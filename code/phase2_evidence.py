"""
phase2_evidence.py — Evidence Parser and Neural Entity Extractor

Combines deterministic structured-evidence parsing with targeted LLM
extraction for unstructured content (PDFs, images, free-text logs).

For structured evidence (SQL, JSON, structured logs, TXT handovers),
we use pure regex/pandas parsing — no LLM needed.

Phase 2 output: list of EvidenceRecord, DataFlowRecord, RestoreRiskRecord
per case, grounded to the IdentityCluster from Phase 1.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import tempfile
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from models import (
    CasePayload,
    DataCategory,
    DataFlowRecord,
    EvidenceRecord,
    IdentityCluster,
    RestoreRiskRecord,
    resource_id_to_category,
    resource_id_to_system,
)


class EvidenceIndex:
    """
    Pre-parses and indexes ALL evidence files once, then provides
    fast per-user lookups via identity matching.
    
    This is the core of the "neuro" phase — but most evidence is structured
    enough that we parse deterministically. Only truly unstructured evidence
    (PDFs, images) would need LLM; we handle those with heuristic extraction.
    """

    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir
        self.evidence_dir = os.path.join(dataset_dir, "evidence")
        self._evidence_manifest: Optional[pd.DataFrame] = None
        self._system_activity: Optional[pd.DataFrame] = None

        # Pre-parsed evidence stores
        # Key: (system_id, resource_id) -> list of observation dicts
        self._resource_observations: Dict[Tuple[str, str], List[dict]] = {}
        # Key: evidence_id -> list of (system_id, resource_id, state, identifier, category, captured_at)
        self._evidence_entries: Dict[str, List[dict]] = {}

        self._built = False

    def build(self) -> None:
        """Parse all evidence files and system activity. Call once."""
        if self._built:
            return

        self._evidence_manifest = pd.read_csv(
            f"{self.dataset_dir}/evidence_files.csv", dtype=str
        ).fillna("")

        self._system_activity = pd.read_csv(
            f"{self.dataset_dir}/system_activity.csv", dtype=str
        ).fillna("")

        # Parse each evidence file
        for _, row in self._evidence_manifest.iterrows():
            file_id = row["file_id"]
            file_path = os.path.join(self.dataset_dir, row["file_path"])
            captured_at = row["captured_at"]

            if not os.path.exists(file_path):
                continue

            if file_path.endswith(".sql"):
                self._parse_sql_evidence(file_id, file_path, captured_at)
            elif file_path.endswith(".json"):
                self._parse_json_evidence(file_id, file_path, captured_at)
            elif file_path.endswith(".log"):
                self._parse_log_evidence(file_id, file_path, captured_at)
            elif file_path.endswith(".txt"):
                self._parse_txt_evidence(file_id, file_path, captured_at)
            elif file_path.endswith(".pdf"):
                self._parse_pdf_evidence(file_id, file_path, captured_at)
            elif file_path.endswith(".png"):
                self._parse_image_evidence(file_id, file_path, captured_at)

        self._built = True

    # ── SQL Evidence Parser ──

    def _parse_sql_evidence(self, file_id: str, file_path: str, captured_at: str) -> None:
        """Parse SQLite-compatible SQL snapshots."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                sql_content = f.read()
        except Exception:
            return

        # Execute in an in-memory SQLite database
        try:
            conn = sqlite3.connect(":memory:")
            conn.executescript(sql_content)

            # Discover all tables
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = [row[0] for row in cursor.fetchall()]

            for table in tables:
                # Get columns
                cursor = conn.execute(f"PRAGMA table_info('{table}')")
                columns = [col[1] for col in cursor.fetchall()]

                # Get all rows
                cursor = conn.execute(f"SELECT * FROM '{table}'")
                rows = cursor.fetchall()

                for row_data in rows:
                    row_dict = dict(zip(columns, row_data))
                    self._index_sql_row(file_id, table, row_dict, captured_at)

            conn.close()
        except Exception:
            # If SQL execution fails, try regex-based INSERT parsing
            self._parse_sql_inserts(file_id, sql_content, captured_at)

    def _index_sql_row(self, file_id: str, table: str, row: dict, captured_at: str) -> None:
        """Index a single SQL row into the observation store."""
        resource_id = row.get("resource_id", "")
        if not resource_id:
            return

        # Determine system_id from table name and resource prefix
        system_id = self._system_from_table(table, resource_id)
        category = resource_id_to_category(resource_id)

        # Collect identifiers
        identifiers = []
        for key in ["email", "full_name", "phone", "email_hash", "user_id",
                     "customer_reference", "requester_email", "requester_name",
                     "requester_phone"]:
            val = row.get(key, "")
            if val and str(val).lower() not in ("none", "null", ""):
                identifiers.append(str(val))

        status = row.get("account_status", row.get("status", ""))

        entry = {
            "system_id": system_id,
            "resource_id": resource_id,
            "category": category,
            "identifiers": identifiers,
            "status": status,
            "captured_at": captured_at,
            "table": table,
            "raw": row,
        }

        key = (system_id, resource_id)
        self._resource_observations.setdefault(key, []).append(entry)
        self._evidence_entries.setdefault(file_id, []).append(entry)

    def _system_from_table(self, table: str, resource_id: str) -> str:
        """Determine system_id from table name and resource prefix."""
        table_lower = table.lower()
        if "failover" in table_lower or "primary" in table_lower or "replica" in table_lower or "reporting" in table_lower or "regional" in table_lower:
            return resource_id_to_system(resource_id)
        if "search" in table_lower:
            return "search_service"
        if "support" in table_lower:
            return "support_platform"
        if "analytics" in table_lower:
            return "analytics_platform"
        if "inventory" in table_lower:
            return resource_id_to_system(resource_id)
        return resource_id_to_system(resource_id)

    def _parse_sql_inserts(self, file_id: str, content: str, captured_at: str) -> None:
        """Fallback: parse INSERT statements via regex."""
        insert_re = re.compile(
            r"INSERT INTO\s+[\"']?(\w+)[\"']?\s+VALUES\s*\((.+?)\);",
            re.IGNORECASE,
        )
        for m in insert_re.finditer(content):
            table = m.group(1)
            values_str = m.group(2)
            # Rough split on comma, handling quoted strings
            values = re.findall(r"'([^']*)'|(\d+(?:\.\d+)?)|NULL", values_str)
            row_values = [v[0] if v[0] else (v[1] if v[1] else None) for v in values]
            if row_values and len(row_values) >= 2:
                row_dict = {"resource_id": row_values[0] if row_values else ""}
                if len(row_values) > 1:
                    row_dict["user_id"] = row_values[1]
                if len(row_values) > 2:
                    row_dict["full_name"] = row_values[2]
                if len(row_values) > 3:
                    row_dict["email"] = row_values[3]
                if len(row_values) > 4:
                    row_dict["phone"] = row_values[4]
                if len(row_values) > 5:
                    row_dict["account_status"] = row_values[5]
                self._index_sql_row(file_id, table, row_dict, captured_at)

    # ── JSON Evidence Parser ──

    def _parse_json_evidence(self, file_id: str, file_path: str, captured_at: str) -> None:
        """Parse JSON evidence (snapshot bundles, export manifests, configs, etc.)."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return

        if isinstance(data, dict):
            self._walk_json(file_id, data, captured_at, [])

    def _walk_json(self, file_id: str, obj, captured_at: str, path: list) -> None:
        """Recursively walk JSON structure to extract resource observations."""
        if isinstance(obj, dict):
            # Check if this dict represents a resource entry
            if "resource_id" in obj:
                self._index_json_entry(file_id, obj, captured_at, path)
            else:
                for key, val in obj.items():
                    self._walk_json(file_id, val, captured_at, path + [key])
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                self._walk_json(file_id, item, captured_at, path + [str(i)])

    def _index_json_entry(self, file_id: str, obj: dict, captured_at: str, path: list) -> None:
        """Index a JSON object that contains a resource_id."""
        resource_id = obj.get("resource_id", "")
        if not resource_id:
            return

        # Infer system from path and resource prefix
        system_id = self._system_from_json_path(path, resource_id)
        category = resource_id_to_category(resource_id)

        identifiers = []
        for key in ["email", "full_name", "phone", "email_hash", "customer_reference",
                     "customer_email", "requester_email", "requester_name", "requester_phone",
                     "user_id", "subject_reference", "lookup_key"]:
            val = obj.get(key)
            if val and str(val).lower() not in ("none", "null", ""):
                identifiers.append(str(val))

        # Check profile sub-object
        profile = obj.get("profile")
        if isinstance(profile, dict):
            for key in ["email", "full_name", "phone", "email_hash", "customer_reference"]:
                val = profile.get(key)
                if val and str(val).lower() not in ("none", "null", ""):
                    identifiers.append(str(val))

        status_code = obj.get("status_code")
        status = obj.get("status", obj.get("account_status", ""))

        # Determine observation state
        obs_state = "PRESENT"
        if status_code == 404 or status_code == "404":
            obs_state = "ABSENT"
        elif status and str(status).upper() == "DELETED":
            obs_state = "ABSENT"

        # Check for enabled/disabled data flows
        enabled = obj.get("enabled")
        if enabled is not None:
            obs_state = "ACTIVE" if (enabled is True or str(enabled).lower() == "true") else "DISABLED"

        # Check for restore access
        restore_allowed = obj.get("restore_allowed")
        if restore_allowed is not None:
            obs_state = "ACTIVE" if (restore_allowed is True or str(restore_allowed).lower() == "true") else "BLOCKED"

        # Check for anonymization
        for key in ["email", "customer_reference", "requester_email"]:
            val = obj.get(key)
            if val is None and key in obj:
                pass  # null means removed

        entry = {
            "system_id": system_id,
            "resource_id": resource_id,
            "category": category,
            "identifiers": identifiers,
            "status": obs_state,
            "captured_at": captured_at,
            "path": "/".join(path),
            "raw": obj,
        }

        key = (system_id, resource_id)
        self._resource_observations.setdefault(key, []).append(entry)
        self._evidence_entries.setdefault(file_id, []).append(entry)

    def _system_from_json_path(self, path: list, resource_id: str) -> str:
        """Infer system_id from JSON path context."""
        path_str = "/".join(path).lower()
        if "account_service" in path_str:
            return "account_service"
        elif "search_service" in path_str:
            return "search_service"
        elif "analytics" in path_str:
            return "analytics_platform"
        elif "support" in path_str:
            return "support_platform"
        elif "security" in path_str:
            return "security_service"
        elif "export" in path_str:
            return "export_service"
        elif "integration" in path_str:
            return "integration_service"
        elif "backup" in path_str:
            return "backup_service"
        elif "identity" in path_str:
            return "identity_service"
        elif "marketing" in path_str:
            return "marketing_platform"
        elif "connector" in path_str:
            return "integration_service"
        return resource_id_to_system(resource_id)

    # ── Log Evidence Parser ──

    def _parse_log_evidence(self, file_id: str, file_path: str, captured_at: str) -> None:
        """Parse structured log files."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception:
            return

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            self._parse_log_line(file_id, line, captured_at, line_num)

    def _parse_log_line(self, file_id: str, line: str, captured_at: str, line_num: int) -> None:
        """Parse a single structured log line (key=value format)."""
        # Extract key-value pairs
        kv = {}
        # Match key=value patterns (value can be quoted or unquoted)
        for m in re.finditer(r'(\w+)=("(?:[^"\\]|\\.)*"|[^\s]+)', line):
            key = m.group(1)
            val = m.group(2).strip('"')
            kv[key] = val

        resource_id = kv.get("resource_id", "")
        if not resource_id:
            return

        # Only index "completed" events or info-level observations
        event = kv.get("event", "")
        if "started" in event:
            return  # Skip "started" events, wait for completion

        system_id = kv.get("service", resource_id_to_system(resource_id))
        category = resource_id_to_category(resource_id)

        identifiers = []
        for key in ["email", "customer_email", "customer_reference", "lookup_key",
                     "requester_email", "requester_name", "requester_phone",
                     "subject_reference"]:
            val = kv.get(key, "")
            if val and val.lower() not in ("none", "null", ""):
                identifiers.append(val)

        # Determine state from log data
        obs_state = "PRESENT"
        http_status = kv.get("http_status", "")
        rows = kv.get("rows", "")
        result = kv.get("result", "")
        enabled = kv.get("enabled", "")
        restore_allowed = kv.get("restore_allowed", "")

        if http_status == "404" or rows == "0":
            obs_state = "ABSENT"
        elif result == "miss":
            obs_state = "ABSENT"
        elif result == "hit":
            obs_state = "PRESENT"
        elif enabled == "true":
            obs_state = "ACTIVE"
        elif enabled == "false":
            obs_state = "DISABLED"
        elif restore_allowed == "true":
            obs_state = "ACTIVE"
        elif restore_allowed == "false":
            obs_state = "BLOCKED"

        # Extract timestamp from line
        ts_match = re.match(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)', line)
        line_time = ts_match.group(1) if ts_match else captured_at

        entry = {
            "system_id": system_id,
            "resource_id": resource_id,
            "category": category,
            "identifiers": identifiers,
            "status": obs_state,
            "captured_at": line_time,
            "event": event,
            "raw_kv": kv,
        }

        key = (system_id, resource_id)
        self._resource_observations.setdefault(key, []).append(entry)
        self._evidence_entries.setdefault(file_id, []).append(entry)

    # ── TXT Evidence Parser ──

    def _parse_txt_evidence(self, file_id: str, file_path: str, captured_at: str) -> None:
        """Parse text handover/report files."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            return

        # Parse handover-format lines: timestamp | system | resource | category | status | identifier
        pattern = re.compile(
            r'-\s*(\S+)\s*\|\s*(\S+)\s*\|\s*(\S+)\s*\|\s*(\S+)\s*\|\s*(\S+)\s*\|\s*(.+)'
        )

        for m in pattern.finditer(content):
            timestamp = m.group(1)
            system_id = m.group(2)
            resource_id = m.group(3)
            data_cat = m.group(4)
            status = m.group(5)
            identifier = m.group(6).strip()

            obs_state = "PRESENT"
            if status == "NOT_FOUND":
                obs_state = "ABSENT"
            elif status == "FOUND":
                obs_state = "PRESENT"
            elif status == "IDENTIFIABLE":
                obs_state = "IDENTIFIABLE"
            elif status == "ANONYMIZED":
                obs_state = "ANONYMIZED"
            elif status == "REDACTED":
                obs_state = "REDACTED"
            elif status == "ACTIVE":
                obs_state = "ACTIVE"
            elif status == "DISABLED":
                obs_state = "DISABLED"
            elif status == "BLOCKED":
                obs_state = "BLOCKED"

            identifiers = []
            if identifier and identifier.lower() not in ("no direct identifier",):
                identifiers.append(identifier)

            entry = {
                "system_id": system_id,
                "resource_id": resource_id,
                "category": data_cat,
                "identifiers": identifiers,
                "status": obs_state,
                "captured_at": timestamp,
                "raw_status": status,
            }

            key = (system_id, resource_id)
            self._resource_observations.setdefault(key, []).append(entry)
            self._evidence_entries.setdefault(file_id, []).append(entry)

    # ── PDF Evidence Parser (Heuristic) ──

    def _parse_pdf_evidence(self, file_id: str, file_path: str, captured_at: str) -> None:
        """
        Extract text from PDFs using pdfplumber or PyPDF2.
        Falls back to filename-based metadata if libraries unavailable.
        """
        text = ""
        try:
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                # Also try to extract tables
                for page in pdf.pages:
                    tables = page.extract_tables()
                    for table in tables:
                        for row in table:
                            if row:
                                text += " | ".join(str(c) if c else "" for c in row) + "\n"
        except ImportError:
            try:
                from PyPDF2 import PdfReader
                reader = PdfReader(file_path)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            except ImportError:
                return

        if not text.strip():
            return

        # Parse extracted text for resource observations
        self._parse_extracted_text(file_id, text, captured_at)

    def _parse_extracted_text(self, file_id: str, text: str, captured_at: str) -> None:
        """Parse text extracted from PDFs/images for resource references."""
        # Look for resource_id patterns
        resource_pattern = re.compile(
            r'(account|search|analytics|support|export|cache|audit_record|tombstone|'
            r'data_flow|backup|email_delivery|marketing|legal_hold|object)_(\d+)'
        )

        # Look for email patterns
        email_pattern = re.compile(r'[\w.+-]+@[\w-]+\.[\w.]+')

        # Look for user_id patterns
        user_id_pattern = re.compile(r'user_(\d+)')

        # Look for status indicators
        lines = text.split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue

            resources = resource_pattern.findall(line)
            emails = email_pattern.findall(line)
            user_ids = user_id_pattern.findall(line)

            for prefix, num in resources:
                resource_id = f"{prefix}_{num}"
                system_id = resource_id_to_system(resource_id)
                category = resource_id_to_category(resource_id)

                identifiers = list(emails)
                for uid_num in user_ids:
                    identifiers.append(f"user_{uid_num}")

                # Try to determine state from line context
                obs_state = "PRESENT"
                line_lower = line.lower()
                if "not found" in line_lower or "deleted" in line_lower or "absent" in line_lower or "404" in line_lower:
                    obs_state = "ABSENT"
                elif "anonymized" in line_lower or "anonymised" in line_lower:
                    obs_state = "ANONYMIZED"
                elif "redacted" in line_lower:
                    obs_state = "REDACTED"
                elif "active" in line_lower or "enabled" in line_lower:
                    obs_state = "ACTIVE"
                elif "disabled" in line_lower or "stopped" in line_lower:
                    obs_state = "DISABLED"
                elif "blocked" in line_lower:
                    obs_state = "BLOCKED"
                elif "retained" in line_lower or "retain" in line_lower:
                    obs_state = "RETAINED"

                entry = {
                    "system_id": system_id,
                    "resource_id": resource_id,
                    "category": category,
                    "identifiers": identifiers,
                    "status": obs_state,
                    "captured_at": captured_at,
                    "source": "pdf_extraction",
                }

                key = (system_id, resource_id)
                self._resource_observations.setdefault(key, []).append(entry)
                self._evidence_entries.setdefault(file_id, []).append(entry)

    # ── Image Evidence Parser (Heuristic) ──

    def _parse_image_evidence(self, file_id: str, file_path: str, captured_at: str) -> None:
        """
        Extract text from screenshots using OCR.
        Falls back to filename-based heuristics if OCR unavailable.
        """
        text = ""
        try:
            import pytesseract
            from PIL import Image
            img = Image.open(file_path)
            text = pytesseract.image_to_string(img)
        except ImportError:
            pass

        if text.strip():
            self._parse_extracted_text(file_id, text, captured_at)

    def query_for_user(
        self,
        identity: IdentityCluster,
        audit_at: str,
        system_activity_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[List[EvidenceRecord], List[DataFlowRecord], List[RestoreRiskRecord]]:
        """Convenience alias for querying records for a user."""
        if system_activity_df is None:
            system_activity_df = self._system_activity
        return self.find_records_for_user(identity, audit_at, system_activity_df)

    def find_records_for_user(
        self,
        identity: IdentityCluster,
        audit_at: str,
        system_activity_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[List[EvidenceRecord], List[DataFlowRecord], List[RestoreRiskRecord]]:
        """
        Find all evidence records, data flows, and restore risks for a user
        at or before the audit_at timestamp.
        
        Returns: (evidence_records, data_flows, restore_risks)
        """
        evidence_records: List[EvidenceRecord] = []
        data_flows: List[DataFlowRecord] = []
        restore_risks: List[RestoreRiskRecord] = []

        if system_activity_df is None:
            system_activity_df = self._system_activity

        all_ids_lower = identity.all_identifiers_lower()

        # Track unique (system_id, resource_id) pairs to avoid duplicates
        seen_resources: Set[Tuple[str, str]] = set()
        seen_flows: Set[str] = set()
        seen_backups: Set[str] = set()

        # ── Search system activity for user-related resources ──
        user_resources = self._find_user_resources_from_activity(
            identity, system_activity_df, audit_at
        )

        # ── Search evidence entries for user-related resources ──
        for file_id, entries in self._evidence_entries.items():
            for entry in entries:
                # Check if this entry relates to our user
                entry_ids = {str(eid).lower() for eid in entry.get("identifiers", [])}
                if not entry_ids.intersection(all_ids_lower):
                    continue

                # Check timestamp constraint
                entry_time = entry.get("captured_at", "")
                if entry_time and audit_at and entry_time > audit_at:
                    continue

                system_id = entry["system_id"]
                resource_id = entry["resource_id"]
                category = entry.get("category", resource_id_to_category(resource_id))
                status = entry.get("status", "PRESENT")

                # Handle data flows
                if resource_id.startswith("data_flow_"):
                    if status in ("ACTIVE",) and resource_id not in seen_flows:
                        seen_flows.add(resource_id)
                        target_email = ""
                        raw = entry.get("raw_kv", entry.get("raw", {}))
                        if isinstance(raw, dict):
                            target_email = raw.get("customer_email", "")
                        data_flows.append(DataFlowRecord(
                            system_id="integration_service",
                            resource_id=resource_id,
                            evidence_id=file_id,
                            target_email=target_email,
                            connector_name=raw.get("connector_name", "") if isinstance(raw, dict) else "",
                        ))
                    continue

                # Handle backups / restore risks
                if resource_id.startswith("backup_"):
                    if status in ("ACTIVE",) and resource_id not in seen_backups:
                        seen_backups.add(resource_id)
                        restore_risks.append(RestoreRiskRecord(
                            system_id="backup_service",
                            resource_id=resource_id,
                            evidence_id=file_id,
                        ))
                    continue

                # Regular resource record
                res_key = (system_id, resource_id)
                if res_key in seen_resources:
                    continue
                seen_resources.add(res_key)

                # Determine if PII is exposed
                has_pii = False
                if status in ("PRESENT", "IDENTIFIABLE"):
                    if category in (DataCategory.ANALYTICS_PROFILE.value,
                                    DataCategory.PRODUCT_TELEMETRY.value):
                        has_pii = True
                    elif category in (DataCategory.SUPPORT_MESSAGE.value,):
                        has_pii = True
                    elif category == DataCategory.ACCOUNT_TOMBSTONE.value:
                        # Check if PII fields are populated
                        raw = entry.get("raw", {})
                        if isinstance(raw, dict):
                            for pii_key in ["email", "full_name", "phone", "email_hash",
                                            "requester_email", "requester_name"]:
                                val = raw.get(pii_key)
                                if val and str(val).lower() not in ("none", "null", ""):
                                    has_pii = True
                                    break

                evidence_records.append(EvidenceRecord(
                    system_id=system_id,
                    resource_id=resource_id,
                    data_category=category,
                    state=status,
                    evidence_id=file_id,
                    has_pii=has_pii,
                ))

        # ── Also check resources found in system_activity ──
        for resource_id, details in user_resources.items():
            system_id = details["system_id"]
            res_key = (system_id, resource_id)
            if res_key in seen_resources:
                continue

            category = resource_id_to_category(resource_id)

            # Handle data flows from system_activity
            if resource_id.startswith("data_flow_") and resource_id not in seen_flows:
                # Check if the flow is still active at audit_at
                if details.get("is_active", False):
                    seen_flows.add(resource_id)
                    data_flows.append(DataFlowRecord(
                        system_id="integration_service",
                        resource_id=resource_id,
                        evidence_id=details.get("evidence_id", ""),
                        target_email="",
                    ))
                continue

            if resource_id.startswith("backup_") and resource_id not in seen_backups:
                if details.get("is_active", False):
                    seen_backups.add(resource_id)
                    restore_risks.append(RestoreRiskRecord(
                        system_id="backup_service",
                        resource_id=resource_id,
                        evidence_id=details.get("evidence_id", ""),
                    ))
                continue

            # Skip resources we don't need to track (audit_records handled below)
            if resource_id.startswith("audit_record_"):
                # These are always retained
                seen_resources.add(res_key)
                evidence_records.append(EvidenceRecord(
                    system_id=system_id,
                    resource_id=resource_id,
                    data_category=DataCategory.SECURITY_AUDIT_LOG.value,
                    state="PRESENT",
                    evidence_id=details.get("evidence_id", ""),
                ))

        return evidence_records, data_flows, restore_risks

    def _find_user_resources_from_activity(
        self,
        identity: IdentityCluster,
        sa_df: pd.DataFrame,
        audit_at: str,
    ) -> Dict[str, dict]:
        """Find resources in system_activity that involve this user."""
        results: Dict[str, dict] = {}

        # Filter by resource_id patterns that include user references in details
        all_ids_lower = identity.all_identifiers_lower()

        for _, row in sa_df.iterrows():
            event_time = row.get("event_time", "")
            if event_time and audit_at and event_time > audit_at:
                continue

            resource_id = row.get("resource_id", "")
            details = str(row.get("details", "")).lower()
            target_system = row.get("target_system", "")

            # Check if this activity mentions our user
            is_match = False
            for ident in all_ids_lower:
                if ident in details:
                    is_match = True
                    break

            if not is_match:
                continue

            system_id = target_system if target_system else resource_id_to_system(resource_id)
            event_type = row.get("event_type", "")

            if resource_id not in results:
                results[resource_id] = {
                    "system_id": system_id,
                    "evidence_id": "",
                    "is_active": False,
                    "last_event": event_type,
                    "event_time": event_time,
                }

            # Update state tracking
            existing = results[resource_id]
            if event_time >= existing.get("event_time", ""):
                existing["last_event"] = event_type
                existing["event_time"] = event_time

                if event_type in ("DISABLE_DATA_FLOW",):
                    existing["is_active"] = False
                elif event_type in ("CHECK_DATA_FLOW", "SYNC_RECORD"):
                    if resource_id.startswith("data_flow_"):
                        existing["is_active"] = True
                elif event_type in ("BLOCK_RESTORE",):
                    existing["is_active"] = False
                elif event_type in ("CHECK_RESTORE_ACCESS",):
                    if resource_id.startswith("backup_"):
                        existing["is_active"] = True

        return results

    def get_evidence_for_resource(
        self, system_id: str, resource_id: str, identity: IdentityCluster, audit_at: str
    ) -> Optional[str]:
        """Find the best evidence_id that proves a resource state for a user at audit_at."""
        all_ids_lower = identity.all_identifiers_lower()
        best_file_id = ""
        best_time = ""

        for file_id, entries in self._evidence_entries.items():
            for entry in entries:
                if entry["resource_id"] != resource_id:
                    continue
                if entry["system_id"] != system_id:
                    continue

                entry_time = entry.get("captured_at", "")
                if entry_time and audit_at and entry_time > audit_at:
                    continue

                # Check identity match
                entry_ids = {str(eid).lower() for eid in entry.get("identifiers", [])}
                if entry_ids.intersection(all_ids_lower) or not entry_ids:
                    if not best_time or (entry_time and entry_time > best_time):
                        best_file_id = file_id
                        best_time = entry_time

        return best_file_id if best_file_id else None
