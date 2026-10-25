
"""
models.py — Pydantic / dataclass schemas for the Neuro-Symbolic Deletion Audit Agent.

Every inter-phase payload is strictly typed to guarantee zero schema drift
between the symbolic graph, neural extractor, rule engine, and DAG planner.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Enumerations — single source of truth for the grader-accepted vocabulary
# ---------------------------------------------------------------------------

class DeletionStatus(str, Enum):
    COMPLETE = "COMPLETE"
    INCOMPLETE = "INCOMPLETE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class Phase(str, Enum):
    FINDING = "FINDING"
    REMEDIATION = "REMEDIATION"
    VERIFICATION = "VERIFICATION"


class FindingOp(str, Enum):
    REPORT_REMAINING_RECORD = "REPORT_REMAINING_RECORD"
    REPORT_EXPOSED_PII = "REPORT_EXPOSED_PII"
    REPORT_ACTIVE_DATA_FLOW = "REPORT_ACTIVE_DATA_FLOW"
    REPORT_RESTORE_RISK = "REPORT_RESTORE_RISK"


class RemediationOp(str, Enum):
    DISABLE_DATA_FLOW = "DISABLE_DATA_FLOW"
    BLOCK_RESTORE = "BLOCK_RESTORE"
    DELETE_RECORD = "DELETE_RECORD"
    REDACT_RECORD = "REDACT_RECORD"
    ANONYMIZE_RECORD = "ANONYMIZE_RECORD"


class VerificationOp(str, Enum):
    VERIFY_RECORD_ABSENT = "VERIFY_RECORD_ABSENT"
    VERIFY_RECORD_PRESENT = "VERIFY_RECORD_PRESENT"
    VERIFY_RECORD_REDACTED = "VERIFY_RECORD_REDACTED"
    VERIFY_RECORD_ANONYMIZED = "VERIFY_RECORD_ANONYMIZED"
    VERIFY_DATA_FLOW_DISABLED = "VERIFY_DATA_FLOW_DISABLED"
    VERIFY_RESTORE_BLOCKED = "VERIFY_RESTORE_BLOCKED"


class ResourceState(str, Enum):
    PRESENT = "PRESENT"
    ABSENT = "ABSENT"
    ACTIVE = "ACTIVE"
    DISABLED = "DISABLED"
    BLOCKED = "BLOCKED"
    REDACTED = "REDACTED"
    ANONYMIZED = "ANONYMIZED"


class DataCategory(str, Enum):
    USER_PROFILE = "USER_PROFILE"
    SECONDARY_IDENTIFIER = "SECONDARY_IDENTIFIER"
    SEARCH_INDEX_ENTRY = "SEARCH_INDEX_ENTRY"
    CACHE_ENTRY = "CACHE_ENTRY"
    SUPPORT_MESSAGE = "SUPPORT_MESSAGE"
    SUPPORT_ATTACHMENT = "SUPPORT_ATTACHMENT"
    ANALYTICS_PROFILE = "ANALYTICS_PROFILE"
    PRODUCT_TELEMETRY = "PRODUCT_TELEMETRY"
    USER_EXPORT = "USER_EXPORT"
    MARKETING_PROFILE = "MARKETING_PROFILE"
    EMAIL_DELIVERY_LOG = "EMAIL_DELIVERY_LOG"
    SECURITY_AUDIT_LOG = "SECURITY_AUDIT_LOG"
    LEGAL_HOLD_RECORD = "LEGAL_HOLD_RECORD"
    BACKUP_ARCHIVE = "BACKUP_ARCHIVE"
    ACCOUNT_TOMBSTONE = "ACCOUNT_TOMBSTONE"


class RequiredAction(str, Enum):
    DELETE = "DELETE"
    REDACT = "REDACT"
    ANONYMIZE = "ANONYMIZE"
    RETAIN = "RETAIN"


# ---------------------------------------------------------------------------
# Operation ordering — used by the DAG planner for deterministic step sorting
# ---------------------------------------------------------------------------

# Within FINDING phase, order by this priority
FINDING_OP_ORDER = {
    FindingOp.REPORT_REMAINING_RECORD: 0,
    FindingOp.REPORT_EXPOSED_PII: 1,
    FindingOp.REPORT_ACTIVE_DATA_FLOW: 2,
    FindingOp.REPORT_RESTORE_RISK: 3,
}

# Within REMEDIATION phase, order by this priority
REMEDIATION_OP_ORDER = {
    RemediationOp.DISABLE_DATA_FLOW: 0,
    RemediationOp.BLOCK_RESTORE: 1,
    RemediationOp.DELETE_RECORD: 2,
    RemediationOp.REDACT_RECORD: 3,
    RemediationOp.ANONYMIZE_RECORD: 4,
}

# Within VERIFICATION phase, order by this priority
VERIFICATION_OP_ORDER = {
    VerificationOp.VERIFY_RECORD_ABSENT: 0,
    VerificationOp.VERIFY_RECORD_PRESENT: 1,
    VerificationOp.VERIFY_RECORD_REDACTED: 2,
    VerificationOp.VERIFY_RECORD_ANONYMIZED: 3,
    VerificationOp.VERIFY_DATA_FLOW_DISABLED: 4,
    VerificationOp.VERIFY_RESTORE_BLOCKED: 5,
}

# System ordering for deterministic tie-breaking
SYSTEM_ORDER = {
    "account_service": 0,
    "export_service": 1,
    "search_service": 2,
    "security_service": 3,
    "identity_service": 4,
    "support_platform": 5,
    "analytics_platform": 6,
    "integration_service": 7,
    "backup_service": 8,
    "marketing_platform": 9,
    "email_delivery_service": 10,
    "cache_service": 11,
    "notification_service": 12,
}


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

@dataclass
class IdentityCluster:
    """Phase 1 output — deterministic identity resolution result."""
    user_id: str
    current_email: str
    current_name: str
    current_phone: str
    current_email_hash: str
    # All identifiers ever used (current + historical)
    all_emails: set = field(default_factory=set)
    all_phones: set = field(default_factory=set)
    all_email_hashes: set = field(default_factory=set)
    all_names: set = field(default_factory=set)

    def all_identifiers_lower(self) -> set:
        """Return all identifiers as lowercase strings for matching."""
        result = set()
        for e in self.all_emails:
            if e:
                result.add(e.lower())
        for p in self.all_phones:
            if p:
                result.add(p.lower())
        for h in self.all_email_hashes:
            if h:
                result.add(h.lower())
        for n in self.all_names:
            if n:
                result.add(n.lower())
        if self.user_id:
            result.add(self.user_id.lower())
        return result


@dataclass
class EvidenceRecord:
    """A single discovered data record from evidence inspection."""
    system_id: str
    resource_id: str
    data_category: str  # Maps to DataCategory
    state: str  # Current observed state: PRESENT, IDENTIFIABLE, ANONYMIZED, etc.
    evidence_id: str  # The file_id from evidence_files.csv
    has_pii: bool = False  # Whether PII is still exposed
    identifier_found: str = ""  # What identifier was found


@dataclass
class DataFlowRecord:
    """An active data flow that could recreate deleted data."""
    system_id: str
    resource_id: str
    evidence_id: str
    target_email: str = ""
    connector_name: str = ""
    state: str = "ACTIVE"


@dataclass
class RestoreRiskRecord:
    """A backup/restore path that could bring data back."""
    system_id: str
    resource_id: str
    evidence_id: str
    target_email: str = ""
    state: str = "ACTIVE"


@dataclass
class Step:
    """One step in the final steps_json output."""
    step_no: int
    phase: str
    system_id: str
    resource_id: str
    operation: str
    state: str
    evidence_id: str

    def to_dict(self) -> dict:
        return {
            "step_no": self.step_no,
            "phase": self.phase,
            "system_id": self.system_id,
            "resource_id": self.resource_id,
            "operation": self.operation,
            "state": self.state,
            "evidence_id": self.evidence_id,
        }


@dataclass
class CasePayload:
    """Full inter-phase payload for a single case."""
    case_id: str
    email: str
    requested_at: str
    reported_status: str
    reported_at: str
    audit_at: str
    request_text: str

    # Phase 1: Identity resolution
    identity: Optional[IdentityCluster] = None
    is_deletion_request: bool = True

    # Phase 2: Evidence discovery
    evidence_records: list = field(default_factory=list)
    data_flows: list = field(default_factory=list)
    restore_risks: list = field(default_factory=list)

    # Phase 3: Rule engine decision
    deletion_status: Optional[DeletionStatus] = None

    # Phase 4: Steps
    steps: list = field(default_factory=list)

    def steps_json(self) -> str:
        """Serialize steps to JSON string for CSV output."""
        return json.dumps([s.to_dict() for s in self.steps], separators=(",", ":"))


# ---------------------------------------------------------------------------
# Mapping helpers
# ---------------------------------------------------------------------------

# resource_id prefix -> data_category
RESOURCE_PREFIX_TO_CATEGORY = {
    "account": DataCategory.USER_PROFILE.value,
    "search": DataCategory.SEARCH_INDEX_ENTRY.value,
    "analytics": DataCategory.ANALYTICS_PROFILE.value,
    "support": DataCategory.SUPPORT_MESSAGE.value,  # default; attachment detected separately
    "export": DataCategory.USER_EXPORT.value,
    "cache": DataCategory.CACHE_ENTRY.value,
    "audit_record": DataCategory.SECURITY_AUDIT_LOG.value,
    "tombstone": DataCategory.ACCOUNT_TOMBSTONE.value,
    "data_flow": "DATA_FLOW",  # Not a data category, handled separately
    "backup": DataCategory.BACKUP_ARCHIVE.value,
    "email_delivery": DataCategory.EMAIL_DELIVERY_LOG.value,
    "marketing": DataCategory.MARKETING_PROFILE.value,
    "legal_hold": DataCategory.LEGAL_HOLD_RECORD.value,
    "object": DataCategory.USER_EXPORT.value,  # object storage = exported data
}

# resource_id prefix -> system_id
RESOURCE_PREFIX_TO_SYSTEM = {
    "account": "account_service",
    "search": "search_service",
    "analytics": "analytics_platform",
    "support": "support_platform",
    "export": "export_service",
    "cache": "cache_service",
    "audit_record": "security_service",
    "tombstone": "identity_service",
    "data_flow": "integration_service",
    "backup": "backup_service",
    "email_delivery": "email_delivery_service",
    "marketing": "marketing_platform",
    "legal_hold": "legal_service",
    "object": "object_store",
}


def resource_id_to_category(resource_id: str) -> str:
    """Determine data category from resource_id prefix."""
    for prefix, category in RESOURCE_PREFIX_TO_CATEGORY.items():
        if resource_id.startswith(prefix + "_"):
            return category
    return DataCategory.USER_PROFILE.value


def resource_id_to_system(resource_id: str) -> str:
    """Determine system_id from resource_id prefix."""
    for prefix, system in RESOURCE_PREFIX_TO_SYSTEM.items():
        if resource_id.startswith(prefix + "_"):
            return system
    return "account_service"
