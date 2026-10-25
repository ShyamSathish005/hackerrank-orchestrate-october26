"""
phase1_identity.py — Deterministic Identity Graph (Symbolic Phase)

Implements Union-Find backed identity resolution:
1. Build identity graph from user_identities.csv + user_audit_log.csv
2. For a given request email, resolve the full IdentityCluster
   (user_id, all current/historical emails, phones, hashes, names)
3. Zero LLM usage — purely deterministic graph traversal
"""
from __future__ import annotations

import pandas as pd
from typing import Dict, Optional, Set

from models import IdentityCluster


class IdentityGraph:
    """
    Deterministic identity resolver using indexed lookups.
    
    Strategy:
    - Build email->user_id index from current identities
    - Build email->user_id index from historical audit log (old email values)
    - For a given request email, resolve user_id, then gather all identifiers
    """

    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir
        self._user_df: Optional[pd.DataFrame] = None
        self._audit_df: Optional[pd.DataFrame] = None
        # Index: lowercase email -> user_id
        self._email_to_user: Dict[str, str] = {}
        # Index: user_id -> IdentityCluster
        self._user_clusters: Dict[str, IdentityCluster] = {}
        self._built = False

    def build(self) -> None:
        """Build the identity graph from CSV data. Call once."""
        if self._built:
            return

        # ── Load data ──
        self._user_df = pd.read_csv(
            f"{self.dataset_dir}/user_identities.csv",
            dtype=str,
        ).fillna("")
        self._audit_df = pd.read_csv(
            f"{self.dataset_dir}/user_audit_log.csv",
            dtype=str,
        ).fillna("")

        # ── Build clusters for every user ──
        for _, row in self._user_df.iterrows():
            uid = row["user_id"]
            cluster = IdentityCluster(
                user_id=uid,
                current_email=row["email"],
                current_name=row["name"],
                current_phone=row["phone"],
                current_email_hash=row["email_hash"],
                all_emails={row["email"]},
                all_phones={row["phone"]} if row["phone"] else set(),
                all_email_hashes={row["email_hash"]} if row["email_hash"] else set(),
                all_names={row["name"]} if row["name"] else set(),
            )
            self._user_clusters[uid] = cluster
            # Index current email
            if row["email"]:
                self._email_to_user[row["email"].lower()] = uid

        # ── Merge historical identifiers from audit log ──
        for _, row in self._audit_df.iterrows():
            uid = row["user_id"]
            field = row["field_name"]
            old_val = row["old_value"]
            new_val = row["new_value"]

            if uid not in self._user_clusters:
                continue

            cluster = self._user_clusters[uid]

            if field == "email":
                if old_val:
                    cluster.all_emails.add(old_val)
                    self._email_to_user[old_val.lower()] = uid
                if new_val:
                    cluster.all_emails.add(new_val)
                    self._email_to_user[new_val.lower()] = uid
            elif field == "phone":
                if old_val:
                    cluster.all_phones.add(old_val)
                if new_val:
                    cluster.all_phones.add(new_val)
            elif field == "email_hash":
                if old_val:
                    cluster.all_email_hashes.add(old_val)
                if new_val:
                    cluster.all_email_hashes.add(new_val)
            elif field == "name":
                if old_val:
                    cluster.all_names.add(old_val)
                if new_val:
                    cluster.all_names.add(new_val)

        self._built = True

    def resolve(self, email: str) -> Optional[IdentityCluster]:
        """
        Resolve an email to a full IdentityCluster.
        Returns None if the email doesn't match any known user.
        """
        if not self._built:
            self.build()

        lookup = email.strip().lower()
        uid = self._email_to_user.get(lookup)
        if uid is None:
            return None
        return self._user_clusters.get(uid)

    def get_cluster_by_user_id(self, user_id: str) -> Optional[IdentityCluster]:
        """Direct lookup by user_id."""
        if not self._built:
            self.build()
        return self._user_clusters.get(user_id)

    def get_all_users(self) -> Dict[str, IdentityCluster]:
        """Return all user clusters."""
        if not self._built:
            self.build()
        return self._user_clusters
