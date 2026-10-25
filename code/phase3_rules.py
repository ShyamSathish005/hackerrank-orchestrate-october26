"""
phase3_rules.py — Process-Aware Verification State Machine (Symbolic Phase)

Implements the strict rule engine from deletion_rules.csv:
1. Classifies whether a request is actually a deletion request
2. Cross-references discovered data against retention policies
3. Determines COMPLETE / INCOMPLETE / NOT_APPLICABLE status
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import pandas as pd

from models import (
    CasePayload,
    DataCategory,
    DataFlowRecord,
    DeletionStatus,
    EvidenceRecord,
    RequiredAction,
    RestoreRiskRecord,
)


# ---------------------------------------------------------------------------
# Request Classification (deterministic)
# ---------------------------------------------------------------------------

# Keywords that indicate a NON-deletion request
NON_DELETION_INDICATORS = [
    r"\bdownload\b.*\b(information|data|account)\b",
    r"\bcopy of\b",
    r"\binformation request\b",
    r"\bnot a request to erase\b",
    r"\bdon'?t\s+(delete|remove|close|erase)\b",
    r"\bdo not\s+(delete|remove|close|erase)\b",
    r"\bkeep\s+(the\s+)?account\b",
    r"\bleave\s+my\b.*\bin place\b",
    r"\breset(ting)?\s+(the\s+)?login\b",
    r"\baccess\s+my\s+profile\b",
    r"\bmisspelled\b",
    r"\bcorrect\s+it\b",
    r"\bchange\s+the\s+email\b",
    r"\bcontinue\s+using\b",
    r"\bopt[\s-]?out\b.*\bmarketing\b",
    r"\bremove\s+me\s+from\s+marketing\b",
    r"\bunsubscribe\b",
    r"\bmarketing\s+lists?\s+only\b",
    r"\bstill\s+(want|use|need)\b",
    r"\bplease\s+don'?t\s+delete\b",
    r"\bwhat\s+personal\s+information\b.*\byou\s+hold\b",
    r"\bsend\s+me\s+a\s+copy\b",
    r"\bportable\s+format\b",
]

# Keywords that indicate a valid DELETION request
DELETION_INDICATORS = [
    r"\bdelete\b",
    r"\bdeletion\b",
    r"\berase\b",
    r"\berased\b",
    r"\bremove\b",
    r"\bremoval\b",
    r"\bclose\s+(my\s+)?account\b",
    r"\bclosed\s+my\s+account\b",
    r"\bpermanently\s+delete\b",
    r"\baccount[\s-]deletion\b",
    r"\bsuppression\b",
    r"\bexclus[ãa]o\b",           # Portuguese
    r"\b削除\b",                  # Japanese
    r"\belimin(ar|ación)\b",     # Spanish
    r"\bsupprimer\b",           # French
    r"\bsuppression\b",         # French
]


def is_deletion_request(request_text: str) -> bool:
    """
    Deterministic classification of whether request_text authorizes deletion.
    
    Rule: A request for access, correction, export, marketing opt-out,
    or another non-deletion action is NOT_APPLICABLE.
    """
    text_lower = request_text.lower()

    # Check non-deletion indicators first (they take priority)
    for pattern in NON_DELETION_INDICATORS:
        if re.search(pattern, text_lower):
            # But verify it's not also a deletion request
            has_deletion = False
            for del_pattern in DELETION_INDICATORS:
                if re.search(del_pattern, text_lower):
                    has_deletion = True
                    break

            # If text explicitly says "don't delete" or "not a deletion request"
            # then it's not a deletion request even if deletion words appear
            if re.search(r"\bnot\s+a\s+request\s+to\s+erase\b", text_lower):
                return False
            if re.search(r"\bdon'?t\s+(delete|remove|close|erase)\b", text_lower):
                return False
            if re.search(r"\bdo\s+not\s+(delete|remove|close|erase)\b", text_lower):
                return False
            if re.search(r"\bplease\s+don'?t\s+delete\b", text_lower):
                return False
            if re.search(r"\bkeep\s+(the\s+)?account\s+as\s+it\s+is\b", text_lower):
                return False
            if re.search(r"\bleave\s+my\b.*\bin\s+place\b", text_lower):
                return False

            if not has_deletion:
                return False

    # Check for deletion indicators
    for pattern in DELETION_INDICATORS:
        if re.search(pattern, text_lower):
            return True

    # Default: treat ambiguous requests as deletion requests
    return True


# ---------------------------------------------------------------------------
# Rule Engine
# ---------------------------------------------------------------------------

class RuleEngine:
    """
    Process-aware verification state machine.
    
    Ingests deletion_rules.csv and uses strict logic gates to determine
    whether each resource requires DELETE, REDACT, ANONYMIZE, or RETAIN.
    """

    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir
        self._rules: Dict[str, str] = {}  # data_category -> required_action
        self._rules_df: Optional[pd.DataFrame] = None
        self._built = False

    def build(self) -> None:
        """Load deletion rules."""
        if self._built:
            return

        self._rules_df = pd.read_csv(
            f"{self.dataset_dir}/deletion_rules.csv", dtype=str
        ).fillna("")

        for _, row in self._rules_df.iterrows():
            category = row["data_category"]
            action = row["required_action"]
            self._rules[category] = action

        self._built = True

    def get_required_action(self, data_category: str) -> str:
        """Get the required action for a data category."""
        if not self._built:
            self.build()
        return self._rules.get(data_category, RequiredAction.DELETE.value)

    def evaluate_case(self, case: CasePayload) -> DeletionStatus:
        """
        Evaluate a case and determine its deletion status.
        
        Logic gates:
        1. IF request doesn't authorize deletion -> NOT_APPLICABLE
        2. IF un-exempted data exists OR active sync/restore detected -> INCOMPLETE
        3. IF all non-exempt data missing AND retained data matches policy -> COMPLETE
        """
        if not self._built:
            self.build()

        # Gate 1: Is this actually a deletion request?
        if not case.is_deletion_request:
            return DeletionStatus.NOT_APPLICABLE

        # Gate 2: Check for violations
        has_violations = False

        # Check evidence records for policy violations
        for record in case.evidence_records:
            category = record.data_category
            required_action = self.get_required_action(category)
            state = record.state

            if required_action == RequiredAction.DELETE.value:
                # Record should be absent
                if state not in ("ABSENT", "NOT_FOUND"):
                    has_violations = True
                    break
            elif required_action == RequiredAction.REDACT.value:
                # Record can exist but must be redacted
                if state in ("PRESENT", "IDENTIFIABLE") and record.has_pii:
                    has_violations = True
                    break
            elif required_action == RequiredAction.ANONYMIZE.value:
                # Record can exist but must be anonymized
                if state in ("PRESENT", "IDENTIFIABLE") and record.has_pii:
                    has_violations = True
                    break
            elif required_action == RequiredAction.RETAIN.value:
                # Record should be retained — no violation
                pass

        # Check for active data flows
        if any(df.state == "ACTIVE" for df in case.data_flows):
            has_violations = True

        # Check for active restore risks
        if any(rr.state == "ACTIVE" for rr in case.restore_risks):
            has_violations = True

        # Gate 3: Final decision
        if has_violations:
            return DeletionStatus.INCOMPLETE
        else:
            return DeletionStatus.COMPLETE

    def categorize_support_record(self, resource_id: str, raw_data: dict = None) -> str:
        """
        Determine whether a support record is a MESSAGE or ATTACHMENT.
        
        Per rules:
        - SUPPORT_MESSAGE with shared context -> REDACT
        - SUPPORT_ATTACHMENT belonging only to subject -> DELETE
        """
        # Default to SUPPORT_MESSAGE (more conservative, leads to REDACT)
        return DataCategory.SUPPORT_MESSAGE.value
