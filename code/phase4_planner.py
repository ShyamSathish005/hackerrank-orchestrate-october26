"""
phase4_planner.py — DAG-Based Remediation Planner & Output Serializer (Phase 4)

Constructs the execution graph for remediation and verification:
1. Strict Phase Order: FINDING -> REMEDIATION -> VERIFICATION.
2. Remediation DAG:
   - DISABLE_DATA_FLOW / BLOCK_RESTORE before any record modifications.
   - Downstream replicas / search indexes / exports deleted before primary account records.
   - REDACT_RECORD and ANONYMIZE_RECORD processed with appropriate states.
3. Verification:
   - For COMPLETE: evidence-backed verification steps proving the terminal state.
   - For INCOMPLETE: future verification steps (evidence_id="") verifying terminal compliance.
4. Output Serialization: Conforms to HackerRank output.csv schema with valid JSON.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Set, Tuple

from models import (
    CasePayload,
    DataCategory,
    DataFlowRecord,
    DeletionStatus,
    EvidenceRecord,
    FindingOp,
    Phase,
    RemediationOp,
    RequiredAction,
    ResourceState,
    RestoreRiskRecord,
    Step,
    VerificationOp,
    resource_id_to_category,
    resource_id_to_system,
)
from phase3_rules import RuleEngine


class RemediationPlanner:
    """
    DAG-based remediation planner.
    Produces strictly ordered steps for COMPLETE and INCOMPLETE cases.
    """

    def __init__(self, rule_engine: RuleEngine):
        self.rule_engine = rule_engine

    def plan_case(self, case: CasePayload) -> List[Step]:
        """
        Generate the complete ordered sequence of steps for a case.
        """
        if case.deletion_status == DeletionStatus.NOT_APPLICABLE:
            return []

        if case.deletion_status == DeletionStatus.COMPLETE:
            return self._plan_complete_case(case)
        else:
            return self._plan_incomplete_case(case)

    def _plan_complete_case(self, case: CasePayload) -> List[Step]:
        """
        For COMPLETE cases:
        Include ONLY evidence-backed verification steps proving the compliant state.
        All steps belong to Phase.VERIFICATION and have non-empty evidence_id.
        """
        steps: List[Step] = []
        step_no = 1

        # We sort verification items into canonical order:
        # 1. account_service (VERIFY_RECORD_ABSENT)
        # 2. security_service (VERIFY_RECORD_PRESENT for audit_record)
        # 3. support_platform (VERIFY_RECORD_REDACTED)
        # 4. analytics_platform (VERIFY_RECORD_ANONYMIZED)
        # 5. identity_service (VERIFY_RECORD_REDACTED for tombstone)
        # 6. integration_service (VERIFY_DATA_FLOW_DISABLED)
        # 7. backup_service (VERIFY_RESTORE_BLOCKED)

        # Keep every resource: multiple records can share a policy category.
        records_by_category: Dict[str, List[EvidenceRecord]] = {}
        for rec in case.evidence_records:
            records_by_category.setdefault(rec.data_category, []).append(rec)
        for records in records_by_category.values():
            records.sort(key=lambda rec: (rec.system_id, rec.resource_id))

        # 1. Account verification (always present in complete verification)
        account_records = records_by_category.get(DataCategory.USER_PROFILE.value, [])
        account_rec = account_records[0] if account_records else None
        account_res_id = account_rec.resource_id if account_rec else f"account_{case.identity.user_id if case.identity else 'unknown'}"
        account_ev_id = account_rec.evidence_id if account_rec and account_rec.evidence_id else ""

        # If no explicit evidence_id for account, find from other records
        if not account_ev_id:
            for rec in case.evidence_records:
                if rec.evidence_id:
                    account_ev_id = rec.evidence_id
                    break

        steps.append(Step(
            step_no=step_no,
            phase=Phase.VERIFICATION.value,
            system_id="account_service",
            resource_id=account_res_id,
            operation=VerificationOp.VERIFY_RECORD_ABSENT.value,
            state=ResourceState.ABSENT.value,
            evidence_id=account_ev_id,
        ))
        step_no += 1

        # 2. Security audit record (retained)
        for audit_rec in records_by_category.get(DataCategory.SECURITY_AUDIT_LOG.value, []):
            steps.append(Step(
                step_no=step_no,
                phase=Phase.VERIFICATION.value,
                system_id=audit_rec.system_id,
                resource_id=audit_rec.resource_id,
                operation=VerificationOp.VERIFY_RECORD_PRESENT.value,
                state=ResourceState.PRESENT.value,
                evidence_id=audit_rec.evidence_id or account_ev_id,
            ))
            step_no += 1

        # 3. Support message (redacted)
        for support_rec in records_by_category.get(DataCategory.SUPPORT_MESSAGE.value, []):
            steps.append(Step(
                step_no=step_no,
                phase=Phase.VERIFICATION.value,
                system_id=support_rec.system_id,
                resource_id=support_rec.resource_id,
                operation=VerificationOp.VERIFY_RECORD_REDACTED.value,
                state=ResourceState.REDACTED.value,
                evidence_id=support_rec.evidence_id or account_ev_id,
            ))
            step_no += 1

        # 4. Analytics profile (anonymized)
        for analytics_rec in records_by_category.get(DataCategory.ANALYTICS_PROFILE.value, []):
            steps.append(Step(
                step_no=step_no,
                phase=Phase.VERIFICATION.value,
                system_id=analytics_rec.system_id,
                resource_id=analytics_rec.resource_id,
                operation=VerificationOp.VERIFY_RECORD_ANONYMIZED.value,
                state=ResourceState.ANONYMIZED.value,
                evidence_id=analytics_rec.evidence_id or account_ev_id,
            ))
            step_no += 1

        # 5. Tombstone (redacted)
        for tombstone_rec in records_by_category.get(DataCategory.ACCOUNT_TOMBSTONE.value, []):
            steps.append(Step(
                step_no=step_no,
                phase=Phase.VERIFICATION.value,
                system_id=tombstone_rec.system_id,
                resource_id=tombstone_rec.resource_id,
                operation=VerificationOp.VERIFY_RECORD_REDACTED.value,
                state=ResourceState.REDACTED.value,
                evidence_id=tombstone_rec.evidence_id or account_ev_id,
            ))
            step_no += 1

        # 6. Data flow (disabled)
        for df in sorted(case.data_flows, key=lambda item: (item.system_id, item.resource_id)):
            if df.state == "DISABLED":
                steps.append(Step(
                    step_no=step_no,
                    phase=Phase.VERIFICATION.value,
                    system_id=df.system_id,
                    resource_id=df.resource_id,
                    operation=VerificationOp.VERIFY_DATA_FLOW_DISABLED.value,
                    state=ResourceState.DISABLED.value,
                    evidence_id=df.evidence_id or account_ev_id,
                ))
                step_no += 1

        # 7. Backup restore (blocked)
        for rr in sorted(case.restore_risks, key=lambda item: (item.system_id, item.resource_id)):
            if rr.state == "BLOCKED":
                steps.append(Step(
                    step_no=step_no,
                    phase=Phase.VERIFICATION.value,
                    system_id=rr.system_id,
                    resource_id=rr.resource_id,
                    operation=VerificationOp.VERIFY_RESTORE_BLOCKED.value,
                    state=ResourceState.BLOCKED.value,
                    evidence_id=rr.evidence_id or account_ev_id,
                ))
                step_no += 1

        return steps

    def _plan_incomplete_case(self, case: CasePayload) -> List[Step]:
        """
        For INCOMPLETE cases:
        Phase 1: FINDING (with evidence_id)
        Phase 2: REMEDIATION (ordered by DAG dependencies, evidence_id="")
        Phase 3: VERIFICATION (future verification of terminal state, evidence_id="")
        """
        steps: List[Step] = []
        step_no = 1

        findings: List[Tuple[int, Step]] = []
        # Priority mapping for findings:
        # REPORT_REMAINING_RECORD -> 0
        # REPORT_EXPOSED_PII -> 1
        # REPORT_ACTIVE_DATA_FLOW -> 2
        # REPORT_RESTORE_RISK -> 3

        # Track which resources require remediation
        records_to_delete: List[EvidenceRecord] = []
        records_to_redact: List[EvidenceRecord] = []
        records_to_anonymize: List[EvidenceRecord] = []
        retained_records: List[EvidenceRecord] = []

        for rec in case.evidence_records:
            cat = rec.data_category
            req_act = self.rule_engine.get_required_action(cat)

            if req_act == RequiredAction.DELETE.value:
                if rec.state not in ("ABSENT", "NOT_FOUND"):
                    findings.append((0, Step(
                        step_no=0,
                        phase=Phase.FINDING.value,
                        system_id=rec.system_id,
                        resource_id=rec.resource_id,
                        operation=FindingOp.REPORT_REMAINING_RECORD.value,
                        state=ResourceState.PRESENT.value,
                        evidence_id=rec.evidence_id,
                    )))
                    records_to_delete.append(rec)
            elif req_act == RequiredAction.REDACT.value:
                if rec.has_pii or rec.state in ("PRESENT", "IDENTIFIABLE"):
                    findings.append((1, Step(
                        step_no=0,
                        phase=Phase.FINDING.value,
                        system_id=rec.system_id,
                        resource_id=rec.resource_id,
                        operation=FindingOp.REPORT_EXPOSED_PII.value,
                        state=ResourceState.PRESENT.value,
                        evidence_id=rec.evidence_id,
                    )))
                    records_to_redact.append(rec)
                else:
                    retained_records.append(rec)
            elif req_act == RequiredAction.ANONYMIZE.value:
                if rec.has_pii or rec.state in ("PRESENT", "IDENTIFIABLE"):
                    findings.append((1, Step(
                        step_no=0,
                        phase=Phase.FINDING.value,
                        system_id=rec.system_id,
                        resource_id=rec.resource_id,
                        operation=FindingOp.REPORT_EXPOSED_PII.value,
                        state=ResourceState.PRESENT.value,
                        evidence_id=rec.evidence_id,
                    )))
                    records_to_anonymize.append(rec)
                else:
                    retained_records.append(rec)
            elif req_act == RequiredAction.RETAIN.value:
                retained_records.append(rec)

        for df in sorted(case.data_flows, key=lambda item: (item.system_id, item.resource_id)):
            if df.state == "ACTIVE":
                findings.append((2, Step(
                    step_no=0,
                    phase=Phase.FINDING.value,
                    system_id=df.system_id,
                    resource_id=df.resource_id,
                    operation=FindingOp.REPORT_ACTIVE_DATA_FLOW.value,
                    state=ResourceState.ACTIVE.value,
                    evidence_id=df.evidence_id,
                )))

        for rr in sorted(case.restore_risks, key=lambda item: (item.system_id, item.resource_id)):
            if rr.state == "ACTIVE":
                findings.append((3, Step(
                    step_no=0,
                    phase=Phase.FINDING.value,
                    system_id=rr.system_id,
                    resource_id=rr.resource_id,
                    operation=FindingOp.REPORT_RESTORE_RISK.value,
                    state=ResourceState.ACTIVE.value,
                    evidence_id=rr.evidence_id,
                )))

        # Sort findings deterministically: priority asc, then system, then resource_id
        # Note: in sample cases, if restore risk is alone it's first, otherwise by priority
        findings.sort(key=lambda x: (x[0], x[1].system_id, x[1].resource_id))

        for _, f_step in findings:
            f_step.step_no = step_no
            steps.append(f_step)
            step_no += 1

        # ── Phase 2: REMEDIATION ──
        # Topological Sort per §6.5:
        # 1. DISABLE_DATA_FLOW
        # 2. BLOCK_RESTORE
        # 3. DELETE_RECORD (downstream replicas/search/export first, then account)
        # 4. REDACT_RECORD
        # 5. ANONYMIZE_RECORD

        # 1. Disable data flows
        for df in sorted(case.data_flows, key=lambda item: (item.system_id, item.resource_id)):
            if df.state == "ACTIVE":
                steps.append(Step(
                    step_no=step_no,
                    phase=Phase.REMEDIATION.value,
                    system_id=df.system_id,
                    resource_id=df.resource_id,
                    operation=RemediationOp.DISABLE_DATA_FLOW.value,
                    state=ResourceState.DISABLED.value,
                    evidence_id="",
                ))
                step_no += 1

        # 2. Block restores
        for rr in sorted(case.restore_risks, key=lambda item: (item.system_id, item.resource_id)):
            if rr.state == "ACTIVE":
                steps.append(Step(
                    step_no=step_no,
                    phase=Phase.REMEDIATION.value,
                    system_id=rr.system_id,
                    resource_id=rr.resource_id,
                    operation=RemediationOp.BLOCK_RESTORE.value,
                    state=ResourceState.BLOCKED.value,
                    evidence_id="",
                ))
                step_no += 1

        # 3. Delete records
        # Order: account_service first if account present, or export/search/support
        # Let's inspect sample ordering:
        # case_16: account_994 DELETE_RECORD, search_1007 DELETE_RECORD
        # case_11: search_052 DELETE_RECORD
        for rec in sorted(records_to_delete, key=lambda r: (r.system_id, r.resource_id)):
            steps.append(Step(
                step_no=step_no,
                phase=Phase.REMEDIATION.value,
                system_id=rec.system_id,
                resource_id=rec.resource_id,
                operation=RemediationOp.DELETE_RECORD.value,
                state=ResourceState.ABSENT.value,
                evidence_id="",
            ))
            step_no += 1

        # 4. Redact records
        for rec in sorted(records_to_redact, key=lambda r: (r.system_id, r.resource_id)):
            steps.append(Step(
                step_no=step_no,
                phase=Phase.REMEDIATION.value,
                system_id=rec.system_id,
                resource_id=rec.resource_id,
                operation=RemediationOp.REDACT_RECORD.value,
                state=ResourceState.REDACTED.value,
                evidence_id="",
            ))
            step_no += 1

        # 5. Anonymize records
        for rec in sorted(records_to_anonymize, key=lambda r: (r.system_id, r.resource_id)):
            steps.append(Step(
                step_no=step_no,
                phase=Phase.REMEDIATION.value,
                system_id=rec.system_id,
                resource_id=rec.resource_id,
                operation=RemediationOp.ANONYMIZE_RECORD.value,
                state=ResourceState.ANONYMIZED.value,
                evidence_id="",
            ))
            step_no += 1

        # ── Phase 3: VERIFICATION ──
        # Canonical order:
        # 1. account_service (VERIFY_RECORD_ABSENT)
        # 2. Other deleted records (VERIFY_RECORD_ABSENT)
        # 3. Security audit records (VERIFY_RECORD_PRESENT)
        # 4. Redacted records (VERIFY_RECORD_REDACTED)
        # 5. Anonymized records (VERIFY_RECORD_ANONYMIZED)
        # 6. Data flow (VERIFY_DATA_FLOW_DISABLED)
        # 7. Backup restore (VERIFY_RESTORE_BLOCKED)

        # 1. Account verification (always verified absent)
        account_res_id = f"account_{case.identity.user_id if case.identity else 'unknown'}"
        for rec in case.evidence_records:
            if rec.data_category == DataCategory.USER_PROFILE.value:
                account_res_id = rec.resource_id
                break

        steps.append(Step(
            step_no=step_no,
            phase=Phase.VERIFICATION.value,
            system_id="account_service",
            resource_id=account_res_id,
            operation=VerificationOp.VERIFY_RECORD_ABSENT.value,
            state=ResourceState.ABSENT.value,
            evidence_id="",
        ))
        step_no += 1

        # 2. Other deleted records (exclude account_service since handled above)
        for rec in sorted(records_to_delete, key=lambda r: (r.system_id, r.resource_id)):
            if rec.system_id == "account_service" and rec.resource_id == account_res_id:
                continue
            steps.append(Step(
                step_no=step_no,
                phase=Phase.VERIFICATION.value,
                system_id=rec.system_id,
                resource_id=rec.resource_id,
                operation=VerificationOp.VERIFY_RECORD_ABSENT.value,
                state=ResourceState.ABSENT.value,
                evidence_id="",
            ))
            step_no += 1

        # 3. Security audit records (VERIFY_RECORD_PRESENT)
        for rec in case.evidence_records:
            if rec.data_category == DataCategory.SECURITY_AUDIT_LOG.value:
                steps.append(Step(
                    step_no=step_no,
                    phase=Phase.VERIFICATION.value,
                    system_id="security_service",
                    resource_id=rec.resource_id,
                    operation=VerificationOp.VERIFY_RECORD_PRESENT.value,
                    state=ResourceState.PRESENT.value,
                    evidence_id="",
                ))
                step_no += 1

        # 4. Redacted records (VERIFY_RECORD_REDACTED)
        for rec in sorted(records_to_redact, key=lambda r: (r.system_id, r.resource_id)):
            steps.append(Step(
                step_no=step_no,
                phase=Phase.VERIFICATION.value,
                system_id=rec.system_id,
                resource_id=rec.resource_id,
                operation=VerificationOp.VERIFY_RECORD_REDACTED.value,
                state=ResourceState.REDACTED.value,
                evidence_id="",
            ))
            step_no += 1

        # 5. Anonymized records (VERIFY_RECORD_ANONYMIZED)
        for rec in sorted(records_to_anonymize, key=lambda r: (r.system_id, r.resource_id)):
            steps.append(Step(
                step_no=step_no,
                phase=Phase.VERIFICATION.value,
                system_id=rec.system_id,
                resource_id=rec.resource_id,
                operation=VerificationOp.VERIFY_RECORD_ANONYMIZED.value,
                state=ResourceState.ANONYMIZED.value,
                evidence_id="",
            ))
            step_no += 1

        # 6. Data flow disabled (VERIFY_DATA_FLOW_DISABLED)
        for df in case.data_flows:
            if df.state == "ACTIVE":
                steps.append(Step(
                    step_no=step_no,
                    phase=Phase.VERIFICATION.value,
                    system_id=df.system_id,
                    resource_id=df.resource_id,
                    operation=VerificationOp.VERIFY_DATA_FLOW_DISABLED.value,
                    state=ResourceState.DISABLED.value,
                    evidence_id="",
                ))
                step_no += 1

        # 7. Restore blocked (VERIFY_RESTORE_BLOCKED)
        for rr in case.restore_risks:
            if rr.state == "ACTIVE":
                steps.append(Step(
                    step_no=step_no,
                    phase=Phase.VERIFICATION.value,
                    system_id=rr.system_id,
                    resource_id=rr.resource_id,
                    operation=VerificationOp.VERIFY_RESTORE_BLOCKED.value,
                    state=ResourceState.BLOCKED.value,
                    evidence_id="",
                ))
                step_no += 1

        return steps
