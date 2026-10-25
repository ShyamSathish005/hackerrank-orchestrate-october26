"""
main.py — Main Entry Point for Northstar Shop Deletion Audit Agent

Orchestrates the 4-phase Neuro-Symbolic Pipeline:
- Phase 1: Deterministic Identity Graph (Symbolic)
- Phase 2: Evidence Parser and Entity Extractor (Probabilistic/Indexed)
- Phase 3: Process-Aware Verification State Machine (Symbolic Rule Engine)
- Phase 4: DAG-Based Remediation Planner & Output Serializer

Conforms to HackerRank challenge specification:
- Reads dataset from `dataset/`
- Processes all cases in `dataset/deletion_requests.csv` (or sample cases)
- Writes valid `output.csv` to repository root with columns:
  case_id,deletion_status,steps_json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, List, Optional

import pandas as pd

from models import CasePayload, DeletionStatus, Step
from phase1_identity import IdentityGraph
from phase2_evidence import EvidenceIndex
from phase3_rules import RuleEngine, is_deletion_request
from phase4_planner import RemediationPlanner


class DeletionAuditPipeline:
    """
    End-to-end pipeline orchestrating all 4 phases.
    """

    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir
        self.identity_graph = IdentityGraph(dataset_dir)
        self.evidence_index = EvidenceIndex(dataset_dir)
        self.rule_engine = RuleEngine(dataset_dir)
        self.planner = RemediationPlanner(self.rule_engine)
        self._initialized = False

    def initialize(self) -> None:
        """Pre-index graph, evidence, and rules."""
        if self._initialized:
            return

        t0 = time.time()
        print("[-] Phase 1: Building Identity Graph...")
        self.identity_graph.build()

        print("[-] Phase 2: Indexing Evidence and Activity...")
        self.evidence_index.build()

        print("[-] Phase 3: Loading Retention Rules...")
        self.rule_engine.build()

        t_elapsed = time.time() - t0
        print(f"[+] All phases initialized in {t_elapsed:.2f}s")
        self._initialized = True

    def audit_case(self, row: dict) -> CasePayload:
        """
        Run the complete 4-phase audit on a single request.
        """
        case_id = row["case_id"]
        email = row["email"].strip()
        requested_at = row.get("requested_at", "")
        reported_status = row.get("reported_status", "")
        reported_at = row.get("reported_at", "")
        audit_at = row.get("audit_at", "")
        request_text = row.get("request_text", "")

        case = CasePayload(
            case_id=case_id,
            email=email,
            requested_at=requested_at,
            reported_status=reported_status,
            reported_at=reported_at,
            audit_at=audit_at,
            request_text=request_text,
        )

        # Gate 1: Check request authorization
        case.is_deletion_request = is_deletion_request(request_text)
        if not case.is_deletion_request:
            case.deletion_status = DeletionStatus.NOT_APPLICABLE
            case.steps = []
            return case

        # Phase 1: Resolve identity cluster
        case.identity = self.identity_graph.resolve(email)
        if not case.identity:
            # Fallback if email not found in identity database
            from models import IdentityCluster
            case.identity = IdentityCluster(
                user_id="",
                current_email=email,
                current_name="",
                current_phone="",
                current_email_hash="",
                all_emails={email},
            )

        # Phase 2: Evidence discovery
        ev_records, data_flows, restore_risks = self.evidence_index.query_for_user(
            identity=case.identity,
            audit_at=audit_at,
        )
        case.evidence_records = ev_records
        case.data_flows = data_flows
        case.restore_risks = restore_risks

        # Phase 3: Rule engine evaluation
        case.deletion_status = self.rule_engine.evaluate_case(case)

        # Phase 4: DAG Remediation Planning
        case.steps = self.planner.plan_case(case)

        return case

    def run(
        self,
        requests_file: str,
        output_file: str,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Process all requests and write output.csv.
        """
        self.initialize()

        req_df = pd.read_csv(requests_file, dtype=str).fillna("")
        if limit:
            req_df = req_df.head(limit)

        print(f"[*] Processing {len(req_df)} cases from {requests_file}...")
        results = []

        status_counts = {
            DeletionStatus.COMPLETE: 0,
            DeletionStatus.INCOMPLETE: 0,
            DeletionStatus.NOT_APPLICABLE: 0,
        }

        t_start = time.time()
        for idx, (_, row) in enumerate(req_df.iterrows(), 1):
            case = self.audit_case(row.to_dict())
            status_counts[case.deletion_status] += 1

            results.append({
                "case_id": case.case_id,
                "deletion_status": case.deletion_status.value,
                "steps_json": case.steps_json(),
            })

            if idx % 50 == 0 or idx == len(req_df):
                print(f"    Processed {idx}/{len(req_df)} cases...")

        t_end = time.time()
        out_df = pd.DataFrame(results)

        # Sort by case_id as required
        out_df = out_df.sort_values("case_id").reset_index(drop=True)

        # Write CSV with exact formatting
        out_df.to_csv(output_file, index=False, quoting=csv.QUOTE_MINIMAL)
        print(f"[+] Output written to {output_file} ({len(out_df)} rows) in {t_end - t_start:.2f}s")
        print(f"    COMPLETE:       {status_counts[DeletionStatus.COMPLETE]}")
        print(f"    INCOMPLETE:     {status_counts[DeletionStatus.INCOMPLETE]}")
        print(f"    NOT_APPLICABLE: {status_counts[DeletionStatus.NOT_APPLICABLE]}")

        return out_df


def main():
    parser = argparse.ArgumentParser(description="Northstar Shop Deletion Audit Agent")
    parser.add_argument(
        "--dataset_dir",
        default="dataset",
        help="Path to dataset directory (default: dataset)",
    )
    parser.add_argument(
        "--requests",
        default="dataset/deletion_requests.csv",
        help="Path to deletion requests CSV",
    )
    parser.add_argument(
        "--output",
        default="output.csv",
        help="Output CSV path (default: output.csv)",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Run on sample_deletion_requests.csv instead of full evaluation set",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of requests to process",
    )

    args = parser.parse_args()

    # Determine requests path
    req_path = args.requests
    if args.sample:
        req_path = os.path.join(args.dataset_dir, "sample_deletion_requests.csv")

    pipeline = DeletionAuditPipeline(dataset_dir=args.dataset_dir)
    pipeline.run(
        requests_file=req_path,
        output_file=args.output,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
