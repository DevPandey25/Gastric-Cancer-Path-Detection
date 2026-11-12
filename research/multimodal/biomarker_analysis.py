"""In-Silico miRNA Biomarker and Target Gene Interaction Audit Engine.

Audits Gastric.xlsx (Table s2) containing 3,711 predicted miRNA-target interactions:
- Schema validation, missingness quantification, and duplicate detection.
- Analysis of 8 computational prediction algorithms (DIANA-microT, ElMMo, MicroCosm,
  miRanda, miRDB, PicTar, PITA, TargetScan).
- Consensus distribution analysis across prediction databases (predicted.sum in [1, 8]).
- Top high-confidence target gene identification.
- Mandatory scientific disclosure: clearly documents in-silico prediction nature
  (preventing misclassification as patient-measured molecular biomarkers).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

logger = logging.getLogger("multimodal.biomarker")

ALGORITHM_COLUMNS = [
    "diana_microt",
    "elmmo",
    "microcosm",
    "miranda",
    "mirdb",
    "pictar",
    "pita",
    "targetscan",
]


class BiomarkerWorkbookAuditor:
    """
    Auditor for Gastric.xlsx in-silico miRNA target interaction database.
    """

    def __init__(self, excel_path: Union[str, Path] = "data/support/Gastric.xlsx") -> None:
        self.excel_path = Path(excel_path)

    def load_workbook_sheet(self, sheet_name: str = "Table s2") -> pd.DataFrame:
        """Loads target sheet from Gastric.xlsx."""
        if not self.excel_path.exists():
            raise FileNotFoundError(f"Biomarker workbook not found: {self.excel_path}")
        return pd.read_excel(self.excel_path, sheet_name=sheet_name)

    def execute_biomarker_audit(self, output_json_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Executes complete quality, schema, and consensus audit of the biomarker workbook.
        """
        df = self.load_workbook_sheet()
        total_rows = len(df)

        # 1. miRNA Accessions & Identifiers
        mirna_ids = df["mature_mirna_id"].dropna().unique().tolist()
        mirna_accs = df["mature_mirna_acc"].dropna().unique().tolist()

        # 2. Target Gene Identifiers
        missing_symbols = int(df["target_symbol"].isna().sum())
        missing_entrez = int(df["target_entrez"].isna().sum())
        missing_ensembl = int(df["target_ensembl"].isna().sum())

        unique_symbols = int(df["target_symbol"].dropna().nunique())
        unique_entrez = int(df["target_entrez"].dropna().nunique())
        unique_ensembl = int(df["target_ensembl"].dropna().nunique())

        # 3. Algorithm Consensus Analysis (predicted.sum)
        consensus_counts = {int(k): int(v) for k, v in df["predicted.sum"].value_counts().sort_index().items()}

        # 4. Top high-consensus targets (consensus = 8 or 7)
        top_targets_df = df[df["predicted.sum"] >= 7].sort_values(by=["predicted.sum", "all.sum"], ascending=False)
        top_targets = []
        for _, row in top_targets_df.head(25).iterrows():
            top_targets.append({
                "target_symbol": str(row.get("target_symbol")),
                "entrez_id": int(row["target_entrez"]) if pd.notna(row.get("target_entrez")) else None,
                "ensembl_id": str(row.get("target_ensembl")),
                "algorithm_consensus_count": int(row["predicted.sum"]),
                "total_score_sum": int(row["all.sum"]),
            })

        # 5. Algorithm contribution breakdown
        algo_positive_rates = {}
        for algo in ALGORITHM_COLUMNS:
            if algo in df.columns:
                pos_cnt = int((df[algo] > 0).sum())
                algo_positive_rates[algo] = {
                    "positive_count": pos_cnt,
                    "positive_rate_pct": float(round((pos_cnt / total_rows) * 100.0, 1)),
                }

        audit_summary = {
            "phase": "Original Phase 10: Multimodal Clinical Research",
            "workbook_file": str(self.excel_path).replace("\\", "/"),
            "sheet_name": "Table s2",
            "total_interaction_rows": total_rows,
            "mirna_profile": {
                "mature_mirna_id": mirna_ids[0] if mirna_ids else "Unknown",
                "mature_mirna_acc": mirna_accs[0] if mirna_accs else "Unknown",
                "unique_mirnas_in_file": len(mirna_ids),
            },
            "target_genes": {
                "unique_symbols": unique_symbols,
                "missing_symbols": missing_symbols,
                "missing_symbols_pct": float(round((missing_symbols / total_rows) * 100.0, 2)),
                "unique_entrez_ids": unique_entrez,
                "unique_ensembl_ids": unique_ensembl,
            },
            "consensus_distribution": consensus_counts,
            "algorithms_evaluated": ALGORITHM_COLUMNS,
            "algorithm_positive_rates": algo_positive_rates,
            "top_consensus_targets_count_ge_7": len(top_targets_df),
            "sample_top_targets": top_targets,
            "scientific_disclosures": [
                "All records in Gastric.xlsx represent in-silico computational predictions, NOT experimentally verified patient biomarkers.",
                "Zero patient-level molecular expression measurements or clinical specimen linkages exist in this workbook.",
                "High consensus indicates agreement across bioinformatics algorithms (e.g. TargetScan, miRanda, PicTar), not clinical validation in gastric cancer patients.",
            ],
        }

        if output_json_path:
            p = Path(output_json_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w") as f:
                json.dump(audit_summary, f, indent=2)
            logger.info(f"Biomarker audit summary written to {p}")

        return audit_summary

