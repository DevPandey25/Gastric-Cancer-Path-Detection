"""In-Silico miRNA Regulatory Network Construction and Visualization Engine.

Builds graph structures connecting microRNA hsa-miR-497-5p to its predicted target genes:
- Graph representation of 3,711 predicted interactions weighted by consensus algorithms.
- Subnetwork extraction based on algorithm consensus filtering (e.g. consensus >= 7).
- Network centrality and connectivity metrics.
- Publication-grade network graph and prediction distribution visualizations.
- Fast query interface for downstream API and dashboard exploration.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from research.multimodal.biomarker_analysis import BiomarkerWorkbookAuditor

logger = logging.getLogger("multimodal.network")


class MolecularNetworkEngine:
    """
    Constructs, queries, and visualizes miRNA-target interaction networks.
    """

    def __init__(self, excel_path: Union[str, Path] = "data/support/Gastric.xlsx") -> None:
        self.auditor = BiomarkerWorkbookAuditor(excel_path)
        self._df: Optional[pd.DataFrame] = None

    def _get_dataframe(self) -> pd.DataFrame:
        if self._df is None:
            self._df = self.auditor.load_workbook_sheet()
        return self._df

    def build_interaction_graph(self, min_consensus: int = 7) -> nx.DiGraph:
        """
        Builds a directed NetworkX graph where edges point from miRNA to target genes,
        weighted by algorithm consensus count.
        """
        df = self._get_dataframe()
        sub_df = df[df["predicted.sum"] >= min_consensus].dropna(subset=["target_symbol"]).copy()

        g = nx.DiGraph()
        mirna_id = "hsa-miR-497-5p"
        g.add_node(mirna_id, node_type="miRNA", label=mirna_id)

        for _, row in sub_df.iterrows():
            gene = str(row["target_symbol"]).strip()
            score = int(row["predicted.sum"])
            all_sum = int(row["all.sum"])

            g.add_node(gene, node_type="target_gene", label=gene, consensus=score, total_score=all_sum)
            g.add_edge(mirna_id, gene, weight=score, total_score=all_sum)

        return g

    def query_targets(
        self,
        gene_symbol: Optional[str] = None,
        min_consensus: int = 5,
        top_k: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Searchable exploration interface for querying predicted miRNA targets.
        """
        df = self._get_dataframe()
        query_df = df[df["predicted.sum"] >= min_consensus].dropna(subset=["target_symbol"]).copy()

        if gene_symbol:
            query_df = query_df[query_df["target_symbol"].str.upper().str.contains(gene_symbol.upper().strip())]

        query_df = query_df.sort_values(by=["predicted.sum", "all.sum"], ascending=False).head(top_k)

        results: List[Dict[str, Any]] = []
        for _, row in query_df.iterrows():
            results.append({
                "mirna_id": str(row.get("mature_mirna_id")),
                "target_symbol": str(row.get("target_symbol")),
                "entrez_id": int(row["target_entrez"]) if pd.notna(row.get("target_entrez")) else None,
                "ensembl_id": str(row.get("target_ensembl")),
                "consensus_databases_count": int(row["predicted.sum"]),
                "total_score_sum": int(row["all.sum"]),
                "supported_algorithms": [
                    algo for algo in ["diana_microt", "elmmo", "microcosm", "miranda", "mirdb", "pictar", "pita", "targetscan"]
                    if int(row.get(algo, 0)) > 0
                ],
            })
        return results

    def render_consensus_distribution_figure(self, output_path: Union[str, Path]) -> str:
        """
        Renders bar chart showing distribution of consensus across 1 to 8 prediction databases.
        """
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        df = self._get_dataframe()
        counts = df["predicted.sum"].value_counts().sort_index()

        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar(counts.index, counts.values, color="#4A90E2", edgecolor="black", alpha=0.85)

        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h + 20, f"{h}", ha="center", fontsize=9, fontweight="bold")

        ax.set_xlabel("Number of Computational Prediction Databases Supporting Interaction (1 to 8)")
        ax.set_ylabel("Number of Predicted Target Genes (Total: 3,711)")
        ax.set_title("hsa-miR-497-5p In-Silico Target Prediction Consensus Distribution", fontweight="bold")
        ax.set_xticks(range(1, 9))
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

        plt.tight_layout()
        fig.savefig(str(out_p), dpi=300, bbox_inches="tight")
        plt.close(fig)
        return str(out_p.resolve()).replace("\\", "/")

    def render_molecular_network_figure(self, output_path: Union[str, Path], top_n_genes: int = 30) -> str:
        """
        Renders a spring-layout bipartite network graph connecting hsa-miR-497-5p to its top target genes.
        """
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        df = self._get_dataframe()
        # Filter top N target genes with highest consensus and score
        top_df = df.dropna(subset=["target_symbol"]).sort_values(by=["predicted.sum", "all.sum"], ascending=False).head(top_n_genes)

        g = nx.Graph()
        mirna_node = "hsa-miR-497-5p"
        g.add_node(mirna_node, node_type="mirna")

        for _, row in top_df.iterrows():
            gene = str(row["target_symbol"]).strip()
            score = int(row["predicted.sum"])
            g.add_node(gene, node_type="gene", score=score)
            g.add_edge(mirna_node, gene, weight=score)

        fig, ax = plt.subplots(figsize=(11, 11))
        pos = nx.spring_layout(g, seed=42, k=0.45)

        # Draw nodes
        gene_nodes = [n for n, attr in g.nodes(data=True) if attr.get("node_type") == "gene"]
        gene_scores = [g.nodes[n]["score"] for n in gene_nodes]

        # Draw miRNA center
        nx.draw_networkx_nodes(
            g, pos, nodelist=[mirna_node], node_color="#E02020", node_size=1800, alpha=0.9, edgecolors="black", ax=ax
        )
        # Draw target genes colored by consensus
        nodes_scatter = nx.draw_networkx_nodes(
            g, pos, nodelist=gene_nodes, node_color=gene_scores, cmap=plt.cm.Blues,
            node_size=800, vmin=6, vmax=8, edgecolors="black", linewidths=0.8, ax=ax
        )
        # Draw edges
        nx.draw_networkx_edges(g, pos, alpha=0.4, edge_color="gray", width=1.5, ax=ax)
        # Labels
        nx.draw_networkx_labels(g, pos, font_size=8, font_weight="bold", ax=ax)

        # Colorbar for consensus score
        cbar = plt.colorbar(nodes_scatter, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label("Algorithm Consensus (Databases: 1-8)", rotation=-90, va="bottom")

        ax.set_title(
            f"In-Silico Regulatory Subnetwork: hsa-miR-497-5p and Top {top_n_genes} High-Confidence Target Genes\n"
            "(Node color represents bioinformatics consensus score)",
            fontweight="bold",
            fontsize=11,
            pad=10,
        )
        ax.axis("off")

        plt.tight_layout()
        fig.savefig(str(out_p), dpi=300, bbox_inches="tight")
        plt.close(fig)
        return str(out_p.resolve()).replace("\\", "/")

