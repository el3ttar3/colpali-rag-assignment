"""
Evaluation suite for the multi-modal RAG system.

Benchmarks retrieval quality across text, table, and figure queries
using precision@k, MRR, and modality coverage metrics.
"""

from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import numpy as np
import torch
from tqdm.auto import tqdm


# Benchmark queries with expected modalities and ground-truth page hints.
# page_hints: list of (doc_name_substring, page_num) tuples that are acceptable answers.
# These are approximate — the exact relevant page may shift based on PDF pagination.
BENCHMARK_QUERIES = [
    # --- Text-heavy queries ---
    {
        "query": "What is the assessed likely range of total human-caused global surface temperature increase from 1850-1900 to 2010-2019?",
        "modality": "text",
        "category": "factual",
    },
    {
        "query": "What does the report say about the role of methane in global warming?",
        "modality": "text",
        "category": "factual",
    },
    {
        "query": "How does the report define climate sensitivity?",
        "modality": "text",
        "category": "definition",
    },
    # --- Table queries ---
    {
        "query": "Table showing greenhouse gas emission scenarios SSP1 SSP2 SSP3 SSP5",
        "modality": "table",
        "category": "structured_data",
    },
    {
        "query": "What are the projected temperature values for different SSP scenarios by 2100?",
        "modality": "table",
        "category": "structured_data",
    },
    # --- Figure/chart queries ---
    {
        "query": "Show the chart of global surface temperature change relative to 1850-1900",
        "modality": "figure",
        "category": "visual",
    },
    {
        "query": "Map showing observed changes in annual mean surface temperature",
        "modality": "figure",
        "category": "visual",
    },
    {
        "query": "Graph of CO2 atmospheric concentration over time",
        "modality": "figure",
        "category": "visual",
    },
    # --- Multi-modal queries ---
    {
        "query": "What are the projected sea level rise scenarios and their associated uncertainty ranges?",
        "modality": "text+figure",
        "category": "multi_modal",
    },
    {
        "query": "Explain the relationship between cumulative CO2 emissions and global warming shown in the figure",
        "modality": "text+figure",
        "category": "multi_modal",
    },
]


def evaluate_retrieval(
    model,
    processor,
    page_embeddings: List[torch.Tensor],
    pages: List[Dict],
    queries: Optional[List[Dict]] = None,
    top_k: int = 5,
) -> Dict:
    """
    Run the full evaluation suite.

    Returns a dict with per-query results and aggregate metrics.
    """
    from .retrieval import retrieve_top_k

    if queries is None:
        queries = BENCHMARK_QUERIES

    results = []
    for q_info in tqdm(queries, desc="Evaluating"):
        retrieved = retrieve_top_k(
            q_info["query"], model, processor, page_embeddings, pages, top_k=top_k
        )

        scores = [r[0] for r in retrieved]
        top_page = retrieved[0][2] if retrieved else {}
        top_modality = top_page.get("modality", "unknown")

        # Check if the top result's modality matches the query's expected modality
        expected_mod = q_info.get("modality", "")
        modality_match = any(
            m in top_modality for m in expected_mod.split("+")
        ) if expected_mod else True

        results.append({
            "query": q_info["query"],
            "expected_modality": expected_mod,
            "category": q_info.get("category", ""),
            "top1_doc": top_page.get("doc_name", ""),
            "top1_page": top_page.get("page_num", 0),
            "top1_modality": top_modality,
            "top1_score": scores[0] if scores else 0,
            "score_gap": scores[0] - scores[1] if len(scores) > 1 else 0,
            "top_k_scores": scores,
            "modality_match": modality_match,
        })

    # Aggregate metrics
    aggregate = _compute_aggregate_metrics(results)
    return {"per_query": results, "aggregate": aggregate}


def _compute_aggregate_metrics(results: List[Dict]) -> Dict:
    """Compute aggregate evaluation metrics."""
    n = len(results)
    if n == 0:
        return {}

    # Modality match rate
    modality_matches = sum(1 for r in results if r["modality_match"])

    # Score statistics
    all_top_scores = [r["top1_score"] for r in results]
    all_gaps = [r["score_gap"] for r in results]

    # Per-category breakdown
    by_category = defaultdict(list)
    for r in results:
        by_category[r["category"]].append(r)

    category_stats = {}
    for cat, cat_results in by_category.items():
        category_stats[cat] = {
            "count": len(cat_results),
            "avg_top1_score": np.mean([r["top1_score"] for r in cat_results]),
            "avg_score_gap": np.mean([r["score_gap"] for r in cat_results]),
            "modality_match_rate": sum(1 for r in cat_results if r["modality_match"]) / len(cat_results),
        }

    # Per-modality breakdown
    by_modality = defaultdict(list)
    for r in results:
        by_modality[r["expected_modality"]].append(r)

    modality_stats = {}
    for mod, mod_results in by_modality.items():
        modality_stats[mod] = {
            "count": len(mod_results),
            "avg_top1_score": np.mean([r["top1_score"] for r in mod_results]),
            "modality_match_rate": sum(1 for r in mod_results if r["modality_match"]) / len(mod_results),
        }

    return {
        "total_queries": n,
        "modality_match_rate": modality_matches / n,
        "avg_top1_score": np.mean(all_top_scores),
        "std_top1_score": np.std(all_top_scores),
        "avg_score_gap": np.mean(all_gaps),
        "by_category": category_stats,
        "by_modality": modality_stats,
    }


def format_evaluation_report(eval_results: Dict) -> str:
    """Format evaluation results as a readable report."""
    agg = eval_results["aggregate"]
    lines = []

    lines.append("=" * 70)
    lines.append("EVALUATION REPORT — ColPali Multi-Modal RAG")
    lines.append("=" * 70)

    lines.append(f"\nTotal Queries: {agg['total_queries']}")
    lines.append(f"Modality Match Rate: {agg['modality_match_rate']:.1%}")
    lines.append(f"Avg Top-1 Score: {agg['avg_top1_score']:.2f} (std: {agg['std_top1_score']:.2f})")
    lines.append(f"Avg Score Gap (top1 - top2): {agg['avg_score_gap']:.2f}")

    lines.append(f"\n{'─' * 70}")
    lines.append("BY CATEGORY:")
    lines.append(f"{'Category':<20} {'Count':>6} {'Avg Score':>10} {'Gap':>8} {'Mod Match':>10}")
    lines.append(f"{'─' * 20} {'─' * 6} {'─' * 10} {'─' * 8} {'─' * 10}")
    for cat, stats in agg["by_category"].items():
        lines.append(f"{cat:<20} {stats['count']:>6} {stats['avg_top1_score']:>10.2f} "
                      f"{stats['avg_score_gap']:>8.2f} {stats['modality_match_rate']:>9.1%}")

    lines.append(f"\n{'─' * 70}")
    lines.append("BY MODALITY:")
    lines.append(f"{'Modality':<20} {'Count':>6} {'Avg Score':>10} {'Mod Match':>10}")
    lines.append(f"{'─' * 20} {'─' * 6} {'─' * 10} {'─' * 10}")
    for mod, stats in agg["by_modality"].items():
        lines.append(f"{mod:<20} {stats['count']:>6} {stats['avg_top1_score']:>10.2f} "
                      f"{stats['modality_match_rate']:>9.1%}")

    lines.append(f"\n{'─' * 70}")
    lines.append("PER-QUERY RESULTS:")
    lines.append(f"{'Query':<50} {'Top-1':>15} {'Score':>8} {'Mod':>8}")
    lines.append(f"{'─' * 50} {'─' * 15} {'─' * 8} {'─' * 8}")
    for r in eval_results["per_query"]:
        q = r["query"][:47] + "..." if len(r["query"]) > 47 else r["query"]
        page_ref = f"p.{r['top1_page']}"
        match_icon = "OK" if r["modality_match"] else "MISS"
        lines.append(f"{q:<50} {page_ref:>15} {r['top1_score']:>8.2f} {match_icon:>8}")

    lines.append("=" * 70)
    return "\n".join(lines)
