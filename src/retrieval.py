"""
MaxSim retrieval engine with late interaction scoring.

Implements the ColBERT-style multi-vector retrieval:
  For each query token, find max similarity across page tokens, then sum.
"""

from typing import List, Dict, Tuple, Optional

import torch
import numpy as np


def compute_maxsim(query_emb: torch.Tensor, page_emb: torch.Tensor) -> float:
    """
    Compute MaxSim (late interaction) score.

    Args:
        query_emb: (n_query_tokens, dim)
        page_emb:  (n_page_tokens, dim)

    Returns:
        Scalar MaxSim score.
    """
    query_norm = query_emb / query_emb.norm(dim=1, keepdim=True)
    page_norm = page_emb / page_emb.norm(dim=1, keepdim=True)
    sim_matrix = query_norm @ page_norm.T
    return sim_matrix.max(dim=1).values.sum().item()


def retrieve_top_k(
    query: str,
    model,
    processor,
    page_embeddings: List[torch.Tensor],
    pages: List[Dict],
    top_k: int = 5,
) -> List[Tuple[float, int, Dict]]:
    """
    Retrieve top-k pages for a query using MaxSim.

    Returns list of (score, page_index, page_dict) sorted by score desc.
    """
    from .embedding import embed_query
    query_emb = embed_query(query, model, processor)

    scores = []
    for idx, page_emb in enumerate(page_embeddings):
        score = compute_maxsim(query_emb, page_emb)
        scores.append((score, idx, pages[idx]))

    scores.sort(key=lambda x: x[0], reverse=True)
    return scores[:top_k]


def retrieve_with_scores(
    query: str,
    model,
    processor,
    page_embeddings: List[torch.Tensor],
    pages: List[Dict],
) -> List[Tuple[float, int, Dict]]:
    """Retrieve ALL pages with scores (for evaluation)."""
    from .embedding import embed_query
    query_emb = embed_query(query, model, processor)

    scores = []
    for idx, page_emb in enumerate(page_embeddings):
        score = compute_maxsim(query_emb, page_emb)
        scores.append((score, idx, pages[idx]))

    scores.sort(key=lambda x: x[0], reverse=True)
    return scores


def get_similarity_map(
    query_emb: torch.Tensor,
    page_emb: torch.Tensor,
) -> np.ndarray:
    """
    Compute per-patch relevance map for visualization.

    Returns 1D array of per-page-token max similarity across query tokens.
    """
    query_norm = query_emb / query_emb.norm(dim=1, keepdim=True)
    page_norm = page_emb / page_emb.norm(dim=1, keepdim=True)
    sim_matrix = (query_norm @ page_norm.T).numpy()
    return sim_matrix.max(axis=0)
