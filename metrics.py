"""
Single shared implementation of WSS@r, RR@10%, and ATD, used by both
simulate.py (GAT/MLP simulation) and elas_simulation.py (ELAS u4 baseline).

Both callers reduce their own data representation down to a pair of
parallel arrays before calling into this module:

    screened : cumulative number of records screened at each step
    found    : cumulative number of relevant records found at that step


"""
from typing import Optional
import numpy as np


def wss_at_recall(
    screened: np.ndarray,
    found:    np.ndarray,
    n_total:     int,
    n_relevant:  int,
    target:      float,
) -> Optional[float]:
    """
    Work Saved over Sampling at a given recall target.

    WSS@r = r - (records_screened_to_reach_r_recall / n_total)

    Returns None if the target recall is never reached (should not happen
    once all relevant records are found, but guards against truncated runs).
    """
    recall_pct = found / n_relevant
    idx = int(np.searchsorted(recall_pct, target))
    if idx >= len(screened):
        return None
    return float(target - screened[idx] / n_total)


def rr_at_fraction(
    screened: np.ndarray,
    found:    np.ndarray,
    n_total:     int,
    n_relevant:  int,
    fraction:    float = 0.10,
) -> float:
    """
    Recall achieved after screening a given fraction of the corpus.
    Default fraction=0.10 gives RR@10%.
    """
    cutoff = max(1, int(fraction * n_total))
    idx = int(np.searchsorted(screened, cutoff))
    if idx < len(found):
        return float(found[idx] / n_relevant)
    return float(found[-1] / n_relevant)


def average_time_to_discovery(
    screened: np.ndarray,
    found:    np.ndarray,
    n_total:     int,
) -> Optional[float]:
    """
    ATD — mean normalised discovery rank across all relevant records found.
    Lower = relevant records are, on average, found earlier in the review.
    """
    steps, prev = [], 0
    for s, f in zip(screened, found):
        if f > prev:
            steps.extend([s / n_total] * int(f - prev))
            prev = int(f)
    return float(np.mean(steps)) if steps else None


def compute_all_metrics(
    screened: np.ndarray,
    found:    np.ndarray,
    n_total:     int,
    n_relevant:  int,
) -> dict:
    """
    Convenience wrapper returning WSS@95, WSS@100, RR@10%, and ATD together.
    `screened` and `found` must be parallel, non-decreasing arrays.
    """
    return {
        "wss95":    wss_at_recall(screened, found, n_total, n_relevant, 0.95),
        "wss100":   wss_at_recall(screened, found, n_total, n_relevant, 1.00),
        "rr_10pct": rr_at_fraction(screened, found, n_total, n_relevant, 0.10),
        "atd":      average_time_to_discovery(screened, found, n_total),
    }