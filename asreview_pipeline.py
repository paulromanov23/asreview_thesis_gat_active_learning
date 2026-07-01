"""
ELAS u4 simulation runner with WSS@95/WSS@100/RR@10%/ATD metric calculation.

Parameters are loaded directly from asreview.models.models.get_ai_config
so they are guaranteed to match the official ELAS u4 definition.

Supports multiple seeds and computes WSS@95, WSS@100, RR@10%, and ATD using
metrics.py's shared implementation.
"""
import pandas as pd
import numpy as np

from asreview import Simulate, ActiveLearningCycle
from asreview.models.classifiers import SVM
from asreview.models.feature_extractors import Tfidf
from asreview.models.balancers import Balanced
from asreview.models.queriers import Max
from asreview.models.models import get_ai_config

from metrics import compute_all_metrics


def n_query(results: list, n_records: int) -> int:
    """
    Determine the number of items to query in each active learning cycle.
    """
    if n_records >= 10000:
        if len(results) >= 10000:
            return 10**5
        if len(results) >= 1000:
            return 1000
        elif len(results) >= 100:
            return 25
        else:
            return 1
    else:
        if len(results) >= 1000:
            return 100
        elif len(results) >= 100:
            return 5
        else:
            return 1


def compute_metrics(sim_results, labels, total_records):
    """
    Compute WSS@95, WSS@100, RR@10%, and ATD from a completed ASReview
    Simulate run, via the shared metrics.py implementation.

    sim_results   : sim._results dataframe
    labels        : full labels array for the dataset
    total_records : total number of records in the dataset
    """
    total_positives = int(labels.sum())

    # Reconstruct full screening order — queried papers first, unqueried appended
    queried_ids   = sim_results["record_id"].values.astype(int)
    all_ids       = np.arange(total_records)
    unqueried_ids = np.setdiff1d(all_ids, queried_ids)
    full_order    = np.concatenate([queried_ids, unqueried_ids])

    # Labels in full screening order
    review_order_labels = labels[full_order]

    # screened/found parallel arrays, one entry per record in review order —
    # this is the same shape of input simulate.py's compute_metrics() builds
    # from its recall_curve, just derived from ELAS's flat order instead of
    # cycle-batched records.
    screened = np.arange(1, total_records + 1)
    found    = np.cumsum(review_order_labels)

    m = compute_all_metrics(screened, found, total_records, total_positives)

    # Diagnostic fields (not part of the shared metric, kept for logging parity
    # with the original script)
    recall_pct    = found / total_positives
    idx_at_95     = int(np.searchsorted(recall_pct, 0.95))
    idx_at_100    = int(np.searchsorted(recall_pct, 1.00))
    n_screened_95  = idx_at_95  + 1 if idx_at_95  < total_records else total_records
    n_screened_100 = idx_at_100 + 1 if idx_at_100 < total_records else total_records

    return {
        "wss95":          m["wss95"],
        "wss100":         m["wss100"],
        "rr_10pct":       m["rr_10pct"],
        "atd":            m["atd"],
        "n_screened_95":  n_screened_95,
        "n_screened_100": n_screened_100,
        "n_relevant":     total_positives,
        "n_total":        int(total_records),
    }


def run_elas_simulation(data_path: str, n_seeds: int = 3, dataset_name: str = None):
    """
    Run ELAS u4 simulation over multiple seeds and report mean ± std.

    data_path   : path to CSV with 'title', 'abstract', 'label_included' columns
    n_seeds     : number of random seeds to run
    dataset_name: optional display name; defaults to data_path
    """
    if dataset_name is None:
        dataset_name = data_path

    # Load data
    data   = pd.read_csv(data_path)
    X      = data[["title", "abstract"]].fillna("")
    labels = data["label_included"].values.astype(int)

    total_records   = len(labels)
    total_positives = int(labels.sum())

    print(f"\n{'='*60}")
    print(f"Dataset : {dataset_name}")
    print(f"N papers: {total_records:,}  |  N relevant: {total_positives}  |  "
          f"Prevalence: {total_positives/total_records:.2%}")
    print(f"Seeds   : {n_seeds}")
    print(f"{'='*60}")

    # Load ELAS u4 config
    config = get_ai_config("elas_u4").get("value")

    # Run over seeds
    all_metrics = []

    for seed in range(n_seeds):
        rng       = np.random.default_rng(seed)
        prior_pos = int(rng.choice(np.where(labels == 1)[0]))
        prior_neg = int(rng.choice(np.where(labels == 0)[0]))

        cycle = ActiveLearningCycle(
            querier=Max(),
            classifier=SVM(**config.classifier_param),
            balancer=Balanced(**config.balancer_param),
            feature_extractor=Tfidf(**config.feature_extractor_param),
            n_query=lambda results, n=total_records: n_query(results, n),
        )

        sim = Simulate(X=X, labels=labels, cycles=[cycle], print_progress=False)
        sim.label([prior_pos, prior_neg])
        sim.review()

        m = compute_metrics(sim._results, labels, total_records)
        all_metrics.append(m)

        print(f"  Seed {seed} | prior_pos={prior_pos:5d}, prior_neg={prior_neg:5d} | "
              f"WSS@95={m['wss95']:.3f}  WSS@100={m['wss100']:.3f}  "
              f"RR@10%={m['rr_10pct']:.3f}  ATD={m['atd']:.4f}")

    # Aggregate
    def agg(key):
        vals = [m[key] for m in all_metrics if m[key] is not None]
        return np.mean(vals), np.std(vals)

    wss95_mean,  wss95_std  = agg("wss95")
    wss100_mean, wss100_std = agg("wss100")
    rr_mean,     rr_std     = agg("rr_10pct")
    atd_mean,    atd_std    = agg("atd")

    print(f"\n── Results: {dataset_name} ({n_seeds} seeds) ──────────────────")
    print(f"  WSS@95  : {wss95_mean:.3f} ± {wss95_std:.3f}")
    print(f"  WSS@100 : {wss100_mean:.3f} ± {wss100_std:.3f}")
    print(f"  RR@10%  : {rr_mean:.3f} ± {rr_std:.3f}")
    print(f"  ATD     : {atd_mean:.4f} ± {atd_std:.4f}")

    return all_metrics


# Entry point

if __name__ == "__main__":
    datasets = {
        "name":   "data/name_of_dataset.csv",
    }

    for name, path in datasets.items():
        run_elas_simulation(path, n_seeds=3, dataset_name=name)