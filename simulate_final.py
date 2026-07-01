"""
ASReview-aligned active learning simulation for GAT (or MLP, for the
ablation) on pre-built PyG graphs.

Loop contract (mirrors ASReview's core cycle):
  1. Prior knowledge: seed 1 relevant + 1 irrelevant (ASReview default).
  2. Train model on current labeled mask.
  3. Query: score all unlabeled nodes → pick top-1 (max strategy, ASReview default).
  4. Oracle: reveal true label (simulation knows ground truth).
  5. Add to labeled set. Retrain. Repeat.
  6. Stop when all relevant records found (ASReview --stop-if min).

Key design decisions:
  - n_instances = 1  (one paper shown per cycle, default in ASReview)
  - Max query strategy (certainty-based, ASReview default)
  - Separate eval forward pass with model.eval() + no_grad
  - No validation set — mirrors ASReview exactly, trains on all labeled nodes
  - Score caching: forward pass only runs when model is retrained
  - Phase switching: cosine fallback triggered when scoring model stagnates
  - unlabeled_np maintained incrementally (O(1) per step, not O(N))

model_class:
  Defaults to GATClassifier for the main experiment. Pass
  model_class=MLPClassifier to run the edge-free ablation (isolates the
  contribution of sBERT features alone, no message passing) — this is how
  the Brouwer raw-graph / MLP ablation comparisons were produced.
  NOTE: a previous local copy of this file had model_class hardcoded to
  MLPClassifier inside run_simulation() left over from that ablation run.
  That hardcoding is removed here — model_class is a first-class parameter
  again, defaulting back to GATClassifier.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import time

from metrics import compute_all_metrics


# Models

class GATClassifier(torch.nn.Module):
    """
    2-layer GAT for binary node classification.
    in_channels    : sBERT embedding dim (768 for all-mpnet-base-v2)
    hidden_channels: per-head hidden dim
    heads          : attention heads in layer 1
    dropout        : applied to input and between layers
    """
    def __init__(self, in_channels: int, hidden_channels: int = 64,
                 heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GATConv(in_channels, hidden_channels,
                             heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, 1,
                             heads=1, concat=False, dropout=dropout)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x.squeeze(-1)   # [num_nodes]


class MLPClassifier(torch.nn.Module):
    """
    Same capacity as GATClassifier but no graph edges.
    Used to isolate the contribution of graph structure vs sBERT features alone.
    """
    def __init__(self, in_channels: int, hidden_channels: int = 64,
                 heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout
        # Match GAT capacity: hidden_channels * heads in layer 1
        self.fc1 = torch.nn.Linear(in_channels, hidden_channels * heads)
        self.fc2 = torch.nn.Linear(hidden_channels * heads, 1)

    def forward(self, x, edge_index=None):   # edge_index accepted but ignored
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc2(x).squeeze(-1)


# Training

def train_one_cycle(
    graph,
    train_mask:      torch.Tensor,
    device:          torch.device,
    model_class,
    epochs:          int   = 200,
    lr:              float = 1e-4,
    hidden_channels: int   = 64,
    heads:           int   = 4,
    dropout:         float = 0.3,
):
    """
    Train a fresh model on the current labeled set (train_mask).
    Returns the final epoch weights — no val set, no checkpointing.

    - pos_weight uses GLOBAL class ratio so score scale is consistent
      across cycles regardless of how many positives are in train_mask.
    - Mirrors ASReview exactly: all labeled nodes used for training.
    """
    graph      = graph.to(device)
    train_mask = train_mask.to(device)

    n_pos_global = graph.y.sum().item()
    n_neg_global = graph.num_nodes - n_pos_global
    pos_weight   = torch.tensor([n_neg_global / n_pos_global], device=device)

    model     = model_class(graph.x.shape[1], hidden_channels,
                            heads, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(graph.x, graph.edge_index)
        loss   = criterion(logits[train_mask], graph.y[train_mask].float())
        loss.backward()
        optimizer.step()

    return model


# Scoring helpers

def gat_scores(model, graph, device: torch.device) -> np.ndarray:
    """
    Full forward pass → sigmoid probabilities for all nodes.
    Returns numpy array [N]. Called only when model has been retrained.
    Works for GATClassifier or MLPClassifier — name kept for continuity
    with earlier scripts, but this is just "model_scores".
    """
    model.eval()
    graph = graph.to(device)
    with torch.no_grad():
        logits = model(graph.x, graph.edge_index)
        probs  = torch.sigmoid(logits)
    return probs.cpu().numpy()


def cosine_scores(graph, labeled_pos: set, device: torch.device) -> np.ndarray:
    """
    Score all nodes by max cosine similarity to any known positive.
    Returns numpy array [N]. Pure content signal, no graph structure.
    """
    if len(labeled_pos) == 0:
        return np.zeros(graph.num_nodes, dtype=np.float32)

    graph    = graph.to(device)
    pos_emb  = graph.x[list(labeled_pos)]       # [k, 768]
    all_emb  = graph.x                          # [N, 768]

    pos_norm = F.normalize(pos_emb, dim=1)      # [k, 768]
    all_norm = F.normalize(all_emb, dim=1)      # [N, 768]

    sim      = torch.mm(all_norm, pos_norm.T)   # [N, k]
    scores   = sim.max(dim=1).values            # [N]

    return scores.cpu().numpy()


#  Stagnation detector

def is_stagnating(recall_curve: list, window: int) -> bool:
    """
    Returns True when no new relevant paper has been found in the last
    `window` cycles — signal that the model-reachable cluster is exhausted.
    """
    if len(recall_curve) < window + 1:
        return False
    found_window_ago = recall_curve[-window][1]
    found_now        = recall_curve[-1][1]
    return found_now == found_window_ago


# Metrics

@dataclass
class SimulationResults:
    dataset_name: str
    n_total:      int
    n_relevant:   int
    prevalence:   float

    recall_curve: List[Tuple[int, int]] = field(default_factory=list)

    wss95:    Optional[float] = None
    wss100:   Optional[float] = None
    rr_10pct: Optional[float] = None
    atd:      Optional[float] = None

    wall_time_s: float = 0.0


def compute_metrics(r: SimulationResults) -> SimulationResults:
    """
    Thin wrapper: unpacks the recall curve and delegates to the shared
    implementation in metrics.py (also used by elas_simulation.py).
    """
    curve           = np.array(r.recall_curve)
    screened, found = curve[:, 0], curve[:, 1]

    m = compute_all_metrics(screened, found, r.n_total, r.n_relevant)
    r.wss95, r.wss100, r.rr_10pct, r.atd = (
        m["wss95"], m["wss100"], m["rr_10pct"], m["atd"]
    )
    return r


# Main simulation loop 

def run_simulation(
    graph,
    dataset_name:      str   = "dataset",
    model_class                = GATClassifier,
    n_prior_included:  int   = 1,
    n_prior_excluded:  int   = 1,
    n_instances:       int   = 1,
    train_epochs:      int   = 200,
    lr:                float = 1e-4,
    hidden_channels:   int   = 64,
    heads:             int   = 4,
    dropout:           float = 0.3,
    seed:              int   = 42,
    retrain_every:     int   = 1,
    weight_decay:      int   = None,
    stagnation_window: int   = 1000,
    verbose:           bool  = True,
    log_every:         int   = 50,
) -> SimulationResults:
    """
    Run one full ASReview-style simulation on a pre-built PyG graph.

    Graph must have:
        graph.x          [N, F]  sBERT node features
        graph.y          [N]     binary labels (1=relevant, 0=irrelevant)
        graph.edge_index [2, E]

    model_class : GATClassifier (default, main experiment) or MLPClassifier
                  (edge-free ablation).

    Phase 1 — model scoring:
        Forward pass cached; only recomputed every `retrain_every` labels.

    Phase 2 — Cosine fallback:
        Triggered when no relevant paper found in last `stagnation_window` cycles.
        Scores by max cosine similarity to known positives in sBERT space.
        Handles structurally isolated relevant papers the model cannot reach.
    """
    t_start = time.time()
    rng     = np.random.default_rng(seed)
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    labels  = graph.y.cpu().numpy()
    n       = graph.num_nodes
    n_rel   = int(labels.sum())
    idx_all = np.arange(n)
    pos_idx = idx_all[labels == 1]
    neg_idx = idx_all[labels == 0]

    if verbose:
        print(f"\n{'='*60}")
        print(f"Dataset : {dataset_name}")
        print(f"Model   : {model_class.__name__}")
        print(f"N papers: {n:,}  |  N relevant: {n_rel}  |  "
              f"Prevalence: {n_rel/n:.2%}")
        print(f"Prior   : {n_prior_included}+ / {n_prior_excluded}-  |  "
              f"Device: {device}  |  retrain_every: {retrain_every}")
        print(f"{'='*60}")

    # Prior knowledge
    prior_pos   = rng.choice(pos_idx, size=n_prior_included, replace=False)
    prior_neg   = rng.choice(neg_idx, size=n_prior_excluded, replace=False)
    labeled     = set(prior_pos.tolist() + prior_neg.tolist())
    labeled_pos = set(prior_pos.tolist())

    # Pool and incremental mask
    pool = set(idx_all.tolist()) - labeled

    # True = node is available for query. Updated O(1) per step.
    unlabeled_np = np.ones(n, dtype=bool)
    unlabeled_np[list(labeled)] = False

    def make_mask(idx_set: set) -> torch.Tensor:
        m = torch.zeros(n, dtype=torch.bool)
        if idx_set:
            m[list(idx_set)] = True
        return m

    # Simulation state
    results    = SimulationResults(dataset_name=dataset_name, n_total=n,
                                   n_relevant=n_rel, prevalence=n_rel / n)
    n_found    = int(sum(labels[i] for i in labeled))
    n_screened = len(labeled)
    results.recall_curve.append((n_screened, n_found))

    model         = None
    cached_scores = None   # numpy [N] — cached model scores
    model_updated = False
    in_fallback   = False  # latched True once stagnation detected; never resets
    cycle         = 0

    if verbose:
        print(f"Prior: screened={n_screened}, found={n_found}/{n_rel}")

    # Main loop
    while n_found < n_rel and pool:
        cycle += 1

        # Retrain (Phase 1 only)
        if not in_fallback and (cycle % retrain_every == 1 or model is None):
            train_mask = make_mask(labeled)
            model = train_one_cycle(
                graph, train_mask, device, model_class=model_class,
                epochs=train_epochs, lr=lr,
                hidden_channels=hidden_channels, heads=heads, dropout=dropout,
            )
            model_updated = True

        # Refresh score cache (Phase 1 only, when model changed)
        if not in_fallback and (model_updated or cached_scores is None):
            cached_scores = gat_scores(model, graph, device)
            model_updated = False

        # Check for stagnation → latch fallback
        if not in_fallback and is_stagnating(results.recall_curve,
                                             stagnation_window):
            in_fallback = True
            if verbose:
                print(f"  [cycle {cycle}] Stagnation detected — "
                      f"switching to cosine fallback "
                      f"({n_found}/{n_rel} found so far)")

        # Score all nodes
        if in_fallback:
            scores = cosine_scores(graph, labeled_pos, device)
        else:
            scores = cached_scores.copy()

        # Mask out labeled nodes
        scores[~unlabeled_np] = -1.0

        # Select top-n
        top_indices = np.argsort(scores)[::-1][:n_instances]

        # Oracle
        for idx in top_indices:
            idx = int(idx)
            if idx not in pool:
                continue

            true_label = int(labels[idx])
            n_screened += 1
            n_found    += true_label
            labeled.add(idx)
            unlabeled_np[idx] = False          # O(1) update
            if true_label == 1:
                labeled_pos.add(idx)
            pool.discard(idx)
            results.recall_curve.append((n_screened, n_found))

        # Log
        if verbose and cycle % log_every == 0:
            phase = "[fallback]" if in_fallback else "[model]   "
            print(f"  Cycle {cycle:>5} {phase} | "
                  f"Screened: {n_screened:>6} ({n_screened/n:.1%}) | "
                  f"Found: {n_found:>4}/{n_rel} ({n_found/n_rel:.1%})")

    # Final metrics
    results.wall_time_s = time.time() - t_start
    results = compute_metrics(results)

    # Stuck relevant papers diagnosis
    stuck_ids = [i for i in range(len(labels))
                 if labels[i] == 1 and i not in labeled_pos]
    if stuck_ids:
        print(f"\nStuck relevant papers: {len(stuck_ids)}")
        for i in stuck_ids:
            in_deg     = (graph.edge_index[1] == i).sum().item()
            out_deg    = (graph.edge_index[0] == i).sum().item()
            sbert_norm = graph.x[i].norm().item()
            print(f"  Node {i:5d}: in={in_deg:3d}, out={out_deg:3d}, "
                  f"total={in_deg+out_deg:3d}, sBERT_norm={sbert_norm:.3f}")

    if verbose:
        print(f"\n── Results: {dataset_name} ──────────────────────────────────")
        print(f"  Screened to find all relevant : {n_screened:,} / {n:,} "
              f"({n_screened/n:.1%})")
        print(f"  WSS@95  : {results.wss95:.3f}"
              if results.wss95  is not None else "  WSS@95  : N/A")
        print(f"  WSS@100 : {results.wss100:.3f}"
              if results.wss100 is not None else "  WSS@100 : N/A")
        print(f"  RR@10%  : {results.rr_10pct:.3f}")
        print(f"  ATD     : {results.atd:.4f}  (lower = finds relevant earlier)"
              if results.atd is not None else "  ATD     : N/A")
        print(f"  Wall time: {results.wall_time_s:.1f}s")

    return results


# Plotting

def plot_recall_curve(
    results_list,
    labels_list = None,
    title:      str = "Recall Curve",
    save_path:  str = None,
):
    """
    Standard ASReview recall curve.
    X: fraction of corpus screened  |  Y: recall
    Diagonal = random baseline.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available")
        return None, None

    if not isinstance(results_list, list):
        results_list = [results_list]
    if labels_list is None:
        labels_list = [r.dataset_name for r in results_list]

    fig, ax = plt.subplots(figsize=(8, 6))

    for r, lbl in zip(results_list, labels_list):
        curve = np.array(r.recall_curve)
        x = curve[:, 0] / r.n_total
        y = curve[:, 1] / r.n_relevant
        wss = f"WSS@95={r.wss95:.3f}" if r.wss95 is not None else "WSS@95=N/A"
        ax.plot(x, y, linewidth=2, label=f"{lbl}  ({wss})")

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1,
            label="Random baseline", alpha=0.5)
    ax.axhline(0.95, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax.text(0.01, 0.96, "95% recall", fontsize=9, color='gray')

    ax.set_xlabel("Fraction of corpus screened", fontsize=12)
    ax.set_ylabel("Recall (fraction of relevant found)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    else:
        plt.show()

    return fig, ax


# Multi-dataset runner

def run_all_datasets(
    graph_paths: dict,
    n_seeds:     int = 3,
    **sim_kwargs,
) -> dict:
    """
    Run simulation across multiple datasets and seeds.

    graph_paths : {dataset_name: path_to_sbert_graph_pt}
    sim_kwargs  : forwarded to run_simulation(), e.g. model_class=MLPClassifier
                  to run the ablation instead of the default GATClassifier.
    Returns     : {dataset_name: [SimulationResults, ...]}
    """
    from collections import defaultdict
    all_results = defaultdict(list)

    for name, path in graph_paths.items():
        print(f"\nLoading {name}")
        graph = torch.load(path, weights_only=False)

        for seed in range(n_seeds):
            r = run_simulation(graph, dataset_name=f"{name}_s{seed}",
                               seed=seed, **sim_kwargs)
            all_results[name].append(r)

        wss_vals  = [r.wss95  for r in all_results[name] if r.wss95  is not None]
        w100_vals = [r.wss100 for r in all_results[name] if r.wss100 is not None]
        rr_vals   = [r.rr_10pct for r in all_results[name] if r.rr_10pct is not None]
        atd_vals  = [r.atd    for r in all_results[name] if r.atd    is not None]

        if wss_vals:
            print(f"  {name} | WSS@95  : {np.mean(wss_vals):.3f} ± {np.std(wss_vals):.3f}")
        if w100_vals:
            print(f"  {name} | WSS@100 : {np.mean(w100_vals):.3f} ± {np.std(w100_vals):.3f}")
        if rr_vals:
            print(f"  {name} | RR@10%  : {np.mean(rr_vals):.3f} ± {np.std(rr_vals):.3f}")
        if atd_vals:
            print(f"  {name} | ATD     : {np.mean(atd_vals):.4f} ± {np.std(atd_vals):.4f}")

    return dict(all_results)