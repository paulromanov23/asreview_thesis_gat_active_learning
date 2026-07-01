"""

Hyperparameter tuning for GATClassifier using Optuna.
Notebook-first design: run everything by calling functions directly.
No globals. CPU fallback for local runs.

Strategy:
  - Tune on Santos (~1 min/trial, high homophily h_adj=0.333)
  - Optimize WSS@95 (primary thesis metric)
  - On Kaggle double T4: spawn two worker processes via run_parallel()
  - On single GPU or CPU: call run_sequential() directly
  - After tuning: call validate_on_walker() to check generalization

Search space:
  lr              : [1e-5, 1e-2]        log-uniform
  hidden_channels : {32, 64, 128, 256}  categorical
  heads           : {2, 4, 8}           categorical
  n_layers        : {2, 3, 4}           categorical
  dropout         : [0.1, 0.5]          uniform
  train_epochs    : {50, 100, 200}      categorical
  weight_decay    : [1e-5, 1e-2]        log-uniform


Notebook usage (Kaggle double T4):

    from optuna_tune import run_parallel, validate_on_walker, print_study_summary

    SANTOS = "path/to/Santos_2018_sbert.pt"
    WALKER = "path/to/Walker_2018_sbert.pt"

    # Phase 1 — tune on Santos (spawns 2 workers, one per GPU)
    run_parallel(
        graph_path   = SANTOS,
        n_trials     = 50,          # per GPU → 100 total
        study_name   = "gat_santos",
        db_path      = "sqlite:///optuna_gat.db",
    )

    # Phase 2 — print summary
    print_study_summary(study_name="gat_santos", db_path="sqlite:///optuna_gat.db")

    # Phase 3 — validate top-3 configs on Walker
    validate_on_walker(
        walker_path  = WALKER,
        study_name   = "gat_santos",
        db_path      = "sqlite:///optuna_gat.db",
        top_n        = 3,
        n_seeds      = 3,
        results_path = "optuna_walker_results.json",
    )


Local / single-GPU usage:

    from optuna_tune import run_sequential, validate_on_walker

    run_sequential(
        graph_path = "path/to/Santos_2018_sbert.pt",
        n_trials   = 100,
        study_name = "gat_santos",
        db_path    = "sqlite:///optuna_gat.db",
        gpu_id     = 0,             # ignored if no CUDA available
    )

"""

import json
import multiprocessing as mp
import time

import numpy as np
import optuna
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

optuna.logging.set_verbosity(optuna.logging.WARNING)

#  Fixed simulation settings (match thesis config, not tuned) 
_SIM_CFG = dict(
    n_prior_included  = 1,
    n_prior_excluded  = 1,
    retrain_every     = 50,
    stagnation_window = 1000,
    seed              = 42,    # fixed across trials for fair comparison
)



# N-layer GATClassifier
# Replaces the hardcoded 2-layer version in simulate.py.
# Copy this class back into simulate.py once best config is found.


class GATClassifier(torch.nn.Module):
    """
    N-layer GAT for binary node classification.

    Architecture:
      Layer 1..n-1 : GATConv(in → hidden_channels, heads=heads) + ELU + dropout
      Layer n      : GATConv(hidden_channels * heads → 1, heads=1, concat=False)

    All tunable: hidden_channels, heads, n_layers, dropout.
    """
    def __init__(
        self,
        in_channels:     int,
        hidden_channels: int   = 64,
        heads:           int   = 4,
        n_layers:        int   = 2,
        dropout:         float = 0.3,
    ):
        super().__init__()
        assert n_layers >= 2, "n_layers must be >= 2"
        self.dropout = dropout
        self.convs   = torch.nn.ModuleList()

        self.convs.append(
            GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        )
        for _ in range(n_layers - 2):
            self.convs.append(
                GATConv(hidden_channels * heads, hidden_channels,
                        heads=heads, dropout=dropout)
            )
        self.convs.append(
            GATConv(hidden_channels * heads, 1,
                    heads=1, concat=False, dropout=dropout)
        )

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        for conv in self.convs[:-1]:
            x = F.elu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index).squeeze(-1)   # [N]



# Training helpers  (mirror simulate.py exactly)


def _train_one_cycle(graph, train_mask, device,
                     epochs, lr, hidden_channels, heads,
                     n_layers, dropout, weight_decay):
    graph      = graph.to(device)
    train_mask = train_mask.to(device)

    n_pos      = graph.y.sum().item()
    n_neg      = graph.num_nodes - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], device=device)

    model     = GATClassifier(graph.x.shape[1], hidden_channels,
                               heads, n_layers, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(graph.x, graph.edge_index)
        loss   = criterion(logits[train_mask], graph.y[train_mask].float())
        loss.backward()
        optimizer.step()

    return model


def _gat_scores(model, graph, device):
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(graph.x.to(device),
                                    graph.edge_index.to(device)))
    return probs.cpu().numpy()


def _cosine_scores(graph, labeled_pos, device):
    if not labeled_pos:
        return np.zeros(graph.num_nodes, dtype=np.float32)
    x        = graph.x.to(device)
    pos_norm = F.normalize(x[list(labeled_pos)], dim=1)
    all_norm = F.normalize(x, dim=1)
    return torch.mm(all_norm, pos_norm.T).max(dim=1).values.cpu().numpy()


def _is_stagnating(recall_curve, window):
    if len(recall_curve) < window + 1:
        return False
    return recall_curve[-window][1] == recall_curve[-1][1]



# Single-seed simulation (returns WSS@95 only — fast inner loop for Optuna)


def _run_trial_simulation(
    graph, device,
    seed, epochs, lr, hidden_channels, heads, n_layers, dropout, weight_decay,
    n_prior_included, n_prior_excluded, retrain_every, stagnation_window,
):
    """
    Stripped-down ASReview simulation returning WSS@95.
    No validation set. No logging. Mirrors simulate.py contract exactly.
    Returns None if 95% recall is never reached (penalized as -1.0 by caller).
    """
    rng     = np.random.default_rng(seed)
    labels  = graph.y.cpu().numpy()
    n       = graph.num_nodes
    n_rel   = int(labels.sum())
    idx_all = np.arange(n)

    prior_pos   = rng.choice(idx_all[labels == 1],
                              size=n_prior_included, replace=False)
    prior_neg   = rng.choice(idx_all[labels == 0],
                              size=n_prior_excluded, replace=False)
    labeled     = set(prior_pos.tolist() + prior_neg.tolist())
    labeled_pos = set(prior_pos.tolist())

    pool         = set(idx_all.tolist()) - labeled
    unlabeled_np = np.ones(n, dtype=bool)
    unlabeled_np[list(labeled)] = False

    def make_mask(idx_set):
        m = torch.zeros(n, dtype=torch.bool)
        if idx_set:
            m[list(idx_set)] = True
        return m

    n_found      = int(sum(labels[i] for i in labeled))
    n_screened   = len(labeled)
    recall_curve = [(n_screened, n_found)]

    model         = None
    cached_scores = None
    model_updated = False
    in_fallback   = False
    cycle         = 0

    while n_found < n_rel and pool:
        cycle += 1

        if not in_fallback and (cycle % retrain_every == 1 or model is None):
            model = _train_one_cycle(
                graph, make_mask(labeled), device,
                epochs=epochs, lr=lr,
                hidden_channels=hidden_channels, heads=heads,
                n_layers=n_layers, dropout=dropout,
                weight_decay=weight_decay,
            )
            model_updated = True

        if not in_fallback and (model_updated or cached_scores is None):
            cached_scores = _gat_scores(model, graph, device)
            model_updated = False

        if not in_fallback and _is_stagnating(recall_curve, stagnation_window):
            in_fallback = True

        scores = (_cosine_scores(graph, labeled_pos, device)
                  if in_fallback else cached_scores.copy())
        scores[~unlabeled_np] = -1.0

        # Top-1 query — with pool-membership safety check
        for candidate in np.argsort(scores)[::-1]:
            idx = int(candidate)
            if idx in pool:
                break

        true_label   = int(labels[idx])
        n_screened  += 1
        n_found     += true_label
        labeled.add(idx)
        unlabeled_np[idx] = False
        if true_label == 1:
            labeled_pos.add(idx)
        pool.discard(idx)
        recall_curve.append((n_screened, n_found))

    # WSS@95
    curve      = np.array(recall_curve)
    recall_pct = curve[:, 1] / n_rel
    idx95      = np.searchsorted(recall_pct, 0.95)
    if idx95 >= len(curve):
        return None
    return float(0.95 - curve[idx95, 0] / n)


# Optuna objective factory


def _make_objective(graph, device, sim_cfg):
    """
    Returns an Optuna objective closure.
    graph and device are captured once — no repeated I/O per trial.
    """
    def objective(trial):
        params = dict(
            lr              = trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            hidden_channels = trial.suggest_categorical("hidden_channels",
                                                         [32, 64, 128, 256]),
            heads           = trial.suggest_categorical("heads", [2, 4, 8]),
            n_layers        = trial.suggest_categorical("n_layers", [2, 3, 4]),
            dropout         = trial.suggest_float("dropout", 0.1, 0.5),
            epochs          = trial.suggest_categorical("train_epochs",
                                                         [50, 100, 200]),
            weight_decay    = trial.suggest_float("weight_decay",
                                                   1e-5, 1e-2, log=True),
        )

        t0    = time.time()
        wss95 = _run_trial_simulation(graph, device, **params, **sim_cfg)
        elapsed = time.time() - t0

        value = wss95 if wss95 is not None else -1.0
        print(
            f"  Trial {trial.number:>4} [{device}] WSS@95={value:.4f} | "
            f"lr={params['lr']:.1e} h={params['hidden_channels']} "
            f"heads={params['heads']} layers={params['n_layers']} "
            f"drop={params['dropout']:.2f} ep={params['epochs']} "
            f"wd={params['weight_decay']:.1e}  ({elapsed:.0f}s)"
        )
        return value

    return objective



# Study factory (shared between sequential and parallel runners)


def _create_or_load_study(study_name, db_path):
    return optuna.create_study(
        study_name   = study_name,
        storage      = db_path,
        direction    = "maximize",
        sampler      = optuna.samplers.TPESampler(
                           n_startup_trials = 20,   # random explore first
                           seed             = 42,
                       ),
        load_if_exists = True,
    )


def _resolve_device(gpu_id):
    """Return cuda:{gpu_id} if available, else cpu."""
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        gid    = gpu_id if gpu_id < n_gpus else 0
        return torch.device(f"cuda:{gid}")
    return torch.device("cpu")



# Public API — Sequential runner


def run_sequential(
    graph_path:  str,
    n_trials:    int  = 100,
    study_name:  str  = "gat_tuning",
    db_path:     str  = "sqlite:///optuna_gat.db",
    gpu_id:      int  = 0,
    sim_cfg:     dict = None,
):
    """
    Run Optuna on a single GPU (or CPU if unavailable).
    Use this locally or on Kaggle single-GPU.

    Args:
        graph_path : path to the tuning dataset .pt file (Santos recommended)
        n_trials   : number of Optuna trials to run
        study_name : Optuna study name (used for persistence)
        db_path    : SQLite path, e.g. "sqlite:///optuna_gat.db"
        gpu_id     : GPU index (0 or 1); ignored if CUDA unavailable
        sim_cfg    : override default simulation settings (optional)
    """
    cfg    = {**_SIM_CFG, **(sim_cfg or {})}
    device = _resolve_device(gpu_id)

    print(f"Device  : {device}")
    print(f"Loading : {graph_path}")
    graph  = torch.load(graph_path, weights_only=False)
    graph  = graph.to(device)

    study  = _create_or_load_study(study_name, db_path)
    print(f"Study   : '{study_name}'  |  existing trials: {len(study.trials)}")
    print(f"Running {n_trials} trials...\n")

    study.optimize(
        _make_objective(graph, device, cfg),
        n_trials        = n_trials,
        gc_after_trial  = True,
        show_progress_bar = False,
    )

    print(f"\nDone. Best WSS@95 = {study.best_value:.4f}")
    print_study_summary(study_name=study_name, db_path=db_path, top_n=5)



# Public API — Parallel runner (Kaggle double T4)


def _worker(gpu_id, graph_path, n_trials, study_name, db_path, sim_cfg):
    """
    Multiprocessing worker — each process owns one GPU.
    Must be a module-level function (not a lambda) for pickle to work.
    """
    cfg    = {**_SIM_CFG, **(sim_cfg or {})}
    device = _resolve_device(gpu_id)

    print(f"[gpu:{gpu_id}] Starting — device={device}")
    graph  = torch.load(graph_path, weights_only=False)
    graph  = graph.to(device)

    study  = _create_or_load_study(study_name, db_path)
    study.optimize(
        _make_objective(graph, device, cfg),
        n_trials        = n_trials,
        gc_after_trial  = True,
        show_progress_bar = False,
    )
    print(f"[gpu:{gpu_id}] Done.")


def run_parallel(
    graph_path:  str,
    n_trials:    int  = 50,
    study_name:  str  = "gat_tuning",
    db_path:     str  = "sqlite:///optuna_gat.db",
    sim_cfg:     dict = None,
):
    """
    Run Optuna across both T4 GPUs in parallel using multiprocessing.
    Each GPU runs n_trials → total = n_trials * 2.
    Falls back to sequential on cpu/single-GPU automatically.

    IMPORTANT — call this inside a notebook cell like:
        if __name__ == "__main__":   # not needed in notebooks but harmless
            run_parallel(...)

    Args:
        graph_path : path to the tuning dataset .pt file (Santos recommended)
        n_trials   : trials per GPU (50 → 100 total on double T4)
        study_name : Optuna study name
        db_path    : SQLite path
        sim_cfg    : override default simulation settings (optional)
    """
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if n_gpus < 2:
        print(f"Only {n_gpus} GPU(s) detected — falling back to sequential.")
        run_sequential(graph_path, n_trials=n_trials * max(1, n_gpus),
                       study_name=study_name, db_path=db_path,
                       gpu_id=0, sim_cfg=sim_cfg)
        return

    print(f"Launching 2 workers across {n_gpus} GPUs "
          f"({n_trials} trials each = {n_trials * 2} total).\n")

    # 'spawn' is required for CUDA + multiprocessing
    ctx = mp.get_context("spawn")

    workers = [
        ctx.Process(
            target = _worker,
            args   = (gpu_id, graph_path, n_trials,
                      study_name, db_path, sim_cfg),
        )
        for gpu_id in range(2)
    ]

    for w in workers:
        w.start()
    for w in workers:
        w.join()

    print("\nBoth workers finished.")
    print_study_summary(study_name=study_name, db_path=db_path, top_n=5)



# Public API — Walker validation


def validate_on_walker(
    walker_path:  str,
    study_name:   str  = "gat_tuning",
    db_path:      str  = "sqlite:///optuna_gat.db",
    top_n:        int  = 3,
    n_seeds:      int  = 3,
    gpu_id:       int  = 0,
    results_path: str  = "optuna_walker_results.json",
    sim_cfg:      dict = None,
):
    """
    Take top-N configs from the Santos study and validate each on Walker.
    Runs n_seeds per config and reports mean ± std WSS@95.

    Args:
        walker_path  : path to Walker_2018_sbert.pt
        study_name   : must match the study used during tuning
        db_path      : must match the db used during tuning
        top_n        : how many top Santos configs to transfer
        n_seeds      : seeds per Walker config (3 matches thesis standard)
        gpu_id       : GPU to use (0 by default)
        results_path : JSON file to save results
        sim_cfg      : override default simulation settings (optional)
    """
    cfg    = {**_SIM_CFG, **(sim_cfg or {})}
    device = _resolve_device(gpu_id)

    print(f"Loading Walker graph from {walker_path}...")
    graph  = torch.load(walker_path, weights_only=False).to(device)

    study  = optuna.load_study(study_name=study_name, storage=db_path)
    trials = (study.trials_dataframe()
              .query("state == 'COMPLETE'")
              .sort_values("value", ascending=False)
              .head(top_n))

    print(f"\n{'='*70}")
    print(f"Walker validation — top {top_n} Santos configs")
    print(f"{'='*70}")

    all_results = []

    for rank, (_, row) in enumerate(trials.iterrows(), 1):
        params = dict(
            lr              = float(row["params_lr"]),
            hidden_channels = int(row["params_hidden_channels"]),
            heads           = int(row["params_heads"]),
            n_layers        = int(row["params_n_layers"]),
            dropout         = float(row["params_dropout"]),
            epochs          = int(row["params_train_epochs"]),
            weight_decay    = float(row["params_weight_decay"]),
        )
        santos_wss = float(row["value"])
        print(f"\nRank {rank}  Santos WSS@95={santos_wss:.4f}")
        print(f"  Params: {params}")

        seed_results = []
        for s in range(n_seeds):
            # Override seed per run for multi-seed averaging
            trial_cfg = {**cfg, "seed": s}
            wss = _run_trial_simulation(graph, device, **params, **trial_cfg)
            val = wss if wss is not None else -1.0
            seed_results.append(val)
            print(f"  seed={s}  Walker WSS@95={val:.4f}")

        mean_wss = float(np.mean(seed_results))
        std_wss  = float(np.std(seed_results))
        print(f"  → Walker WSS@95: {mean_wss:.4f} ± {std_wss:.4f}")

        all_results.append(dict(
            rank             = rank,
            santos_wss95     = santos_wss,
            walker_wss95_mean= mean_wss,
            walker_wss95_std = std_wss,
            params           = params,
        ))

    # Save to JSON
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to {results_path}")

    # Summary table
    print(f"\n{'─'*50}")
    print(f"  {'Rank':<6} {'Santos WSS@95':>14} {'Walker WSS@95':>20}")
    print(f"{'─'*50}")
    for r in all_results:
        print(f"  {r['rank']:<6} {r['santos_wss95']:>14.4f} "
              f"  {r['walker_wss95_mean']:.4f} ± {r['walker_wss95_std']:.4f}")
    print(f"{'─'*50}")

    return all_results



# Public API — Study summary


def print_study_summary(
    study_name: str = "gat_tuning",
    db_path:    str = "sqlite:///optuna_gat.db",
    top_n:      int = 10,
):
    """Print top-N trials and best params from a completed study."""
    study  = optuna.load_study(study_name=study_name, storage=db_path)
    df     = (study.trials_dataframe()
              .query("state == 'COMPLETE'")
              .sort_values("value", ascending=False))

    cols = [
        "number", "value",
        "params_lr", "params_hidden_channels", "params_heads",
        "params_n_layers", "params_dropout",
        "params_train_epochs", "params_weight_decay",
    ]
    # Keep only columns that exist (safety for partially-run studies)
    cols = [c for c in cols if c in df.columns]

    print(f"\nTop {top_n} trials — {study_name}")
    print(df[cols].head(top_n).to_string(index=False))

    best = study.best_trial
    print(f"\nBest trial #{best.number}  WSS@95 = {best.value:.4f}")
    print("Params:")
    for k, v in best.params.items():
        print(f"  {k:<20}: {v}")