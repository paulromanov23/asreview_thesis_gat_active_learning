# asreview_thesis_gat_active_learning

Thesis: *Do your friends make you visible?* — Graph Attention Networks
incorporating citation graph structure for active learning in ASReview
systematic review screening, compared against the default ELAS u4 baseline
(TF-IDF + SVM).

MSc Applied Data Science, Utrecht University. Supervisors: Rens van de
Schoot, Timo van der Kuil.

## Data access

Datasets come from the SYNERGY+ collection (OpenAlex-sourced systematic
review corpora): Walker, Santos, Sep, Lewowski, Burska, Lauper, Nelson,
Dolinska, Brouwer, Muthu, Leenaars.

SYNERGY+ does not yet have its own published dataset paper — it is
currently a roadmap item on the ASReview GitHub. This repo cites the base
SYNERGY v1 dataset (DataverseNL DOI: `10.34894/HE6NAQ`) with transparent
disclosure that SYNERGY+ access was preliminary. If a published SYNERGY+
reference exists by the time you're reproducing this, cite that instead.

## Environment setup

```bash
pip install -r requirements.txt
```

Two packages (`optuna`, `sentence-transformers`) are listed unpinned —
pin them to your working versions once you've confirmed compatibility with
your `torch`/`torch-geometric` build, then update `requirements.txt`.

## Run order

1. **Clean + build graphs**
   ```bash
   python pipeline.py
   ```
   Runs `data_processing.clean_dataset()`, auto-selects the MeSH
   co-occurrence threshold per dataset (`find_mesh_threshold`, targets mean
   degree in [3, 15]), computes Table 1 statistics
   (`network_statistics.py`), and builds each dataset's PyG graph
   (`graph_builder.build_pyg_graph`).

   This step needs sBERT embeddings already computed (see step 0 below) —
   without `embeddings_dir` set, graphs are built with placeholder zero
   node features, which is only useful for checking edge structure, not
   for running simulations.

   **Step 0 — sBERT embeddings** (run before `pipeline.py`, typically on a
   GPU instance):
   ```bash
   python sbert_encode.py
   ```
   Encodes title+abstract per dataset with `all-mpnet-base-v2` (768-dim),
   saving `{dataset}_sbert_embeddings.npy` files that `pipeline.py` loads
   via its `embeddings_dir` argument.

2. **Hyperparameter tuning** (optional — final values already in
   `hyperparameters.tex`; re-run only to reproduce the tuning process
   itself)
   ```bash
   python optuna_tune.py
   ```
   Tunes on Santos (fast, high homophily), validates top configs on
   Walker. Final chosen config: `hidden_channels=32, heads=4, lr=0.0005,
   train_epochs=100, dropout=0.3` (2-layer GAT).

3. **GAT / MLP simulation**
   ```python
   from simulate import run_all_datasets, GATClassifier, MLPClassifier

   graph_paths = {"Walker": "outputs/Walker_graph.pt", ...}

   # Main experiment
   gat_results = run_all_datasets(graph_paths, n_seeds=3, model_class=GATClassifier)

   # Edge-free ablation (isolates sBERT-feature contribution from graph structure)
   mlp_results = run_all_datasets(graph_paths, n_seeds=3, model_class=MLPClassifier)
   ```
   WSS@95, WSS@100, RR@10%, and ATD are computed via `metrics.py`.

4. **ELAS u4 baseline**
   ```bash
   python elas_simulation.py
   ```
   Runs the official ELAS u4 config (`asreview.models.models.get_ai_config`)
   through ASReview's `Simulate`/`ActiveLearningCycle`, then scores the
   exported labeling order using the **same** `metrics.py` implementation
   as step 3 — this is what backs the head-to-head comparability claim
   between GAT and ELAS.

5. **Figures**
   ```bash
   python wss_95_chart.py
   ```
   Dumbbell plot of GAT vs ELAS WSS@95 per dataset, ordered by adjusted
   homophily (`h_adj`). An equivalent `atd_dumbbell`-style script can be
   adapted from this one for the ATD figure.

## Files

| File | Role |
|---|---|
| `data_processing.py` | Cleaning, MeSH threshold search (`find_mesh_threshold`) |
| `graph_builder.py` | PyG graph construction (citation + MeSH edges, sBERT node features) |
| `network_statistics.py` | Table 1 statistics (degree, homophily, connectivity, MeSH coverage) |
| `pipeline.py` | Orchestrates cleaning → statistics → graph building |
| `sbert_encode.py` | sBERT (all-mpnet-base-v2) title+abstract embeddings |
| `simulate.py` | GAT/MLP models, ASReview-aligned simulation loop |
| `elas_simulation.py` | ELAS u4 baseline runner |
| `optuna_tune.py` | Hyperparameter search (Santos → Walker validation) |
| `metrics.py` | Shared WSS@95/WSS@100/RR@10%/ATD implementation used by both `simulate.py` and `elas_simulation.py` |
| `wss_95_chart.py` | GAT vs ELAS WSS@95 dumbbell figure |
