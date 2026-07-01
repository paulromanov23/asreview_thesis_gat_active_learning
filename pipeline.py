import os
import time
import pandas as pd
import numpy as np
import torch
import logging
from pathlib import Path
from torch_geometric.data import Data

from data_processing import clean_dataset, find_mesh_threshold
from network_statistics import node_statistics, graph_statistics
from graph_builder import build_pyg_graph 

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s — %(levelname)s — %(message)s')
log = logging.getLogger(__name__)


def load_datasets(data_dir):
    """Load all CSV files from data directory."""
    data_dir = Path(data_dir)
    datasets = {}
    for csv_file in sorted(data_dir.glob('*.csv')):
        name = csv_file.stem  # filename without extension
        log.info(f"Loading {name}...")
        try:
            df = pd.read_csv(csv_file, low_memory=False)
            if 'label_included' not in df.columns:
                log.warning(f"  Skipping {name} — no label_included column")
                continue
            datasets[name] = df
            log.info(f"  {len(df):,} papers, "
                     f"{df['label_included'].sum()} relevant "
                     f"({df['label_included'].mean():.1%})")
        except Exception as e:
            log.error(f"  Failed to load {name}: {e}")
    return datasets


def run_pipeline(data_dir, target_degree=5, save_dir=None):
    """
    Full pipeline:
      1. Load all datasets from data_dir
      2. Clean each dataset
      3. Collect statistics
      4. Build PyG graph objects
      5. Optionally save outputs

    Returns:
        stats_df  — DataFrame with one row per dataset
        graphs    — dict of {dataset_name: torch_geometric.data.Data}
    """
    t_start = time.time()
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    log.info("="*60)
    log.info("STEP 1: Loading datasets")
    datasets = load_datasets(data_dir)
    if not datasets:
        raise ValueError(f"No valid datasets found in {data_dir}")

    # ── 2. Clean ──────────────────────────────────────────────────────────────
    log.info("="*60)
    log.info("STEP 2: Cleaning datasets")
    cleaned = {}
    for name, df in datasets.items():
        t0 = time.time()
        log.info(f"  Cleaning {name}...")
        cleaned[name] = clean_dataset(df)
        log.info(f"  Done in {time.time()-t0:.1f}s")

    # ── 3. Statistics ─────────────────────────────────────────────────────────
    log.info("="*60)
    log.info("STEP 3: Computing statistics")
    all_stats = {}
    for name, df in cleaned.items():
        t0 = time.time()
        log.info(f"  {name}...")
        
        # Auto-select threshold
        threshold = find_mesh_threshold(df, 'mesh_terms_specific', 
                                        target_degree=target_degree)
        log.info(f"    Auto-selected MeSH threshold: {threshold}")
        
        node_stats  = node_statistics(df)
        graph_stats = graph_statistics(df, threshold)
        all_stats[name] = {
            'mesh_threshold': threshold,
            **node_stats,
            **graph_stats
        }
        log.info(f"    Done in {time.time()-t0:.1f}s")

    stats_df = pd.DataFrame(all_stats).T
    stats_df.index.name = 'dataset'

    if save_dir:
        stats_df.to_csv(save_dir / 'statistics.csv')
        log.info(f"  Statistics saved to {save_dir / 'statistics.csv'}")

    # ── 4. Build graphs ───────────────────────────────────────────────────────
    log.info("="*60)
    log.info("STEP 4: Building PyG graph objects")
    graphs = {}
    for name, df in cleaned.items():
        t0 = time.time()
        threshold = all_stats[name]['mesh_threshold']
        log.info(f"  Building graph for {name} (MeSH threshold={threshold})...")
        
        graphs[name] = build_pyg_graph(df, mesh_threshold=threshold)
        
        g = graphs[name]
        log.info(f"    Nodes: {g.num_nodes:,}, Edges: {g.num_edges:,} "
                 f"— {time.time()-t0:.1f}s")
        
        if save_dir:
            torch.save(g, save_dir / f'{name}_graph.pt')
            log.info(f"    Saved to {save_dir / f'{name}_graph.pt'}")

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("="*60)
    log.info(f"Pipeline complete in {time.time()-t_start:.1f}s")
    log.info(f"\n{stats_df[['n_papers','prevalence','degree_mean','homophily_adj','homophily_target_class']].to_string()}")

    return stats_df, graphs


if __name__ == '__main__':
    stats, graphs = run_pipeline(
        data_dir='data/',
        target_degree=5,
        save_dir='outputs_directed/'
    )