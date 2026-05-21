import random
import numpy as np
import networkx as nx
from collections import Counter

from graph_builder import build_nx_graph
# network_statistics.py — top of file
from graph_builder import build_mesh_edges_fast 


# ── Node-level statistics ─────────────────────────────────────────────────────

def node_statistics(df):
    """Statistics about individual node properties."""
    citation_counts = df['in_corpus_refs'].apply(len)
    related_counts  = df['in_corpus_related'].apply(len)
    mesh_counts     = df['mesh_terms'].apply(len)
    kw_counts       = df['keywords'].apply(len)

    return {
        'n_papers':           len(df),
        'n_relevant':         int(df['label_included'].sum()),
        'prevalence':         df['label_included'].mean(),

        # Citation
        'citation_isolated':  (citation_counts == 0).mean(),
        'citation_mean':      citation_counts.mean(),
        'citation_median':    citation_counts.median(),
        'citation_90pct':     citation_counts.quantile(0.9),
        'citation_max':       citation_counts.max(),

        # Related works
        'related_mean':       related_counts.mean(),
        'related_isolated':   (related_counts == 0).mean(),

        # MeSH
        'mesh_coverage':      (mesh_counts > 0).mean(),
        'mesh_mean_terms':    mesh_counts.mean(),

        # Keywords
        'keyword_coverage':   (kw_counts > 0).mean(),
        'keyword_mean_terms': kw_counts.mean(),

        # Metadata completeness
        'has_abstract':       df['abstract'].notna().mean(),
        'has_fwci':           df['fwci'].notna().mean(),
        'year_min':           df['publication_year'].min(),
        'year_max':           df['publication_year'].max(),
    }


# ── Graph-level statistics ────────────────────────────────────────────────────

def graph_statistics(df, mesh_threshold, mesh_col='mesh_terms_specific'):
    """
    Statistics about the full graph structure.

    Diameter and avg path length are computed on the largest connected
    component only (undefined on disconnected graphs).
    For large components they are approximated via BFS from 50 sampled nodes.
    Graphs with largest CC > 5000 nodes skip these metrics entirely (-1).

    Key metric for GNN viability: homophily_vs_random.
    Values > 1.0 mean relevant papers connect to each other more than
    random chance predicts — the property GNN message passing exploits.
    """
    G = build_nx_graph(df, mesh_threshold, mesh_col)

    # ── Edge type counts ──────────────────────────────────────────────────────
    citation_edges = [(u, v) for u, v, d in G.edges(data=True)
                      if d.get('edge_type') == 'citation']
    mesh_edges     = [(u, v) for u, v, d in G.edges(data=True)
                      if d.get('edge_type') == 'mesh']

    # ── Degree distribution ───────────────────────────────────────────────────
    degrees = [d for _, d in G.degree()]

    # ── Connected components ──────────────────────────────────────────────────
    components = list(nx.connected_components(G))
    largest_cc  = G.subgraph(max(components, key=len))
    n_largest   = largest_cc.number_of_nodes()

    # ── Diameter / avg path length ────────────────────────────────────────────
    if n_largest <= 500:
        diameter        = nx.diameter(largest_cc)
        avg_path_length = nx.average_shortest_path_length(largest_cc)
    elif n_largest <= 5000:
        sample  = random.sample(list(largest_cc.nodes()), 50)
        lengths = []
        for src in sample:
            paths = nx.single_source_shortest_path_length(largest_cc, src)
            lengths.extend(paths.values())
        diameter        = int(max(lengths))
        avg_path_length = float(np.mean(lengths))
    else:
        diameter        = -1
        avg_path_length = -1.0

    # ── Clustering coefficient (sampled for large graphs) ─────────────────────
    sample_nodes = random.sample(list(G.nodes()), min(500, G.number_of_nodes()))
    clustering   = nx.average_clustering(G, nodes=sample_nodes)

    # ── Degree assortativity ──────────────────────────────────────────────────
    try:
        assortativity = nx.degree_assortativity_coefficient(G)
    except Exception:
        assortativity = None

    # ── Label homophily ───────────────────────────────────────────────────────
    # Do relevant papers connect to each other more than random chance predicts?
    labels = dict(enumerate(df['label_included'].values))
    same_label_edges = sum(
        1 for u, v in G.edges()
        if labels.get(u) == labels.get(v) == 1
    )
    total_relevant   = df['label_included'].sum()
    expected_homophily = (total_relevant / len(df)) ** 2  # random baseline

    n_total_edges = max(G.number_of_edges(), 1)
    n_mesh_cite   = max(len(mesh_edges) + len(citation_edges), 1)

    return {
        # Size
        'n_nodes':                 G.number_of_nodes(),
        'n_edges_total':           G.number_of_edges(),
        'n_edges_citation':        len(citation_edges),
        'n_edges_mesh':            len(mesh_edges),

        # Degree
        'degree_mean':             np.mean(degrees),
        'degree_median':           np.median(degrees),
        'degree_90pct':            np.percentile(degrees, 90),
        'degree_max':              max(degrees),
        'pct_isolated':            sum(1 for d in degrees if d == 0) / len(degrees),

        # Connectivity
        'n_components':            len(components),
        'largest_cc_pct':          n_largest / len(df),
        'diameter_approx':         diameter,
        'avg_path_length_approx':  avg_path_length,

        # Structure
        'clustering_coefficient':  clustering,
        'degree_assortativity':    assortativity,

        # Label homophily — key GNN viability metric
        'relevant_edge_density':   same_label_edges / n_mesh_cite,
        'homophily_vs_random':     (same_label_edges / n_total_edges)
                                   / max(expected_homophily, 1e-9),
    }


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_statistics(name, df, mesh_threshold, mesh_col='mesh_terms_specific'):
    """Compute and pretty-print all statistics for one dataset."""
    print(f"\n{'='*60}")
    print(f"Dataset: {name}")
    print(f"{'='*60}")

    node_stats  = node_statistics(df)
    graph_stats = graph_statistics(df, mesh_threshold, mesh_col)

    print(f"\n--- Node Statistics ---")
    print(f"  Papers:          {node_stats['n_papers']:,}")
    print(f"  Relevant:        {node_stats['n_relevant']:,} "
          f"({node_stats['prevalence']:.1%})")
    print(f"  Abstract cover:  {node_stats['has_abstract']:.1%}")
    print(f"  FWCI available:  {node_stats['has_fwci']:.1%}")
    print(f"  Year range:      {node_stats['year_min']} – {node_stats['year_max']}")
    print(f"  MeSH coverage:   {node_stats['mesh_coverage']:.1%} "
          f"(mean {node_stats['mesh_mean_terms']:.1f} terms)")
    print(f"  Keyword cover:   {node_stats['keyword_coverage']:.1%}")
    print(f"  Citation edges:  {node_stats['citation_mean']:.2f} mean, "
          f"{node_stats['citation_isolated']:.1%} isolated")

    print(f"\n--- Graph Statistics ---")
    print(f"  Edges (total):   {graph_stats['n_edges_total']:,}")
    print(f"  Edges citation:  {graph_stats['n_edges_citation']:,}")
    print(f"  Edges MeSH:      {graph_stats['n_edges_mesh']:,}")
    print(f"  Mean degree:     {graph_stats['degree_mean']:.2f}")
    print(f"  Median degree:   {graph_stats['degree_median']:.0f}")
    print(f"  90th pct degree: {graph_stats['degree_90pct']:.0f}")
    print(f"  Isolated nodes:  {graph_stats['pct_isolated']:.1%}")
    print(f"  Components:      {graph_stats['n_components']:,}")
    print(f"  Largest CC:      {graph_stats['largest_cc_pct']:.1%} of nodes")
    print(f"  Diameter(approx):{graph_stats['diameter_approx']}")
    print(f"  Avg path length: {graph_stats['avg_path_length_approx']:.2f}")
    print(f"  Clustering coef: {graph_stats['clustering_coefficient']:.4f}")
    print(f"  Assortativity:   {graph_stats['degree_assortativity']}")

    print(f"\n--- Label Homophily ---")
    print(f"  Relevant-relevant edge density: "
          f"{graph_stats['relevant_edge_density']:.4f}")
    print(f"  Homophily vs random baseline:   "
          f"{graph_stats['homophily_vs_random']:.2f}x")
    print(f"  (>1.0 = relevant papers cluster more than chance)")

    return {**node_stats, **graph_stats}