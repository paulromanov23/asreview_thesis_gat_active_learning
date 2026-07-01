import torch
import numpy as np
from torch_geometric.data import Data
from scipy.sparse import csr_matrix
from collections import defaultdict
import networkx as nx


def build_mesh_edges_fast(df, mesh_col, threshold, max_term_frequency=500):
    n = len(df)
    all_terms = list({t for terms in df[mesh_col] for t in terms})
    term_to_idx = {t: i for i, t in enumerate(all_terms)}

    # Count term frequencies
    term_freq = defaultdict(int)
    for terms in df[mesh_col]:
        for term in terms:
            term_freq[term] += 1

    rows, cols = [], []
    for doc_idx, terms in enumerate(df[mesh_col]):
        for term in terms:
            if term_freq[term] < max_term_frequency:  # same filter as threshold search
                rows.append(doc_idx)
                cols.append(term_to_idx[term])

    if not rows:
        return []

    data = np.ones(len(rows))
    td_matrix = csr_matrix((data, (rows, cols)), shape=(n, len(all_terms)))
    td_binary = td_matrix.copy()
    td_binary.data = np.ones_like(td_binary.data)
    cooccurrence = (td_binary @ td_binary.T)
    cooccurrence.setdiag(0)
    cooccurrence.data[cooccurrence.data < threshold] = 0
    cooccurrence.eliminate_zeros()

    edges = [(i, j) for i, j in zip(*cooccurrence.nonzero()) if i < j]
    return edges


def build_nx_graph(df, mesh_threshold, mesh_col='mesh_terms_specific'):
    """NetworkX graph — used for statistics only."""
    df = df.reset_index(drop=True)
    id_to_idx = {oid: idx for idx, oid in enumerate(df['openalex_id'])}
    G = nx.DiGraph()
    G.add_nodes_from(range(len(df)))

    # Citation edges — vectorized using apply instead of iterrows
    citation_edges = []
    for idx, (oid, refs) in enumerate(zip(df['openalex_id'], df['in_corpus_refs'])):
        src = id_to_idx[oid]
        for ref in refs:
            if ref in id_to_idx:
                citation_edges.append((src, id_to_idx[ref]))
    G.add_edges_from(citation_edges, edge_type='citation')

    # MeSH edges — fast sparse matrix method
    mesh_pairs = build_mesh_edges_fast(df, mesh_col, mesh_threshold)
    G.add_edges_from(mesh_pairs, edge_type='mesh')

    return G


def build_pyg_graph(df, mesh_threshold, mesh_col='mesh_terms_specific'):
    """
    Build a PyTorch Geometric Data object from a cleaned dataframe.
    Node features are placeholder zeros until SPECTER2 embeddings are added.
    Edge types: 0 = citation, 1 = MeSH
    """
    df = df.reset_index(drop=True)
    id_to_idx = {oid: idx for idx, oid in enumerate(df['openalex_id'])}
    n = len(df)

    # ── Citation edges ────────────────────────────────────────────────────────
    citation_src, citation_dst = [], []
    for idx, row in df.iterrows():
        src = id_to_idx[row['openalex_id']]
        for ref in row['in_corpus_refs']:
            if ref in id_to_idx:
                citation_src.append(src)
                citation_dst.append(id_to_idx[ref])

    # ── MeSH edges ────────────────────────────────────────────────────────────
    mesh_pairs = build_mesh_edges_fast(df, mesh_col, mesh_threshold)
    mesh_src = [i for i, j in mesh_pairs]
    mesh_dst = [j for i, j in mesh_pairs]

    # ── Combine — undirected (add both directions) ────────────────────────────
    #all_src = citation_src + citation_dst + mesh_src + mesh_dst
    #all_dst = citation_dst + citation_src + mesh_dst + mesh_src
    #edge_type = (
    #[0] * len(citation_src) +    # outgoing citations (A cites B)
    #[2] * len(citation_src) +    # incoming citations (B cited by A)
    #[1] * len(mesh_src) * 2      # MeSH co-occurrence, stays undirected
    #)

    #  ──────────────────────────── Directed citations — only A→B  ────────────────────────────
    all_src = citation_src + mesh_src + mesh_dst
    all_dst = citation_dst + mesh_dst + mesh_src

    edge_type = (
        [0] * len(citation_src) +   # outgoing citations (A cites B) only
        [1] * len(mesh_src) * 2     # MeSH stays undirected
    )

    edge_index = torch.tensor([all_src, all_dst], dtype=torch.long)
    edge_attr  = torch.tensor(edge_type, dtype=torch.long)

    # ── Node features — zeros for now, replaced by SPECTER2 later ────────────
    x = torch.zeros((n, 1), dtype=torch.float)

    # ── Labels ────────────────────────────────────────────────────────────────
    y = torch.tensor(df['label_included'].values, dtype=torch.long)

    # ── Metadata — useful for active learning loop ────────────────────────────
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.num_nodes = n
    #print(data.edge_index.shape[1])
    data.openalex_ids = df['openalex_id'].tolist()
    data.titles = df['title'].tolist()

    return data