import pandas as pd
import numpy as np
import ast
from collections import Counter, defaultdict

def find_mesh_threshold(df, mesh_col, target_degree=5, min_threshold=2, max_threshold=15):
    """
    Automatically find the MeSH co-occurrence threshold that produces
    a mean node degree closest to target_degree.
    """
    from itertools import combinations
    
    term_to_papers = defaultdict(list)
    for idx, terms in enumerate(df[mesh_col]):
        for term in terms:
            term_to_papers[term].append(idx)
    
    pair_counts = defaultdict(int)
    for term, papers in term_to_papers.items():
        if len(papers) < 500:
            for i, j in combinations(papers, 2):
                pair_counts[(min(i,j), max(i,j))] += 1
    
    best_threshold = min_threshold
    best_diff = float('inf')
    
    for threshold in range(min_threshold, 16):
        edges = [(i,j) for (i,j), cnt in pair_counts.items() if cnt >= threshold]
        mean_degree = 2 * len(edges) / len(df)
        diff = abs(mean_degree - target_degree)
        print(f"threshold={threshold}: mean_degree={mean_degree:.2f}")
        
        if diff < best_diff:
            best_diff = diff
            best_threshold = threshold
            
        # Early stop if we've gone below target
        if mean_degree < target_degree * 0.5:
            break
    
    return best_threshold

def find_top_n_filter(df):
    """
    Automatically determine how many top MeSH terms to filter out
    by finding the elbow in the term frequency distribution.
    """
    all_terms = [t for terms in df['mesh_terms'] for t in terms]
    counts = Counter(all_terms)
    
    # Terms appearing in more than 20% of papers are too generic
    n_papers = len(df)
    threshold_count = n_papers * 0.20
    
    n_to_filter = sum(1 for _, cnt in counts.items() if cnt > threshold_count)
    return max(n_to_filter, 10)  # always filter at least 10

GENERIC_MESH = {
    'Humans', 'Male', 'Female', 'Adult', 'Middle Aged',
    'Animals', 'Aged', 'Young Adult', 'Adolescent', 'Child',
    'Aged, 80 and over', 'Child, Preschool', 'Infant',
    'Mice', 'Rats', 'Prospective Studies', 'Retrospective Studies',
    'Treatment Outcome', 'Follow-Up Studies', 'Time Factors',
    'Risk Factors', 'Reproducibility of Results', 'Sensitivity and Specificity'
}

def clean_dataset(df):
    
    df = df.copy()

    # OpenAlex IDs
    df['openalex_id'] = df['openalex_id'].str.upper().str.replace(
        'HTTPS://OPENALEX.ORG/', '', regex=False
    ).str.replace('https://openalex.org/', '', regex=False)
    
    # Parse list-of-ID columns
    for col in ['referenced_works', 'related_works']:
        df[col] = df[col].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else (x if isinstance(x, list) else [])
        )
        df[col] = df[col].apply(
            lambda refs: [r.replace('https://openalex.org/', '').upper() for r in refs]
        )
    
    # Parse MeSH — list of dicts, extract descriptor names
    df['mesh'] = df['mesh'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else (x if isinstance(x, list) else [])
    )
    
    df['mesh_terms'] = df['mesh'].apply(
    lambda terms: list(set(
        t['descriptor_name'] for t in terms 
        if isinstance(t, dict) and 'descriptor_name' in t and t['descriptor_name'] is not None
    ))
)
     # Generic MeSH filter — must happen before top_n filter
    df['mesh_terms'] = df['mesh_terms'].apply(
        lambda terms: [t for t in terms if t not in GENERIC_MESH]
    )
    
    # Parse keywords — simple list of strings
    df['keywords'] = df['keywords'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else (x if isinstance(x, list) else [])
    )
    
    # In-corpus edges
    all_ids = set(df['openalex_id'])
    df['in_corpus_refs'] = df['referenced_works'].apply(
        lambda refs: [r for r in refs if r in all_ids]
    )
    df['in_corpus_related'] = df['related_works'].apply(
        lambda refs: [r for r in refs if r in all_ids]
    )

    # Automatic top_n based on frequency distribution
    top_n = find_top_n_filter(df)
    all_terms = [t for terms in df['mesh_terms'] for t in terms]
    top_n_terms = {term for term, _ in Counter(all_terms).most_common(top_n)}
    df['mesh_terms_specific'] = df['mesh_terms'].apply(
        lambda terms: [t for t in terms if t not in top_n_terms]
    )
    df['_mesh_top_n_filtered'] = top_n  # store for reproducibility logging
    
    return df