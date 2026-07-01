import os
import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

def run_embedding_pipeline(list_dfs):
    """
    Processes dataframes, generates embeddings using multi-GPU if available,
    and saves results directly to disk.
    """
    # Initialize the model
    model = SentenceTransformer("all-mpnet-base-v2")

    # Automatically detect all available Kaggle GPUs (T4 x2 or P100)
    if torch.cuda.is_available():
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        print(f"--- Using {len(devices)} GPUs for inference: {devices} ---")
    else:
        devices = None
        print("--- No GPU detected. Falling back to CPU ---")

    dataset_texts = {}
    embeddings = {}

    # Process each DataFrame
    for name, df in list_dfs:
        print(f"\nProcessing dataset: {name}...")
        
        # Combine title and abstract, handling missing values cleanly
        combined_text = df['title'].fillna('') + '. ' + df['abstract'].fillna('')
        dataset_texts[name] = combined_text.tolist()
        
        # Encode directly using integrated multi-process functionality
        embeddings[name] = model.encode(
            sentences=dataset_texts[name],  
            batch_size=128,                 # Optimal for Kaggle's 15GB-16GB GPU VRAM
            show_progress_bar=True,
            convert_to_numpy=True,  
            device=devices                  
        )
        
        # Save output to the current working directory (/kaggle/working/)
        output_filename = f"{name}_sbert_embeddings.npy"
        np.save(output_filename, embeddings[name])
        print(f"Saved: {output_filename} (Shape: {embeddings[name].shape})")

# Mandatory safeguard for PyTorch/Python multi-GPU processing
if __name__ == "__main__":   
    # Execute pipeline
    run_embedding_pipeline(list_dfs)
