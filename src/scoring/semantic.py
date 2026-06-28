"""
Vectorized semantic scoring module.
"""
import numpy as np
import pandas as pd

def calculate_semantic_score(df: pd.DataFrame, jd_vector: np.ndarray) -> np.ndarray:
    """
    Executes an ultra-fast vectorized dot product for cosine similarity.
    Requires vectors to be pre-normalized.
    """
    candidate_vectors = np.vstack(df['profile_embedding'].values)
    return np.dot(candidate_vectors, jd_vector)