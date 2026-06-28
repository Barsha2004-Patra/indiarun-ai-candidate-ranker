"""
Main Entry Point for the Online Ranking System.
Strict constraint: Must run in under 5 minutes on CPU, NO network access.
"""
import os
# CRITICAL: Disable HuggingFace Network calls to comply with offline constraint
os.environ["HF_HUB_OFFLINE"] = "1"

import time
import logging
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import load_config
from src.scoring.semantic import calculate_semantic_score
from src.scoring.behavioral import calculate_behavioral_score
from src.scoring.risk import apply_risk_penalties
from src.reasoning.generator import generate_reasoning

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RedrobRanker:
    """Executes the ultra-fast vectorized ranking pipeline."""
    
    def __init__(self, config_path: Path):
        self.config = load_config(config_path)
        logging.info("Loading local embedding model (Cached, Offline)...")
        self.model = SentenceTransformer(self.config.model_name)

    def run(self, features_path: Path, output_path: Path, jd_text: str) -> None:
        start_time = time.time()
        
        logging.info(f"Loading pre-computed candidate features from {features_path}...")
        df = pd.read_parquet(features_path)
        
        logging.info("Embedding Job Description...")
        jd_vector = self.model.encode(jd_text, normalize_embeddings=True)
        
        logging.info("Calculating Semantic Matches (Vectorized)...")
        df['semantic_score'] = calculate_semantic_score(df, jd_vector)
        
        logging.info("Calculating Behavioral Signals...")
        df['behavioral_score'] = calculate_behavioral_score(df, self.config.behavioral, self.config.thresholds)
        
        # Experience Normalization
        max_exp = self.config.thresholds.max_ideal_experience_years
        df['exp_score'] = (df['years_of_experience'] / max_exp).clip(0, 1)
        
        logging.info("Fusing final scores and applying risk penalties...")
        df['base_score'] = (
            (df['semantic_score'] * self.config.weights.semantic) +
            (df['behavioral_score'] * self.config.weights.behavioral) +
            (df['exp_score'] * self.config.weights.experience)
        )
        
        df['final_score'] = apply_risk_penalties(df, df['base_score'], self.config.thresholds)
        
        logging.info("Extracting and sorting Top 100 candidates...")
        top_100 = df.sort_values(by='final_score', ascending=False).head(100).copy()
        top_100['rank'] = range(1, 101)
        
        logging.info("Generating explainable reasoning...")
        top_100['reasoning'] = top_100.apply(lambda row: generate_reasoning(row, self.config.thresholds), axis=1)
        
        logging.info("Exporting to CSV...")
        submission = top_100[['candidate_id', 'rank', 'final_score', 'reasoning']]
        submission = submission.rename(columns={'final_score': 'score'}) 
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(output_path, index=False)
        
        logging.info(f"Success! Saved to {output_path}")
        logging.info(f"Total Online Ranking Time: {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Redrob Candidate Ranker.")
    parser.add_argument('--features', type=str, required=True, help="Path to features.parquet")
    parser.add_argument('--jd', type=str, required=True, help="Path to JD text file")
    parser.add_argument('--output', type=str, required=True, help="Path to output CSV")
    parser.add_argument('--config', type=str, default="configs/settings.yaml", help="Path to config yaml")
    args = parser.parse_args()
    
    with open(args.jd, 'r', encoding='utf-8') as f:
        job_desc_text = f.read()
        
    ranker = RedrobRanker(Path(args.config))
    ranker.run(Path(args.features), Path(args.output), job_desc_text)