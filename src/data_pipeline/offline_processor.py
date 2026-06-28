"""
Offline Data Pipeline: Parses nested JSONL, extracts heuristic flags,
and generates dense semantic embeddings. Outputs optimized Parquet artifacts.
"""
import json
import gzip
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CandidatePipeline:
    """Handles the transformation of raw JSONL candidate data into vectorized Parquet format."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        logging.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def extract_features(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """Flattens schema and engineers derived features & risk flags."""
        profile = candidate.get('profile', {})
        signals = candidate.get('redrob_signals', {})
        career = candidate.get('career_history', [])
        skills = candidate.get('skills', [])

        years_exp = profile.get('years_of_experience', 0.0)
        response_rate = signals.get('recruiter_response_rate', 0.0)
        profile_completeness = signals.get('profile_completeness_score', 0)
        notice_period = signals.get('notice_period_days', 90)
        
        total_months_worked = sum([job.get('duration_months', 0) for job in career])
        avg_tenure = total_months_worked / len(career) if career else 0
        expert_skills = sum(1 for s in skills if s.get('proficiency') == 'expert')
        
        # Risk: Honeypot trap (high expertise, zero experience)
        is_risk_honeypot = bool(expert_skills >= 5 and years_exp < 1)
        
        # Text aggregation for Semantic Model
        career_text = " ".join([
            f"{job.get('title', '')} at {job.get('company', '')}: {job.get('description', '')}" 
            for job in career[:3]
        ])
        
        semantic_text = (
            f"Title: {profile.get('current_title', '')}. "
            f"Summary: {profile.get('summary', '')}. "
            f"Experience: {career_text}"
        )

        return {
            'candidate_id': candidate['candidate_id'],
            'current_title': profile.get('current_title', ''),
            'years_of_experience': years_exp,
            'recruiter_response_rate': response_rate,
            'profile_completeness': profile_completeness,
            'notice_period_days': notice_period,
            'avg_tenure_months': avg_tenure,
            'expert_skill_count': expert_skills,
            'is_risk_honeypot': is_risk_honeypot,
            'raw_text_for_embedding': semantic_text
        }

    def process_file(self, input_filepath: Path, output_filepath: Path, batch_size: int = 2048) -> None:
        """Processes JSONL, vectorizes text, and saves to Parquet."""
        logging.info(f"Starting processing of {input_filepath}")
        
        open_func = gzip.open if input_filepath.suffix == '.gz' else open
        mode = 'rt' if input_filepath.suffix == '.gz' else 'r'
        records: List[Dict[str, Any]] = []
        
        logging.info("Pass 1: Extracting schema and engineering features...")
        with open_func(input_filepath, mode) as f:
            for line in tqdm(f, desc="Parsing JSONL"):
                if not line.strip():
                    continue
                features = self.extract_features(json.loads(line))
                records.append(features)

        df = pd.DataFrame(records)
        
        logging.info("Pass 2: Generating dense semantic embeddings (batched)...")
        texts_to_embed = df['raw_text_for_embedding'].tolist()
        
        embeddings = self.model.encode(
            texts_to_embed, 
            batch_size=batch_size, 
            show_progress_bar=True,
            normalize_embeddings=True
        )
        
        df['profile_embedding'] = list(embeddings)
        df = df.drop(columns=['raw_text_for_embedding'])
        
        output_filepath.parent.mkdir(parents=True, exist_ok=True)
        logging.info(f"Pass 3: Saving highly optimized dataset to {output_filepath}")
        df.to_parquet(output_filepath, engine='pyarrow', compression='snappy')
        logging.info("Offline processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process offline candidate data.")
    parser.add_argument('--input', type=str, required=True, help="Path to candidates.jsonl")
    parser.add_argument('--output', type=str, required=True, help="Path to save features.parquet")
    parser.add_argument('--batch', type=int, default=2048, help="Batch size for embedding")
    args = parser.parse_args()
    
    pipeline = CandidatePipeline()
    pipeline.process_file(Path(args.input), Path(args.output), args.batch)