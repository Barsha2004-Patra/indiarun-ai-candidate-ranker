"""
Deterministic reason generation module. No LLMs to ensure strict offline compliance and speed.
"""
import pandas as pd
from src.config import ThresholdsConfig

def generate_reasoning(row: pd.Series, t_config: ThresholdsConfig) -> str:
    """Generates a dynamic, factual 1-2 sentence reasoning."""
    reasons = []
    
    if row['semantic_score'] > t_config.good_semantic_score:
        reasons.append(f"Strong semantic alignment with core JD requirements ({row['semantic_score']:.2f} score).")
    else:
        reasons.append("Moderate baseline technical match.")
        
    reasons.append(f"Brings {row['years_of_experience']:.1f} years of experience as a {row['current_title']}.")
    
    if row['recruiter_response_rate'] >= t_config.good_response_rate:
        reasons.append("Highly engaged candidate with excellent recruiter response rates.")
    elif row['notice_period_days'] <= t_config.immediate_notice_days:
        reasons.append("Available for immediate hiring (sub-30 day notice).")
        
    return " ".join(reasons)