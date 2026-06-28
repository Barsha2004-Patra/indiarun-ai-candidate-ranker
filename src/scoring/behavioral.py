"""
Behavioral signal heuristic module.
"""
import pandas as pd
from src.config import BehavioralMultipliers, ThresholdsConfig

def calculate_behavioral_score(df: pd.DataFrame, b_config: BehavioralMultipliers, t_config: ThresholdsConfig) -> pd.Series:
    """Fuses Redrob behavioral signals into a normalized multiplier."""
    response_rate = df['recruiter_response_rate'].fillna(b_config.fallback_response_rate)
    completeness = (df['profile_completeness'] / 100.0)
    
    # Notice period normalization
    notice_days_capped = df['notice_period_days'].clip(0, t_config.max_ideal_notice_days)
    notice_score = (t_config.max_ideal_notice_days - notice_days_capped) / t_config.max_ideal_notice_days
    
    return (
        (response_rate * b_config.response_rate_weight) + 
        (completeness * b_config.completeness_weight) + 
        (notice_score * b_config.notice_period_weight)
    )