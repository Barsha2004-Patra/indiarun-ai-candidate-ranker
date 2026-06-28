"""
Risk detection and penalty module.
"""
import numpy as np
import pandas as pd
from src.config import ThresholdsConfig

def apply_risk_penalties(df: pd.DataFrame, base_score: pd.Series, t_config: ThresholdsConfig) -> pd.Series:
    """Applies multiplier penalties to identified honeypots or risky candidates."""
    return np.where(
        df['is_risk_honeypot'], 
        base_score * t_config.honeypot_penalty_multiplier, 
        base_score
    )