"""
Configuration loader for the Redrob Ranker.
Parses settings.yaml into strongly typed dataclasses.
"""
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class WeightsConfig:
    semantic: float = 0.55
    behavioral: float = 0.25
    experience: float = 0.20

@dataclass
class ThresholdsConfig:
    max_ideal_experience_years: float = 15.0
    max_ideal_notice_days: float = 90.0
    honeypot_penalty_multiplier: float = 0.1
    good_semantic_score: float = 0.60
    good_response_rate: float = 0.80
    immediate_notice_days: float = 30.0

@dataclass
class BehavioralMultipliers:
    response_rate_weight: float = 0.6
    completeness_weight: float = 0.2
    notice_period_weight: float = 0.2
    fallback_response_rate: float = 0.1

@dataclass
class AppConfig:
    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 2048
    weights: WeightsConfig = field(default_factory=WeightsConfig)
    thresholds: ThresholdsConfig = field(default_factory=ThresholdsConfig)
    behavioral: BehavioralMultipliers = field(default_factory=BehavioralMultipliers)

def load_config(config_path: Path) -> AppConfig:
    """Loads configuration from YAML file safely."""
    if not config_path.exists():
        import logging
        logging.warning(f"Config file {config_path} not found. Using defaults.")
        return AppConfig()
    
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
        
    return AppConfig(
        model_name=data.get("model", {}).get("name", "all-MiniLM-L6-v2"),
        batch_size=data.get("model", {}).get("batch_size", 2048),
        weights=WeightsConfig(**data.get("weights", {})),
        thresholds=ThresholdsConfig(**data.get("thresholds", {})),
        behavioral=BehavioralMultipliers(**data.get("behavioral_multipliers", {}))
    )