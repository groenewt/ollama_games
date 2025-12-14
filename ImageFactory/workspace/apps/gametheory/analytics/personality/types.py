"""Data types for personality profiling."""

from dataclasses import dataclass
from typing import List, Dict


@dataclass
class ModelPersonalityProfile:
    """Complete personality profile for a model."""
    model_name: str
    bias_score: float              # -1 (clustered) to +1 (uniform tendency)
    symmetry_breaking_score: float # How much model favors specific fields (0-1)
    preferred_fields: List[int]    # Fields ranked by preference (0-indexed)
    game_fingerprints: Dict[str, Dict]  # game_id -> allocation patterns
    consistency_score: float       # Cross-game stability (0-1)
    total_sessions: int
    total_rounds: int
    avg_concentration: float       # Average HHI across all games
    avg_entropy: float             # Average normalized entropy
    dominant_strategy: str         # Most common strategy type
    # Enhanced personality metrics
    adaptability_score: float = 0.0      # 0-1: How much strategy changes vs opponent
    risk_tolerance: float = 0.5          # 0-1: Preference for high-variance plays
    temporal_pattern: str = "consistent" # "early_aggressive", "late_aggressive", "consistent"
    early_game_concentration: float = 0.0  # Avg HHI in first half of sessions
    late_game_concentration: float = 0.0   # Avg HHI in second half of sessions


@dataclass
class FieldPreference:
    """Field preference analysis for a model."""
    model_name: str
    game_type: str
    num_fields: int
    avg_allocations: List[float]   # Average allocation per field
    std_allocations: List[float]   # Std dev per field
    preference_ranking: List[int]  # Fields ranked by preference (0-indexed)
    preference_strength: float     # How strong the preference is (0-1)


@dataclass
class GameFingerprint:
    """Allocation fingerprint for a model in a specific game."""
    model_name: str
    game_type: str
    avg_concentration: float
    avg_entropy: float
    dominant_strategy: str
    strategy_distribution: Dict[str, float]  # strategy_type -> frequency
    field_preferences: List[float]  # Average proportion per field
    num_sessions: int
    num_rounds: int
