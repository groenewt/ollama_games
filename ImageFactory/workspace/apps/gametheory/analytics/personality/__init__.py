"""Personality profiling module.

Provides comprehensive model personality analysis through:
- BiasAnalyzer: Concentration bias and symmetry detection
- BehaviorAnalyzer: Adaptability, risk, temporal patterns
- GameFingerprintBuilder: Per-game allocation patterns
- ModelPersonalityProfiler: Complete profile orchestration
"""

from .types import (
    ModelPersonalityProfile,
    FieldPreference,
    GameFingerprint,
)
from .base import BasePersonalityAnalyzer
from .bias_analyzer import BiasAnalyzer
from .behavior_analyzer import BehaviorAnalyzer
from .game_fingerprint import GameFingerprintBuilder
from .profiler import ModelPersonalityProfiler

__all__ = [
    # Data types
    "ModelPersonalityProfile",
    "FieldPreference",
    "GameFingerprint",
    # Analyzers (new granular access)
    "BasePersonalityAnalyzer",
    "BiasAnalyzer",
    "BehaviorAnalyzer",
    "GameFingerprintBuilder",
    # Main profiler (backward compatible)
    "ModelPersonalityProfiler",
]
