"""Burr-based game engine for complex game types."""

from .burr_app import (
    BurrGameRunner,
    build_game_app,
    build_allocation_game_app,  # Deprecated alias for backwards compatibility
    AllocationSpace,
    BurrGameDefinition,
    play_game_round,
)
from .discrete_space import DiscreteActionSpace
from .permutation_space import PermutationSpace, SumoCoachSpace

__all__ = [
    "BurrGameRunner",
    "build_game_app",
    "build_allocation_game_app",  # Deprecated
    "play_game_round",
    "AllocationSpace",
    "DiscreteActionSpace",
    "PermutationSpace",
    "SumoCoachSpace",
    "BurrGameDefinition",
]
