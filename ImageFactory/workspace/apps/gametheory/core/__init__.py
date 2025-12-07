"""Core module for game theory package."""

from .config import (
    OLLAMA_MODELS,
    OLLAMA_ENDPOINTS,
    DEFAULT_OLLAMA_MODELS,
    DEFAULT_OLLAMA_ENDPOINTS,
    DEFAULT_TIMEOUT,
    DISCOVERY_TIMEOUT,
    SERIES_TIMEOUT,
    DEFAULT_NUM_GAMES,
    MAX_NUM_GAMES,
    discover_all_available,
)
from .types import (
    RuntimeMode,
    GameDefinition,
    PlayerConfig,
    RoundResult,
    SessionMetadata,
    GameSession,
)
from .utils import detect_num_players

__all__ = [
    # Configuration constants
    "OLLAMA_MODELS",
    "OLLAMA_ENDPOINTS",
    "DEFAULT_OLLAMA_MODELS",
    "DEFAULT_OLLAMA_ENDPOINTS",
    "DEFAULT_TIMEOUT",
    "DISCOVERY_TIMEOUT",
    "SERIES_TIMEOUT",
    "DEFAULT_NUM_GAMES",
    "MAX_NUM_GAMES",
    # Discovery
    "discover_all_available",
    # Types
    "RuntimeMode",
    "GameDefinition",
    "PlayerConfig",
    "RoundResult",
    "SessionMetadata",
    "GameSession",
    # Utilities
    "detect_num_players",
]
