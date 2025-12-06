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
    discover_models,
    check_endpoint_available,
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
    "OLLAMA_MODELS",
    "OLLAMA_ENDPOINTS",
    "DEFAULT_OLLAMA_MODELS",
    "DEFAULT_OLLAMA_ENDPOINTS",
    "DEFAULT_TIMEOUT",
    "DISCOVERY_TIMEOUT",
    "SERIES_TIMEOUT",
    "DEFAULT_NUM_GAMES",
    "MAX_NUM_GAMES",
    "discover_all_available",
    "discover_models",
    "check_endpoint_available",
    "RuntimeMode",
    "GameDefinition",
    "PlayerConfig",
    "RoundResult",
    "SessionMetadata",
    "GameSession",
    "detect_num_players",
]
