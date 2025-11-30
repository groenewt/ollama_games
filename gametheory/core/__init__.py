"""Core module for game theory package."""

from .config import (
    OLLAMA_MODELS,
    OLLAMA_ENDPOINTS,
    DEFAULT_TIMEOUT,
    DEFAULT_NUM_GAMES,
    MAX_NUM_GAMES,
)
from .types import (
    RuntimeMode,
    GameDefinition,
    PlayerConfig,
    RoundResult,
    SessionMetadata,
    GameSession,
)

__all__ = [
    "OLLAMA_MODELS",
    "OLLAMA_ENDPOINTS",
    "DEFAULT_TIMEOUT",
    "DEFAULT_NUM_GAMES",
    "MAX_NUM_GAMES",
    "RuntimeMode",
    "GameDefinition",
    "PlayerConfig",
    "RoundResult",
    "SessionMetadata",
    "GameSession",
]
