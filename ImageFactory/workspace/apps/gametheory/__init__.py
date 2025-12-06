"""Game Theory LLM Arena - A framework for testing AI models in strategic games."""

from .core import (
    OLLAMA_MODELS,
    OLLAMA_ENDPOINTS,
    DEFAULT_OLLAMA_MODELS,
    DEFAULT_OLLAMA_ENDPOINTS,
    DEFAULT_TIMEOUT,
    DEFAULT_NUM_GAMES,
    RuntimeMode,
    GameDefinition,
    PlayerConfig,
    RoundResult,
    SessionMetadata,
    GameSession,
    discover_all_available,
    discover_models,
    check_endpoint_available,
)
from .games import (
    GAME_REGISTRY,
    get_game,
    list_games,
    get_game_names,
)
from .model import OllamaClient, GameRunner
from .metrics import MetricsTracker, SessionManager, CrossGameAnalyzer
from .analytics import AnalyticsService

__version__ = "0.1.0"

__all__ = [
    # Core
    "OLLAMA_MODELS",
    "OLLAMA_ENDPOINTS",
    "DEFAULT_TIMEOUT",
    "DEFAULT_NUM_GAMES",
    "RuntimeMode",
    "GameDefinition",
    "PlayerConfig",
    "RoundResult",
    "SessionMetadata",
    "GameSession",
    # Games
    "GAME_REGISTRY",
    "get_game",
    "list_games",
    "get_game_names",
    # Model
    "OllamaClient",
    "GameRunner",
    # Metrics
    "MetricsTracker",
    "SessionManager",
    "CrossGameAnalyzer",
    # Analytics
    "AnalyticsService",
]
