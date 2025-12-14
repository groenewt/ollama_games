"""Game Theory LLM Arena - A framework for testing AI models in strategic games."""

from .core import (
    OLLAMA_MODELS,
    OLLAMA_ENDPOINTS,
    DEFAULT_OLLAMA_MODELS,
    DEFAULT_OLLAMA_ENDPOINTS,
    DEFAULT_TIMEOUT,
    DISCOVERY_TIMEOUT,
    SERIES_TIMEOUT,
    DEFAULT_NUM_GAMES,
    MAX_NUM_GAMES,
    RuntimeMode,
    GameDefinition,
    PlayerConfig,
    RoundResult,
    SessionMetadata,
    GameSession,
    RoleConfig,
    discover_all_available,
)
from .storage import RoleRepository
from .games import (
    GAME_REGISTRY,
    BURR_GAME_REGISTRY,
    get_game,
    list_games,
    get_game_names,
    is_burr_game,
    is_discrete_game,
)
from .model import OllamaClient, GameRunner
from .engine import BurrGameRunner, BurrGameDefinition, AllocationSpace, DiscreteActionSpace
from .metrics import MetricsTracker, SessionManager, CrossGameAnalyzer
from .analytics import AnalyticsService

__version__ = "0.1.0"

__all__ = [
    # Configuration
    "OLLAMA_MODELS",
    "OLLAMA_ENDPOINTS",
    "DEFAULT_OLLAMA_MODELS",
    "DEFAULT_OLLAMA_ENDPOINTS",
    "DEFAULT_TIMEOUT",
    "DISCOVERY_TIMEOUT",
    "SERIES_TIMEOUT",
    "DEFAULT_NUM_GAMES",
    "MAX_NUM_GAMES",
    # Types
    "RuntimeMode",
    "GameDefinition",
    "PlayerConfig",
    "RoundResult",
    "SessionMetadata",
    "GameSession",
    "RoleConfig",
    # Storage
    "RoleRepository",
    # Discovery
    "discover_all_available",
    # Games
    "GAME_REGISTRY",
    "BURR_GAME_REGISTRY",
    "get_game",
    "list_games",
    "get_game_names",
    "is_burr_game",
    "is_discrete_game",
    # Model (legacy - GameRunner deprecated, use BurrGameRunner)
    "OllamaClient",
    "GameRunner",
    # Burr Engine (unified runner for all games)
    "BurrGameRunner",
    "BurrGameDefinition",
    "AllocationSpace",
    "DiscreteActionSpace",
    # Metrics
    "MetricsTracker",
    "SessionManager",
    "CrossGameAnalyzer",
    # Analytics
    "AnalyticsService",
]
