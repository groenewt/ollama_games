"""Play tab modules for arena."""

from .player_config import build_player_section
from .payoff_editor import build_payoff_editor, apply_custom_payoffs_to_game
from .game_runner import (
    GameRunnerCallbacks,
    GameRunnerConfig,
    GameRunnerResult,
    GameRunnerService,
    extract_player_settings,
    format_action,
    build_session_config,
)

__all__ = [
    # Player config
    "build_player_section",
    # Payoff editor
    "build_payoff_editor",
    "apply_custom_payoffs_to_game",
    # Game runner
    "GameRunnerCallbacks",
    "GameRunnerConfig",
    "GameRunnerResult",
    "GameRunnerService",
    "extract_player_settings",
    "format_action",
    "build_session_config",
]
