"""Arena modules for organizing arena.py components.

This package provides helper modules for the main arena.py Marimo application.
The Marimo cell structure is preserved in arena.py to maintain reactivity,
but business logic is delegated to these modules.
"""

from .shared_services import ArenaServices, init_services
from .formatters import format_activity_log, format_prompt_log
from .custom_game import parse_actions_list, create_custom_game
from .roles import RolesManager
from .play import (
    build_player_section,
    build_payoff_editor,
    apply_custom_payoffs_to_game,
    GameRunnerCallbacks,
    GameRunnerConfig,
    GameRunnerResult,
    GameRunnerService,
    extract_player_settings,
    format_action,
    build_session_config,
)
from .queue import QueuedGame, QueueExecutionResult, QueueUIBuilder

__all__ = [
    # Services
    "ArenaServices",
    "init_services",
    # Formatters
    "format_activity_log",
    "format_prompt_log",
    # Custom game
    "parse_actions_list",
    "create_custom_game",
    # Roles
    "RolesManager",
    # Play - UI builders
    "build_player_section",
    "build_payoff_editor",
    "apply_custom_payoffs_to_game",
    # Play - Game runner
    "GameRunnerCallbacks",
    "GameRunnerConfig",
    "GameRunnerResult",
    "GameRunnerService",
    "extract_player_settings",
    "format_action",
    "build_session_config",
    # Queue
    "QueuedGame",
    "QueueExecutionResult",
    "QueueUIBuilder",
]
