"""Game execution service with callback-based UI decoupling.

This module extracts the core game execution logic from arena.py,
allowing the Marimo cell to focus on UI integration while the
service handles all execution details.
"""

from dataclasses import dataclass, field
from typing import (
    Callable, Optional, List, Dict, Any, Tuple,
    TYPE_CHECKING
)
from datetime import datetime
import asyncio

if TYPE_CHECKING:
    from gametheory import PlayerConfig, BurrGameRunner, MetricsTracker, SessionManager


@dataclass
class GameRunnerCallbacks:
    """Callbacks for UI integration during game execution.

    All callbacks are optional - if not provided, operations are no-ops.
    """
    on_activity: Callable[[Dict[str, Any]], None] = field(
        default_factory=lambda: lambda _: None
    )
    on_prompt: Callable[[Dict[str, Any]], None] = field(
        default_factory=lambda: lambda _: None
    )
    on_round_complete: Callable[[Dict[str, Any]], None] = field(
        default_factory=lambda: lambda _: None
    )


@dataclass
class GameRunnerConfig:
    """Configuration for game execution."""
    num_games: int
    runtime_mode: str  # "one_off", "repeated", "sequential", "multi_player"
    payoff_display: str
    timeout: float = 300.0


@dataclass
class GameRunnerResult:
    """Results from a game series execution."""
    results: List[Dict[str, Any]]
    prompt_log: List[Dict[str, Any]]
    activity_log: List[Dict[str, Any]]
    session_id: str
    elapsed: float
    success: bool
    error_message: Optional[str] = None


def extract_player_settings(
    role: Optional[Any],
    model_dropdown: Any,
    endpoint_dropdown: Any,
    temp_slider: Any,
    top_p_slider: Any,
    top_k_slider: Any,
    repeat_slider: Any,
    system_textarea: Any,
    strategy_textarea: Any,
    game_id: str,
) -> Dict[str, Any]:
    """Extract player settings from role or manual configuration.

    Args:
        role: RoleConfig if using a role, None for manual config
        model_dropdown: Model dropdown (manual config)
        endpoint_dropdown: Endpoint dropdown (manual config)
        temp_slider: Temperature slider (manual config)
        top_p_slider: Top-p slider (manual config)
        top_k_slider: Top-k slider (manual config)
        repeat_slider: Repeat penalty slider (manual config)
        system_textarea: System prompt textarea (manual config)
        strategy_textarea: Strategy hints textarea (both modes)
        game_id: Current game ID for game-specific instructions

    Returns:
        Dict with player configuration
    """
    if role:
        # Use role settings with optional session override
        strategy_hints = (
            strategy_textarea.value
            if strategy_textarea.value
            else role.get_game_instructions(game_id) or None
        )
        return {
            "model": role.model,
            "endpoint": role.endpoint,
            "temperature": role.temperature,
            "top_p": role.top_p,
            "top_k": role.top_k,
            "repeat_penalty": role.repeat_penalty,
            "system_prompt": role.system_prompt or None,
            "strategy_hints": strategy_hints,
            "role_id": role.role_id,
            "role_name": role.name,
        }
    else:
        # Use manual configuration
        return {
            "model": model_dropdown.value,
            "endpoint": endpoint_dropdown.value,
            "temperature": temp_slider.value,
            "top_p": top_p_slider.value,
            "top_k": top_k_slider.value,
            "repeat_penalty": repeat_slider.value,
            "system_prompt": system_textarea.value if system_textarea.value else None,
            "strategy_hints": strategy_textarea.value if strategy_textarea.value else None,
            "role_id": None,
            "role_name": None,
        }


def format_action(action: Any, game: Any) -> str:
    """Format action for display based on game type.

    Args:
        action: The action to format
        game: Game definition for type checking

    Returns:
        Formatted action string
    """
    if game.is_discrete:
        return str(action)
    elif hasattr(game.action_space, 'num_fields'):  # AllocationSpace
        return f"[{', '.join(f'{a:.0f}' for a in action)}]"
    return str(action)


def build_session_config(
    game: Any,
    game_id: str,
    all_player_settings: List[Dict[str, Any]],
    custom_payoffs: Dict[str, Any],
    has_custom_payoffs: bool,
    is_discrete: bool,
    num_players: int,
) -> Dict[str, Any]:
    """Build session configuration dict.

    Args:
        game: Game definition
        game_id: Game identifier
        all_player_settings: List of player settings dicts
        custom_payoffs: Custom payoff values
        has_custom_payoffs: Whether custom payoffs are used
        is_discrete: Whether this is a discrete game
        num_players: Number of players

    Returns:
        Session config dict
    """
    player_settings = {
        f"p{idx+1}": {
            "temperature": settings["temperature"],
            "top_p": settings["top_p"]
        }
        for idx, settings in enumerate(all_player_settings[:num_players])
    }

    if hasattr(game, 'payoff_matrix'):
        serialized_matrix = {
            "_".join(actions): list(payoffs)
            for actions, payoffs in game.payoff_matrix.items()
        }
        game_actions = game.actions
    else:
        serialized_matrix = None
        game_actions = f"ActionSpace:{type(game.action_space).__name__}"

    return {
        "custom_payoffs": custom_payoffs,
        "player_settings": player_settings,
        "uses_custom_payoffs": has_custom_payoffs,
        "payoff_matrix": serialized_matrix,
        "game_actions": game_actions,
        "game_name": game.name,
        "num_players": game.num_players,
        "is_discrete_game": is_discrete,
    }


class GameRunnerService:
    """Executes game series with callback-based progress reporting.

    Decouples game execution logic from Marimo UI, enabling
    testability and cleaner code organization.
    """

    def __init__(
        self,
        game: Any,
        players: List['PlayerConfig'],
        config: GameRunnerConfig,
        metrics: 'MetricsTracker',
        session_manager: 'SessionManager',
        session: Any,
        BurrGameRunner: type,
        callbacks: Optional[GameRunnerCallbacks] = None,
    ):
        """Initialize game runner service.

        Args:
            game: BurrGameDefinition to play
            players: List of PlayerConfig objects
            config: GameRunnerConfig with execution settings
            metrics: MetricsTracker for API metrics
            session_manager: SessionManager for persistence
            session: Pre-created session object
            BurrGameRunner: BurrGameRunner class
            callbacks: Optional callbacks for UI updates
        """
        self.game = game
        self.players = players
        self.config = config
        self.metrics = metrics
        self.session_manager = session_manager
        self.session = session
        self.BurrGameRunner = BurrGameRunner
        self.callbacks = callbacks or GameRunnerCallbacks()

        self.is_discrete = game.is_discrete if hasattr(game, 'is_discrete') else False
        self.num_players = game.num_players

        # Internal state
        self._activity_log: List[Dict[str, Any]] = []
        self._prompt_log: List[Dict[str, Any]] = []

    def _log_activity(self, level: str, message: str) -> None:
        """Log an activity entry."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
        }
        self._activity_log.append(entry)
        self.callbacks.on_activity(entry)

    def _log_prompt(self, entry: Dict[str, Any]) -> None:
        """Log a prompt entry."""
        self._prompt_log.append(entry)
        self.callbacks.on_prompt(entry)

    async def run(self) -> GameRunnerResult:
        """Execute the game series.

        Returns:
            GameRunnerResult with results, logs, and metadata
        """
        import time

        self._activity_log.clear()
        self._prompt_log.clear()

        self._log_activity("INFO", "Starting game series...")
        self.metrics.soft_reset()

        self._log_activity(
            "INFO",
            f"Game: {self.game.name} | Players: {self.num_players} | "
            f"Rounds: {self.config.num_games}"
        )

        # Log player configurations
        for idx, player in enumerate(self.players):
            role_info = "[Role]" if hasattr(player, 'role_id') and player.role_id else "[Manual]"
            self._log_activity(
                "INFO",
                f"P{idx+1}: {player.model} @ {player.endpoint} {role_info} "
                f"(temp={player.temperature}, sys_prompt={'YES' if player.system_prompt else 'NO'})"
            )

        self._log_activity("INFO", f"Session {self.session.session_id} started")

        # Create runner
        runner = self.BurrGameRunner(self.game, enable_tracking=True)

        start_time = time.time()
        results = []
        error_message = None

        try:
            results = await asyncio.wait_for(
                self._run_game_loop(runner),
                timeout=self.config.timeout
            )
            success = True
        except asyncio.TimeoutError:
            self._log_activity(
                "ERROR",
                f"Game series timed out after {self.config.timeout}s"
            )
            success = False
            error_message = f"Timeout after {self.config.timeout}s"

        elapsed = time.time() - start_time

        self._log_activity("SUCCESS", f"Completed {len(results)} games in {elapsed:.2f}s")

        # Save results
        self.session_manager.save_session_metadata(self.session)
        self.session_manager.save_results(
            self.session.session_id,
            results,
            self.session.game_type
        )

        self._log_activity("INFO", f"Results saved for session {self.session.session_id}")

        return GameRunnerResult(
            results=results,
            prompt_log=self._prompt_log.copy(),
            activity_log=self._activity_log.copy(),
            session_id=self.session.session_id,
            elapsed=elapsed,
            success=success,
            error_message=error_message,
        )

    async def _run_game_loop(self, runner) -> List[Dict[str, Any]]:
        """Run the main game loop using Burr state machine for tracking.

        Delegates to runner.run_series() which now handles all game types
        with full Burr tracking enabled.
        """
        import aiohttp

        self.metrics.log_request_start(self.config.num_games * self.num_players)

        # Create wrapper callbacks that do internal logging AND call external callbacks
        class WrappedCallbacks:
            def __init__(wrapper_self):
                wrapper_self.service = self

            def on_activity(wrapper_self, entry: Dict[str, Any]) -> None:
                # Store to internal log
                wrapper_self.service._activity_log.append(entry)
                # Call external callback
                wrapper_self.service.callbacks.on_activity(entry)

            def on_prompt(wrapper_self, entry: Dict[str, Any]) -> None:
                # Store to internal log
                wrapper_self.service._prompt_log.append(entry)
                # Call external callback
                wrapper_self.service.callbacks.on_prompt(entry)

            def on_round_complete(wrapper_self, result: Dict[str, Any]) -> None:
                # Call external callback
                wrapper_self.service.callbacks.on_round_complete(result)

        wrapped = WrappedCallbacks()

        connector = aiohttp.TCPConnector(limit=20, keepalive_timeout=30)
        async with aiohttp.ClientSession(connector=connector) as aio_session:
            # Use run_series which uses Burr state machine with tracking
            results = await runner.run_series(
                players=self.players,
                num_rounds=self.config.num_games,
                session=aio_session,
                payoff_display=self.config.payoff_display,
                runtime_mode=self.config.runtime_mode,
                callbacks=wrapped,
            )

        return results
