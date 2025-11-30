"""Reusable Marimo UI component factories."""

from typing import Dict, List, Tuple, Any, Optional
import marimo as mo
import polars as pl

from ..core.config import OLLAMA_MODELS, OLLAMA_ENDPOINTS, DEFAULT_NUM_GAMES, MAX_NUM_GAMES
from ..core.types import GameDefinition
from ..games import get_game_names


def create_model_selector(
    player_num: int,
    default_index: int = 0,
) -> mo.ui.dropdown:
    """Create a model selector dropdown for a player.

    Args:
        player_num: The player number (1-indexed).
        default_index: Index of default model in OLLAMA_MODELS.

    Returns:
        A Marimo dropdown UI element.
    """
    return mo.ui.dropdown(
        options=OLLAMA_MODELS,
        label=f"Player {player_num} Model",
        value=OLLAMA_MODELS[min(default_index, len(OLLAMA_MODELS) - 1)],
    )


def create_endpoint_selector(
    player_num: int,
    default_index: int = 0,
) -> mo.ui.dropdown:
    """Create an endpoint selector dropdown for a player.

    Args:
        player_num: The player number (1-indexed).
        default_index: Index of default endpoint in OLLAMA_ENDPOINTS.

    Returns:
        A Marimo dropdown UI element.
    """
    return mo.ui.dropdown(
        options=OLLAMA_ENDPOINTS,
        label=f"Player {player_num} Endpoint",
        value=OLLAMA_ENDPOINTS[min(default_index, len(OLLAMA_ENDPOINTS) - 1)],
    )


def create_player_selector(
    player_num: int,
    default_model_index: int = 0,
    default_endpoint_index: int = 0,
) -> Tuple[mo.ui.dropdown, mo.ui.dropdown]:
    """Create model and endpoint selectors for a player.

    Args:
        player_num: The player number (1-indexed).
        default_model_index: Index of default model.
        default_endpoint_index: Index of default endpoint.

    Returns:
        Tuple of (model_dropdown, endpoint_dropdown).
    """
    model = create_model_selector(player_num, default_model_index)
    endpoint = create_endpoint_selector(player_num, default_endpoint_index)
    return model, endpoint


def create_game_selector() -> mo.ui.dropdown:
    """Create a game type selector dropdown.

    Returns:
        A Marimo dropdown UI element with all available games.
    """
    game_names = get_game_names()
    return mo.ui.dropdown(
        options={v: k for k, v in game_names.items()},  # Display name -> game_id
        label="Select Game Type",
        value="Prisoner's Dilemma",
    )


def create_runtime_selector() -> mo.ui.dropdown:
    """Create a runtime mode selector dropdown.

    Returns:
        A Marimo dropdown UI element.
    """
    return mo.ui.dropdown(
        options={
            "One-off": "one_off",
            "Repeated": "repeated",
            "Sequential": "sequential",
            "Multiplayer": "multi_player",
        },
        label="Select Runtime Mode",
        value="One-off",
    )


def create_game_controls(
    num_games_default: int = DEFAULT_NUM_GAMES,
    max_games: int = MAX_NUM_GAMES,
) -> Dict[str, Any]:
    """Create common game control UI elements.

    Args:
        num_games_default: Default number of games.
        max_games: Maximum number of games allowed.

    Returns:
        Dictionary with 'num_games' slider and 'run_button'.
    """
    return {
        "num_games": mo.ui.slider(
            1, max_games, value=num_games_default, label="Number of games to play"
        ),
        "run_button": mo.ui.run_button(
            label="Run Game Series",
            tooltip="Start the game series with selected configuration",
        ),
    }


def create_payoff_matrix_display(game: GameDefinition) -> mo.Html:
    """Render payoff matrix as formatted table.

    Args:
        game: The game definition.

    Returns:
        A Marimo HTML element with the payoff matrix.
    """
    if game.num_players == 2:
        rows = [
            {
                "Player 1 Action": a1,
                "Player 2 Action": a2,
                "Player 1 Payoff": p1,
                "Player 2 Payoff": p2,
            }
            for (a1, a2), (p1, p2) in game.payoff_matrix.items()
        ]
    else:
        # Handle 3+ player games
        rows = []
        for actions, payoffs in game.payoff_matrix.items():
            row = {}
            for i, action in enumerate(actions):
                row[f"Player {i+1} Action"] = action
            for i, payoff in enumerate(payoffs):
                row[f"Player {i+1} Payoff"] = payoff
            rows.append(row)

    df = pl.DataFrame(rows)
    return df


def create_metrics_panel(metrics: Dict[str, Any]) -> mo.Html:
    """Create a metrics summary panel.

    Args:
        metrics: Dictionary of metrics from MetricsTracker.to_dict().

    Returns:
        A Marimo vstack with metrics display.
    """
    return mo.vstack([
        mo.hstack([
            mo.stat(
                label="Total Requests",
                value=str(metrics.get("total_requests", 0)),
            ),
            mo.stat(
                label="Success Rate",
                value=f"{metrics.get('success_rate', 100):.1f}%",
            ),
            mo.stat(
                label="Avg Response Time",
                value=f"{metrics.get('avg_response_time', 0):.2f}s",
            ),
            mo.stat(
                label="Elapsed Time",
                value=f"{metrics.get('elapsed_seconds', 0):.1f}s",
            ),
        ]),
    ])


def create_game_info_panel(game: GameDefinition) -> mo.Html:
    """Create a game information panel.

    Args:
        game: The game definition.

    Returns:
        A Marimo vstack with game info and payoff matrix.
    """
    return mo.vstack([
        mo.md(f"### {game.name}"),
        mo.md(game.description),
        mo.md("#### Payoff Matrix:"),
        create_payoff_matrix_display(game),
    ])


def create_config_panel(
    game_selector: mo.ui.dropdown,
    runtime_selector: mo.ui.dropdown,
    num_games: mo.ui.slider,
    player_selectors: List[Tuple[mo.ui.dropdown, mo.ui.dropdown]],
    run_button: mo.ui.run_button,
    show_player3: bool = False,
) -> mo.Html:
    """Create a complete configuration panel.

    Args:
        game_selector: Game type dropdown.
        runtime_selector: Runtime mode dropdown.
        num_games: Number of games slider.
        player_selectors: List of (model, endpoint) tuples per player.
        run_button: Run button element.
        show_player3: Whether to show Player 3 selectors.

    Returns:
        A Marimo vstack with all configuration elements.
    """
    rows = [
        mo.md("# Game Configuration"),
        game_selector,
        runtime_selector,
        num_games,
    ]

    # Add player selectors
    for i, (model, endpoint) in enumerate(player_selectors[:2]):
        rows.append(mo.hstack([model, endpoint]))

    # Optionally add Player 3
    if show_player3 and len(player_selectors) > 2:
        model3, endpoint3 = player_selectors[2]
        rows.append(mo.hstack([model3, endpoint3]))

    rows.append(run_button)

    return mo.vstack(rows)
