"""Payoff editor utilities for arena."""

from typing import Dict, Tuple, Any, List, Optional


def build_payoff_inputs(
    base_game: Any,
    current_custom_payoffs: Dict[str, Any],
    mo,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Build payoff input widgets and track changes.

    Args:
        base_game: GameDefinition with payoff_matrix
        current_custom_payoffs: Dict of custom payoff values
        mo: Marimo module

    Returns:
        Tuple of (payoff_inputs dict, changes list)
    """
    payoff_inputs = {}
    custom_payoffs_changes = []

    for actions, payoffs in base_game.payoff_matrix.items():
        action_key = "_".join(actions)
        for i, default_val in enumerate(payoffs):
            input_key = f"{action_key}_p{i+1}"
            current_val = current_custom_payoffs.get(input_key, default_val)

            # Track changes for visual indicator
            if current_val != default_val:
                custom_payoffs_changes.append({
                    "action": " vs ".join(actions),
                    "player": i + 1,
                    "default": default_val,
                    "custom": current_val,
                })

            payoff_inputs[input_key] = mo.ui.number(
                value=current_val,
                label=f"P{i+1}",
                start=-100,
                stop=100,
            )

    return payoff_inputs, custom_payoffs_changes


def build_payoff_rows(
    base_game: Any,
    payoff_inputs: Dict[str, Any],
    mo,
) -> List[Any]:
    """Build UI rows for payoff inputs grouped by action.

    Args:
        base_game: GameDefinition with payoff_matrix
        payoff_inputs: Dict of payoff input widgets
        mo: Marimo module

    Returns:
        List of Marimo hstack rows
    """
    payoff_rows = []
    for actions in base_game.payoff_matrix.keys():
        action_key = "_".join(actions)
        action_label = " vs ".join(actions)
        row_inputs = [
            payoff_inputs[f"{action_key}_p{i+1}"]
            for i in range(base_game.num_players)
        ]
        payoff_rows.append(mo.hstack([mo.md(f"**{action_label}:**")] + row_inputs))
    return payoff_rows


def build_payoff_editor(
    base_game: Any,
    current_custom_payoffs: Dict[str, Any],
    game_id: str,
    get_custom_payoffs,
    set_custom_payoffs,
    mo,
) -> Tuple[Any, Dict[str, Any], List[Dict[str, Any]]]:
    """Build the complete payoff editor UI.

    Args:
        base_game: GameDefinition with payoff_matrix
        current_custom_payoffs: Current custom payoffs for this game
        game_id: Game identifier
        get_custom_payoffs: Reactive getter function
        set_custom_payoffs: Reactive setter function
        mo: Marimo module

    Returns:
        Tuple of (payoff_editor_ui, payoff_inputs, changes)
    """
    payoff_inputs, custom_payoffs_changes = build_payoff_inputs(
        base_game, current_custom_payoffs, mo
    )
    payoff_rows = build_payoff_rows(base_game, payoff_inputs, mo)

    def _apply_custom_payoffs():
        new_payoffs = {}
        for key, inp in payoff_inputs.items():
            new_payoffs[key] = inp.value
        current = get_custom_payoffs()
        set_custom_payoffs({**current, game_id: new_payoffs})

    apply_button = mo.ui.button(
        label="Apply Custom Payoffs",
        on_click=lambda _: _apply_custom_payoffs()
    )

    def _reset_payoffs():
        current = get_custom_payoffs()
        set_custom_payoffs({k: v for k, v in current.items() if k != game_id})

    reset_button = mo.ui.button(
        label="Reset to Default",
        on_click=lambda _: _reset_payoffs()
    )

    payoff_editor_ui = mo.accordion({
        "Custom Payoffs": mo.vstack([
            mo.md("Edit payoff values for each action combination:"),
            *payoff_rows,
            mo.hstack([apply_button, reset_button]),
        ])
    })

    return payoff_editor_ui, payoff_inputs, custom_payoffs_changes


def apply_custom_payoffs_to_game(
    base_game: Any,
    custom_payoffs: Dict[str, Any],
    GameDefinition,
) -> Any:
    """Create a new game with custom payoffs applied.

    Args:
        base_game: Original GameDefinition
        custom_payoffs: Dict of custom payoff values keyed by action_p{n}
        GameDefinition: GameDefinition class to instantiate

    Returns:
        New GameDefinition with custom payoffs, or original if no changes
    """
    if not custom_payoffs:
        return base_game

    new_matrix = {}
    for actions, payoffs in base_game.payoff_matrix.items():
        action_key = "_".join(actions)
        new_matrix[actions] = tuple(
            custom_payoffs.get(f"{action_key}_p{i+1}", payoffs[i])
            for i in range(base_game.num_players)
        )

    return GameDefinition(
        id=base_game.id,
        name=base_game.name + " (Custom)",
        description=base_game.description,
        payoff_matrix=new_matrix,
        actions=base_game.actions,
        num_players=base_game.num_players,
        is_sequential=getattr(base_game, 'is_sequential', False),
        memory_depth=getattr(base_game, 'memory_depth', 0),
    )
