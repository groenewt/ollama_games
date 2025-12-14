"""Factory for creating discrete matrix games.

This module provides a factory function to create BurrGameDefinition
instances from simple payoff matrices, reducing boilerplate code.
"""

from typing import Dict, List, Tuple, Union

from ..engine import BurrGameDefinition, DiscreteActionSpace

PayoffValue = Union[int, float]


def create_matrix_game(
    id: str,
    name: str,
    description: str,
    actions: List[str],
    matrix: Dict[Tuple[str, ...], Tuple[PayoffValue, ...]],
    num_players: int = 2,
    default_payoff: Tuple[PayoffValue, ...] = None,
) -> BurrGameDefinition:
    """Create a discrete matrix game from a payoff matrix.

    Args:
        id: Unique game identifier (e.g., "prisoners_dilemma")
        name: Display name (e.g., "Prisoner's Dilemma")
        description: Full game description with strategic considerations
        actions: List of valid action strings
        matrix: Dict mapping action tuples to payoff tuples
        num_players: Number of players (default 2)
        default_payoff: Payoff for invalid action combinations (default: zeros)

    Returns:
        A BurrGameDefinition ready for use with BurrGameRunner.

    Raises:
        ValueError: If matrix entries don't match num_players or contain
            invalid actions.

    Example:
        GAME = create_matrix_game(
            id="prisoners_dilemma",
            name="Prisoner's Dilemma",
            description="A classic game...",
            actions=["cooperate", "defect"],
            matrix={
                ("cooperate", "cooperate"): (3, 3),
                ("cooperate", "defect"): (0, 5),
                ("defect", "cooperate"): (5, 0),
                ("defect", "defect"): (1, 1),
            },
        )
    """
    if default_payoff is None:
        default_payoff = tuple(0 for _ in range(num_players))

    # Validate matrix entries
    for action_tuple, payoff_tuple in matrix.items():
        if len(action_tuple) != num_players:
            raise ValueError(
                f"Action tuple {action_tuple} has {len(action_tuple)} entries, "
                f"expected {num_players}"
            )
        if len(payoff_tuple) != num_players:
            raise ValueError(
                f"Payoff tuple {payoff_tuple} has {len(payoff_tuple)} entries, "
                f"expected {num_players}"
            )
        for action in action_tuple:
            if action not in actions:
                raise ValueError(
                    f"Unknown action '{action}' in matrix. Valid actions: {actions}"
                )

    # Create closure for payoff function
    def payoff_fn(player_actions: Tuple[str, ...]) -> Tuple[PayoffValue, ...]:
        return matrix.get(player_actions, default_payoff)

    return BurrGameDefinition(
        id=id,
        name=name,
        description=description,
        action_space=DiscreteActionSpace(actions=actions),
        payoff_fn=payoff_fn,
        num_players=num_players,
    )
