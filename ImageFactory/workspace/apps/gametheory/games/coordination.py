"""Coordination Game definition."""

from ..core.types import GameDefinition

GAME = GameDefinition(
    id="coordination_game",
    name="Coordination Game",
    description="Players benefit from selecting the same strategy. "
                "There's no dominant choice - coordination itself is the goal.",
    payoff_matrix={
        ("A", "A"): (2, 2),
        ("A", "B"): (0, 0),
        ("B", "A"): (0, 0),
        ("B", "B"): (2, 2),
    },
    actions=["A", "B"],
    num_players=2,
)
