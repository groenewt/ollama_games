"""Matching Pennies game definition."""

from ..core.types import GameDefinition

GAME = GameDefinition(
    id="matching_pennies",
    name="Matching Pennies",
    description="Zero-sum game where one player wins if the choices match, "
                "the other wins if they differ.",
    payoff_matrix={
        ("heads", "heads"): (1, -1),
        ("heads", "tails"): (-1, 1),
        ("tails", "heads"): (-1, 1),
        ("tails", "tails"): (1, -1),
    },
    actions=["heads", "tails"],
    num_players=2,
)
