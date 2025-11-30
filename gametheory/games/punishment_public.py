"""Punishment Public Goods game definition."""

from ..core.types import GameDefinition

GAME = GameDefinition(
    id="punishment_public_goods",
    name="Punishment Public Goods",
    description="Players can contribute, free-ride, or punish a free-rider at a cost. "
                "Introduces enforcement mechanisms into the public goods dilemma.",
    payoff_matrix={
        # No punishment scenarios
        ("contribute", "contribute"): (2, 2),
        ("contribute", "free_ride"): (0, 3),
        ("free_ride", "contribute"): (3, 0),
        ("free_ride", "free_ride"): (1, 1),
        # Punishment scenarios
        ("punish", "free_ride"): (-1, 1),  # Punisher pays cost, free-rider penalized
        ("contribute", "punish"): (1, -1),
        ("punish", "contribute"): (1, -1),
        ("free_ride", "punish"): (-1, 1),
        # Edge cases
        ("punish", "punish"): (-2, -2),  # Both punishing each other
    },
    actions=["contribute", "free_ride", "punish"],
    num_players=2,
)
