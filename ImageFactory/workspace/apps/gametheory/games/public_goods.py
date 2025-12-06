"""Public Goods Game definition."""

from ..core.types import GameDefinition

GAME = GameDefinition(
    id="public_goods",
    name="Public Goods Game",
    description="Players decide whether to contribute to a public good or free-ride. "
                "Free-riding is tempting but mutual contribution benefits everyone.",
    payoff_matrix={
        ("contribute", "contribute"): (2, 2),
        ("contribute", "free_ride"): (0, 3),
        ("free_ride", "contribute"): (3, 0),
        ("free_ride", "free_ride"): (1, 1),
    },
    actions=["contribute", "free_ride"],
    num_players=2,
)
