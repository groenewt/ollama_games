"""Stag Hunt game definition."""

from ..core.types import GameDefinition

GAME = GameDefinition(
    id="stag_hunt",
    name="Stag Hunt",
    description="Players choose to hunt stag or hare. Hunting stag requires "
                "cooperation and yields the best outcome, but hunting hare is safer.",
    payoff_matrix={
        ("stag", "stag"): (5, 5),
        ("stag", "hare"): (0, 3),
        ("hare", "stag"): (3, 0),
        ("hare", "hare"): (2, 2),
    },
    actions=["stag", "hare"],
    num_players=2,
)
