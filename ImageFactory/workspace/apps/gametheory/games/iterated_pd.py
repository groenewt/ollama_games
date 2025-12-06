"""Iterated Prisoner's Dilemma game definition."""

from ..core.types import GameDefinition

GAME = GameDefinition(
    id="iterated_prisoners_dilemma",
    name="Iterated Prisoner's Dilemma",
    description="Repeated Prisoner's Dilemma where each player can condition "
                "on the opponent's previous move. Enables strategies like tit-for-tat.",
    payoff_matrix={
        ("cooperate", "cooperate"): (3, 3),
        ("cooperate", "defect"): (0, 5),
        ("defect", "cooperate"): (5, 0),
        ("defect", "defect"): (1, 1),
    },
    actions=["cooperate", "defect"],
    num_players=2,
    is_sequential=True,
    memory_depth=1,
)
