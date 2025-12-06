"""Prisoner's Dilemma game definition."""

from ..core.types import GameDefinition

GAME = GameDefinition(
    id="prisoners_dilemma",
    name="Prisoner's Dilemma",
    description="A classic game where players choose to cooperate or defect. "
                "Mutual cooperation yields good payoffs for both, but defection "
                "tempts with higher individual gains at the other's expense.",
    payoff_matrix={
        ("cooperate", "cooperate"): (3, 3),
        ("cooperate", "defect"): (0, 5),
        ("defect", "cooperate"): (5, 0),
        ("defect", "defect"): (1, 1),
    },
    actions=["cooperate", "defect"],
    num_players=2,
)
