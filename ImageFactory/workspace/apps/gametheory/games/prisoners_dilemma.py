"""Prisoner's Dilemma game definition."""

from .matrix_factory import create_matrix_game

GAME = create_matrix_game(
    id="prisoners_dilemma",
    name="Prisoner's Dilemma",
    description="""A classic game where two players simultaneously choose to cooperate or defect.

Strategic considerations:
- Mutual cooperation (3,3) yields the best collective outcome
- But defection tempts with a higher payoff (5) if opponent cooperates
- Mutual defection (1,1) is the Nash equilibrium - neither can improve by changing alone
- The dilemma: individual rationality leads to worse collective outcome
- In repeated games, strategies like tit-for-tat can sustain cooperation""",
    actions=["cooperate", "defect"],
    matrix={
        ("cooperate", "cooperate"): (3, 3),
        ("cooperate", "defect"): (0, 5),
        ("defect", "cooperate"): (5, 0),
        ("defect", "defect"): (1, 1),
    },
)
