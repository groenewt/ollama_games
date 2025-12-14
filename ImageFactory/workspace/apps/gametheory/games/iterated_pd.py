"""Iterated Prisoner's Dilemma game definition."""

from .matrix_factory import create_matrix_game

GAME = create_matrix_game(
    id="iterated_prisoners_dilemma",
    name="Iterated Prisoner's Dilemma",
    description="""Repeated Prisoner's Dilemma where players can condition on previous rounds.

Strategic considerations:
- In a single round, defection dominates - but this is REPEATED
- The shadow of the future enables cooperation strategies
- Tit-for-tat: cooperate first, then mirror opponent's last move
- Grim trigger: cooperate until opponent defects, then always defect
- Forgiveness matters: occasional cooperation can restore trust
- Watch opponent's pattern: are they cooperative, exploitative, or reactive?
- Your reputation matters: consistent behavior signals your strategy""",
    actions=["cooperate", "defect"],
    matrix={
        ("cooperate", "cooperate"): (3, 3),
        ("cooperate", "defect"): (0, 5),
        ("defect", "cooperate"): (5, 0),
        ("defect", "defect"): (1, 1),
    },
)
