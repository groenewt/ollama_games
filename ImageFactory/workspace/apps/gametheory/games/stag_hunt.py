"""Stag Hunt game definition."""

from .matrix_factory import create_matrix_game

GAME = create_matrix_game(
    id="stag_hunt",
    name="Stag Hunt",
    description="""Players choose to hunt stag (cooperatively) or hare (alone).

Strategic considerations:
- Mutual stag hunting (5,5) yields the best collective outcome
- Hunting hare (2,2) is the safe, risk-dominant equilibrium
- If you hunt stag alone while opponent hunts hare, you get nothing (0)
- Hunting hare guarantees at least 2-3 points regardless of opponent
- The question: can you trust your partner to cooperate on the risky but rewarding stag?""",
    actions=["stag", "hare"],
    matrix={
        ("stag", "stag"): (5, 5),
        ("stag", "hare"): (0, 3),
        ("hare", "stag"): (3, 0),
        ("hare", "hare"): (2, 2),
    },
)
