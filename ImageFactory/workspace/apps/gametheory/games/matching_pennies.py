"""Matching Pennies game definition."""

from .matrix_factory import create_matrix_game

GAME = create_matrix_game(
    id="matching_pennies",
    name="Matching Pennies",
    description="""A zero-sum game where Player 1 wins if pennies match, Player 2 wins if they differ.

Strategic considerations:
- This is a strictly competitive (zero-sum) game
- Player 1 wants to match; Player 2 wants to mismatch
- No pure strategy Nash equilibrium exists
- Optimal play is random: 50% heads, 50% tails
- Any predictable pattern can be exploited by the opponent
- In repeated play, try to detect patterns in opponent's choices""",
    actions=["heads", "tails"],
    matrix={
        ("heads", "heads"): (1, -1),
        ("heads", "tails"): (-1, 1),
        ("tails", "heads"): (-1, 1),
        ("tails", "tails"): (1, -1),
    },
)
