"""Chicken Game definition."""

from .matrix_factory import create_matrix_game

GAME = create_matrix_game(
    id="chicken",
    name="Chicken Game",
    description="""Players simultaneously choose to swerve or dare in a head-on confrontation.

Strategic considerations:
- Both swerving (0,0) is safe but neither gains advantage
- If you dare and opponent swerves, you win (+1 vs -1)
- If both dare, catastrophic mutual destruction (-10,-10)
- The key is predicting whether opponent will "chicken out"
- No dominant strategy - outcome depends on reading opponent's commitment""",
    actions=["swerve", "dare"],
    matrix={
        ("swerve", "swerve"): (0, 0),
        ("swerve", "dare"): (-1, 1),
        ("dare", "swerve"): (1, -1),
        ("dare", "dare"): (-10, -10),
    },
)
