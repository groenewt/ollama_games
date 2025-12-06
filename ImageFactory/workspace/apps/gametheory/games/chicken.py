"""Chicken Game definition."""

from ..core.types import GameDefinition

GAME = GameDefinition(
    id="chicken",
    name="Chicken Game",
    description="Players choose to swerve or dare. If both dare, catastrophic "
                "mutual destruction ensues. The 'chicken' who swerves first loses face.",
    payoff_matrix={
        ("swerve", "swerve"): (0, 0),
        ("swerve", "dare"): (-1, 1),
        ("dare", "swerve"): (1, -1),
        ("dare", "dare"): (-10, -10),
    },
    actions=["swerve", "dare"],
    num_players=2,
)
