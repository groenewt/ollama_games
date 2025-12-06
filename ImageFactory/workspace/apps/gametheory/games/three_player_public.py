"""Three-Player Public Good game definition."""

from ..core.types import GameDefinition

GAME = GameDefinition(
    id="three_player_public_good",
    name="Three-Player Public Good",
    description="Three players decide to contribute or free-ride. The public good "
                "yields a benefit if at least two contribute.",
    payoff_matrix={
        # All contribute
        ("contribute", "contribute", "contribute"): (3, 3, 3),
        # Two contribute, one free-ride
        ("contribute", "contribute", "free_ride"): (1, 1, 5),
        ("contribute", "free_ride", "contribute"): (1, 5, 1),
        ("free_ride", "contribute", "contribute"): (5, 1, 1),
        # One contributes, two free-ride (public good fails)
        ("contribute", "free_ride", "free_ride"): (0, 2, 2),
        ("free_ride", "contribute", "free_ride"): (2, 0, 2),
        ("free_ride", "free_ride", "contribute"): (2, 2, 0),
        # All free-ride
        ("free_ride", "free_ride", "free_ride"): (1, 1, 1),
    },
    actions=["contribute", "free_ride"],
    num_players=3,
)
