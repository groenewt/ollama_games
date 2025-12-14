"""Three-Player Public Good game definition."""

from .matrix_factory import create_matrix_game

GAME = create_matrix_game(
    id="three_player_public_good",
    name="Three-Player Public Good",
    description="""Three players decide whether to contribute to a public good. The good is created if at least two contribute.

Strategic considerations:
- Universal contribution (3,3,3) benefits everyone equally
- Being the lone free-rider when two others contribute yields 5 points
- Being a lone contributor yields 0 while free-riders get 2 each
- Universal free-riding (1,1,1) is a poor equilibrium
- The threshold mechanic (need 2 contributors) creates strategic complexity
- You need to predict whether BOTH others will contribute or free-ride""",
    actions=["contribute", "free_ride"],
    matrix={
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
    num_players=3,
)
