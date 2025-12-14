"""Punishment Public Goods game definition."""

from .matrix_factory import create_matrix_game

GAME = create_matrix_game(
    id="punishment_public_goods",
    name="Punishment Public Goods",
    description="""Public goods game with an added punishment option for enforcing cooperation.

Strategic considerations:
- Standard public goods: contribute (2,2), free_ride temptation (3), sucker payoff (0)
- Punishment costs you 1 point but also costs the opponent 2 points
- Punishing a free-rider: you get -1, they get +1 (reduced from +3)
- Punishing a contributor: you get +1, they get -1 (they suffer for cooperating)
- Mutual punishment (-2,-2) is the worst outcome
- The threat of punishment can deter free-riding
- But actually punishing is costly - it's a second-order dilemma""",
    actions=["contribute", "free_ride", "punish"],
    matrix={
        # No punishment scenarios
        ("contribute", "contribute"): (2, 2),
        ("contribute", "free_ride"): (0, 3),
        ("free_ride", "contribute"): (3, 0),
        ("free_ride", "free_ride"): (1, 1),
        # Punishment scenarios
        ("punish", "free_ride"): (-1, 1),
        ("contribute", "punish"): (1, -1),
        ("punish", "contribute"): (1, -1),
        ("free_ride", "punish"): (-1, 1),
        # Edge cases
        ("punish", "punish"): (-2, -2),
    },
)
