"""Battle of the Sexes game definition."""

from .matrix_factory import create_matrix_game

GAME = create_matrix_game(
    id="battle_of_the_sexes",
    name="Battle of the Sexes",
    description="""A coordination game where both players prefer being together but have different preferred activities.

Strategic considerations:
- Player 1 prefers opera (2,1), Player 2 prefers football (1,2)
- Miscoordination (0,0) is the worst outcome for both
- The key is coordinating on SOMETHING rather than disagreeing
- Two pure Nash equilibria exist: (opera,opera) and (football,football)
- Communication or focal points help resolve which equilibrium to pick""",
    actions=["opera", "football"],
    matrix={
        ("opera", "opera"): (2, 1),
        ("opera", "football"): (0, 0),
        ("football", "opera"): (0, 0),
        ("football", "football"): (1, 2),
    },
)
