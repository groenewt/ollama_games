"""Public Goods Game definition."""

from .matrix_factory import create_matrix_game

GAME = create_matrix_game(
    id="public_goods",
    name="Public Goods Game",
    description="""Players decide whether to contribute to a public good or free-ride on others' contributions.

Strategic considerations:
- Mutual contribution (2,2) creates the public good, benefiting everyone
- Free-riding on a contributor yields the best individual payoff (3)
- Being the only contributor while others free-ride is costly (0)
- Mutual free-riding (1,1) means no public good is created
- The dilemma: individual incentives to free-ride undermine collective benefit
- In larger groups, the free-rider problem becomes more severe""",
    actions=["contribute", "free_ride"],
    matrix={
        ("contribute", "contribute"): (2, 2),
        ("contribute", "free_ride"): (0, 3),
        ("free_ride", "contribute"): (3, 0),
        ("free_ride", "free_ride"): (1, 1),
    },
)
