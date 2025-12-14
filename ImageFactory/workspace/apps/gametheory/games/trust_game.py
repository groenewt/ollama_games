"""Trust Game definition."""

from .matrix_factory import create_matrix_game

GAME = create_matrix_game(
    id="trust_game",
    name="Trust Game",
    description="""A game where players choose to trust or betray each other.

Strategic considerations:
- Mutual trust (3,3) creates the best collective outcome
- Betraying a trusting opponent yields the highest individual payoff (5)
- Being betrayed while trusting is the worst outcome (0)
- Mutual betrayal (1,1) is the safe but suboptimal equilibrium
- Building reputation for trustworthiness can encourage cooperation
- Similar to Prisoner's Dilemma but framed around social trust""",
    actions=["trust", "betray"],
    matrix={
        ("trust", "trust"): (3, 3),
        ("trust", "betray"): (0, 5),
        ("betray", "trust"): (5, 0),
        ("betray", "betray"): (1, 1),
    },
)
