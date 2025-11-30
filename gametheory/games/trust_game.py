"""Trust Game definition."""

from ..core.types import GameDefinition

GAME = GameDefinition(
    id="trust_game",
    name="Trust Game",
    description="A game where players can choose to trust or betray each other. "
                "Similar structure to Prisoner's Dilemma with trust/betray framing.",
    payoff_matrix={
        ("trust", "trust"): (3, 3),
        ("trust", "betray"): (0, 5),
        ("betray", "trust"): (5, 0),
        ("betray", "betray"): (1, 1),
    },
    actions=["trust", "betray"],
    num_players=2,
)
