"""Battle of the Sexes game definition."""

from ..core.types import GameDefinition

GAME = GameDefinition(
    id="battle_of_the_sexes",
    name="Battle of the Sexes",
    description="Coordinating game where players prefer different outcomes but "
                "both prefer coordination to miscoordination.",
    payoff_matrix={
        ("opera", "opera"): (2, 1),
        ("opera", "football"): (0, 0),
        ("football", "opera"): (0, 0),
        ("football", "football"): (1, 2),
    },
    actions=["opera", "football"],
    num_players=2,
)
