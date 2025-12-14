"""Colonel Blotto game implementation using Burr engine.

Colonel Blotto is a resource allocation game where players distribute
limited resources across multiple battlefields. The player who allocates
more resources to a battlefield wins it. The player who wins the majority
of battlefields wins the game.

This game has no pure strategy Nash equilibrium, making it ideal for
testing LLM strategic reasoning vs pattern matching.

References:
- Borel, E. (1921). "La théorie du jeu les équations intégrales à noyau symétrique"
- Roberson, B. (2006). "The Colonel Blotto game"
"""

from typing import Tuple, List

from ..engine.burr_app import BurrGameDefinition, AllocationSpace


def blotto_payoff(
    allocations: Tuple[Tuple[float, ...], Tuple[float, ...]]
) -> Tuple[int, int]:
    """Calculate payoffs for 2-player Colonel Blotto.

    Each player wins a battlefield if they allocated more resources to it.
    Ties go to neither player. The payoff is the number of battlefields won.

    Args:
        allocations: Tuple of (player1_allocation, player2_allocation)
                    Each allocation is a tuple of floats for each battlefield.

    Returns:
        Tuple of (player1_wins, player2_wins)
    """
    p1_alloc, p2_alloc = allocations

    p1_wins = 0
    p2_wins = 0

    for a1, a2 in zip(p1_alloc, p2_alloc):
        if a1 > a2:
            p1_wins += 1
        elif a2 > a1:
            p2_wins += 1
        # Ties: no one wins

    return (p1_wins, p2_wins)


def blotto_payoff_weighted(
    allocations: Tuple[Tuple[float, ...], Tuple[float, ...]],
    weights: Tuple[float, ...] = None,
) -> Tuple[float, float]:
    """Calculate weighted payoffs for Colonel Blotto.

    Some battlefields may be worth more than others.

    Args:
        allocations: Tuple of allocations per player
        weights: Value of each battlefield (defaults to 1.0 each)

    Returns:
        Tuple of weighted scores
    """
    p1_alloc, p2_alloc = allocations
    n_fields = len(p1_alloc)

    if weights is None:
        weights = tuple(1.0 for _ in range(n_fields))

    p1_score = 0.0
    p2_score = 0.0

    for a1, a2, w in zip(p1_alloc, p2_alloc, weights):
        if a1 > a2:
            p1_score += w
        elif a2 > a1:
            p2_score += w
        # Ties: split the value
        else:
            p1_score += w / 2
            p2_score += w / 2

    return (p1_score, p2_score)


# --- Pre-configured Blotto variants ---

COLONEL_BLOTTO_5 = BurrGameDefinition(
    id="colonel_blotto_5",
    name="Colonel Blotto (5 Battlefields)",
    description="""You are a general in Colonel Blotto's army. You have 100 troops to allocate across 5 battlefields.
Each battlefield is won by whoever commits more troops. Win the majority of battlefields (3+) to win the war.

Strategic considerations:
- Spreading evenly (20-20-20-20-20) is predictable
- Concentrating forces wins some but loses others
- There is no single "best" strategy - it depends on your opponent

Think carefully about how your opponent might allocate, then choose your distribution.""",
    action_space=AllocationSpace(num_fields=5, budget=100.0),
    payoff_fn=blotto_payoff,
    num_players=2,
)

COLONEL_BLOTTO_3 = BurrGameDefinition(
    id="colonel_blotto_3",
    name="Colonel Blotto (3 Battlefields)",
    description="""A simpler Colonel Blotto variant with only 3 battlefields and 100 troops.
Win 2 out of 3 battlefields to win. This version allows for clearer strategic reasoning.""",
    action_space=AllocationSpace(num_fields=3, budget=100.0),
    payoff_fn=blotto_payoff,
    num_players=2,
)

COLONEL_BLOTTO_7 = BurrGameDefinition(
    id="colonel_blotto_7",
    name="Colonel Blotto (7 Battlefields)",
    description="""An extended Colonel Blotto with 7 battlefields and 100 troops.
Win 4 out of 7 battlefields to win. More battlefields means more strategic complexity.""",
    action_space=AllocationSpace(num_fields=7, budget=100.0),
    payoff_fn=blotto_payoff,
    num_players=2,
)

# Registry of Blotto games
BLOTTO_GAMES = {
    "colonel_blotto_3": COLONEL_BLOTTO_3,
    "colonel_blotto_5": COLONEL_BLOTTO_5,
    "colonel_blotto_7": COLONEL_BLOTTO_7,
}
