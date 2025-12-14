"""Tennis Coach Problem implementation.

The Tennis Coach Problem (Arad, 2012) is a permutation-based allocation game
where each coach assigns 4 differently-skilled players to 4 match positions.

Unlike Colonel Blotto (continuous allocation), this is a discrete assignment
problem with n! = 24 possible pure strategies.

References:
- Arad, A. (2012). "The Tennis Coach Problem: A Game-Theoretic and Experimental Study"
  The B.E. Journal of Theoretical Economics, vol. 12, no. 1.
"""

from typing import Tuple

from ..engine.burr_app import BurrGameDefinition
from ..engine.permutation_space import PermutationSpace


def tennis_coach_payoff(
    assignments: Tuple[Tuple[int, ...], Tuple[int, ...]]
) -> Tuple[float, float]:
    """Calculate payoffs for Tennis Coach Problem.

    At each position:
    - Better player (higher skill) wins 1 point
    - Equal players each get 0.5 points

    Total: n points divided between teams (constant-sum game).

    Args:
        assignments: Tuple of (team1_assignment, team2_assignment)
                    Each assignment is a tuple of skill levels for each position.

    Returns:
        Tuple of (team1_score, team2_score)
    """
    a1, a2 = assignments
    p1_score = 0.0
    p2_score = 0.0

    for skill1, skill2 in zip(a1, a2):
        if skill1 > skill2:
            p1_score += 1.0
        elif skill2 > skill1:
            p2_score += 1.0
        else:
            p1_score += 0.5
            p2_score += 0.5

    return (p1_score, p2_score)


# --- Pre-configured Tennis Coach variants ---

TENNIS_COACH_4 = BurrGameDefinition(
    id="tennis_coach_4",
    name="Tennis Coach (4 Players)",
    description="""You are a tennis team coach assigning 4 players to 4 match positions.
Your players have skill levels 1 (weakest) through 4 (strongest).

At each position, the team with the better player wins that point.
Equal players split the point. Total: 4 points per match.

Strategic considerations:
- Matching your best (4) against their worst (1) guarantees a win there
- But your opponent may anticipate this and counter-assign
- The "shifted" strategy (2,3,4,1) beats the "identity" (1,2,3,4)
- There is no single dominant strategy - it depends on your opponent

Respond with a permutation of [1, 2, 3, 4] indicating which skill player
you assign to each position.""",
    action_space=PermutationSpace(num_positions=4, skill_levels=(1, 2, 3, 4)),
    payoff_fn=tennis_coach_payoff,
    num_players=2,
)

TENNIS_COACH_3 = BurrGameDefinition(
    id="tennis_coach_3",
    name="Tennis Coach (3 Players)",
    description="""Simplified Tennis Coach with 3 players per team.
Assign players with skills 1, 2, 3 to 3 match positions.
Win 2 out of 3 positions to win the match.""",
    action_space=PermutationSpace(num_positions=3, skill_levels=(1, 2, 3)),
    payoff_fn=tennis_coach_payoff,
    num_players=2,
)

# Registry of Tennis Coach games
TENNIS_GAMES = {
    "tennis_coach_3": TENNIS_COACH_3,
    "tennis_coach_4": TENNIS_COACH_4,
}
