"""Sumo Coach Problem implementation.

The Sumo Coach Problem (Rehsmann, 2023) extends the Tennis Coach Problem
with custom skill values, weighted positions, and potentially asymmetric teams.

This implementation provides a deterministic variant suitable for LLM testing,
where skill values are known (not stochastic as in the original formulation).

References:
- Rehsmann, D. (2023). "The Sumo Coach Problem"
  Review of Economic Design, vol. 27(3), pages 669-700.
"""

from typing import Tuple, Optional

from ..engine.burr_app import BurrGameDefinition
from ..engine.permutation_space import SumoCoachSpace


def sumo_coach_payoff(
    assignments: Tuple[Tuple[int, ...], Tuple[int, ...]],
    team1_skills: Tuple[float, ...],
    team2_skills: Tuple[float, ...],
    position_weights: Optional[Tuple[float, ...]] = None,
) -> Tuple[float, float]:
    """Calculate Sumo Coach payoffs.

    At each bout position:
    - Stronger wrestler wins (gets position weight, default 1.0)
    - Equal wrestlers split the position weight

    Args:
        assignments: ((team1_wrestler_indices), (team2_wrestler_indices))
                    Each index maps to a skill value from respective team.
        team1_skills: Tuple of skill values for team 1
        team2_skills: Tuple of skill values for team 2
        position_weights: Optional weights for each position (default: all 1.0)

    Returns:
        Tuple of (team1_score, team2_score)
    """
    a1, a2 = assignments
    n = len(a1)

    if position_weights is None:
        position_weights = tuple(1.0 for _ in range(n))

    p1_score = 0.0
    p2_score = 0.0

    for i, (idx1, idx2) in enumerate(zip(a1, a2)):
        skill1 = team1_skills[idx1]
        skill2 = team2_skills[idx2]
        weight = position_weights[i]

        if skill1 > skill2:
            p1_score += weight
        elif skill2 > skill1:
            p2_score += weight
        else:
            # Tie: split the weight
            p1_score += weight / 2
            p2_score += weight / 2

    return (p1_score, p2_score)


# --- Pre-configured Sumo Coach variants ---

# Standard 4-wrestler symmetric variant
SUMO_COACH_4 = BurrGameDefinition(
    id="sumo_coach_4",
    name="Sumo Coach (4 Wrestlers)",
    description="""You are a sumo stable master assigning 4 wrestlers to 4 bouts.
Your wrestlers have strength ratings: 60, 70, 80, 90.
At each bout, the stronger wrestler wins. Equally matched wrestlers split.

You must assign each wrestler to exactly one bout (use indices 0-3).
Win the most bouts to win the tournament.

Strategic insight: Like Tennis Coach, but with non-uniform skill gaps.
The gap between 90 and 80 is the same as 70 to 60, creating symmetry.""",
    action_space=SumoCoachSpace(
        num_positions=4,
        team_skills=(60, 70, 80, 90),
    ),
    payoff_fn=lambda a: sumo_coach_payoff(a, (60, 70, 80, 90), (60, 70, 80, 90)),
    num_players=2,
)

# Asymmetric variant: teams have different strength distributions
SUMO_COACH_ASYMMETRIC = BurrGameDefinition(
    id="sumo_coach_asymmetric",
    name="Sumo Coach (Asymmetric Teams)",
    description="""Asymmetric sumo tournament with unequal teams.

Team 1 wrestlers: 50, 70, 80, 100 (has one weak wrestler at 50, one dominant at 100)
Team 2 wrestlers: 60, 65, 85, 90 (more balanced distribution)

Key insight: Team 1's 100-rated wrestler beats everyone on Team 2.
But Team 1's 50-rated wrestler loses to everyone on Team 2.
The strategic question: where do you sacrifice the guaranteed loss?

You play as Team 1. Assign wrestlers by index (0=50, 1=70, 2=80, 3=100).""",
    action_space=SumoCoachSpace(
        num_positions=4,
        team_skills=(50, 70, 80, 100),
    ),
    payoff_fn=lambda a: sumo_coach_payoff(a, (50, 70, 80, 100), (60, 65, 85, 90)),
    num_players=2,
)

# Weighted variant: championship bout worth more
SUMO_COACH_WEIGHTED = BurrGameDefinition(
    id="sumo_coach_weighted",
    name="Sumo Coach (Weighted Bouts)",
    description="""Sumo tournament with weighted bout importance.

All wrestlers have equal skills (strength 75 each), so individual bouts are 50-50.
BUT the bouts have different values:
- Bout 1: 1 point (undercard)
- Bout 2: 1 point (undercard)
- Bout 3: 2 points (semi-main)
- Bout 4: 3 points (championship bout)

Total: 7 points available. Win 4+ to win the tournament.

With equal skills, this becomes purely a coordination/assignment game.""",
    action_space=SumoCoachSpace(
        num_positions=4,
        team_skills=(75, 75, 75, 75),
        position_weights=(1.0, 1.0, 2.0, 3.0),
    ),
    payoff_fn=lambda a: sumo_coach_payoff(
        a, (75, 75, 75, 75), (75, 75, 75, 75),
        position_weights=(1.0, 1.0, 2.0, 3.0)
    ),
    num_players=2,
)

# Registry of Sumo Coach games
SUMO_GAMES = {
    "sumo_coach_4": SUMO_COACH_4,
    "sumo_coach_asymmetric": SUMO_COACH_ASYMMETRIC,
    "sumo_coach_weighted": SUMO_COACH_WEIGHTED,
}
