"""Permutation action spaces for assignment games.

This module provides action spaces for games where players assign
items (players, wrestlers) to positions via permutations.

Games using these spaces:
- Tennis Coach: Assign skill-level players to match positions
- Sumo Coach: Assign wrestler indices to bout positions
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class PermutationSpace:
    """Action space for permutation assignment games (Tennis Coach).

    Players must assign each skill level to exactly one position.
    The assignment is a permutation of skill levels.
    """

    num_positions: int
    skill_levels: Tuple[int, ...]  # e.g., (1, 2, 3, 4)

    def validate(self, assignment: List[int]) -> Tuple[bool, str]:
        """Validate that assignment is a valid permutation of skill levels."""
        if len(assignment) != self.num_positions:
            return False, f"Expected {self.num_positions} positions, got {len(assignment)}"
        if sorted(assignment) != sorted(self.skill_levels):
            return False, f"Must be permutation of {self.skill_levels}"
        return True, ""

    def default_action(self) -> List[int]:
        """Return default action (sorted skill levels - identity permutation)."""
        return list(sorted(self.skill_levels))

    def prompt_instructions(self) -> str:
        """Generate prompt instructions for this action space."""
        return f"""Assign players with skills {self.skill_levels} to {self.num_positions} positions.
Respond with JSON: {{"assignment": [pos1_skill, pos2_skill, ...], "reasoning": "your strategy"}}
Each skill level must appear exactly once."""


@dataclass
class SumoCoachSpace:
    """Action space for Sumo Coach assignment.

    Players assign wrestler indices to bout positions.
    Indices map to skill values in the team configuration.
    """

    num_positions: int
    team_skills: Tuple[float, ...]
    position_weights: Optional[Tuple[float, ...]] = None

    def validate(self, assignment: List[int]) -> Tuple[bool, str]:
        """Validate that assignment is a valid permutation of wrestler indices."""
        if len(assignment) != self.num_positions:
            return False, f"Need {self.num_positions} assignments, got {len(assignment)}"
        expected_indices = list(range(len(self.team_skills)))
        if sorted(assignment) != expected_indices:
            return (
                False,
                f"Must assign each wrestler (indices 0-{len(self.team_skills)-1}) exactly once",
            )
        return True, ""

    def default_action(self) -> List[int]:
        """Return default action (sequential indices - identity permutation)."""
        return list(range(self.num_positions))

    def prompt_instructions(self) -> str:
        """Generate prompt instructions for this action space."""
        skills_desc = ", ".join(
            f"wrestler {i}={s}" for i, s in enumerate(self.team_skills)
        )
        weights_desc = ""
        if self.position_weights:
            weights_desc = f"\nBout weights: {self.position_weights}"

        return f"""Assign your wrestlers to bout positions.
Your team: {skills_desc}
{weights_desc}
Respond with JSON: {{"assignment": [bout1_wrestler_idx, bout2_wrestler_idx, ...], "reasoning": "your strategy"}}
Each wrestler index (0 to {len(self.team_skills)-1}) must appear exactly once."""
