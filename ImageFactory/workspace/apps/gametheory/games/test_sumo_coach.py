"""Mathematical validation tests for Sumo Coach Problem.

Test cases based on:
- Rehsmann (2023) "The Sumo Coach Problem"

These tests verify the payoff function matches documented game theory properties.
"""

import pytest

from .sumo_coach import (
    sumo_coach_payoff,
    SumoCoachSpace,
    SUMO_COACH_4,
    SUMO_COACH_ASYMMETRIC,
    SUMO_COACH_WEIGHTED,
)


class TestSumoCoachPayoff:
    """Tests for the sumo_coach_payoff function."""

    def test_symmetric_identity_all_ties(self):
        """Same teams, same assignment → all ties."""
        skills = (60, 70, 80, 90)
        result = sumo_coach_payoff(
            ((0, 1, 2, 3), (0, 1, 2, 3)),
            skills, skills
        )
        assert result == (2.0, 2.0)  # 4 ties × 0.5 = 2.0 each

    def test_reversed_vs_identity(self):
        """Reversed assignment creates symmetric wins."""
        skills = (60, 70, 80, 90)
        # (3,2,1,0) means: pos0=90, pos1=80, pos2=70, pos3=60
        # vs (0,1,2,3): pos0=60, pos1=70, pos2=80, pos3=90
        result = sumo_coach_payoff(
            ((3, 2, 1, 0), (0, 1, 2, 3)),
            skills, skills
        )
        # Pos0: 90>60 → P1
        # Pos1: 80>70 → P1
        # Pos2: 70<80 → P2
        # Pos3: 60<90 → P2
        assert result == (2.0, 2.0)

    def test_asymmetric_teams(self):
        """Test with different team skill distributions."""
        team1 = (50, 70, 80, 100)  # Has dominant 100
        team2 = (60, 65, 85, 90)   # More balanced

        # Both use identity assignment
        result = sumo_coach_payoff(
            ((0, 1, 2, 3), (0, 1, 2, 3)),
            team1, team2
        )
        # Pos0: 50<60 → P2
        # Pos1: 70>65 → P1
        # Pos2: 80<85 → P2
        # Pos3: 100>90 → P1
        assert result == (2.0, 2.0)

    def test_dominant_wrestler_strategy(self):
        """Team 1 puts dominant wrestler against weakest opponent."""
        team1 = (50, 70, 80, 100)
        team2 = (60, 65, 85, 90)

        # T1: (3,0,1,2) = [100, 50, 70, 80]
        # T2: (0,1,2,3) = [60, 65, 85, 90]
        result = sumo_coach_payoff(
            ((3, 0, 1, 2), (0, 1, 2, 3)),
            team1, team2
        )
        # Pos0: 100>60 → P1
        # Pos1: 50<65 → P2
        # Pos2: 70<85 → P2
        # Pos3: 80<90 → P2
        assert result == (1.0, 3.0)

    def test_weighted_positions(self):
        """Championship bout worth more than undercard."""
        skills = (60, 70, 80, 90)
        weights = (1.0, 1.0, 1.0, 3.0)  # Final bout worth 3x

        # Identity vs reversed
        result = sumo_coach_payoff(
            ((0, 1, 2, 3), (3, 2, 1, 0)),
            skills, skills,
            position_weights=weights
        )
        # Pos0: 60<90 → P2 gets 1.0
        # Pos1: 70<80 → P2 gets 1.0
        # Pos2: 80>70 → P1 gets 1.0
        # Pos3: 90>60 → P1 gets 3.0
        assert result == (4.0, 2.0)

    def test_constant_sum_symmetric(self):
        """With equal weights, total = num_positions."""
        skills = (60, 70, 80, 90)

        # Test several assignment pairs
        test_cases = [
            ((0, 1, 2, 3), (0, 1, 2, 3)),
            ((3, 2, 1, 0), (0, 1, 2, 3)),
            ((1, 0, 3, 2), (2, 3, 0, 1)),
        ]

        for a1, a2 in test_cases:
            p1, p2 = sumo_coach_payoff((a1, a2), skills, skills)
            assert p1 + p2 == 4.0, f"Failed for {a1} vs {a2}"

    def test_weighted_constant_sum(self):
        """With weighted positions, total = sum of weights."""
        skills = (60, 70, 80, 90)
        weights = (1.0, 2.0, 3.0, 4.0)  # Total = 10

        p1, p2 = sumo_coach_payoff(
            ((0, 1, 2, 3), (3, 2, 1, 0)),
            skills, skills,
            position_weights=weights
        )
        assert p1 + p2 == 10.0


class TestSumoCoachSpace:
    """Tests for SumoCoachSpace validation."""

    def test_valid_assignment(self):
        """Valid index permutation should pass."""
        space = SumoCoachSpace(num_positions=4, team_skills=(60, 70, 80, 90))

        valid, msg = space.validate([3, 1, 0, 2])
        assert valid is True
        assert msg == ""

    def test_wrong_length(self):
        """Wrong number of assignments should fail."""
        space = SumoCoachSpace(num_positions=4, team_skills=(60, 70, 80, 90))

        valid, msg = space.validate([0, 1, 2])  # Only 3
        assert valid is False
        assert "4" in msg

    def test_invalid_indices(self):
        """Non-permutation of indices should fail."""
        space = SumoCoachSpace(num_positions=4, team_skills=(60, 70, 80, 90))

        # Duplicate index
        valid, msg = space.validate([0, 0, 2, 3])
        assert valid is False

        # Out of range index
        valid, msg = space.validate([0, 1, 2, 5])
        assert valid is False

    def test_prompt_instructions(self):
        """Prompt should include skill info."""
        space = SumoCoachSpace(
            num_positions=4,
            team_skills=(60, 70, 80, 90),
            position_weights=(1, 1, 2, 3)
        )
        instructions = space.prompt_instructions()

        assert "60" in instructions
        assert "90" in instructions


class TestGameDefinitions:
    """Test pre-configured game definitions."""

    def test_sumo_coach_4(self):
        """Test SUMO_COACH_4 configuration."""
        game = SUMO_COACH_4

        assert game.id == "sumo_coach_4"
        assert game.num_players == 2
        assert game.action_space.num_positions == 4
        assert game.action_space.team_skills == (60, 70, 80, 90)

        # Test payoff function works
        result = game.payoff_fn(((0, 1, 2, 3), (3, 2, 1, 0)))
        assert result == (2.0, 2.0)

    def test_sumo_coach_asymmetric(self):
        """Test SUMO_COACH_ASYMMETRIC configuration."""
        game = SUMO_COACH_ASYMMETRIC

        assert game.id == "sumo_coach_asymmetric"
        assert game.action_space.team_skills == (50, 70, 80, 100)

        # Team 1 has 100-rated wrestler (index 3)
        # Putting 100 in position 0 vs Team 2's weakest
        result = game.payoff_fn(((3, 0, 1, 2), (0, 1, 2, 3)))
        # This matches our earlier test
        assert result == (1.0, 3.0)

    def test_sumo_coach_weighted(self):
        """Test SUMO_COACH_WEIGHTED configuration."""
        game = SUMO_COACH_WEIGHTED

        assert game.id == "sumo_coach_weighted"
        # All skills are equal (75)
        assert game.action_space.team_skills == (75, 75, 75, 75)
        assert game.action_space.position_weights == (1.0, 1.0, 2.0, 3.0)

        # With equal skills, any matchup is all ties
        result = game.payoff_fn(((0, 1, 2, 3), (3, 2, 1, 0)))
        # Total weight = 7, split evenly = 3.5 each
        assert result == (3.5, 3.5)


class TestStrategicProperties:
    """Test game-theoretic properties."""

    def test_skill_advantage_matters(self):
        """Higher total skill should translate to advantage."""
        strong_team = (80, 85, 90, 95)  # Total = 350
        weak_team = (60, 65, 70, 75)    # Total = 270

        # Both use identity - strong team wins all positions
        result = sumo_coach_payoff(
            ((0, 1, 2, 3), (0, 1, 2, 3)),
            strong_team, weak_team
        )
        assert result == (4.0, 0.0)  # Strong team sweeps

    def test_strategic_assignment_can_help_weaker_team(self):
        """Smart assignment can help weaker team salvage draws."""
        strong_team = (80, 85, 90, 95)
        weak_team = (60, 65, 70, 75)

        # Weak team reverses assignment to create some wins
        result = sumo_coach_payoff(
            ((0, 1, 2, 3), (3, 2, 1, 0)),  # Strong identity, weak reversed
            strong_team, weak_team
        )
        # Pos0: 80>75 → P1
        # Pos1: 85>70 → P1
        # Pos2: 90>65 → P1
        # Pos3: 95>60 → P1
        # Still loses all! The skill gap is too large
        assert result == (4.0, 0.0)

    def test_marginal_skill_differences(self):
        """When skills are close, assignment strategy matters more."""
        team1 = (70, 75, 80, 85)
        team2 = (72, 76, 79, 83)  # Very close

        # Identity vs identity
        result = sumo_coach_payoff(
            ((0, 1, 2, 3), (0, 1, 2, 3)),
            team1, team2
        )
        # Pos0: 70<72 → P2
        # Pos1: 75<76 → P2
        # Pos2: 80>79 → P1
        # Pos3: 85>83 → P1
        assert result == (2.0, 2.0)
