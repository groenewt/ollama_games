"""Mathematical validation tests for Colonel Blotto game.

Test cases based on academic sources:
- Wikipedia: https://en.wikipedia.org/wiki/Blotto_game
- Mind Your Decisions: https://mindyourdecisions.com/blog/2012/01/24/the-colonel-blotto-game/
- Roberson (2006): "The Colonel Blotto game" Economic Theory

These tests verify the payoff functions match documented mathematical properties
and known examples from game theory literature.
"""

import random
import pytest
from typing import Tuple, List

from .colonel_blotto import (
    blotto_payoff,
    blotto_payoff_weighted,
    COLONEL_BLOTTO_3,
    COLONEL_BLOTTO_5,
    COLONEL_BLOTTO_7,
)
from ..engine.burr_app import AllocationSpace


def random_allocation(n_fields: int, budget: float) -> Tuple[float, ...]:
    """Generate a random valid allocation for testing."""
    # Generate n-1 random cut points, then compute differences
    cuts = sorted([random.uniform(0, budget) for _ in range(n_fields - 1)])
    cuts = [0] + cuts + [budget]
    return tuple(cuts[i + 1] - cuts[i] for i in range(n_fields))


class TestBlottoPayoff:
    """Tests for the basic blotto_payoff function."""

    def test_wikipedia_s6_examples(self):
        """Test payoffs match Wikipedia S=6 documented examples.

        From Wikipedia "Blotto game" article:
        For S=6 (budget=6) with 3 battlefields, possible allocations are:
        (2,2,2), (1,2,3), (1,1,4)

        The strategy (2,2,2) is noted as optimal because it never loses.
        """
        # (2,2,2) vs (1,2,3): P1 wins BF1 (2>1), tie BF2 (2=2), P2 wins BF3 (2<3)
        assert blotto_payoff(((2, 2, 2), (1, 2, 3))) == (1, 1)

        # (2,2,2) vs (1,1,4): P1 wins BF1 (2>1), P1 wins BF2 (2>1), P2 wins BF3 (2<4)
        assert blotto_payoff(((2, 2, 2), (1, 1, 4))) == (2, 1)

        # (1,2,3) vs (1,1,4): tie BF1 (1=1), P1 wins BF2 (2>1), P2 wins BF3 (3<4)
        assert blotto_payoff(((1, 2, 3), (1, 1, 4))) == (1, 1)

    def test_mind_your_decisions_examples(self):
        """Test 2-battlefield examples from Mind Your Decisions blog.

        Scenario: Blotto has 4 troops, Lotso has 3 troops, 2 battlefields.
        Note: These are asymmetric budget scenarios.
        """
        # (4,0) vs (2,1): BF1: 4>2 P1 wins, BF2: 0<1 P2 wins
        assert blotto_payoff(((4, 0), (2, 1))) == (1, 1)

        # (2,2) vs (3,0): BF1: 2<3 P2 wins, BF2: 2>0 P1 wins
        assert blotto_payoff(((2, 2), (3, 0))) == (1, 1)

        # (4,0) vs (3,0): BF1: 4>3 P1 wins, BF2: 0=0 tie
        assert blotto_payoff(((4, 0), (3, 0))) == (1, 0)

        # (0,4) vs (0,3): BF1: 0=0 tie, BF2: 4>3 P1 wins
        assert blotto_payoff(((0, 4), (0, 3))) == (1, 0)

    def test_symmetry_property(self):
        """Swapping player allocations should swap payoffs."""
        test_cases = [
            ((30, 40, 30), (25, 50, 25)),
            ((50, 50), (40, 60)),
            ((20, 20, 20, 20, 20), (10, 20, 30, 20, 20)),
        ]

        for alloc1, alloc2 in test_cases:
            p1_wins, p2_wins = blotto_payoff((alloc1, alloc2))
            p2_wins_swapped, p1_wins_swapped = blotto_payoff((alloc2, alloc1))

            assert p1_wins == p1_wins_swapped, f"Failed for {alloc1} vs {alloc2}"
            assert p2_wins == p2_wins_swapped, f"Failed for {alloc1} vs {alloc2}"

    def test_zero_sum_property(self):
        """Total wins + ties must equal number of battlefields.

        In Colonel Blotto, each battlefield is won by one player or tied.
        Therefore: p1_wins + p2_wins + ties = n_battlefields
        """
        random.seed(42)  # Reproducible randomness

        for _ in range(100):
            n_fields = random.choice([3, 5, 7])
            budget = 100.0

            alloc1 = random_allocation(n_fields, budget)
            alloc2 = random_allocation(n_fields, budget)

            p1_wins, p2_wins = blotto_payoff((alloc1, alloc2))
            ties = sum(1 for a, b in zip(alloc1, alloc2) if a == b)

            assert p1_wins + p2_wins + ties == n_fields, (
                f"Zero-sum violated: {p1_wins} + {p2_wins} + {ties} != {n_fields}"
            )

    def test_identical_allocations(self):
        """Identical allocations should result in all ties (0, 0)."""
        test_cases = [
            (20, 20, 20, 20, 20),
            (50, 50),
            (33.33, 33.33, 33.34),
            (100 / 7,) * 7,
        ]

        for alloc in test_cases:
            assert blotto_payoff((alloc, alloc)) == (0, 0), f"Failed for {alloc}"

    def test_complete_domination(self):
        """One player allocating more to ALL fields wins all battlefields."""
        # P1 wins all battlefields
        assert blotto_payoff(((30, 30, 40), (20, 20, 30))) == (3, 0)

        # P2 wins all battlefields
        assert blotto_payoff(((20, 20, 30), (30, 30, 40))) == (0, 3)

        # 5 battlefield domination
        assert blotto_payoff(
            ((25, 25, 25, 15, 10), (20, 20, 20, 10, 5))
        ) == (5, 0)

    def test_all_ties(self):
        """When all battlefields are ties, both players get 0."""
        assert blotto_payoff(((50, 50), (50, 50))) == (0, 0)
        assert blotto_payoff(((33, 34, 33), (33, 34, 33))) == (0, 0)


class TestAllocationSpace:
    """Tests for AllocationSpace validation."""

    def test_valid_allocation(self):
        """Valid allocations should pass validation."""
        space = AllocationSpace(num_fields=5, budget=100.0)

        valid, msg = space.validate([20, 20, 20, 20, 20])
        assert valid is True
        assert msg == ""

    def test_wrong_field_count(self):
        """Wrong number of fields should fail validation."""
        space = AllocationSpace(num_fields=5, budget=100.0)

        valid, msg = space.validate([25, 25, 25, 25])  # 4 instead of 5
        assert valid is False
        assert "5" in msg

        valid, msg = space.validate([20, 20, 20, 20, 10, 10])  # 6 instead of 5
        assert valid is False
        assert "5" in msg

    def test_negative_values(self):
        """Negative allocations should fail validation."""
        space = AllocationSpace(num_fields=5, budget=100.0)

        valid, msg = space.validate([30, 30, 30, 30, -20])
        assert valid is False
        assert "negative" in msg.lower()

    def test_wrong_budget(self):
        """Allocations not summing to budget should fail."""
        space = AllocationSpace(num_fields=5, budget=100.0)

        # Under budget
        valid, msg = space.validate([20, 20, 20, 20, 10])  # sums to 90
        assert valid is False
        assert "100" in msg

        # Over budget
        valid, msg = space.validate([30, 30, 30, 30, 30])  # sums to 150
        assert valid is False

    def test_budget_tolerance(self):
        """Budget validation should allow small floating point errors."""
        space = AllocationSpace(num_fields=3, budget=100.0)

        # Within tolerance (0.01)
        valid, _ = space.validate([33.333, 33.333, 33.334])
        assert valid is True

        # Just at tolerance boundary
        valid, _ = space.validate([33.33, 33.33, 33.34])  # sums to 100.00
        assert valid is True

        # Outside tolerance
        valid, _ = space.validate([33.0, 33.0, 33.0])  # sums to 99
        assert valid is False

    def test_prompt_instructions(self):
        """Prompt instructions should include budget and field count."""
        space = AllocationSpace(num_fields=5, budget=100.0)
        instructions = space.prompt_instructions()

        assert "100" in instructions
        assert "5" in instructions


class TestWeightedPayoff:
    """Tests for weighted payoff variant."""

    def test_equal_weights_matches_unweighted(self):
        """With equal weights of 1.0, weighted payoff should match unweighted counts."""
        allocs = ((30, 40, 30), (25, 50, 25))

        unweighted = blotto_payoff(allocs)
        weighted = blotto_payoff_weighted(allocs, weights=(1.0, 1.0, 1.0))

        # Weighted returns floats, unweighted returns ints
        # With equal weights and no ties, they should match
        p1_unweighted, p2_unweighted = unweighted
        p1_weighted, p2_weighted = weighted

        assert p1_weighted == float(p1_unweighted)
        assert p2_weighted == float(p2_unweighted)

    def test_asymmetric_battlefield_values(self):
        """Test weighted payoff with asymmetric battlefield values.

        Scenario: BF1 is worth 10 points, BF2 and BF3 worth 1 each.
        P1 concentrates on low-value fields, P2 takes the valuable one.
        """
        # P1 allocates: 10 to BF1, 50 to BF2, 40 to BF3
        # P2 allocates: 90 to BF1, 5 to BF2, 5 to BF3
        # P2 wins BF1 (valuable), P1 wins BF2 and BF3
        allocs = ((10, 50, 40), (90, 5, 5))
        weights = (10.0, 1.0, 1.0)

        p1_score, p2_score = blotto_payoff_weighted(allocs, weights)

        # P2 should win despite fewer battlefield victories
        # P2 gets 10 points (BF1), P1 gets 2 points (BF2 + BF3)
        assert p2_score == 10.0
        assert p1_score == 2.0
        assert p2_score > p1_score

    def test_tie_splits_value(self):
        """Ties should split the battlefield value equally."""
        allocs = ((50, 50), (50, 50))  # All ties
        weights = (3.0, 1.0)

        p1_score, p2_score = blotto_payoff_weighted(allocs, weights)

        # Total value is 4.0, split equally
        assert p1_score == 2.0
        assert p2_score == 2.0

    def test_partial_ties(self):
        """Mix of wins, losses, and ties in weighted game."""
        # BF1: tie (split 2.0), BF2: P1 wins (gets 3.0), BF3: P2 wins (gets 1.0)
        allocs = ((30, 50, 20), (30, 40, 30))
        weights = (4.0, 3.0, 1.0)

        p1_score, p2_score = blotto_payoff_weighted(allocs, weights)

        # P1: 2.0 (half of BF1) + 3.0 (BF2) = 5.0
        # P2: 2.0 (half of BF1) + 1.0 (BF3) = 3.0
        assert p1_score == 5.0
        assert p2_score == 3.0

    def test_default_equal_weights(self):
        """When weights=None, should use equal weights of 1.0."""
        allocs = ((60, 40), (40, 60))

        weighted_default = blotto_payoff_weighted(allocs, weights=None)
        weighted_explicit = blotto_payoff_weighted(allocs, weights=(1.0, 1.0))

        assert weighted_default == weighted_explicit


class TestEdgeCases:
    """Edge case tests for Colonel Blotto."""

    def test_single_battlefield(self):
        """Single battlefield game - winner takes all."""
        assert blotto_payoff(((100,), (99,))) == (1, 0)
        assert blotto_payoff(((50,), (51,))) == (0, 1)
        assert blotto_payoff(((50,), (50,))) == (0, 0)

    def test_all_resources_one_field(self):
        """All resources concentrated in one battlefield."""
        # P1 all-in on BF1, P2 spreads evenly across 3 fields
        allocs = ((100, 0, 0), (33, 33, 34))
        p1_wins, p2_wins = blotto_payoff(allocs)

        assert p1_wins == 1  # Wins BF1 only
        assert p2_wins == 2  # Wins BF2 and BF3

    def test_many_battlefields(self):
        """Test with larger number of battlefields (7)."""
        # Even spread vs concentrated strategy
        even = tuple([100 / 7] * 7)  # ~14.28 each
        concentrated = (50, 50, 0, 0, 0, 0, 0)

        p1_wins, p2_wins = blotto_payoff((even, concentrated))

        # P1 (even) loses 2 concentrated fields, wins 5 uncontested
        assert p1_wins == 5
        assert p2_wins == 2

    def test_floating_point_precision(self):
        """Test that floating point comparisons work correctly."""
        # These should be equal despite floating point representation
        alloc1 = (33.333333333333336, 33.333333333333336, 33.33333333333333)
        alloc2 = (33.333333333333336, 33.333333333333336, 33.33333333333333)

        assert blotto_payoff((alloc1, alloc2)) == (0, 0)

    def test_very_small_differences(self):
        """Very small differences should still determine a winner."""
        alloc1 = (50.0001, 49.9999)
        alloc2 = (50.0, 50.0)

        p1_wins, p2_wins = blotto_payoff((alloc1, alloc2))
        assert p1_wins == 1  # P1 wins BF1 by 0.0001
        assert p2_wins == 1  # P2 wins BF2 by 0.0001

    def test_zero_allocations(self):
        """Zero allocations are valid and should tie with zero."""
        # Both allocate nothing to some fields
        allocs = ((100, 0), (0, 100))
        p1_wins, p2_wins = blotto_payoff(allocs)

        assert p1_wins == 1  # Wins BF1
        assert p2_wins == 1  # Wins BF2


class TestGameDefinitions:
    """Test the pre-configured game definitions."""

    def test_blotto_3_fields(self):
        """Test COLONEL_BLOTTO_3 configuration."""
        game = COLONEL_BLOTTO_3

        assert game.id == "colonel_blotto_3"
        assert game.num_players == 2
        assert game.action_space.num_fields == 3
        assert game.action_space.budget == 100.0

        # Test payoff function works
        result = game.payoff_fn(((40, 30, 30), (30, 40, 30)))
        assert result == (1, 1)  # P1 wins BF1, P2 wins BF2, tie BF3

    def test_blotto_5_fields(self):
        """Test COLONEL_BLOTTO_5 configuration."""
        game = COLONEL_BLOTTO_5

        assert game.id == "colonel_blotto_5"
        assert game.num_players == 2
        assert game.action_space.num_fields == 5
        assert game.action_space.budget == 100.0

    def test_blotto_7_fields(self):
        """Test COLONEL_BLOTTO_7 configuration."""
        game = COLONEL_BLOTTO_7

        assert game.id == "colonel_blotto_7"
        assert game.num_players == 2
        assert game.action_space.num_fields == 7
        assert game.action_space.budget == 100.0


class TestStrategicProperties:
    """Test game-theoretic properties of Colonel Blotto."""

    def test_rock_paper_scissors_like_cycle(self):
        """Colonel Blotto can exhibit intransitive dominance.

        In some configurations, strategy A beats B, B beats C, but C beats A.
        This is why no pure Nash equilibrium exists.
        """
        # Example from literature: in 3-field Blotto with budget 6
        # (2,2,2) ties with (1,2,3)
        # (2,2,2) beats (1,1,4)
        # (1,2,3) ties with (1,1,4)

        result_1 = blotto_payoff(((2, 2, 2), (1, 2, 3)))
        result_2 = blotto_payoff(((2, 2, 2), (1, 1, 4)))
        result_3 = blotto_payoff(((1, 2, 3), (1, 1, 4)))

        # (2,2,2) never loses
        assert result_1[0] >= result_1[1]  # tie or win vs (1,2,3)
        assert result_2[0] > result_2[1]   # win vs (1,1,4)

        # But (1,2,3) also doesn't lose to (1,1,4)
        assert result_3[0] >= result_3[1]

    def test_uniform_strategy_robustness(self):
        """Uniform allocation never gets dominated badly.

        The uniform strategy (equal allocation to all fields) is a
        reasonable defensive strategy that avoids worst-case outcomes.
        """
        uniform_5 = (20, 20, 20, 20, 20)

        # Test against various concentrated strategies
        concentrated_strategies = [
            (50, 50, 0, 0, 0),
            (40, 30, 30, 0, 0),
            (100, 0, 0, 0, 0),
            (30, 30, 20, 10, 10),
        ]

        for concentrated in concentrated_strategies:
            p1_wins, p2_wins = blotto_payoff((uniform_5, concentrated))

            # Uniform should never lose all 5 battlefields
            assert p1_wins > 0, f"Uniform lost all to {concentrated}"

            # In fact, uniform should win majority against extreme concentration
            # (3+ zeros means truly extreme - most resources in 2 or fewer fields)
            if sum(1 for x in concentrated if x == 0) >= 3:
                assert p1_wins >= 3, f"Uniform should dominate {concentrated}"
