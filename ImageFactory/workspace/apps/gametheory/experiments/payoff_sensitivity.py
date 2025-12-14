"""Payoff function sensitivity analysis.

Tests how different payoff function designs affect
strategy selection and game outcomes.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable, Tuple
import asyncio
import aiohttp
import numpy as np
from datetime import datetime
import uuid


@dataclass
class PayoffVariant:
    """A variant payoff function configuration."""
    name: str
    description: str
    payoff_fn: Callable[[Tuple[Any, ...]], Tuple[float, ...]]
    parameters: Dict[str, Any]


@dataclass
class PayoffSensitivityResult:
    """Results from testing a single payoff variant."""
    variant_name: str
    parameters: Dict[str, Any]
    num_rounds: int
    avg_payoff_p1: float
    avg_payoff_p2: float
    win_rate_p1: float
    avg_concentration_p1: float
    avg_concentration_p2: float
    dominant_strategy_p1: str
    dominant_strategy_p2: str


@dataclass
class PayoffSensitivitySummary:
    """Summary of payoff sensitivity analysis."""
    analysis_id: str
    base_game: str
    num_variants: int
    results: List[PayoffSensitivityResult]
    sensitivity_by_parameter: Dict[str, float]  # param -> sensitivity score
    most_sensitive_param: str
    recommendations: List[str]


class PayoffSensitivityAnalyzer:
    """Analyzes how payoff function changes affect outcomes."""

    def __init__(
        self,
        base_game_id: str,
        runner_factory: Callable,
        endpoint: str = "http://localhost:11434",
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ):
        """Initialize analyzer.

        Args:
            base_game_id: Base game to derive variants from
            runner_factory: Function(game_id) -> BurrGameRunner
            endpoint: Ollama endpoint
            progress_callback: Optional callback(completed, total, message)
        """
        self.base_game_id = base_game_id
        self.runner_factory = runner_factory
        self.endpoint = endpoint
        self.progress_callback = progress_callback

        # Get base game
        base_runner = runner_factory(base_game_id)
        self.base_game = base_runner.game

    def create_linear_payoff(
        self,
        num_fields: int
    ) -> Callable[[Tuple[Any, ...]], Tuple[float, ...]]:
        """Create linear payoff: fields won = payoff."""

        def payoff_fn(allocations: Tuple[Any, ...]) -> Tuple[float, ...]:
            p1_alloc, p2_alloc = allocations
            p1_wins = 0
            p2_wins = 0

            for a1, a2 in zip(p1_alloc, p2_alloc):
                if a1 > a2:
                    p1_wins += 1
                elif a2 > a1:
                    p2_wins += 1
                # Ties = no points

            return (float(p1_wins), float(p2_wins))

        return payoff_fn

    def create_threshold_payoff(
        self,
        num_fields: int,
        threshold: float = 0.1
    ) -> Callable[[Tuple[Any, ...]], Tuple[float, ...]]:
        """Create threshold payoff: must exceed by threshold % to win.

        Args:
            num_fields: Number of fields
            threshold: Minimum margin to win (0.1 = 10% more)
        """

        def payoff_fn(allocations: Tuple[Any, ...]) -> Tuple[float, ...]:
            p1_alloc, p2_alloc = allocations
            p1_wins = 0
            p2_wins = 0

            for a1, a2 in zip(p1_alloc, p2_alloc):
                if a1 > a2 * (1 + threshold):
                    p1_wins += 1
                elif a2 > a1 * (1 + threshold):
                    p2_wins += 1
                # Close contests = no points

            return (float(p1_wins), float(p2_wins))

        return payoff_fn

    def create_diminishing_payoff(
        self,
        num_fields: int,
        decay: float = 0.5
    ) -> Callable[[Tuple[Any, ...]], Tuple[float, ...]]:
        """Create diminishing returns payoff: log-scaled field values.

        Args:
            num_fields: Number of fields
            decay: Decay factor for diminishing returns
        """
        import math

        def payoff_fn(allocations: Tuple[Any, ...]) -> Tuple[float, ...]:
            p1_alloc, p2_alloc = allocations
            p1_score = 0.0
            p2_score = 0.0

            for i, (a1, a2) in enumerate(zip(p1_alloc, p2_alloc)):
                # Field value diminishes with allocation amount
                if a1 > a2:
                    value = 1.0 + decay * math.log1p(a1 - a2)
                    p1_score += min(value, 2.0)  # Cap at 2x
                elif a2 > a1:
                    value = 1.0 + decay * math.log1p(a2 - a1)
                    p2_score += min(value, 2.0)

            return (p1_score, p2_score)

        return payoff_fn

    def create_weighted_payoff(
        self,
        weights: List[float]
    ) -> Callable[[Tuple[Any, ...]], Tuple[float, ...]]:
        """Create weighted field payoff: different fields worth different points.

        Args:
            weights: Weight per field (e.g., [1, 1, 2, 3] for 4 fields)
        """

        def payoff_fn(allocations: Tuple[Any, ...]) -> Tuple[float, ...]:
            p1_alloc, p2_alloc = allocations
            p1_score = 0.0
            p2_score = 0.0

            for i, (a1, a2) in enumerate(zip(p1_alloc, p2_alloc)):
                weight = weights[i] if i < len(weights) else 1.0
                if a1 > a2:
                    p1_score += weight
                elif a2 > a1:
                    p2_score += weight
                else:
                    # Ties split the weight
                    p1_score += weight / 2
                    p2_score += weight / 2

            return (p1_score, p2_score)

        return payoff_fn

    def create_winner_takes_all_payoff(
        self,
        num_fields: int
    ) -> Callable[[Tuple[Any, ...]], Tuple[float, ...]]:
        """Create winner-takes-all: only majority winner gets points."""

        def payoff_fn(allocations: Tuple[Any, ...]) -> Tuple[float, ...]:
            p1_alloc, p2_alloc = allocations
            p1_fields = 0
            p2_fields = 0

            for a1, a2 in zip(p1_alloc, p2_alloc):
                if a1 > a2:
                    p1_fields += 1
                elif a2 > a1:
                    p2_fields += 1

            if p1_fields > p2_fields:
                return (float(num_fields), 0.0)
            elif p2_fields > p1_fields:
                return (0.0, float(num_fields))
            else:
                return (0.0, 0.0)  # Draw

        return payoff_fn

    def create_synergy_payoff(
        self,
        num_fields: int,
        synergy_bonus: float = 0.5
    ) -> Callable[[Tuple[Any, ...]], Tuple[float, ...]]:
        """Create synergy payoff: adjacent field wins give bonus.

        Args:
            num_fields: Number of fields
            synergy_bonus: Bonus per adjacent pair won
        """

        def payoff_fn(allocations: Tuple[Any, ...]) -> Tuple[float, ...]:
            p1_alloc, p2_alloc = allocations
            p1_score = 0.0
            p2_score = 0.0
            p1_won = []
            p2_won = []

            for i, (a1, a2) in enumerate(zip(p1_alloc, p2_alloc)):
                if a1 > a2:
                    p1_score += 1
                    p1_won.append(i)
                elif a2 > a1:
                    p2_score += 1
                    p2_won.append(i)

            # Add synergy bonuses for adjacent wins
            for i in range(len(p1_won) - 1):
                if p1_won[i+1] - p1_won[i] == 1:  # Adjacent
                    p1_score += synergy_bonus

            for i in range(len(p2_won) - 1):
                if p2_won[i+1] - p2_won[i] == 1:
                    p2_score += synergy_bonus

            return (p1_score, p2_score)

        return payoff_fn

    def create_variants(
        self,
        variant_type: str = "all"
    ) -> List[PayoffVariant]:
        """Create payoff variants for testing.

        Args:
            variant_type: "all", "threshold", "weighted", "diminishing", etc.

        Returns:
            List of PayoffVariant configurations
        """
        num_fields = self.base_game.action_space.num_fields
        variants = []

        if variant_type in ["all", "linear"]:
            variants.append(PayoffVariant(
                name="linear",
                description="Standard: 1 point per field won",
                payoff_fn=self.create_linear_payoff(num_fields),
                parameters={"type": "linear"},
            ))

        if variant_type in ["all", "threshold"]:
            for threshold in [0.1, 0.2, 0.3]:
                variants.append(PayoffVariant(
                    name=f"threshold_{int(threshold*100)}pct",
                    description=f"Must exceed by {int(threshold*100)}% to win field",
                    payoff_fn=self.create_threshold_payoff(num_fields, threshold),
                    parameters={"type": "threshold", "threshold": threshold},
                ))

        if variant_type in ["all", "diminishing"]:
            for decay in [0.3, 0.5, 0.7]:
                variants.append(PayoffVariant(
                    name=f"diminishing_{int(decay*10)}",
                    description=f"Log-scaled returns (decay={decay})",
                    payoff_fn=self.create_diminishing_payoff(num_fields, decay),
                    parameters={"type": "diminishing", "decay": decay},
                ))

        if variant_type in ["all", "weighted"]:
            # Create different weight configurations
            if num_fields == 5:
                weight_configs = [
                    [1, 1, 2, 1, 1],  # Center weighted
                    [2, 1, 1, 1, 2],  # Edge weighted
                    [1, 1, 1, 2, 3],  # Ascending
                ]
            elif num_fields == 3:
                weight_configs = [
                    [1, 2, 1],  # Center weighted
                    [2, 1, 2],  # Edge weighted
                ]
            else:
                weight_configs = [[1] * num_fields]

            for weights in weight_configs:
                name = f"weighted_{'_'.join(str(w) for w in weights)}"
                variants.append(PayoffVariant(
                    name=name,
                    description=f"Field weights: {weights}",
                    payoff_fn=self.create_weighted_payoff(weights),
                    parameters={"type": "weighted", "weights": weights},
                ))

        if variant_type in ["all", "winner_takes_all"]:
            variants.append(PayoffVariant(
                name="winner_takes_all",
                description="Only majority winner gets points",
                payoff_fn=self.create_winner_takes_all_payoff(num_fields),
                parameters={"type": "winner_takes_all"},
            ))

        if variant_type in ["all", "synergy"]:
            for bonus in [0.3, 0.5, 1.0]:
                variants.append(PayoffVariant(
                    name=f"synergy_{int(bonus*10)}",
                    description=f"Adjacent wins give +{bonus} bonus",
                    payoff_fn=self.create_synergy_payoff(num_fields, bonus),
                    parameters={"type": "synergy", "bonus": bonus},
                ))

        return variants

    def _classify_allocation(self, allocation: List[float]) -> str:
        """Classify allocation strategy type."""
        budget = sum(allocation)
        if budget == 0:
            return "unknown"

        proportions = [a / budget for a in allocation]
        hhi = sum(p ** 2 for p in proportions)
        num_fields = len(allocation)
        uniform_hhi = 1.0 / num_fields

        if hhi >= 0.5:
            return "concentrated"
        elif hhi <= uniform_hhi * 1.2:
            return "uniform"
        elif hhi <= uniform_hhi * 2.0:
            return "hedged"
        else:
            return "asymmetric"

    async def test_variant(
        self,
        session: aiohttp.ClientSession,
        variant: PayoffVariant,
        model1: str,
        model2: str,
        num_rounds: int = 10
    ) -> PayoffSensitivityResult:
        """Test a single payoff variant.

        Args:
            session: aiohttp session
            variant: Payoff variant to test
            model1: First model
            model2: Second model
            num_rounds: Rounds to play

        Returns:
            PayoffSensitivityResult with test results
        """
        from ..core.types import PlayerConfig
        from ..engine.burr_app import BurrGameDefinition

        # Create modified game with variant payoff
        modified_game = BurrGameDefinition(
            id=f"{self.base_game.id}_{variant.name}",
            name=f"{self.base_game.name} ({variant.name})",
            description=f"{self.base_game.description}\n\nPayoff: {variant.description}",
            action_space=self.base_game.action_space,
            payoff_fn=variant.payoff_fn,
            num_players=self.base_game.num_players,
        )

        # Create temporary runner
        from ..engine.burr_app import BurrGameRunner
        runner = BurrGameRunner(modified_game, enable_tracking=False)

        player1 = PlayerConfig(
            player_id=1,
            model=model1,
            endpoint=self.endpoint,
        )
        player2 = PlayerConfig(
            player_id=2,
            model=model2,
            endpoint=self.endpoint,
        )
        players = [player1, player2]

        # Run rounds
        history = []
        history_payoffs = []
        cumulative_payoffs = (0.0, 0.0)

        p1_payoffs = []
        p2_payoffs = []
        p1_wins = 0
        p1_allocations = []
        p2_allocations = []

        for round_num in range(num_rounds):
            try:
                result = await runner.play_round(
                    session,
                    players,
                    history,
                    history_payoffs=history_payoffs,
                    cumulative_payoffs=cumulative_payoffs,
                    is_repeated=round_num > 0,
                )

                actions, payoffs, response_times, prompts, raw_responses, was_parsed, was_normalized = result

                p1_payoffs.append(payoffs[0])
                p2_payoffs.append(payoffs[1])
                p1_allocations.append(list(actions[0]))
                p2_allocations.append(list(actions[1]))

                if payoffs[0] > payoffs[1]:
                    p1_wins += 1

                # Update history
                history.append(tuple(tuple(a) for a in actions))
                history_payoffs.append(payoffs)
                cumulative_payoffs = (
                    cumulative_payoffs[0] + payoffs[0],
                    cumulative_payoffs[1] + payoffs[1],
                )

            except Exception:
                pass

        # Calculate metrics
        n = len(p1_payoffs)
        if n == 0:
            return PayoffSensitivityResult(
                variant_name=variant.name,
                parameters=variant.parameters,
                num_rounds=0,
                avg_payoff_p1=0,
                avg_payoff_p2=0,
                win_rate_p1=0,
                avg_concentration_p1=0,
                avg_concentration_p2=0,
                dominant_strategy_p1="unknown",
                dominant_strategy_p2="unknown",
            )

        # Calculate concentrations
        p1_concentrations = []
        p2_concentrations = []
        p1_strategies = []
        p2_strategies = []

        for alloc in p1_allocations:
            budget = sum(alloc)
            if budget > 0:
                props = [a / budget for a in alloc]
                p1_concentrations.append(sum(p ** 2 for p in props))
                p1_strategies.append(self._classify_allocation(alloc))

        for alloc in p2_allocations:
            budget = sum(alloc)
            if budget > 0:
                props = [a / budget for a in alloc]
                p2_concentrations.append(sum(p ** 2 for p in props))
                p2_strategies.append(self._classify_allocation(alloc))

        from collections import Counter
        dom_p1 = Counter(p1_strategies).most_common(1)[0][0] if p1_strategies else "unknown"
        dom_p2 = Counter(p2_strategies).most_common(1)[0][0] if p2_strategies else "unknown"

        return PayoffSensitivityResult(
            variant_name=variant.name,
            parameters=variant.parameters,
            num_rounds=n,
            avg_payoff_p1=sum(p1_payoffs) / n,
            avg_payoff_p2=sum(p2_payoffs) / n,
            win_rate_p1=p1_wins / n,
            avg_concentration_p1=sum(p1_concentrations) / len(p1_concentrations) if p1_concentrations else 0,
            avg_concentration_p2=sum(p2_concentrations) / len(p2_concentrations) if p2_concentrations else 0,
            dominant_strategy_p1=dom_p1,
            dominant_strategy_p2=dom_p2,
        )

    async def sweep_payoff_sensitivity(
        self,
        model1: str,
        model2: str,
        variants: Optional[List[PayoffVariant]] = None,
        rounds_per_variant: int = 10,
        timeout_per_variant: float = 120.0
    ) -> PayoffSensitivitySummary:
        """Test multiple payoff variants.

        Args:
            model1: First model
            model2: Second model
            variants: List of variants to test (defaults to all)
            rounds_per_variant: Rounds per variant
            timeout_per_variant: Timeout per variant

        Returns:
            PayoffSensitivitySummary with all results
        """
        if variants is None:
            variants = self.create_variants("all")

        analysis_id = str(uuid.uuid4())[:12]
        results = []

        connector = aiohttp.TCPConnector(limit=20, keepalive_timeout=30)
        async with aiohttp.ClientSession(connector=connector) as session:
            for i, variant in enumerate(variants):
                if self.progress_callback:
                    self.progress_callback(
                        i, len(variants),
                        f"Testing {variant.name}"
                    )

                try:
                    result = await asyncio.wait_for(
                        self.test_variant(session, variant, model1, model2, rounds_per_variant),
                        timeout=timeout_per_variant,
                    )
                    results.append(result)
                except asyncio.TimeoutError:
                    pass
                except Exception:
                    pass

        # Calculate sensitivity by parameter type
        sensitivity_by_param = self._calculate_parameter_sensitivity(results)

        # Find most sensitive parameter
        if sensitivity_by_param:
            most_sensitive = max(sensitivity_by_param.items(), key=lambda x: x[1])[0]
        else:
            most_sensitive = "none"

        # Generate recommendations
        recommendations = self._generate_recommendations(results, sensitivity_by_param)

        if self.progress_callback:
            self.progress_callback(len(variants), len(variants), "Analysis complete")

        return PayoffSensitivitySummary(
            analysis_id=analysis_id,
            base_game=self.base_game_id,
            num_variants=len(variants),
            results=results,
            sensitivity_by_parameter=sensitivity_by_param,
            most_sensitive_param=most_sensitive,
            recommendations=recommendations,
        )

    def _calculate_parameter_sensitivity(
        self,
        results: List[PayoffSensitivityResult]
    ) -> Dict[str, float]:
        """Calculate sensitivity score for each parameter type."""
        # Group results by parameter type
        by_type = {}
        for r in results:
            ptype = r.parameters.get("type", "unknown")
            if ptype not in by_type:
                by_type[ptype] = []
            by_type[ptype].append(r)

        sensitivity = {}
        for ptype, type_results in by_type.items():
            if len(type_results) < 2:
                continue

            # Calculate variance in outcomes for this type
            win_rates = [r.win_rate_p1 for r in type_results]
            concentrations = [r.avg_concentration_p1 for r in type_results]

            # Sensitivity = variance in outcomes
            wr_var = np.var(win_rates) if len(win_rates) > 1 else 0
            conc_var = np.var(concentrations) if len(concentrations) > 1 else 0

            sensitivity[ptype] = float(wr_var + conc_var)

        # Normalize
        max_sens = max(sensitivity.values()) if sensitivity else 1
        if max_sens > 0:
            sensitivity = {k: v / max_sens for k, v in sensitivity.items()}

        return sensitivity

    def _generate_recommendations(
        self,
        results: List[PayoffSensitivityResult],
        sensitivity: Dict[str, float]
    ) -> List[str]:
        """Generate recommendations based on results."""
        recommendations = []

        if not results:
            return ["Insufficient data for recommendations"]

        # Find best variant for each metric
        best_balance = min(results, key=lambda r: abs(r.win_rate_p1 - 0.5))
        recommendations.append(
            f"Most balanced: {best_balance.variant_name} ({best_balance.win_rate_p1:.0%} P1 win rate)"
        )

        # Strategy diversity
        strategies_seen = set()
        for r in results:
            strategies_seen.add(r.dominant_strategy_p1)
            strategies_seen.add(r.dominant_strategy_p2)

        if len(strategies_seen) > 2:
            recommendations.append(
                f"Payoff design affects strategy: {len(strategies_seen)} different archetypes observed"
            )

        # Sensitivity insights
        if sensitivity:
            most_sensitive = max(sensitivity.items(), key=lambda x: x[1])
            if most_sensitive[1] > 0.5:
                recommendations.append(
                    f"High sensitivity to {most_sensitive[0]} payoff design"
                )

        return recommendations

    def results_to_dict(self, summary: PayoffSensitivitySummary) -> Dict[str, Any]:
        """Convert summary to serializable dict."""
        return {
            "analysis_id": summary.analysis_id,
            "base_game": summary.base_game,
            "num_variants": summary.num_variants,
            "most_sensitive_param": summary.most_sensitive_param,
            "sensitivity_by_parameter": summary.sensitivity_by_parameter,
            "recommendations": summary.recommendations,
            "results": [
                {
                    "variant": r.variant_name,
                    "parameters": r.parameters,
                    "num_rounds": r.num_rounds,
                    "avg_payoff_p1": r.avg_payoff_p1,
                    "win_rate_p1": r.win_rate_p1,
                    "concentration_p1": r.avg_concentration_p1,
                    "strategy_p1": r.dominant_strategy_p1,
                }
                for r in summary.results
            ],
        }
