"""Meta-strategy learning analysis for allocation games.

Detects adaptation, learning, and counter-strategy development
during repeated game play.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
import numpy as np
from collections import defaultdict

from ..core.utils import parse_allocation


@dataclass
class AdaptationMetrics:
    """Metrics describing a player's adaptation behavior."""
    player_id: int
    model: str
    adaptation_detected: bool
    adaptation_type: str           # "reactive", "anticipatory", "none", "random"
    memory_correlation: float      # Correlation with opponent history (-1 to 1)
    improvement_trajectory: List[float]  # Win rate by round window
    improvement_rate: float        # Linear slope of improvement
    strategy_stability: float      # How consistent strategy is (0-1)
    counter_strategy_score: float  # Evidence of deliberate countering


@dataclass
class MemoryEffectResult:
    """Results of memory effect analysis."""
    player_id: int
    lag: int                       # How many rounds back we're looking
    correlation_by_field: List[float]  # Correlation per field
    overall_correlation: float
    significant: bool              # Statistically significant effect
    effect_direction: str          # "mimic", "counter", "none"


@dataclass
class MultiLagMemoryAnalysis:
    """Results of multi-lag memory effect analysis (lags 1-3)."""
    player_id: int
    model: str
    lag_results: Dict[int, "MemoryEffectResult"]  # lag -> result
    strongest_lag: int             # Which lag has highest correlation
    strongest_correlation: float   # Correlation at strongest lag
    dominant_effect: str           # Overall effect type
    memory_depth: int              # Estimated memory depth (0-3)
    combined_score: float          # Weighted combination of lag effects


@dataclass
class LearningCurve:
    """Learning curve data for a player."""
    player_id: int
    model: str
    round_windows: List[int]       # Window start rounds
    win_rates: List[float]         # Win rate per window
    concentration_trend: List[float]  # Concentration over time
    trend_direction: str           # "improving", "declining", "stable"
    learning_rate: float           # Slope of improvement


class MetaStrategyAnalyzer:
    """Analyzes meta-strategic behavior and learning during tournaments."""

    def __init__(self, results: List[Dict], num_players: int = 2):
        """Initialize with game results.

        Args:
            results: List of round result dictionaries
            num_players: Number of players in the game
        """
        self.results = results
        self.num_players = num_players
        self._allocations_cache = {}

    def _get_allocations(self, player_num: int) -> List[Optional[List[float]]]:
        """Extract allocation sequence for a player.

        Args:
            player_num: Player number (1-indexed)

        Returns:
            List of allocations (or None for failed rounds)
        """
        if player_num in self._allocations_cache:
            return self._allocations_cache[player_num]

        allocations = []

        for result in self.results:
            allocation = None

            # Try allocation field first, then action field
            alloc_key = f"player{player_num}_allocation"
            action_key = f"player{player_num}_action"

            for key in [alloc_key, action_key]:
                if key in result:
                    allocation = parse_allocation(result[key])
                    if allocation is not None:
                        break

            allocations.append(allocation)

        self._allocations_cache[player_num] = allocations
        return allocations

    def _get_payoffs(self, player_num: int) -> List[float]:
        """Extract payoff sequence for a player."""
        payoffs = []
        key = f"player{player_num}_payoff"
        for result in self.results:
            payoffs.append(result.get(key, 0))
        return payoffs

    def _get_model(self, player_num: int) -> str:
        """Get model name for a player."""
        key = f"player{player_num}_model"
        if self.results:
            return self.results[0].get(key, f"Player {player_num}")
        return f"Player {player_num}"

    def detect_adaptation(self, player_num: int) -> AdaptationMetrics:
        """Detect if player adapts strategy over time.

        Args:
            player_num: Player number (1-indexed)

        Returns:
            AdaptationMetrics describing adaptation behavior
        """
        allocations = self._get_allocations(player_num)
        valid_allocations = [a for a in allocations if a is not None]

        if len(valid_allocations) < 4:
            return AdaptationMetrics(
                player_id=player_num,
                model=self._get_model(player_num),
                adaptation_detected=False,
                adaptation_type="insufficient_data",
                memory_correlation=0.0,
                improvement_trajectory=[],
                improvement_rate=0.0,
                strategy_stability=0.0,
                counter_strategy_score=0.0,
            )

        # Calculate memory correlation
        memory_result = self.analyze_memory_effects(player_num)
        memory_correlation = memory_result.overall_correlation

        # Calculate improvement trajectory
        learning = self.calculate_learning_curve(player_num)
        improvement_trajectory = learning.win_rates
        improvement_rate = learning.learning_rate

        # Calculate strategy stability (variance of concentration over time)
        concentrations = []
        for alloc in valid_allocations:
            budget = sum(alloc)
            if budget > 0:
                proportions = [a / budget for a in alloc]
                hhi = sum(p ** 2 for p in proportions)
                concentrations.append(hhi)

        strategy_stability = 1.0 - np.std(concentrations) if concentrations else 0.0
        strategy_stability = max(0, min(1, strategy_stability))

        # Detect counter-strategies
        counter_result = self.detect_counter_strategies(player_num)
        counter_strategy_score = counter_result.get("counter_score", 0)

        # Determine adaptation type
        if abs(memory_correlation) > 0.3:
            if memory_correlation > 0:
                adaptation_type = "reactive"  # Mimics opponent
            else:
                adaptation_type = "anticipatory"  # Counters opponent
            adaptation_detected = True
        elif improvement_rate > 0.02:
            adaptation_type = "learning"
            adaptation_detected = True
        elif strategy_stability < 0.5:
            adaptation_type = "random"
            adaptation_detected = False
        else:
            adaptation_type = "none"
            adaptation_detected = False

        return AdaptationMetrics(
            player_id=player_num,
            model=self._get_model(player_num),
            adaptation_detected=adaptation_detected,
            adaptation_type=adaptation_type,
            memory_correlation=memory_correlation,
            improvement_trajectory=improvement_trajectory,
            improvement_rate=improvement_rate,
            strategy_stability=strategy_stability,
            counter_strategy_score=counter_strategy_score,
        )

    def analyze_memory_effects(
        self,
        player_num: int,
        lag: int = 1
    ) -> MemoryEffectResult:
        """Analyze correlation between player's strategy and opponent's history.

        Tests if player's allocation at time t correlates with opponent's
        allocation at time t-lag.

        Args:
            player_num: Player number to analyze (1-indexed)
            lag: Number of rounds to look back

        Returns:
            MemoryEffectResult with correlation data
        """
        # Get opponent number
        opponent_num = 2 if player_num == 1 else 1

        player_allocs = self._get_allocations(player_num)
        opponent_allocs = self._get_allocations(opponent_num)

        # Need at least lag+3 rounds for meaningful analysis
        if len(player_allocs) < lag + 3:
            return MemoryEffectResult(
                player_id=player_num,
                lag=lag,
                correlation_by_field=[],
                overall_correlation=0.0,
                significant=False,
                effect_direction="none",
            )

        # Build paired observations: (opponent[t-lag], player[t])
        player_vectors = []
        opponent_vectors = []

        for t in range(lag, len(player_allocs)):
            player_alloc = player_allocs[t]
            opponent_alloc = opponent_allocs[t - lag]

            if player_alloc is not None and opponent_alloc is not None:
                # Normalize to proportions
                p_budget = sum(player_alloc)
                o_budget = sum(opponent_alloc)

                if p_budget > 0 and o_budget > 0:
                    player_vectors.append([a / p_budget for a in player_alloc])
                    opponent_vectors.append([a / o_budget for a in opponent_alloc])

        if len(player_vectors) < 3:
            return MemoryEffectResult(
                player_id=player_num,
                lag=lag,
                correlation_by_field=[],
                overall_correlation=0.0,
                significant=False,
                effect_direction="none",
            )

        # Calculate correlation per field
        num_fields = len(player_vectors[0])
        correlations = []

        for field_idx in range(num_fields):
            player_field = [v[field_idx] for v in player_vectors]
            opponent_field = [v[field_idx] for v in opponent_vectors]

            # Pearson correlation
            if np.std(player_field) > 0 and np.std(opponent_field) > 0:
                corr = np.corrcoef(player_field, opponent_field)[0, 1]
                correlations.append(float(corr) if not np.isnan(corr) else 0.0)
            else:
                correlations.append(0.0)

        # Overall correlation (average across fields)
        overall_corr = sum(correlations) / len(correlations) if correlations else 0.0

        # Determine significance (rough heuristic: |r| > 0.3 with n > 5)
        significant = abs(overall_corr) > 0.3 and len(player_vectors) > 5

        # Effect direction
        if overall_corr > 0.3:
            effect_direction = "mimic"  # Player follows opponent
        elif overall_corr < -0.3:
            effect_direction = "counter"  # Player opposes opponent
        else:
            effect_direction = "none"

        return MemoryEffectResult(
            player_id=player_num,
            lag=lag,
            correlation_by_field=correlations,
            overall_correlation=overall_corr,
            significant=significant,
            effect_direction=effect_direction,
        )

    def analyze_multi_lag_memory(
        self,
        player_num: int,
        max_lag: int = 3,
    ) -> MultiLagMemoryAnalysis:
        """Analyze memory effects across multiple time lags.

        Tests correlations between player's strategy at time t and opponent's
        strategies at times t-1, t-2, t-3 to understand memory depth.

        Args:
            player_num: Player number to analyze (1-indexed)
            max_lag: Maximum lag to test (default 3)

        Returns:
            MultiLagMemoryAnalysis with results for all tested lags
        """
        lag_results: Dict[int, MemoryEffectResult] = {}
        correlations: List[Tuple[int, float]] = []

        for lag in range(1, max_lag + 1):
            result = self.analyze_memory_effects(player_num, lag=lag)
            lag_results[lag] = result
            correlations.append((lag, abs(result.overall_correlation)))

        # Find strongest lag
        if correlations:
            strongest_lag, strongest_abs_corr = max(correlations, key=lambda x: x[1])
            strongest_result = lag_results[strongest_lag]
            strongest_correlation = strongest_result.overall_correlation
        else:
            strongest_lag = 1
            strongest_correlation = 0.0

        # Determine dominant effect across all lags
        mimic_count = sum(1 for r in lag_results.values() if r.effect_direction == "mimic")
        counter_count = sum(1 for r in lag_results.values() if r.effect_direction == "counter")

        if mimic_count > counter_count and mimic_count > 0:
            dominant_effect = "mimic"
        elif counter_count > mimic_count and counter_count > 0:
            dominant_effect = "counter"
        else:
            dominant_effect = "none"

        # Estimate memory depth (how many lags back show significant effects)
        significant_lags = sum(1 for r in lag_results.values() if r.significant)
        memory_depth = significant_lags

        # Combined score: weighted average favoring recent lags
        # Weights: lag1=0.5, lag2=0.3, lag3=0.2
        weights = {1: 0.5, 2: 0.3, 3: 0.2}
        combined_score = sum(
            weights.get(lag, 0.1) * abs(r.overall_correlation)
            for lag, r in lag_results.items()
        )

        return MultiLagMemoryAnalysis(
            player_id=player_num,
            model=self._get_model(player_num),
            lag_results=lag_results,
            strongest_lag=strongest_lag,
            strongest_correlation=strongest_correlation,
            dominant_effect=dominant_effect,
            memory_depth=memory_depth,
            combined_score=combined_score,
        )

    def calculate_learning_curve(
        self,
        player_num: int,
        window_size: int = 5
    ) -> LearningCurve:
        """Calculate win rate trajectory over time.

        Args:
            player_num: Player number (1-indexed)
            window_size: Number of rounds per window

        Returns:
            LearningCurve with win rate trajectory
        """
        opponent_num = 2 if player_num == 1 else 1

        player_payoffs = self._get_payoffs(player_num)
        opponent_payoffs = self._get_payoffs(opponent_num)

        if len(player_payoffs) < window_size:
            return LearningCurve(
                player_id=player_num,
                model=self._get_model(player_num),
                round_windows=[],
                win_rates=[],
                concentration_trend=[],
                trend_direction="insufficient_data",
                learning_rate=0.0,
            )

        # Calculate win rate in sliding windows
        round_windows = []
        win_rates = []

        for start in range(0, len(player_payoffs) - window_size + 1, max(1, window_size // 2)):
            end = start + window_size
            window_player = player_payoffs[start:end]
            window_opponent = opponent_payoffs[start:end]

            wins = sum(1 for p, o in zip(window_player, window_opponent) if p > o)
            win_rate = wins / window_size

            round_windows.append(start + 1)  # 1-indexed
            win_rates.append(win_rate)

        # Calculate concentration trend
        allocations = self._get_allocations(player_num)
        concentration_trend = []

        for start in range(0, len(allocations) - window_size + 1, max(1, window_size // 2)):
            window_allocs = allocations[start:start + window_size]
            valid_allocs = [a for a in window_allocs if a is not None]

            if valid_allocs:
                avg_hhi = 0
                for alloc in valid_allocs:
                    budget = sum(alloc)
                    if budget > 0:
                        proportions = [a / budget for a in alloc]
                        hhi = sum(p ** 2 for p in proportions)
                        avg_hhi += hhi
                avg_hhi /= len(valid_allocs)
                concentration_trend.append(avg_hhi)

        # Calculate learning rate (slope of win rate over time)
        if len(win_rates) >= 2:
            x = np.arange(len(win_rates))
            learning_rate = np.polyfit(x, win_rates, 1)[0]
        else:
            learning_rate = 0.0

        # Determine trend direction
        if learning_rate > 0.02:
            trend_direction = "improving"
        elif learning_rate < -0.02:
            trend_direction = "declining"
        else:
            trend_direction = "stable"

        return LearningCurve(
            player_id=player_num,
            model=self._get_model(player_num),
            round_windows=round_windows,
            win_rates=win_rates,
            concentration_trend=concentration_trend,
            trend_direction=trend_direction,
            learning_rate=float(learning_rate),
        )

    def calculate_improvement_rate(
        self,
        player_num: int,
        window: int = 5
    ) -> float:
        """Calculate win rate improvement slope.

        Args:
            player_num: Player number (1-indexed)
            window: Window size for smoothing

        Returns:
            Improvement rate (positive = improving)
        """
        learning = self.calculate_learning_curve(player_num, window)
        return learning.learning_rate

    def detect_counter_strategies(self, player_num: int) -> Dict[str, Any]:
        """Detect if player develops counter-strategies.

        Looks for patterns where player inverts or shifts relative to opponent.

        Args:
            player_num: Player number (1-indexed)

        Returns:
            Dict with counter-strategy analysis
        """
        opponent_num = 2 if player_num == 1 else 1

        player_allocs = self._get_allocations(player_num)
        opponent_allocs = self._get_allocations(opponent_num)

        # Analyze if player shifts allocation opposite to opponent's strength
        shift_scores = []

        for t in range(1, len(player_allocs)):
            player_alloc = player_allocs[t]
            opponent_prev = opponent_allocs[t - 1]

            if player_alloc is None or opponent_prev is None:
                continue

            # Normalize
            p_budget = sum(player_alloc)
            o_budget = sum(opponent_prev)

            if p_budget == 0 or o_budget == 0:
                continue

            player_props = [a / p_budget for a in player_alloc]
            opponent_props = [a / o_budget for a in opponent_prev]

            # Find opponent's strongest field
            opponent_max_field = opponent_props.index(max(opponent_props))

            # Check if player reduced allocation to that field
            if t > 1 and player_allocs[t - 1] is not None:
                prev_budget = sum(player_allocs[t - 1])
                if prev_budget > 0:
                    prev_props = [a / prev_budget for a in player_allocs[t - 1]]
                    change_at_opponent_max = player_props[opponent_max_field] - prev_props[opponent_max_field]

                    # Negative change = shifted away from opponent's strength
                    if change_at_opponent_max < -0.05:
                        shift_scores.append(1)
                    elif change_at_opponent_max > 0.05:
                        shift_scores.append(-1)
                    else:
                        shift_scores.append(0)

        # Calculate counter-strategy score
        if shift_scores:
            counter_score = sum(shift_scores) / len(shift_scores)
            counter_detected = counter_score > 0.2
        else:
            counter_score = 0
            counter_detected = False

        return {
            "player_id": player_num,
            "counter_detected": counter_detected,
            "counter_score": counter_score,
            "num_shifts_analyzed": len(shift_scores),
            "explanation": (
                "Player shifts away from opponent's strongest field"
                if counter_detected else
                "No significant counter-strategy detected"
            ),
        }

    def analyze_strategy_distribution_shift(
        self,
        player_num: int
    ) -> Dict[str, Any]:
        """Compare strategy distribution between first and last thirds of game.

        Args:
            player_num: Player number (1-indexed)

        Returns:
            Dict with distribution shift analysis
        """
        allocations = self._get_allocations(player_num)
        valid_allocs = [a for a in allocations if a is not None]

        if len(valid_allocs) < 6:
            return {
                "player_id": player_num,
                "shift_detected": False,
                "explanation": "Insufficient data for distribution analysis",
            }

        third = len(valid_allocs) // 3
        first_third = valid_allocs[:third]
        last_third = valid_allocs[-third:]

        # Calculate average allocation proportions for each third
        def avg_proportions(allocs):
            num_fields = len(allocs[0])
            totals = [0.0] * num_fields

            for alloc in allocs:
                budget = sum(alloc)
                if budget > 0:
                    for i, a in enumerate(alloc):
                        totals[i] += a / budget

            return [t / len(allocs) for t in totals]

        first_avg = avg_proportions(first_third)
        last_avg = avg_proportions(last_third)

        # Calculate shift magnitude (sum of absolute differences)
        shift_magnitude = sum(abs(f - l) for f, l in zip(first_avg, last_avg))

        # Calculate which fields changed most
        field_changes = [
            (i, last_avg[i] - first_avg[i])
            for i in range(len(first_avg))
        ]
        field_changes.sort(key=lambda x: abs(x[1]), reverse=True)

        shift_detected = shift_magnitude > 0.2

        return {
            "player_id": player_num,
            "shift_detected": shift_detected,
            "shift_magnitude": shift_magnitude,
            "first_third_distribution": first_avg,
            "last_third_distribution": last_avg,
            "largest_changes": field_changes[:3],
            "explanation": (
                f"Strategy shifted by {shift_magnitude:.2%} between first and last thirds"
                if shift_detected else
                "Strategy remained relatively stable throughout"
            ),
        }

    def summarize_all_players(self) -> Dict[str, Any]:
        """Generate summary for all players.

        Returns:
            Dict with analysis for each player
        """
        summary = {
            "num_rounds": len(self.results),
            "players": {},
        }

        for player_num in range(1, self.num_players + 1):
            adaptation = self.detect_adaptation(player_num)
            memory = self.analyze_memory_effects(player_num)
            learning = self.calculate_learning_curve(player_num)
            counter = self.detect_counter_strategies(player_num)
            shift = self.analyze_strategy_distribution_shift(player_num)

            summary["players"][player_num] = {
                "model": adaptation.model,
                "adaptation": {
                    "detected": adaptation.adaptation_detected,
                    "type": adaptation.adaptation_type,
                    "memory_correlation": adaptation.memory_correlation,
                    "improvement_rate": adaptation.improvement_rate,
                    "stability": adaptation.strategy_stability,
                },
                "memory_effect": {
                    "direction": memory.effect_direction,
                    "correlation": memory.overall_correlation,
                    "significant": memory.significant,
                },
                "learning": {
                    "trend": learning.trend_direction,
                    "rate": learning.learning_rate,
                    "trajectory": learning.win_rates,
                },
                "counter_strategy": {
                    "detected": counter["counter_detected"],
                    "score": counter["counter_score"],
                },
                "distribution_shift": {
                    "detected": shift["shift_detected"],
                    "magnitude": shift.get("shift_magnitude", 0),
                },
            }

        return summary
