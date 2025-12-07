"""Learning curve and trend analysis for educational insights."""

from typing import Dict, Any, List, Optional, Tuple
import polars as pl

from ..core.utils import detect_num_players


class LearningAnalyzer:
    """Analyzes payoff trends over rounds to detect learning behavior."""

    def __init__(self, min_rounds: int = 10):
        """Initialize the learning analyzer.

        Args:
            min_rounds: Minimum rounds needed for reliable trend analysis.
        """
        self.min_rounds = min_rounds

    def detect_learning(
        self,
        results_df: pl.DataFrame,
        player_num: int = 1,
    ) -> Dict[str, Any]:
        """Detect if a player shows learning behavior.

        Args:
            results_df: DataFrame with game results.
            player_num: Player number (1-indexed).

        Returns:
            Dictionary with learning analysis metrics.
        """
        payoff_col = f"player{player_num}_payoff"

        if payoff_col not in results_df.columns:
            return {"has_learning": False, "reason": "payoff_column_missing"}

        payoffs = results_df.sort("game_number")[payoff_col].to_list()

        if len(payoffs) < self.min_rounds:
            return {
                "has_learning": False,
                "reason": "insufficient_data",
                "rounds_available": len(payoffs),
                "rounds_needed": self.min_rounds,
            }

        # Split into halves
        mid = len(payoffs) // 2
        first_half = payoffs[:mid]
        second_half = payoffs[mid:]

        first_half_avg = sum(first_half) / len(first_half)
        second_half_avg = sum(second_half) / len(second_half)

        # Calculate trend slope using simple linear regression
        slope, intercept = self._calculate_trend(payoffs)

        # Calculate rolling average improvement
        window_size = max(3, len(payoffs) // 5)
        rolling_avgs = self._rolling_average(payoffs, window_size)

        # Determine if there's learning (improvement over time)
        improvement_pct = (
            ((second_half_avg - first_half_avg) / abs(first_half_avg) * 100)
            if first_half_avg != 0
            else 0
        )

        # Determine trend direction based on slope
        if slope > 0.05:
            trend_direction = "improving"
        elif slope < -0.05:
            trend_direction = "declining"
        else:
            trend_direction = "stable"

        # Consider it "learning" if second half is significantly better
        has_learning = second_half_avg > first_half_avg * 1.1 and slope > 0

        return {
            "has_learning": has_learning,
            "first_half_avg": round(first_half_avg, 2),
            "second_half_avg": round(second_half_avg, 2),
            "improvement_pct": round(improvement_pct, 1),
            "trend_slope": round(slope, 4),
            "trend_intercept": round(intercept, 2),
            "trend_direction": trend_direction,
            "rolling_averages": rolling_avgs,
            "total_rounds": len(payoffs),
            "explanation": self._generate_explanation(
                has_learning, trend_direction, improvement_pct, first_half_avg, second_half_avg
            ),
        }

    def _calculate_trend(self, payoffs: List[int]) -> Tuple[float, float]:
        """Calculate linear regression slope and intercept.

        Args:
            payoffs: List of payoff values.

        Returns:
            Tuple of (slope, intercept).
        """
        n = len(payoffs)
        if n < 2:
            return (0.0, payoffs[0] if payoffs else 0.0)

        x = list(range(n))
        y = payoffs

        # Calculate sums
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_xx = sum(xi * xi for xi in x)

        # Calculate slope and intercept
        denominator = n * sum_xx - sum_x * sum_x
        if denominator == 0:
            return (0.0, sum_y / n)

        slope = (n * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / n

        return (slope, intercept)

    def _rolling_average(self, values: List[int], window: int) -> List[float]:
        """Calculate rolling average.

        Args:
            values: List of values.
            window: Window size.

        Returns:
            List of rolling averages.
        """
        if len(values) < window:
            return [sum(values) / len(values)] if values else []

        result = []
        for i in range(len(values) - window + 1):
            avg = sum(values[i : i + window]) / window
            result.append(round(avg, 2))

        return result

    def _generate_explanation(
        self,
        has_learning: bool,
        trend_direction: str,
        improvement_pct: float,
        first_avg: float,
        second_avg: float,
    ) -> str:
        """Generate human-readable explanation of learning analysis.

        Args:
            has_learning: Whether learning was detected.
            trend_direction: "improving", "declining", or "stable".
            improvement_pct: Percentage improvement.
            first_avg: First half average payoff.
            second_avg: Second half average payoff.

        Returns:
            Explanation string.
        """
        if has_learning:
            return (
                f"Player shows learning behavior with {improvement_pct:.1f}% improvement. "
                f"Average payoff increased from {first_avg:.1f} (first half) to {second_avg:.1f} (second half)."
            )
        elif trend_direction == "improving":
            return (
                f"Slight upward trend detected ({improvement_pct:.1f}% change), but not strong enough to indicate learning."
            )
        elif trend_direction == "declining":
            return (
                f"Performance declined over time ({improvement_pct:.1f}% change). "
                f"Average dropped from {first_avg:.1f} to {second_avg:.1f}."
            )
        else:
            return (
                f"Performance remained stable throughout the session. "
                f"First half avg: {first_avg:.1f}, Second half avg: {second_avg:.1f}."
            )

    def analyze_session(self, results_df: pl.DataFrame) -> Dict[str, Any]:
        """Analyze learning for all players in a session.

        Args:
            results_df: DataFrame with session results.

        Returns:
            Dictionary with learning analysis per player.
        """
        analysis = {}

        # Detect number of players (cached)
        num_players = detect_num_players(tuple(results_df.columns))

        if num_players == 0:
            return {"error": "No player payoff columns found"}

        for p in range(1, num_players + 1):
            analysis[f"player{p}"] = self.detect_learning(results_df, player_num=p)

        # Add comparison
        if num_players >= 2:
            learners = [
                p for p in range(1, num_players + 1)
                if analysis[f"player{p}"].get("has_learning", False)
            ]
            analysis["summary"] = {
                "players_showing_learning": learners,
                "best_improver": self._find_best_improver(analysis, num_players),
            }

        return analysis

    def _find_best_improver(self, analysis: Dict, num_players: int) -> Optional[int]:
        """Find the player with the best improvement.

        Args:
            analysis: Analysis dictionary with player data.
            num_players: Total number of players.

        Returns:
            Player number of best improver, or None.
        """
        best_player = None
        best_improvement = float("-inf")

        for p in range(1, num_players + 1):
            player_data = analysis.get(f"player{p}", {})
            improvement = player_data.get("improvement_pct", 0)
            if improvement > best_improvement:
                best_improvement = improvement
                best_player = p

        return best_player if best_improvement > 0 else None
