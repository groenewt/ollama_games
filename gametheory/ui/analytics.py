"""Analytics panel builders for the Analytics tab."""

from typing import Dict, List, Any, Optional
import marimo as mo
import polars as pl

from ..visualization.charts import (
    create_response_time_chart,
    create_model_comparison_heatmap,
    create_leaderboard_chart,
    create_cooperation_rate_chart,
    create_cumulative_payoff_chart,
    create_action_distribution_chart,
    create_payoff_comparison_chart,
    create_avg_payoff_chart,
)
from .components import create_metrics_panel


class AnalyticsPanelBuilder:
    """Static factory methods for building analytics UI sections."""

    @staticmethod
    def build_metrics_section(cumulative: Dict[str, Any]) -> mo.Html:
        """Build the cumulative API metrics section.

        Args:
            cumulative: Dictionary of cumulative metrics from MetricsTracker.

        Returns:
            A Marimo vstack with metrics display.
        """
        return mo.vstack([
            mo.md("### Cumulative API Metrics"),
            mo.hstack([
                mo.stat(
                    label="Total API Calls",
                    value=str(cumulative.get("total_requests", 0)),
                ),
                mo.stat(
                    label="Success Rate",
                    value=f"{cumulative.get('success_rate', 100):.1f}%",
                ),
                mo.stat(
                    label="Avg Response",
                    value=f"{cumulative.get('avg_response_time', 0):.2f}s",
                ),
                mo.stat(
                    label="Min Response",
                    value=f"{cumulative.get('min_response_time', 0):.2f}s",
                ),
                mo.stat(
                    label="Max Response",
                    value=f"{cumulative.get('max_response_time', 0):.2f}s",
                ),
                mo.stat(
                    label="Failed Calls",
                    value=str(cumulative.get("failed_requests", 0)),
                ),
                mo.stat(
                    label="Sessions",
                    value=str(cumulative.get("sessions_count", 0)),
                ),
                mo.stat(
                    label="Req/Sec",
                    value=f"{cumulative.get('requests_per_second', 0):.2f}",
                ),
            ], justify="start"),
            mo.md(f"**Models Tested:** {', '.join(cumulative.get('models_tested', [])) or 'None'}"),
        ])

    @staticmethod
    def build_response_time_section(times: List[float]) -> mo.Html:
        """Build the response time chart section.

        Args:
            times: List of response times.

        Returns:
            A Marimo vstack with response time chart.
        """
        if not times:
            return mo.vstack([
                mo.md("### API Response Times"),
                mo.callout(mo.md("No response time data available yet."), kind="neutral"),
            ])

        chart = create_response_time_chart(times)
        return mo.vstack([
            mo.md("### API Response Times"),
            chart,
        ])

    @staticmethod
    def build_leaderboard_section(leaderboard_df: pl.DataFrame) -> mo.Html:
        """Build the model leaderboard section with chart and table.

        Args:
            leaderboard_df: DataFrame with model rankings.

        Returns:
            A Marimo vstack with leaderboard chart and table.
        """
        if leaderboard_df.is_empty():
            return mo.vstack([
                mo.md("### Model Leaderboard"),
                mo.callout(mo.md("No leaderboard data available yet."), kind="neutral"),
            ])

        # Add weighted_score column using avg_payoff
        leaderboard_with_score = leaderboard_df.with_columns([
            pl.col("avg_payoff").alias("weighted_score")
        ])

        chart = create_leaderboard_chart(leaderboard_with_score, metric="weighted_score")

        return mo.vstack([
            mo.md("### Model Leaderboard"),
            chart,
            mo.accordion({
                "Detailed Rankings": leaderboard_df,
            }),
        ])

    @staticmethod
    def build_heatmap_section(comparison_df: pl.DataFrame) -> mo.Html:
        """Build the model comparison heatmap section.

        Args:
            comparison_df: DataFrame with model/game aggregated stats.

        Returns:
            A Marimo vstack with heatmap chart.
        """
        if comparison_df.is_empty():
            return mo.vstack([
                mo.md("### Model Performance by Game Type"),
                mo.callout(mo.md("No comparison data available yet."), kind="neutral"),
            ])

        chart = create_model_comparison_heatmap(comparison_df, metric="avg_payoff")

        return mo.vstack([
            mo.md("### Model Performance by Game Type"),
            chart,
        ])

    @staticmethod
    def build_cooperation_section(coop_df: pl.DataFrame) -> mo.Html:
        """Build the cooperation rate chart section.

        Args:
            coop_df: DataFrame with cooperation rates.

        Returns:
            A Marimo vstack with cooperation rate chart.
        """
        if coop_df.is_empty():
            return mo.vstack([
                mo.md("### Cooperation Rates"),
                mo.callout(mo.md("No cooperation data available yet."), kind="neutral"),
            ])

        # Only show if there's meaningful cooperation data
        if coop_df["cooperation_rate"].sum() == 0:
            return mo.vstack([
                mo.md("### Cooperation Rates"),
                mo.callout(
                    mo.md("No cooperative actions detected. This may be normal for competitive games."),
                    kind="neutral",
                ),
            ])

        chart = create_cooperation_rate_chart(coop_df)

        return mo.vstack([
            mo.md("### Cooperation Rates by Model"),
            mo.md("_Cooperative actions include: cooperate, contribute, trust, share, stag_"),
            chart,
        ])

    @staticmethod
    def build_game_summary_section(summary_df: pl.DataFrame) -> mo.Html:
        """Build the game type summary section.

        Args:
            summary_df: DataFrame with per-game statistics.

        Returns:
            A Marimo vstack with game summary table.
        """
        if summary_df.is_empty():
            return mo.vstack([
                mo.md("### Game Type Summary"),
                mo.callout(mo.md("No game data available yet."), kind="neutral"),
            ])

        return mo.vstack([
            mo.md("### Game Type Summary"),
            summary_df,
        ])

    @staticmethod
    def build_sessions_section(sessions: List[Dict[str, Any]], limit: int = 10) -> mo.Html:
        """Build the recent sessions section.

        Args:
            sessions: List of session metadata dictionaries.
            limit: Maximum number of sessions to display.

        Returns:
            A Marimo vstack with sessions table.
        """
        if not sessions:
            return mo.vstack([
                mo.md("### Recent Sessions"),
                mo.callout(mo.md("No sessions recorded yet."), kind="neutral"),
            ])

        return mo.vstack([
            mo.md("### Recent Sessions"),
            mo.ui.table(sessions[:limit]),
        ])

    @staticmethod
    def build_session_detail_section(
        results_df: pl.DataFrame,
        session_id: str,
    ) -> mo.Html:
        """Build session-specific charts for a selected session.

        Args:
            results_df: DataFrame with session results.
            session_id: The session identifier.

        Returns:
            A Marimo vstack with session detail charts.
        """
        if results_df.is_empty():
            return mo.callout(mo.md(f"No data found for session {session_id}"), kind="warn")

        # Add game_number column if not present
        if "game_number" not in results_df.columns:
            results_df = results_df.with_row_index("game_number")
            results_df = results_df.with_columns([
                (pl.col("game_number") + 1).alias("game_number")
            ])

        # Create charts
        payoff_chart_p1 = create_cumulative_payoff_chart(results_df, player_num=1, color="steelblue")
        payoff_chart_p2 = create_cumulative_payoff_chart(results_df, player_num=2, color="coral")
        action_chart = create_action_distribution_chart(results_df)
        comparison_chart = create_payoff_comparison_chart(results_df)
        avg_chart = create_avg_payoff_chart(results_df)

        return mo.vstack([
            mo.md(f"### Session: {session_id}"),
            mo.md("#### Cumulative Payoffs"),
            mo.hstack([payoff_chart_p1, payoff_chart_p2]),
            mo.md("#### Action Distribution"),
            action_chart,
            mo.md("#### Payoff Analysis"),
            mo.hstack([comparison_chart, avg_chart]),
            mo.accordion({
                "Raw Results": results_df,
            }),
        ])
