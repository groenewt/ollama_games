"""Analytics panel builders for the Analytics tab."""

from typing import Dict, List, Any, Optional
import marimo as mo
import polars as pl

from ..core.utils import detect_num_players
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
from ..analytics.strategy import StrategyDetector, StrategyType, STRATEGY_DESCRIPTIONS
from ..analytics.learning import LearningAnalyzer
from ..analytics.equilibrium import EquilibriumAnalyzer
from ..core.types import GameDefinition


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
    def _detect_num_players(df: pl.DataFrame) -> int:
        """Detect number of players from DataFrame columns (cached)."""
        return detect_num_players(tuple(df.columns))

    @staticmethod
    def reconstruct_game_from_config(session_config: Dict[str, Any], game_type: str) -> Optional[GameDefinition]:
        """Reconstruct a GameDefinition from stored session config.

        Args:
            session_config: The session's config dictionary containing payoff_matrix etc.
            game_type: The game type ID.

        Returns:
            GameDefinition with the stored payoffs, or None if config is missing.
        """
        if not session_config:
            return None

        config = session_config.get("config", {})
        payoff_matrix_raw = config.get("payoff_matrix")

        if not payoff_matrix_raw:
            return None

        # Deserialize the payoff matrix (string keys -> tuple keys)
        payoff_matrix = {}
        for key_str, payoffs in payoff_matrix_raw.items():
            actions = tuple(key_str.split("_"))
            payoff_matrix[actions] = tuple(payoffs)

        return GameDefinition(
            id=game_type,
            name=config.get("game_name", game_type),
            description="Reconstructed from session data",
            payoff_matrix=payoff_matrix,
            actions=config.get("game_actions", []),
            num_players=config.get("num_players", 2),
        )

    @staticmethod
    def build_payoff_matrix_section(session_config: Dict[str, Any]) -> mo.Html:
        """Build the payoff matrix display from session config.

        Args:
            session_config: The session's config dictionary.

        Returns:
            A Marimo vstack with payoff matrix display.
        """
        if not session_config:
            return mo.callout(mo.md("No session configuration available."), kind="neutral")

        config = session_config.get("config", {})
        payoff_matrix_raw = config.get("payoff_matrix")

        if not payoff_matrix_raw:
            return mo.callout(mo.md("No payoff matrix stored for this session."), kind="neutral")

        num_players = config.get("num_players", 2)
        uses_custom = config.get("uses_custom_payoffs", False)
        game_name = config.get("game_name", "Unknown")

        # Build rows for display
        rows = []
        for key_str, payoffs in payoff_matrix_raw.items():
            actions = key_str.split("_")
            row = {}
            for i, action in enumerate(actions):
                row[f"P{i+1} Action"] = action
            for i, payoff in enumerate(payoffs):
                row[f"P{i+1} Payoff"] = payoff
            rows.append(row)

        matrix_df = pl.DataFrame(rows)

        elements = [mo.md(f"#### Payoff Matrix ({game_name})")]

        if uses_custom:
            elements.append(
                mo.callout(
                    mo.md("**Custom Payoffs** - This session used modified payoff values"),
                    kind="warn",
                )
            )

        elements.append(matrix_df)

        return mo.vstack(elements)

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

        # Detect number of players
        num_players = AnalyticsPanelBuilder._detect_num_players(results_df)

        # Add game_number column if not present
        if "game_number" not in results_df.columns:
            results_df = results_df.with_row_index("game_number")
            results_df = results_df.with_columns([
                (pl.col("game_number") + 1).alias("game_number")
            ])

        # Create cumulative payoff charts for all players
        colors = ["steelblue", "coral", "seagreen", "purple", "orange", "brown"]
        payoff_charts = []
        for p in range(1, num_players + 1):
            color = colors[(p - 1) % len(colors)]
            chart = create_cumulative_payoff_chart(results_df, player_num=p, color=color)
            payoff_charts.append(chart)

        # Action distribution and payoff charts with num_players parameter
        action_chart = create_action_distribution_chart(results_df, num_players=num_players)
        comparison_chart = create_payoff_comparison_chart(results_df, num_players=num_players)
        avg_chart = create_avg_payoff_chart(results_df, num_players=num_players)

        return mo.vstack([
            mo.md(f"### Session: {session_id}"),
            mo.md(f"#### Cumulative Payoffs ({num_players} players)"),
            mo.hstack(payoff_charts),
            mo.md("#### Action Distribution"),
            action_chart,
            mo.md("#### Payoff Analysis"),
            mo.hstack([comparison_chart, avg_chart]),
            mo.accordion({
                "Raw Results": results_df,
            }),
        ])

    @staticmethod
    def build_strategy_section(
        results_df: pl.DataFrame,
        player_models: Optional[Dict[int, str]] = None,
    ) -> mo.Html:
        """Build the strategy detection section with educational insights.

        Args:
            results_df: DataFrame with game results.
            player_models: Optional mapping of player number to model name.

        Returns:
            A Marimo vstack with strategy analysis.
        """
        if results_df.is_empty():
            return mo.vstack([
                mo.md("### Strategy Analysis"),
                mo.callout(mo.md("No data available for strategy analysis."), kind="neutral"),
            ])

        detector = StrategyDetector()
        analysis = detector.analyze_session(results_df)

        if "error" in analysis:
            return mo.vstack([
                mo.md("### Strategy Analysis"),
                mo.callout(mo.md(analysis["error"]), kind="warn"),
            ])

        elements = [mo.md("### Strategy Analysis")]
        elements.append(mo.md("_Detected strategies based on action patterns during the session._"))

        # Build strategy cards for each player
        strategy_cards = []
        for key, data in analysis.items():
            if not key.startswith("player"):
                continue

            player_num = int(key.replace("player", ""))
            model_name = player_models.get(player_num, f"Player {player_num}") if player_models else f"Player {player_num}"

            strategy = data.get("strategy", "unknown")
            confidence = data.get("confidence", 0)
            explanation = data.get("explanation", "")
            description = data.get("description", "")

            # Color-code by strategy type
            if strategy in ["always_cooperate", "tit_for_tat"]:
                kind = "success"
            elif strategy in ["always_defect", "grim_trigger"]:
                kind = "danger"
            elif strategy == "random":
                kind = "warn"
            else:
                kind = "neutral"

            card_content = mo.vstack([
                mo.md(f"**{model_name}**"),
                mo.md(f"Strategy: **{strategy.replace('_', ' ').title()}**"),
                mo.md(f"Confidence: {confidence*100:.0f}%"),
                mo.md(f"_{explanation}_"),
                mo.md(f"<small>{description}</small>"),
            ])

            strategy_cards.append(mo.callout(card_content, kind=kind))

        elements.append(mo.hstack(strategy_cards, justify="start"))

        # Add strategy legend
        legend_items = []
        for strategy_type, desc in STRATEGY_DESCRIPTIONS.items():
            legend_items.append(f"- **{strategy_type.value.replace('_', ' ').title()}**: {desc}")

        elements.append(mo.accordion({
            "Strategy Definitions": mo.md("\n".join(legend_items)),
        }))

        return mo.vstack(elements)

    @staticmethod
    def build_learning_curve_section(
        results_df: pl.DataFrame,
        player_models: Optional[Dict[int, str]] = None,
    ) -> mo.Html:
        """Build the learning curve analysis section.

        Args:
            results_df: DataFrame with game results.
            player_models: Optional mapping of player number to model name.

        Returns:
            A Marimo vstack with learning curve analysis.
        """
        if results_df.is_empty():
            return mo.vstack([
                mo.md("### Learning Curve Analysis"),
                mo.callout(mo.md("No data available for learning analysis."), kind="neutral"),
            ])

        analyzer = LearningAnalyzer()
        analysis = analyzer.analyze_session(results_df)

        if "error" in analysis:
            return mo.vstack([
                mo.md("### Learning Curve Analysis"),
                mo.callout(mo.md(analysis["error"]), kind="warn"),
            ])

        elements = [mo.md("### Learning Curve Analysis")]
        elements.append(mo.md("_Analyzing payoff trends to detect learning behavior over rounds._"))

        # Build learning cards for each player
        learning_cards = []
        for key, data in analysis.items():
            if not key.startswith("player"):
                continue

            player_num = int(key.replace("player", ""))
            model_name = player_models.get(player_num, f"Player {player_num}") if player_models else f"Player {player_num}"

            has_learning = data.get("has_learning", False)
            trend_direction = data.get("trend_direction", "stable")
            improvement_pct = data.get("improvement_pct", 0)
            first_half_avg = data.get("first_half_avg", 0)
            second_half_avg = data.get("second_half_avg", 0)
            explanation = data.get("explanation", "")

            # Color-code by trend
            if has_learning:
                kind = "success"
                icon = "📈"
            elif trend_direction == "improving":
                kind = "info"
                icon = "↗️"
            elif trend_direction == "declining":
                kind = "danger"
                icon = "📉"
            else:
                kind = "neutral"
                icon = "➡️"

            card_content = mo.vstack([
                mo.md(f"**{model_name}** {icon}"),
                mo.md(f"Trend: **{trend_direction.title()}**"),
                mo.md(f"Change: {improvement_pct:+.1f}%"),
                mo.md(f"First half avg: {first_half_avg:.1f}"),
                mo.md(f"Second half avg: {second_half_avg:.1f}"),
                mo.md(f"_{explanation}_"),
            ])

            learning_cards.append(mo.callout(card_content, kind=kind))

        elements.append(mo.hstack(learning_cards, justify="start"))

        # Summary if available
        if "summary" in analysis:
            summary = analysis["summary"]
            learners = summary.get("players_showing_learning", [])
            best = summary.get("best_improver")

            summary_text = []
            if learners:
                learner_names = [player_models.get(p, f"Player {p}") if player_models else f"Player {p}" for p in learners]
                summary_text.append(f"**Players showing learning**: {', '.join(learner_names)}")
            else:
                summary_text.append("No players showed significant learning behavior.")

            if best:
                best_name = player_models.get(best, f"Player {best}") if player_models else f"Player {best}"
                summary_text.append(f"**Best improver**: {best_name}")

            elements.append(mo.callout(mo.md("\n\n".join(summary_text)), kind="info"))

        return mo.vstack(elements)

    @staticmethod
    def build_equilibrium_section(
        game: Any,
        results_df: pl.DataFrame,
    ) -> mo.Html:
        """Build the equilibrium analysis section with educational insights.

        Args:
            game: The GameDefinition object.
            results_df: DataFrame with game results.

        Returns:
            A Marimo vstack with equilibrium analysis.
        """
        if game is None:
            return mo.vstack([
                mo.md("### Equilibrium Analysis"),
                mo.callout(mo.md("No game selected for equilibrium analysis."), kind="neutral"),
            ])

        analyzer = EquilibriumAnalyzer()

        # Get game summary (theoretical properties)
        game_summary = analyzer.get_game_summary(game)

        elements = [mo.md("### Equilibrium Analysis")]
        elements.append(mo.md(f"_Game-theoretic analysis of **{game.name}**_"))

        # Game classification
        game_type = game_summary.get("game_type", "unknown")
        game_type_display = game_type.replace("_", " ").title()

        if game_type == "social_dilemma":
            type_kind = "warn"
            type_desc = "Individual rationality leads to collectively suboptimal outcomes."
        elif game_type == "coordination":
            type_kind = "success"
            type_desc = "Players benefit from coordinating on the same action."
        elif game_type == "multiple_equilibria":
            type_kind = "info"
            type_desc = "Multiple stable outcomes exist - coordination is key."
        else:
            type_kind = "neutral"
            type_desc = ""

        elements.append(mo.callout(
            mo.md(f"**Game Type**: {game_type_display}\n\n{type_desc}"),
            kind=type_kind,
        ))

        # Theoretical equilibria
        theory_items = []

        nash = game_summary.get("nash_equilibria", [])
        if nash:
            nash_str = ", ".join([str(e) for e in nash])
            theory_items.append(f"**Nash Equilibria**: {nash_str}")
        else:
            theory_items.append("**Nash Equilibria**: None (in pure strategies)")

        pareto = game_summary.get("pareto_optimal", [])
        if pareto:
            pareto_str = ", ".join([str(p) for p in pareto])
            theory_items.append(f"**Pareto Optimal**: {pareto_str}")

        dominant = game_summary.get("dominant_strategies", {})
        if any(v is not None for v in dominant.values()):
            dom_str = ", ".join([f"P{k}={v}" for k, v in dominant.items() if v is not None])
            theory_items.append(f"**Dominant Strategies**: {dom_str}")

        elements.append(mo.vstack([
            mo.md("#### Theoretical Properties"),
            mo.md("\n\n".join(theory_items)),
        ]))

        # Compare with LLM behavior if we have results
        if not results_df.is_empty():
            outcome_analysis = analyzer.analyze_game_outcomes(game, results_df)

            nash_rate = outcome_analysis.get("nash_play_rate", 0)
            pareto_rate = outcome_analysis.get("pareto_play_rate", 0)
            total_rounds = outcome_analysis.get("total_rounds", 0)

            comparison_items = [
                f"**Total Rounds**: {total_rounds}",
                f"**Nash Equilibrium Play Rate**: {nash_rate*100:.1f}%",
                f"**Pareto Optimal Play Rate**: {pareto_rate*100:.1f}%",
            ]

            elements.append(mo.vstack([
                mo.md("#### LLM Behavior vs Theory"),
                mo.md("\n\n".join(comparison_items)),
            ]))

            # Educational insights
            insights = outcome_analysis.get("insights", [])
            if insights:
                elements.append(mo.vstack([
                    mo.md("#### Educational Insights"),
                    mo.md("\n\n".join([f"- {insight}" for insight in insights])),
                ]))

            # Outcome distribution
            outcome_dist = outcome_analysis.get("outcome_distribution", {})
            if outcome_dist:
                dist_items = []
                for outcome, data in outcome_dist.items():
                    count = data.get("count", 0)
                    rate = data.get("rate", 0)
                    outcome_str = str(outcome)
                    is_nash = outcome in nash
                    is_pareto = outcome in pareto
                    markers = []
                    if is_nash:
                        markers.append("Nash")
                    if is_pareto:
                        markers.append("Pareto")
                    marker_str = f" [{', '.join(markers)}]" if markers else ""
                    dist_items.append(f"- {outcome_str}: {count} times ({rate*100:.1f}%){marker_str}")

                elements.append(mo.accordion({
                    "Outcome Distribution": mo.md("\n".join(dist_items)),
                }))

        return mo.vstack(elements)
