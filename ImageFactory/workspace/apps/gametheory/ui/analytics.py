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

    # =====================================================================
    # ALLOCATION GAME ANALYTICS (Burr games: Blotto, Tennis, Sumo)
    # =====================================================================

    @staticmethod
    def build_allocation_analysis_section(
        results: List[Dict[str, Any]],
        num_players: int = 2,
        budget: float = 100.0
    ) -> mo.Html:
        """Build allocation analysis section for Burr games.

        Args:
            results: List of round result dictionaries
            num_players: Number of players
            budget: Allocation budget (e.g., 100 troops)

        Returns:
            Marimo vstack with allocation analytics
        """
        from ..analytics.allocation import AllocationAnalyzer
        from ..visualization.allocation_charts import (
            create_allocation_heatmap,
            create_concentration_timeline,
            create_field_preference_bars,
        )

        analyzer = AllocationAnalyzer()
        summaries = analyzer.analyze_session(results, num_players, budget)

        if not summaries:
            return mo.md("*No allocation data available for analysis.*")

        elements = [mo.md("### Allocation Analysis")]

        # Heatmaps side by side
        heatmaps = []
        for player_num in range(1, num_players + 1):
            heatmap = create_allocation_heatmap(results, player_num)
            heatmaps.append(heatmap)

        if heatmaps:
            elements.append(mo.hstack(heatmaps, justify="start"))

        # Strategy metrics summary
        metrics_rows = []
        for player_num, summary in summaries.items():
            metrics_rows.append(
                f"**P{player_num} ({summary.model[:15]})**: "
                f"{summary.dominant_strategy.title()} strategy "
                f"(HHI={summary.avg_concentration:.2f}, "
                f"consistency={summary.strategy_consistency*100:.0f}%)"
            )

        if metrics_rows:
            elements.append(mo.callout(
                mo.md("**Strategy Classification**\n\n" + "\n\n".join(metrics_rows)),
                kind="info",
            ))

        # Concentration timeline
        concentration_chart = create_concentration_timeline(results, num_players, budget)
        elements.append(concentration_chart)

        # Field preferences
        num_fields = len(next(iter(summaries.values())).field_preferences) if summaries else 0
        if num_fields > 0:
            pref_chart = create_field_preference_bars(summaries, num_fields)
            elements.append(pref_chart)

        return mo.vstack(elements)

    @staticmethod
    def build_compliance_metrics_section(
        results: List[Dict[str, Any]],
        num_players: int = 2
    ) -> mo.Html:
        """Build compliance metrics section showing parse/normalize rates.

        Args:
            results: List of round result dictionaries
            num_players: Number of players

        Returns:
            Marimo vstack with compliance charts
        """
        from ..analytics.allocation import AllocationAnalyzer
        from ..visualization.allocation_charts import create_compliance_summary_chart

        analyzer = AllocationAnalyzer()
        elements = [mo.md("### Response Compliance")]

        charts = []
        stats = []

        for player_num in range(1, num_players + 1):
            compliance = analyzer.calculate_compliance_metrics(results, player_num)

            # Chart
            chart = create_compliance_summary_chart(results, player_num)
            charts.append(chart)

            # Stats
            model_key = f"player{player_num}_model"
            model = results[0].get(model_key, f"Player {player_num}") if results else f"Player {player_num}"

            stats.append(mo.stat(
                label=f"P{player_num} Parse Rate",
                value=f"{compliance.parse_rate*100:.0f}%",
                caption=f"{model[:12]}: {compliance.parsed_ok}/{compliance.total_rounds} OK",
            ))

        if charts:
            elements.append(mo.hstack(charts, justify="start"))

        if stats:
            elements.append(mo.hstack(stats, justify="start"))

        return mo.vstack(elements)

    @staticmethod
    def build_strategy_evolution_section(
        results: List[Dict[str, Any]],
        player_num: int,
        budget: float = 100.0
    ) -> mo.Html:
        """Build strategy evolution analysis section.

        Args:
            results: List of round result dictionaries
            player_num: Player number to analyze
            budget: Allocation budget

        Returns:
            Marimo vstack with evolution analysis
        """
        from ..analytics.allocation import AllocationAnalyzer
        from ..visualization.allocation_charts import create_evolution_summary_chart

        analyzer = AllocationAnalyzer()
        evolution = analyzer.detect_strategy_evolution(results, player_num, budget)

        elements = [mo.md(f"### Strategy Evolution (Player {player_num})")]

        if not evolution.get("has_evolution"):
            elements.append(mo.callout(
                mo.md(evolution.get("explanation", "No significant evolution detected.")),
                kind="neutral",
            ))
        else:
            # Evolution chart
            chart = create_evolution_summary_chart(evolution)
            elements.append(chart)

            # Evolution summary
            trend = evolution.get("trend", "unknown")
            change = evolution.get("concentration_change", 0)

            trend_icons = {
                "concentrating": "focusing resources",
                "spreading": "diversifying resources",
                "stable": "maintaining strategy",
                "slight_shift": "minor adjustment",
            }

            elements.append(mo.callout(
                mo.md(f"""
**Trend:** {trend.title()} ({trend_icons.get(trend, '')})

**First Half:** {evolution.get('first_half_strategy', 'unknown').title()} (HHI={evolution.get('first_half_concentration', 0):.2f})

**Second Half:** {evolution.get('second_half_strategy', 'unknown').title()} (HHI={evolution.get('second_half_concentration', 0):.2f})

**Change:** {change:+.3f} HHI
                """),
                kind="success" if evolution.get("strategy_shift") else "info",
            ))

        return mo.vstack(elements)

    @staticmethod
    def build_allocation_session_detail(
        results: List[Dict[str, Any]],
        num_players: int = 2,
        budget: float = 100.0,
        game_name: str = "Allocation Game"
    ) -> mo.Html:
        """Build complete session detail view for allocation games.

        This is the main entry point for Burr game session analysis,
        replacing the equilibrium-based analysis used for discrete games.

        Args:
            results: List of round result dictionaries
            num_players: Number of players
            budget: Allocation budget
            game_name: Name of the game for display

        Returns:
            Marimo vstack with complete allocation analysis
        """
        elements = [mo.md(f"## {game_name} Analysis")]

        # Main allocation analysis
        allocation_section = AnalyticsPanelBuilder.build_allocation_analysis_section(
            results, num_players, budget
        )
        elements.append(allocation_section)

        # Compliance metrics
        compliance_section = AnalyticsPanelBuilder.build_compliance_metrics_section(
            results, num_players
        )
        elements.append(compliance_section)

        # Evolution analysis for each player (in accordion)
        evolution_accordions = {}
        for player_num in range(1, num_players + 1):
            evolution_section = AnalyticsPanelBuilder.build_strategy_evolution_section(
                results, player_num, budget
            )
            model_key = f"player{player_num}_model"
            model = results[0].get(model_key, f"Player {player_num}") if results else f"Player {player_num}"
            evolution_accordions[f"P{player_num} Evolution ({model[:12]})"] = evolution_section

        if evolution_accordions:
            elements.append(mo.accordion(evolution_accordions))

        return mo.vstack(elements)

    # =====================================================================
    # NEW ANALYSIS FEATURES
    # =====================================================================

    @staticmethod
    def build_sensitivity_section(
        sweep_summary: 'SweepSummary'
    ) -> mo.Html:
        """Build hyperparameter sensitivity analysis section.

        Args:
            sweep_summary: SweepSummary from HyperparameterSweeper

        Returns:
            Marimo vstack with sensitivity analysis
        """
        from ..analytics.sensitivity import HyperparameterSensitivityAnalyzer
        from ..visualization.allocation_charts import (
            create_compliance_by_penalty_chart,
            create_sensitivity_heatmap,
        )

        analyzer = HyperparameterSensitivityAnalyzer(sweep_summary)
        summary = analyzer.summarize()

        elements = [mo.md("### Hyperparameter Sensitivity Analysis")]

        # Parameter importance
        importance = summary.get("parameter_importance", {})
        if importance:
            importance_items = []
            for param, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
                bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
                importance_items.append(f"- **{param}**: [{bar}] {score:.2f}")

            elements.append(mo.callout(
                mo.md("**Parameter Importance**\n\n" + "\n".join(importance_items)),
                kind="info",
            ))

        # Compliance by repeat_penalty
        compliance_result = analyzer.analyze_compliance_by_repeat_penalty()
        if compliance_result.values:
            compliance_chart = create_compliance_by_penalty_chart(compliance_result)
            elements.append(compliance_chart)

        # Strategy variance interaction
        variance_result = analyzer.analyze_strategy_variance()
        if variance_result.x_values and variance_result.y_values:
            variance_chart = create_sensitivity_heatmap(variance_result)
            elements.append(variance_chart)

        # Optimal parameters
        opt = summary.get("optimal_balanced", {})
        if opt.get("parameters"):
            opt_items = [f"- {k}: {v}" for k, v in opt.get("parameters", {}).items()]
            elements.append(mo.callout(
                mo.md(f"**Optimal Parameters (balanced)**\n\n" +
                      "\n".join(opt_items) +
                      f"\n\nWin Rate: {opt.get('achieved_value', 0):.1%}"),
                kind="success",
            ))

        return mo.vstack(elements)

    @staticmethod
    def build_personality_section(
        profiler: 'ModelPersonalityProfiler',
        model: str
    ) -> mo.Html:
        """Build model personality profile section.

        Args:
            profiler: ModelPersonalityProfiler instance
            model: Model name to profile

        Returns:
            Marimo vstack with personality analysis
        """
        from ..visualization.allocation_charts import create_personality_comparison_chart

        summary = profiler.summarize(model)
        profile = summary.get("profile", {})

        elements = [mo.md(f"### Model Personality: {model}")]

        # Summary text
        elements.append(mo.callout(mo.md(summary.get("summary", "")), kind="info"))

        # Profile stats
        stats = [
            mo.stat(
                label="Bias Score",
                value=f"{profile.get('bias_score', 0):.2f}",
                caption="-1=clustered, +1=uniform",
            ),
            mo.stat(
                label="Symmetry Breaking",
                value=f"{profile.get('symmetry_breaking_score', 0):.2f}",
                caption="Field preference strength",
            ),
            mo.stat(
                label="Consistency",
                value=f"{profile.get('consistency_score', 0):.2f}",
                caption="Cross-game stability",
            ),
            mo.stat(
                label="Total Sessions",
                value=str(profile.get("total_sessions", 0)),
            ),
        ]

        elements.append(mo.hstack(stats, justify="start"))

        # Game fingerprints
        fingerprints = summary.get("game_fingerprints", {})
        if fingerprints:
            fp_items = []
            for game, fp in fingerprints.items():
                fp_items.append(
                    f"- **{game}**: {fp.get('dominant_strategy', 'unknown').title()} "
                    f"(HHI={fp.get('avg_concentration', 0):.2f}, {fp.get('num_rounds', 0)} rounds)"
                )

            elements.append(mo.accordion({
                "Game-Specific Fingerprints": mo.md("\n".join(fp_items)),
            }))

        return mo.vstack(elements)

    @staticmethod
    def build_meta_learning_section(
        results: List[Dict[str, Any]],
        num_players: int = 2
    ) -> mo.Html:
        """Build meta-strategy learning analysis section.

        Args:
            results: List of round result dictionaries
            num_players: Number of players

        Returns:
            Marimo vstack with meta-learning analysis
        """
        from ..analytics.meta_learning import MetaStrategyAnalyzer

        analyzer = MetaStrategyAnalyzer(results, num_players)
        summary = analyzer.summarize_all_players()

        elements = [mo.md("### Meta-Strategy Learning Analysis")]

        if not summary.get("players"):
            elements.append(mo.callout(
                mo.md("Insufficient data for meta-learning analysis."),
                kind="neutral",
            ))
            return mo.vstack(elements)

        # Player adaptation cards
        cards = []
        for player_num, data in summary.get("players", {}).items():
            adaptation = data.get("adaptation", {})
            memory = data.get("memory_effect", {})
            learning = data.get("learning", {})

            # Determine card kind
            if adaptation.get("detected"):
                kind = "success"
            elif learning.get("trend") == "improving":
                kind = "info"
            else:
                kind = "neutral"

            card_content = mo.vstack([
                mo.md(f"**{data.get('model', f'Player {player_num}')}**"),
                mo.md(f"Adaptation: **{adaptation.get('type', 'none').title()}**"),
                mo.md(f"Memory Effect: {memory.get('direction', 'none').title()} "
                      f"(r={memory.get('correlation', 0):.2f})"),
                mo.md(f"Learning Trend: {learning.get('trend', 'stable').title()} "
                      f"(rate={learning.get('rate', 0):.3f})"),
            ])

            cards.append(mo.callout(card_content, kind=kind))

        if cards:
            elements.append(mo.hstack(cards, justify="start"))

        return mo.vstack(elements)

    @staticmethod
    def build_tournament_section(
        result: 'TournamentResult'
    ) -> mo.Html:
        """Build tournament results section.

        Args:
            result: TournamentResult from TournamentRunner

        Returns:
            Marimo vstack with tournament results
        """
        from ..visualization.allocation_charts import (
            create_tournament_standings_chart,
            create_matchup_matrix_heatmap,
        )

        elements = [mo.md(f"### Tournament Results: {result.tournament_id}")]

        # Tournament info
        elements.append(mo.hstack([
            mo.stat(label="Format", value=result.config.format.title()),
            mo.stat(label="Models", value=str(len(result.config.models))),
            mo.stat(label="Games", value=str(len(result.config.games))),
            mo.stat(label="Matches", value=str(len(result.match_results))),
        ], justify="start"))

        # Standings chart
        standings_chart = create_tournament_standings_chart(result.standings)
        elements.append(standings_chart)

        # Matchup matrix
        matchup_chart = create_matchup_matrix_heatmap(
            result.matchup_matrix,
            result.model_indices
        )
        elements.append(matchup_chart)

        # Detailed standings table
        standings_rows = [
            {
                "Rank": i + 1,
                "Model": s.model,
                "Points": s.points,
                "W": s.wins,
                "L": s.losses,
                "D": s.draws,
                "Win Rate": f"{s.win_rate:.1%}",
            }
            for i, s in enumerate(result.standings)
        ]

        elements.append(mo.accordion({
            "Detailed Standings": mo.ui.table(standings_rows),
        }))

        return mo.vstack(elements)

    @staticmethod
    def build_ecosystem_section(
        result: 'EcosystemResult'
    ) -> mo.Html:
        """Build ecosystem simulation results section.

        Args:
            result: EcosystemResult from EcosystemSimulator

        Returns:
            Marimo vstack with ecosystem analysis
        """
        from ..visualization.allocation_charts import (
            create_ecosystem_timeline,
            create_diversity_timeline,
        )

        elements = [mo.md(f"### Ecosystem Simulation: {result.simulation_id}")]

        # Summary stats
        eq = result.equilibrium_analysis
        cycles = result.cyclical_patterns

        elements.append(mo.hstack([
            mo.stat(label="Total Rounds", value=str(result.total_rounds)),
            mo.stat(label="Models", value=str(len(result.models))),
            mo.stat(label="Converged", value="Yes" if eq.converged else "No"),
            mo.stat(label="Equilibrium", value=eq.equilibrium_type.title()),
        ], justify="start"))

        # Equilibrium analysis
        eq_kind = "success" if eq.converged else "warn" if eq.stability_score > 0.5 else "neutral"
        elements.append(mo.callout(
            mo.md(f"**Equilibrium Analysis**\n\n{eq.explanation}\n\n"
                  f"Dominant Archetype: {eq.dominant_archetype.title()}\n"
                  f"Stability Score: {eq.stability_score:.1%}"),
            kind=eq_kind,
        ))

        # Cyclical patterns
        if cycles.detected:
            elements.append(mo.callout(
                mo.md(f"**Cyclical Pattern Detected**\n\n{cycles.explanation}\n\n"
                      f"Cycle: {' -> '.join(cycles.cycle_archetypes)}"),
                kind="info",
            ))

        # Timeline charts
        if result.states:
            eco_timeline = create_ecosystem_timeline(result.states)
            elements.append(eco_timeline)

            diversity_chart = create_diversity_timeline(result.states)
            elements.append(diversity_chart)

        # Final standings
        standings_items = []
        sorted_standings = sorted(
            result.final_standings.items(),
            key=lambda x: x[1].get("win_rate", 0),
            reverse=True
        )
        for model, stats in sorted_standings:
            standings_items.append(
                f"- **{model}**: {stats.get('win_rate', 0):.1%} win rate, "
                f"{stats.get('total_wins', 0)} wins / {stats.get('total_games', 0)} games"
            )

        elements.append(mo.accordion({
            "Final Standings": mo.md("\n".join(standings_items)),
        }))

        return mo.vstack(elements)

    @staticmethod
    def build_intelligence_leaderboard_section(
        cross_game_analyzer: 'CrossGameComparativeAnalyzer',
        models: Optional[List[str]] = None
    ) -> mo.Html:
        """Build intelligence proxy leaderboard section.

        Args:
            cross_game_analyzer: CrossGameComparativeAnalyzer instance
            models: Optional list of models to include

        Returns:
            Marimo vstack with intelligence leaderboard
        """
        from ..visualization.allocation_charts import (
            create_intelligence_leaderboard_chart,
            create_intelligence_breakdown_chart,
        )

        leaderboard = cross_game_analyzer.create_intelligence_leaderboard(models)

        elements = [mo.md("### Model Intelligence Proxy Leaderboard")]

        if leaderboard.is_empty():
            elements.append(mo.callout(
                mo.md("No intelligence data available. Run some games first!"),
                kind="neutral",
            ))
            return mo.vstack(elements)

        elements.append(mo.md(
            "_Intelligence proxy: Composite score based on compliance (25%), "
            "efficiency (25%), adaptation (25%), and meta-awareness (25%)._"
        ))

        # Leaderboard chart
        leaderboard_chart = create_intelligence_leaderboard_chart(leaderboard)
        elements.append(leaderboard_chart)

        # Breakdown chart
        breakdown_chart = create_intelligence_breakdown_chart(leaderboard)
        elements.append(breakdown_chart)

        # Top 3 callout
        top_3 = leaderboard.head(3).to_dicts()
        if top_3:
            medals = ["🥇", "🥈", "🥉"]
            top_items = [
                f"{medals[i]} **{row['model']}**: IQ={row['composite_iq']:.0f}"
                for i, row in enumerate(top_3)
            ]
            elements.append(mo.callout(
                mo.md("**Top Performers**\n\n" + "\n".join(top_items)),
                kind="success",
            ))

        return mo.vstack(elements)

    @staticmethod
    def build_payoff_sensitivity_section(
        summary: 'PayoffSensitivitySummary'
    ) -> mo.Html:
        """Build payoff function sensitivity analysis section.

        Args:
            summary: PayoffSensitivitySummary from PayoffSensitivityAnalyzer

        Returns:
            Marimo vstack with payoff sensitivity analysis
        """
        from ..visualization.allocation_charts import create_payoff_sensitivity_chart

        elements = [mo.md(f"### Payoff Function Sensitivity: {summary.base_game}")]

        elements.append(mo.hstack([
            mo.stat(label="Variants Tested", value=str(summary.num_variants)),
            mo.stat(label="Most Sensitive", value=summary.most_sensitive_param.title()),
        ], justify="start"))

        # Results chart
        if summary.results:
            chart = create_payoff_sensitivity_chart(summary.results)
            elements.append(chart)

        # Sensitivity by parameter
        if summary.sensitivity_by_parameter:
            sens_items = []
            for param, score in sorted(
                summary.sensitivity_by_parameter.items(),
                key=lambda x: x[1],
                reverse=True
            ):
                bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
                sens_items.append(f"- **{param}**: [{bar}] {score:.2f}")

            elements.append(mo.callout(
                mo.md("**Sensitivity by Payoff Type**\n\n" + "\n".join(sens_items)),
                kind="info",
            ))

        # Recommendations
        if summary.recommendations:
            elements.append(mo.accordion({
                "Recommendations": mo.md("\n".join([f"- {r}" for r in summary.recommendations])),
            }))

        return mo.vstack(elements)
