"""Game Analytics panel builder - Game/Outcome-focused analysis."""

from typing import Dict, List, Any, Optional
import marimo as mo
import polars as pl

from ..core.utils import detect_num_players
from ..core.types import GameDefinition
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
from ..analytics.equilibrium import EquilibriumAnalyzer


class GameAnalyticsPanelBuilder:
    """Static factory methods for building Game Analytics UI sections.

    Game Analytics focuses on game outcomes and performance:
    - API metrics and response times
    - Model leaderboards and performance rankings
    - Equilibrium analysis (Nash, Pareto)
    - Allocation metrics (HHI, compliance)
    - Tournament and ecosystem results
    """

    # =====================================================================
    # API METRICS & PERFORMANCE
    # =====================================================================

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

    # =====================================================================
    # LEADERBOARDS & MODEL COMPARISON
    # =====================================================================

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

    # =====================================================================
    # SESSION DETAIL & PAYOFF ANALYSIS
    # =====================================================================

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
        num_players = GameAnalyticsPanelBuilder._detect_num_players(results_df)

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

    # =====================================================================
    # EQUILIBRIUM ANALYSIS
    # =====================================================================

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
            create_allocation_violin,
            create_concentration_timeline,
            create_field_preference_bars,
            create_placement_grid,
            create_allocation_comparison,
            create_allocation_difference_heatmap,
            create_allocation_evolution_area,
            create_field_win_analysis,
            create_resource_efficiency_chart,
        )

        analyzer = AllocationAnalyzer()
        summaries = analyzer.analyze_session(results, num_players, budget)

        if not summaries:
            return mo.md("*No allocation data available for analysis.*")

        elements = [mo.md("### Allocation Analysis")]

        # Strategy metrics summary (moved up for immediate visibility)
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

        # =================================================================
        # ROW 1: PLACEMENT GRIDS + AGGREGATED COMPARISON (same row)
        # =================================================================
        elements.append(mo.md("#### Resource Placement & Comparison"))
        elements.append(mo.md("_Per-round placement grids (individual) + average comparison (aggregated)._"))

        # Build row with: P1 Grid | P2 Grid | Aggregated Comparison
        row1_charts = []
        for player_num in range(1, num_players + 1):
            grid = create_placement_grid(results, player_num, budget)
            row1_charts.append(grid)

        comparison_chart = create_allocation_comparison(results, num_players, aggregate=True)
        row1_charts.append(comparison_chart)

        elements.append(mo.hstack(row1_charts, justify="start", wrap=True))

        # =================================================================
        # ROW 2: DIFFERENCE MATRIX + FIELD WIN ANALYSIS (same row)
        # =================================================================
        if num_players == 2:
            elements.append(mo.md("#### Comparative Analysis"))
            elements.append(mo.md("_Allocation difference (P1-P2) + field win rates._"))

            diff_chart = create_allocation_difference_heatmap(results)
            field_win_chart = create_field_win_analysis(results, num_players)

            elements.append(mo.hstack([diff_chart, field_win_chart], justify="start", wrap=True))

        # =================================================================
        # ROW 3: ALLOCATION EVOLUTION (side by side)
        # =================================================================
        elements.append(mo.md("#### Allocation Evolution Over Rounds"))
        elements.append(mo.md("_Stacked area showing how each player shifts resources between fields._"))

        evolution_charts = []
        for player_num in range(1, num_players + 1):
            evo_chart = create_allocation_evolution_area(results, player_num)
            evolution_charts.append(evo_chart)

        if evolution_charts:
            elements.append(mo.hstack(evolution_charts, justify="start", wrap=True))

        # =================================================================
        # ROW 4: RESOURCE EFFICIENCY
        # =================================================================
        if num_players == 2:
            elements.append(mo.md("#### Resource Efficiency Over Rounds"))
            elements.append(mo.md("_Win rate (fields won / total) tracking over time._"))
            efficiency_chart = create_resource_efficiency_chart(results)
            elements.append(efficiency_chart)

        # =================================================================
        # SECTION 7: CONCENTRATION TIMELINE (existing)
        # =================================================================
        elements.append(mo.md("#### Strategy Concentration Timeline"))
        concentration_chart = create_concentration_timeline(results, num_players, budget)
        elements.append(concentration_chart)

        # =================================================================
        # SECTION 8: FIELD PREFERENCES (existing)
        # =================================================================
        num_fields = len(next(iter(summaries.values())).field_preferences) if summaries else 0
        if num_fields > 0:
            elements.append(mo.md("#### Average Field Preferences"))
            pref_chart = create_field_preference_bars(summaries, num_fields)
            elements.append(pref_chart)

        # =================================================================
        # SECTION 9: DISTRIBUTIONS (existing, in accordion)
        # =================================================================
        # Distribution views (violin plots) - in accordion to save space
        violins = []
        for player_num in range(1, num_players + 1):
            violin = create_allocation_violin(results, player_num)
            violins.append(violin)

        if violins:
            violin_content = mo.vstack([
                mo.md("_Shows probability density of allocations per battlefield._"),
                mo.hstack(violins, justify="start", wrap=True),
            ])
            elements.append(mo.accordion({
                "Allocation Distributions (Violin Plots)": violin_content
            }))

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
        allocation_section = GameAnalyticsPanelBuilder.build_allocation_analysis_section(
            results, num_players, budget
        )
        elements.append(allocation_section)

        # Compliance metrics
        compliance_section = GameAnalyticsPanelBuilder.build_compliance_metrics_section(
            results, num_players
        )
        elements.append(compliance_section)

        # Evolution analysis for each player (in accordion)
        evolution_accordions = {}
        for player_num in range(1, num_players + 1):
            evolution_section = GameAnalyticsPanelBuilder.build_strategy_evolution_section(
                results, player_num, budget
            )
            model_key = f"player{player_num}_model"
            model = results[0].get(model_key, f"Player {player_num}") if results else f"Player {player_num}"
            evolution_accordions[f"P{player_num} Evolution ({model[:12]})"] = evolution_section

        if evolution_accordions:
            elements.append(mo.accordion(evolution_accordions))

        return mo.vstack(elements)

    @staticmethod
    def build_allocation_aggregate_section(
        all_sessions_results: List[Dict[str, Any]],
        num_players: int = 2,
        budget: float = 100.0,
        game_name: str = "Allocation Game"
    ) -> mo.Html:
        """Build allocation analysis across ALL filtered sessions (aggregate view).

        This is used in the All Sessions dashboard view when filtering to an
        allocation game. Shows combined patterns across multiple sessions.

        Args:
            all_sessions_results: Combined results from all filtered sessions
            num_players: Number of players
            budget: Allocation budget
            game_name: Name of the game for display

        Returns:
            Marimo vstack with aggregate allocation analytics
        """
        from ..analytics.allocation import AllocationAnalyzer
        from ..visualization.allocation_charts import (
            create_allocation_comparison,
            create_field_preference_bars,
            create_concentration_timeline,
            create_strategy_type_pie,
            create_field_win_analysis,
        )

        if not all_sessions_results:
            return mo.callout(
                mo.md("*No allocation data available. Run some games to see aggregate analytics.*"),
                kind="info",
            )

        analyzer = AllocationAnalyzer()
        summaries = analyzer.analyze_session(all_sessions_results, num_players, budget)

        elements = [mo.md(f"### {game_name} - Aggregate Analysis (All Sessions)")]

        if not summaries:
            elements.append(mo.md("*No allocation data available for analysis.*"))
            return mo.vstack(elements)

        # Strategy classification summary
        metrics_rows = []
        for player_num, summary in summaries.items():
            metrics_rows.append(
                f"**P{player_num}**: {summary.dominant_strategy.title()} strategy "
                f"(HHI={summary.avg_concentration:.2f}, "
                f"consistency={summary.strategy_consistency*100:.0f}%)"
            )

        if metrics_rows:
            elements.append(mo.callout(
                mo.md("**Aggregate Strategy Classification**\n\n" + "\n\n".join(metrics_rows)),
                kind="info",
            ))

        # Row 1: Head-to-head comparison + Field win analysis
        elements.append(mo.md("#### Cross-Session Allocation Patterns"))

        comparison_chart = create_allocation_comparison(all_sessions_results, num_players, aggregate=True)
        field_win_chart = create_field_win_analysis(all_sessions_results, num_players) if num_players == 2 else None

        if field_win_chart:
            elements.append(mo.hstack([comparison_chart, field_win_chart], justify="start", wrap=True))
        else:
            elements.append(comparison_chart)

        # Row 2: Concentration timeline
        elements.append(mo.md("#### Strategy Concentration Trends"))
        concentration_chart = create_concentration_timeline(all_sessions_results, num_players, budget)
        elements.append(concentration_chart)

        # Row 3: Field preferences
        num_fields = len(next(iter(summaries.values())).field_preferences) if summaries else 0
        if num_fields > 0:
            elements.append(mo.md("#### Field Preference Distribution"))
            pref_chart = create_field_preference_bars(summaries, num_fields)
            elements.append(pref_chart)

        # Strategy type distribution (aggregate across all rounds)
        all_metrics = []
        for result in all_sessions_results:
            for player_num in range(1, num_players + 1):
                allocation = analyzer.parse_allocation_from_result(result, player_num)
                if allocation:
                    metrics = analyzer.analyze_allocation(allocation, budget)
                    all_metrics.append(metrics)

        if all_metrics:
            elements.append(mo.md("#### Strategy Type Distribution"))
            strategy_pie = create_strategy_type_pie(all_metrics)
            elements.append(strategy_pie)

        return mo.vstack(elements)

    # =====================================================================
    # SENSITIVITY & ADVANCED ANALYSIS
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

    # =====================================================================
    # TOURNAMENT & ECOSYSTEM
    # =====================================================================

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
    def build_model_game_specialization_section(
        cross_game_analyzer: 'CrossGameComparativeAnalyzer',
        models: Optional[List[str]] = None
    ) -> mo.Html:
        """Build model-game performance matrix showing specialization.

        Displays a heatmap of models (rows) x games (columns) colored by win rate,
        with specialization scores showing whether a model is a generalist or specialist.

        Args:
            cross_game_analyzer: CrossGameComparativeAnalyzer instance
            models: Optional list of models to include

        Returns:
            Marimo vstack with specialization heatmap and stats
        """
        import altair as alt

        summary = cross_game_analyzer.summarize_all()

        elements = [mo.md("### Model-Game Performance Matrix")]

        if summary.get("error"):
            elements.append(mo.callout(
                mo.md(summary["error"]),
                kind="neutral",
            ))
            return mo.vstack(elements)

        model_performance = summary.get("model_performance", {})
        game_types = summary.get("game_types", [])

        if not model_performance or not game_types:
            elements.append(mo.callout(
                mo.md("No cross-game data available. Run games across multiple game types."),
                kind="neutral",
            ))
            return mo.vstack(elements)

        elements.append(mo.md(
            "_Win rate performance across game types. Darker = higher win rate._"
        ))

        # Build matrix data for heatmap
        matrix_rows = []
        specialization_data = []

        for model, perf in model_performance.items():
            win_rates = perf.get("win_rates_by_game", {})
            spec_score = perf.get("specialization_score", 0)
            overall_wr = perf.get("overall_win_rate", 0)
            best_game = perf.get("best_game", "")

            # Add row for each game type
            for game in game_types:
                wr = win_rates.get(game, None)
                if wr is not None:
                    matrix_rows.append({
                        "model": model,
                        "game": game,
                        "win_rate": wr,
                    })

            # Specialization data
            specialization_data.append({
                "model": model,
                "specialization": spec_score,
                "overall_win_rate": overall_wr,
                "best_game": best_game,
                "type": "Specialist" if spec_score > 0.3 else "Generalist",
            })

        if matrix_rows:
            import polars as pl
            matrix_df = pl.DataFrame(matrix_rows)

            # Create heatmap
            heatmap = (
                alt.Chart(matrix_df.to_pandas())
                .mark_rect()
                .encode(
                    x=alt.X("game:N", title="Game Type", axis=alt.Axis(labelAngle=-45)),
                    y=alt.Y("model:N", title="Model"),
                    color=alt.Color(
                        "win_rate:Q",
                        scale=alt.Scale(scheme="viridis", domain=[0, 1]),
                        title="Win Rate",
                    ),
                    tooltip=[
                        alt.Tooltip("model:N", title="Model"),
                        alt.Tooltip("game:N", title="Game"),
                        alt.Tooltip("win_rate:Q", title="Win Rate", format=".1%"),
                    ],
                )
                .properties(width=400, height=max(150, len(model_performance) * 30))
            )
            elements.append(heatmap)

        # Specialization stats
        if specialization_data:
            elements.append(mo.md("#### Model Specialization"))
            elements.append(mo.md(
                "_Specialist models excel at specific games. Generalists perform consistently across games._"
            ))

            # Create cards for each model
            spec_cards = []
            for data in sorted(specialization_data, key=lambda x: -x["overall_win_rate"]):
                model = data["model"]
                spec = data["specialization"]
                wr = data["overall_win_rate"]
                best = data["best_game"]
                model_type = data["type"]

                # Color based on specialization type
                kind = "warn" if model_type == "Specialist" else "info"

                card = mo.callout(
                    mo.vstack([
                        mo.md(f"**{model[:20]}**"),
                        mo.md(f"Type: {model_type}"),
                        mo.md(f"Overall WR: {wr:.1%}"),
                        mo.md(f"Best: {best[:12]}") if best else mo.md(""),
                        mo.md(f"Spec Score: {spec:.2f}"),
                    ]),
                    kind=kind,
                )
                spec_cards.append(card)

            if spec_cards:
                # Show max 4 cards per row
                for i in range(0, len(spec_cards), 4):
                    elements.append(mo.hstack(spec_cards[i:i+4], justify="start"))

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
