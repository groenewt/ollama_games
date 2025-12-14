"""Role Analytics panel builder - Player/Model-focused analysis."""

from typing import Dict, List, Any, Optional
from dataclasses import asdict
import marimo as mo
import polars as pl

from ..analytics.strategy import StrategyDetector, StrategyType, STRATEGY_DESCRIPTIONS
from ..analytics.learning import LearningAnalyzer
from ..core.role import RoleConfig


# =============================================================================
# FILTER COMPONENTS
# =============================================================================


class RoleAnalyticsFilterBuilder:
    """Builds the smart chip filtering UI for Role Analytics."""

    @staticmethod
    def build_role_selector(
        roles: List[RoleConfig],
        selected_role_id: Optional[str] = None,
    ) -> mo.ui.dropdown:
        """Build the primary role selector dropdown.

        Args:
            roles: List of available roles
            selected_role_id: Currently selected role ID (for persistence)

        Returns:
            Dropdown for role selection
        """
        if not roles:
            return mo.ui.dropdown(
                options={},
                label="Select Role to Analyze",
                value=None,
            )

        options = {f"{r.name} ({r.model})": r.role_id for r in roles}

        # Find the display key for selected role
        selected_key = None
        if selected_role_id:
            for key, rid in options.items():
                if rid == selected_role_id:
                    selected_key = key
                    break

        return mo.ui.dropdown(
            options=options,
            label="Select Role to Analyze",
            value=selected_key,
        )

    @staticmethod
    def build_game_type_chips(
        available_games: List[str],
        game_names: Dict[str, str],
        selected: Optional[List[str]] = None,
    ) -> mo.ui.multiselect:
        """Build game type filter chips.

        Only shows games where the selected role has data.

        Args:
            available_games: List of game IDs with data
            game_names: Dict mapping game_id to display name
            selected: Currently selected game types

        Returns:
            Multiselect for game type filtering
        """
        if not available_games:
            return mo.ui.multiselect(
                options={},
                label="Filter by Game Type (optional)",
                value=[],
            )

        options = {game_names.get(g, g): g for g in available_games}
        return mo.ui.multiselect(
            options=options,
            label="Filter by Game Type (optional)",
            value=selected or [],
        )

    @staticmethod
    def build_session_chips(
        sessions: List[Dict[str, Any]],
        selected: Optional[List[str]] = None,
    ) -> mo.ui.multiselect:
        """Build session filter chips.

        Only shows sessions where the selected role participated.

        Args:
            sessions: List of session metadata dicts
            selected: Currently selected session IDs

        Returns:
            Multiselect for session filtering
        """
        if not sessions:
            return mo.ui.multiselect(
                options={},
                label="Filter by Session (optional)",
                value=[],
            )

        options = {
            f"{s.get('session_id', '')[:8]} ({s.get('game_type', 'unknown')})": s.get('session_id', '')
            for s in sessions[:50]  # Limit to most recent 50
        }
        return mo.ui.multiselect(
            options=options,
            label="Filter by Session (optional)",
            value=selected or [],
        )

    @staticmethod
    def build_filter_bar(
        role_selector: mo.ui.dropdown,
        game_chips: Optional[mo.ui.multiselect] = None,
        session_chips: Optional[mo.ui.multiselect] = None,
    ) -> mo.Html:
        """Compose the complete filter bar.

        Progressive disclosure: chips only appear after role selection.

        Args:
            role_selector: The role dropdown
            game_chips: Optional game type multiselect
            session_chips: Optional session multiselect

        Returns:
            Complete filter bar as Marimo Html
        """
        elements = [
            mo.md("## Role Analytics"),
            role_selector,
        ]

        # Only show secondary filters if role is selected
        if role_selector.value:
            chip_row = []
            if game_chips is not None:
                chip_row.append(game_chips)
            if session_chips is not None:
                chip_row.append(session_chips)

            if chip_row:
                elements.append(mo.hstack(chip_row, justify="start", wrap=True))

                # Filter summary
                filters_active = []
                if game_chips is not None and game_chips.value:
                    filters_active.append(f"{len(game_chips.value)} game(s)")
                if session_chips is not None and session_chips.value:
                    filters_active.append(f"{len(session_chips.value)} session(s)")

                if filters_active:
                    elements.append(
                        mo.callout(
                            mo.md(f"**Active filters**: {', '.join(filters_active)}"),
                            kind="info",
                        )
                    )

        return mo.vstack(elements)


# =============================================================================
# EMPTY STATE HANDLERS
# =============================================================================


class RoleAnalyticsEmptyStates:
    """Standardized empty state handlers for Role Analytics."""

    @staticmethod
    def no_role_selected() -> mo.Html:
        """Display when no role is selected."""
        return mo.vstack([
            mo.md("## Role Analytics"),
            mo.callout(
                mo.md(
                    "**Select a role to analyze**\n\n"
                    "Choose a role from the dropdown above to view detailed "
                    "behavioral analysis, performance metrics, and personality profiling."
                ),
                kind="neutral",
            ),
            mo.md("_Tip: Create roles in the Roles tab to enable analytics._"),
        ])

    @staticmethod
    def no_data_for_role(role_name: str) -> mo.Html:
        """Display when role exists but has no session data."""
        return mo.vstack([
            mo.md(f"## Role Analytics: {role_name}"),
            mo.callout(
                mo.md(
                    f"**No session data available for {role_name}**\n\n"
                    "This role hasn't participated in any games yet. "
                    "Go to the Play tab to run some sessions with this role."
                ),
                kind="warn",
            ),
        ])

    @staticmethod
    def no_data_for_filters(
        role_name: str,
        game_filters: List[str],
        session_filters: List[str],
    ) -> mo.Html:
        """Display when filters exclude all data."""
        filter_desc = []
        if game_filters:
            filter_desc.append(f"games: {', '.join(game_filters[:3])}")
        if session_filters:
            filter_desc.append(f"sessions: {', '.join([s[:8] for s in session_filters[:3]])}")

        return mo.vstack([
            mo.md(f"## Role Analytics: {role_name}"),
            mo.callout(
                mo.md(
                    f"**No data matches current filters**\n\n"
                    f"Active filters: {'; '.join(filter_desc)}\n\n"
                    "Try removing some filters to see more data."
                ),
                kind="info",
            ),
        ])

    @staticmethod
    def insufficient_data(role_name: str, reasons: List[str]) -> mo.Html:
        """Display when data is insufficient for certain analytics."""
        reasons_md = "\n".join([f"- {r}" for r in reasons])
        return mo.callout(
            mo.md(f"**Limited analytics for {role_name}**\n\n{reasons_md}"),
            kind="neutral",
        )


# =============================================================================
# IMPROVED PANEL BUILDER WITH ACCORDION ORGANIZATION
# =============================================================================


class ImprovedRoleAnalyticsPanelBuilder:
    """Refactored Role Analytics panel builder with smart organization.

    Provides accordion-based organization and uses the new RoleAnalyticsService
    for cross-session role data.
    """

    @staticmethod
    def build_summary_section(
        role: RoleConfig,
        stats: 'RoleStatistics',
        personality_summary: Optional[str] = None,
    ) -> mo.Html:
        """Build the always-visible summary overview.

        Args:
            role: The selected RoleConfig
            stats: RoleStatistics from RoleAnalyticsService
            personality_summary: Optional one-sentence personality description

        Returns:
            Marimo vstack with summary
        """
        elements = [
            mo.md(f"### {role.name}"),
            mo.md(f"_Model: {role.model} | Endpoint: {role.endpoint}_"),
        ]

        # Stats row
        stat_cards = [
            mo.stat(label="Sessions", value=str(stats.total_sessions)),
            mo.stat(label="Total Rounds", value=str(stats.total_rounds)),
            mo.stat(
                label="Avg Payoff",
                value=f"{stats.avg_payoff:.1f}",
                caption=f"Range: {stats.min_payoff:.0f} - {stats.max_payoff:.0f}",
            ),
            mo.stat(
                label="Win Rate",
                value=f"{stats.win_rate:.1f}%",
            ),
        ]

        if stats.cooperation_rate is not None:
            stat_cards.append(
                mo.stat(
                    label="Coop Rate",
                    value=f"{stats.cooperation_rate:.0f}%",
                )
            )

        elements.append(mo.hstack(stat_cards, justify="start", wrap=True))

        # Games played
        if stats.games_played:
            elements.append(
                mo.md(f"**Games played**: {', '.join(stats.games_played)}")
            )

        # Personality summary if provided
        if personality_summary:
            elements.append(
                mo.callout(mo.md(f"**Personality**: {personality_summary}"), kind="info")
            )

        return mo.vstack(elements)

    @staticmethod
    def build_performance_accordion(
        timeline_data: List[Dict[str, Any]],
        breakdown_data: List[Dict[str, Any]],
        round_data: pl.DataFrame,
        role_name: str,
    ) -> mo.Html:
        """Build the Performance Overview accordion section.

        Args:
            timeline_data: List of RoleTimeline dicts
            breakdown_data: List of RoleGameBreakdown dicts
            round_data: DataFrame with round-level data
            role_name: Role name for chart titles

        Returns:
            Accordion with performance sections
        """
        from ..visualization.role_charts import (
            create_session_timeline,
            create_game_breakdown_bars,
            create_learning_curve_multi_session,
            create_cumulative_reward_chart,
        )

        sections = {}

        # Session Timeline
        if timeline_data:
            timeline_chart = create_session_timeline(
                [asdict(t) if hasattr(t, '__dataclass_fields__') else t for t in timeline_data],
                role_name,
                metric="avg_payoff",
            )
            sections["Session Timeline"] = mo.vstack([
                mo.md("_Performance trend across sessions_"),
                timeline_chart,
            ])

        # Game-by-Game Breakdown
        if breakdown_data:
            breakdown_chart = create_game_breakdown_bars(
                [asdict(b) if hasattr(b, '__dataclass_fields__') else b for b in breakdown_data],
                role_name,
                metric="avg_payoff",
            )
            sections["Game Performance"] = mo.vstack([
                mo.md("_Average payoff by game type_"),
                breakdown_chart,
            ])

        # Learning Curve
        if not round_data.is_empty():
            learning_chart = create_learning_curve_multi_session(
                round_data, role_name
            )
            cumulative_chart = create_cumulative_reward_chart(
                round_data, role_name, by_game=True
            )
            sections["Learning & Rewards"] = mo.vstack([
                mo.md("_Learning progression and cumulative rewards_"),
                mo.hstack([learning_chart, cumulative_chart], justify="start", wrap=True, gap=1.0),
            ])

        if not sections:
            return mo.callout(
                mo.md("No performance data available."),
                kind="neutral",
            )

        return mo.accordion(sections, multiple=True)

    @staticmethod
    def build_behavioral_accordion(
        results_df: pl.DataFrame,
        results_list: List[Dict[str, Any]],
        player_models: Dict[int, str],
        num_players: int = 2,
    ) -> mo.Html:
        """Build the Behavioral Analysis accordion section.

        Reuses existing section builders from RoleAnalyticsPanelBuilder.

        Args:
            results_df: DataFrame with game results
            results_list: List of round result dicts
            player_models: Mapping of player number to model name
            num_players: Number of players

        Returns:
            Accordion with behavioral sections
        """
        sections = {}

        # Strategy Detection (reuse existing)
        if not results_df.is_empty():
            strategy_section = RoleAnalyticsPanelBuilder.build_strategy_section(
                results_df, player_models
            )
            sections["Strategy Detection"] = strategy_section

        # Learning Curves (reuse existing)
        if not results_df.is_empty():
            learning_section = RoleAnalyticsPanelBuilder.build_learning_curve_section(
                results_df, player_models
            )
            sections["Session Learning Curves"] = learning_section

        # Meta-Learning (reuse existing)
        if results_list and len(results_list) >= 5:
            meta_section = RoleAnalyticsPanelBuilder.build_meta_learning_section(
                results_list, num_players
            )
            sections["Meta-Learning Analysis"] = meta_section

        if not sections:
            return mo.callout(
                mo.md("No behavioral data available for analysis."),
                kind="neutral",
            )

        return mo.accordion(sections, multiple=True)

    @staticmethod
    def build_personality_accordion(
        profiler: Optional['ModelPersonalityProfiler'],
        model: str,
        round_data: pl.DataFrame,
        role_name: str,
        data_sufficiency: Optional['DataSufficiency'] = None,
    ) -> mo.Html:
        """Build the Personality Profile accordion section.

        Args:
            profiler: ModelPersonalityProfiler instance (optional)
            model: Model name to profile
            round_data: DataFrame with action data
            role_name: Role name for chart titles
            data_sufficiency: Optional data sufficiency check results

        Returns:
            Accordion with personality sections
        """
        from ..visualization.role_charts import (
            create_personality_radar,
            create_bias_gauge,
            create_symmetry_gauge,
            create_adaptability_gauge,
            create_risk_tolerance_gauge,
            create_temporal_pattern_chart,
            create_action_frequency_bars,
            create_fingerprint_matrix,
        )
        import logging

        sections = {}

        # Check data sufficiency first
        if data_sufficiency and not data_sufficiency.has_personality_metrics:
            reasons = data_sufficiency.insufficient_reasons
            return mo.callout(
                mo.md(
                    f"**Insufficient data for personality analysis**\n\n"
                    + "\n".join(f"- {r}" for r in reasons)
                ),
                kind="neutral",
            )

        # Personality Profile from profiler
        if profiler:
            try:
                summary = profiler.summarize(model)
                profile = summary.get("profile", {})

                # Build radar chart with enhanced traits
                traits = {
                    "Bias": (profile.get("bias_score", 0) + 1) / 2,  # Normalize -1..1 to 0..1
                    "Symmetry": profile.get("symmetry_breaking_score", 0),
                    "Consistency": profile.get("consistency_score", 0),
                    "Adaptability": profile.get("adaptability_score", 0),
                    "Risk": profile.get("risk_tolerance", 0.5),
                }
                radar_chart = create_personality_radar(traits, role_name)

                # Gauge charts - original pair
                bias_gauge = create_bias_gauge(profile.get("bias_score", 0))
                symmetry_gauge = create_symmetry_gauge(profile.get("symmetry_breaking_score", 0))

                # NEW: Enhanced gauge charts
                adaptability_gauge = create_adaptability_gauge(profile.get("adaptability_score", 0))
                risk_gauge = create_risk_tolerance_gauge(profile.get("risk_tolerance", 0.5))

                # Build profile section content
                profile_elements = [
                    mo.md(f"_{summary.get('summary', 'No summary available')}_"),
                    radar_chart,
                    mo.md("**Core Metrics**"),
                    mo.hstack([bias_gauge, symmetry_gauge], justify="start", wrap=True, gap=0.5),
                    mo.md("**Behavioral Metrics**"),
                    mo.hstack([adaptability_gauge, risk_gauge], justify="start", wrap=True, gap=0.5),
                ]

                # Add temporal pattern visualization if data exists
                early = profile.get("early_game_concentration", 0)
                late = profile.get("late_game_concentration", 0)
                temporal = profile.get("temporal_pattern", "consistent")

                if early > 0 or late > 0:
                    temporal_chart = create_temporal_pattern_chart(early, late, role_name)
                    profile_elements.append(mo.md("**Temporal Pattern**"))
                    profile_elements.append(temporal_chart)
                    if temporal != "consistent":
                        profile_elements.append(
                            mo.callout(
                                mo.md(f"Pattern: {temporal.replace('_', ' ').title()}"),
                                kind="info",
                            )
                        )

                sections["Personality Profile"] = mo.vstack(profile_elements)

                # Fingerprint matrix
                fingerprints = summary.get("game_fingerprints", {})
                if fingerprints:
                    fingerprint_chart = create_fingerprint_matrix(fingerprints, role_name)
                    sections["Behavior Fingerprint"] = mo.vstack([
                        mo.md("_Game-specific behavior patterns (HHI/entropy use absolute 0-1 scale)_"),
                        fingerprint_chart,
                    ])

            except Exception as e:
                # Log the error for debugging
                logging.warning(f"Personality profiler failed for {model}: {e}")
                sections["Personality Profile"] = mo.callout(
                    mo.md(f"Could not generate personality profile: {str(e)}"),
                    kind="warn",
                )

        # Action frequency from round data (per game type)
        if not round_data.is_empty() and "action" in round_data.columns:
            action_chart = create_action_frequency_bars(round_data, role_name, by_game=True)
            sections["Action Distribution"] = mo.vstack([
                mo.md("_How often each action was chosen per game type_"),
                action_chart,
            ])

        if not sections:
            return mo.callout(
                mo.md("Insufficient data for personality analysis. Need more sessions across multiple games."),
                kind="neutral",
            )

        return mo.accordion(sections, multiple=True)

    @staticmethod
    def build_sessions_accordion(
        sessions: List[Dict[str, Any]],
        session_manager: 'SessionManager',
        max_sessions: int = 20,
    ) -> mo.Html:
        """Build the Session Details accordion with drilldown capability.

        Args:
            sessions: List of session metadata dicts
            session_manager: SessionManager for loading session data
            max_sessions: Maximum sessions to show

        Returns:
            Accordion with session details
        """
        if not sessions:
            return mo.callout(
                mo.md("No sessions available."),
                kind="neutral",
            )

        session_items = {}
        for session in sessions[:max_sessions]:
            session_id = session.get("session_id", "")
            game_type = session.get("game_type", "unknown")
            num_rounds = session.get("num_rounds", 0)
            timestamp = session.get("timestamp", "")[:10] if session.get("timestamp") else ""

            label = f"{session_id[:8]} - {game_type} ({num_rounds} rounds)"
            if timestamp:
                label += f" - {timestamp}"

            # Session detail content
            detail_content = mo.vstack([
                mo.md(f"**Session ID**: {session_id}"),
                mo.md(f"**Game Type**: {game_type}"),
                mo.md(f"**Rounds**: {num_rounds}"),
                mo.md(f"**Date**: {timestamp or 'Unknown'}"),
            ])

            session_items[label] = detail_content

        return mo.accordion(session_items, multiple=False)

    @staticmethod
    def build_complete_panel(
        role: RoleConfig,
        stats: 'RoleStatistics',
        timeline: List[Any],
        breakdown: List[Any],
        round_data: pl.DataFrame,
        sessions: List[Dict[str, Any]],
        profiler: Optional['ModelPersonalityProfiler'] = None,
        session_manager: Optional['SessionManager'] = None,
        results_df: Optional[pl.DataFrame] = None,
        results_list: Optional[List[Dict[str, Any]]] = None,
    ) -> mo.Html:
        """Build the complete role analytics panel with all sections.

        Args:
            role: The selected RoleConfig
            stats: RoleStatistics from RoleAnalyticsService
            timeline: List of RoleTimeline entries
            breakdown: List of RoleGameBreakdown entries
            round_data: DataFrame with round-level data
            sessions: List of session metadata
            profiler: Optional ModelPersonalityProfiler
            session_manager: Optional SessionManager for drilldown
            results_df: Optional DataFrame for behavioral analysis
            results_list: Optional list of results for meta-learning

        Returns:
            Complete panel with summary and accordions
        """
        # Convert dataclass objects to dicts for visualization
        timeline_dicts = [
            asdict(t) if hasattr(t, '__dataclass_fields__') else t
            for t in timeline
        ]
        breakdown_dicts = [
            asdict(b) if hasattr(b, '__dataclass_fields__') else b
            for b in breakdown
        ]

        elements = []

        # Summary (always visible)
        summary = ImprovedRoleAnalyticsPanelBuilder.build_summary_section(
            role, stats
        )
        elements.append(summary)

        # Performance Overview accordion
        elements.append(mo.md("---"))
        elements.append(mo.md("### Performance Overview"))
        performance = ImprovedRoleAnalyticsPanelBuilder.build_performance_accordion(
            timeline_dicts, breakdown_dicts, round_data, role.name
        )
        elements.append(performance)

        # Behavioral Analysis accordion
        if results_df is not None and not results_df.is_empty():
            elements.append(mo.md("---"))
            elements.append(mo.md("### Behavioral Analysis"))
            player_models = {1: role.model}  # Simplified for single role view
            behavioral = ImprovedRoleAnalyticsPanelBuilder.build_behavioral_accordion(
                results_df, results_list or [], player_models
            )
            elements.append(behavioral)

        # Personality Profile accordion
        elements.append(mo.md("---"))
        elements.append(mo.md("### Personality Profile"))
        personality = ImprovedRoleAnalyticsPanelBuilder.build_personality_accordion(
            profiler, role.model, round_data, role.name
        )
        elements.append(personality)

        # Session Details accordion
        elements.append(mo.md("---"))
        elements.append(mo.md("### Session Details"))
        session_details = ImprovedRoleAnalyticsPanelBuilder.build_sessions_accordion(
            sessions, session_manager
        )
        elements.append(session_details)

        return mo.vstack(elements)


# =============================================================================
# LEGACY COMPATIBILITY - Original class preserved below
# =============================================================================


class RoleAnalyticsPanelBuilder:
    """Static factory methods for building Role Analytics UI sections.

    Role Analytics focuses on player/model behavior analysis:
    - Strategy detection (TFT, defection, etc.)
    - Learning curves and improvement
    - Model personality profiling
    - Meta-learning and adaptation
    """

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

        elements.append(mo.hstack(strategy_cards, justify="start", wrap=True))

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

        elements.append(mo.hstack(learning_cards, justify="start", wrap=True))

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

        elements.append(mo.hstack(stats, justify="start", wrap=True))

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
        """Build meta-strategy learning analysis section with mini-charts.

        Args:
            results: List of round result dictionaries
            num_players: Number of players

        Returns:
            Marimo vstack with meta-learning analysis and visualizations
        """
        import altair as alt
        import polars as pl
        from ..analytics.meta_learning import MetaStrategyAnalyzer

        analyzer = MetaStrategyAnalyzer(results, num_players)
        summary = analyzer.summarize_all_players()

        elements = [mo.md("### Meta-Strategy Learning Analysis")]
        elements.append(mo.md("_Detecting adaptation, learning, and opponent modeling over rounds._"))

        if not summary.get("players"):
            elements.append(mo.callout(
                mo.md("Insufficient data for meta-learning analysis."),
                kind="neutral",
            ))
            return mo.vstack(elements)

        # Build enhanced player cards with mini-charts
        player_sections = []

        for player_num, data in summary.get("players", {}).items():
            adaptation = data.get("adaptation", {})
            memory = data.get("memory_effect", {})
            learning = data.get("learning", {})
            counter = data.get("counter_strategy", {})
            shift = data.get("distribution_shift", {})

            model_name = data.get('model', f'Player {player_num}')

            # Determine card kind
            if adaptation.get("detected"):
                kind = "success"
            elif learning.get("trend") == "improving":
                kind = "info"
            elif learning.get("trend") == "declining":
                kind = "danger"
            else:
                kind = "neutral"

            # Stats row
            stats_row = mo.hstack([
                mo.stat(
                    label="Adaptation",
                    value=adaptation.get('type', 'none').title(),
                    caption="Detected" if adaptation.get('detected') else "None",
                ),
                mo.stat(
                    label="Memory",
                    value=f"r={memory.get('correlation', 0):.2f}",
                    caption=memory.get('direction', 'none').title(),
                ),
                mo.stat(
                    label="Learning",
                    value=learning.get('trend', 'stable').title(),
                    caption=f"rate={learning.get('rate', 0):.3f}",
                ),
                mo.stat(
                    label="Stability",
                    value=f"{adaptation.get('stability', 0):.0%}",
                    caption="Strategy consistency",
                ),
            ], justify="start", wrap=True)

            # Mini-charts row
            charts_row_items = []

            # Learning trajectory mini-chart
            trajectory = learning.get('trajectory', [])
            if len(trajectory) >= 2:
                traj_data = pl.DataFrame({
                    "window": list(range(1, len(trajectory) + 1)),
                    "win_rate": trajectory,
                })

                # Trend line color based on direction
                line_color = (
                    "green" if learning.get('trend') == "improving"
                    else "red" if learning.get('trend') == "declining"
                    else "steelblue"
                )

                learning_chart = (
                    alt.Chart(traj_data.to_pandas())
                    .mark_line(point=True, color=line_color)
                    .encode(
                        x=alt.X("window:O", title="Window", axis=alt.Axis(labelAngle=0)),
                        y=alt.Y("win_rate:Q", title="Win Rate", scale=alt.Scale(domain=[0, 1])),
                        tooltip=[
                            alt.Tooltip("window:O", title="Window"),
                            alt.Tooltip("win_rate:Q", title="Win Rate", format=".1%"),
                        ],
                    )
                    .properties(width=180, height=100, title="Learning Curve")
                )
                charts_row_items.append(learning_chart)

            # Memory correlation visual bar
            mem_corr = memory.get('correlation', 0)
            mem_data = pl.DataFrame({
                "type": ["Memory Correlation"],
                "value": [mem_corr],
                "abs_value": [abs(mem_corr)],
            })

            # Color: green for positive (mimic), red for negative (counter), gray for neutral
            mem_color = (
                "green" if mem_corr > 0.3
                else "red" if mem_corr < -0.3
                else "gray"
            )

            memory_chart = (
                alt.Chart(mem_data.to_pandas())
                .mark_bar(color=mem_color)
                .encode(
                    x=alt.X("value:Q", title="Correlation", scale=alt.Scale(domain=[-1, 1])),
                    tooltip=[
                        alt.Tooltip("value:Q", title="r", format=".2f"),
                    ],
                )
                .properties(width=150, height=40, title="Opponent Memory")
            )
            charts_row_items.append(memory_chart)

            # Counter-strategy indicator
            counter_score = counter.get('score', 0)
            if abs(counter_score) > 0.1:
                counter_data = pl.DataFrame({
                    "type": ["Counter Score"],
                    "value": [counter_score],
                })
                counter_chart = (
                    alt.Chart(counter_data.to_pandas())
                    .mark_bar(color="purple" if counter_score > 0 else "orange")
                    .encode(
                        x=alt.X("value:Q", title="Score", scale=alt.Scale(domain=[-1, 1])),
                    )
                    .properties(width=100, height=40, title="Counter-Strategy")
                )
                charts_row_items.append(counter_chart)

            # Build player section
            player_content = mo.vstack([
                mo.md(f"#### {model_name}"),
                stats_row,
                mo.hstack(charts_row_items, justify="start", wrap=True, gap=0.75) if charts_row_items else mo.md(""),
            ])

            player_sections.append(mo.callout(player_content, kind=kind))

        # Arrange player sections
        for section in player_sections:
            elements.append(section)

        # Add summary insights
        num_adapters = sum(
            1 for d in summary.get("players", {}).values()
            if d.get("adaptation", {}).get("detected")
        )
        num_improvers = sum(
            1 for d in summary.get("players", {}).values()
            if d.get("learning", {}).get("trend") == "improving"
        )

        insight_parts = []
        if num_adapters > 0:
            insight_parts.append(f"{num_adapters} player(s) showing adaptation behavior")
        if num_improvers > 0:
            insight_parts.append(f"{num_improvers} player(s) improving over time")

        if insight_parts:
            elements.append(mo.callout(
                mo.md("**Summary**: " + ", ".join(insight_parts)),
                kind="info",
            ))

        return mo.vstack(elements)

    @staticmethod
    def build_archetype_section(
        results: List[Dict[str, Any]],
        player_num: int,
        budget: float = 100.0
    ) -> mo.Html:
        """Build strategy archetype clustering analysis section.

        Uses dimensionality reduction (PCA) and K-Means clustering to identify
        strategy archetypes from allocation patterns.

        Args:
            results: List of round result dictionaries
            player_num: Player number to analyze
            budget: Allocation budget for normalization

        Returns:
            Marimo vstack with archetype analysis and visualizations
        """
        from ..analytics.clustering import StrategyClusterer, find_optimal_clusters
        from ..analytics.allocation import AllocationAnalyzer
        from ..visualization.allocation_charts import (
            create_strategy_cluster_scatter,
            create_strategy_type_pie,
        )

        elements = [mo.md(f"### Strategy Archetypes (Player {player_num})")]
        elements.append(mo.md("_Clustering analysis to identify distinct strategic patterns._"))

        # Extract allocations for this player
        analyzer = AllocationAnalyzer()
        allocations = []
        metrics_list = []

        for result in results:
            allocation = analyzer.parse_allocation_from_result(result, player_num)
            if allocation:
                allocations.append(allocation)
                metrics = analyzer.analyze_allocation(allocation, budget)
                metrics_list.append(metrics)

        if len(allocations) < 4:
            elements.append(mo.callout(
                mo.md(f"Need at least 4 rounds for clustering analysis. Found {len(allocations)} rounds."),
                kind="neutral",
            ))
            return mo.vstack(elements)

        # Perform clustering analysis
        clusterer = StrategyClusterer()
        features = clusterer.extract_features(allocations, budget)

        if features.size == 0:
            elements.append(mo.callout(
                mo.md("Could not extract features for clustering."),
                kind="warn",
            ))
            return mo.vstack(elements)

        # Find optimal number of clusters
        optimal_k = find_optimal_clusters(features, max_clusters=min(6, len(allocations) // 2))
        cluster_result = clusterer.analyze_allocations(
            allocations, budget, n_clusters=optimal_k, dim_reduction='pca'
        )

        # Get archetype summaries
        archetypes = clusterer.get_archetype_summaries(allocations, cluster_result, budget)

        # Summary stats
        stats = [
            mo.stat(
                label="Clusters Found",
                value=str(len(set(cluster_result.cluster_labels))),
                caption=f"Silhouette: {cluster_result.silhouette_score:.2f}",
            ),
            mo.stat(
                label="Total Rounds",
                value=str(len(allocations)),
            ),
        ]
        elements.append(mo.hstack(stats, justify="start", wrap=True))

        # Create visualizations side by side
        charts = []

        # Strategy type pie chart
        if metrics_list:
            pie_chart = create_strategy_type_pie(metrics_list)
            charts.append(pie_chart)

        # Cluster scatter plot
        if cluster_result.features_2d.size > 0:
            # Create labels from archetype names
            labels = [
                cluster_result.archetype_names[label] if label < len(cluster_result.archetype_names) else f"Cluster {label}"
                for label in cluster_result.cluster_labels
            ]

            # Add metadata for tooltips
            metadata = [
                {"round": i + 1, "hhi": f"{metrics_list[i].concentration_index:.2f}"}
                for i in range(len(allocations))
            ]

            scatter_chart = create_strategy_cluster_scatter(
                cluster_result.features_2d.tolist(),
                labels,
                metadata
            )
            charts.append(scatter_chart)

        if charts:
            elements.append(mo.hstack(charts, justify="start", wrap=True))

        # Archetype descriptions
        if archetypes:
            archetype_items = []
            colors = {
                "concentrated": "danger",
                "uniform": "success",
                "spread": "success",
                "hedged": "info",
                "front-heavy": "warn",
                "back-heavy": "warn",
            }

            for arch in archetypes:
                # Determine color based on name
                kind = "neutral"
                name_lower = arch.name.lower()
                for pattern, color in colors.items():
                    if pattern in name_lower:
                        kind = color
                        break

                arch_content = mo.vstack([
                    mo.md(f"**{arch.name}**"),
                    mo.md(f"_{arch.description}_"),
                    mo.md(f"Rounds: {arch.count} | Avg HHI: {arch.avg_concentration:.2f}"),
                ])
                archetype_items.append(mo.callout(arch_content, kind=kind))

            elements.append(mo.md("#### Identified Archetypes"))
            elements.append(mo.hstack(archetype_items, justify="start", wrap=True))

        return mo.vstack(elements)
