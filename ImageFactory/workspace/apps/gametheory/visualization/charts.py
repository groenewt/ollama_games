"""Altair chart factories for game visualization."""

from typing import List, Dict, Any, Optional
import altair as alt
import pandas as pd
import polars as pl

from ..core.utils import detect_num_players


def _get_num_players(df: pl.DataFrame) -> int:
    """Detect number of players from DataFrame columns (cached)."""
    return detect_num_players(tuple(df.columns))


def _empty_chart_placeholder(message: str, width: int, height: int) -> alt.Chart:
    """Create placeholder chart for empty/invalid data.

    Args:
        message: Message to display
        width: Chart width
        height: Chart height

    Returns:
        Altair text chart with message
    """
    return alt.Chart(pd.DataFrame({"text": [message]})).mark_text(
        fontSize=14, color="#888"
    ).encode(text="text:N").properties(width=width, height=height, title="")


def create_cumulative_payoff_chart(
    results_df: pl.DataFrame,
    player_num: int = 1,
    color: str = "blue",
    width: int = 600,
    height: int = 300,
    per_session: bool = False,
) -> alt.Chart:
    """Generate cumulative payoff line chart for a player.

    Args:
        results_df: DataFrame with game results.
        player_num: Player number (1 or 2).
        color: Line color.
        width: Chart width.
        height: Chart height.
        per_session: If True, reset cumulative at session boundaries.

    Returns:
        An Altair chart object.
    """
    # Validate inputs
    if results_df is None or results_df.is_empty():
        return _empty_chart_placeholder("No game results data", width, height)

    payoff_col = f"player{player_num}_payoff"
    cumulative_col = f"cumulative_payoff_player{player_num}"
    action_col = f"player{player_num}_action"

    # Validate required column exists
    if payoff_col not in results_df.columns:
        return _empty_chart_placeholder(f"Missing column: {payoff_col}", width, height)

    # Add game_number if missing
    if "game_number" not in results_df.columns:
        results_df = results_df.with_row_index("game_number")
        results_df = results_df.with_columns(pl.col("game_number") + 1)  # 1-indexed
    else:
        # Sort by game_number to ensure correct cumulative order
        results_df = results_df.sort("game_number")

    # Calculate cumulative payoffs if not already present
    if cumulative_col not in results_df.columns:
        if per_session and "session_id" in results_df.columns:
            # Reset cumulative at session boundaries
            results_df = results_df.with_columns([
                pl.col(payoff_col).cum_sum().over("session_id").alias(cumulative_col)
            ])
        else:
            results_df = results_df.with_columns([
                pl.col(payoff_col).cum_sum().alias(cumulative_col)
            ])

    # Build tooltip columns - only include columns that exist
    tooltip_cols = ["game_number", cumulative_col]
    if action_col in results_df.columns:
        tooltip_cols.append(action_col)

    return (
        alt.Chart(results_df.to_pandas())
        .mark_line(point=True)
        .encode(
            x=alt.X("game_number:O", title="Game Number"),
            y=alt.Y(f"{cumulative_col}:Q", title="Cumulative Payoff"),
            color=alt.value(color),
            tooltip=tooltip_cols,
        )
        .properties(
            title=f"Player {player_num} Cumulative Payoff",
            width=width,
            height=height,
        )
    )


def create_action_distribution_chart(
    results_df: pl.DataFrame,
    num_players: Optional[int] = None,
    width: int = 200,
    height: int = 200,
) -> alt.Chart:
    """Generate action combination frequency chart.

    Args:
        results_df: DataFrame with game results.
        num_players: Number of players (auto-detected if None).
        width: Chart width per facet.
        height: Chart height.

    Returns:
        An Altair chart object.
    """
    # Validate inputs
    if results_df is None or results_df.is_empty():
        return _empty_chart_placeholder("No game results data", width, height)

    if num_players is None:
        num_players = _get_num_players(results_df)

    if num_players == 2:
        # 2-player: show combined action distribution as faceted chart
        action_counts = (
            results_df
            .group_by(["player1_action", "player2_action"])
            .agg(pl.len().alias("count"))
            .to_pandas()
        )

        return (
            alt.Chart(action_counts)
            .mark_bar()
            .encode(
                x=alt.X("player1_action:N", title="Player 1 Action"),
                y=alt.Y("count:Q", title="Frequency"),
                color="player2_action:N",
                column="player2_action:N",
                tooltip=["player1_action", "player2_action", "count"],
            )
            .properties(
                title="Action Combinations Distribution",
                width=width,
                height=height,
            )
        )
    else:
        # N-player: show individual player action distributions side by side
        charts = []
        colors = ["steelblue", "coral", "seagreen", "purple", "orange", "brown"]

        for p in range(1, num_players + 1):
            action_col = f"player{p}_action"
            if action_col not in results_df.columns:
                continue

            player_counts = (
                results_df
                .group_by(action_col)
                .agg(pl.len().alias("count"))
                .to_pandas()
            )

            color = colors[(p - 1) % len(colors)]
            chart = (
                alt.Chart(player_counts)
                .mark_bar(color=color)
                .encode(
                    x=alt.X(f"{action_col}:N", title="Action"),
                    y=alt.Y("count:Q", title="Frequency"),
                    tooltip=[action_col, "count"],
                )
                .properties(title=f"Player {p} Actions", width=width, height=height)
            )
            charts.append(chart)

        if not charts:
            return alt.Chart().mark_text().encode(text=alt.value("No action data"))

        return alt.hconcat(*charts).properties(title="Action Distributions by Player")


def create_payoff_comparison_chart(
    results_df: pl.DataFrame,
    num_players: Optional[int] = None,
    width: int = 400,
    height: int = 300,
) -> alt.Chart:
    """Generate total payoff comparison bar chart.

    Args:
        results_df: DataFrame with game results.
        num_players: Number of players (auto-detected if None).
        width: Chart width.
        height: Chart height.

    Returns:
        An Altair chart object.
    """
    if num_players is None:
        num_players = _get_num_players(results_df)

    # Dynamically build summary for N players
    summary_data = []
    for p in range(1, num_players + 1):
        col = f"player{p}_payoff"
        if col in results_df.columns:
            summary_data.append({
                "Player": f"Player {p}",
                "Total Payoff": results_df[col].sum()
            })

    if not summary_data:
        return alt.Chart().mark_text().encode(text=alt.value("No payoff data"))

    summary = pl.DataFrame(summary_data).to_pandas()

    return (
        alt.Chart(summary)
        .mark_bar()
        .encode(
            x=alt.X("Player:N", title="Player"),
            y=alt.Y("Total Payoff:Q", title="Total Payoff"),
            color=alt.Color("Player:N", legend=alt.Legend(title="Player")),
            tooltip=["Player", "Total Payoff"],
        )
        .properties(
            title="Total Payoffs Comparison",
            width=width,
            height=height,
        )
    )


def create_avg_payoff_chart(
    results_df: pl.DataFrame,
    num_players: Optional[int] = None,
    width: int = 400,
    height: int = 300,
) -> alt.Chart:
    """Generate average payoff comparison bar chart.

    Args:
        results_df: DataFrame with game results.
        num_players: Number of players (auto-detected if None).
        width: Chart width.
        height: Chart height.

    Returns:
        An Altair chart object.
    """
    if num_players is None:
        num_players = _get_num_players(results_df)

    # Dynamically build summary for N players
    summary_data = []
    for p in range(1, num_players + 1):
        col = f"player{p}_payoff"
        if col in results_df.columns:
            summary_data.append({
                "Player": f"Player {p}",
                "Avg Payoff": results_df[col].mean()
            })

    if not summary_data:
        return alt.Chart().mark_text().encode(text=alt.value("No payoff data"))

    summary = pl.DataFrame(summary_data).to_pandas()

    return (
        alt.Chart(summary)
        .mark_bar()
        .encode(
            x=alt.X("Player:N", title="Player"),
            y=alt.Y("Avg Payoff:Q", title="Average Payoff"),
            color=alt.Color("Player:N", legend=alt.Legend(title="Player")),
            tooltip=["Player", "Avg Payoff"],
        )
        .properties(
            title="Average Payoffs Comparison",
            width=width,
            height=height,
        )
    )


def create_response_time_chart(
    response_times: List[float],
    width: int = 600,
    height: int = 200,
) -> alt.Chart:
    """Generate response time line chart.

    Args:
        response_times: List of response times.
        width: Chart width.
        height: Chart height.

    Returns:
        An Altair chart object.
    """
    df = pl.DataFrame({
        "request": list(range(1, len(response_times) + 1)),
        "response_time": response_times,
    }).to_pandas()

    return (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("request:O", title="Request #"),
            y=alt.Y("response_time:Q", title="Response Time (s)"),
            tooltip=["request", "response_time"],
        )
        .properties(
            title="Response Times Over Time",
            width=width,
            height=height,
        )
    )


def create_model_comparison_heatmap(
    aggregated_df: pl.DataFrame,
    metric: str = "avg_payoff",
    width: int = 400,
    height: int = 300,
) -> alt.Chart:
    """Generate heatmap comparing models across games.

    Args:
        aggregated_df: DataFrame with model/game aggregated stats.
        metric: Column name for the metric to display.
        width: Chart width.
        height: Chart height.

    Returns:
        An Altair chart object.
    """
    return (
        alt.Chart(aggregated_df.to_pandas())
        .mark_rect()
        .encode(
            x=alt.X("game_type:N", title="Game Type"),
            y=alt.Y("model:N", title="Model"),
            color=alt.Color(f"{metric}:Q", title=metric.replace("_", " ").title()),
            tooltip=["model", "game_type", metric],
        )
        .properties(
            title=f"Model Performance: {metric.replace('_', ' ').title()}",
            width=width,
            height=height,
        )
    )


def create_leaderboard_chart(
    rankings_df: pl.DataFrame,
    metric: str = "weighted_score",
    width: int = 600,
    height: int = 300,
) -> alt.Chart:
    """Generate horizontal bar chart for model rankings.

    Args:
        rankings_df: DataFrame with model rankings.
        metric: Column name for the ranking metric.
        width: Chart width.
        height: Chart height.

    Returns:
        An Altair chart object.
    """
    return (
        alt.Chart(rankings_df.to_pandas())
        .mark_bar()
        .encode(
            x=alt.X(f"{metric}:Q", title=metric.replace("_", " ").title()),
            y=alt.Y("model:N", sort="-x", title="Model"),
            color=alt.Color("model:N", legend=None),
            tooltip=["model", metric, "total_plays"],
        )
        .properties(
            title="Model Leaderboard",
            width=width,
            height=height,
        )
    )


def create_cooperation_rate_chart(
    coop_df: pl.DataFrame,
    width: int = 500,
    height: int = 300,
) -> alt.Chart:
    """Generate cooperation rate chart by model and game.

    Args:
        coop_df: DataFrame with cooperation rates.
        width: Chart width.
        height: Chart height.

    Returns:
        An Altair chart object.
    """
    return (
        alt.Chart(coop_df.to_pandas())
        .mark_bar()
        .encode(
            x=alt.X("model:N", title="Model"),
            y=alt.Y("cooperation_rate:Q", title="Cooperation Rate (%)"),
            color=alt.Color("game_type:N", title="Game Type"),
            column=alt.Column("game_type:N", title="Game"),
            tooltip=["model", "game_type", "cooperation_rate", "total_decisions"],
        )
        .properties(
            title="Cooperation Rates by Model",
            width=width // 3,
            height=height,
        )
    )
