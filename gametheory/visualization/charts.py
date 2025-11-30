"""Altair chart factories for game visualization."""

from typing import List, Dict, Any, Optional
import altair as alt
import polars as pl


def create_cumulative_payoff_chart(
    results_df: pl.DataFrame,
    player_num: int = 1,
    color: str = "blue",
    width: int = 600,
    height: int = 300,
) -> alt.Chart:
    """Generate cumulative payoff line chart for a player.

    Args:
        results_df: DataFrame with game results.
        player_num: Player number (1 or 2).
        color: Line color.
        width: Chart width.
        height: Chart height.

    Returns:
        An Altair chart object.
    """
    payoff_col = f"player{player_num}_payoff"
    cumulative_col = f"cumulative_payoff_player{player_num}"
    action_col = f"player{player_num}_action"

    # Calculate cumulative payoffs if not already present
    if cumulative_col not in results_df.columns:
        results_df = results_df.with_columns([
            pl.col(payoff_col).cum_sum().alias(cumulative_col)
        ])

    return (
        alt.Chart(results_df.to_pandas())
        .mark_line(point=True)
        .encode(
            x=alt.X("game_number:O", title="Game Number"),
            y=alt.Y(f"{cumulative_col}:Q", title="Cumulative Payoff"),
            color=alt.value(color),
            tooltip=["game_number", cumulative_col, action_col],
        )
        .properties(
            title=f"Player {player_num} Cumulative Payoff",
            width=width,
            height=height,
        )
    )


def create_action_distribution_chart(
    results_df: pl.DataFrame,
    width: int = 200,
    height: int = 200,
) -> alt.Chart:
    """Generate action combination frequency chart.

    Args:
        results_df: DataFrame with game results.
        width: Chart width per facet.
        height: Chart height.

    Returns:
        An Altair chart object.
    """
    # Aggregate action counts
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


def create_payoff_comparison_chart(
    results_df: pl.DataFrame,
    width: int = 400,
    height: int = 300,
) -> alt.Chart:
    """Generate total payoff comparison bar chart.

    Args:
        results_df: DataFrame with game results.
        width: Chart width.
        height: Chart height.

    Returns:
        An Altair chart object.
    """
    # Aggregate payoffs
    summary = (
        results_df
        .select([
            pl.col("player1_payoff").sum().alias("Player 1"),
            pl.col("player2_payoff").sum().alias("Player 2"),
        ])
        .to_pandas()
        .melt(var_name="Player", value_name="Total Payoff")
    )

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
    width: int = 400,
    height: int = 300,
) -> alt.Chart:
    """Generate average payoff comparison bar chart.

    Args:
        results_df: DataFrame with game results.
        width: Chart width.
        height: Chart height.

    Returns:
        An Altair chart object.
    """
    # Aggregate payoffs
    summary = (
        results_df
        .select([
            pl.col("player1_payoff").mean().alias("Player 1"),
            pl.col("player2_payoff").mean().alias("Player 2"),
        ])
        .to_pandas()
        .melt(var_name="Player", value_name="Avg Payoff")
    )

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
