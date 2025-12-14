"""Visualization charts for role-centric analytics.

Provides Altair charts for analyzing role behavior across sessions and games.
"""

from typing import List, Dict, Optional, Any
import altair as alt
import polars as pl
import math


# Color schemes
TREND_COLORS = {
    "improving": "#2ecc71",
    "declining": "#e74c3c",
    "stable": "#95a5a6",
}

ROLE_PALETTE = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#f39c12", "#1abc9c"]

STRATEGY_COLORS = {
    "concentrated": "#e74c3c",
    "uniform": "#3498db",
    "asymmetric": "#9b59b6",
    "hedged": "#2ecc71",
    "unknown": "#95a5a6",
}


# =============================================================================
# RESPONSIVE SIZING UTILITIES
# =============================================================================


def calculate_adaptive_dimensions(
    data_count: int,
    num_categories: int = 1,
    base_width: int = 500,
    base_height: int = 250,
    max_width: int = 900,
    min_width: int = 300,
) -> tuple[int, int]:
    """Calculate adaptive chart dimensions based on data volume.

    Args:
        data_count: Number of data points
        num_categories: Number of categories (games, sessions, etc.)
        base_width: Default width for moderate data
        base_height: Default height
        max_width: Maximum allowed width
        min_width: Minimum allowed width

    Returns:
        Tuple of (width, height)
    """
    # Scale width with categories
    width = min(max_width, max(min_width, base_width + (num_categories - 2) * 80))

    # Scale height slightly with data volume for scatter/point charts
    if data_count > 500:
        height = min(350, base_height + 50)
    else:
        height = base_height

    return width, height


def calculate_facet_columns(num_facets: int, max_columns: int = 4) -> int:
    """Calculate optimal number of facet columns.

    Args:
        num_facets: Number of facet panels
        max_columns: Maximum columns allowed

    Returns:
        Optimal column count
    """
    if num_facets <= 2:
        return num_facets
    elif num_facets <= 4:
        return 2
    elif num_facets <= 6:
        return 3
    else:
        return max_columns


def calculate_adaptive_window_size(
    total_rounds: int,
    min_window: int = 3,
    max_window: int = 50
) -> int:
    """Calculate adaptive rolling window size based on total rounds.

    Args:
        total_rounds: Total number of rounds in data
        min_window: Minimum window size
        max_window: Maximum window size

    Returns:
        Optimal window size (roughly 5% of data, bounded)
    """
    # Use ~5% of data as window, bounded
    adaptive = max(min_window, min(max_window, total_rounds // 20))
    return adaptive


def get_x_axis_config(data_count: int) -> dict:
    """Get Altair x-axis configuration for high-volume data.

    Args:
        data_count: Number of data points

    Returns:
        Dict with axis parameters for clean x-axis display
    """
    if data_count <= 50:
        return {}  # Default behavior
    elif data_count <= 200:
        return {"labelOverlap": True, "tickCount": 20}
    elif data_count <= 500:
        return {"labelOverlap": True, "tickCount": 15}
    else:
        return {"labelOverlap": True, "tickCount": 10}


def aggregate_for_visualization(
    df: pl.DataFrame,
    x_col: str,
    y_cols: List[str],
    target_points: int = 200,
) -> pl.DataFrame:
    """Aggregate data for visualization when point count exceeds threshold.

    Args:
        df: Input DataFrame
        x_col: Column to use for x-axis (will be binned)
        y_cols: Columns to aggregate (mean)
        target_points: Target number of output points

    Returns:
        Aggregated DataFrame or original if small enough
    """
    if len(df) <= target_points:
        return df

    # Calculate bin size
    bin_size = max(1, len(df) // target_points)

    # Add bin column
    df = df.with_columns(
        (pl.col(x_col) // bin_size).alias("_bin")
    )

    # Build aggregation expressions
    agg_exprs = [
        pl.col(x_col).mean().alias(x_col),
    ]
    for col in y_cols:
        if col in df.columns:
            agg_exprs.append(pl.col(col).mean().alias(col))

    # Preserve categorical columns (first value per bin)
    for col in df.columns:
        if col not in [x_col, "_bin"] + y_cols:
            agg_exprs.append(pl.col(col).first().alias(col))

    return df.group_by("_bin").agg(agg_exprs).drop("_bin").sort(x_col)


def _empty_chart_placeholder(message: str, width: int, height: int) -> alt.Chart:
    """Create placeholder chart for insufficient data scenarios."""
    return (
        alt.Chart(pl.DataFrame([{"text": message}]).to_pandas())
        .mark_text(fontSize=14, color="#666", align="center", baseline="middle")
        .encode(text=alt.value(message))
        .properties(width=width, height=height, title="")
    )


# =============================================================================
# PRIORITY 1: Core Session/Role Charts
# =============================================================================


def create_session_timeline(
    timeline_data: List[Dict[str, Any]],
    role_name: str,
    metric: str = "avg_payoff",
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> alt.Chart:
    """Create timeline showing role's metrics over sessions.

    Args:
        timeline_data: List of RoleTimeline dicts (session_id, timestamp, avg_payoff, etc.)
        role_name: Name of the role for title
        metric: Which metric to plot (avg_payoff, win_rate, total_payoff)
        width: Chart width (adaptive if None)
        height: Chart height (adaptive if None)

    Returns:
        Altair layered chart with line + points
    """
    # Calculate adaptive dimensions
    num_sessions = len(timeline_data) if timeline_data else 0
    calc_width, calc_height = calculate_adaptive_dimensions(
        num_sessions,
        num_categories=1,
        base_width=500,
        base_height=250,
    )
    width = width if width is not None else calc_width
    height = height if height is not None else calc_height

    if not timeline_data:
        return _empty_chart_placeholder("No session data available", width, height)

    # Convert to DataFrame
    rows = []
    for i, entry in enumerate(timeline_data):
        rows.append({
            "session_num": i + 1,
            "session_id": entry.get("session_id", "")[:8],
            "game_type": entry.get("game_type", "unknown"),
            "avg_payoff": entry.get("avg_payoff", 0),
            "win_rate": entry.get("win_rate", 0),
            "total_payoff": entry.get("total_payoff", 0),
            "num_rounds": entry.get("num_rounds", 0),
            "opponents": ", ".join(entry.get("opponent_models", [])[:2]),
        })

    df = pl.DataFrame(rows)

    metric_titles = {
        "avg_payoff": "Average Payoff",
        "win_rate": "Win Rate (%)",
        "total_payoff": "Total Payoff",
    }

    # Base chart
    base = alt.Chart(df.to_pandas())

    # Line with points
    line = base.mark_line(point=True, color="#3498db").encode(
        x=alt.X("session_num:O", title="Session #", axis=alt.Axis(labelAngle=0)),
        y=alt.Y(f"{metric}:Q", title=metric_titles.get(metric, metric)),
        tooltip=[
            alt.Tooltip("session_id:N", title="Session"),
            alt.Tooltip("game_type:N", title="Game"),
            alt.Tooltip("avg_payoff:Q", title="Avg Payoff", format=".1f"),
            alt.Tooltip("win_rate:Q", title="Win Rate", format=".1f"),
            alt.Tooltip("num_rounds:Q", title="Rounds"),
            alt.Tooltip("opponents:N", title="Opponents"),
        ],
    )

    # Add trend line
    trend = base.transform_regression("session_num", metric).mark_line(
        strokeDash=[5, 5],
        color="#e74c3c",
        opacity=0.7,
    ).encode(
        x="session_num:O",
        y=f"{metric}:Q",
    )

    return (line + trend).properties(
        title=f"{role_name} - Performance Over Sessions",
        width=width,
        height=height,
    )


def create_learning_curve_multi_session(
    round_data: pl.DataFrame,
    role_name: str,
    window_size: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> alt.Chart:
    """Create learning curve visualization across all rounds.

    Args:
        round_data: DataFrame with round-level data (round_number, payoff, session_id)
        role_name: Role name for title
        window_size: Rolling average window (adaptive if None)
        width: Chart width (adaptive if None)
        height: Chart height (adaptive if None)

    Returns:
        Altair chart showing learning progression
    """
    # Calculate adaptive dimensions and window size
    total_rounds = len(round_data) if not round_data.is_empty() else 0
    calc_width, calc_height = calculate_adaptive_dimensions(
        total_rounds,
        base_width=500,
        base_height=250,
    )
    width = width if width is not None else calc_width
    height = height if height is not None else calc_height

    if round_data.is_empty():
        return _empty_chart_placeholder("No round data available", width, height)

    # Calculate cumulative round number across sessions
    df = round_data.sort(["session_id", "round_number"]).with_row_index("global_round")
    df = df.with_columns((pl.col("global_round") + 1).alias("global_round"))

    # Calculate adaptive window size
    if window_size is None:
        window_size = calculate_adaptive_window_size(total_rounds)

    # Calculate rolling average
    df = df.with_columns(
        pl.col("payoff").rolling_mean(window_size=window_size).alias("rolling_avg")
    )

    # Aggregate for visualization if too many points
    if len(df) > 500:
        df = aggregate_for_visualization(
            df,
            x_col="global_round",
            y_cols=["payoff", "rolling_avg"],
            target_points=300
        )

    # Determine trend
    if len(df) >= 2:
        first_half = df.head(len(df) // 2)["payoff"].mean()
        second_half = df.tail(len(df) // 2)["payoff"].mean()
        trend = "improving" if second_half > first_half * 1.05 else (
            "declining" if second_half < first_half * 0.95 else "stable"
        )
    else:
        trend = "stable"

    trend_color = TREND_COLORS.get(trend, "#95a5a6")

    base = alt.Chart(df.to_pandas())

    # Get x-axis config for high round counts
    x_axis_config = get_x_axis_config(total_rounds)
    x_axis = alt.Axis(**x_axis_config) if x_axis_config else alt.Axis()

    # Raw payoff points
    points = base.mark_point(opacity=0.3, size=20).encode(
        x=alt.X("global_round:Q", title="Round", axis=x_axis),
        y=alt.Y("payoff:Q", title="Payoff"),
        tooltip=[
            alt.Tooltip("global_round:Q", title="Round"),
            alt.Tooltip("payoff:Q", title="Payoff"),
            alt.Tooltip("session_id:N", title="Session"),
        ],
    )

    # Rolling average line
    rolling_line = base.mark_line(color=trend_color, strokeWidth=2).encode(
        x="global_round:Q",
        y=alt.Y("rolling_avg:Q", title="Payoff (rolling avg)"),
    )

    # Trend line
    trend_line = base.transform_regression("global_round", "payoff").mark_line(
        strokeDash=[5, 5],
        color=trend_color,
        opacity=0.5,
    ).encode(
        x="global_round:Q",
        y="payoff:Q",
    )

    return (points + rolling_line + trend_line).properties(
        title=f"{role_name} - Learning Curve ({trend.title()})",
        width=width,
        height=height,
    )


def create_role_comparison_heatmap(
    comparison_df: pl.DataFrame,
    metrics: List[str] = None,
    width: int = 400,
    height: int = 300,
) -> alt.Chart:
    """Create heatmap comparing roles across normalized metrics.

    Args:
        comparison_df: DataFrame with columns: role_name, and metric columns
        metrics: List of metric columns to include
        width: Chart width
        height: Chart height

    Returns:
        Altair heatmap with roles on Y-axis, metrics on X-axis
    """
    if comparison_df.is_empty():
        return _empty_chart_placeholder("No comparison data available", width, height)

    if metrics is None:
        metrics = ["avg_payoff", "win_rate", "cooperation_rate", "total_rounds"]

    # Filter to available metrics
    available_metrics = [m for m in metrics if m in comparison_df.columns]
    if not available_metrics:
        return _empty_chart_placeholder("No metrics available for comparison", width, height)

    # Normalize metrics to 0-1 scale
    df = comparison_df.to_pandas()

    for metric in available_metrics:
        if metric in df.columns:
            min_val = df[metric].min()
            max_val = df[metric].max()
            if max_val > min_val:
                df[f"{metric}_norm"] = (df[metric] - min_val) / (max_val - min_val)
            else:
                df[f"{metric}_norm"] = 0.5

    # Melt to long format
    df_long = df.melt(
        id_vars=["role_name"],
        value_vars=[f"{m}_norm" for m in available_metrics if f"{m}_norm" in df.columns],
        var_name="metric",
        value_name="value",
    )

    # Clean metric names
    df_long["metric"] = df_long["metric"].str.replace("_norm", "").str.replace("_", " ").str.title()

    chart = alt.Chart(df_long).mark_rect().encode(
        x=alt.X("metric:N", title="Metric"),
        y=alt.Y("role_name:N", title="Role"),
        color=alt.Color(
            "value:Q",
            title="Normalized Score",
            scale=alt.Scale(scheme="viridis", domain=[0, 1]),
        ),
        tooltip=[
            alt.Tooltip("role_name:N", title="Role"),
            alt.Tooltip("metric:N", title="Metric"),
            alt.Tooltip("value:Q", title="Score", format=".2f"),
        ],
    ).properties(
        title="Role Performance Comparison",
        width=width,
        height=height,
    )

    return chart


def create_cumulative_reward_chart(
    round_data: pl.DataFrame,
    role_name: str,
    by_game: bool = False,
    per_session: bool = False,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> alt.Chart:
    """Create cumulative payoff chart over time.

    Args:
        round_data: DataFrame with payoff data
        role_name: Role name for title
        by_game: If True, create faceted chart per game type
        per_session: If True, reset cumulative at session boundaries
        width: Chart width (adaptive if None)
        height: Chart height (adaptive if None)

    Returns:
        Altair line chart with cumulative payoff
    """
    # Calculate adaptive dimensions
    total_rounds = len(round_data) if not round_data.is_empty() else 0
    num_games = round_data.select("game_type").n_unique() if (
        not round_data.is_empty() and "game_type" in round_data.columns
    ) else 1
    calc_width, calc_height = calculate_adaptive_dimensions(
        total_rounds,
        num_categories=num_games,
        base_width=450,
        base_height=250,
    )
    width = width if width is not None else calc_width
    height = height if height is not None else calc_height

    if round_data.is_empty():
        return _empty_chart_placeholder("No payoff data available", width, height)

    # Ensure round_number column exists
    if "round_number" not in round_data.columns:
        round_data = round_data.with_row_index("round_number")

    # Sort data properly - session then round for correct cumulative order
    sort_cols = []
    if "session_id" in round_data.columns:
        sort_cols.append("session_id")
    if "game_type" in round_data.columns:
        sort_cols.append("game_type")
    sort_cols.append("round_number")
    df = round_data.sort(sort_cols)

    if per_session and "session_id" in df.columns:
        # Reset cumulative at each session boundary
        df = df.with_columns(
            pl.col("payoff").cum_sum().over("session_id").alias("cumulative_payoff")
        ).with_row_index("global_idx")

        chart = alt.Chart(df.to_pandas()).mark_line(point=True).encode(
            x=alt.X("global_idx:Q", title="Round"),
            y=alt.Y("cumulative_payoff:Q", title="Cumulative Payoff"),
            color=alt.Color("session_id:N", title="Session"),
            tooltip=[
                alt.Tooltip("session_id:N", title="Session"),
                alt.Tooltip("cumulative_payoff:Q", title="Cumulative"),
                alt.Tooltip("payoff:Q", title="Round Payoff"),
            ],
        )
    elif by_game and "game_type" in df.columns:
        # Faceted approach - one chart per game type (avoids visual overlap)
        df = df.with_columns(
            pl.col("payoff").cum_sum().over("game_type").alias("cumulative_payoff")
        )
        # Add within-game round index for cleaner x-axis
        df = df.with_columns(
            pl.col("round_number").rank("ordinal").over("game_type").alias("game_round")
        )

        # Calculate dynamic facet columns based on number of games
        facet_cols = calculate_facet_columns(num_games)

        chart = alt.Chart(df.to_pandas()).mark_line(point=True, color="#3498db").encode(
            x=alt.X("game_round:Q", title="Round (within game)"),
            y=alt.Y("cumulative_payoff:Q", title="Cumulative Payoff"),
            tooltip=[
                alt.Tooltip("game_type:N", title="Game"),
                alt.Tooltip("cumulative_payoff:Q", title="Cumulative"),
                alt.Tooltip("payoff:Q", title="Round Payoff"),
            ],
        ).facet(
            facet=alt.Facet("game_type:N", title="Game Type"),
            columns=facet_cols,
        ).resolve_scale(x="independent", y="independent")
    else:
        # Single cumulative line across all games/sessions
        df = df.with_columns(
            pl.col("payoff").cum_sum().alias("cumulative_payoff")
        ).with_row_index("global_idx")

        chart = alt.Chart(df.to_pandas()).mark_line(
            point=True, color="#3498db"
        ).encode(
            x=alt.X("global_idx:Q", title="Round"),
            y=alt.Y("cumulative_payoff:Q", title="Cumulative Payoff"),
            tooltip=[
                alt.Tooltip("cumulative_payoff:Q", title="Cumulative"),
                alt.Tooltip("payoff:Q", title="Round Payoff"),
            ],
        )

    # For faceted charts, title is set differently
    if by_game and "game_type" in round_data.columns:
        return chart.properties(title=f"{role_name} - Cumulative Payoff by Game")

    return chart.properties(
        title=f"{role_name} - Cumulative Payoff",
        width=width,
        height=height,
    )


def create_game_breakdown_bars(
    breakdown_data: List[Dict[str, Any]],
    role_name: str,
    metric: str = "avg_payoff",
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> alt.Chart:
    """Create bar chart showing performance by game type.

    Args:
        breakdown_data: List of RoleGameBreakdown dicts
        role_name: Role name for title
        metric: Metric to display (avg_payoff, win_rate, rounds_count)
        width: Chart width (adaptive if None)
        height: Chart height (adaptive if None)

    Returns:
        Altair bar chart
    """
    # Calculate adaptive dimensions based on number of games
    num_games = len(breakdown_data) if breakdown_data else 0
    # Scale width with number of bars, with minimum and maximum bounds
    calc_width = max(300, min(600, 80 * num_games + 100)) if num_games > 0 else 400
    calc_height = 250
    width = width if width is not None else calc_width
    height = height if height is not None else calc_height

    if not breakdown_data:
        return _empty_chart_placeholder("No game breakdown data", width, height)

    df = pl.DataFrame(breakdown_data)

    metric_titles = {
        "avg_payoff": "Average Payoff",
        "win_rate": "Win Rate (%)",
        "rounds_count": "Total Rounds",
        "sessions_count": "Sessions",
    }

    chart = alt.Chart(df.to_pandas()).mark_bar().encode(
        x=alt.X("game_type:N", title="Game Type", sort="-y"),
        y=alt.Y(f"{metric}:Q", title=metric_titles.get(metric, metric)),
        color=alt.Color(
            f"{metric}:Q",
            scale=alt.Scale(scheme="viridis"),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("game_type:N", title="Game"),
            alt.Tooltip("avg_payoff:Q", title="Avg Payoff", format=".1f"),
            alt.Tooltip("win_rate:Q", title="Win Rate", format=".1f"),
            alt.Tooltip("rounds_count:Q", title="Rounds"),
            alt.Tooltip("sessions_count:Q", title="Sessions"),
        ],
    ).properties(
        title=f"{role_name} - Performance by Game",
        width=width,
        height=height,
    )

    return chart


# =============================================================================
# PRIORITY 2: Personality & Behavior Charts
# =============================================================================


def create_personality_radar(
    traits: Dict[str, float],
    role_name: str,
    width: int = 350,
    height: int = 350,
) -> alt.Chart:
    """Create radar-style chart for personality traits.

    Uses parallel coordinates as Altair doesn't support native radar charts.

    Args:
        traits: Dict mapping trait names to values (should be normalized 0-1)
        role_name: Role name for title
        width: Chart width
        height: Chart height

    Returns:
        Altair parallel coordinates chart
    """
    if not traits:
        return _empty_chart_placeholder("No personality data", width, height)

    # Convert to long format for parallel coordinates
    rows = [{"trait": k, "value": v} for k, v in traits.items()]
    df = pl.DataFrame(rows)

    chart = alt.Chart(df.to_pandas()).mark_line(point=True, color="#3498db").encode(
        x=alt.X("trait:N", title="Personality Dimension", sort=None),
        y=alt.Y("value:Q", title="Score", scale=alt.Scale(domain=[0, 1])),
        tooltip=[
            alt.Tooltip("trait:N", title="Trait"),
            alt.Tooltip("value:Q", title="Score", format=".2f"),
        ],
    ).properties(
        title=f"{role_name} - Personality Profile",
        width=width,
        height=height,
    )

    # Add fill area
    area = alt.Chart(df.to_pandas()).mark_area(
        opacity=0.3,
        color="#3498db",
    ).encode(
        x=alt.X("trait:N", sort=None),
        y=alt.Y("value:Q"),
    )

    return area + chart


def create_bias_gauge(
    bias_score: float,
    label: str = "Bias Score",
    width: int = 150,
    height: int = 100,
) -> alt.Chart:
    """Create gauge-style visualization for bias score.

    Args:
        bias_score: Value from -1 (clustered) to +1 (uniform)
        label: Label text
        width: Chart width
        height: Chart height

    Returns:
        Altair gauge chart
    """
    # Normalize to 0-1 for display
    normalized = (bias_score + 1) / 2

    # Create arc data
    arc_data = pl.DataFrame([
        {"category": "value", "value": normalized},
        {"category": "remaining", "value": 1 - normalized},
    ])

    # Determine color based on value
    if bias_score < -0.3:
        color = "#e74c3c"  # Red for clustered
    elif bias_score > 0.3:
        color = "#2ecc71"  # Green for uniform
    else:
        color = "#f39c12"  # Orange for neutral

    chart = alt.Chart(arc_data.to_pandas()).mark_arc(innerRadius=30, outerRadius=50).encode(
        theta=alt.Theta("value:Q", stack=True),
        color=alt.Color(
            "category:N",
            scale=alt.Scale(domain=["value", "remaining"], range=[color, "#ecf0f1"]),
            legend=None,
        ),
    ).properties(width=width, height=height)

    # Add text annotation
    text = alt.Chart(pl.DataFrame([{"text": f"{bias_score:.2f}"}]).to_pandas()).mark_text(
        fontSize=16,
        fontWeight="bold",
    ).encode(text="text:N")

    return (chart + text).properties(title=label)


def create_symmetry_gauge(
    symmetry_score: float,
    label: str = "Symmetry Breaking",
    width: int = 150,
    height: int = 100,
) -> alt.Chart:
    """Create gauge for symmetry breaking score.

    Args:
        symmetry_score: Value from 0 (symmetric) to 1 (asymmetric)
        label: Label text
        width: Chart width
        height: Chart height

    Returns:
        Altair gauge chart
    """
    arc_data = pl.DataFrame([
        {"category": "value", "value": symmetry_score},
        {"category": "remaining", "value": 1 - symmetry_score},
    ])

    # Color gradient: green (symmetric) to red (asymmetric)
    if symmetry_score < 0.3:
        color = "#2ecc71"
    elif symmetry_score > 0.7:
        color = "#e74c3c"
    else:
        color = "#f39c12"

    chart = alt.Chart(arc_data.to_pandas()).mark_arc(innerRadius=30, outerRadius=50).encode(
        theta=alt.Theta("value:Q", stack=True),
        color=alt.Color(
            "category:N",
            scale=alt.Scale(domain=["value", "remaining"], range=[color, "#ecf0f1"]),
            legend=None,
        ),
    ).properties(width=width, height=height)

    text = alt.Chart(pl.DataFrame([{"text": f"{symmetry_score:.2f}"}]).to_pandas()).mark_text(
        fontSize=16,
        fontWeight="bold",
    ).encode(text="text:N")

    return (chart + text).properties(title=label)


def create_adaptability_gauge(
    adaptability_score: float,
    label: str = "Adaptability",
    width: int = 150,
    height: int = 100,
) -> alt.Chart:
    """Create gauge for adaptability score.

    Args:
        adaptability_score: Value from 0 (rigid) to 1 (highly adaptive)
        label: Label text
        width: Chart width
        height: Chart height

    Returns:
        Altair gauge chart
    """
    arc_data = pl.DataFrame([
        {"category": "value", "value": adaptability_score},
        {"category": "remaining", "value": 1 - adaptability_score},
    ])

    # Color: blue for low (rigid), green for high (adaptive)
    if adaptability_score > 0.6:
        color = "#2ecc71"  # Green for adaptive
    elif adaptability_score < 0.3:
        color = "#3498db"  # Blue for rigid
    else:
        color = "#f39c12"  # Orange for moderate

    chart = alt.Chart(arc_data.to_pandas()).mark_arc(innerRadius=30, outerRadius=50).encode(
        theta=alt.Theta("value:Q", stack=True),
        color=alt.Color(
            "category:N",
            scale=alt.Scale(domain=["value", "remaining"], range=[color, "#ecf0f1"]),
            legend=None,
        ),
    ).properties(width=width, height=height)

    text = alt.Chart(pl.DataFrame([{"text": f"{adaptability_score:.2f}"}]).to_pandas()).mark_text(
        fontSize=16,
        fontWeight="bold",
    ).encode(text="text:N")

    return (chart + text).properties(title=label)


def create_risk_tolerance_gauge(
    risk_tolerance: float,
    label: str = "Risk Tolerance",
    width: int = 150,
    height: int = 100,
) -> alt.Chart:
    """Create gauge for risk tolerance score.

    Args:
        risk_tolerance: Value from 0 (risk-averse) to 1 (risk-seeking)
        label: Label text
        width: Chart width
        height: Chart height

    Returns:
        Altair gauge chart
    """
    arc_data = pl.DataFrame([
        {"category": "value", "value": risk_tolerance},
        {"category": "remaining", "value": 1 - risk_tolerance},
    ])

    # Color gradient: green (safe) to red (risky)
    if risk_tolerance > 0.6:
        color = "#e74c3c"  # Red for risk-seeking
    elif risk_tolerance < 0.4:
        color = "#2ecc71"  # Green for risk-averse
    else:
        color = "#f39c12"  # Orange for neutral

    chart = alt.Chart(arc_data.to_pandas()).mark_arc(innerRadius=30, outerRadius=50).encode(
        theta=alt.Theta("value:Q", stack=True),
        color=alt.Color(
            "category:N",
            scale=alt.Scale(domain=["value", "remaining"], range=[color, "#ecf0f1"]),
            legend=None,
        ),
    ).properties(width=width, height=height)

    text = alt.Chart(pl.DataFrame([{"text": f"{risk_tolerance:.2f}"}]).to_pandas()).mark_text(
        fontSize=16,
        fontWeight="bold",
    ).encode(text="text:N")

    return (chart + text).properties(title=label)


def create_temporal_pattern_chart(
    early_concentration: float,
    late_concentration: float,
    role_name: str,
    width: int = 250,
    height: int = 150,
) -> alt.Chart:
    """Create chart showing early vs late game concentration.

    Args:
        early_concentration: Average HHI in first half of sessions
        late_concentration: Average HHI in second half of sessions
        role_name: Role name for title
        width: Chart width
        height: Chart height

    Returns:
        Altair bar chart comparing early vs late concentration
    """
    df = pl.DataFrame([
        {"phase": "Early Game", "concentration": early_concentration},
        {"phase": "Late Game", "concentration": late_concentration},
    ])

    # Determine trend color
    change = late_concentration - early_concentration
    if change > 0.05:
        colors = ["#3498db", "#e74c3c"]  # Blue to red (concentrating)
    elif change < -0.05:
        colors = ["#e74c3c", "#3498db"]  # Red to blue (spreading)
    else:
        colors = ["#95a5a6", "#95a5a6"]  # Gray (stable)

    chart = alt.Chart(df.to_pandas()).mark_bar().encode(
        x=alt.X("phase:N", title="Game Phase", sort=["Early Game", "Late Game"]),
        y=alt.Y("concentration:Q", title="Avg Concentration (HHI)", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color(
            "phase:N",
            scale=alt.Scale(domain=["Early Game", "Late Game"], range=colors),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("phase:N", title="Phase"),
            alt.Tooltip("concentration:Q", title="HHI", format=".3f"),
        ],
    ).properties(
        title=f"{role_name} - Temporal Pattern",
        width=width,
        height=height,
    )

    return chart


def create_action_frequency_bars(
    round_data: pl.DataFrame,
    role_name: str,
    by_game: bool = False,
    width: int = 350,
    height: int = 200,
) -> alt.Chart:
    """Create bar chart showing action frequency distribution.

    Args:
        round_data: DataFrame with action column
        role_name: Role name for title
        by_game: If True, create faceted chart per game type
        width: Chart width
        height: Chart height

    Returns:
        Altair bar chart
    """
    if round_data.is_empty() or "action" not in round_data.columns:
        return _empty_chart_placeholder("No action data available", width, height)

    # Handle per-game faceting
    if by_game and "game_type" in round_data.columns:
        # Group by game and action
        action_counts = (
            round_data.filter(pl.col("action").is_not_null())
            .group_by(["game_type", "action"])
            .agg(pl.len().alias("count"))
            .sort(["game_type", "count"], descending=[False, True])
        )

        if action_counts.is_empty():
            return _empty_chart_placeholder("No valid actions recorded", width, height)

        # Calculate percentages within each game
        action_counts = action_counts.with_columns(
            (pl.col("count") / pl.col("count").sum().over("game_type") * 100).alias("percentage")
        )

        num_games = action_counts.select("game_type").n_unique()
        facet_cols = calculate_facet_columns(num_games)

        chart = alt.Chart(action_counts.to_pandas()).mark_bar().encode(
            x=alt.X("action:N", title="Action", sort="-y"),
            y=alt.Y("count:Q", title="Frequency"),
            color=alt.Color(
                "action:N",
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("game_type:N", title="Game"),
                alt.Tooltip("action:N", title="Action"),
                alt.Tooltip("count:Q", title="Count"),
                alt.Tooltip("percentage:Q", title="Percentage", format=".1f"),
            ],
        ).facet(
            facet=alt.Facet("game_type:N", title="Game Type"),
            columns=facet_cols,
        ).resolve_scale(x="independent", y="independent")

        return chart.properties(title=f"{role_name} - Action Distribution by Game")

    # Original single-chart logic
    action_counts = (
        round_data.filter(pl.col("action").is_not_null())
        .group_by("action")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )

    if action_counts.is_empty():
        return _empty_chart_placeholder("No valid actions recorded", width, height)

    # Calculate percentages
    total = action_counts["count"].sum()
    action_counts = action_counts.with_columns(
        (pl.col("count") / total * 100).alias("percentage")
    )

    chart = alt.Chart(action_counts.to_pandas()).mark_bar().encode(
        x=alt.X("action:N", title="Action", sort="-y"),
        y=alt.Y("count:Q", title="Frequency"),
        color=alt.Color(
            "percentage:Q",
            scale=alt.Scale(scheme="viridis"),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("action:N", title="Action"),
            alt.Tooltip("count:Q", title="Count"),
            alt.Tooltip("percentage:Q", title="Percentage", format=".1f"),
        ],
    ).properties(
        title=f"{role_name} - Action Distribution",
        width=width,
        height=height,
    )

    return chart


def create_fingerprint_matrix(
    fingerprints: Dict[str, Dict[str, Any]],
    role_name: str,
    width: int = 400,
    height: int = 200,
) -> alt.Chart:
    """Create compact matrix showing game-specific behavior fingerprints.

    Uses absolute scales for metrics with known ranges (HHI, entropy: 0-1)
    and percentile ranking for unbounded metrics (num_rounds, num_sessions).

    Args:
        fingerprints: Dict from ModelPersonalityProfile.game_fingerprints
                     {game_type: {avg_concentration, dominant_strategy, ...}}
        role_name: Role name for title
        width: Chart width
        height: Chart height

    Returns:
        Altair heatmap-style fingerprint matrix
    """
    if not fingerprints:
        return _empty_chart_placeholder("No fingerprint data available", width, height)

    # Define known metric ranges for absolute scaling
    METRIC_RANGES = {
        "avg_concentration": (0.0, 1.0),  # HHI is 0-1
        "avg_entropy": (0.0, 1.0),         # Normalized entropy 0-1
    }

    # Extract metrics into rows with appropriate normalization
    rows = []
    metrics = ["avg_concentration", "avg_entropy", "num_rounds", "num_sessions"]

    for game_type, fp in fingerprints.items():
        for metric in metrics:
            if metric in fp:
                value = fp[metric]
                # Use absolute scale for known ranges
                if metric in METRIC_RANGES:
                    min_v, max_v = METRIC_RANGES[metric]
                    normalized = (value - min_v) / (max_v - min_v) if max_v > min_v else 0.5
                else:
                    # Mark for percentile normalization later
                    normalized = None  # Will be calculated below

                rows.append({
                    "game": game_type,
                    "metric": metric.replace("_", " ").title(),
                    "value": value,
                    "normalized": normalized,
                    "has_known_range": metric in METRIC_RANGES,
                })

    if not rows:
        return _empty_chart_placeholder("No fingerprint metrics available", width, height)

    df = pl.DataFrame(rows)

    # Apply percentile normalization only to metrics without known ranges
    # Calculate min/max for unbounded metrics and normalize
    unbounded_metrics = df.filter(~pl.col("has_known_range")).select("metric").unique().to_series().to_list()

    for metric in unbounded_metrics:
        metric_values = df.filter(pl.col("metric") == metric).select("value").to_series()
        if len(metric_values) > 0:
            min_v = metric_values.min()
            max_v = metric_values.max()
            # Use log scale for count metrics to handle large ranges
            if max_v > min_v:
                # Log transform for better visual distribution
                import math
                log_min = math.log1p(min_v)
                log_max = math.log1p(max_v)
                df = df.with_columns(
                    pl.when((pl.col("metric") == metric) & pl.col("normalized").is_null())
                    .then((pl.col("value").log1p() - log_min) / (log_max - log_min + 0.001))
                    .otherwise(pl.col("normalized"))
                    .alias("normalized")
                )
            else:
                df = df.with_columns(
                    pl.when((pl.col("metric") == metric) & pl.col("normalized").is_null())
                    .then(0.5)
                    .otherwise(pl.col("normalized"))
                    .alias("normalized")
                )

    # Fill any remaining nulls
    df = df.with_columns(pl.col("normalized").fill_null(0.5))

    chart = alt.Chart(df.to_pandas()).mark_rect().encode(
        x=alt.X("metric:N", title="Metric"),
        y=alt.Y("game:N", title="Game Type"),
        color=alt.Color(
            "normalized:Q",
            title="Score",
            scale=alt.Scale(scheme="viridis", domain=[0, 1]),
        ),
        tooltip=[
            alt.Tooltip("game:N", title="Game"),
            alt.Tooltip("metric:N", title="Metric"),
            alt.Tooltip("value:Q", title="Raw Value", format=".2f"),
            alt.Tooltip("normalized:Q", title="Normalized", format=".2f"),
        ],
    ).properties(
        title=f"{role_name} - Behavior Fingerprint",
        width=width,
        height=height,
    )

    return chart


# =============================================================================
# PRIORITY 3: Advanced Charts
# =============================================================================


def create_role_cluster_scatter(
    features_2d: List[List[float]],
    role_names: List[str],
    metadata: Optional[List[Dict[str, Any]]] = None,
    width: int = 450,
    height: int = 350,
) -> alt.Chart:
    """Create scatter plot of roles in 2D strategy space.

    Args:
        features_2d: List of [x, y] coordinates from PCA/t-SNE
        role_names: Names for each point
        metadata: Optional additional info per point
        width: Chart width
        height: Chart height

    Returns:
        Altair scatter plot
    """
    if not features_2d or not role_names:
        return _empty_chart_placeholder("No clustering data available", width, height)

    rows = []
    for i, (coords, name) in enumerate(zip(features_2d, role_names)):
        row = {
            "x": coords[0],
            "y": coords[1],
            "role": name,
        }
        if metadata and i < len(metadata):
            row.update(metadata[i])
        rows.append(row)

    df = pl.DataFrame(rows)

    chart = alt.Chart(df.to_pandas()).mark_circle(size=100).encode(
        x=alt.X("x:Q", title="Component 1"),
        y=alt.Y("y:Q", title="Component 2"),
        color=alt.Color("role:N", title="Role"),
        tooltip=list(rows[0].keys()) if rows else [],
    ).properties(
        title="Role Strategy Space (PCA)",
        width=width,
        height=height,
    )

    # Add role labels
    labels = alt.Chart(df.to_pandas()).mark_text(
        align="left",
        dx=7,
        fontSize=10,
    ).encode(
        x="x:Q",
        y="y:Q",
        text="role:N",
    )

    return chart + labels


def create_consistency_overlap(
    game_metrics: Dict[str, List[float]],
    role_name: str,
    metric_name: str = "payoff",
    width: int = 400,
    height: int = 250,
) -> alt.Chart:
    """Create chart showing consistency of metric across games.

    Args:
        game_metrics: Dict mapping game_type to list of metric values
        role_name: Role name for title
        metric_name: Name of the metric being compared
        width: Chart width
        height: Chart height

    Returns:
        Strip/jitter plot showing distribution per game
    """
    if not game_metrics:
        return _empty_chart_placeholder("No consistency data available", width, height)

    rows = []
    for game_type, values in game_metrics.items():
        for v in values:
            rows.append({
                "game": game_type,
                "value": v,
            })

    if not rows:
        return _empty_chart_placeholder("No metric values available", width, height)

    df = pl.DataFrame(rows)

    # Calculate mean per game for annotation
    means = df.group_by("game").agg(pl.col("value").mean().alias("mean"))

    # Strip plot with jitter
    strip = alt.Chart(df.to_pandas()).mark_circle(opacity=0.5, size=30).encode(
        x=alt.X("game:N", title="Game Type"),
        y=alt.Y("value:Q", title=metric_name.title()),
        color=alt.Color("game:N", legend=None),
    )

    # Mean markers
    mean_marks = alt.Chart(means.to_pandas()).mark_tick(
        color="red",
        thickness=2,
        size=30,
    ).encode(
        x="game:N",
        y="mean:Q",
    )

    # Overall mean line
    overall_mean = df["value"].mean()
    mean_rule = alt.Chart(pl.DataFrame([{"y": overall_mean}]).to_pandas()).mark_rule(
        strokeDash=[5, 5],
        color="#e74c3c",
    ).encode(y="y:Q")

    return (strip + mean_marks + mean_rule).properties(
        title=f"{role_name} - {metric_name.title()} Consistency Across Games",
        width=width,
        height=height,
    )


def create_decision_trajectory(
    round_data: pl.DataFrame,
    role_name: str,
    action_mapping: Optional[Dict[str, int]] = None,
    width: int = 500,
    height: int = 250,
) -> alt.Chart:
    """Create trajectory chart showing decision flow over time.

    Args:
        round_data: DataFrame with action data
        role_name: Role name for title
        action_mapping: Optional mapping of actions to numeric values
        width: Chart width
        height: Chart height

    Returns:
        Altair line chart showing action trajectory
    """
    if round_data.is_empty() or "action" not in round_data.columns:
        return _empty_chart_placeholder("No action data for trajectory", width, height)

    df = round_data.filter(pl.col("action").is_not_null()).with_row_index("idx")

    if df.is_empty():
        return _empty_chart_placeholder("No valid actions recorded", width, height)

    # Map actions to numeric if not provided
    if action_mapping is None:
        unique_actions = df["action"].unique().to_list()
        action_mapping = {a: i for i, a in enumerate(unique_actions)}

    df = df.with_columns(
        pl.col("action").replace_strict(action_mapping, default=0).alias("action_num")
    )

    chart = alt.Chart(df.to_pandas()).mark_line(
        point=True,
        interpolate="step-after",
    ).encode(
        x=alt.X("idx:Q", title="Decision #"),
        y=alt.Y(
            "action_num:Q",
            title="Action",
            axis=alt.Axis(
                values=list(action_mapping.values()),
                labelExpr=f"datum.value == 0 ? '{list(action_mapping.keys())[0] if action_mapping else ''}' : "
                          f"datum.value == 1 ? '{list(action_mapping.keys())[1] if len(action_mapping) > 1 else ''}' : "
                          f"'{list(action_mapping.keys())[2] if len(action_mapping) > 2 else ''}'",
            ),
        ),
        color=alt.Color("game_type:N", title="Game") if "game_type" in df.columns else alt.value("#3498db"),
        tooltip=[
            alt.Tooltip("idx:Q", title="Decision #"),
            alt.Tooltip("action:N", title="Action"),
            alt.Tooltip("payoff:Q", title="Payoff") if "payoff" in df.columns else alt.value(""),
        ],
    ).properties(
        title=f"{role_name} - Decision Trajectory",
        width=width,
        height=height,
    )

    return chart


def create_response_time_histogram(
    round_data: pl.DataFrame,
    role_name: str,
    bins: int = 20,
    width: int = 350,
    height: int = 200,
) -> alt.Chart:
    """Create histogram of response times.

    Args:
        round_data: DataFrame with response_time column
        role_name: Role name for title
        bins: Number of histogram bins
        width: Chart width
        height: Chart height

    Returns:
        Altair histogram
    """
    if round_data.is_empty() or "response_time" not in round_data.columns:
        return _empty_chart_placeholder("No response time data", width, height)

    df = round_data.filter(pl.col("response_time").is_not_null())

    if df.is_empty():
        return _empty_chart_placeholder("No valid response times", width, height)

    chart = alt.Chart(df.to_pandas()).mark_bar().encode(
        x=alt.X("response_time:Q", bin=alt.Bin(maxbins=bins), title="Response Time (s)"),
        y=alt.Y("count():Q", title="Frequency"),
        tooltip=[
            alt.Tooltip("count():Q", title="Count"),
        ],
    ).properties(
        title=f"{role_name} - Response Time Distribution",
        width=width,
        height=height,
    )

    # Add mean line
    mean_rt = df["response_time"].mean()
    mean_rule = alt.Chart(pl.DataFrame([{"x": mean_rt}]).to_pandas()).mark_rule(
        color="#e74c3c",
        strokeDash=[5, 5],
    ).encode(x="x:Q")

    return chart + mean_rule
