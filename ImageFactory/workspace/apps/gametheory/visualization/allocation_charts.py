"""Visualization charts for allocation game analytics.

Provides Altair charts for analyzing Colonel Blotto, Tennis Coach, Sumo Coach
and other allocation-based games.
"""

from typing import List, Dict, Optional, Any
from collections import Counter
import altair as alt
import numpy as np
import polars as pl

from ..core.utils import parse_allocation_from_result


def create_allocation_heatmap(
    results: List[Dict],
    player_num: int,
    title: str = None
) -> alt.Chart:
    """Create heatmap showing allocation per field across rounds.

    Args:
        results: List of round result dictionaries
        player_num: Player number (1-indexed)
        title: Optional chart title

    Returns:
        Altair Chart
    """
    # Extract allocations
    data_rows = []
    for round_idx, result in enumerate(results):
        allocation = parse_allocation_from_result(result, player_num)

        if allocation:
            for field_idx, value in enumerate(allocation):
                data_rows.append({
                    "round": round_idx + 1,
                    "field": f"Field {field_idx + 1}",
                    "allocation": float(value),
                })

    if not data_rows:
        return alt.Chart().mark_text().encode(
            text=alt.value("No allocation data available")
        )

    df = pl.DataFrame(data_rows)

    chart = alt.Chart(df.to_pandas()).mark_rect().encode(
        x=alt.X("field:N", title="Battlefield/Target", sort=None),
        y=alt.Y("round:O", title="Round", sort="descending"),
        color=alt.Color(
            "allocation:Q",
            title="Allocation",
            scale=alt.Scale(scheme="viridis"),
        ),
        tooltip=["round", "field", "allocation"],
    ).properties(
        title=title or f"Player {player_num} Allocation Heatmap",
        width=250,
        height=min(300, len(results) * 25 + 50),
    )

    return chart


def create_concentration_timeline(
    results: List[Dict],
    num_players: int = 2,
    budget: float = 100.0
) -> alt.Chart:
    """Create line chart showing concentration index over rounds.

    Args:
        results: List of round result dictionaries
        num_players: Number of players
        budget: Allocation budget for HHI calculation

    Returns:
        Altair Chart
    """
    from ..analytics.allocation import AllocationAnalyzer
    import ast

    analyzer = AllocationAnalyzer()
    data_rows = []

    for round_idx, result in enumerate(results):
        for player_num in range(1, num_players + 1):
            allocation = analyzer.parse_allocation_from_result(result, player_num)

            if allocation:
                metrics = analyzer.analyze_allocation(allocation, budget)
                model_key = f"player{player_num}_model"
                model = result.get(model_key, f"Player {player_num}")

                data_rows.append({
                    "round": round_idx + 1,
                    "player": f"P{player_num}: {model[:15]}",
                    "concentration": metrics.concentration_index,
                    "strategy": metrics.strategy_type,
                })

    if not data_rows:
        return alt.Chart().mark_text().encode(
            text=alt.value("No allocation data available")
        )

    df = pl.DataFrame(data_rows)

    chart = alt.Chart(df.to_pandas()).mark_line(point=True).encode(
        x=alt.X("round:Q", title="Round"),
        y=alt.Y("concentration:Q", title="Concentration Index (HHI)", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("player:N", title="Player"),
        tooltip=["round", "player", "concentration", "strategy"],
    ).properties(
        title="Strategy Concentration Over Rounds",
        width=400,
        height=250,
    )

    return chart


def create_compliance_summary_chart(
    results: List[Dict],
    player_num: int
) -> alt.Chart:
    """Create donut chart showing parse/normalize/fail rates.

    Args:
        results: List of round result dictionaries
        player_num: Player number (1-indexed)

    Returns:
        Altair Chart
    """
    from ..analytics.allocation import AllocationAnalyzer

    analyzer = AllocationAnalyzer()
    compliance = analyzer.calculate_compliance_metrics(results, player_num)

    data = [
        {"status": "Parsed OK", "count": compliance.parsed_ok, "color": "#2ecc71"},
        {"status": "Normalized", "count": compliance.normalized, "color": "#f39c12"},
        {"status": "Failed", "count": compliance.failed, "color": "#e74c3c"},
    ]

    # Filter out zero counts
    data = [d for d in data if d["count"] > 0]

    if not data:
        data = [{"status": "No Data", "count": 1, "color": "#95a5a6"}]

    df = pl.DataFrame(data)

    chart = alt.Chart(df.to_pandas()).mark_arc(innerRadius=40).encode(
        theta=alt.Theta("count:Q"),
        color=alt.Color(
            "status:N",
            title="Status",
            scale=alt.Scale(
                domain=["Parsed OK", "Normalized", "Failed"],
                range=["#2ecc71", "#f39c12", "#e74c3c"]
            ),
        ),
        tooltip=["status", "count"],
    ).properties(
        title=f"Player {player_num} Compliance",
        width=200,
        height=200,
    )

    return chart


def create_allocation_boxplot(
    results: List[Dict],
    player_num: int,
    budget: float = 100.0
) -> alt.Chart:
    """Create box plot showing allocation distribution per field.

    Args:
        results: List of round result dictionaries
        player_num: Player number (1-indexed)
        budget: Allocation budget

    Returns:
        Altair Chart
    """
    from ..analytics.allocation import AllocationAnalyzer
    import ast

    analyzer = AllocationAnalyzer()
    data_rows = []

    for result in results:
        allocation = analyzer.parse_allocation_from_result(result, player_num)
        if allocation:
            for field_idx, value in enumerate(allocation):
                data_rows.append({
                    "field": f"Field {field_idx + 1}",
                    "allocation": float(value),
                })

    if not data_rows:
        return alt.Chart().mark_text().encode(
            text=alt.value("No allocation data available")
        )

    df = pl.DataFrame(data_rows)

    chart = alt.Chart(df.to_pandas()).mark_boxplot().encode(
        x=alt.X("field:N", title="Battlefield/Target", sort=None),
        y=alt.Y("allocation:Q", title="Allocation"),
        color=alt.Color("field:N", legend=None),
    ).properties(
        title=f"Player {player_num} Allocation Distribution",
        width=300,
        height=200,
    )

    return chart


def create_field_preference_bars(
    summaries: Dict[int, Any],
    num_fields: int
) -> alt.Chart:
    """Create grouped bar chart showing average field preferences.

    Args:
        summaries: Dict mapping player_num to SessionAllocationSummary
        num_fields: Number of fields/targets

    Returns:
        Altair Chart
    """
    data_rows = []

    for player_num, summary in summaries.items():
        if hasattr(summary, 'field_preferences') and summary.field_preferences:
            for field_idx, avg_alloc in enumerate(summary.field_preferences):
                data_rows.append({
                    "player": f"P{player_num}: {summary.model[:12]}",
                    "field": f"Field {field_idx + 1}",
                    "avg_allocation": avg_alloc,
                })

    if not data_rows:
        return alt.Chart().mark_text().encode(
            text=alt.value("No field preference data available")
        )

    df = pl.DataFrame(data_rows)

    chart = alt.Chart(df.to_pandas()).mark_bar().encode(
        x=alt.X("field:N", title="Battlefield/Target", sort=None),
        y=alt.Y("avg_allocation:Q", title="Average Allocation"),
        color=alt.Color("player:N", title="Player"),
        xOffset="player:N",
        tooltip=["player", "field", "avg_allocation"],
    ).properties(
        title="Average Field Preferences by Player",
        width=350,
        height=200,
    )

    return chart


def create_strategy_type_pie(
    metrics_list: List[Any]
) -> alt.Chart:
    """Create pie chart showing strategy type distribution.

    Args:
        metrics_list: List of AllocationMetrics

    Returns:
        Altair Chart
    """
    strategy_counts = Counter(m.strategy_type for m in metrics_list)

    data = [
        {"strategy": strategy, "count": count}
        for strategy, count in strategy_counts.items()
    ]

    if not data:
        return alt.Chart().mark_text().encode(
            text=alt.value("No strategy data available")
        )

    df = pl.DataFrame(data)

    colors = {
        "concentrated": "#e74c3c",
        "uniform": "#3498db",
        "asymmetric": "#9b59b6",
        "hedged": "#2ecc71",
    }

    chart = alt.Chart(df.to_pandas()).mark_arc().encode(
        theta=alt.Theta("count:Q"),
        color=alt.Color(
            "strategy:N",
            title="Strategy Type",
            scale=alt.Scale(
                domain=list(colors.keys()),
                range=list(colors.values())
            ),
        ),
        tooltip=["strategy", "count"],
    ).properties(
        title="Strategy Type Distribution",
        width=200,
        height=200,
    )

    return chart


def create_hyperparameter_heatmap(
    aggregated_data: List[Dict],
    x_param: str = "temperature",
    y_param: str = "top_k",
    metric: str = "avg_concentration"
) -> alt.Chart:
    """Create heatmap showing metric by hyperparameter combination.

    Args:
        aggregated_data: List of aggregated metrics from CrossSessionAnalyzer
        x_param: Parameter for X axis
        y_param: Parameter for Y axis
        metric: Metric to visualize (avg_concentration, avg_compliance)

    Returns:
        Altair Chart
    """
    if not aggregated_data:
        return alt.Chart().mark_text().encode(
            text=alt.value("No hyperparameter data available")
        )

    df = pl.DataFrame(aggregated_data)

    chart = alt.Chart(df.to_pandas()).mark_rect().encode(
        x=alt.X(f"{x_param}:O", title=x_param.replace("_", " ").title()),
        y=alt.Y(f"{y_param}:O", title=y_param.replace("_", " ").title()),
        color=alt.Color(
            f"{metric}:Q",
            title=metric.replace("_", " ").title(),
            scale=alt.Scale(scheme="viridis"),
        ),
        tooltip=[x_param, y_param, metric, "num_sessions"],
    ).properties(
        title=f"{metric.replace('_', ' ').title()} by {x_param} x {y_param}",
        width=300,
        height=250,
    )

    return chart


def create_strategy_cluster_scatter(
    features_2d: List[List[float]],
    labels: List[str],
    metadata: Optional[List[Dict]] = None
) -> alt.Chart:
    """Create scatter plot of strategies in reduced 2D space.

    Args:
        features_2d: List of [x, y] coordinates from PCA/t-SNE
        labels: Cluster labels or strategy types
        metadata: Optional additional info per point

    Returns:
        Altair Chart
    """
    data_rows = []
    for i, (coords, label) in enumerate(zip(features_2d, labels)):
        row = {
            "x": coords[0],
            "y": coords[1],
            "cluster": label,
        }
        if metadata and i < len(metadata):
            row.update(metadata[i])
        data_rows.append(row)

    if not data_rows:
        return alt.Chart().mark_text().encode(
            text=alt.value("No clustering data available")
        )

    df = pl.DataFrame(data_rows)

    chart = alt.Chart(df.to_pandas()).mark_circle(size=60).encode(
        x=alt.X("x:Q", title="Component 1"),
        y=alt.Y("y:Q", title="Component 2"),
        color=alt.Color("cluster:N", title="Cluster/Type"),
        tooltip=list(data_rows[0].keys()) if data_rows else [],
    ).properties(
        title="Strategy Clusters (PCA/t-SNE)",
        width=400,
        height=300,
    )

    return chart


def create_model_comparison_radar(
    fingerprints: Dict[str, Dict],
    num_fields: int
) -> alt.Chart:
    """Create radar-like chart comparing model allocation patterns.

    Note: Altair doesn't natively support radar charts, so we create
    a parallel coordinates-style visualization.

    Args:
        fingerprints: Dict mapping model name to fingerprint dict
        num_fields: Number of fields/targets

    Returns:
        Altair Chart
    """
    data_rows = []

    for model, fp in fingerprints.items():
        prefs = fp.get("avg_field_preferences", [])
        if prefs:
            # Normalize to 0-1 range
            max_pref = max(prefs) if prefs else 1
            for field_idx, pref in enumerate(prefs):
                data_rows.append({
                    "model": model[:15],
                    "field": f"F{field_idx + 1}",
                    "preference": pref / max_pref if max_pref > 0 else 0,
                    "raw_value": pref,
                })

    if not data_rows:
        return alt.Chart().mark_text().encode(
            text=alt.value("No model comparison data available")
        )

    df = pl.DataFrame(data_rows)

    chart = alt.Chart(df.to_pandas()).mark_line(point=True).encode(
        x=alt.X("field:N", title="Field", sort=None),
        y=alt.Y("preference:Q", title="Relative Preference", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("model:N", title="Model"),
        tooltip=["model", "field", "raw_value"],
    ).properties(
        title="Model Field Preference Patterns",
        width=350,
        height=200,
    )

    return chart


def create_allocation_violin(
    results: List[Dict],
    player_num: int
) -> alt.Chart:
    """Create violin plot showing allocation distribution per field.

    Args:
        results: List of round result dictionaries
        player_num: Player number (1-indexed)

    Returns:
        Altair Chart
    """
    from ..analytics.allocation import AllocationAnalyzer

    analyzer = AllocationAnalyzer()
    data_rows = []

    for result in results:
        allocation = analyzer.parse_allocation_from_result(result, player_num)
        if allocation:
            for field_idx, value in enumerate(allocation):
                data_rows.append({
                    "field": f"Field {field_idx + 1}",
                    "allocation": float(value),
                })

    if not data_rows:
        return alt.Chart().mark_text().encode(
            text=alt.value("No allocation data available")
        )

    df = pl.DataFrame(data_rows)

    # Violin plot using layered density + boxplot
    violin = alt.Chart(df.to_pandas()).transform_density(
        "allocation",
        as_=["allocation", "density"],
        extent=[0, max(d["allocation"] for d in data_rows) * 1.1],
        groupby=["field"],
    ).mark_area(orient="horizontal").encode(
        y=alt.Y("allocation:Q", title="Allocation"),
        x=alt.X(
            "density:Q",
            stack="center",
            impute=None,
            title=None,
            axis=alt.Axis(labels=False, values=[0], grid=False, ticks=False),
        ),
        column=alt.Column("field:N", title="Battlefield/Target", header=alt.Header(labelOrient="bottom")),
        color=alt.Color("field:N", legend=None),
    ).properties(
        width=60,
        height=200,
    )

    return violin


def create_evolution_summary_chart(
    evolution_data: Dict[str, Any]
) -> alt.Chart:
    """Create chart showing strategy evolution over session.

    Args:
        evolution_data: Dict from AllocationAnalyzer.detect_strategy_evolution()

    Returns:
        Altair Chart
    """
    concentrations = evolution_data.get("concentration_by_round", [])

    if not concentrations:
        return alt.Chart().mark_text().encode(
            text=alt.value("No evolution data available")
        )

    data_rows = [
        {"round": i + 1, "concentration": c}
        for i, c in enumerate(concentrations)
    ]

    df = pl.DataFrame(data_rows)

    # Add trend line
    base = alt.Chart(df.to_pandas())

    line = base.mark_line(point=True, color="#3498db").encode(
        x=alt.X("round:Q", title="Round"),
        y=alt.Y("concentration:Q", title="Concentration (HHI)", scale=alt.Scale(domain=[0, 1])),
        tooltip=["round", "concentration"],
    )

    trend = base.transform_regression("round", "concentration").mark_line(
        strokeDash=[5, 5],
        color="#e74c3c",
    ).encode(
        x="round:Q",
        y="concentration:Q",
    )

    # Add mid-point annotation
    mid = len(concentrations) // 2
    rule = alt.Chart(pl.DataFrame([{"x": mid + 0.5}]).to_pandas()).mark_rule(
        strokeDash=[3, 3],
        color="#95a5a6",
    ).encode(x="x:Q")

    chart = (line + trend + rule).properties(
        title=f"Strategy Evolution: {evolution_data.get('trend', 'unknown').title()}",
        width=400,
        height=200,
    )

    return chart


# --- Sensitivity Analysis Charts ---

def create_compliance_by_penalty_chart(
    sensitivity_result: 'SensitivityResult'
) -> alt.Chart:
    """Create bar chart showing compliance rate by repeat_penalty.

    Args:
        sensitivity_result: SensitivityResult from HyperparameterSensitivityAnalyzer

    Returns:
        Altair Chart
    """
    if not sensitivity_result.values:
        return alt.Chart().mark_text().encode(
            text=alt.value("No sensitivity data available")
        )

    data_rows = [
        {
            "repeat_penalty": value,
            "compliance_rate": sensitivity_result.compliance_by_value.get(value, 0),
            "sample_count": sensitivity_result.sample_counts.get(value, 0),
        }
        for value in sensitivity_result.values
    ]

    df = pl.DataFrame(data_rows)

    chart = alt.Chart(df.to_pandas()).mark_bar().encode(
        x=alt.X("repeat_penalty:O", title="Repeat Penalty"),
        y=alt.Y("compliance_rate:Q", title="Compliance Rate", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color(
            "compliance_rate:Q",
            scale=alt.Scale(scheme="redyellowgreen"),
            legend=None,
        ),
        tooltip=["repeat_penalty", "compliance_rate", "sample_count"],
    ).properties(
        title="Format Compliance by Repeat Penalty",
        width=300,
        height=200,
    )

    return chart


def create_sensitivity_heatmap(
    interaction_result: 'InteractionResult'
) -> alt.Chart:
    """Create heatmap showing metric across two parameters.

    Args:
        interaction_result: InteractionResult from sensitivity analyzer

    Returns:
        Altair Chart
    """
    if interaction_result.metric_surface.size == 0:
        return alt.Chart().mark_text().encode(
            text=alt.value("No interaction data available")
        )

    # Convert surface to long format
    data_rows = []
    for y_idx, y_val in enumerate(interaction_result.y_values):
        for x_idx, x_val in enumerate(interaction_result.x_values):
            metric_val = interaction_result.metric_surface[y_idx, x_idx]
            if not np.isnan(metric_val):
                data_rows.append({
                    interaction_result.x_param: x_val,
                    interaction_result.y_param: y_val,
                    "metric": metric_val,
                })

    if not data_rows:
        return alt.Chart().mark_text().encode(
            text=alt.value("No valid data points")
        )

    df = pl.DataFrame(data_rows)

    chart = alt.Chart(df.to_pandas()).mark_rect().encode(
        x=alt.X(f"{interaction_result.x_param}:O", title=interaction_result.x_param.replace("_", " ").title()),
        y=alt.Y(f"{interaction_result.y_param}:O", title=interaction_result.y_param.replace("_", " ").title()),
        color=alt.Color(
            "metric:Q",
            title=interaction_result.metric_name,
            scale=alt.Scale(scheme="viridis"),
        ),
        tooltip=[interaction_result.x_param, interaction_result.y_param, "metric"],
    ).properties(
        title=f"{interaction_result.metric_name} by {interaction_result.x_param} x {interaction_result.y_param}",
        width=300,
        height=250,
    )

    # Add optimal point marker
    optimal_data = pl.DataFrame([{
        interaction_result.x_param: interaction_result.optimal_x,
        interaction_result.y_param: interaction_result.optimal_y,
    }])

    optimal_point = alt.Chart(optimal_data.to_pandas()).mark_point(
        shape="star",
        size=200,
        color="red",
        filled=True,
    ).encode(
        x=f"{interaction_result.x_param}:O",
        y=f"{interaction_result.y_param}:O",
    )

    return chart + optimal_point


# --- Tournament Charts ---

def create_tournament_standings_chart(
    standings: List['TournamentStanding']
) -> alt.Chart:
    """Create bar chart of tournament standings.

    Args:
        standings: List of TournamentStanding

    Returns:
        Altair Chart
    """
    if not standings:
        return alt.Chart().mark_text().encode(
            text=alt.value("No standings data available")
        )

    data_rows = [
        {
            "model": s.model[:15],
            "points": s.points,
            "wins": s.wins,
            "losses": s.losses,
            "win_rate": s.win_rate,
        }
        for s in standings
    ]

    df = pl.DataFrame(data_rows)

    chart = alt.Chart(df.to_pandas()).mark_bar().encode(
        x=alt.X("model:N", title="Model", sort="-y"),
        y=alt.Y("points:Q", title="Points"),
        color=alt.Color(
            "win_rate:Q",
            title="Win Rate",
            scale=alt.Scale(scheme="redyellowgreen", domain=[0, 1]),
        ),
        tooltip=["model", "points", "wins", "losses", "win_rate"],
    ).properties(
        title="Tournament Standings",
        width=400,
        height=250,
    )

    return chart


def create_matchup_matrix_heatmap(
    matchup_matrix: 'np.ndarray',
    model_indices: Dict[str, int]
) -> alt.Chart:
    """Create heatmap of head-to-head matchup results.

    Args:
        matchup_matrix: NxN matrix of win rates
        model_indices: Dict mapping model name to matrix index

    Returns:
        Altair Chart
    """
    if matchup_matrix.size == 0:
        return alt.Chart().mark_text().encode(
            text=alt.value("No matchup data available")
        )

    # Invert indices
    index_to_model = {v: k[:12] for k, v in model_indices.items()}

    data_rows = []
    n = matchup_matrix.shape[0]
    for i in range(n):
        for j in range(n):
            if i != j:
                data_rows.append({
                    "player": index_to_model.get(i, f"M{i}"),
                    "opponent": index_to_model.get(j, f"M{j}"),
                    "win_rate": matchup_matrix[i, j],
                })

    if not data_rows:
        return alt.Chart().mark_text().encode(
            text=alt.value("No matchup data")
        )

    df = pl.DataFrame(data_rows)

    chart = alt.Chart(df.to_pandas()).mark_rect().encode(
        x=alt.X("opponent:N", title="Opponent"),
        y=alt.Y("player:N", title="Player"),
        color=alt.Color(
            "win_rate:Q",
            title="Win Rate",
            scale=alt.Scale(scheme="redyellowgreen", domain=[0, 1]),
        ),
        tooltip=["player", "opponent", "win_rate"],
    ).properties(
        title="Head-to-Head Win Rates",
        width=300,
        height=300,
    )

    return chart


# --- Ecosystem Charts ---

def create_ecosystem_timeline(
    states: List['EcosystemState']
) -> alt.Chart:
    """Create stacked area chart of archetype distribution over time.

    Args:
        states: List of EcosystemState snapshots

    Returns:
        Altair Chart
    """
    if not states:
        return alt.Chart().mark_text().encode(
            text=alt.value("No ecosystem data available")
        )

    # Convert to long format
    data_rows = []
    for state in states:
        total = sum(state.archetype_distribution.values())
        for archetype, count in state.archetype_distribution.items():
            data_rows.append({
                "round": state.round_num,
                "archetype": archetype,
                "proportion": count / total if total > 0 else 0,
                "count": count,
            })

    if not data_rows:
        return alt.Chart().mark_text().encode(
            text=alt.value("No archetype data")
        )

    df = pl.DataFrame(data_rows)

    colors = {
        "concentrated": "#e74c3c",
        "uniform": "#3498db",
        "asymmetric": "#9b59b6",
        "hedged": "#2ecc71",
        "unknown": "#95a5a6",
    }

    chart = alt.Chart(df.to_pandas()).mark_area().encode(
        x=alt.X("round:Q", title="Round"),
        y=alt.Y("proportion:Q", title="Proportion", stack="normalize"),
        color=alt.Color(
            "archetype:N",
            title="Strategy Type",
            scale=alt.Scale(
                domain=list(colors.keys()),
                range=list(colors.values())
            ),
        ),
        tooltip=["round", "archetype", "count"],
    ).properties(
        title="Strategy Archetype Distribution Over Time",
        width=500,
        height=250,
    )

    return chart


def create_diversity_timeline(
    states: List['EcosystemState']
) -> alt.Chart:
    """Create line chart of diversity entropy over time.

    Args:
        states: List of EcosystemState snapshots

    Returns:
        Altair Chart
    """
    if not states:
        return alt.Chart().mark_text().encode(
            text=alt.value("No ecosystem data available")
        )

    data_rows = [
        {
            "round": s.round_num,
            "diversity": s.diversity_entropy,
            "concentration": s.avg_concentration,
        }
        for s in states
    ]

    df = pl.DataFrame(data_rows)

    base = alt.Chart(df.to_pandas())

    diversity_line = base.mark_line(color="#3498db", point=True).encode(
        x=alt.X("round:Q", title="Round"),
        y=alt.Y("diversity:Q", title="Diversity Entropy", scale=alt.Scale(domain=[0, 1])),
        tooltip=["round", "diversity"],
    )

    concentration_line = base.mark_line(color="#e74c3c", strokeDash=[5, 5]).encode(
        x="round:Q",
        y=alt.Y("concentration:Q", title="Avg Concentration"),
        tooltip=["round", "concentration"],
    )

    chart = alt.layer(diversity_line, concentration_line).resolve_scale(
        y="independent"
    ).properties(
        title="Ecosystem Diversity & Concentration",
        width=500,
        height=200,
    )

    return chart


# --- Intelligence & Cross-Game Charts ---

def create_intelligence_leaderboard_chart(
    leaderboard_df: 'pl.DataFrame'
) -> alt.Chart:
    """Create bar chart of intelligence proxy scores.

    Args:
        leaderboard_df: DataFrame with model intelligence scores

    Returns:
        Altair Chart
    """
    if leaderboard_df.is_empty():
        return alt.Chart().mark_text().encode(
            text=alt.value("No intelligence data available")
        )

    # Convert to pandas for Altair
    df = leaderboard_df.to_pandas()

    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("model:N", title="Model", sort="-y"),
        y=alt.Y("composite_iq:Q", title="Intelligence Score"),
        color=alt.Color(
            "composite_iq:Q",
            scale=alt.Scale(scheme="viridis"),
            legend=None,
        ),
        tooltip=["model", "composite_iq", "compliance_score", "efficiency_score", "adaptation_score"],
    ).properties(
        title="Model Intelligence Proxy Leaderboard",
        width=400,
        height=250,
    )

    return chart


def create_intelligence_breakdown_chart(
    leaderboard_df: 'pl.DataFrame'
) -> alt.Chart:
    """Create stacked bar chart showing intelligence component breakdown.

    Args:
        leaderboard_df: DataFrame with model intelligence scores

    Returns:
        Altair Chart
    """
    if leaderboard_df.is_empty():
        return alt.Chart().mark_text().encode(
            text=alt.value("No intelligence data available")
        )

    # Melt to long format
    df = leaderboard_df.select([
        "model",
        "compliance_score",
        "efficiency_score",
        "adaptation_score",
        "meta_awareness_score",
    ]).to_pandas()

    df_long = df.melt(
        id_vars=["model"],
        value_vars=["compliance_score", "efficiency_score", "adaptation_score", "meta_awareness_score"],
        var_name="component",
        value_name="score",
    )

    # Clean up component names
    df_long["component"] = df_long["component"].str.replace("_score", "").str.title()

    chart = alt.Chart(df_long).mark_bar().encode(
        x=alt.X("model:N", title="Model"),
        y=alt.Y("score:Q", title="Score", stack="zero"),
        color=alt.Color(
            "component:N",
            title="Component",
            scale=alt.Scale(scheme="category10"),
        ),
        tooltip=["model", "component", "score"],
    ).properties(
        title="Intelligence Score Breakdown",
        width=400,
        height=250,
    )

    return chart


# --- Personality Charts ---

def create_personality_comparison_chart(
    profiles_df: 'pl.DataFrame'
) -> alt.Chart:
    """Create parallel coordinates style chart comparing model personalities.

    Args:
        profiles_df: DataFrame from ModelPersonalityProfiler.compare_personalities()

    Returns:
        Altair Chart
    """
    if profiles_df.is_empty():
        return alt.Chart().mark_text().encode(
            text=alt.value("No personality data available")
        )

    # Normalize metrics to 0-1 for fair comparison
    df = profiles_df.to_pandas()

    metrics = ["bias_score", "symmetry_breaking_score", "consistency_score", "avg_concentration"]
    for metric in metrics:
        if metric in df.columns:
            min_val = df[metric].min()
            max_val = df[metric].max()
            if max_val > min_val:
                df[f"{metric}_norm"] = (df[metric] - min_val) / (max_val - min_val)
            else:
                df[f"{metric}_norm"] = 0.5

    # Melt for parallel coordinates
    df_long = df.melt(
        id_vars=["model"],
        value_vars=[f"{m}_norm" for m in metrics if f"{m}_norm" in df.columns],
        var_name="metric",
        value_name="value",
    )

    df_long["metric"] = df_long["metric"].str.replace("_norm", "").str.replace("_", " ").str.title()

    chart = alt.Chart(df_long).mark_line(point=True).encode(
        x=alt.X("metric:N", title="Personality Dimension"),
        y=alt.Y("value:Q", title="Normalized Score", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("model:N", title="Model"),
        tooltip=["model", "metric", "value"],
    ).properties(
        title="Model Personality Profiles",
        width=400,
        height=250,
    )

    return chart


def create_payoff_sensitivity_chart(
    results: List['PayoffSensitivityResult']
) -> alt.Chart:
    """Create chart showing win rate across payoff variants.

    Args:
        results: List of PayoffSensitivityResult

    Returns:
        Altair Chart
    """
    if not results:
        return alt.Chart().mark_text().encode(
            text=alt.value("No payoff sensitivity data available")
        )

    data_rows = [
        {
            "variant": r.variant_name[:15],
            "win_rate_p1": r.win_rate_p1,
            "strategy_p1": r.dominant_strategy_p1,
            "concentration_p1": r.avg_concentration_p1,
        }
        for r in results
    ]

    df = pl.DataFrame(data_rows)

    chart = alt.Chart(df.to_pandas()).mark_bar().encode(
        x=alt.X("variant:N", title="Payoff Variant", sort=None),
        y=alt.Y("win_rate_p1:Q", title="P1 Win Rate", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color(
            "strategy_p1:N",
            title="Dominant Strategy",
            scale=alt.Scale(
                domain=["concentrated", "uniform", "hedged", "asymmetric", "unknown"],
                range=["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#95a5a6"]
            ),
        ),
        tooltip=["variant", "win_rate_p1", "strategy_p1", "concentration_p1"],
    ).properties(
        title="Win Rate by Payoff Function Variant",
        width=500,
        height=250,
    )

    # Add balance line at 0.5
    balance_line = alt.Chart(pl.DataFrame([{"y": 0.5}]).to_pandas()).mark_rule(
        strokeDash=[5, 5],
        color="#95a5a6",
    ).encode(y="y:Q")

    return chart + balance_line


# =============================================================================
# DEEP COMPARATIVE ALLOCATION ANALYSIS
# =============================================================================

def _parse_allocation(result: Dict, player_num: int) -> Optional[List[float]]:
    """Parse allocation from result dictionary.

    Args:
        result: Round result dictionary
        player_num: Player number (1-indexed)

    Returns:
        List of allocation values or None
    """
    return parse_allocation_from_result(result, player_num)


def create_placement_grid(
    results: List[Dict],
    player_num: int,
    budget: float = 100.0
) -> alt.Chart:
    """Create detailed placement grid with text annotations.

    Shows exact resource allocation per field per round in a heatmap
    with values displayed in cells.

    Args:
        results: List of round result dictionaries
        player_num: Player number (1-indexed)
        budget: Allocation budget (for color scaling)

    Returns:
        Altair Chart with heatmap + text annotations
    """
    data_rows = []
    for round_idx, result in enumerate(results):
        allocation = _parse_allocation(result, player_num)
        if allocation:
            for field_idx, value in enumerate(allocation):
                data_rows.append({
                    "round": round_idx + 1,
                    "field": f"F{field_idx + 1}",
                    "field_num": field_idx + 1,
                    "allocation": float(value),
                    "alloc_label": f"{value:.0f}",
                })

    if not data_rows:
        return alt.Chart().mark_text().encode(
            text=alt.value("No allocation data available")
        )

    df = pl.DataFrame(data_rows)
    num_fields = df["field_num"].max()
    num_rounds = len(results)

    # Base heatmap
    heatmap = alt.Chart(df.to_pandas()).mark_rect(stroke="white", strokeWidth=1).encode(
        x=alt.X("field:N", title="Field", sort=None),
        y=alt.Y("round:O", title="Round", sort="descending"),
        color=alt.Color(
            "allocation:Q",
            title="Units",
            scale=alt.Scale(scheme="blues", domain=[0, budget * 0.6]),
        ),
        tooltip=["round", "field", "allocation"],
    )

    # Text annotations
    text = alt.Chart(df.to_pandas()).mark_text(
        baseline="middle",
        fontSize=11,
        fontWeight="bold"
    ).encode(
        x=alt.X("field:N", sort=None),
        y=alt.Y("round:O", sort="descending"),
        text="alloc_label:N",
        color=alt.condition(
            alt.datum.allocation > budget * 0.35,
            alt.value("white"),
            alt.value("black")
        ),
    )

    model_key = f"player{player_num}_model"
    model = results[0].get(model_key, f"Player {player_num}") if results else f"Player {player_num}"

    return (heatmap + text).properties(
        title=f"P{player_num} Placement Grid ({model[:15]})",
        width=min(300, num_fields * 50),
        height=min(400, num_rounds * 25 + 50),
    )


def create_allocation_comparison(
    results: List[Dict],
    num_players: int = 2,
    aggregate: bool = True
) -> alt.Chart:
    """Create head-to-head allocation comparison chart.

    Shows both players' allocations side by side for direct comparison.

    Args:
        results: List of round result dictionaries
        num_players: Number of players
        aggregate: If True, show averages with error bars. If False, facet by round.

    Returns:
        Altair Chart
    """
    data_rows = []
    for round_idx, result in enumerate(results):
        for player_num in range(1, num_players + 1):
            allocation = _parse_allocation(result, player_num)
            if allocation:
                model_key = f"player{player_num}_model"
                model = result.get(model_key, f"Player {player_num}")

                for field_idx, value in enumerate(allocation):
                    data_rows.append({
                        "round": round_idx + 1,
                        "player": f"P{player_num}: {model[:10]}",
                        "player_num": player_num,
                        "field": f"Field {field_idx + 1}",
                        "field_num": field_idx + 1,
                        "allocation": float(value),
                    })

    if not data_rows:
        return alt.Chart().mark_text().encode(
            text=alt.value("No allocation data available")
        )

    df = pl.DataFrame(data_rows)

    if aggregate:
        # Aggregated view with error bars
        base = alt.Chart(df.to_pandas())

        bars = base.mark_bar().encode(
            x=alt.X("field:N", title="Battlefield", sort=None),
            y=alt.Y("mean(allocation):Q", title="Average Allocation"),
            color=alt.Color("player:N", title="Player"),
            xOffset="player:N",
            tooltip=["player", "field", "mean(allocation):Q", "stdev(allocation):Q"],
        )

        error_bars = base.mark_errorbar(extent="stdev").encode(
            x=alt.X("field:N", sort=None),
            y=alt.Y("allocation:Q"),
            color=alt.Color("player:N"),
            xOffset="player:N",
        )

        chart = (bars + error_bars).properties(
            title="Head-to-Head: Average Allocation by Field",
            width=400,
            height=250,
        )
    else:
        # Faceted by round
        chart = alt.Chart(df.to_pandas()).mark_bar().encode(
            x=alt.X("field:N", title="Field", sort=None),
            y=alt.Y("allocation:Q", title="Allocation"),
            color=alt.Color("player:N", title="Player"),
            xOffset="player:N",
            tooltip=["round", "player", "field", "allocation"],
        ).properties(
            width=150,
            height=120,
        ).facet(
            column=alt.Column("round:O", title="Round"),
        ).properties(
            title="Round-by-Round Allocation Comparison",
        )

    return chart


def create_allocation_difference_heatmap(
    results: List[Dict],
    num_fields: int = None
) -> alt.Chart:
    """Create diverging heatmap showing P1 - P2 allocation differences.

    Positive values (red) = P1 allocated more
    Negative values (blue) = P2 allocated more

    Args:
        results: List of round result dictionaries
        num_fields: Number of fields (auto-detected if None)

    Returns:
        Altair Chart
    """
    data_rows = []
    for round_idx, result in enumerate(results):
        p1_alloc = _parse_allocation(result, 1)
        p2_alloc = _parse_allocation(result, 2)

        if p1_alloc and p2_alloc and len(p1_alloc) == len(p2_alloc):
            for field_idx in range(len(p1_alloc)):
                diff = p1_alloc[field_idx] - p2_alloc[field_idx]
                data_rows.append({
                    "round": round_idx + 1,
                    "field": f"F{field_idx + 1}",
                    "field_num": field_idx + 1,
                    "difference": diff,
                    "p1_alloc": p1_alloc[field_idx],
                    "p2_alloc": p2_alloc[field_idx],
                    "diff_label": f"{diff:+.0f}",
                })

    if not data_rows:
        return alt.Chart().mark_text().encode(
            text=alt.value("No allocation difference data available")
        )

    df = pl.DataFrame(data_rows)
    max_diff = max(abs(df["difference"].max()), abs(df["difference"].min()))
    num_rounds = len(results)
    detected_fields = df["field_num"].max()

    # Heatmap with diverging color scale
    heatmap = alt.Chart(df.to_pandas()).mark_rect(stroke="white", strokeWidth=1).encode(
        x=alt.X("field:N", title="Field", sort=None),
        y=alt.Y("round:O", title="Round", sort="descending"),
        color=alt.Color(
            "difference:Q",
            title="P1 - P2",
            scale=alt.Scale(
                scheme="redblue",
                domain=[-max_diff, max_diff],
                reverse=True,  # Red = P1 higher
            ),
        ),
        tooltip=["round", "field", "p1_alloc", "p2_alloc", "difference"],
    )

    # Text annotations
    text = alt.Chart(df.to_pandas()).mark_text(
        baseline="middle",
        fontSize=10,
        fontWeight="bold"
    ).encode(
        x=alt.X("field:N", sort=None),
        y=alt.Y("round:O", sort="descending"),
        text="diff_label:N",
        color=alt.condition(
            (alt.datum.difference > max_diff * 0.5) | (alt.datum.difference < -max_diff * 0.5),
            alt.value("white"),
            alt.value("black")
        ),
    )

    return (heatmap + text).properties(
        title="Allocation Difference (P1 - P2): Red=P1 Higher, Blue=P2 Higher",
        width=min(350, detected_fields * 55),
        height=min(400, num_rounds * 25 + 50),
    )


def create_allocation_evolution_area(
    results: List[Dict],
    player_num: int
) -> alt.Chart:
    """Create stacked area chart showing allocation proportions over rounds.

    Visualizes how the player shifts resources between fields over time.

    Args:
        results: List of round result dictionaries
        player_num: Player number (1-indexed)

    Returns:
        Altair Chart
    """
    data_rows = []
    for round_idx, result in enumerate(results):
        allocation = _parse_allocation(result, player_num)
        if allocation:
            total = sum(allocation)
            for field_idx, value in enumerate(allocation):
                proportion = value / total if total > 0 else 0
                data_rows.append({
                    "round": round_idx + 1,
                    "field": f"Field {field_idx + 1}",
                    "field_num": field_idx + 1,
                    "allocation": float(value),
                    "proportion": proportion,
                })

    if not data_rows:
        return alt.Chart().mark_text().encode(
            text=alt.value("No allocation data available")
        )

    df = pl.DataFrame(data_rows)
    num_fields = df["field_num"].max()

    # Generate distinct colors for fields
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#f39c12", "#1abc9c", "#e67e22"]
    field_colors = colors[:num_fields] if num_fields <= len(colors) else colors

    model_key = f"player{player_num}_model"
    model = results[0].get(model_key, f"Player {player_num}") if results else f"Player {player_num}"

    chart = alt.Chart(df.to_pandas()).mark_area().encode(
        x=alt.X("round:Q", title="Round", axis=alt.Axis(tickMinStep=1)),
        y=alt.Y("proportion:Q", title="Proportion of Budget", stack="normalize"),
        color=alt.Color(
            "field:N",
            title="Field",
            sort=None,
            scale=alt.Scale(range=field_colors),
        ),
        tooltip=["round", "field", "allocation", alt.Tooltip("proportion:Q", format=".1%")],
    ).properties(
        title=f"P{player_num} Allocation Evolution ({model[:15]})",
        width=450,
        height=200,
    )

    return chart


def create_field_win_analysis(
    results: List[Dict],
    num_players: int = 2
) -> alt.Chart:
    """Create chart analyzing which fields each player tends to win.

    Compares allocations field-by-field and shows win rates per field.

    Args:
        results: List of round result dictionaries
        num_players: Number of players

    Returns:
        Altair Chart
    """
    field_wins = {}  # field_num -> {player_num: wins}

    for result in results:
        p1_alloc = _parse_allocation(result, 1)
        p2_alloc = _parse_allocation(result, 2)

        if p1_alloc and p2_alloc and len(p1_alloc) == len(p2_alloc):
            for field_idx in range(len(p1_alloc)):
                field_num = field_idx + 1
                if field_num not in field_wins:
                    field_wins[field_num] = {1: 0, 2: 0, "ties": 0}

                if p1_alloc[field_idx] > p2_alloc[field_idx]:
                    field_wins[field_num][1] += 1
                elif p2_alloc[field_idx] > p1_alloc[field_idx]:
                    field_wins[field_num][2] += 1
                else:
                    field_wins[field_num]["ties"] += 1

    if not field_wins:
        return alt.Chart().mark_text().encode(
            text=alt.value("No field win data available")
        )

    # Convert to data rows
    data_rows = []
    for field_num, wins in sorted(field_wins.items()):
        total = wins[1] + wins[2] + wins["ties"]
        for player_num in [1, 2]:
            win_rate = wins[player_num] / total if total > 0 else 0
            data_rows.append({
                "field": f"Field {field_num}",
                "player": f"P{player_num}",
                "wins": wins[player_num],
                "win_rate": win_rate,
            })

    df = pl.DataFrame(data_rows)

    chart = alt.Chart(df.to_pandas()).mark_bar().encode(
        x=alt.X("field:N", title="Battlefield", sort=None),
        y=alt.Y("win_rate:Q", title="Field Win Rate", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color(
            "player:N",
            title="Player",
            scale=alt.Scale(range=["#3498db", "#e74c3c"]),
        ),
        xOffset="player:N",
        tooltip=["field", "player", "wins", alt.Tooltip("win_rate:Q", format=".1%")],
    ).properties(
        title="Field Win Rate by Player (Higher Allocation = Win)",
        width=400,
        height=200,
    )

    # Add balance line
    balance = alt.Chart(pl.DataFrame([{"y": 0.5}]).to_pandas()).mark_rule(
        strokeDash=[3, 3],
        color="#95a5a6",
    ).encode(y="y:Q")

    return chart + balance


def create_resource_efficiency_chart(
    results: List[Dict]
) -> alt.Chart:
    """Create chart showing resource efficiency metrics per round.

    Efficiency = fields won / total fields when accounting for overbidding.

    Args:
        results: List of round result dictionaries

    Returns:
        Altair Chart
    """
    data_rows = []

    for round_idx, result in enumerate(results):
        p1_alloc = _parse_allocation(result, 1)
        p2_alloc = _parse_allocation(result, 2)

        if p1_alloc and p2_alloc and len(p1_alloc) == len(p2_alloc):
            num_fields = len(p1_alloc)

            for player_num in [1, 2]:
                my_alloc = p1_alloc if player_num == 1 else p2_alloc
                opp_alloc = p2_alloc if player_num == 1 else p1_alloc

                wins = 0
                overbid_total = 0
                total_spent = sum(my_alloc)

                for i in range(num_fields):
                    if my_alloc[i] > opp_alloc[i]:
                        wins += 1
                        # Overbid = excess resources beyond what was needed
                        overbid = my_alloc[i] - opp_alloc[i] - 1  # -1 for min win margin
                        if overbid > 0:
                            overbid_total += overbid

                efficiency = wins / num_fields if num_fields > 0 else 0
                waste_ratio = overbid_total / total_spent if total_spent > 0 else 0

                model_key = f"player{player_num}_model"
                model = result.get(model_key, f"Player {player_num}")

                data_rows.append({
                    "round": round_idx + 1,
                    "player": f"P{player_num}: {model[:10]}",
                    "player_num": player_num,
                    "efficiency": efficiency,
                    "waste_ratio": waste_ratio,
                    "fields_won": wins,
                    "overbid": overbid_total,
                })

    if not data_rows:
        return alt.Chart().mark_text().encode(
            text=alt.value("No efficiency data available")
        )

    df = pl.DataFrame(data_rows)

    # Efficiency line chart
    chart = alt.Chart(df.to_pandas()).mark_line(point=True).encode(
        x=alt.X("round:Q", title="Round", axis=alt.Axis(tickMinStep=1)),
        y=alt.Y("efficiency:Q", title="Win Rate (Fields Won / Total)", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("player:N", title="Player"),
        strokeDash=alt.StrokeDash("player:N"),
        tooltip=["round", "player", "fields_won", alt.Tooltip("efficiency:Q", format=".1%"), "overbid"],
    ).properties(
        title="Resource Efficiency Over Rounds",
        width=450,
        height=200,
    )

    # Balance line at 50%
    balance = alt.Chart(pl.DataFrame([{"y": 0.5}]).to_pandas()).mark_rule(
        strokeDash=[3, 3],
        color="#95a5a6",
    ).encode(y="y:Q")

    return chart + balance
