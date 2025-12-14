"""Hyperparameter sensitivity analysis for allocation games.

Analyzes how hyperparameters (temperature, top_k, repeat_penalty)
affect model behavior, compliance, and strategy selection.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
import numpy as np
from collections import defaultdict


@dataclass
class SensitivityResult:
    """Result of sensitivity analysis for a single parameter or parameter pair."""
    parameter: str
    values: List[float]
    compliance_by_value: Dict[float, float]  # param_value -> compliance_rate
    win_rate_by_value: Dict[float, float]
    variance_by_value: Dict[float, float]    # Strategy variance (1-HHI)
    concentration_by_value: Dict[float, float]  # HHI by param value
    sample_counts: Dict[float, int]          # Number of samples per value
    surface_data: Optional[np.ndarray] = None  # 2D grid for interaction effects


@dataclass
class InteractionResult:
    """Result of two-parameter interaction analysis."""
    x_param: str
    y_param: str
    x_values: List[float]
    y_values: List[float]
    metric_surface: np.ndarray  # 2D grid: metric[y_idx, x_idx]
    metric_name: str
    optimal_x: float
    optimal_y: float
    optimal_value: float


@dataclass
class OptimalParams:
    """Optimal hyperparameters for a given objective."""
    objective: str
    parameters: Dict[str, float]
    achieved_value: float
    confidence: float  # Based on sample size


class HyperparameterSensitivityAnalyzer:
    """Analyzes sensitivity of game outcomes to hyperparameter changes."""

    def __init__(self, sweep_summary: 'SweepSummary'):
        """Initialize with sweep results.

        Args:
            sweep_summary: SweepSummary from HyperparameterSweeper.run_sweep()
        """
        self.summary = sweep_summary
        self._indexed_results = None

    def _index_results(self) -> Dict[Tuple, Dict]:
        """Index results by parameter tuple for fast lookup."""
        if self._indexed_results is not None:
            return self._indexed_results

        self._indexed_results = {}
        for result in self.summary.results:
            if "error" in result.metrics:
                continue
            params = result.parameters
            key = tuple(sorted(params.items()))
            self._indexed_results[key] = result

        return self._indexed_results

    def _group_by_parameter(self, param: str) -> Dict[float, List[Dict]]:
        """Group sweep results by a single parameter value.

        Args:
            param: Parameter name (e.g., "temperature")

        Returns:
            Dict mapping param value to list of result metrics
        """
        groups = defaultdict(list)

        for result in self.summary.results:
            if "error" in result.metrics:
                continue
            if param in result.parameters:
                value = result.parameters[param]
                groups[value].append(result.metrics)

        return dict(groups)

    def analyze_compliance_by_repeat_penalty(self) -> SensitivityResult:
        """Analyze how repeat_penalty affects JSON format compliance.

        Returns:
            SensitivityResult with compliance rates by repeat_penalty value
        """
        return self.analyze_single_parameter("repeat_penalty", "parse_rate")

    def analyze_single_parameter(
        self,
        param: str,
        primary_metric: str = "win_rate"
    ) -> SensitivityResult:
        """Analyze sensitivity to a single parameter.

        Args:
            param: Parameter name to analyze
            primary_metric: Primary metric to track (win_rate, parse_rate, etc.)

        Returns:
            SensitivityResult with metrics aggregated by parameter value
        """
        groups = self._group_by_parameter(param)

        if not groups:
            return SensitivityResult(
                parameter=param,
                values=[],
                compliance_by_value={},
                win_rate_by_value={},
                variance_by_value={},
                concentration_by_value={},
                sample_counts={},
            )

        values = sorted(groups.keys())
        compliance_by_value = {}
        win_rate_by_value = {}
        variance_by_value = {}
        concentration_by_value = {}
        sample_counts = {}

        for value in values:
            metrics_list = groups[value]
            n = len(metrics_list)
            sample_counts[value] = n

            # Average compliance (parse_rate)
            parse_rates = [m.get("parse_rate", 0) for m in metrics_list]
            compliance_by_value[value] = sum(parse_rates) / n if n else 0

            # Average win rate
            win_rates = [m.get("win_rate", 0) for m in metrics_list]
            win_rate_by_value[value] = sum(win_rates) / n if n else 0

            # Average concentration (HHI)
            concentrations = [m.get("avg_concentration", 0) for m in metrics_list]
            avg_concentration = sum(concentrations) / n if n else 0
            concentration_by_value[value] = avg_concentration

            # Strategy variance = 1 - HHI (higher = more spread)
            variance_by_value[value] = 1.0 - avg_concentration

        return SensitivityResult(
            parameter=param,
            values=values,
            compliance_by_value=compliance_by_value,
            win_rate_by_value=win_rate_by_value,
            variance_by_value=variance_by_value,
            concentration_by_value=concentration_by_value,
            sample_counts=sample_counts,
        )

    def analyze_strategy_variance(
        self,
        x_param: str = "temperature",
        y_param: str = "top_k"
    ) -> InteractionResult:
        """Analyze strategy variance across two parameters.

        Creates a 2D surface showing how strategy variance (1-HHI)
        changes with both parameters.

        Args:
            x_param: Parameter for X axis
            y_param: Parameter for Y axis

        Returns:
            InteractionResult with 2D surface data
        """
        return self.analyze_interaction(x_param, y_param, "avg_concentration", invert=True)

    def analyze_interaction(
        self,
        x_param: str,
        y_param: str,
        metric: str,
        invert: bool = False
    ) -> InteractionResult:
        """Analyze interaction effects between two parameters.

        Args:
            x_param: Parameter for X axis
            y_param: Parameter for Y axis
            metric: Metric to analyze (win_rate, parse_rate, avg_concentration)
            invert: If True, return 1-metric (useful for concentration -> variance)

        Returns:
            InteractionResult with 2D surface and optimal values
        """
        # Collect unique values for each parameter
        x_values = sorted(set(
            r.parameters.get(x_param)
            for r in self.summary.results
            if x_param in r.parameters and "error" not in r.metrics
        ))
        y_values = sorted(set(
            r.parameters.get(y_param)
            for r in self.summary.results
            if y_param in r.parameters and "error" not in r.metrics
        ))

        if not x_values or not y_values:
            return InteractionResult(
                x_param=x_param,
                y_param=y_param,
                x_values=[],
                y_values=[],
                metric_surface=np.array([[]]),
                metric_name=metric,
                optimal_x=0,
                optimal_y=0,
                optimal_value=0,
            )

        # Build 2D surface
        surface = np.full((len(y_values), len(x_values)), np.nan)
        x_idx_map = {v: i for i, v in enumerate(x_values)}
        y_idx_map = {v: i for i, v in enumerate(y_values)}

        # Aggregate metrics for each (x, y) combination
        cell_values = defaultdict(list)
        for result in self.summary.results:
            if "error" in result.metrics:
                continue
            x_val = result.parameters.get(x_param)
            y_val = result.parameters.get(y_param)
            if x_val is not None and y_val is not None:
                metric_val = result.metrics.get(metric, 0)
                if invert:
                    metric_val = 1.0 - metric_val
                cell_values[(x_val, y_val)].append(metric_val)

        # Fill surface with averages
        for (x_val, y_val), values in cell_values.items():
            x_idx = x_idx_map[x_val]
            y_idx = y_idx_map[y_val]
            surface[y_idx, x_idx] = sum(values) / len(values)

        # Find optimal (maximum) value
        valid_mask = ~np.isnan(surface)
        if valid_mask.any():
            max_idx = np.unravel_index(np.nanargmax(surface), surface.shape)
            optimal_y = y_values[max_idx[0]]
            optimal_x = x_values[max_idx[1]]
            optimal_value = surface[max_idx]
        else:
            optimal_x, optimal_y, optimal_value = 0, 0, 0

        metric_name = f"1 - {metric}" if invert else metric

        return InteractionResult(
            x_param=x_param,
            y_param=y_param,
            x_values=x_values,
            y_values=y_values,
            metric_surface=surface,
            metric_name=metric_name,
            optimal_x=optimal_x,
            optimal_y=optimal_y,
            optimal_value=optimal_value,
        )

    def generate_surface_data(
        self,
        x_param: str = "temperature",
        y_param: str = "top_k",
        metric: str = "win_rate"
    ) -> np.ndarray:
        """Generate 2D surface data for visualization.

        Args:
            x_param: Parameter for X axis
            y_param: Parameter for Y axis
            metric: Metric to visualize

        Returns:
            2D numpy array suitable for heatmap/contour plots
        """
        result = self.analyze_interaction(x_param, y_param, metric)
        return result.metric_surface

    def find_optimal_params(
        self,
        objective: str = "win_rate",
        constraints: Optional[Dict[str, float]] = None
    ) -> OptimalParams:
        """Find optimal hyperparameters for a given objective.

        Args:
            objective: Metric to maximize (win_rate, parse_rate)
            constraints: Optional minimum values for other metrics
                        e.g., {"parse_rate": 0.8} requires 80% compliance

        Returns:
            OptimalParams with best configuration found
        """
        constraints = constraints or {}
        best_result = None
        best_value = float('-inf')

        for result in self.summary.results:
            if "error" in result.metrics:
                continue

            # Check constraints
            meets_constraints = True
            for constraint_metric, min_val in constraints.items():
                if result.metrics.get(constraint_metric, 0) < min_val:
                    meets_constraints = False
                    break

            if not meets_constraints:
                continue

            # Check objective
            obj_value = result.metrics.get(objective, 0)
            if obj_value > best_value:
                best_value = obj_value
                best_result = result

        if best_result is None:
            return OptimalParams(
                objective=objective,
                parameters={},
                achieved_value=0,
                confidence=0,
            )

        # Calculate confidence based on sample consistency
        param_key = tuple(sorted(best_result.parameters.items()))
        similar_results = [
            r for r in self.summary.results
            if tuple(sorted(r.parameters.items())) == param_key
            and "error" not in r.metrics
        ]
        n_samples = len(similar_results)
        confidence = min(1.0, n_samples / 5)  # Full confidence at 5+ samples

        return OptimalParams(
            objective=objective,
            parameters=best_result.parameters.copy(),
            achieved_value=best_value,
            confidence=confidence,
        )

    def get_parameter_importance(self) -> Dict[str, float]:
        """Estimate relative importance of each parameter.

        Uses variance in outcomes across parameter values as proxy for importance.

        Returns:
            Dict mapping parameter name to importance score (0-1)
        """
        importance = {}
        all_params = set()

        for result in self.summary.results:
            all_params.update(result.parameters.keys())

        for param in all_params:
            sensitivity = self.analyze_single_parameter(param, "win_rate")
            if sensitivity.values:
                win_rates = list(sensitivity.win_rate_by_value.values())
                variance = np.var(win_rates) if len(win_rates) > 1 else 0
                importance[param] = float(variance)

        # Normalize to 0-1 scale
        max_importance = max(importance.values()) if importance else 1
        if max_importance > 0:
            importance = {k: v / max_importance for k, v in importance.items()}

        return importance

    def summarize(self) -> Dict[str, Any]:
        """Generate a summary of sensitivity analysis.

        Returns:
            Dict with key findings from sensitivity analysis
        """
        summary = {
            "sweep_id": self.summary.sweep_id,
            "game_id": self.summary.game_id,
            "total_configs": self.summary.total_configs,
            "completed_configs": self.summary.completed_configs,
            "parameter_importance": self.get_parameter_importance(),
        }

        # Analyze each parameter in the grid
        param_sensitivities = {}
        for param in self.summary.parameter_grid.keys():
            result = self.analyze_single_parameter(param)
            if result.values:
                param_sensitivities[param] = {
                    "values": result.values,
                    "compliance_range": (
                        min(result.compliance_by_value.values()),
                        max(result.compliance_by_value.values()),
                    ),
                    "win_rate_range": (
                        min(result.win_rate_by_value.values()),
                        max(result.win_rate_by_value.values()),
                    ),
                }

        summary["parameter_sensitivities"] = param_sensitivities

        # Find optimal configurations
        summary["optimal_for_win_rate"] = self.find_optimal_params("win_rate").__dict__
        summary["optimal_for_compliance"] = self.find_optimal_params("parse_rate").__dict__
        summary["optimal_balanced"] = self.find_optimal_params(
            "win_rate",
            constraints={"parse_rate": 0.7}
        ).__dict__

        return summary
