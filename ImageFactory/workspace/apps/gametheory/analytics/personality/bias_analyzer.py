"""Bias and symmetry analysis for personality profiling."""

from typing import Dict, Any, Optional
from collections import defaultdict

from .base import BasePersonalityAnalyzer


class BiasAnalyzer(BasePersonalityAnalyzer):
    """Analyzes allocation biases and symmetry breaking patterns."""

    def detect_inherent_biases(self, model: str) -> Dict[str, Any]:
        """Analyze clustered vs uniform tendencies.

        Args:
            model: Model name

        Returns:
            Dict with bias analysis including:
            - bias_detected: Whether significant bias was found
            - bias_score: -1 (clustered) to +1 (uniform)
            - tendency: "clustered", "uniform", or "balanced"
            - avg_concentration: Average HHI
            - explanation: Human-readable explanation
        """
        df = self._load_model_data(model)
        allocations = self._extract_allocations(df, model)

        if not allocations:
            return {
                "model": model,
                "bias_detected": False,
                "bias_score": 0.0,
                "explanation": "No allocation data available",
            }

        concentrations = []
        for alloc, _ in allocations:
            budget = sum(alloc)
            if budget > 0:
                proportions = [a / budget for a in alloc]
                hhi = sum(p ** 2 for p in proportions)
                concentrations.append(hhi)

        if not concentrations:
            return {
                "model": model,
                "bias_detected": False,
                "bias_score": 0.0,
                "explanation": "No valid allocations",
            }

        avg_concentration = sum(concentrations) / len(concentrations)

        # Bias score: -1 (always concentrated) to +1 (always uniform)
        # At HHI = 0.2 (uniform for 5 fields), bias = +1
        # At HHI = 0.5 (concentrated), bias = -1
        uniform_hhi = 0.2
        concentrated_hhi = 0.5

        if avg_concentration <= uniform_hhi:
            bias_score = 1.0
        elif avg_concentration >= concentrated_hhi:
            bias_score = -1.0
        else:
            # Linear interpolation
            bias_score = 1.0 - 2.0 * (avg_concentration - uniform_hhi) / (concentrated_hhi - uniform_hhi)

        bias_detected = abs(bias_score) > 0.3

        if bias_score > 0.3:
            tendency = "uniform"
            explanation = f"Model tends toward uniform allocation (avg HHI={avg_concentration:.3f})"
        elif bias_score < -0.3:
            tendency = "clustered"
            explanation = f"Model tends toward concentrated allocation (avg HHI={avg_concentration:.3f})"
        else:
            tendency = "balanced"
            explanation = f"Model shows balanced allocation patterns (avg HHI={avg_concentration:.3f})"

        return {
            "model": model,
            "bias_detected": bias_detected,
            "bias_score": bias_score,
            "tendency": tendency,
            "avg_concentration": avg_concentration,
            "num_allocations": len(concentrations),
            "explanation": explanation,
        }

    def detect_symmetry_breaking(
        self,
        model: str,
        game_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Detect if model favors specific fields.

        Args:
            model: Model name
            game_type: Optional game type filter

        Returns:
            Dict with symmetry breaking analysis including:
            - symmetry_breaking_detected: Whether significant asymmetry found
            - symmetry_score: 0 (symmetric) to 1 (highly asymmetric)
            - preferred_fields: Fields ranked by preference
            - explanation: Human-readable explanation
        """
        df = self._load_model_data(model, game_type)
        allocations = self._extract_allocations(df, model)

        if not allocations:
            return {
                "model": model,
                "symmetry_breaking_detected": False,
                "symmetry_score": 0.0,
                "preferred_fields": [],
                "explanation": "No allocation data available",
            }

        # Group allocations by number of fields
        by_num_fields = defaultdict(list)
        for alloc, _ in allocations:
            num_fields = len(alloc)
            by_num_fields[num_fields].append(alloc)

        # Analyze the most common field count
        most_common = max(by_num_fields.items(), key=lambda x: len(x[1]))
        num_fields, field_allocations = most_common

        # Calculate average allocation per field
        field_totals = [0.0] * num_fields
        valid_count = 0

        for alloc in field_allocations:
            budget = sum(alloc)
            if budget > 0:
                for i, a in enumerate(alloc):
                    field_totals[i] += a / budget
                valid_count += 1

        if valid_count == 0:
            return {
                "model": model,
                "symmetry_breaking_detected": False,
                "symmetry_score": 0.0,
                "preferred_fields": [],
                "explanation": "No valid allocations",
            }

        avg_proportions = [t / valid_count for t in field_totals]

        # Calculate symmetry breaking score
        uniform = 1.0 / num_fields
        deviations = [abs(p - uniform) for p in avg_proportions]
        max_deviation = max(deviations)

        # Symmetry score: 0 = perfectly symmetric, 1 = highly asymmetric
        symmetry_score = min(1.0, max_deviation * num_fields)

        # Rank fields by preference
        field_rankings = sorted(range(num_fields), key=lambda i: avg_proportions[i], reverse=True)

        symmetry_breaking_detected = symmetry_score > 0.15

        if symmetry_breaking_detected:
            top_field = field_rankings[0]
            explanation = f"Model prefers Field {top_field + 1} ({avg_proportions[top_field]:.1%} vs uniform {uniform:.1%})"
        else:
            explanation = "Model allocates relatively evenly across fields"

        return {
            "model": model,
            "symmetry_breaking_detected": symmetry_breaking_detected,
            "symmetry_score": symmetry_score,
            "preferred_fields": field_rankings,
            "avg_proportions": avg_proportions,
            "num_fields": num_fields,
            "num_allocations": valid_count,
            "explanation": explanation,
        }
