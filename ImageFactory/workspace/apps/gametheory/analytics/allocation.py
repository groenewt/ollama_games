"""Allocation game analytics for Burr/continuous action space games.

Provides metrics and analysis for Colonel Blotto, Tennis Coach, Sumo Coach
and other allocation-based games.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from math import log2

from ..core.utils import parse_allocation


@dataclass
class AllocationMetrics:
    """Metrics for a single allocation."""
    concentration_index: float  # HHI: 1/n (uniform) to 1.0 (concentrated)
    entropy: float              # Shannon entropy (higher = more diverse)
    normalized_entropy: float   # Entropy / max_entropy (0-1 scale)
    strategy_type: str          # "concentrated", "uniform", "asymmetric", "random"
    max_field: int              # Field index with highest allocation (0-indexed)
    max_allocation: float       # Value at max field
    spread_score: float         # 1 - HHI (higher = more spread)
    num_fields: int             # Number of fields/targets
    budget: float               # Total allocation sum


@dataclass
class SessionAllocationSummary:
    """Summary of allocation patterns across a session."""
    player_id: int
    model: str
    num_rounds: int
    avg_concentration: float
    avg_entropy: float
    dominant_strategy: str
    strategy_consistency: float  # How often same strategy type used
    field_preferences: List[float]  # Average allocation per field
    metrics_by_round: List[AllocationMetrics] = field(default_factory=list)


@dataclass
class ComplianceMetrics:
    """Metrics for response quality/compliance."""
    total_rounds: int
    parsed_ok: int
    normalized: int
    failed: int
    parse_rate: float          # % successfully parsed
    normalization_rate: float  # % that needed normalization
    failure_rate: float        # % that completely failed


class AllocationAnalyzer:
    """Analyzer for allocation game results."""

    @staticmethod
    def calculate_hhi(allocation: List[float], budget: float = None) -> float:
        """Calculate Herfindahl-Hirschman Index (concentration).

        Returns value between 1/n (perfectly uniform) and 1.0 (all in one field).
        """
        if budget is None:
            budget = sum(allocation)
        if budget == 0:
            return 1.0 / len(allocation)  # Treat as uniform

        proportions = [a / budget for a in allocation]
        return sum(p ** 2 for p in proportions)

    @staticmethod
    def calculate_entropy(allocation: List[float], budget: float = None) -> float:
        """Calculate Shannon entropy of allocation proportions.

        Higher entropy = more diverse allocation.
        """
        if budget is None:
            budget = sum(allocation)
        if budget == 0:
            return 0.0

        proportions = [a / budget for a in allocation if a > 0]
        if not proportions:
            return 0.0

        return -sum(p * log2(p) for p in proportions)

    @staticmethod
    def classify_strategy(hhi: float, num_fields: int) -> str:
        """Classify strategy type based on concentration.

        Args:
            hhi: Herfindahl-Hirschman Index
            num_fields: Number of fields/targets

        Returns:
            Strategy type: "concentrated", "uniform", "asymmetric", or "hedged"
        """
        uniform_hhi = 1.0 / num_fields
        concentrated_threshold = 0.5  # >50% in one field dominates

        if hhi >= concentrated_threshold:
            return "concentrated"
        elif hhi <= uniform_hhi * 1.2:  # Within 20% of perfectly uniform
            return "uniform"
        elif hhi <= uniform_hhi * 2.0:
            return "hedged"  # Moderately spread
        else:
            return "asymmetric"  # Uneven but not fully concentrated

    def analyze_allocation(
        self,
        allocation: List[float],
        budget: float = None
    ) -> AllocationMetrics:
        """Analyze a single allocation.

        Args:
            allocation: List of resource allocations per field
            budget: Expected total (uses sum if not provided)

        Returns:
            AllocationMetrics with computed values
        """
        if budget is None:
            budget = sum(allocation)

        num_fields = len(allocation)
        hhi = self.calculate_hhi(allocation, budget)
        entropy = self.calculate_entropy(allocation, budget)
        max_entropy = log2(num_fields) if num_fields > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        max_field = allocation.index(max(allocation))
        max_allocation = allocation[max_field]

        strategy_type = self.classify_strategy(hhi, num_fields)

        return AllocationMetrics(
            concentration_index=hhi,
            entropy=entropy,
            normalized_entropy=normalized_entropy,
            strategy_type=strategy_type,
            max_field=max_field,
            max_allocation=max_allocation,
            spread_score=1.0 - hhi,
            num_fields=num_fields,
            budget=budget,
        )

    def parse_allocation_from_result(
        self,
        result: Dict,
        player_num: int
    ) -> Optional[List[float]]:
        """Extract allocation list from result dict.

        Args:
            result: Round result dictionary
            player_num: Player number (1-indexed)

        Returns:
            Allocation list or None if not parseable
        """
        # Try allocation field first (raw list)
        alloc_key = f"player{player_num}_allocation"
        if alloc_key in result:
            parsed = parse_allocation(result[alloc_key])
            if parsed is not None:
                return parsed

        # Try action field (might be formatted string like "[50, 25, 25]")
        action_key = f"player{player_num}_action"
        if action_key in result:
            parsed = parse_allocation(result[action_key])
            if parsed is not None:
                return parsed

        logging.debug(f"Could not parse allocation for player {player_num} from result")
        return None

    def analyze_session(
        self,
        results: List[Dict],
        num_players: int = 2,
        budget: float = 100.0
    ) -> Dict[int, SessionAllocationSummary]:
        """Analyze all allocations in a session.

        Args:
            results: List of round result dictionaries
            num_players: Number of players
            budget: Expected allocation budget

        Returns:
            Dict mapping player_num to SessionAllocationSummary
        """
        summaries = {}

        for player_num in range(1, num_players + 1):
            metrics_list = []
            strategy_types = []
            field_totals = None

            # Get model name from first result
            model_key = f"player{player_num}_model"
            model = results[0].get(model_key, f"Player {player_num}") if results else f"Player {player_num}"

            for result in results:
                allocation = self.parse_allocation_from_result(result, player_num)
                if allocation:
                    metrics = self.analyze_allocation(allocation, budget)
                    metrics_list.append(metrics)
                    strategy_types.append(metrics.strategy_type)

                    # Accumulate field totals
                    if field_totals is None:
                        field_totals = [0.0] * len(allocation)
                    for i, val in enumerate(allocation):
                        field_totals[i] += val

            if not metrics_list:
                continue

            num_rounds = len(metrics_list)

            # Calculate averages
            avg_concentration = sum(m.concentration_index for m in metrics_list) / num_rounds
            avg_entropy = sum(m.entropy for m in metrics_list) / num_rounds

            # Dominant strategy (most common type)
            from collections import Counter
            strategy_counts = Counter(strategy_types)
            dominant_strategy = strategy_counts.most_common(1)[0][0]
            strategy_consistency = strategy_counts[dominant_strategy] / num_rounds

            # Field preferences (average per field)
            field_preferences = [t / num_rounds for t in field_totals] if field_totals else []

            summaries[player_num] = SessionAllocationSummary(
                player_id=player_num,
                model=model,
                num_rounds=num_rounds,
                avg_concentration=avg_concentration,
                avg_entropy=avg_entropy,
                dominant_strategy=dominant_strategy,
                strategy_consistency=strategy_consistency,
                field_preferences=field_preferences,
                metrics_by_round=metrics_list,
            )

        return summaries

    def calculate_compliance_metrics(
        self,
        results: List[Dict],
        player_num: int
    ) -> ComplianceMetrics:
        """Calculate parse/normalization compliance metrics.

        Args:
            results: List of round result dictionaries
            player_num: Player number (1-indexed)

        Returns:
            ComplianceMetrics with rates
        """
        total = len(results)
        parsed_ok = 0
        normalized = 0
        failed = 0

        parsed_key = f"player{player_num}_was_parsed"
        normalized_key = f"player{player_num}_was_normalized"

        for result in results:
            was_parsed = result.get(parsed_key, True)
            was_normalized = result.get(normalized_key, False)

            if not was_parsed and not was_normalized:
                failed += 1
            elif was_normalized:
                normalized += 1
            else:
                parsed_ok += 1

        return ComplianceMetrics(
            total_rounds=total,
            parsed_ok=parsed_ok,
            normalized=normalized,
            failed=failed,
            parse_rate=parsed_ok / total if total > 0 else 0.0,
            normalization_rate=normalized / total if total > 0 else 0.0,
            failure_rate=failed / total if total > 0 else 0.0,
        )

    def detect_strategy_evolution(
        self,
        results: List[Dict],
        player_num: int,
        budget: float = 100.0
    ) -> Dict[str, Any]:
        """Detect how strategy changes over rounds.

        Args:
            results: List of round result dictionaries
            player_num: Player number (1-indexed)
            budget: Expected allocation budget

        Returns:
            Dict with evolution analysis
        """
        metrics_list = []

        for result in results:
            allocation = self.parse_allocation_from_result(result, player_num)
            if allocation:
                metrics = self.analyze_allocation(allocation, budget)
                metrics_list.append(metrics)

        if len(metrics_list) < 2:
            return {
                "has_evolution": False,
                "trend": "insufficient_data",
                "explanation": "Need at least 2 rounds for evolution analysis"
            }

        # Compare first half vs second half
        mid = len(metrics_list) // 2
        first_half = metrics_list[:mid]
        second_half = metrics_list[mid:]

        first_concentration = sum(m.concentration_index for m in first_half) / len(first_half)
        second_concentration = sum(m.concentration_index for m in second_half) / len(second_half)

        concentration_change = second_concentration - first_concentration

        # Strategy type shifts
        first_strategies = [m.strategy_type for m in first_half]
        second_strategies = [m.strategy_type for m in second_half]

        from collections import Counter
        first_dominant = Counter(first_strategies).most_common(1)[0][0]
        second_dominant = Counter(second_strategies).most_common(1)[0][0]

        # Determine trend
        if abs(concentration_change) < 0.05:
            trend = "stable"
            explanation = "Strategy concentration remained stable throughout"
        elif concentration_change > 0.1:
            trend = "concentrating"
            explanation = f"Became more concentrated (+{concentration_change:.2f} HHI)"
        elif concentration_change < -0.1:
            trend = "spreading"
            explanation = f"Became more spread (-{abs(concentration_change):.2f} HHI)"
        else:
            trend = "slight_shift"
            explanation = f"Minor concentration change ({concentration_change:+.2f} HHI)"

        strategy_shift = first_dominant != second_dominant

        return {
            "has_evolution": abs(concentration_change) > 0.05 or strategy_shift,
            "trend": trend,
            "concentration_change": concentration_change,
            "first_half_concentration": first_concentration,
            "second_half_concentration": second_concentration,
            "first_half_strategy": first_dominant,
            "second_half_strategy": second_dominant,
            "strategy_shift": strategy_shift,
            "explanation": explanation,
            "concentration_by_round": [m.concentration_index for m in metrics_list],
        }


class CrossSessionAnalyzer:
    """Analyzer for comparing allocation patterns across sessions."""

    def __init__(self, analyzer: AllocationAnalyzer = None):
        self.analyzer = analyzer or AllocationAnalyzer()

    def aggregate_by_hyperparameters(
        self,
        sessions: List[Dict]
    ) -> List[Dict]:
        """Group session metrics by hyperparameter configurations.

        Args:
            sessions: List of session data dictionaries with config and results

        Returns:
            List of aggregated metrics per unique hyperparameter combo
        """
        # Group by (temperature, top_p, top_k, repeat_penalty)
        groups = {}

        for session in sessions:
            config = session.get("config", {})
            players = session.get("players", [])
            results = session.get("results", [])

            if not results:
                continue

            for i, player in enumerate(players):
                player_num = i + 1

                # Extract hyperparameters
                temp = player.get("temperature", 0.7)
                top_p = player.get("top_p", 0.9)
                top_k = player.get("top_k", 40)
                repeat_penalty = player.get("repeat_penalty", 1.1)
                model = player.get("model", "unknown")

                key = (model, temp, top_p, top_k, repeat_penalty)

                if key not in groups:
                    groups[key] = {
                        "model": model,
                        "temperature": temp,
                        "top_p": top_p,
                        "top_k": top_k,
                        "repeat_penalty": repeat_penalty,
                        "sessions": [],
                        "concentrations": [],
                        "compliance_rates": [],
                    }

                # Analyze this player's allocations
                summary = self.analyzer.analyze_session(
                    results, num_players=len(players)
                ).get(player_num)

                if summary:
                    groups[key]["concentrations"].append(summary.avg_concentration)
                    groups[key]["sessions"].append(session.get("session_id", ""))

                compliance = self.analyzer.calculate_compliance_metrics(results, player_num)
                groups[key]["compliance_rates"].append(compliance.parse_rate)

        # Aggregate each group
        aggregated = []
        for key, group in groups.items():
            if group["concentrations"]:
                aggregated.append({
                    **{k: v for k, v in group.items() if k not in ["sessions", "concentrations", "compliance_rates"]},
                    "num_sessions": len(group["sessions"]),
                    "avg_concentration": sum(group["concentrations"]) / len(group["concentrations"]),
                    "avg_compliance": sum(group["compliance_rates"]) / len(group["compliance_rates"]) if group["compliance_rates"] else 0,
                })

        return aggregated

    def compare_models_allocation_style(
        self,
        sessions: List[Dict]
    ) -> Dict[str, Dict]:
        """Compare allocation patterns across different models.

        Args:
            sessions: List of session data dictionaries

        Returns:
            Dict mapping model name to allocation fingerprint
        """
        model_data = {}

        for session in sessions:
            players = session.get("players", [])
            results = session.get("results", [])

            if not results:
                continue

            summaries = self.analyzer.analyze_session(results, num_players=len(players))

            for player_num, summary in summaries.items():
                model = summary.model

                if model not in model_data:
                    model_data[model] = {
                        "concentrations": [],
                        "entropies": [],
                        "strategy_types": [],
                        "field_preferences": [],
                    }

                model_data[model]["concentrations"].append(summary.avg_concentration)
                model_data[model]["entropies"].append(summary.avg_entropy)
                model_data[model]["strategy_types"].append(summary.dominant_strategy)
                if summary.field_preferences:
                    model_data[model]["field_preferences"].append(summary.field_preferences)

        # Calculate fingerprints
        fingerprints = {}
        for model, data in model_data.items():
            from collections import Counter

            fingerprints[model] = {
                "avg_concentration": sum(data["concentrations"]) / len(data["concentrations"]) if data["concentrations"] else 0,
                "avg_entropy": sum(data["entropies"]) / len(data["entropies"]) if data["entropies"] else 0,
                "dominant_strategy": Counter(data["strategy_types"]).most_common(1)[0][0] if data["strategy_types"] else "unknown",
                "num_sessions": len(data["concentrations"]),
            }

            # Average field preferences if available
            if data["field_preferences"]:
                num_fields = len(data["field_preferences"][0])
                avg_prefs = []
                for field_idx in range(num_fields):
                    field_vals = [fp[field_idx] for fp in data["field_preferences"]]
                    avg_prefs.append(sum(field_vals) / len(field_vals))
                fingerprints[model]["avg_field_preferences"] = avg_prefs

        return fingerprints
