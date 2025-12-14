"""Behavioral analysis for personality profiling."""

from typing import Dict, Any
import polars as pl

from .base import BasePersonalityAnalyzer


class BehaviorAnalyzer(BasePersonalityAnalyzer):
    """Analyzes behavioral patterns like adaptability, risk, and temporal changes."""

    def detect_adaptability(self, model: str) -> Dict[str, Any]:
        """Analyze how model changes strategy based on opponent moves.

        Measures correlation between opponent's previous round outcome and
        model's strategy change in the current round.

        Args:
            model: Model name

        Returns:
            Dict with adaptability analysis including:
            - adaptability_detected: Whether significant adaptability found
            - adaptability_score: 0-1 score
            - explanation: Human-readable explanation
        """
        df = self._load_model_data(model)

        if df.is_empty() or len(df) < 3:
            return {
                "model": model,
                "adaptability_detected": False,
                "adaptability_score": 0.5,
                "explanation": "Insufficient data for adaptability analysis",
            }

        # Group by session to analyze round-over-round changes
        session_ids = df.select("session_id").unique().to_series().to_list()
        strategy_changes = []
        opponent_outcomes = []

        for session_id in session_ids:
            session_df = df.filter(pl.col("session_id") == session_id)
            if len(session_df) < 2:
                continue

            prev_alloc = None
            # Get model's allocations and opponent's payoffs
            for idx, row in enumerate(session_df.iter_rows(named=True)):
                if idx == 0:
                    prev_alloc = None
                    # Find model's player position for next iteration
                    for p in range(1, 7):
                        model_col = f"player{p}_model"
                        if model_col not in row:
                            break
                        if row.get(model_col) == model:
                            prev_alloc = self._parse_allocation(
                                row.get(f"player{p}_allocation") or row.get(f"player{p}_action")
                            )
                            break
                    continue

                # Find model's player position and extract allocation
                for p in range(1, 7):
                    model_col = f"player{p}_model"
                    if model_col not in row:
                        break
                    if row.get(model_col) == model:
                        curr_alloc = self._parse_allocation(
                            row.get(f"player{p}_allocation") or row.get(f"player{p}_action")
                        )

                        # Get opponent payoff from previous round
                        prev_row = list(session_df.iter_rows(named=True))[idx - 1]
                        opp_payoffs = []
                        for op in range(1, 7):
                            if op != p and f"player{op}_payoff" in prev_row:
                                opp_payoffs.append(prev_row.get(f"player{op}_payoff", 0) or 0)

                        if curr_alloc and prev_alloc and opp_payoffs:
                            # Calculate strategy change (HHI difference)
                            curr_budget = sum(curr_alloc) if curr_alloc else 1
                            prev_budget = sum(prev_alloc) if prev_alloc else 1
                            if curr_budget > 0 and prev_budget > 0:
                                curr_hhi = sum((a/curr_budget)**2 for a in curr_alloc)
                                prev_hhi = sum((a/prev_budget)**2 for a in prev_alloc)
                                strategy_changes.append(abs(curr_hhi - prev_hhi))
                                opponent_outcomes.append(max(opp_payoffs) if opp_payoffs else 0)

                        prev_alloc = curr_alloc
                        break

        if len(strategy_changes) < 2:
            return {
                "model": model,
                "adaptability_detected": False,
                "adaptability_score": 0.5,
                "explanation": "Insufficient rounds for adaptability analysis",
            }

        # Calculate correlation between opponent success and strategy change
        mean_change = sum(strategy_changes) / len(strategy_changes)
        mean_opp = sum(opponent_outcomes) / len(opponent_outcomes)

        if mean_change > 0 and mean_opp > 0:
            variance_change = sum((c - mean_change)**2 for c in strategy_changes) / len(strategy_changes)
            if variance_change > 0:
                adaptability_score = min(1.0, mean_change * 5)  # Scale to 0-1
            else:
                adaptability_score = 0.5
        else:
            adaptability_score = 0.5

        adaptability_detected = adaptability_score > 0.3

        if adaptability_score > 0.6:
            explanation = f"Model shows high adaptability (score={adaptability_score:.2f})"
        elif adaptability_score < 0.3:
            explanation = f"Model shows rigid strategy patterns (score={adaptability_score:.2f})"
        else:
            explanation = f"Model shows moderate adaptability (score={adaptability_score:.2f})"

        return {
            "model": model,
            "adaptability_detected": adaptability_detected,
            "adaptability_score": adaptability_score,
            "num_transitions": len(strategy_changes),
            "avg_strategy_change": mean_change,
            "explanation": explanation,
        }

    def detect_risk_tolerance(self, model: str) -> Dict[str, Any]:
        """Analyze variance in allocation patterns to detect risk preference.

        High variance = risk-seeking, low variance = risk-averse.

        Args:
            model: Model name

        Returns:
            Dict with risk tolerance analysis including:
            - risk_tolerance: 0-1 score
            - risk_category: "risk_seeking", "risk_averse", or "neutral"
            - explanation: Human-readable explanation
        """
        df = self._load_model_data(model)
        allocations = self._extract_allocations(df, model)

        if len(allocations) < 3:
            return {
                "model": model,
                "risk_tolerance": 0.5,
                "risk_category": "neutral",
                "explanation": "Insufficient data for risk analysis",
            }

        # Calculate HHI for each allocation
        hhis = []
        for alloc, _ in allocations:
            budget = sum(alloc)
            if budget > 0:
                proportions = [a / budget for a in alloc]
                hhi = sum(p ** 2 for p in proportions)
                hhis.append(hhi)

        if len(hhis) < 2:
            return {
                "model": model,
                "risk_tolerance": 0.5,
                "risk_category": "neutral",
                "explanation": "No valid allocations for risk analysis",
            }

        # Calculate variance in HHI as risk proxy
        mean_hhi = sum(hhis) / len(hhis)
        variance_hhi = sum((h - mean_hhi)**2 for h in hhis) / len(hhis)
        std_hhi = variance_hhi ** 0.5

        # Combine variance and mean for risk score
        variance_component = min(1.0, std_hhi * 5)  # Scale std to 0-1
        concentration_component = mean_hhi  # Already 0-1

        risk_tolerance = 0.5 * variance_component + 0.5 * concentration_component

        if risk_tolerance > 0.6:
            risk_category = "risk_seeking"
            explanation = f"Model shows risk-seeking behavior (variance={std_hhi:.3f}, avg HHI={mean_hhi:.3f})"
        elif risk_tolerance < 0.4:
            risk_category = "risk_averse"
            explanation = f"Model shows risk-averse behavior (variance={std_hhi:.3f}, avg HHI={mean_hhi:.3f})"
        else:
            risk_category = "neutral"
            explanation = f"Model shows balanced risk tolerance (variance={std_hhi:.3f}, avg HHI={mean_hhi:.3f})"

        return {
            "model": model,
            "risk_tolerance": risk_tolerance,
            "risk_category": risk_category,
            "hhi_variance": variance_hhi,
            "hhi_std": std_hhi,
            "avg_hhi": mean_hhi,
            "num_allocations": len(hhis),
            "explanation": explanation,
        }

    def detect_temporal_patterns(self, model: str) -> Dict[str, Any]:
        """Compare early-game vs late-game behavior.

        Analyzes whether model becomes more aggressive/conservative over time.

        Args:
            model: Model name

        Returns:
            Dict with temporal pattern analysis including:
            - temporal_pattern: "early_aggressive", "late_aggressive", or "consistent"
            - early_concentration: Average HHI in first half
            - late_concentration: Average HHI in second half
            - explanation: Human-readable explanation
        """
        df = self._load_model_data(model)

        if df.is_empty():
            return {
                "model": model,
                "temporal_pattern": "consistent",
                "early_concentration": 0.0,
                "late_concentration": 0.0,
                "explanation": "No data for temporal analysis",
            }

        # Group by session and split each into early/late halves
        session_ids = df.select("session_id").unique().to_series().to_list()
        early_hhis = []
        late_hhis = []

        for session_id in session_ids:
            session_df = df.filter(pl.col("session_id") == session_id)
            if len(session_df) < 4:  # Need at least 4 rounds to split
                continue

            allocations = self._extract_allocations(session_df, model)
            if len(allocations) < 4:
                continue

            mid_point = len(allocations) // 2
            early_allocs = allocations[:mid_point]
            late_allocs = allocations[mid_point:]

            # Calculate HHI for each half
            for alloc, _ in early_allocs:
                budget = sum(alloc)
                if budget > 0:
                    hhi = sum((a/budget)**2 for a in alloc)
                    early_hhis.append(hhi)

            for alloc, _ in late_allocs:
                budget = sum(alloc)
                if budget > 0:
                    hhi = sum((a/budget)**2 for a in alloc)
                    late_hhis.append(hhi)

        if not early_hhis or not late_hhis:
            return {
                "model": model,
                "temporal_pattern": "consistent",
                "early_concentration": 0.0,
                "late_concentration": 0.0,
                "explanation": "Insufficient data for temporal analysis",
            }

        early_avg = sum(early_hhis) / len(early_hhis)
        late_avg = sum(late_hhis) / len(late_hhis)

        # Determine pattern based on concentration change
        change = late_avg - early_avg
        threshold = 0.05  # 5% change threshold

        if change > threshold:
            pattern = "late_aggressive"
            explanation = f"Model becomes more concentrated late-game ({early_avg:.3f} → {late_avg:.3f})"
        elif change < -threshold:
            pattern = "early_aggressive"
            explanation = f"Model starts concentrated, spreads late-game ({early_avg:.3f} → {late_avg:.3f})"
        else:
            pattern = "consistent"
            explanation = f"Model maintains consistent strategy ({early_avg:.3f} → {late_avg:.3f})"

        return {
            "model": model,
            "temporal_pattern": pattern,
            "early_concentration": early_avg,
            "late_concentration": late_avg,
            "concentration_change": change,
            "num_early_rounds": len(early_hhis),
            "num_late_rounds": len(late_hhis),
            "explanation": explanation,
        }
