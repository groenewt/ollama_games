"""Model personality profiling for allocation games.

Builds profiles of model behavior patterns, detecting inherent biases,
field preferences, and cross-game consistency.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
import numpy as np
from collections import defaultdict
import polars as pl

from ..core.utils import parse_allocation as _parse_allocation_util


@dataclass
class ModelPersonalityProfile:
    """Complete personality profile for a model."""
    model_name: str
    bias_score: float              # -1 (clustered) to +1 (uniform tendency)
    symmetry_breaking_score: float # How much model favors specific fields (0-1)
    preferred_fields: List[int]    # Fields ranked by preference (0-indexed)
    game_fingerprints: Dict[str, Dict]  # game_id -> allocation patterns
    consistency_score: float       # Cross-game stability (0-1)
    total_sessions: int
    total_rounds: int
    avg_concentration: float       # Average HHI across all games
    avg_entropy: float             # Average normalized entropy
    dominant_strategy: str         # Most common strategy type
    # Enhanced personality metrics
    adaptability_score: float = 0.0      # 0-1: How much strategy changes vs opponent
    risk_tolerance: float = 0.5          # 0-1: Preference for high-variance plays
    temporal_pattern: str = "consistent" # "early_aggressive", "late_aggressive", "consistent"
    early_game_concentration: float = 0.0  # Avg HHI in first half of sessions
    late_game_concentration: float = 0.0   # Avg HHI in second half of sessions


@dataclass
class FieldPreference:
    """Field preference analysis for a model."""
    model_name: str
    game_type: str
    num_fields: int
    avg_allocations: List[float]   # Average allocation per field
    std_allocations: List[float]   # Std dev per field
    preference_ranking: List[int]  # Fields ranked by preference (0-indexed)
    preference_strength: float     # How strong the preference is (0-1)


@dataclass
class GameFingerprint:
    """Allocation fingerprint for a model in a specific game."""
    model_name: str
    game_type: str
    avg_concentration: float
    avg_entropy: float
    dominant_strategy: str
    strategy_distribution: Dict[str, float]  # strategy_type -> frequency
    field_preferences: List[float]  # Average proportion per field
    num_sessions: int
    num_rounds: int


class ModelPersonalityProfiler:
    """Builds personality profiles from cross-session data."""

    def __init__(self, session_manager: 'SessionManager'):
        """Initialize profiler with session manager.

        Args:
            session_manager: SessionManager for data access
        """
        self.session_manager = session_manager
        self._cache = {}

    def _load_model_data(
        self,
        model: str,
        game_type: Optional[str] = None
    ) -> pl.DataFrame:
        """Load all sessions where model participated.

        Args:
            model: Model name to filter for
            game_type: Optional game type filter

        Returns:
            DataFrame with model's games
        """
        cache_key = (model, game_type)
        if cache_key in self._cache:
            return self._cache[cache_key]

        df = self.session_manager.load_all_sessions(game_type=game_type)

        if df.is_empty():
            self._cache[cache_key] = df
            return df

        # Find which player positions this model played
        model_filters = []
        for col in df.columns:
            if col.endswith("_model"):
                model_filters.append(pl.col(col) == model)

        if not model_filters:
            self._cache[cache_key] = pl.DataFrame()
            return pl.DataFrame()

        combined_filter = model_filters[0]
        for f in model_filters[1:]:
            combined_filter = combined_filter | f

        result = df.filter(combined_filter)
        self._cache[cache_key] = result
        return result

    def _parse_allocation(self, value: Any) -> Optional[List[float]]:
        """Parse allocation from various formats."""
        return _parse_allocation_util(value)

    def _extract_allocations(
        self,
        df: pl.DataFrame,
        model: str
    ) -> List[Tuple[List[float], str]]:
        """Extract all allocations for a model with game type.

        Args:
            df: DataFrame with game data
            model: Model name

        Returns:
            List of (allocation, game_type) tuples
        """
        allocations = []

        for row in df.iter_rows(named=True):
            game_type = row.get("game_type", "unknown")

            # Find which player this model was
            for p in range(1, 7):  # Support up to 6 players
                model_col = f"player{p}_model"
                alloc_col = f"player{p}_allocation"
                action_col = f"player{p}_action"

                if model_col not in row:
                    break

                if row.get(model_col) == model:
                    # Try allocation column first
                    alloc = None
                    for col in [alloc_col, action_col]:
                        if col in row:
                            alloc = self._parse_allocation(row.get(col))
                            if alloc:
                                break

                    if alloc:
                        allocations.append((alloc, game_type))
                    break

        return allocations

    def detect_inherent_biases(self, model: str) -> Dict[str, Any]:
        """Analyze clustered vs uniform tendencies.

        Args:
            model: Model name

        Returns:
            Dict with bias analysis
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
        # Linear scale between
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
            Dict with symmetry breaking analysis
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
        # Perfectly symmetric = all fields get 1/n
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

    def detect_adaptability(self, model: str) -> Dict[str, Any]:
        """Analyze how model changes strategy based on opponent moves.

        Measures correlation between opponent's previous round outcome and
        model's strategy change in the current round.

        Args:
            model: Model name

        Returns:
            Dict with adaptability analysis
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

            # Get model's allocations and opponent's payoffs
            for idx, row in enumerate(session_df.iter_rows(named=True)):
                if idx == 0:
                    prev_alloc = None
                    continue

                # Find model's player position and extract allocation
                for p in range(1, 7):
                    model_col = f"player{p}_model"
                    if model_col not in row:
                        break
                    if row.get(model_col) == model:
                        curr_alloc = self._parse_allocation(row.get(f"player{p}_allocation") or row.get(f"player{p}_action"))

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
        # High correlation = model adapts to opponent performance
        mean_change = sum(strategy_changes) / len(strategy_changes)
        mean_opp = sum(opponent_outcomes) / len(opponent_outcomes)

        if mean_change > 0 and mean_opp > 0:
            # Normalize and calculate simple correlation proxy
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
            Dict with risk tolerance analysis
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
        # High variance in concentration = risk-seeking (tries different strategies)
        # Low variance = risk-averse (sticks to one approach)
        mean_hhi = sum(hhis) / len(hhis)
        variance_hhi = sum((h - mean_hhi)**2 for h in hhis) / len(hhis)
        std_hhi = variance_hhi ** 0.5

        # Also consider mean HHI: high concentration = risky bets
        # Combine variance and mean for risk score
        # Risk tolerance = 0.5 * (normalized variance) + 0.5 * (mean HHI)
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
            Dict with temporal pattern analysis
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

    def build_game_fingerprint(
        self,
        model: str,
        game_type: str
    ) -> GameFingerprint:
        """Build allocation fingerprint for a model in a specific game.

        Args:
            model: Model name
            game_type: Game type ID

        Returns:
            GameFingerprint for this model/game combination
        """
        df = self._load_model_data(model, game_type)
        allocations = self._extract_allocations(df, model)

        game_allocations = [(a, g) for a, g in allocations if g == game_type]

        if not game_allocations:
            return GameFingerprint(
                model_name=model,
                game_type=game_type,
                avg_concentration=0.0,
                avg_entropy=0.0,
                dominant_strategy="unknown",
                strategy_distribution={},
                field_preferences=[],
                num_sessions=0,
                num_rounds=0,
            )

        concentrations = []
        entropies = []
        strategies = []
        field_totals = None
        num_fields = 0

        for alloc, _ in game_allocations:
            budget = sum(alloc)
            if budget == 0:
                continue

            num_fields = len(alloc)
            proportions = [a / budget for a in alloc]

            # HHI
            hhi = sum(p ** 2 for p in proportions)
            concentrations.append(hhi)

            # Entropy
            from math import log2
            entropy = -sum(p * log2(p) for p in proportions if p > 0)
            max_entropy = log2(num_fields) if num_fields > 1 else 1.0
            norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
            entropies.append(norm_entropy)

            # Strategy classification
            uniform_hhi = 1.0 / num_fields
            if hhi >= 0.5:
                strategy = "concentrated"
            elif hhi <= uniform_hhi * 1.2:
                strategy = "uniform"
            elif hhi <= uniform_hhi * 2.0:
                strategy = "hedged"
            else:
                strategy = "asymmetric"
            strategies.append(strategy)

            # Field totals
            if field_totals is None:
                field_totals = [0.0] * num_fields
            for i, p in enumerate(proportions):
                field_totals[i] += p

        n = len(concentrations)
        if n == 0:
            return GameFingerprint(
                model_name=model,
                game_type=game_type,
                avg_concentration=0.0,
                avg_entropy=0.0,
                dominant_strategy="unknown",
                strategy_distribution={},
                field_preferences=[],
                num_sessions=0,
                num_rounds=0,
            )

        # Strategy distribution
        from collections import Counter
        strategy_counts = Counter(strategies)
        strategy_distribution = {s: c / n for s, c in strategy_counts.items()}
        dominant_strategy = strategy_counts.most_common(1)[0][0]

        # Field preferences
        field_preferences = [t / n for t in field_totals] if field_totals else []

        # Count unique sessions
        session_ids = df.filter(pl.col("game_type") == game_type).select("session_id").unique()
        num_sessions = len(session_ids)

        return GameFingerprint(
            model_name=model,
            game_type=game_type,
            avg_concentration=sum(concentrations) / n,
            avg_entropy=sum(entropies) / n,
            dominant_strategy=dominant_strategy,
            strategy_distribution=strategy_distribution,
            field_preferences=field_preferences,
            num_sessions=num_sessions,
            num_rounds=n,
        )

    def build_profile(self, model: str) -> ModelPersonalityProfile:
        """Build complete personality profile for a model.

        Args:
            model: Model name

        Returns:
            ModelPersonalityProfile with all metrics
        """
        df = self._load_model_data(model)

        if df.is_empty():
            return ModelPersonalityProfile(
                model_name=model,
                bias_score=0.0,
                symmetry_breaking_score=0.0,
                preferred_fields=[],
                game_fingerprints={},
                consistency_score=0.0,
                total_sessions=0,
                total_rounds=0,
                avg_concentration=0.0,
                avg_entropy=0.0,
                dominant_strategy="unknown",
                adaptability_score=0.0,
                risk_tolerance=0.5,
                temporal_pattern="consistent",
                early_game_concentration=0.0,
                late_game_concentration=0.0,
            )

        # Bias analysis
        bias_result = self.detect_inherent_biases(model)
        bias_score = bias_result.get("bias_score", 0.0)

        # Symmetry analysis
        symmetry_result = self.detect_symmetry_breaking(model)
        symmetry_score = symmetry_result.get("symmetry_score", 0.0)
        preferred_fields = symmetry_result.get("preferred_fields", [])

        # NEW: Enhanced personality metrics
        adaptability_result = self.detect_adaptability(model)
        adaptability_score = adaptability_result.get("adaptability_score", 0.0)

        risk_result = self.detect_risk_tolerance(model)
        risk_tolerance = risk_result.get("risk_tolerance", 0.5)

        temporal_result = self.detect_temporal_patterns(model)
        temporal_pattern = temporal_result.get("temporal_pattern", "consistent")
        early_concentration = temporal_result.get("early_concentration", 0.0)
        late_concentration = temporal_result.get("late_concentration", 0.0)

        # Build fingerprints for each game type
        game_types = df.select("game_type").unique().to_series().to_list()
        game_fingerprints = {}

        concentrations = []
        entropies = []
        strategies = []

        for game_type in game_types:
            fp = self.build_game_fingerprint(model, game_type)
            game_fingerprints[game_type] = {
                "avg_concentration": fp.avg_concentration,
                "avg_entropy": fp.avg_entropy,
                "dominant_strategy": fp.dominant_strategy,
                "strategy_distribution": fp.strategy_distribution,
                "field_preferences": fp.field_preferences,
                "num_rounds": fp.num_rounds,
                "num_sessions": fp.num_sessions,
            }

            if fp.num_rounds > 0:
                concentrations.append(fp.avg_concentration)
                entropies.append(fp.avg_entropy)
                strategies.append(fp.dominant_strategy)

        # Calculate cross-game consistency
        # Based on variance in concentration across games
        if len(concentrations) >= 2:
            concentration_variance = np.var(concentrations)
            # Consistency: 0 = highly variable, 1 = very consistent
            # Scale so variance of 0.1 maps to ~0.5 consistency
            consistency_score = max(0, 1.0 - concentration_variance * 10)
        else:
            consistency_score = 1.0 if concentrations else 0.0

        # Overall averages
        avg_concentration = sum(concentrations) / len(concentrations) if concentrations else 0.0
        avg_entropy = sum(entropies) / len(entropies) if entropies else 0.0

        # Dominant strategy overall
        from collections import Counter
        if strategies:
            dominant_strategy = Counter(strategies).most_common(1)[0][0]
        else:
            dominant_strategy = "unknown"

        # Count totals
        total_sessions = df.select("session_id").n_unique()
        total_rounds = len(df)

        return ModelPersonalityProfile(
            model_name=model,
            bias_score=bias_score,
            symmetry_breaking_score=symmetry_score,
            preferred_fields=preferred_fields,
            game_fingerprints=game_fingerprints,
            consistency_score=consistency_score,
            total_sessions=total_sessions,
            total_rounds=total_rounds,
            avg_concentration=avg_concentration,
            avg_entropy=avg_entropy,
            dominant_strategy=dominant_strategy,
            adaptability_score=adaptability_score,
            risk_tolerance=risk_tolerance,
            temporal_pattern=temporal_pattern,
            early_game_concentration=early_concentration,
            late_game_concentration=late_concentration,
        )

    def compare_personalities(
        self,
        models: List[str]
    ) -> pl.DataFrame:
        """Compare personality profiles across models.

        Args:
            models: List of model names to compare

        Returns:
            DataFrame with profile comparison
        """
        profiles = []

        for model in models:
            profile = self.build_profile(model)
            profiles.append({
                "model": profile.model_name,
                "bias_score": profile.bias_score,
                "symmetry_breaking_score": profile.symmetry_breaking_score,
                "consistency_score": profile.consistency_score,
                "avg_concentration": profile.avg_concentration,
                "avg_entropy": profile.avg_entropy,
                "dominant_strategy": profile.dominant_strategy,
                "total_sessions": profile.total_sessions,
                "total_rounds": profile.total_rounds,
            })

        return pl.DataFrame(profiles)

    def get_available_models(self) -> List[str]:
        """Get list of all models with session data.

        Returns:
            List of model names
        """
        df = self.session_manager.load_all_sessions()

        if df.is_empty():
            return []

        models = set()
        for col in df.columns:
            if col.endswith("_model"):
                models.update(df.select(col).unique().to_series().to_list())

        return sorted([m for m in models if m])

    def summarize(self, model: str) -> Dict[str, Any]:
        """Generate human-readable summary of model personality.

        Args:
            model: Model name

        Returns:
            Dict with summary and explanations
        """
        profile = self.build_profile(model)

        # Generate descriptive text
        descriptions = []

        # Bias description
        if profile.bias_score > 0.3:
            descriptions.append(f"tends toward uniform/spread allocation patterns")
        elif profile.bias_score < -0.3:
            descriptions.append(f"tends toward concentrated allocation patterns")
        else:
            descriptions.append(f"shows balanced allocation tendencies")

        # Symmetry description
        if profile.symmetry_breaking_score > 0.15:
            if profile.preferred_fields:
                pref_str = ", ".join([f"Field {f+1}" for f in profile.preferred_fields[:2]])
                descriptions.append(f"prefers {pref_str}")
        else:
            descriptions.append(f"allocates evenly across fields")

        # Consistency description
        if profile.consistency_score > 0.7:
            descriptions.append(f"very consistent across different games")
        elif profile.consistency_score < 0.4:
            descriptions.append(f"adapts strategy significantly by game type")
        else:
            descriptions.append(f"moderately consistent across games")

        # NEW: Enhanced personality descriptions
        # Adaptability
        if profile.adaptability_score > 0.6:
            descriptions.append("highly adaptive to opponent behavior")
        elif profile.adaptability_score < 0.3:
            descriptions.append("follows rigid strategic patterns")

        # Risk tolerance
        if profile.risk_tolerance > 0.6:
            descriptions.append("risk-seeking")
        elif profile.risk_tolerance < 0.4:
            descriptions.append("risk-averse")

        # Temporal patterns
        if profile.temporal_pattern == "early_aggressive":
            descriptions.append("starts aggressive then spreads")
        elif profile.temporal_pattern == "late_aggressive":
            descriptions.append("becomes more concentrated over time")

        summary_text = f"{model} {'; '.join(descriptions)}."

        return {
            "model": model,
            "summary": summary_text,
            "profile": {
                "bias_score": profile.bias_score,
                "symmetry_breaking_score": profile.symmetry_breaking_score,
                "consistency_score": profile.consistency_score,
                "dominant_strategy": profile.dominant_strategy,
                "total_sessions": profile.total_sessions,
                "total_rounds": profile.total_rounds,
                # NEW: Enhanced metrics
                "adaptability_score": profile.adaptability_score,
                "risk_tolerance": profile.risk_tolerance,
                "temporal_pattern": profile.temporal_pattern,
                "early_game_concentration": profile.early_game_concentration,
                "late_game_concentration": profile.late_game_concentration,
                "avg_concentration": profile.avg_concentration,
            },
            "game_fingerprints": profile.game_fingerprints,
        }
