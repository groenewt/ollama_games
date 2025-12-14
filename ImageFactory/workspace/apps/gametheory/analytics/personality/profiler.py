"""Model personality profiler - main orchestrator."""

from typing import List, Dict, Any, Optional
from collections import Counter
import numpy as np
import polars as pl

from .types import ModelPersonalityProfile, GameFingerprint
from .bias_analyzer import BiasAnalyzer
from .behavior_analyzer import BehaviorAnalyzer
from .game_fingerprint import GameFingerprintBuilder


class ModelPersonalityProfiler:
    """Builds comprehensive personality profiles from cross-session data.

    Composes specialized analyzers for different aspects of personality:
    - BiasAnalyzer: Concentration bias and symmetry breaking
    - BehaviorAnalyzer: Adaptability, risk tolerance, temporal patterns
    - GameFingerprintBuilder: Per-game allocation patterns
    """

    def __init__(self, session_manager: 'SessionManager'):
        """Initialize profiler with session manager.

        Args:
            session_manager: SessionManager for data access
        """
        self.session_manager = session_manager
        self._bias_analyzer = BiasAnalyzer(session_manager)
        self._behavior_analyzer = BehaviorAnalyzer(session_manager)
        self._fingerprint_builder = GameFingerprintBuilder(session_manager)

    # Delegate methods to specialized analyzers
    def detect_inherent_biases(self, model: str) -> Dict[str, Any]:
        """Analyze clustered vs uniform tendencies."""
        return self._bias_analyzer.detect_inherent_biases(model)

    def detect_symmetry_breaking(
        self,
        model: str,
        game_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Detect if model favors specific fields."""
        return self._bias_analyzer.detect_symmetry_breaking(model, game_type)

    def detect_adaptability(self, model: str) -> Dict[str, Any]:
        """Analyze how model changes strategy based on opponent moves."""
        return self._behavior_analyzer.detect_adaptability(model)

    def detect_risk_tolerance(self, model: str) -> Dict[str, Any]:
        """Analyze variance in allocation patterns."""
        return self._behavior_analyzer.detect_risk_tolerance(model)

    def detect_temporal_patterns(self, model: str) -> Dict[str, Any]:
        """Compare early-game vs late-game behavior."""
        return self._behavior_analyzer.detect_temporal_patterns(model)

    def build_game_fingerprint(self, model: str, game_type: str) -> GameFingerprint:
        """Build allocation fingerprint for a model in a specific game."""
        return self._fingerprint_builder.build_game_fingerprint(model, game_type)

    def build_profile(self, model: str) -> ModelPersonalityProfile:
        """Build complete personality profile for a model.

        Args:
            model: Model name

        Returns:
            ModelPersonalityProfile with all metrics
        """
        df = self._bias_analyzer._load_model_data(model)

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

        # Enhanced personality metrics
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
        if len(concentrations) >= 2:
            concentration_variance = np.var(concentrations)
            consistency_score = max(0, 1.0 - concentration_variance * 10)
        else:
            consistency_score = 1.0 if concentrations else 0.0

        # Overall averages
        avg_concentration = sum(concentrations) / len(concentrations) if concentrations else 0.0
        avg_entropy = sum(entropies) / len(entropies) if entropies else 0.0

        # Dominant strategy overall
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
            descriptions.append("tends toward uniform/spread allocation patterns")
        elif profile.bias_score < -0.3:
            descriptions.append("tends toward concentrated allocation patterns")
        else:
            descriptions.append("shows balanced allocation tendencies")

        # Symmetry description
        if profile.symmetry_breaking_score > 0.15:
            if profile.preferred_fields:
                pref_str = ", ".join([f"Field {f+1}" for f in profile.preferred_fields[:2]])
                descriptions.append(f"prefers {pref_str}")
        else:
            descriptions.append("allocates evenly across fields")

        # Consistency description
        if profile.consistency_score > 0.7:
            descriptions.append("very consistent across different games")
        elif profile.consistency_score < 0.4:
            descriptions.append("adapts strategy significantly by game type")
        else:
            descriptions.append("moderately consistent across games")

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
                "adaptability_score": profile.adaptability_score,
                "risk_tolerance": profile.risk_tolerance,
                "temporal_pattern": profile.temporal_pattern,
                "early_game_concentration": profile.early_game_concentration,
                "late_game_concentration": profile.late_game_concentration,
                "avg_concentration": profile.avg_concentration,
            },
            "game_fingerprints": profile.game_fingerprints,
        }
