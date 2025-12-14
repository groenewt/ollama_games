"""Game fingerprint building for personality profiling."""

from collections import Counter
from math import log2
import polars as pl

from .base import BasePersonalityAnalyzer
from .types import GameFingerprint


class GameFingerprintBuilder(BasePersonalityAnalyzer):
    """Builds allocation fingerprints for models in specific games."""

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
