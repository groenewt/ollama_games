"""Cross-game comparative analysis and intelligence proxy scoring.

Analyzes strategy transferability across games and computes
composite intelligence metrics for model ranking.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
import numpy as np
from collections import defaultdict
import polars as pl

from ..core.utils import parse_allocation


@dataclass
class StrategyTransferMetrics:
    """Metrics for strategy transfer between games."""
    source_game: str
    target_game: str
    strategy_similarity: float     # Cosine similarity of allocation patterns (-1 to 1)
    performance_correlation: float # Win rate correlation
    concentration_diff: float      # Difference in avg concentration
    transferability_score: float   # Composite transfer score (0-1)


@dataclass
class IntelligenceScores:
    """Intelligence proxy scores for a model."""
    model: str
    compliance_score: float        # Parse success rate (0-1)
    efficiency_score: float        # Payoff efficiency (0-1)
    adaptation_score: float        # Learning/adaptation ability (0-1)
    meta_awareness_score: float    # Opponent modeling ability (0-1)
    composite_iq: float            # Weighted aggregate (0-100)
    rank: int                      # Rank among compared models
    confidence: float              # Confidence based on sample size


@dataclass
class CrossGameComparison:
    """Comparison results across multiple games."""
    model: str
    games_played: List[str]
    overall_win_rate: float
    win_rates_by_game: Dict[str, float]
    concentration_variance: float  # How much strategy varies by game
    best_game: str
    worst_game: str
    specialization_score: float    # 0 = generalist, 1 = specialist


class CrossGameComparativeAnalyzer:
    """Compares model behavior and performance across different game types."""

    def __init__(self, session_manager: 'SessionManager'):
        """Initialize analyzer.

        Args:
            session_manager: SessionManager for data access
        """
        self.session_manager = session_manager
        self._cache = {}

    def _load_model_game_data(
        self,
        model: str,
        game_type: str
    ) -> pl.DataFrame:
        """Load data for a model in a specific game."""
        cache_key = (model, game_type)
        if cache_key in self._cache:
            return self._cache[cache_key]

        df = self.session_manager.load_all_sessions(game_type=game_type)

        if df.is_empty():
            self._cache[cache_key] = df
            return df

        # Filter for games where this model played
        model_filters = []
        for col in df.columns:
            if col.endswith("_model"):
                model_filters.append(pl.col(col) == model)

        if not model_filters:
            result = pl.DataFrame()
        else:
            combined_filter = model_filters[0]
            for f in model_filters[1:]:
                combined_filter = combined_filter | f
            result = df.filter(combined_filter)

        self._cache[cache_key] = result
        return result

    def _parse_allocation(self, value: Any) -> Optional[List[float]]:
        """Parse allocation from various formats."""
        return parse_allocation(value)

    def _extract_model_allocations(
        self,
        df: pl.DataFrame,
        model: str
    ) -> List[List[float]]:
        """Extract all allocations for a model from a DataFrame."""
        allocations = []

        for row in df.iter_rows(named=True):
            for p in range(1, 7):
                model_col = f"player{p}_model"
                if model_col not in row:
                    break
                if row.get(model_col) == model:
                    for col in [f"player{p}_allocation", f"player{p}_action"]:
                        if col in row:
                            alloc = self._parse_allocation(row.get(col))
                            if alloc:
                                allocations.append(alloc)
                                break
                    break

        return allocations

    def _calculate_win_rate(
        self,
        df: pl.DataFrame,
        model: str
    ) -> float:
        """Calculate win rate for a model from DataFrame."""
        if df.is_empty():
            return 0.0

        wins = 0
        total = 0

        for row in df.iter_rows(named=True):
            # Find which player this model was
            model_player = None
            for p in range(1, 7):
                model_col = f"player{p}_model"
                if model_col not in row:
                    break
                if row.get(model_col) == model:
                    model_player = p
                    break

            if model_player is None:
                continue

            # Get payoffs
            model_payoff = row.get(f"player{model_player}_payoff", 0)

            # Compare against all opponents
            opponent_payoffs = []
            for p in range(1, 7):
                if p == model_player:
                    continue
                payoff_col = f"player{p}_payoff"
                if payoff_col in row:
                    opponent_payoffs.append(row.get(payoff_col, 0))

            if opponent_payoffs:
                max_opponent = max(opponent_payoffs)
                if model_payoff > max_opponent:
                    wins += 1
                total += 1

        return wins / total if total > 0 else 0.0

    def calculate_strategy_transfer(
        self,
        model: str,
        source_game: str,
        target_game: str
    ) -> StrategyTransferMetrics:
        """Measure how strategy patterns transfer between games.

        Args:
            model: Model name
            source_game: Source game type
            target_game: Target game type

        Returns:
            StrategyTransferMetrics with similarity and correlation
        """
        source_df = self._load_model_game_data(model, source_game)
        target_df = self._load_model_game_data(model, target_game)

        source_allocs = self._extract_model_allocations(source_df, model)
        target_allocs = self._extract_model_allocations(target_df, model)

        if not source_allocs or not target_allocs:
            return StrategyTransferMetrics(
                source_game=source_game,
                target_game=target_game,
                strategy_similarity=0.0,
                performance_correlation=0.0,
                concentration_diff=0.0,
                transferability_score=0.0,
            )

        # Calculate average allocation proportions for each game
        def avg_proportions(allocs: List[List[float]]) -> List[float]:
            num_fields = len(allocs[0])
            totals = [0.0] * num_fields
            count = 0
            for alloc in allocs:
                if len(alloc) != num_fields:
                    continue
                budget = sum(alloc)
                if budget > 0:
                    for i, a in enumerate(alloc):
                        totals[i] += a / budget
                    count += 1
            return [t / count for t in totals] if count > 0 else totals

        source_avg = avg_proportions(source_allocs)
        target_avg = avg_proportions(target_allocs)

        # Strategy similarity via cosine similarity
        # Handle different field counts by comparing concentration pattern
        def concentration(allocs: List[List[float]]) -> float:
            hhis = []
            for alloc in allocs:
                budget = sum(alloc)
                if budget > 0:
                    props = [a / budget for a in alloc]
                    hhis.append(sum(p ** 2 for p in props))
            return sum(hhis) / len(hhis) if hhis else 0.0

        source_conc = concentration(source_allocs)
        target_conc = concentration(target_allocs)

        # If same field count, use cosine similarity
        if len(source_avg) == len(target_avg) and source_avg and target_avg:
            source_vec = np.array(source_avg)
            target_vec = np.array(target_avg)
            norm_product = np.linalg.norm(source_vec) * np.linalg.norm(target_vec)
            if norm_product > 0:
                strategy_similarity = float(np.dot(source_vec, target_vec) / norm_product)
            else:
                strategy_similarity = 0.0
        else:
            # Different field counts - compare concentration patterns
            # Similarity based on concentration difference
            strategy_similarity = 1.0 - abs(source_conc - target_conc)

        # Performance correlation (win rate comparison)
        source_wr = self._calculate_win_rate(source_df, model)
        target_wr = self._calculate_win_rate(target_df, model)
        performance_correlation = 1.0 - abs(source_wr - target_wr)

        # Concentration difference
        concentration_diff = target_conc - source_conc

        # Composite transferability score
        transferability_score = (strategy_similarity + performance_correlation) / 2

        return StrategyTransferMetrics(
            source_game=source_game,
            target_game=target_game,
            strategy_similarity=strategy_similarity,
            performance_correlation=performance_correlation,
            concentration_diff=concentration_diff,
            transferability_score=transferability_score,
        )

    def calculate_intelligence_proxy(
        self,
        model: str,
        weights: Optional[Dict[str, float]] = None
    ) -> IntelligenceScores:
        """Compute composite intelligence proxy scores.

        Intelligence proxy components:
        1. Compliance (25%): Parse success rate
        2. Efficiency (25%): Payoff per game
        3. Adaptation (25%): Learning/improvement rate
        4. Meta-awareness (25%): Evidence of opponent modeling

        Args:
            model: Model name
            weights: Optional custom weights for components

        Returns:
            IntelligenceScores with all metrics
        """
        weights = weights or {
            "compliance": 0.25,
            "efficiency": 0.25,
            "adaptation": 0.25,
            "meta_awareness": 0.25,
        }

        df = self.session_manager.load_all_sessions()

        if df.is_empty():
            return IntelligenceScores(
                model=model,
                compliance_score=0.0,
                efficiency_score=0.0,
                adaptation_score=0.0,
                meta_awareness_score=0.0,
                composite_iq=0.0,
                rank=0,
                confidence=0.0,
            )

        # Filter for games where this model played
        model_filters = []
        for col in df.columns:
            if col.endswith("_model"):
                model_filters.append(pl.col(col) == model)

        if not model_filters:
            return IntelligenceScores(
                model=model,
                compliance_score=0.0,
                efficiency_score=0.0,
                adaptation_score=0.0,
                meta_awareness_score=0.0,
                composite_iq=0.0,
                rank=0,
                confidence=0.0,
            )

        combined_filter = model_filters[0]
        for f in model_filters[1:]:
            combined_filter = combined_filter | f

        model_df = df.filter(combined_filter)

        if model_df.is_empty():
            return IntelligenceScores(
                model=model,
                compliance_score=0.0,
                efficiency_score=0.0,
                adaptation_score=0.0,
                meta_awareness_score=0.0,
                composite_iq=0.0,
                rank=0,
                confidence=0.0,
            )

        # 1. Compliance Score - parse success rate
        compliance_score = self._calculate_compliance_score(model_df, model)

        # 2. Efficiency Score - win rate normalized
        efficiency_score = self._calculate_efficiency_score(model_df, model)

        # 3. Adaptation Score - improvement over rounds
        adaptation_score = self._calculate_adaptation_score(model_df, model)

        # 4. Meta-awareness Score - evidence of opponent modeling
        meta_awareness_score = self._calculate_meta_awareness_score(model_df, model)

        # Composite IQ (0-100 scale)
        composite_iq = (
            compliance_score * weights["compliance"] +
            efficiency_score * weights["efficiency"] +
            adaptation_score * weights["adaptation"] +
            meta_awareness_score * weights["meta_awareness"]
        ) * 100

        # Confidence based on sample size
        num_rounds = len(model_df)
        confidence = min(1.0, num_rounds / 100)  # Full confidence at 100+ rounds

        return IntelligenceScores(
            model=model,
            compliance_score=compliance_score,
            efficiency_score=efficiency_score,
            adaptation_score=adaptation_score,
            meta_awareness_score=meta_awareness_score,
            composite_iq=composite_iq,
            rank=0,  # Set later in leaderboard
            confidence=confidence,
        )

    def _calculate_compliance_score(
        self,
        df: pl.DataFrame,
        model: str
    ) -> float:
        """Calculate compliance (parse success) score."""
        parsed_total = 0
        total = 0

        for row in df.iter_rows(named=True):
            for p in range(1, 7):
                model_col = f"player{p}_model"
                if model_col not in row:
                    break
                if row.get(model_col) == model:
                    parsed_col = f"player{p}_was_parsed"
                    if parsed_col in row:
                        total += 1
                        if row.get(parsed_col, True):
                            parsed_total += 1
                    break

        return parsed_total / total if total > 0 else 0.5  # Default to 0.5 if no data

    def _calculate_efficiency_score(
        self,
        df: pl.DataFrame,
        model: str
    ) -> float:
        """Calculate efficiency (win rate) score."""
        return self._calculate_win_rate(df, model)

    def _calculate_adaptation_score(
        self,
        df: pl.DataFrame,
        model: str
    ) -> float:
        """Calculate adaptation (learning) score based on win rate improvement."""
        # Group by session and calculate win rate progression
        sessions = df.select("session_id").unique().to_series().to_list()

        if len(sessions) < 2:
            return 0.5  # Neutral score with insufficient data

        session_win_rates = []
        for session_id in sessions:
            session_df = df.filter(pl.col("session_id") == session_id)
            wr = self._calculate_win_rate(session_df, model)
            session_win_rates.append(wr)

        if len(session_win_rates) < 2:
            return 0.5

        # Calculate improvement trend
        x = np.arange(len(session_win_rates))
        slope = np.polyfit(x, session_win_rates, 1)[0]

        # Convert slope to 0-1 scale
        # Positive slope = improving, negative = declining
        # Scale so slope of 0.1 per session = score of 1.0
        adaptation_score = 0.5 + slope * 5  # slope of 0.1 -> 1.0
        return max(0.0, min(1.0, adaptation_score))

    def _calculate_meta_awareness_score(
        self,
        df: pl.DataFrame,
        model: str
    ) -> float:
        """Calculate meta-awareness score based on response to opponent patterns.

        This is a simplified heuristic - checks if model's strategy variance
        correlates with opponent's strategy variance (indicating awareness).
        """
        from .meta_learning import MetaStrategyAnalyzer

        sessions = df.select("session_id").unique().to_series().to_list()

        if not sessions:
            return 0.5

        adaptation_scores = []

        for session_id in sessions:
            session_df = df.filter(pl.col("session_id") == session_id)
            results = session_df.to_dicts()

            if len(results) < 5:
                continue

            # Find which player position the model is
            model_player = None
            for p in range(1, 7):
                model_col = f"player{p}_model"
                if model_col in results[0] and results[0].get(model_col) == model:
                    model_player = p
                    break

            if model_player is None:
                continue

            # Use MetaStrategyAnalyzer
            try:
                analyzer = MetaStrategyAnalyzer(results)
                memory_result = analyzer.analyze_memory_effects(model_player)

                # Correlation magnitude indicates awareness
                if memory_result.significant:
                    adaptation_scores.append(abs(memory_result.overall_correlation))
                else:
                    adaptation_scores.append(0.0)
            except Exception as e:
                logging.debug(f"Meta-awareness analysis failed for session: {e}")

        if not adaptation_scores:
            return 0.5

        # Average correlation magnitude as meta-awareness score
        avg_score = sum(adaptation_scores) / len(adaptation_scores)
        return min(1.0, avg_score * 2)  # Scale up slightly

    def create_intelligence_leaderboard(
        self,
        models: Optional[List[str]] = None
    ) -> pl.DataFrame:
        """Create ranked leaderboard of model intelligence.

        Args:
            models: Optional list of models to include. If None, includes all.

        Returns:
            DataFrame with ranked intelligence scores
        """
        if models is None:
            # Get all models from data
            df = self.session_manager.load_all_sessions()
            if df.is_empty():
                return pl.DataFrame()

            models_set = set()
            for col in df.columns:
                if col.endswith("_model"):
                    models_set.update(df.select(col).unique().to_series().to_list())
            models = sorted([m for m in models_set if m])

        scores = []
        for model in models:
            score = self.calculate_intelligence_proxy(model)
            scores.append({
                "model": score.model,
                "composite_iq": score.composite_iq,
                "compliance_score": score.compliance_score,
                "efficiency_score": score.efficiency_score,
                "adaptation_score": score.adaptation_score,
                "meta_awareness_score": score.meta_awareness_score,
                "confidence": score.confidence,
            })

        if not scores:
            return pl.DataFrame()

        # Sort by composite IQ and assign ranks
        scores.sort(key=lambda x: x["composite_iq"], reverse=True)
        for i, score in enumerate(scores):
            score["rank"] = i + 1

        return pl.DataFrame(scores)

    def analyze_cross_game_performance(
        self,
        model: str
    ) -> CrossGameComparison:
        """Analyze model's performance across all game types.

        Args:
            model: Model name

        Returns:
            CrossGameComparison with performance breakdown
        """
        df = self.session_manager.load_all_sessions()

        if df.is_empty():
            return CrossGameComparison(
                model=model,
                games_played=[],
                overall_win_rate=0.0,
                win_rates_by_game={},
                concentration_variance=0.0,
                best_game="",
                worst_game="",
                specialization_score=0.0,
            )

        # Get game types where model participated
        game_types = df.select("game_type").unique().to_series().to_list()

        win_rates_by_game = {}
        concentrations_by_game = {}
        games_played = []

        for game_type in game_types:
            game_df = self._load_model_game_data(model, game_type)

            if game_df.is_empty():
                continue

            games_played.append(game_type)

            # Win rate
            wr = self._calculate_win_rate(game_df, model)
            win_rates_by_game[game_type] = wr

            # Concentration
            allocations = self._extract_model_allocations(game_df, model)
            if allocations:
                hhis = []
                for alloc in allocations:
                    budget = sum(alloc)
                    if budget > 0:
                        props = [a / budget for a in alloc]
                        hhis.append(sum(p ** 2 for p in props))
                concentrations_by_game[game_type] = sum(hhis) / len(hhis) if hhis else 0

        if not games_played:
            return CrossGameComparison(
                model=model,
                games_played=[],
                overall_win_rate=0.0,
                win_rates_by_game={},
                concentration_variance=0.0,
                best_game="",
                worst_game="",
                specialization_score=0.0,
            )

        # Overall win rate
        overall_win_rate = sum(win_rates_by_game.values()) / len(win_rates_by_game)

        # Best/worst games
        best_game = max(win_rates_by_game.items(), key=lambda x: x[1])[0]
        worst_game = min(win_rates_by_game.items(), key=lambda x: x[1])[0]

        # Concentration variance
        if concentrations_by_game:
            concentration_variance = float(np.var(list(concentrations_by_game.values())))
        else:
            concentration_variance = 0.0

        # Specialization score: high win rate variance = specialist
        win_rate_variance = np.var(list(win_rates_by_game.values())) if win_rates_by_game else 0
        specialization_score = min(1.0, win_rate_variance * 10)  # Scale to 0-1

        return CrossGameComparison(
            model=model,
            games_played=games_played,
            overall_win_rate=overall_win_rate,
            win_rates_by_game=win_rates_by_game,
            concentration_variance=concentration_variance,
            best_game=best_game,
            worst_game=worst_game,
            specialization_score=specialization_score,
        )

    def summarize_all(self) -> Dict[str, Any]:
        """Generate comprehensive cross-game analysis summary.

        Returns:
            Dict with all cross-game analysis
        """
        df = self.session_manager.load_all_sessions()

        if df.is_empty():
            return {"error": "No session data available"}

        # Get all models
        models_set = set()
        for col in df.columns:
            if col.endswith("_model"):
                models_set.update(df.select(col).unique().to_series().to_list())
        models = sorted([m for m in models_set if m])

        # Get game types
        game_types = df.select("game_type").unique().to_series().to_list()

        # Intelligence leaderboard
        leaderboard = self.create_intelligence_leaderboard(models)

        # Cross-game performance
        model_performance = {}
        for model in models:
            comparison = self.analyze_cross_game_performance(model)
            model_performance[model] = {
                "overall_win_rate": comparison.overall_win_rate,
                "win_rates_by_game": comparison.win_rates_by_game,
                "specialization_score": comparison.specialization_score,
                "best_game": comparison.best_game,
            }

        # Strategy transfer matrix (if multiple game types)
        transfer_matrix = {}
        if len(game_types) >= 2 and models:
            for source in game_types:
                transfer_matrix[source] = {}
                for target in game_types:
                    if source != target:
                        # Average transferability across models
                        transfers = []
                        for model in models[:5]:  # Limit for performance
                            transfer = self.calculate_strategy_transfer(model, source, target)
                            transfers.append(transfer.transferability_score)
                        transfer_matrix[source][target] = sum(transfers) / len(transfers) if transfers else 0

        return {
            "num_models": len(models),
            "num_game_types": len(game_types),
            "game_types": game_types,
            "leaderboard": leaderboard.to_dicts() if not leaderboard.is_empty() else [],
            "model_performance": model_performance,
            "strategy_transfer_matrix": transfer_matrix,
        }
