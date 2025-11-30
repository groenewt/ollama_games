"""Analytics service for orchestrating game theory analytics."""

from typing import Dict, List, Any, Optional
import polars as pl

from ..metrics.persistence import SessionManager, CrossGameAnalyzer
from ..metrics.tracker import MetricsTracker


class AnalyticsService:
    """Service class that orchestrates analytics data retrieval."""

    # Actions considered cooperative across different game types
    COOPERATIVE_ACTIONS = [
        "cooperate", "Cooperate", "COOPERATE",
        "contribute", "Contribute", "CONTRIBUTE",
        "trust", "Trust", "TRUST",
        "share", "Share", "SHARE",
        "stag", "Stag", "STAG",
    ]

    def __init__(
        self,
        session_manager: SessionManager,
        metrics_tracker: MetricsTracker,
    ):
        """Initialize the analytics service.

        Args:
            session_manager: The session manager for data access.
            metrics_tracker: The metrics tracker for API metrics.
        """
        self.session_manager = session_manager
        self.metrics = metrics_tracker
        self.analyzer = CrossGameAnalyzer(session_manager)

    def get_dashboard_data(self, game_type: Optional[str] = None) -> Dict[str, Any]:
        """Return all data needed for dashboard overview.

        Args:
            game_type: Optional game type to filter by.

        Returns:
            Dictionary containing:
            - sessions: List of session metadata
            - sessions_count: Number of sessions
            - cumulative: Cumulative API metrics
            - leaderboard: Model leaderboard DataFrame
            - game_summary: Game type summary DataFrame
            - has_data: Boolean indicating if any data exists
            - filtered_game_type: The game type filter applied (if any)
        """
        sessions = self.session_manager.list_sessions()
        cumulative = self.metrics.get_cumulative()

        # Filter sessions by game type if specified
        if game_type:
            sessions = [s for s in sessions if s.get("game_type") == game_type]

        # Get filtered data for analytics
        all_data = self.session_manager.load_all_sessions(game_type=game_type)

        # Calculate leaderboard from filtered data
        if all_data.is_empty():
            leaderboard = pl.DataFrame()
            game_summary = pl.DataFrame()
        else:
            # Leaderboard from filtered data
            p1_stats = all_data.select([
                pl.col("player1_model").alias("model"),
                pl.col("player1_payoff").alias("payoff"),
                pl.col("game_type"),
            ])
            p2_stats = all_data.select([
                pl.col("player2_model").alias("model"),
                pl.col("player2_payoff").alias("payoff"),
                pl.col("game_type"),
            ])
            all_plays = pl.concat([p1_stats, p2_stats])
            leaderboard = all_plays.group_by("model").agg([
                pl.len().alias("total_plays"),
                pl.col("game_type").n_unique().alias("games_played"),
                pl.col("payoff").mean().alias("avg_payoff"),
                pl.col("payoff").sum().alias("total_payoff"),
            ]).sort("avg_payoff", descending=True)

            # Game summary from filtered data
            game_summary = all_data.group_by("game_type").agg([
                pl.col("session_id").n_unique().alias("total_sessions"),
                pl.len().alias("total_rounds"),
                pl.col("player1_payoff").mean().alias("avg_p1_payoff"),
                pl.col("player2_payoff").mean().alias("avg_p2_payoff"),
            ]).sort("total_sessions", descending=True)

        has_data = bool(sessions) or cumulative["total_requests"] > 0

        return {
            "sessions": sessions,
            "sessions_count": len(sessions),
            "cumulative": cumulative,
            "leaderboard": leaderboard,
            "game_summary": game_summary,
            "has_data": has_data,
            "filtered_game_type": game_type,
        }

    def get_response_times(self) -> List[float]:
        """Return combined cumulative + current response times.

        Returns:
            List of all response times across all sessions.
        """
        # Access the internal response times (current + historical)
        all_times = list(self.metrics._all_response_times) + list(self.metrics.response_times)
        return all_times

    def get_cooperation_rates(self, game_type: Optional[str] = None) -> pl.DataFrame:
        """Calculate cooperation rates by model and game type.

        Args:
            game_type: Optional game type to filter by.

        Returns:
            DataFrame with columns: model, game_type, cooperation_rate, total_decisions
        """
        all_data = self.session_manager.load_all_sessions(game_type=game_type)

        if all_data.is_empty():
            return pl.DataFrame()

        # Check if action columns exist
        if "player1_action" not in all_data.columns:
            return pl.DataFrame()

        # Combine player1 and player2 data
        p1_data = all_data.select([
            pl.col("player1_model").alias("model"),
            pl.col("player1_action").alias("action"),
            pl.col("game_type"),
        ])

        p2_data = all_data.select([
            pl.col("player2_model").alias("model"),
            pl.col("player2_action").alias("action"),
            pl.col("game_type"),
        ])

        combined = pl.concat([p1_data, p2_data])

        if combined.is_empty():
            return pl.DataFrame()

        # Calculate cooperation rate per model and game type
        coop_stats = combined.group_by(["model", "game_type"]).agg([
            (
                pl.col("action").is_in(self.COOPERATIVE_ACTIONS).sum()
                / pl.len() * 100
            ).alias("cooperation_rate"),
            pl.len().alias("total_decisions"),
        ])

        return coop_stats.sort(["model", "game_type"])

    def get_model_comparison_data(self, game_type: Optional[str] = None) -> pl.DataFrame:
        """Aggregate model performance across game types for heatmap.

        Args:
            game_type: Optional game type to filter by.

        Returns:
            DataFrame with columns: model, game_type, avg_payoff, total_payoff, games_played
        """
        all_data = self.session_manager.load_all_sessions(game_type=game_type)

        if all_data.is_empty():
            return pl.DataFrame()

        # Combine player1 and player2 payoffs
        p1_data = all_data.select([
            pl.col("player1_model").alias("model"),
            pl.col("player1_payoff").alias("payoff"),
            pl.col("game_type"),
        ])

        p2_data = all_data.select([
            pl.col("player2_model").alias("model"),
            pl.col("player2_payoff").alias("payoff"),
            pl.col("game_type"),
        ])

        combined = pl.concat([p1_data, p2_data])

        if combined.is_empty():
            return pl.DataFrame()

        # Aggregate by model and game type
        model_game_stats = combined.group_by(["model", "game_type"]).agg([
            pl.col("payoff").mean().alias("avg_payoff"),
            pl.col("payoff").sum().alias("total_payoff"),
            pl.len().alias("games_played"),
        ])

        return model_game_stats.sort(["model", "game_type"])

    def get_session_results(self, session_id: str) -> Optional[pl.DataFrame]:
        """Load specific session for detail view.

        Args:
            session_id: The session identifier.

        Returns:
            DataFrame with session results, or None if not found.
        """
        try:
            return self.session_manager.load_session(session_id)
        except FileNotFoundError:
            return None

    def get_all_session_data(self) -> pl.DataFrame:
        """Load all session data.

        Returns:
            Combined DataFrame of all sessions.
        """
        return self.session_manager.load_all_sessions()
