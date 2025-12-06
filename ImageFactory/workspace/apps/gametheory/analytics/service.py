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

    @staticmethod
    def _get_player_columns(df: pl.DataFrame) -> List[int]:
        """Detect which player numbers have columns in the DataFrame.

        Args:
            df: DataFrame to check for player columns.

        Returns:
            List of player numbers (1-indexed) that exist in the DataFrame.
        """
        players = []
        p = 1
        while f"player{p}_model" in df.columns or f"player{p}_payoff" in df.columns:
            players.append(p)
            p += 1
        return players if players else [1, 2]  # Default to 2-player for backwards compat

    def get_dashboard_data(
        self,
        game_type: Optional[str] = None,
        uses_custom_payoffs: Optional[bool] = None,
        runtime_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return all data needed for dashboard overview.

        Args:
            game_type: Optional game type to filter by.
            uses_custom_payoffs: Optional filter for custom payoffs sessions.
            runtime_mode: Optional filter for runtime mode (one-off, repeated, sequential).

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

        # Apply additional filters to the data
        if not all_data.is_empty():
            if uses_custom_payoffs is not None and "uses_custom_payoffs" in all_data.columns:
                all_data = all_data.filter(pl.col("uses_custom_payoffs") == uses_custom_payoffs)
            if runtime_mode is not None and "runtime_mode" in all_data.columns:
                all_data = all_data.filter(pl.col("runtime_mode") == runtime_mode)

        # Calculate leaderboard from filtered data
        if all_data.is_empty():
            leaderboard = pl.DataFrame()
            game_summary = pl.DataFrame()
        else:
            # Detect players dynamically
            player_nums = self._get_player_columns(all_data)

            # Leaderboard from filtered data - support N players
            player_dfs = []
            for p in player_nums:
                model_col = f"player{p}_model"
                payoff_col = f"player{p}_payoff"
                if model_col in all_data.columns and payoff_col in all_data.columns:
                    p_df = all_data.select([
                        pl.col(model_col).alias("model"),
                        pl.col(payoff_col).alias("payoff"),
                        pl.col("game_type"),
                    ])
                    player_dfs.append(p_df)

            all_plays = pl.concat(player_dfs, how="diagonal") if player_dfs else pl.DataFrame()

            if all_plays.is_empty():
                leaderboard = pl.DataFrame()
            else:
                leaderboard = all_plays.group_by("model").agg([
                    pl.len().alias("total_plays"),
                    pl.col("game_type").n_unique().alias("games_played"),
                    pl.col("payoff").mean().alias("avg_payoff"),
                    pl.col("payoff").sum().alias("total_payoff"),
                ]).sort("avg_payoff", descending=True)

            # Game summary from filtered data - dynamic player stats
            agg_exprs = [
                pl.col("session_id").n_unique().alias("total_sessions"),
                pl.len().alias("total_rounds"),
            ]
            for p in player_nums:
                payoff_col = f"player{p}_payoff"
                if payoff_col in all_data.columns:
                    agg_exprs.append(pl.col(payoff_col).mean().alias(f"avg_p{p}_payoff"))

            game_summary = all_data.group_by("game_type").agg(agg_exprs).sort("total_sessions", descending=True)

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

        # Detect players dynamically
        player_nums = self._get_player_columns(all_data)

        # Combine all player data
        player_dfs = []
        for p in player_nums:
            model_col = f"player{p}_model"
            action_col = f"player{p}_action"
            if model_col in all_data.columns and action_col in all_data.columns:
                p_data = all_data.select([
                    pl.col(model_col).alias("model"),
                    pl.col(action_col).alias("action"),
                    pl.col("game_type"),
                ])
                player_dfs.append(p_data)

        combined = pl.concat(player_dfs, how="diagonal") if player_dfs else pl.DataFrame()

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

        # Detect players dynamically
        player_nums = self._get_player_columns(all_data)

        # Combine all player payoffs
        player_dfs = []
        for p in player_nums:
            model_col = f"player{p}_model"
            payoff_col = f"player{p}_payoff"
            if model_col in all_data.columns and payoff_col in all_data.columns:
                p_data = all_data.select([
                    pl.col(model_col).alias("model"),
                    pl.col(payoff_col).alias("payoff"),
                    pl.col("game_type"),
                ])
                player_dfs.append(p_data)

        combined = pl.concat(player_dfs, how="diagonal") if player_dfs else pl.DataFrame()

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
