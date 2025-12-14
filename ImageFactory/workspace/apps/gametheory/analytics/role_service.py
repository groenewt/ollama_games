"""Role-centric analytics service for cross-session analysis."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional
import polars as pl

from ..core.role import RoleConfig
from ..core.utils import detect_num_players
from ..metrics.persistence import SessionManager
from ..storage.role_repository import RoleRepository


# Actions considered cooperative across different game types
COOPERATIVE_ACTIONS = [
    "cooperate", "Cooperate", "COOPERATE",
    "contribute", "Contribute", "CONTRIBUTE",
    "trust", "Trust", "TRUST",
    "share", "Share", "SHARE",
    "stag", "Stag", "STAG",
]


@dataclass
class RoleFilterParams:
    """Filter parameters for role analytics queries."""
    role_id: str                              # Required
    game_types: Optional[List[str]] = None    # Optional filter
    session_ids: Optional[List[str]] = None   # Optional filter


@dataclass
class RoleStatistics:
    """Aggregated statistics for a role."""
    role_id: str
    role_name: str
    model: str
    total_sessions: int
    total_rounds: int
    games_played: List[str]
    avg_payoff: float
    min_payoff: float
    max_payoff: float
    total_payoff: int
    win_rate: float
    cooperation_rate: Optional[float]  # None for allocation-only games
    first_session: Optional[datetime]
    last_session: Optional[datetime]


@dataclass
class RoleGameBreakdown:
    """Per-game type breakdown for a role."""
    role_id: str
    game_type: str
    sessions_count: int
    rounds_count: int
    avg_payoff: float
    win_rate: float
    cooperation_rate: Optional[float]


@dataclass
class RoleTimeline:
    """Temporal data for session timeline."""
    role_id: str
    session_id: str
    game_type: str
    timestamp: datetime
    num_rounds: int
    total_payoff: int
    avg_payoff: float
    win_rate: float
    opponent_models: List[str]


@dataclass
class DataSufficiency:
    """Tracks data sufficiency for various analytics."""
    has_basic_stats: bool          # >= 1 session
    has_game_breakdown: bool       # >= 1 session per game type
    has_learning_analysis: bool    # >= 10 rounds total
    has_personality_metrics: bool  # >= 20 rounds across >= 2 games
    has_cross_role_comparison: bool  # Other roles with shared games
    insufficient_reasons: List[str] = field(default_factory=list)


class RoleAnalyticsService:
    """Service for role-centric analytics with filtering support.

    This service provides role-focused queries spanning entire role history
    with optional filtering by game type and session.
    """

    def __init__(
        self,
        session_manager: SessionManager,
        role_repository: RoleRepository,
    ):
        """Initialize the role analytics service.

        Args:
            session_manager: The session manager for data access.
            role_repository: The role repository for role metadata.
        """
        self.session_manager = session_manager
        self.role_repository = role_repository
        self._stats_cache: Dict[str, RoleStatistics] = {}

    def _get_player_columns(self, df: pl.DataFrame) -> List[int]:
        """Detect which player numbers have columns in the DataFrame."""
        num_players = detect_num_players(tuple(df.columns))
        return list(range(1, num_players + 1)) if num_players > 0 else [1, 2]

    def _find_role_player_positions(
        self,
        df: pl.DataFrame,
        model: str,
    ) -> List[int]:
        """Find which player position(s) a model played in.

        Args:
            df: DataFrame with player model columns.
            model: Model name to find.

        Returns:
            List of player numbers (1-indexed) where this model played.
        """
        positions = []
        for p in self._get_player_columns(df):
            model_col = f"player{p}_model"
            if model_col in df.columns:
                # Check if this model appears in this position
                if df.filter(pl.col(model_col) == model).height > 0:
                    positions.append(p)
        return positions

    def _extract_role_metrics_vectorized(
        self,
        df: pl.DataFrame,
        model: str,
        player_nums: List[int],
    ) -> pl.DataFrame:
        """Extract role-specific payoffs and actions using vectorized operations.

        Args:
            df: DataFrame with player columns
            model: Model name to extract data for
            player_nums: List of player position numbers

        Returns:
            DataFrame with columns: payoff, action, is_win, session_id, game_type
        """
        if df.is_empty():
            return pl.DataFrame()

        # Build position-aware extraction: for each row, extract data from
        # whichever position this model plays in
        result_frames = []

        for p in player_nums:
            model_col = f"player{p}_model"
            payoff_col = f"player{p}_payoff"
            action_col = f"player{p}_action"

            if model_col not in df.columns or payoff_col not in df.columns:
                continue

            # Filter to rows where this player position has our model
            player_df = df.filter(pl.col(model_col) == model)

            if player_df.is_empty():
                continue

            # Build opponent max payoff expression
            opponent_cols = [
                f"player{op}_payoff" for op in player_nums
                if op != p and f"player{op}_payoff" in df.columns
            ]

            select_exprs = [
                pl.col(payoff_col).fill_null(0).alias("payoff"),
                pl.col("session_id") if "session_id" in df.columns else pl.lit("").alias("session_id"),
                pl.col("game_type") if "game_type" in df.columns else pl.lit("").alias("game_type"),
            ]

            # Add action if column exists
            if action_col in player_df.columns:
                select_exprs.append(pl.col(action_col).alias("action"))
            else:
                select_exprs.append(pl.lit(None).alias("action"))

            # Calculate win status
            if opponent_cols:
                max_opponent = pl.max_horizontal([pl.col(c).fill_null(0) for c in opponent_cols])
                select_exprs.append(
                    (pl.col(payoff_col).fill_null(0) > max_opponent).alias("is_win")
                )
            else:
                select_exprs.append(pl.lit(False).alias("is_win"))

            extracted = player_df.select(select_exprs)
            result_frames.append(extracted)

        if not result_frames:
            return pl.DataFrame()

        return pl.concat(result_frames)

    def get_role_sessions(
        self,
        filters: RoleFilterParams,
    ) -> pl.DataFrame:
        """Load all sessions for a role with optional filtering.

        Args:
            filters: Filter parameters including role_id and optional game/session filters.

        Returns:
            DataFrame with session data filtered to this role's participation.
        """
        role = self.role_repository.get_by_id(filters.role_id)
        if not role:
            return pl.DataFrame()

        # Load all sessions, optionally filtered by game type
        if filters.game_types and len(filters.game_types) == 1:
            # Single game type optimization
            df = self.session_manager.load_all_sessions(game_type=filters.game_types[0])
        else:
            df = self.session_manager.load_all_sessions()

        if df.is_empty():
            return df

        # Filter by multiple game types if specified
        if filters.game_types and len(filters.game_types) > 1:
            df = df.filter(pl.col("game_type").is_in(filters.game_types))

        # Filter to rounds where this role's model participated
        model = role.model
        model_filters = []
        for p in self._get_player_columns(df):
            model_col = f"player{p}_model"
            if model_col in df.columns:
                model_filters.append(pl.col(model_col) == model)

        if not model_filters:
            return pl.DataFrame()

        combined_filter = model_filters[0]
        for f in model_filters[1:]:
            combined_filter = combined_filter | f
        df = df.filter(combined_filter)

        # Apply session_id filter if specified
        if filters.session_ids:
            df = df.filter(pl.col("session_id").is_in(filters.session_ids))

        return df

    def get_role_statistics(
        self,
        filters: RoleFilterParams,
    ) -> RoleStatistics:
        """Get aggregated statistics for a role.

        Args:
            filters: Filter parameters.

        Returns:
            RoleStatistics with aggregated metrics.
        """
        role = self.role_repository.get_by_id(filters.role_id)
        if not role:
            return RoleStatistics(
                role_id=filters.role_id,
                role_name="Unknown",
                model="Unknown",
                total_sessions=0,
                total_rounds=0,
                games_played=[],
                avg_payoff=0.0,
                min_payoff=0.0,
                max_payoff=0.0,
                total_payoff=0,
                win_rate=0.0,
                cooperation_rate=None,
                first_session=None,
                last_session=None,
            )

        df = self.get_role_sessions(filters)

        if df.is_empty():
            return RoleStatistics(
                role_id=filters.role_id,
                role_name=role.name,
                model=role.model,
                total_sessions=0,
                total_rounds=0,
                games_played=[],
                avg_payoff=0.0,
                min_payoff=0.0,
                max_payoff=0.0,
                total_payoff=0,
                win_rate=0.0,
                cooperation_rate=None,
                first_session=None,
                last_session=None,
            )

        model = role.model
        player_nums = self._get_player_columns(df)

        # Use vectorized extraction for better performance with large datasets
        role_data = self._extract_role_metrics_vectorized(df, model, player_nums)

        if role_data.is_empty():
            # Fallback: no data found for this role
            return RoleStatistics(
                role_id=filters.role_id,
                role_name=role.name,
                model=role.model,
                total_sessions=0,
                total_rounds=0,
                games_played=[],
                avg_payoff=0.0,
                min_payoff=0.0,
                max_payoff=0.0,
                total_payoff=0,
                win_rate=0.0,
                cooperation_rate=None,
                first_session=None,
                last_session=None,
            )

        # Calculate aggregated statistics using vectorized operations
        stats = role_data.select([
            pl.col("payoff").sum().alias("total_payoff"),
            pl.col("payoff").mean().alias("avg_payoff"),
            pl.col("payoff").min().alias("min_payoff"),
            pl.col("payoff").max().alias("max_payoff"),
            pl.col("is_win").sum().alias("wins"),
            pl.len().alias("total"),
        ]).row(0, named=True)

        # Calculate cooperation rate for discrete games
        actions_df = role_data.filter(pl.col("action").is_not_null())
        if not actions_df.is_empty():
            coop_count = actions_df.filter(
                pl.col("action").is_in(COOPERATIVE_ACTIONS)
            ).height
            coop_rate = (coop_count / actions_df.height * 100)
        else:
            coop_rate = None

        # Parse timestamps
        first_ts = None
        last_ts = None
        if "timestamp" in df.columns:
            try:
                timestamps = df.select("timestamp").to_series().to_list()
                parsed_times = []
                for ts in timestamps:
                    if isinstance(ts, str):
                        parsed_times.append(datetime.fromisoformat(ts.replace('Z', '+00:00')))
                    elif isinstance(ts, datetime):
                        parsed_times.append(ts)
                if parsed_times:
                    first_ts = min(parsed_times)
                    last_ts = max(parsed_times)
            except Exception:
                pass

        total = stats["total"]
        wins = stats["wins"]

        return RoleStatistics(
            role_id=filters.role_id,
            role_name=role.name,
            model=role.model,
            total_sessions=df.select("session_id").n_unique(),
            total_rounds=len(df),
            games_played=df.select("game_type").unique().to_series().to_list(),
            avg_payoff=stats["avg_payoff"] or 0.0,
            min_payoff=stats["min_payoff"] or 0.0,
            max_payoff=stats["max_payoff"] or 0.0,
            total_payoff=int(stats["total_payoff"] or 0),
            win_rate=(wins / total * 100) if total > 0 else 0.0,
            cooperation_rate=coop_rate,
            first_session=first_ts,
            last_session=last_ts,
        )

    def get_game_breakdown(
        self,
        filters: RoleFilterParams,
    ) -> List[RoleGameBreakdown]:
        """Get per-game type performance breakdown.

        Args:
            filters: Filter parameters.

        Returns:
            List of RoleGameBreakdown for each game type.
        """
        role = self.role_repository.get_by_id(filters.role_id)
        if not role:
            return []

        df = self.get_role_sessions(filters)
        if df.is_empty():
            return []

        model = role.model
        game_types = df.select("game_type").unique().to_series().to_list()
        breakdowns = []

        player_nums = self._get_player_columns(df)

        for game_type in game_types:
            game_df = df.filter(pl.col("game_type") == game_type)

            # Use vectorized extraction for better performance
            role_data = self._extract_role_metrics_vectorized(game_df, model, player_nums)

            if role_data.is_empty():
                continue

            # Calculate aggregated statistics using vectorized operations
            stats = role_data.select([
                pl.col("payoff").mean().alias("avg_payoff"),
                pl.col("is_win").sum().alias("wins"),
                pl.len().alias("total"),
            ]).row(0, named=True)

            # Calculate cooperation rate
            actions_df = role_data.filter(pl.col("action").is_not_null())
            if not actions_df.is_empty():
                coop_count = actions_df.filter(
                    pl.col("action").is_in(COOPERATIVE_ACTIONS)
                ).height
                coop_rate = (coop_count / actions_df.height * 100)
            else:
                coop_rate = None

            total = stats["total"]
            wins = stats["wins"]

            breakdown = RoleGameBreakdown(
                role_id=filters.role_id,
                game_type=game_type,
                sessions_count=game_df.select("session_id").n_unique(),
                rounds_count=len(game_df),
                avg_payoff=stats["avg_payoff"] or 0.0,
                win_rate=(wins / total * 100) if total > 0 else 0.0,
                cooperation_rate=coop_rate,
            )
            breakdowns.append(breakdown)

        return sorted(breakdowns, key=lambda b: b.rounds_count, reverse=True)

    def get_session_timeline(
        self,
        filters: RoleFilterParams,
    ) -> List[RoleTimeline]:
        """Get chronological session timeline.

        Args:
            filters: Filter parameters.

        Returns:
            List of RoleTimeline entries ordered by time.
        """
        role = self.role_repository.get_by_id(filters.role_id)
        if not role:
            return []

        df = self.get_role_sessions(filters)
        if df.is_empty():
            return []

        model = role.model
        player_nums = self._get_player_columns(df)

        # Group by session
        session_ids = df.select("session_id").unique().to_series().to_list()
        timeline = []

        for session_id in session_ids:
            session_df = df.filter(pl.col("session_id") == session_id)

            # Get session info
            game_type = session_df.select("game_type").head(1).item()

            # Parse timestamp
            timestamp = datetime.now()
            if "timestamp" in session_df.columns:
                try:
                    ts_str = session_df.select("timestamp").head(1).item()
                    if isinstance(ts_str, str):
                        timestamp = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                    elif isinstance(ts_str, datetime):
                        timestamp = ts_str
                except Exception:
                    pass

            # Use vectorized extraction for metrics
            role_data = self._extract_role_metrics_vectorized(session_df, model, player_nums)

            if role_data.is_empty():
                continue

            stats = role_data.select([
                pl.col("payoff").sum().alias("total_payoff"),
                pl.col("payoff").mean().alias("avg_payoff"),
                pl.col("is_win").sum().alias("wins"),
                pl.len().alias("total"),
            ]).row(0, named=True)

            # Find opponent models (vectorized)
            opponent_models = set()
            for p in player_nums:
                model_col = f"player{p}_model"
                if model_col in session_df.columns:
                    models = session_df.select(model_col).unique().to_series().to_list()
                    for m in models:
                        if m and m != model:
                            opponent_models.add(m)

            total = stats["total"]
            wins = stats["wins"]

            timeline.append(RoleTimeline(
                role_id=filters.role_id,
                session_id=session_id,
                game_type=game_type,
                timestamp=timestamp,
                num_rounds=len(session_df),
                total_payoff=stats["total_payoff"] or 0,
                avg_payoff=stats["avg_payoff"] or 0.0,
                win_rate=(wins / total * 100) if total > 0 else 0.0,
                opponent_models=list(opponent_models),
            ))

        # Sort by timestamp
        return sorted(timeline, key=lambda t: t.timestamp)

    def get_available_games_for_role(
        self,
        role_id: str,
    ) -> List[str]:
        """Get list of game types where this role has data.

        Args:
            role_id: The role ID.

        Returns:
            List of game type IDs.
        """
        filters = RoleFilterParams(role_id=role_id)
        df = self.get_role_sessions(filters)
        if df.is_empty():
            return []
        return df.select("game_type").unique().to_series().to_list()

    def get_available_sessions_for_role(
        self,
        role_id: str,
        game_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get list of sessions where this role participated.

        Args:
            role_id: The role ID.
            game_type: Optional game type filter.

        Returns:
            List of session metadata dictionaries.
        """
        filters = RoleFilterParams(
            role_id=role_id,
            game_types=[game_type] if game_type else None,
        )
        df = self.get_role_sessions(filters)
        if df.is_empty():
            return []

        # Get unique sessions with their metadata
        sessions = []
        session_ids = df.select("session_id").unique().to_series().to_list()

        for session_id in session_ids:
            session_df = df.filter(pl.col("session_id") == session_id)
            game = session_df.select("game_type").head(1).item()

            timestamp = None
            if "timestamp" in session_df.columns:
                try:
                    ts_str = session_df.select("timestamp").head(1).item()
                    if isinstance(ts_str, str):
                        timestamp = ts_str
                except Exception:
                    pass

            sessions.append({
                "session_id": session_id,
                "game_type": game,
                "num_rounds": len(session_df),
                "timestamp": timestamp,
            })

        # Sort by timestamp descending (most recent first)
        return sorted(sessions, key=lambda s: s.get("timestamp") or "", reverse=True)

    def check_data_sufficiency(
        self,
        filters: RoleFilterParams,
    ) -> DataSufficiency:
        """Check what analytics are available given the data.

        Args:
            filters: Filter parameters.

        Returns:
            DataSufficiency indicating available analytics.
        """
        df = self.get_role_sessions(filters)
        total_rounds = len(df)
        unique_games = df.select("game_type").n_unique() if not df.is_empty() else 0
        unique_sessions = df.select("session_id").n_unique() if not df.is_empty() else 0

        reasons = []

        if total_rounds < 1:
            reasons.append("No session data found for this role")
        if total_rounds < 10:
            reasons.append("Need at least 10 rounds for learning analysis")
        if unique_games < 2:
            reasons.append("Need at least 2 game types for full personality metrics")
        if total_rounds < 20:
            reasons.append("Need at least 20 rounds for reliable personality metrics")

        # Check for other roles with shared games for comparison
        has_comparison = False
        if not df.is_empty():
            game_types = df.select("game_type").unique().to_series().to_list()
            all_roles = self.role_repository.list_all()
            for other_role in all_roles:
                if other_role.role_id != filters.role_id:
                    other_filters = RoleFilterParams(
                        role_id=other_role.role_id,
                        game_types=game_types,
                    )
                    other_df = self.get_role_sessions(other_filters)
                    if not other_df.is_empty():
                        has_comparison = True
                        break

        return DataSufficiency(
            has_basic_stats=total_rounds >= 1,
            has_game_breakdown=unique_sessions >= 1,
            has_learning_analysis=total_rounds >= 10,
            has_personality_metrics=total_rounds >= 20 and unique_games >= 2,
            has_cross_role_comparison=has_comparison,
            insufficient_reasons=reasons,
        )

    def get_round_level_data(
        self,
        filters: RoleFilterParams,
    ) -> pl.DataFrame:
        """Get round-level data for detailed analysis.

        Returns DataFrame with role's payoffs, actions, and round numbers.

        Args:
            filters: Filter parameters.

        Returns:
            DataFrame with columns: session_id, game_type, round_number,
            action, payoff, response_time (if available).
        """
        role = self.role_repository.get_by_id(filters.role_id)
        if not role:
            return pl.DataFrame()

        df = self.get_role_sessions(filters)
        if df.is_empty():
            return pl.DataFrame()

        model = role.model
        player_nums = self._get_player_columns(df)

        # Use vectorized extraction for each position the role plays
        result_frames = []

        for p in player_nums:
            model_col = f"player{p}_model"
            if model_col not in df.columns:
                continue

            # Filter to rows where this player position has our model
            player_df = df.filter(pl.col(model_col) == model)

            if player_df.is_empty():
                continue

            # Build select expressions
            select_exprs = [
                pl.col("session_id"),
                pl.col("game_type"),
            ]

            # Round number with fallback
            if "game_number" in player_df.columns:
                select_exprs.append(
                    pl.col("game_number").fill_null(0).alias("round_number")
                )
            elif "round" in player_df.columns:
                select_exprs.append(
                    pl.col("round").fill_null(0).alias("round_number")
                )
            else:
                select_exprs.append(pl.lit(0).alias("round_number"))

            # Action and payoff
            action_col = f"player{p}_action"
            payoff_col = f"player{p}_payoff"
            select_exprs.append(
                pl.col(action_col).alias("action") if action_col in player_df.columns
                else pl.lit(None).alias("action")
            )
            select_exprs.append(
                pl.col(payoff_col).fill_null(0).alias("payoff")
            )

            # Response time if available
            rt_col = f"player{p}_response_time"
            if rt_col in player_df.columns:
                select_exprs.append(pl.col(rt_col).alias("response_time"))

            extracted = player_df.select(select_exprs)
            result_frames.append(extracted)

        if not result_frames:
            return pl.DataFrame()

        return pl.concat(result_frames)

    def compare_roles(
        self,
        role_ids: List[str],
        game_type: Optional[str] = None,
    ) -> pl.DataFrame:
        """Compare multiple roles side-by-side.

        Args:
            role_ids: List of role IDs to compare.
            game_type: Optional game type filter.

        Returns:
            DataFrame with role comparison metrics.
        """
        rows = []
        for role_id in role_ids:
            filters = RoleFilterParams(
                role_id=role_id,
                game_types=[game_type] if game_type else None,
            )
            stats = self.get_role_statistics(filters)
            rows.append({
                "role_id": stats.role_id,
                "role_name": stats.role_name,
                "model": stats.model,
                "total_sessions": stats.total_sessions,
                "total_rounds": stats.total_rounds,
                "avg_payoff": stats.avg_payoff,
                "win_rate": stats.win_rate,
                "cooperation_rate": stats.cooperation_rate,
            })

        return pl.DataFrame(rows)
