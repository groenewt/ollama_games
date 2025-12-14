"""Session persistence and cross-game analysis."""

import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Any, TYPE_CHECKING
import polars as pl

from ..core.types import SessionMetadata, RoundResult, PlayerConfig, LLMResponse
from ..core.utils import detect_num_players

if TYPE_CHECKING:
    from ..storage.interaction_store import InteractionStore


class SessionManager:
    """Manages persistent session data across games.

    Supports optional InteractionStore integration for full prompt/response
    persistence alongside game results.
    """

    def __init__(
        self,
        storage_path: str = "data/sessions",
        interaction_store: Optional["InteractionStore"] = None,
    ):
        """Initialize the session manager.

        Args:
            storage_path: Path to store session data.
            interaction_store: Optional InteractionStore for prompt/response persistence.
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.current_session: Optional[SessionMetadata] = None
        self.interaction_store = interaction_store
        # Buffer for round interactions (cleared on session save)
        self._interaction_buffer: List[List[LLMResponse]] = []

    def _get_session_dir(self, date_: Optional[date] = None) -> Path:
        """Get date-partitioned session directory."""
        date_ = date_ or date.today()
        path = self.storage_path / date_.isoformat()
        path.mkdir(exist_ok=True)
        return path

    def create_session(
        self,
        game_type: str,
        players: List[PlayerConfig],
        num_rounds: int,
    ) -> SessionMetadata:
        """Create a new session.

        Args:
            game_type: The game type ID.
            players: List of player configurations.
            num_rounds: Number of rounds to play.

        Returns:
            The created SessionMetadata.
        """
        self.current_session = SessionMetadata(
            session_id=str(uuid.uuid4())[:8],
            created_at=datetime.utcnow(),
            game_type=game_type,
            num_rounds=num_rounds,
            players=players,
        )
        # Clear interaction buffer for new session
        self._interaction_buffer = []
        return self.current_session

    def buffer_round_interactions(self, interactions: List[LLMResponse]) -> None:
        """Buffer interactions from a round for later persistence.

        Call this after each round to collect LLMResponse objects.
        Interactions are saved when save_results() is called.

        Args:
            interactions: List of LLMResponse objects from the round.
        """
        self._interaction_buffer.append(interactions)

    def set_interaction_store(self, store: "InteractionStore") -> None:
        """Set or update the interaction store.

        Args:
            store: InteractionStore instance for persistence.
        """
        self.interaction_store = store

    def save_session_metadata(self, metadata: SessionMetadata) -> Path:
        """Save session metadata as JSON.

        Args:
            metadata: The session metadata to save.

        Returns:
            Path to the saved file.
        """
        session_dir = self._get_session_dir()
        filepath = session_dir / f"session_{metadata.session_id}.json"

        data = {
            "session_id": metadata.session_id,
            "created_at": metadata.created_at.isoformat(),
            "game_type": metadata.game_type,
            "runtime_mode": metadata.runtime_mode.value if hasattr(metadata.runtime_mode, 'value') else str(metadata.runtime_mode),
            "num_rounds": metadata.num_rounds,
            "players": [
                {
                    "player_id": p.player_id,
                    "model": p.model,
                    "endpoint": p.endpoint,
                }
                for p in metadata.players
            ],
            "config": metadata.config,
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        return filepath

    def save_results(
        self,
        session_id: str,
        results: List[Dict[str, Any]],
        game_type: str,
        persist_interactions: bool = True,
    ) -> Path:
        """Save game results to Parquet.

        Also persists buffered interactions if InteractionStore is configured.

        Args:
            session_id: The session identifier.
            results: List of game result dictionaries.
            game_type: The game type ID.
            persist_interactions: Whether to save buffered interactions.

        Returns:
            Path to the saved results file.
        """
        session_dir = self._get_session_dir()
        filepath = session_dir / f"session_{session_id}.parquet"

        # Add session info to results
        for r in results:
            r["session_id"] = session_id
            r["game_type"] = game_type
            r["timestamp"] = datetime.utcnow().isoformat()

        df = pl.DataFrame(results)
        df.write_parquet(filepath, compression="zstd")

        # Persist interactions if store is configured and we have buffered data
        if persist_interactions and self.interaction_store and self._interaction_buffer:
            try:
                self.interaction_store.save_session_interactions(
                    session_id=session_id,
                    all_interactions=self._interaction_buffer,
                )
            except Exception as e:
                # Log but don't fail - interaction persistence is optional
                import warnings
                warnings.warn(f"Failed to persist interactions: {e}")

            # Clear buffer after save
            self._interaction_buffer = []

        return filepath

    def save_results_with_interactions(
        self,
        session_id: str,
        results: List[Dict[str, Any]],
        game_type: str,
        interactions: List[List[LLMResponse]],
    ) -> Path:
        """Save game results and interactions in one call.

        Convenience method for saving all session data at once.

        Args:
            session_id: The session identifier.
            results: List of game result dictionaries.
            game_type: The game type ID.
            interactions: List of round interactions (each is list of LLMResponse).

        Returns:
            Path to the saved results file.
        """
        # Set buffer and save
        self._interaction_buffer = interactions
        return self.save_results(session_id, results, game_type, persist_interactions=True)

    def load_session(
        self,
        session_id: str,
        date_: Optional[date] = None,
    ) -> pl.DataFrame:
        """Load a specific session's results.

        Args:
            session_id: The session identifier.
            date_: Optional date to search in.

        Returns:
            DataFrame with session results.

        Raises:
            FileNotFoundError: If session not found.
        """
        if date_:
            filepath = self._get_session_dir(date_) / f"session_{session_id}.parquet"
            if filepath.exists():
                return pl.read_parquet(filepath)

        # Search across all dates
        for date_dir in sorted(self.storage_path.iterdir(), reverse=True):
            if not date_dir.is_dir():
                continue
            filepath = date_dir / f"session_{session_id}.parquet"
            if filepath.exists():
                return pl.read_parquet(filepath)

        raise FileNotFoundError(f"Session {session_id} not found")

    def load_all_sessions(
        self,
        game_type: Optional[str] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pl.DataFrame:
        """Load multiple sessions with optional filtering.

        Uses parallel I/O for faster loading with many session files.

        Args:
            game_type: Filter by game type.
            start_date: Filter by start date.
            end_date: Filter by end date.

        Returns:
            Combined DataFrame of all matching sessions.
        """
        parquet_files = []

        for date_dir in self.storage_path.iterdir():
            if not date_dir.is_dir():
                continue

            try:
                dir_date = date.fromisoformat(date_dir.name)
            except ValueError:
                continue

            if start_date and dir_date < start_date:
                continue
            if end_date and dir_date > end_date:
                continue

            parquet_files.extend(date_dir.glob("session_*.parquet"))

        if not parquet_files:
            return pl.DataFrame()

        # Use parallel I/O for faster loading (3-4x speedup with many files)
        if len(parquet_files) > 3:
            with ThreadPoolExecutor(max_workers=4) as executor:
                dfs = list(executor.map(pl.read_parquet, parquet_files))
        else:
            dfs = [pl.read_parquet(f) for f in parquet_files]

        df = pl.concat(dfs, how="diagonal_relaxed")

        if game_type:
            df = df.filter(pl.col("game_type") == game_type)

        return df

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all available sessions.

        Returns:
            List of session metadata dictionaries.
        """
        sessions = []

        for date_dir in sorted(self.storage_path.iterdir(), reverse=True):
            if not date_dir.is_dir():
                continue

            for json_file in date_dir.glob("session_*.json"):
                with open(json_file) as f:
                    sessions.append(json.load(f))

        return sessions

    def load_session_metadata(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load session metadata (config) by session ID.

        Args:
            session_id: The session identifier.

        Returns:
            Session metadata dictionary or None if not found.
        """
        # Search across all dates
        for date_dir in sorted(self.storage_path.iterdir(), reverse=True):
            if not date_dir.is_dir():
                continue
            filepath = date_dir / f"session_{session_id}.json"
            if filepath.exists():
                with open(filepath) as f:
                    return json.load(f)

        return None


class CrossGameAnalyzer:
    """Cross-game analysis and comparison utilities."""

    def __init__(self, session_manager: SessionManager):
        """Initialize the analyzer.

        Args:
            session_manager: The session manager to use for data access.
        """
        self.session_manager = session_manager

    @staticmethod
    def _get_player_nums(df: pl.DataFrame) -> List[int]:
        """Detect player numbers from DataFrame columns (cached)."""
        num_players = detect_num_players(tuple(df.columns))
        return list(range(1, num_players + 1)) if num_players > 0 else [1, 2]

    def model_performance_across_games(self, model: str) -> pl.DataFrame:
        """Analyze a model's performance across all game types.

        Args:
            model: The model identifier.

        Returns:
            DataFrame with performance metrics per game type.
        """
        df = self.session_manager.load_all_sessions()

        if df.is_empty():
            return pl.DataFrame()

        player_nums = self._get_player_nums(df)

        # Build filter for games where this model participated (any player position)
        model_filters = []
        for p in player_nums:
            col = f"player{p}_model"
            if col in df.columns:
                model_filters.append(pl.col(col) == model)

        if not model_filters:
            return pl.DataFrame()

        combined_filter = model_filters[0]
        for f in model_filters[1:]:
            combined_filter = combined_filter | f

        model_games = df.filter(combined_filter)

        if model_games.is_empty():
            return pl.DataFrame()

        # Build dynamic aggregations for each player position
        agg_exprs = [pl.len().alias("total_rounds")]
        for p in player_nums:
            payoff_col = f"player{p}_payoff"
            if payoff_col in model_games.columns:
                agg_exprs.append(pl.col(payoff_col).mean().alias(f"avg_payoff_as_p{p}"))

        return model_games.group_by("game_type").agg(agg_exprs)

    def model_leaderboard(self) -> pl.DataFrame:
        """Generate overall model leaderboard.

        Returns:
            DataFrame with model rankings.
        """
        df = self.session_manager.load_all_sessions()

        if df.is_empty():
            return pl.DataFrame()

        player_nums = self._get_player_nums(df)

        # Combine all player stats dynamically
        player_dfs = []
        for p in player_nums:
            model_col = f"player{p}_model"
            payoff_col = f"player{p}_payoff"
            if model_col in df.columns and payoff_col in df.columns:
                p_stats = df.select([
                    pl.col(model_col).alias("model"),
                    pl.col(payoff_col).alias("payoff"),
                    pl.col("game_type"),
                ])
                player_dfs.append(p_stats)

        if not player_dfs:
            return pl.DataFrame()

        all_plays = pl.concat(player_dfs, how="diagonal")

        return all_plays.group_by("model").agg([
            pl.len().alias("total_plays"),
            pl.col("game_type").n_unique().alias("games_played"),
            pl.col("payoff").mean().alias("avg_payoff"),
            pl.col("payoff").sum().alias("total_payoff"),
        ]).sort("avg_payoff", descending=True)

    def game_type_summary(self) -> pl.DataFrame:
        """Summary statistics per game type.

        Returns:
            DataFrame with per-game statistics.
        """
        df = self.session_manager.load_all_sessions()

        if df.is_empty():
            return pl.DataFrame()

        player_nums = self._get_player_nums(df)

        # Build dynamic aggregations
        agg_exprs = [
            pl.col("session_id").n_unique().alias("total_sessions"),
            pl.len().alias("total_rounds"),
        ]
        for p in player_nums:
            payoff_col = f"player{p}_payoff"
            if payoff_col in df.columns:
                agg_exprs.append(pl.col(payoff_col).mean().alias(f"avg_p{p}_payoff"))

        return df.group_by("game_type").agg(agg_exprs).sort("total_sessions", descending=True)

    def head_to_head(self, model1: str, model2: str) -> pl.DataFrame:
        """Compare two models' performance against each other.

        Args:
            model1: First model identifier.
            model2: Second model identifier.

        Returns:
            DataFrame with head-to-head statistics.
        """
        df = self.session_manager.load_all_sessions()

        if df.is_empty():
            return pl.DataFrame()

        player_nums = self._get_player_nums(df)

        # Build filters for any configuration where both models played
        # This handles N-player games where model1 and model2 both participated
        matchup_filters = []
        for p1 in player_nums:
            for p2 in player_nums:
                if p1 >= p2:
                    continue
                col1 = f"player{p1}_model"
                col2 = f"player{p2}_model"
                if col1 in df.columns and col2 in df.columns:
                    matchup_filters.append(
                        ((pl.col(col1) == model1) & (pl.col(col2) == model2))
                        | ((pl.col(col1) == model2) & (pl.col(col2) == model1))
                    )

        if not matchup_filters:
            return pl.DataFrame()

        combined_filter = matchup_filters[0]
        for f in matchup_filters[1:]:
            combined_filter = combined_filter | f

        matchups = df.filter(combined_filter)

        if matchups.is_empty():
            return pl.DataFrame()

        # Build dynamic aggregations
        agg_exprs = [pl.len().alias("total_rounds")]
        for p in player_nums:
            payoff_col = f"player{p}_payoff"
            if payoff_col in matchups.columns:
                agg_exprs.append(pl.col(payoff_col).mean().alias(f"avg_p{p}_payoff"))

        return matchups.group_by("game_type").agg(agg_exprs)
