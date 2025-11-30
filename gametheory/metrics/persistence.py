"""Session persistence and cross-game analysis."""

import json
import uuid
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Any
import polars as pl

from ..core.types import SessionMetadata, RoundResult, PlayerConfig


class SessionManager:
    """Manages persistent session data across games."""

    def __init__(self, storage_path: str = "data/sessions"):
        """Initialize the session manager.

        Args:
            storage_path: Path to store session data.
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.current_session: Optional[SessionMetadata] = None

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
        return self.current_session

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
    ) -> Path:
        """Save game results to Parquet.

        Args:
            session_id: The session identifier.
            results: List of game result dictionaries.
            game_type: The game type ID.

        Returns:
            Path to the saved file.
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

        return filepath

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

        df = pl.concat([pl.read_parquet(f) for f in parquet_files])

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


class CrossGameAnalyzer:
    """Cross-game analysis and comparison utilities."""

    def __init__(self, session_manager: SessionManager):
        """Initialize the analyzer.

        Args:
            session_manager: The session manager to use for data access.
        """
        self.session_manager = session_manager

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

        # Find games where this model participated
        model_games = df.filter(
            (pl.col("player1_model") == model) | (pl.col("player2_model") == model)
        )

        if model_games.is_empty():
            return pl.DataFrame()

        return model_games.group_by("game_type").agg([
            pl.len().alias("total_rounds"),
            pl.col("player1_payoff").mean().alias("avg_payoff_as_p1"),
            pl.col("player2_payoff").mean().alias("avg_payoff_as_p2"),
        ])

    def model_leaderboard(self) -> pl.DataFrame:
        """Generate overall model leaderboard.

        Returns:
            DataFrame with model rankings.
        """
        df = self.session_manager.load_all_sessions()

        if df.is_empty():
            return pl.DataFrame()

        # Combine player1 and player2 stats
        p1_stats = df.select([
            pl.col("player1_model").alias("model"),
            pl.col("player1_payoff").alias("payoff"),
            pl.col("game_type"),
        ])

        p2_stats = df.select([
            pl.col("player2_model").alias("model"),
            pl.col("player2_payoff").alias("payoff"),
            pl.col("game_type"),
        ])

        all_plays = pl.concat([p1_stats, p2_stats])

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

        return df.group_by("game_type").agg([
            pl.col("session_id").n_unique().alias("total_sessions"),
            pl.len().alias("total_rounds"),
            pl.col("player1_payoff").mean().alias("avg_p1_payoff"),
            pl.col("player2_payoff").mean().alias("avg_p2_payoff"),
        ]).sort("total_sessions", descending=True)

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

        matchups = df.filter(
            ((pl.col("player1_model") == model1) & (pl.col("player2_model") == model2))
            | ((pl.col("player1_model") == model2) & (pl.col("player2_model") == model1))
        )

        if matchups.is_empty():
            return pl.DataFrame()

        return matchups.group_by("game_type").agg([
            pl.len().alias("total_rounds"),
            pl.col("player1_payoff").mean().alias("avg_p1_payoff"),
            pl.col("player2_payoff").mean().alias("avg_p2_payoff"),
        ])
