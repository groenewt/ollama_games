"""Interaction storage for persisting LLM prompts and responses.

Stores full prompt/response data in Parquet format with LZ4 compression
for efficient storage and fast analytical queries.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import json

import polars as pl

from ..core.types import LLMResponse


@dataclass
class InteractionRecord:
    """A single LLM interaction record for storage."""
    session_id: str
    round_number: int
    player_id: int
    model: str
    endpoint: str
    prompt: str
    raw_response: str
    parsed_action: str  # Serialized as string
    was_parsed: bool
    was_normalized: bool
    response_time_ms: float
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    reasoning_trace: Optional[str]
    timestamp: datetime
    inference_params: Dict[str, Any]


class InteractionStore:
    """Persistent storage for LLM prompts and responses.

    Stores interaction data in Parquet format with:
    - LZ4 compression for text fields (prompts/responses)
    - Date-partitioned storage for efficient queries
    - Session-grouped files for atomic writes

    Storage layout:
        {storage_path}/
            interactions/
                {YYYY-MM-DD}/
                    session_{session_id}_interactions.parquet
    """

    def __init__(self, storage_path: str = "data"):
        """Initialize the interaction store.

        Args:
            storage_path: Base path for data storage.
        """
        self.storage_path = Path(storage_path)
        self.interactions_dir = self.storage_path / "interactions"

    def _get_session_path(self, session_id: str, date: Optional[datetime] = None) -> Path:
        """Get the path for a session's interaction file.

        Args:
            session_id: The session identifier.
            date: Optional date for partitioning. Defaults to today.

        Returns:
            Path to the session's interaction parquet file.
        """
        date = date or datetime.utcnow()
        date_str = date.strftime("%Y-%m-%d")
        date_dir = self.interactions_dir / date_str
        return date_dir / f"session_{session_id}_interactions.parquet"

    def save_round_interactions(
        self,
        session_id: str,
        round_number: int,
        interactions: List[LLMResponse],
        append: bool = True,
    ) -> Path:
        """Save interaction data from a single round.

        Args:
            session_id: The session identifier.
            round_number: Current round number.
            interactions: List of LLMResponse objects from this round.
            append: If True, append to existing file. If False, overwrite.

        Returns:
            Path to the saved file.
        """
        # Convert to records
        records = []
        for resp in interactions:
            # Serialize action for storage
            if isinstance(resp.parsed_action, (list, tuple)):
                action_str = json.dumps(list(resp.parsed_action))
            else:
                action_str = str(resp.parsed_action)

            records.append({
                "session_id": session_id,
                "round_number": round_number,
                "player_id": resp.player_id,
                "model": resp.model,
                "endpoint": resp.endpoint,
                "prompt": resp.prompt,
                "raw_response": resp.raw_response,
                "parsed_action": action_str,
                "was_parsed": resp.was_parsed,
                "was_normalized": resp.was_normalized,
                "response_time_ms": resp.response_time_ms,
                "prompt_tokens": resp.prompt_tokens,
                "completion_tokens": resp.completion_tokens,
                "reasoning_trace": resp.reasoning_trace,
                "timestamp": resp.timestamp,
                "inference_params": json.dumps(resp.inference_params),
            })

        # Create DataFrame
        new_df = pl.DataFrame(records)

        # Get file path
        file_path = self._get_session_path(session_id)

        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Append or write
        if append and file_path.exists():
            existing_df = pl.read_parquet(file_path)
            combined_df = pl.concat([existing_df, new_df])
            combined_df.write_parquet(
                file_path,
                compression="lz4",
                use_pyarrow=True,
            )
        else:
            new_df.write_parquet(
                file_path,
                compression="lz4",
                use_pyarrow=True,
            )

        return file_path

    def save_session_interactions(
        self,
        session_id: str,
        all_interactions: List[List[LLMResponse]],
    ) -> Path:
        """Save all interactions from a complete session.

        More efficient than round-by-round saves for batch operations.

        Args:
            session_id: The session identifier.
            all_interactions: List of round interactions, where each element
                             is a list of LLMResponse objects for that round.

        Returns:
            Path to the saved file.
        """
        records = []
        for round_num, round_interactions in enumerate(all_interactions, start=1):
            for resp in round_interactions:
                # Serialize action
                if isinstance(resp.parsed_action, (list, tuple)):
                    action_str = json.dumps(list(resp.parsed_action))
                else:
                    action_str = str(resp.parsed_action)

                records.append({
                    "session_id": session_id,
                    "round_number": round_num,
                    "player_id": resp.player_id,
                    "model": resp.model,
                    "endpoint": resp.endpoint,
                    "prompt": resp.prompt,
                    "raw_response": resp.raw_response,
                    "parsed_action": action_str,
                    "was_parsed": resp.was_parsed,
                    "was_normalized": resp.was_normalized,
                    "response_time_ms": resp.response_time_ms,
                    "prompt_tokens": resp.prompt_tokens,
                    "completion_tokens": resp.completion_tokens,
                    "reasoning_trace": resp.reasoning_trace,
                    "timestamp": resp.timestamp,
                    "inference_params": json.dumps(resp.inference_params),
                })

        df = pl.DataFrame(records)

        # Get file path
        file_path = self._get_session_path(session_id)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        df.write_parquet(
            file_path,
            compression="lz4",
            use_pyarrow=True,
        )

        return file_path

    def load_session_interactions(
        self,
        session_id: str,
        round_number: Optional[int] = None,
    ) -> Optional[pl.DataFrame]:
        """Load interactions for a session.

        Args:
            session_id: The session identifier.
            round_number: Optional filter for specific round.

        Returns:
            DataFrame with interaction data, or None if not found.
        """
        # Search in all date directories
        if not self.interactions_dir.exists():
            return None

        for date_dir in sorted(self.interactions_dir.iterdir(), reverse=True):
            if not date_dir.is_dir():
                continue

            file_path = date_dir / f"session_{session_id}_interactions.parquet"
            if file_path.exists():
                df = pl.read_parquet(file_path)

                if round_number is not None:
                    df = df.filter(pl.col("round_number") == round_number)

                return df

        return None

    def load_interactions_by_date(
        self,
        date: datetime,
        model: Optional[str] = None,
    ) -> pl.DataFrame:
        """Load all interactions from a specific date.

        Args:
            date: The date to load interactions for.
            model: Optional filter by model name.

        Returns:
            DataFrame with all interactions from the date.
        """
        date_str = date.strftime("%Y-%m-%d")
        date_dir = self.interactions_dir / date_str

        if not date_dir.exists():
            return pl.DataFrame()

        # Load all parquet files from the date directory
        dfs = []
        for file_path in date_dir.glob("*.parquet"):
            df = pl.read_parquet(file_path)
            dfs.append(df)

        if not dfs:
            return pl.DataFrame()

        combined = pl.concat(dfs)

        if model:
            combined = combined.filter(pl.col("model") == model)

        return combined

    def get_token_summary(
        self,
        session_id: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get summary of token usage from stored interactions.

        Args:
            session_id: Optional filter by session.
            model: Optional filter by model.

        Returns:
            Dict with token usage statistics.
        """
        if session_id:
            df = self.load_session_interactions(session_id)
            if df is None:
                return {"error": "Session not found"}
        else:
            # Load all recent interactions (last 7 days)
            dfs = []
            for date_dir in sorted(self.interactions_dir.iterdir(), reverse=True)[:7]:
                if date_dir.is_dir():
                    for file_path in date_dir.glob("*.parquet"):
                        dfs.append(pl.read_parquet(file_path))

            if not dfs:
                return {"total_interactions": 0}

            df = pl.concat(dfs)

        if model:
            df = df.filter(pl.col("model") == model)

        # Filter to rows with token data
        df_with_tokens = df.filter(
            pl.col("prompt_tokens").is_not_null() &
            pl.col("completion_tokens").is_not_null()
        )

        if len(df_with_tokens) == 0:
            return {
                "total_interactions": len(df),
                "interactions_with_tokens": 0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
            }

        return {
            "total_interactions": len(df),
            "interactions_with_tokens": len(df_with_tokens),
            "total_prompt_tokens": int(df_with_tokens["prompt_tokens"].sum()),
            "total_completion_tokens": int(df_with_tokens["completion_tokens"].sum()),
            "total_tokens": int(
                df_with_tokens["prompt_tokens"].sum() +
                df_with_tokens["completion_tokens"].sum()
            ),
            "avg_prompt_tokens": float(df_with_tokens["prompt_tokens"].mean()),
            "avg_completion_tokens": float(df_with_tokens["completion_tokens"].mean()),
            "unique_models": df_with_tokens["model"].unique().to_list(),
            "unique_sessions": df_with_tokens["session_id"].unique().len(),
        }

    def list_sessions(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """List all sessions with interaction data.

        Args:
            start_date: Optional start date filter.
            end_date: Optional end date filter.

        Returns:
            List of session summaries.
        """
        if not self.interactions_dir.exists():
            return []

        sessions = []
        for date_dir in sorted(self.interactions_dir.iterdir()):
            if not date_dir.is_dir():
                continue

            # Check date filter
            try:
                dir_date = datetime.strptime(date_dir.name, "%Y-%m-%d")
            except ValueError:
                continue

            if start_date and dir_date < start_date:
                continue
            if end_date and dir_date > end_date:
                continue

            for file_path in date_dir.glob("*.parquet"):
                # Extract session_id from filename
                session_id = file_path.stem.replace("session_", "").replace("_interactions", "")

                # Load metadata
                df = pl.read_parquet(file_path)
                sessions.append({
                    "session_id": session_id,
                    "date": date_dir.name,
                    "num_rounds": df["round_number"].max(),
                    "num_interactions": len(df),
                    "models": df["model"].unique().to_list(),
                    "file_path": str(file_path),
                })

        return sessions
