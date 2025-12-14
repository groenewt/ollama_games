"""Base analyzer for personality profiling."""

from typing import List, Optional, Any, Tuple
import polars as pl

from ...core.utils import parse_allocation


class BasePersonalityAnalyzer:
    """Base class providing data loading and parsing for personality analysis."""

    def __init__(self, session_manager: 'SessionManager'):
        """Initialize analyzer with session manager.

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
        return parse_allocation(value)

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
                    # Try allocation column first, then action
                    alloc = self._parse_allocation(row.get(alloc_col))
                    if alloc is None:
                        alloc = self._parse_allocation(row.get(action_col))

                    if alloc:
                        allocations.append((alloc, game_type))
                    break

        return allocations

    def clear_cache(self):
        """Clear the data cache."""
        self._cache.clear()
