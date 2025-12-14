"""Utility functions for game theory package."""

import ast
import logging
from functools import lru_cache
from typing import Any, List, Optional, Tuple


def parse_allocation(raw: Any) -> Optional[List[float]]:
    """Parse allocation from various formats.

    Handles: list, string repr of list, JSON, comma-separated.
    Returns None on failure (with debug logging).

    Args:
        raw: Raw allocation value (list, string, or None)

    Returns:
        List of floats if parsing succeeds, None otherwise
    """
    if raw is None:
        return None

    # Already a list - convert to floats
    if isinstance(raw, (list, tuple)):
        try:
            return [float(x) for x in raw]
        except (TypeError, ValueError) as e:
            logging.debug(f"Failed to convert list elements to float: {e}")
            return None

    # String parsing
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return None

        # Try ast.literal_eval for Python-style lists
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, (list, tuple)):
                return [float(x) for x in parsed]
        except (ValueError, SyntaxError) as e:
            logging.debug(f"AST parse failed for '{raw[:50]}...': {e}")

        # Try comma-separated format
        try:
            parts = [p.strip() for p in raw.split(",")]
            if all(p.replace(".", "").replace("-", "").isdigit() for p in parts if p):
                return [float(p) for p in parts if p]
        except ValueError as e:
            logging.debug(f"CSV parse failed for '{raw[:50]}...': {e}")

    return None


@lru_cache(maxsize=64)
def detect_num_players(columns: Tuple[str, ...]) -> int:
    """Detect player count from DataFrame column names.

    Args:
        columns: Tuple of column names (must be tuple for hashability).

    Returns:
        Number of players detected (minimum 2).

    Usage:
        num_players = detect_num_players(tuple(df.columns))
    """
    for p in range(1, 10):
        if f"player{p}_payoff" not in columns:
            return max(p - 1, 2)
    return 2


def normalize_allocation(
    allocation: List[float],
    target_budget: float = 100.0,
    tolerance: float = 0.01,
) -> Tuple[List[float], bool]:
    """Normalize allocation to target budget.

    Args:
        allocation: Raw allocation values
        target_budget: Target sum (default 100.0)
        tolerance: Acceptable deviation before normalization (default 1%)

    Returns:
        Tuple of (normalized_allocation, was_normalized)
    """
    current_sum = sum(allocation)

    if abs(current_sum - target_budget) <= tolerance * target_budget:
        return allocation, False

    if current_sum == 0:
        # Distribute evenly if all zeros
        n = len(allocation)
        return [target_budget / n] * n, True

    # Scale to target
    scale = target_budget / current_sum
    return [v * scale for v in allocation], True


def validate_allocation(
    allocation: List[float],
    expected_fields: Optional[int] = None,
    budget: float = 100.0,
    tolerance: float = 0.05,
    allow_negative: bool = False,
) -> Tuple[bool, List[str]]:
    """Validate allocation against constraints.

    Args:
        allocation: Allocation to validate
        expected_fields: Expected number of fields (None to skip check)
        budget: Expected total budget
        tolerance: Acceptable budget deviation (default 5%)
        allow_negative: Whether negative values are allowed

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    if expected_fields is not None and len(allocation) != expected_fields:
        errors.append(f"Expected {expected_fields} fields, got {len(allocation)}")

    if not allow_negative and any(v < 0 for v in allocation):
        errors.append("Allocation contains negative values")

    total = sum(allocation)
    if abs(total - budget) > tolerance * budget:
        errors.append(f"Budget mismatch: {total:.2f} vs expected {budget:.2f}")

    return len(errors) == 0, errors


def parse_allocation_from_result(
    result: dict,
    player_num: int,
) -> Optional[List[float]]:
    """Parse allocation from a result dictionary.

    Tries player{n}_allocation first, then player{n}_action.

    Args:
        result: Round result dictionary
        player_num: Player number (1-indexed)

    Returns:
        Parsed allocation or None
    """
    for key in [f"player{player_num}_allocation", f"player{player_num}_action"]:
        if key in result:
            parsed = parse_allocation(result[key])
            if parsed is not None:
                return parsed
    return None
