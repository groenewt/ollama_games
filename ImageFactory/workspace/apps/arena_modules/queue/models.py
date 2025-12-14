"""Data models for the game queue system."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional
import uuid


@dataclass
class QueuedGame:
    """A game configuration captured for queue execution."""
    queue_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    # Game configuration
    game_type: str = ""
    game_name: str = ""
    runtime_mode: str = "one_off"
    num_games: int = 10
    payoff_display: str = "full"

    # Player configurations
    players: List[Dict[str, Any]] = field(default_factory=list)

    # Custom payoffs
    custom_payoffs: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    status: str = "pending"  # pending, running, completed, failed

    # Results (populated after execution)
    session_id: Optional[str] = None
    results_summary: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


@dataclass
class QueueExecutionResult:
    """Aggregated results from executing all queued games."""
    total_games: int
    completed_games: int
    failed_games: int
    total_rounds: int
    total_elapsed: float
    session_ids: List[str]
    results_by_game: Dict[str, Dict[str, Any]]
