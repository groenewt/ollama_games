"""Type definitions for game theory package."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from enum import Enum
import uuid


class RuntimeMode(str, Enum):
    """Game runtime modes."""
    ONE_OFF = "one_off"
    REPEATED = "repeated"
    SEQUENTIAL = "sequential"
    MULTI_PLAYER = "multi_player"


@dataclass
class GameDefinition:
    """Complete game definition."""
    id: str
    name: str
    description: str
    payoff_matrix: Dict[Tuple[str, ...], Tuple[int, ...]]
    actions: List[str]
    num_players: int = 2
    is_sequential: bool = False
    memory_depth: int = 0


@dataclass
class PlayerConfig:
    """Configuration for a player."""
    player_id: int
    model: str
    endpoint: str
    temperature: float = 0.7
    top_p: float = 0.9
    system_prompt: Optional[str] = None


@dataclass
class RoundResult:
    """Result of a single game round."""
    session_id: str
    game_type: str
    round_number: int
    timestamp: datetime
    actions: Tuple[str, ...]
    payoffs: Tuple[int, ...]
    player_configs: List[PlayerConfig]
    response_times: Dict[int, float] = field(default_factory=dict)
    request_success: Dict[int, bool] = field(default_factory=dict)
    history_length: int = 0


@dataclass
class SessionMetadata:
    """Metadata for a game session."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    created_at: datetime = field(default_factory=datetime.utcnow)
    game_type: str = ""
    runtime_mode: RuntimeMode = RuntimeMode.REPEATED
    num_rounds: int = 10
    players: List[PlayerConfig] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GameSession:
    """Complete game session with results."""
    metadata: SessionMetadata
    results: List[RoundResult] = field(default_factory=list)
    status: str = "pending"  # pending, running, completed, error
