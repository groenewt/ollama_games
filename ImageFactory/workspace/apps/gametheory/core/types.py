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
    top_k: int = 40  # Limit token choices (lower = more focused)
    repeat_penalty: float = 1.1  # Penalize repetition
    system_prompt: Optional[str] = None
    strategy_hints: Optional[str] = None  # Custom hints to replace default game hints


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
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:16])
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


@dataclass
class LLMResponse:
    """Complete LLM response with metadata for comprehensive analysis.

    Captures full context of each LLM interaction including:
    - The prompt sent and raw response received
    - Parsing status and action extracted
    - Token usage for cost tracking
    - Optional reasoning trace and confidence metrics
    """
    player_id: int
    model: str
    endpoint: str
    prompt: str
    raw_response: str
    parsed_action: Any
    was_parsed: bool
    was_normalized: bool  # For allocation games - was budget normalization applied?
    response_time_ms: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    # Token metrics (populated from Ollama response if available)
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    # Decision reasoning capture (experimental)
    reasoning_trace: Optional[str] = None  # Extracted chain-of-thought if present
    alternatives_considered: Optional[List[str]] = None  # Other actions mentioned
    confidence_score: Optional[float] = None  # Estimated decision confidence
    # Inference parameters used
    inference_params: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> Optional[int]:
        """Calculate total tokens if both counts available."""
        if self.prompt_tokens is not None and self.completion_tokens is not None:
            return self.prompt_tokens + self.completion_tokens
        return None


@dataclass
class TokenMetrics:
    """Token usage and cost tracking for a single LLM request.

    Aggregated for session-level and cumulative cost analysis.
    """
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    model: str
    session_id: str
    round_number: int
    player_id: int
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @classmethod
    def from_llm_response(
        cls,
        response: "LLMResponse",
        session_id: str,
        round_number: int,
        cost_per_prompt_token: float = 0.0,
        cost_per_completion_token: float = 0.0,
    ) -> Optional["TokenMetrics"]:
        """Create TokenMetrics from an LLMResponse if token data available."""
        if response.prompt_tokens is None or response.completion_tokens is None:
            return None

        total = response.prompt_tokens + response.completion_tokens
        cost = (
            response.prompt_tokens * cost_per_prompt_token / 1000 +
            response.completion_tokens * cost_per_completion_token / 1000
        )

        return cls(
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=total,
            estimated_cost_usd=cost,
            model=response.model,
            session_id=session_id,
            round_number=round_number,
            player_id=response.player_id,
            timestamp=response.timestamp,
        )
