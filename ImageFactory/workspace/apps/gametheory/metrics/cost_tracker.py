"""Token cost tracking and estimation for LLM API usage.

Provides cost estimation based on model pricing configurations.
Supports both local (free) models via Ollama and commercial API pricing.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any


# Default cost configurations per 1K tokens
# Local models (Ollama) are free, commercial APIs have associated costs
DEFAULT_MODEL_COSTS: Dict[str, Dict[str, float]] = {
    # Local Ollama models (free)
    "llama3.2": {"prompt": 0.0, "completion": 0.0},
    "llama3": {"prompt": 0.0, "completion": 0.0},
    "llama2": {"prompt": 0.0, "completion": 0.0},
    "qwen3": {"prompt": 0.0, "completion": 0.0},
    "qwen2.5": {"prompt": 0.0, "completion": 0.0},
    "tinyllama": {"prompt": 0.0, "completion": 0.0},
    "granite": {"prompt": 0.0, "completion": 0.0},
    "mistral": {"prompt": 0.0, "completion": 0.0},
    "phi3": {"prompt": 0.0, "completion": 0.0},
    "gemma": {"prompt": 0.0, "completion": 0.0},
    # OpenAI models (per 1K tokens, USD)
    "gpt-4": {"prompt": 0.03, "completion": 0.06},
    "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},
    "gpt-4o": {"prompt": 0.005, "completion": 0.015},
    "gpt-3.5-turbo": {"prompt": 0.0005, "completion": 0.0015},
    # Anthropic models (per 1K tokens, USD)
    "claude-3-opus": {"prompt": 0.015, "completion": 0.075},
    "claude-3-sonnet": {"prompt": 0.003, "completion": 0.015},
    "claude-3-haiku": {"prompt": 0.00025, "completion": 0.00125},
    "claude-3.5-sonnet": {"prompt": 0.003, "completion": 0.015},
}


@dataclass
class CostRecord:
    """Record of cost from a single API call."""
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_cost_usd: float
    completion_cost_usd: float
    total_cost_usd: float
    session_id: Optional[str] = None
    round_number: Optional[int] = None
    player_id: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CostSummary:
    """Summary of costs across multiple records."""
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    total_cost_usd: float
    by_model: Dict[str, Dict[str, Any]]
    num_requests: int
    avg_cost_per_request: float


class CostTracker:
    """Track and estimate costs for LLM API usage.

    Maintains running totals and per-model breakdowns.
    Supports custom cost configurations for different providers.
    """

    def __init__(
        self,
        model_costs: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        """Initialize the cost tracker.

        Args:
            model_costs: Custom model cost configurations (per 1K tokens).
                        If None, uses DEFAULT_MODEL_COSTS.
        """
        self.model_costs = model_costs or DEFAULT_MODEL_COSTS.copy()
        self.records: List[CostRecord] = []
        # Running totals
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_cost = 0.0

    def get_model_cost(
        self,
        model: str,
    ) -> Dict[str, float]:
        """Get cost configuration for a model.

        Args:
            model: Model name (exact match or prefix match).

        Returns:
            Dict with 'prompt' and 'completion' costs per 1K tokens.
        """
        # Try exact match first
        if model in self.model_costs:
            return self.model_costs[model]

        # Try prefix match for versioned models
        for key in self.model_costs:
            if model.startswith(key) or key.startswith(model):
                return self.model_costs[key]

        # Default to free (assume local model)
        return {"prompt": 0.0, "completion": 0.0}

    def set_model_cost(
        self,
        model: str,
        prompt_cost: float,
        completion_cost: float,
    ) -> None:
        """Set or update cost configuration for a model.

        Args:
            model: Model name.
            prompt_cost: Cost per 1K prompt tokens (USD).
            completion_cost: Cost per 1K completion tokens (USD).
        """
        self.model_costs[model] = {
            "prompt": prompt_cost,
            "completion": completion_cost,
        }

    def estimate_cost(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> Dict[str, float]:
        """Estimate cost for a request without recording.

        Args:
            model: Model name.
            prompt_tokens: Number of prompt tokens.
            completion_tokens: Number of completion tokens.

        Returns:
            Dict with prompt_cost, completion_cost, and total_cost (USD).
        """
        costs = self.get_model_cost(model)
        prompt_cost = prompt_tokens * costs["prompt"] / 1000
        completion_cost = completion_tokens * costs["completion"] / 1000

        return {
            "prompt_cost_usd": prompt_cost,
            "completion_cost_usd": completion_cost,
            "total_cost_usd": prompt_cost + completion_cost,
        }

    def record_usage(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        session_id: Optional[str] = None,
        round_number: Optional[int] = None,
        player_id: Optional[int] = None,
    ) -> CostRecord:
        """Record token usage and calculate cost.

        Args:
            model: Model name.
            prompt_tokens: Number of prompt tokens.
            completion_tokens: Number of completion tokens.
            session_id: Optional session identifier.
            round_number: Optional round number.
            player_id: Optional player identifier.

        Returns:
            CostRecord with calculated costs.
        """
        costs = self.estimate_cost(model, prompt_tokens, completion_tokens)

        record = CostRecord(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            prompt_cost_usd=costs["prompt_cost_usd"],
            completion_cost_usd=costs["completion_cost_usd"],
            total_cost_usd=costs["total_cost_usd"],
            session_id=session_id,
            round_number=round_number,
            player_id=player_id,
        )

        self.records.append(record)

        # Update running totals
        self._total_prompt_tokens += prompt_tokens
        self._total_completion_tokens += completion_tokens
        self._total_cost += costs["total_cost_usd"]

        return record

    def get_session_cost(
        self,
        session_id: str,
    ) -> CostSummary:
        """Get cost summary for a specific session.

        Args:
            session_id: Session identifier.

        Returns:
            CostSummary for the session.
        """
        session_records = [r for r in self.records if r.session_id == session_id]
        return self._summarize_records(session_records)

    def get_model_cost_summary(
        self,
        model: str,
    ) -> CostSummary:
        """Get cost summary for a specific model.

        Args:
            model: Model name.

        Returns:
            CostSummary for the model.
        """
        model_records = [r for r in self.records if r.model == model]
        return self._summarize_records(model_records)

    def get_cumulative_cost(self) -> CostSummary:
        """Get cumulative cost across all tracked records.

        Returns:
            CostSummary with all costs.
        """
        return self._summarize_records(self.records)

    def _summarize_records(self, records: List[CostRecord]) -> CostSummary:
        """Summarize a list of cost records.

        Args:
            records: List of CostRecord objects.

        Returns:
            CostSummary aggregation.
        """
        if not records:
            return CostSummary(
                total_prompt_tokens=0,
                total_completion_tokens=0,
                total_tokens=0,
                total_cost_usd=0.0,
                by_model={},
                num_requests=0,
                avg_cost_per_request=0.0,
            )

        total_prompt = sum(r.prompt_tokens for r in records)
        total_completion = sum(r.completion_tokens for r in records)
        total_cost = sum(r.total_cost_usd for r in records)

        # Group by model
        by_model: Dict[str, Dict[str, Any]] = {}
        for r in records:
            if r.model not in by_model:
                by_model[r.model] = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "total_cost_usd": 0.0,
                    "num_requests": 0,
                }
            by_model[r.model]["prompt_tokens"] += r.prompt_tokens
            by_model[r.model]["completion_tokens"] += r.completion_tokens
            by_model[r.model]["total_tokens"] += r.total_tokens
            by_model[r.model]["total_cost_usd"] += r.total_cost_usd
            by_model[r.model]["num_requests"] += 1

        return CostSummary(
            total_prompt_tokens=total_prompt,
            total_completion_tokens=total_completion,
            total_tokens=total_prompt + total_completion,
            total_cost_usd=total_cost,
            by_model=by_model,
            num_requests=len(records),
            avg_cost_per_request=total_cost / len(records) if records else 0.0,
        )

    def reset(self) -> None:
        """Clear all recorded costs and reset totals."""
        self.records.clear()
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_cost = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Export tracker state as dictionary.

        Returns:
            Dict with all tracked metrics.
        """
        summary = self.get_cumulative_cost()
        return {
            "total_prompt_tokens": summary.total_prompt_tokens,
            "total_completion_tokens": summary.total_completion_tokens,
            "total_tokens": summary.total_tokens,
            "total_cost_usd": summary.total_cost_usd,
            "num_requests": summary.num_requests,
            "avg_cost_per_request": summary.avg_cost_per_request,
            "by_model": summary.by_model,
            "model_costs_config": self.model_costs,
        }
