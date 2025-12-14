"""Thread-safe metrics tracking for game sessions."""

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Set, Any, Optional

from ..core.types import TokenMetrics


@dataclass
class MetricsTracker:
    """Thread-safe metrics tracking for game sessions.

    Tracks API performance, token usage, and cost metrics.
    """

    total_requests: int = 0
    successful_responses: int = 0
    failed_requests: int = 0
    response_times: List[float] = field(default_factory=list)
    detailed_logs: List[Dict[str, Any]] = field(default_factory=list)
    models_tested: Set[str] = field(default_factory=set)
    endpoints_used: Set[str] = field(default_factory=set)
    start_time: Optional[float] = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # Token tracking (new)
    token_usage: List[TokenMetrics] = field(default_factory=list)
    session_prompt_tokens: int = 0
    session_completion_tokens: int = 0
    session_estimated_cost: float = 0.0

    # Cumulative totals (persist across runs)
    _cumulative_requests: int = field(default=0, repr=False)
    _cumulative_successes: int = field(default=0, repr=False)
    _cumulative_failures: int = field(default=0, repr=False)
    _all_response_times: List[float] = field(default_factory=list, repr=False)
    _session_count: int = field(default=0, repr=False)
    # Cumulative token tracking
    _cumulative_prompt_tokens: int = field(default=0, repr=False)
    _cumulative_completion_tokens: int = field(default=0, repr=False)
    _cumulative_estimated_cost: float = field(default=0.0, repr=False)

    def reset(self) -> None:
        """Reset all metrics for a new run."""
        with self._lock:
            self.total_requests = 0
            self.successful_responses = 0
            self.failed_requests = 0
            self.response_times.clear()
            self.detailed_logs.clear()
            self.models_tested.clear()
            self.endpoints_used.clear()
            self.start_time = time.time()
            # Reset token tracking
            self.token_usage.clear()
            self.session_prompt_tokens = 0
            self.session_completion_tokens = 0
            self.session_estimated_cost = 0.0

    def soft_reset(self) -> None:
        """Reset session metrics but accumulate totals."""
        with self._lock:
            # Accumulate into cumulative totals
            self._cumulative_requests += self.total_requests
            self._cumulative_successes += self.successful_responses
            self._cumulative_failures += self.failed_requests
            self._all_response_times.extend(self.response_times)
            self._session_count += 1
            # Accumulate token metrics
            self._cumulative_prompt_tokens += self.session_prompt_tokens
            self._cumulative_completion_tokens += self.session_completion_tokens
            self._cumulative_estimated_cost += self.session_estimated_cost

            # Reset session-specific metrics
            self.total_requests = 0
            self.successful_responses = 0
            self.failed_requests = 0
            self.response_times.clear()
            self.detailed_logs.clear()
            self.start_time = time.time()
            # Reset token tracking
            self.token_usage.clear()
            self.session_prompt_tokens = 0
            self.session_completion_tokens = 0
            self.session_estimated_cost = 0.0
            # Keep models_tested and endpoints_used for cross-session tracking

    def log_request_start(self, num_requests: int = 1) -> None:
        """Log the start of request(s)."""
        with self._lock:
            self.total_requests += num_requests
            if self.start_time is None:
                self.start_time = time.time()

    def log_success(
        self,
        model: str,
        endpoint: str,
        response_time: float,
        player_id: int,
        response: str,
    ) -> None:
        """Log a successful response."""
        with self._lock:
            self.successful_responses += 1
            self.response_times.append(response_time)
            self.models_tested.add(model)
            self.endpoints_used.add(endpoint)
            self.detailed_logs.append({
                "timestamp": datetime.now().isoformat(),
                "model": model,
                "endpoint": endpoint,
                "player_id": player_id,
                "response_time": response_time,
                "status": "success",
                "response": response,
            })

    def log_error(
        self,
        model: str,
        endpoint: str,
        response_time: float,
        player_id: int,
        error_type: str,
        error_message: str,
    ) -> None:
        """Log a failed request."""
        with self._lock:
            self.failed_requests += 1
            self.detailed_logs.append({
                "timestamp": datetime.now().isoformat(),
                "model": model,
                "endpoint": endpoint,
                "player_id": player_id,
                "response_time": response_time,
                "status": error_type,
                "response": error_message,
            })

    def log_token_usage(
        self,
        session_id: str,
        round_number: int,
        player_id: int,
        model: str,
        prompt_tokens: Optional[int],
        completion_tokens: Optional[int],
        cost_per_prompt_token: float = 0.0,
        cost_per_completion_token: float = 0.0,
    ) -> Optional[TokenMetrics]:
        """Log token usage from an LLM request.

        Args:
            session_id: Current session identifier
            round_number: Current round number
            player_id: Player making the request
            model: Model name
            prompt_tokens: Number of prompt tokens (None if unavailable)
            completion_tokens: Number of completion tokens (None if unavailable)
            cost_per_prompt_token: Cost per 1K prompt tokens (default 0.0 for local)
            cost_per_completion_token: Cost per 1K completion tokens

        Returns:
            TokenMetrics object if token data available, None otherwise
        """
        if prompt_tokens is None or completion_tokens is None:
            return None

        total_tokens = prompt_tokens + completion_tokens
        estimated_cost = (
            prompt_tokens * cost_per_prompt_token / 1000 +
            completion_tokens * cost_per_completion_token / 1000
        )

        metrics = TokenMetrics(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=estimated_cost,
            model=model,
            session_id=session_id,
            round_number=round_number,
            player_id=player_id,
        )

        with self._lock:
            self.token_usage.append(metrics)
            self.session_prompt_tokens += prompt_tokens
            self.session_completion_tokens += completion_tokens
            self.session_estimated_cost += estimated_cost

        return metrics

    def get_avg_response_time(self) -> float:
        """Get average response time."""
        with self._lock:
            if not self.response_times:
                return 0.0
            return sum(self.response_times) / len(self.response_times)

    def get_success_rate(self) -> float:
        """Get success rate as a percentage."""
        with self._lock:
            total = self.successful_responses + self.failed_requests
            if total == 0:
                return 100.0
            return (self.successful_responses / total) * 100

    def get_elapsed_time(self) -> float:
        """Get elapsed time since start."""
        with self._lock:
            if self.start_time is None:
                return 0.0
            return time.time() - self.start_time

    def get_token_summary(self) -> Dict[str, Any]:
        """Get summary of token usage for current session.

        Returns:
            Dict with token counts, costs, and per-model breakdowns.
        """
        with self._lock:
            total_tokens = self.session_prompt_tokens + self.session_completion_tokens

            # Per-model breakdown
            model_breakdown: Dict[str, Dict[str, Any]] = {}
            for tm in self.token_usage:
                if tm.model not in model_breakdown:
                    model_breakdown[tm.model] = {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                        "estimated_cost_usd": 0.0,
                        "request_count": 0,
                    }
                model_breakdown[tm.model]["prompt_tokens"] += tm.prompt_tokens
                model_breakdown[tm.model]["completion_tokens"] += tm.completion_tokens
                model_breakdown[tm.model]["total_tokens"] += tm.total_tokens
                model_breakdown[tm.model]["estimated_cost_usd"] += tm.estimated_cost_usd
                model_breakdown[tm.model]["request_count"] += 1

            return {
                "session_prompt_tokens": self.session_prompt_tokens,
                "session_completion_tokens": self.session_completion_tokens,
                "session_total_tokens": total_tokens,
                "session_estimated_cost_usd": self.session_estimated_cost,
                "requests_with_token_data": len(self.token_usage),
                "avg_tokens_per_request": (
                    total_tokens / len(self.token_usage) if self.token_usage else 0
                ),
                "by_model": model_breakdown,
            }

    def to_dict(self) -> Dict[str, Any]:
        """Export metrics as dictionary for visualization."""
        with self._lock:
            elapsed = time.time() - self.start_time if self.start_time else 0
            total_tokens = self.session_prompt_tokens + self.session_completion_tokens
            return {
                "total_requests": self.total_requests,
                "successful_responses": self.successful_responses,
                "failed_requests": self.failed_requests,
                "avg_response_time": (
                    sum(self.response_times) / len(self.response_times)
                    if self.response_times
                    else 0
                ),
                "min_response_time": min(self.response_times) if self.response_times else 0,
                "max_response_time": max(self.response_times) if self.response_times else 0,
                "success_rate": (
                    self.successful_responses / (self.successful_responses + self.failed_requests) * 100
                    if (self.successful_responses + self.failed_requests) > 0
                    else 100
                ),
                "models_tested": list(self.models_tested),
                "endpoints_used": list(self.endpoints_used),
                "elapsed_seconds": elapsed,
                "requests_per_second": (
                    self.total_requests / elapsed if elapsed > 0 else 0
                ),
                # Token metrics
                "prompt_tokens": self.session_prompt_tokens,
                "completion_tokens": self.session_completion_tokens,
                "total_tokens": total_tokens,
                "estimated_cost_usd": self.session_estimated_cost,
                "tokens_per_request": (
                    total_tokens / len(self.token_usage) if self.token_usage else 0
                ),
            }

    def get_cumulative(self) -> Dict[str, Any]:
        """Get cumulative metrics across all sessions."""
        with self._lock:
            # Include current session in cumulative
            total_requests = self._cumulative_requests + self.total_requests
            total_successes = self._cumulative_successes + self.successful_responses
            total_failures = self._cumulative_failures + self.failed_requests
            all_times = self._all_response_times + self.response_times
            # Cumulative token metrics
            total_prompt_tokens = self._cumulative_prompt_tokens + self.session_prompt_tokens
            total_completion_tokens = self._cumulative_completion_tokens + self.session_completion_tokens
            total_tokens = total_prompt_tokens + total_completion_tokens
            total_cost = self._cumulative_estimated_cost + self.session_estimated_cost

            return {
                "total_requests": total_requests,
                "successful_responses": total_successes,
                "failed_requests": total_failures,
                "avg_response_time": (
                    sum(all_times) / len(all_times) if all_times else 0
                ),
                "min_response_time": min(all_times) if all_times else 0,
                "max_response_time": max(all_times) if all_times else 0,
                "success_rate": (
                    total_successes / (total_successes + total_failures) * 100
                    if (total_successes + total_failures) > 0
                    else 100
                ),
                "models_tested": list(self.models_tested),
                "endpoints_used": list(self.endpoints_used),
                "sessions_count": self._session_count + (1 if self.total_requests > 0 else 0),
                # Cumulative token metrics
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
                "total_tokens": total_tokens,
                "total_estimated_cost_usd": total_cost,
            }

    def get_recent_logs(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get the n most recent log entries."""
        with self._lock:
            return self.detailed_logs[-n:]
