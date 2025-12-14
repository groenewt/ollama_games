"""Hamilton-Burr tracing bridge for unified observability."""
from typing import Any, Dict, Optional
from hamilton import lifecycle as h_lifecycle


class HamiltonBurrTracer(h_lifecycle.NodeExecutionHook):
    """Bridge Hamilton node execution to Burr spans."""

    def __init__(self, tracer=None):
        self._tracer = tracer
        self.active_spans: Dict[str, Any] = {}
        self.node_timings: Dict[str, float] = {}

    def run_before_node_execution(
        self,
        *,
        node_name: str,
        node_tags: Dict[str, Any],
        node_kwargs: Dict[str, Any],
        node_return_type: type,
        **kwargs,
    ):
        """Called before each Hamilton node executes."""
        import time
        self.node_timings[node_name] = time.time()

        if self._tracer:
            try:
                context_manager = self._tracer(node_name)
                context_manager.__enter__()
                self.active_spans[node_name] = context_manager
            except Exception:
                pass

    def run_after_node_execution(
        self,
        *,
        node_name: str,
        node_tags: Dict[str, Any],
        node_kwargs: Dict[str, Any],
        node_return_type: type,
        result: Any,
        **kwargs,
    ):
        """Called after each Hamilton node executes."""
        if node_name in self.active_spans:
            try:
                context_manager = self.active_spans.pop(node_name)
                context_manager.__exit__(None, None, None)
            except Exception:
                pass
