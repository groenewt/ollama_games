"""Hyperparameter sweep infrastructure for allocation game research.

Enables systematic exploration of model parameter effects on
strategy selection and compliance in allocation games.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Tuple
from itertools import product
import asyncio
import aiohttp
from datetime import datetime
import uuid


@dataclass
class SweepConfig:
    """Configuration for a hyperparameter sweep."""
    parameter_grid: Dict[str, List[Any]]  # e.g., {'temperature': [0.1, 0.5, 0.9]}
    num_rounds_per_config: int = 10
    game_id: str = "colonel_blotto_5"
    endpoint: str = "http://localhost:11434"
    model: str = "llama3.2:1b"

    def num_configurations(self) -> int:
        """Calculate total number of configurations."""
        if not self.parameter_grid:
            return 0
        return len(list(product(*self.parameter_grid.values())))


@dataclass
class SweepResult:
    """Result from a single configuration in a sweep."""
    config_id: str
    parameters: Dict[str, Any]
    rounds: List[Dict[str, Any]]
    metrics: Dict[str, float]  # Aggregated metrics
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SweepSummary:
    """Summary of a complete sweep."""
    sweep_id: str
    game_id: str
    total_configs: int
    completed_configs: int
    results: List[SweepResult]
    parameter_grid: Dict[str, List[Any]]
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: str = "running"  # running, completed, failed


class HyperparameterSweeper:
    """Executes hyperparameter sweeps for allocation games."""

    def __init__(
        self,
        game_runner_factory: Callable,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ):
        """Initialize sweeper.

        Args:
            game_runner_factory: Function that creates a BurrGameRunner for a game_id
            progress_callback: Optional callback(completed, total, message)
        """
        self.game_runner_factory = game_runner_factory
        self.progress_callback = progress_callback

    def generate_configs(
        self,
        grid: Dict[str, List[Any]]
    ) -> List[Dict[str, Any]]:
        """Generate all parameter combinations from grid.

        Args:
            grid: Dict mapping parameter names to lists of values

        Returns:
            List of parameter dicts, one per combination
        """
        if not grid:
            return [{}]

        keys = list(grid.keys())
        values = list(grid.values())

        configs = []
        for combo in product(*values):
            config = dict(zip(keys, combo))
            configs.append(config)

        return configs

    async def run_single_config(
        self,
        config: Dict[str, Any],
        sweep_config: SweepConfig,
        session: aiohttp.ClientSession
    ) -> SweepResult:
        """Run a single parameter configuration.

        Args:
            config: Parameter values for this run
            sweep_config: Overall sweep configuration
            session: aiohttp session for API calls

        Returns:
            SweepResult with rounds and metrics
        """
        from ..core.types import PlayerConfig
        from ..analytics.allocation import AllocationAnalyzer

        # Create player config with sweep parameters
        player_config = PlayerConfig(
            player_id=1,
            model=sweep_config.model,
            endpoint=sweep_config.endpoint,
            temperature=config.get("temperature", 0.7),
            top_p=config.get("top_p", 0.9),
            top_k=config.get("top_k", 40),
            repeat_penalty=config.get("repeat_penalty", 1.1),
            system_prompt=config.get("system_prompt"),
            strategy_hints=config.get("strategy_hints"),
        )

        # Create second player with default settings for comparison
        opponent_config = PlayerConfig(
            player_id=2,
            model=sweep_config.model,
            endpoint=sweep_config.endpoint,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.1,
        )

        players = [player_config, opponent_config]

        # Get game runner
        runner = self.game_runner_factory(sweep_config.game_id)
        game = runner.game
        # Budget is only available for allocation games (e.g., Colonel Blotto)
        budget = game.action_space.budget if game.is_allocation else None

        # Run rounds
        rounds = []
        history = []
        history_payoffs = []
        cumulative_payoffs = (0.0, 0.0)

        for round_num in range(sweep_config.num_rounds_per_config):
            try:
                result = await runner.play_round(
                    session,
                    players,
                    history,
                    history_payoffs=history_payoffs,
                    cumulative_payoffs=cumulative_payoffs,
                    is_repeated=round_num > 0,
                )

                actions, payoffs, response_times, prompts, raw_responses, was_parsed, was_normalized = result

                # Store round result
                round_data = {
                    "round": round_num + 1,
                    "player1_allocation": list(actions[0]),
                    "player2_allocation": list(actions[1]),
                    "player1_payoff": payoffs[0],
                    "player2_payoff": payoffs[1],
                    "player1_response_time": response_times[0],
                    "player2_response_time": response_times[1],
                    "player1_was_parsed": was_parsed[0],
                    "player2_was_parsed": was_parsed[1],
                    "player1_was_normalized": was_normalized[0],
                    "player2_was_normalized": was_normalized[1],
                }
                rounds.append(round_data)

                # Update history
                history.append(tuple(tuple(a) for a in actions))
                history_payoffs.append(payoffs)
                cumulative_payoffs = (
                    cumulative_payoffs[0] + payoffs[0],
                    cumulative_payoffs[1] + payoffs[1],
                )

            except Exception as e:
                rounds.append({
                    "round": round_num + 1,
                    "error": str(e),
                })

        # Calculate metrics
        analyzer = AllocationAnalyzer()
        metrics = self._calculate_metrics(rounds, analyzer, budget)

        return SweepResult(
            config_id=str(uuid.uuid4())[:8],
            parameters=config,
            rounds=rounds,
            metrics=metrics,
        )

    def _calculate_metrics(
        self,
        rounds: List[Dict],
        analyzer: 'AllocationAnalyzer',
        budget: Optional[float]
    ) -> Dict[str, float]:
        """Calculate aggregated metrics from rounds.

        Args:
            rounds: List of round results
            analyzer: AllocationAnalyzer instance
            budget: Allocation budget (None for non-allocation games)

        Returns:
            Dict of metric name to value
        """
        valid_rounds = [r for r in rounds if "error" not in r]

        if not valid_rounds:
            return {
                "total_payoff": 0.0,
                "avg_payoff": 0.0,
                "win_rate": 0.0,
                "parse_rate": 0.0,
                "normalize_rate": 0.0,
                "avg_concentration": 0.0,
                "num_rounds": 0,
            }

        # Payoff metrics
        total_payoff = sum(r.get("player1_payoff", 0) for r in valid_rounds)
        avg_payoff = total_payoff / len(valid_rounds)

        wins = sum(1 for r in valid_rounds if r.get("player1_payoff", 0) > r.get("player2_payoff", 0))
        win_rate = wins / len(valid_rounds)

        # Compliance metrics
        parsed = sum(1 for r in valid_rounds if r.get("player1_was_parsed", False))
        normalized = sum(1 for r in valid_rounds if r.get("player1_was_normalized", False))
        parse_rate = parsed / len(valid_rounds)
        normalize_rate = normalized / len(valid_rounds)

        # Strategy metrics (only for allocation games with budget)
        avg_concentration = 0.0
        if budget is not None:
            concentrations = []
            for r in valid_rounds:
                alloc = r.get("player1_allocation", [])
                if alloc:
                    hhi = analyzer.calculate_hhi(alloc, budget)
                    concentrations.append(hhi)

            avg_concentration = sum(concentrations) / len(concentrations) if concentrations else 0.0

        return {
            "total_payoff": total_payoff,
            "avg_payoff": avg_payoff,
            "win_rate": win_rate,
            "parse_rate": parse_rate,
            "normalize_rate": normalize_rate,
            "avg_concentration": avg_concentration,
            "num_rounds": len(valid_rounds),
        }

    async def run_sweep(
        self,
        sweep_config: SweepConfig,
        timeout_per_config: float = 120.0
    ) -> SweepSummary:
        """Execute a complete hyperparameter sweep.

        Args:
            sweep_config: Sweep configuration
            timeout_per_config: Timeout per configuration in seconds

        Returns:
            SweepSummary with all results
        """
        sweep_id = str(uuid.uuid4())[:12]
        started_at = datetime.utcnow()

        configs = self.generate_configs(sweep_config.parameter_grid)
        total_configs = len(configs)

        summary = SweepSummary(
            sweep_id=sweep_id,
            game_id=sweep_config.game_id,
            total_configs=total_configs,
            completed_configs=0,
            results=[],
            parameter_grid=sweep_config.parameter_grid,
            started_at=started_at,
        )

        async with aiohttp.ClientSession() as session:
            for i, config in enumerate(configs):
                if self.progress_callback:
                    self.progress_callback(
                        i, total_configs,
                        f"Running config {i+1}/{total_configs}: {config}"
                    )

                try:
                    result = await asyncio.wait_for(
                        self.run_single_config(config, sweep_config, session),
                        timeout=timeout_per_config,
                    )
                    summary.results.append(result)
                    summary.completed_configs += 1

                except asyncio.TimeoutError:
                    summary.results.append(SweepResult(
                        config_id=str(uuid.uuid4())[:8],
                        parameters=config,
                        rounds=[],
                        metrics={"error": "timeout"},
                    ))

                except Exception as e:
                    summary.results.append(SweepResult(
                        config_id=str(uuid.uuid4())[:8],
                        parameters=config,
                        rounds=[],
                        metrics={"error": str(e)},
                    ))

        summary.completed_at = datetime.utcnow()
        summary.status = "completed"

        if self.progress_callback:
            self.progress_callback(
                total_configs, total_configs,
                f"Sweep complete: {summary.completed_configs}/{total_configs} configs"
            )

        return summary

    def results_to_dataframe(
        self,
        summary: SweepSummary
    ) -> 'pl.DataFrame':
        """Convert sweep results to a Polars DataFrame.

        Args:
            summary: Sweep summary

        Returns:
            DataFrame with one row per configuration
        """
        import polars as pl

        rows = []
        for result in summary.results:
            row = {
                "config_id": result.config_id,
                **result.parameters,
                **result.metrics,
            }
            rows.append(row)

        return pl.DataFrame(rows)


def create_default_grid() -> Dict[str, List[Any]]:
    """Create a default parameter grid for exploration.

    Returns:
        Parameter grid dict
    """
    return {
        "temperature": [0.1, 0.3, 0.5, 0.7, 0.9],
        "top_k": [10, 20, 40, 60, 100],
        "repeat_penalty": [1.0, 1.1, 1.2, 1.3],
    }


def create_quick_grid() -> Dict[str, List[Any]]:
    """Create a quick parameter grid for rapid testing.

    Returns:
        Parameter grid dict with fewer combinations
    """
    return {
        "temperature": [0.2, 0.5, 0.8],
        "top_k": [20, 50],
    }


def create_compliance_grid() -> Dict[str, List[Any]]:
    """Create a grid focused on compliance testing.

    Returns:
        Parameter grid emphasizing format compliance
    """
    return {
        "temperature": [0.1, 0.3, 0.5],
        "top_k": [10, 30, 50],
        "repeat_penalty": [1.0, 1.2, 1.5],
    }
