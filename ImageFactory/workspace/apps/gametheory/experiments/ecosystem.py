"""Multi-agent ecosystem simulation for extended tournaments.

Simulates 100+ round tournaments to study strategy evolution,
equilibrium emergence, and cyclical dynamics.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable, Tuple
import asyncio
import aiohttp
import numpy as np
from datetime import datetime
from collections import Counter
import uuid


@dataclass
class EcosystemState:
    """Snapshot of ecosystem state at a given round."""
    round_num: int
    archetype_distribution: Dict[str, int]  # strategy_type -> count
    dominant_strategy: str
    diversity_entropy: float               # Shannon entropy of archetypes
    win_rates: Dict[str, float]           # model -> win rate this round
    avg_concentration: float              # Average HHI across models


@dataclass
class EquilibriumAnalysis:
    """Analysis of equilibrium emergence."""
    converged: bool
    convergence_round: Optional[int]
    equilibrium_type: str          # "nash_approx", "cycling", "chaos", "none"
    dominant_archetype: str
    stability_score: float         # How stable the equilibrium is (0-1)
    explanation: str


@dataclass
class CyclicalPattern:
    """Detected cyclical pattern in strategy evolution."""
    detected: bool
    cycle_length: int
    cycle_archetypes: List[str]   # Sequence of dominant archetypes
    confidence: float
    explanation: str


@dataclass
class EcosystemResult:
    """Complete ecosystem simulation results."""
    simulation_id: str
    game_id: str
    models: List[str]
    total_rounds: int
    states: List[EcosystemState]
    equilibrium_analysis: EquilibriumAnalysis
    cyclical_patterns: CyclicalPattern
    final_standings: Dict[str, Dict[str, float]]  # model -> metrics
    started_at: datetime
    completed_at: Optional[datetime]


class EcosystemSimulator:
    """Simulates extended multi-agent tournaments."""

    def __init__(
        self,
        models: List[str],
        game_id: str,
        runner_factory: Callable,
        endpoint: str = "http://localhost:11434",
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ):
        """Initialize ecosystem simulator.

        Args:
            models: List of model names
            game_id: Game to simulate
            runner_factory: Function(game_id) -> BurrGameRunner
            endpoint: Ollama endpoint
            progress_callback: Optional callback(completed, total, message)
        """
        self.models = models
        self.game_id = game_id
        self.runner_factory = runner_factory
        self.endpoint = endpoint
        self.progress_callback = progress_callback

        # Per-model tracking
        self.model_allocations: Dict[str, List[List[float]]] = {m: [] for m in models}
        self.model_payoffs: Dict[str, List[float]] = {m: [] for m in models}
        self.model_wins: Dict[str, int] = {m: 0 for m in models}
        self.model_games: Dict[str, int] = {m: 0 for m in models}

    def _create_player_config(
        self,
        model: str,
        player_id: int,
        temperature: float = 0.7
    ) -> 'PlayerConfig':
        """Create PlayerConfig for a model."""
        from ..core.types import PlayerConfig

        return PlayerConfig(
            player_id=player_id,
            model=model,
            endpoint=self.endpoint,
            temperature=temperature,
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.1,
        )

    def _classify_allocation(self, allocation: List[float]) -> str:
        """Classify allocation into strategy archetype."""
        budget = sum(allocation)
        if budget == 0:
            return "unknown"

        proportions = [a / budget for a in allocation]
        hhi = sum(p ** 2 for p in proportions)
        num_fields = len(allocation)
        uniform_hhi = 1.0 / num_fields

        if hhi >= 0.5:
            return "concentrated"
        elif hhi <= uniform_hhi * 1.2:
            return "uniform"
        elif hhi <= uniform_hhi * 2.0:
            return "hedged"
        else:
            return "asymmetric"

    def _calculate_diversity_entropy(
        self,
        archetype_counts: Dict[str, int]
    ) -> float:
        """Calculate Shannon entropy of archetype distribution."""
        from math import log2

        total = sum(archetype_counts.values())
        if total == 0:
            return 0.0

        entropy = 0.0
        for count in archetype_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * log2(p)

        # Normalize by max possible entropy
        max_entropy = log2(len(archetype_counts)) if len(archetype_counts) > 1 else 1.0
        return entropy / max_entropy if max_entropy > 0 else 0.0

    async def _run_round(
        self,
        session: aiohttp.ClientSession,
        round_num: int,
        history: Dict[str, List[List[float]]]
    ) -> Dict[str, Any]:
        """Run a single round with all models.

        For simplicity, plays round-robin matches each round.

        Args:
            session: aiohttp session
            round_num: Current round number
            history: Per-model allocation history

        Returns:
            Dict with round results
        """
        from itertools import combinations
        import random

        runner = self.runner_factory(self.game_id)
        game = runner.game

        round_allocations = {}
        round_payoffs = {}
        round_wins = {m: 0 for m in self.models}
        round_games = {m: 0 for m in self.models}

        # Play each pair
        for model1, model2 in combinations(self.models, 2):
            player1 = self._create_player_config(model1, 1)
            player2 = self._create_player_config(model2, 2)
            players = [player1, player2]

            # Build history from last few rounds
            m1_history = history.get(model1, [])[-5:]
            m2_history = history.get(model2, [])[-5:]

            combined_history = []
            for h1, h2 in zip(m1_history, m2_history):
                combined_history.append((tuple(h1), tuple(h2)))

            try:
                result = await runner.play_round(
                    session,
                    players,
                    combined_history,
                    is_repeated=round_num > 0,
                )

                actions, payoffs, response_times, prompts, raw_responses, was_parsed, was_normalized = result

                # Track allocations
                if model1 not in round_allocations:
                    round_allocations[model1] = list(actions[0])
                if model2 not in round_allocations:
                    round_allocations[model2] = list(actions[1])

                # Track payoffs
                round_payoffs.setdefault(model1, []).append(payoffs[0])
                round_payoffs.setdefault(model2, []).append(payoffs[1])

                # Track wins
                round_games[model1] += 1
                round_games[model2] += 1
                if payoffs[0] > payoffs[1]:
                    round_wins[model1] += 1
                elif payoffs[1] > payoffs[0]:
                    round_wins[model2] += 1

            except Exception:
                pass

        return {
            "allocations": round_allocations,
            "payoffs": round_payoffs,
            "wins": round_wins,
            "games": round_games,
        }

    def _capture_state(
        self,
        round_num: int,
        round_results: Dict[str, Any]
    ) -> EcosystemState:
        """Capture ecosystem state for a round."""
        # Count archetypes
        archetype_counts = Counter()
        concentrations = []

        for model, allocation in round_results.get("allocations", {}).items():
            archetype = self._classify_allocation(allocation)
            archetype_counts[archetype] += 1

            # Track concentration
            budget = sum(allocation)
            if budget > 0:
                proportions = [a / budget for a in allocation]
                hhi = sum(p ** 2 for p in proportions)
                concentrations.append(hhi)

        # Dominant strategy
        if archetype_counts:
            dominant = archetype_counts.most_common(1)[0][0]
        else:
            dominant = "unknown"

        # Diversity
        diversity = self._calculate_diversity_entropy(dict(archetype_counts))

        # Win rates this round
        win_rates = {}
        for model in self.models:
            games = round_results.get("games", {}).get(model, 0)
            wins = round_results.get("wins", {}).get(model, 0)
            win_rates[model] = wins / games if games > 0 else 0.0

        # Average concentration
        avg_concentration = sum(concentrations) / len(concentrations) if concentrations else 0.0

        return EcosystemState(
            round_num=round_num,
            archetype_distribution=dict(archetype_counts),
            dominant_strategy=dominant,
            diversity_entropy=diversity,
            win_rates=win_rates,
            avg_concentration=avg_concentration,
        )

    def detect_equilibrium(
        self,
        states: List[EcosystemState],
        window: int = 10
    ) -> EquilibriumAnalysis:
        """Detect if ecosystem has reached equilibrium.

        Args:
            states: List of ecosystem states
            window: Window size for stability check

        Returns:
            EquilibriumAnalysis with convergence info
        """
        if len(states) < window * 2:
            return EquilibriumAnalysis(
                converged=False,
                convergence_round=None,
                equilibrium_type="insufficient_data",
                dominant_archetype="unknown",
                stability_score=0.0,
                explanation="Not enough rounds for equilibrium analysis",
            )

        # Check stability in recent rounds
        recent_states = states[-window:]
        dominant_strategies = [s.dominant_strategy for s in recent_states]

        # Check if dominant strategy is stable
        strategy_counts = Counter(dominant_strategies)
        most_common, count = strategy_counts.most_common(1)[0]
        stability = count / window

        # Check diversity trend
        diversities = [s.diversity_entropy for s in recent_states]
        diversity_stable = np.std(diversities) < 0.1

        # Detect convergence point
        convergence_round = None
        if stability > 0.7:
            # Find when this stability emerged
            for i in range(len(states) - window):
                window_strategies = [s.dominant_strategy for s in states[i:i+window]]
                if window_strategies.count(most_common) / window > 0.7:
                    convergence_round = states[i].round_num
                    break

        # Determine equilibrium type
        if stability > 0.8 and diversity_stable:
            equilibrium_type = "nash_approx"
            explanation = f"Converged to {most_common} strategy with {stability:.0%} stability"
            converged = True
        elif stability < 0.4:
            equilibrium_type = "chaos"
            explanation = "No dominant strategy, high variance"
            converged = False
        else:
            equilibrium_type = "partial"
            explanation = f"Partial convergence to {most_common} ({stability:.0%})"
            converged = stability > 0.6

        return EquilibriumAnalysis(
            converged=converged,
            convergence_round=convergence_round,
            equilibrium_type=equilibrium_type,
            dominant_archetype=most_common,
            stability_score=stability,
            explanation=explanation,
        )

    def detect_cyclical_patterns(
        self,
        states: List[EcosystemState],
        min_cycle_length: int = 3,
        max_cycle_length: int = 10
    ) -> CyclicalPattern:
        """Detect cyclical patterns in strategy evolution.

        Args:
            states: List of ecosystem states
            min_cycle_length: Minimum cycle to detect
            max_cycle_length: Maximum cycle to detect

        Returns:
            CyclicalPattern with cycle info
        """
        if len(states) < max_cycle_length * 3:
            return CyclicalPattern(
                detected=False,
                cycle_length=0,
                cycle_archetypes=[],
                confidence=0.0,
                explanation="Not enough rounds for cycle detection",
            )

        strategies = [s.dominant_strategy for s in states]

        best_cycle_length = 0
        best_cycle = []
        best_confidence = 0.0

        for cycle_len in range(min_cycle_length, max_cycle_length + 1):
            # Try to find repeating pattern of this length
            matches = 0
            total_checks = 0

            for i in range(cycle_len, len(strategies) - cycle_len):
                for j in range(cycle_len):
                    if strategies[i + j] == strategies[i - cycle_len + j]:
                        matches += 1
                    total_checks += 1

            if total_checks > 0:
                confidence = matches / total_checks
                if confidence > best_confidence and confidence > 0.5:
                    best_confidence = confidence
                    best_cycle_length = cycle_len
                    best_cycle = strategies[-cycle_len:]

        if best_confidence > 0.5:
            return CyclicalPattern(
                detected=True,
                cycle_length=best_cycle_length,
                cycle_archetypes=best_cycle,
                confidence=best_confidence,
                explanation=f"Detected {best_cycle_length}-round cycle with {best_confidence:.0%} confidence",
            )
        else:
            return CyclicalPattern(
                detected=False,
                cycle_length=0,
                cycle_archetypes=[],
                confidence=best_confidence,
                explanation="No significant cyclical patterns detected",
            )

    async def run_extended_tournament(
        self,
        rounds: int = 100,
        track_interval: int = 1,
        timeout_per_round: float = 60.0
    ) -> EcosystemResult:
        """Run extended tournament simulation.

        Args:
            rounds: Total rounds to simulate
            track_interval: Interval for state tracking
            timeout_per_round: Timeout per round

        Returns:
            EcosystemResult with full simulation data
        """
        simulation_id = str(uuid.uuid4())[:12]
        started_at = datetime.utcnow()

        states = []
        history: Dict[str, List[List[float]]] = {m: [] for m in self.models}

        connector = aiohttp.TCPConnector(limit=20, keepalive_timeout=30)
        async with aiohttp.ClientSession(connector=connector) as session:
            for round_num in range(rounds):
                if self.progress_callback:
                    self.progress_callback(
                        round_num, rounds,
                        f"Round {round_num + 1}/{rounds}"
                    )

                try:
                    round_results = await asyncio.wait_for(
                        self._run_round(session, round_num, history),
                        timeout=timeout_per_round,
                    )

                    # Update history
                    for model, allocation in round_results.get("allocations", {}).items():
                        history[model].append(allocation)
                        self.model_allocations[model].append(allocation)

                    # Update cumulative stats
                    for model, payoffs in round_results.get("payoffs", {}).items():
                        self.model_payoffs[model].extend(payoffs)

                    for model, wins in round_results.get("wins", {}).items():
                        self.model_wins[model] += wins
                        self.model_games[model] += round_results.get("games", {}).get(model, 0)

                    # Capture state
                    if round_num % track_interval == 0:
                        state = self._capture_state(round_num, round_results)
                        states.append(state)

                except asyncio.TimeoutError:
                    pass
                except Exception:
                    pass

        # Final analysis
        equilibrium = self.detect_equilibrium(states)
        cycles = self.detect_cyclical_patterns(states)

        # Final standings
        final_standings = {}
        for model in self.models:
            total_games = self.model_games[model]
            total_wins = self.model_wins[model]
            total_payoff = sum(self.model_payoffs[model])
            num_payoffs = len(self.model_payoffs[model])

            final_standings[model] = {
                "total_games": total_games,
                "total_wins": total_wins,
                "win_rate": total_wins / total_games if total_games > 0 else 0,
                "total_payoff": total_payoff,
                "avg_payoff": total_payoff / num_payoffs if num_payoffs > 0 else 0,
            }

        if self.progress_callback:
            self.progress_callback(rounds, rounds, "Simulation complete")

        return EcosystemResult(
            simulation_id=simulation_id,
            game_id=self.game_id,
            models=self.models,
            total_rounds=rounds,
            states=states,
            equilibrium_analysis=equilibrium,
            cyclical_patterns=cycles,
            final_standings=final_standings,
            started_at=started_at,
            completed_at=datetime.utcnow(),
        )

    def results_to_dict(self, result: EcosystemResult) -> Dict[str, Any]:
        """Convert result to serializable dict."""
        return {
            "simulation_id": result.simulation_id,
            "game_id": result.game_id,
            "models": result.models,
            "total_rounds": result.total_rounds,
            "started_at": result.started_at.isoformat(),
            "completed_at": result.completed_at.isoformat() if result.completed_at else None,
            "equilibrium": {
                "converged": result.equilibrium_analysis.converged,
                "type": result.equilibrium_analysis.equilibrium_type,
                "dominant": result.equilibrium_analysis.dominant_archetype,
                "stability": result.equilibrium_analysis.stability_score,
            },
            "cycles": {
                "detected": result.cyclical_patterns.detected,
                "length": result.cyclical_patterns.cycle_length,
                "confidence": result.cyclical_patterns.confidence,
            },
            "final_standings": result.final_standings,
            "states": [
                {
                    "round": s.round_num,
                    "dominant": s.dominant_strategy,
                    "diversity": s.diversity_entropy,
                    "archetypes": s.archetype_distribution,
                }
                for s in result.states
            ],
        }
