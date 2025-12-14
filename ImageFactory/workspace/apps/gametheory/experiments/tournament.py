"""Tournament infrastructure for multi-model competitions.

Supports round-robin, elimination, and Swiss-style tournaments
with comprehensive result tracking and analysis.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable, Tuple
import asyncio
import aiohttp
import numpy as np
from datetime import datetime
from itertools import combinations
import uuid


@dataclass
class TournamentConfig:
    """Configuration for a tournament."""
    models: List[str]
    games: List[str]
    endpoint: str = "http://localhost:11434"
    rounds_per_matchup: int = 10
    format: str = "round_robin"    # "round_robin", "elimination", "swiss"
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1


@dataclass
class MatchResult:
    """Result of a single match between players."""
    player1_model: str
    player2_model: str
    game_type: str
    rounds_played: int
    player1_wins: int
    player2_wins: int
    draws: int
    player1_total_payoff: float
    player2_total_payoff: float
    avg_response_times: Tuple[float, float]


@dataclass
class TournamentStanding:
    """Standing for a single model in tournament."""
    model: str
    wins: int
    losses: int
    draws: int
    points: float
    matches_played: int
    total_payoff: float
    avg_payoff_per_round: float
    win_rate: float


@dataclass
class TournamentResult:
    """Complete tournament results."""
    tournament_id: str
    config: TournamentConfig
    started_at: datetime
    completed_at: Optional[datetime]
    status: str  # "running", "completed", "failed"
    standings: List[TournamentStanding]
    match_results: List[MatchResult]
    matchup_matrix: np.ndarray     # NxN win rates (model i vs model j)
    model_indices: Dict[str, int]  # model name -> index in matrix


class TournamentRunner:
    """Runs multi-model tournaments with various formats."""

    def __init__(
        self,
        config: TournamentConfig,
        runner_factory: Callable,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ):
        """Initialize tournament runner.

        Args:
            config: Tournament configuration
            runner_factory: Function(game_id) -> BurrGameRunner
            progress_callback: Optional callback(completed, total, message)
        """
        self.config = config
        self.runner_factory = runner_factory
        self.progress_callback = progress_callback
        self.match_results: List[MatchResult] = []

    def _create_player_config(
        self,
        model: str,
        player_id: int
    ) -> 'PlayerConfig':
        """Create PlayerConfig for a model."""
        from ..core.types import PlayerConfig

        return PlayerConfig(
            player_id=player_id,
            model=model,
            endpoint=self.config.endpoint,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            repeat_penalty=self.config.repeat_penalty,
        )

    async def _run_match(
        self,
        session: aiohttp.ClientSession,
        model1: str,
        model2: str,
        game_id: str
    ) -> MatchResult:
        """Run a match between two models.

        Args:
            session: aiohttp session
            model1: First model
            model2: Second model
            game_id: Game to play

        Returns:
            MatchResult with match statistics
        """
        runner = self.runner_factory(game_id)
        game = runner.game

        player1 = self._create_player_config(model1, 1)
        player2 = self._create_player_config(model2, 2)
        players = [player1, player2]

        # Run rounds
        history = []
        history_payoffs = []
        cumulative_payoffs = (0.0, 0.0)

        p1_wins = 0
        p2_wins = 0
        draws = 0
        p1_total_payoff = 0.0
        p2_total_payoff = 0.0
        response_times_1 = []
        response_times_2 = []

        for round_num in range(self.config.rounds_per_matchup):
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

                # Track results
                p1_payoff, p2_payoff = payoffs
                p1_total_payoff += p1_payoff
                p2_total_payoff += p2_payoff

                if p1_payoff > p2_payoff:
                    p1_wins += 1
                elif p2_payoff > p1_payoff:
                    p2_wins += 1
                else:
                    draws += 1

                response_times_1.append(response_times[0])
                response_times_2.append(response_times[1])

                # Update history
                history.append(tuple(tuple(a) for a in actions))
                history_payoffs.append(payoffs)
                cumulative_payoffs = (
                    cumulative_payoffs[0] + p1_payoff,
                    cumulative_payoffs[1] + p2_payoff,
                )

            except Exception as e:
                # Continue with remaining rounds on error
                pass

        rounds_played = p1_wins + p2_wins + draws

        return MatchResult(
            player1_model=model1,
            player2_model=model2,
            game_type=game_id,
            rounds_played=rounds_played,
            player1_wins=p1_wins,
            player2_wins=p2_wins,
            draws=draws,
            player1_total_payoff=p1_total_payoff,
            player2_total_payoff=p2_total_payoff,
            avg_response_times=(
                sum(response_times_1) / len(response_times_1) if response_times_1 else 0,
                sum(response_times_2) / len(response_times_2) if response_times_2 else 0,
            ),
        )

    def _generate_round_robin_matchups(self) -> List[Tuple[str, str, str]]:
        """Generate all round-robin matchups.

        Returns:
            List of (model1, model2, game_id) tuples
        """
        matchups = []

        for model1, model2 in combinations(self.config.models, 2):
            for game_id in self.config.games:
                matchups.append((model1, model2, game_id))

        return matchups

    def _generate_swiss_matchups(
        self,
        standings: List[TournamentStanding],
        round_num: int
    ) -> List[Tuple[str, str, str]]:
        """Generate Swiss-style matchups based on current standings.

        Pairs models with similar scores together.

        Args:
            standings: Current standings sorted by points
            round_num: Current round number

        Returns:
            List of matchups for this round
        """
        # Sort by points
        sorted_standings = sorted(standings, key=lambda x: x.points, reverse=True)
        models = [s.model for s in sorted_standings]

        matchups = []
        used = set()

        for i, model1 in enumerate(models):
            if model1 in used:
                continue

            # Find closest unmatched opponent
            for model2 in models[i+1:]:
                if model2 not in used:
                    for game_id in self.config.games:
                        matchups.append((model1, model2, game_id))
                    used.add(model1)
                    used.add(model2)
                    break

        return matchups

    def _calculate_standings(self) -> List[TournamentStanding]:
        """Calculate current standings from match results."""
        model_stats = {}

        for model in self.config.models:
            model_stats[model] = {
                "wins": 0,
                "losses": 0,
                "draws": 0,
                "matches": 0,
                "total_payoff": 0.0,
                "total_rounds": 0,
            }

        for match in self.match_results:
            # Player 1
            if match.player1_wins > match.player2_wins:
                model_stats[match.player1_model]["wins"] += 1
                model_stats[match.player2_model]["losses"] += 1
            elif match.player2_wins > match.player1_wins:
                model_stats[match.player1_model]["losses"] += 1
                model_stats[match.player2_model]["wins"] += 1
            else:
                model_stats[match.player1_model]["draws"] += 1
                model_stats[match.player2_model]["draws"] += 1

            model_stats[match.player1_model]["matches"] += 1
            model_stats[match.player2_model]["matches"] += 1
            model_stats[match.player1_model]["total_payoff"] += match.player1_total_payoff
            model_stats[match.player2_model]["total_payoff"] += match.player2_total_payoff
            model_stats[match.player1_model]["total_rounds"] += match.rounds_played
            model_stats[match.player2_model]["total_rounds"] += match.rounds_played

        standings = []
        for model, stats in model_stats.items():
            matches = stats["matches"]
            total_games = stats["wins"] + stats["losses"] + stats["draws"]

            standings.append(TournamentStanding(
                model=model,
                wins=stats["wins"],
                losses=stats["losses"],
                draws=stats["draws"],
                points=stats["wins"] + 0.5 * stats["draws"],
                matches_played=matches,
                total_payoff=stats["total_payoff"],
                avg_payoff_per_round=(
                    stats["total_payoff"] / stats["total_rounds"]
                    if stats["total_rounds"] > 0 else 0
                ),
                win_rate=(
                    stats["wins"] / total_games
                    if total_games > 0 else 0
                ),
            ))

        return sorted(standings, key=lambda x: x.points, reverse=True)

    def _build_matchup_matrix(self) -> Tuple[np.ndarray, Dict[str, int]]:
        """Build head-to-head win rate matrix.

        Returns:
            Tuple of (NxN matrix, model_name -> index dict)
        """
        models = self.config.models
        n = len(models)
        model_indices = {m: i for i, m in enumerate(models)}

        matrix = np.zeros((n, n))

        for match in self.match_results:
            i = model_indices[match.player1_model]
            j = model_indices[match.player2_model]

            total = match.player1_wins + match.player2_wins + match.draws
            if total > 0:
                # Win rate for model i against model j
                matrix[i, j] += match.player1_wins / total
                matrix[j, i] += match.player2_wins / total

        # Normalize by number of games between each pair
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Count matches between these models
                    match_count = sum(
                        1 for m in self.match_results
                        if (m.player1_model == models[i] and m.player2_model == models[j]) or
                           (m.player1_model == models[j] and m.player2_model == models[i])
                    )
                    if match_count > 1:
                        matrix[i, j] /= match_count

        return matrix, model_indices

    async def run_tournament(
        self,
        session: aiohttp.ClientSession,
        timeout_per_match: float = 300.0
    ) -> TournamentResult:
        """Execute the tournament.

        Args:
            session: aiohttp session
            timeout_per_match: Timeout per match in seconds

        Returns:
            TournamentResult with complete results
        """
        tournament_id = str(uuid.uuid4())[:12]
        started_at = datetime.utcnow()

        self.match_results = []

        if self.config.format == "round_robin":
            matchups = self._generate_round_robin_matchups()
        elif self.config.format == "swiss":
            # Swiss needs rounds
            num_swiss_rounds = min(len(self.config.models) - 1, 5)
            matchups = []
        else:
            matchups = self._generate_round_robin_matchups()

        total_matchups = len(matchups) if matchups else (
            len(self.config.models) * len(self.config.games) // 2
        )
        completed = 0

        if self.config.format == "swiss":
            # Swiss tournament
            standings = self._calculate_standings()

            for round_num in range(num_swiss_rounds):
                round_matchups = self._generate_swiss_matchups(standings, round_num)

                for model1, model2, game_id in round_matchups:
                    if self.progress_callback:
                        self.progress_callback(
                            completed, total_matchups,
                            f"Swiss R{round_num+1}: {model1} vs {model2} ({game_id})"
                        )

                    try:
                        result = await asyncio.wait_for(
                            self._run_match(session, model1, model2, game_id),
                            timeout=timeout_per_match,
                        )
                        self.match_results.append(result)
                    except asyncio.TimeoutError:
                        pass

                    completed += 1

                standings = self._calculate_standings()
        else:
            # Round-robin or other formats
            for model1, model2, game_id in matchups:
                if self.progress_callback:
                    self.progress_callback(
                        completed, total_matchups,
                        f"{model1} vs {model2} ({game_id})"
                    )

                try:
                    result = await asyncio.wait_for(
                        self._run_match(session, model1, model2, game_id),
                        timeout=timeout_per_match,
                    )
                    self.match_results.append(result)
                except asyncio.TimeoutError:
                    pass
                except Exception:
                    pass

                completed += 1

        # Calculate final results
        standings = self._calculate_standings()
        matchup_matrix, model_indices = self._build_matchup_matrix()

        if self.progress_callback:
            self.progress_callback(
                total_matchups, total_matchups,
                "Tournament complete"
            )

        return TournamentResult(
            tournament_id=tournament_id,
            config=self.config,
            started_at=started_at,
            completed_at=datetime.utcnow(),
            status="completed",
            standings=standings,
            match_results=self.match_results,
            matchup_matrix=matchup_matrix,
            model_indices=model_indices,
        )

    def results_to_dict(self, result: TournamentResult) -> Dict[str, Any]:
        """Convert tournament result to serializable dict.

        Args:
            result: TournamentResult to convert

        Returns:
            Dict suitable for JSON serialization
        """
        return {
            "tournament_id": result.tournament_id,
            "format": result.config.format,
            "models": result.config.models,
            "games": result.config.games,
            "rounds_per_matchup": result.config.rounds_per_matchup,
            "started_at": result.started_at.isoformat(),
            "completed_at": result.completed_at.isoformat() if result.completed_at else None,
            "status": result.status,
            "standings": [
                {
                    "model": s.model,
                    "wins": s.wins,
                    "losses": s.losses,
                    "draws": s.draws,
                    "points": s.points,
                    "matches_played": s.matches_played,
                    "win_rate": s.win_rate,
                    "avg_payoff_per_round": s.avg_payoff_per_round,
                }
                for s in result.standings
            ],
            "match_results": [
                {
                    "player1": m.player1_model,
                    "player2": m.player2_model,
                    "game": m.game_type,
                    "p1_wins": m.player1_wins,
                    "p2_wins": m.player2_wins,
                    "draws": m.draws,
                }
                for m in result.match_results
            ],
            "matchup_matrix": result.matchup_matrix.tolist(),
            "model_indices": result.model_indices,
        }


def create_quick_tournament(
    models: List[str],
    game_id: str = "colonel_blotto_5",
    rounds: int = 5,
    endpoint: str = "http://localhost:11434"
) -> TournamentConfig:
    """Create a quick tournament configuration.

    Args:
        models: List of model names
        game_id: Single game to play
        rounds: Rounds per matchup
        endpoint: Ollama endpoint

    Returns:
        TournamentConfig for quick testing
    """
    return TournamentConfig(
        models=models,
        games=[game_id],
        endpoint=endpoint,
        rounds_per_matchup=rounds,
        format="round_robin",
    )


def create_full_tournament(
    models: List[str],
    endpoint: str = "http://localhost:11434"
) -> TournamentConfig:
    """Create a full tournament across all allocation games.

    Args:
        models: List of model names
        endpoint: Ollama endpoint

    Returns:
        TournamentConfig for comprehensive testing
    """
    return TournamentConfig(
        models=models,
        games=["colonel_blotto_3", "colonel_blotto_5", "colonel_blotto_7"],
        endpoint=endpoint,
        rounds_per_matchup=10,
        format="round_robin",
    )
