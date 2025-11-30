"""Async client for Ollama API interactions."""

import asyncio
import time
import aiohttp
from typing import List, Tuple, Optional

from ..core.types import GameDefinition, PlayerConfig
from ..core.config import DEFAULT_TIMEOUT
from ..metrics.tracker import MetricsTracker


class OllamaClient:
    """Async client for querying Ollama models for game decisions."""

    def __init__(self, metrics: Optional[MetricsTracker] = None):
        """Initialize the client.

        Args:
            metrics: Optional MetricsTracker for logging requests.
        """
        self.metrics = metrics or MetricsTracker()

    def _build_prompt(
        self,
        game: GameDefinition,
        history: List[Tuple[str, ...]],
        player_id: int,
    ) -> str:
        """Generate contextual prompt for the game.

        Args:
            game: The game definition.
            history: List of previous round actions.
            player_id: The player's ID (1-indexed).

        Returns:
            The formatted prompt string.
        """
        # Format history for context
        if history:
            if game.num_players == 2:
                history_str = "\n".join([
                    f"Round {i+1}: Player 1 chose {actions[0]}, Player 2 chose {actions[1]}"
                    for i, actions in enumerate(history)
                ])
            else:
                history_str = "\n".join([
                    f"Round {i+1}: " + ", ".join([
                        f"Player {j+1} chose {action}"
                        for j, action in enumerate(actions)
                    ])
                    for i, actions in enumerate(history)
                ])
        else:
            history_str = "No previous rounds."

        actions_str = ", ".join([f'"{a}"' for a in game.actions])

        return f"""You are playing a repeated {game.name} game.
Previous rounds:
{history_str}

You are Player {player_id}. Your available choices are: {actions_str}.
What is your choice for this round? Respond with only one word from the available choices."""

    async def get_action(
        self,
        session: aiohttp.ClientSession,
        player: PlayerConfig,
        game: GameDefinition,
        history: List[Tuple[str, ...]],
        timeout: float = DEFAULT_TIMEOUT,
    ) -> Tuple[str, float, str, str]:
        """Query model for action decision.

        Args:
            session: The aiohttp session.
            player: Player configuration with model and endpoint.
            game: The game definition.
            history: List of previous round actions.
            timeout: Request timeout in seconds.

        Returns:
            Tuple of (action, response_time, prompt, raw_response).
        """
        prompt = self._build_prompt(game, history, player.player_id)
        start_time = time.time()

        try:
            request_json = {
                "model": player.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": player.temperature,
                    "top_p": player.top_p,
                },
            }
            if player.system_prompt:
                request_json["system"] = player.system_prompt

            async with session.post(
                f"{player.endpoint}/api/generate",
                json=request_json,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as response:
                response_time = time.time() - start_time

                if response.status == 200:
                    data = await response.json()
                    response_text = data.get("response", "").strip().lower()

                    # Extract action from response - look for any valid action
                    action = game.actions[0]  # Default to first action
                    for a in game.actions:
                        if a.lower() in response_text:
                            action = a
                            break

                    self.metrics.log_success(
                        model=player.model,
                        endpoint=player.endpoint,
                        response_time=response_time,
                        player_id=player.player_id,
                        response=response_text,
                    )
                    return action, response_time, prompt, response_text
                else:
                    error_text = await response.text()
                    self.metrics.log_error(
                        model=player.model,
                        endpoint=player.endpoint,
                        response_time=response_time,
                        player_id=player.player_id,
                        error_type=f"error_{response.status}",
                        error_message=error_text,
                    )
                    return game.actions[0], response_time, prompt, f"[ERROR {response.status}] {error_text}"

        except asyncio.TimeoutError:
            response_time = time.time() - start_time
            self.metrics.log_error(
                model=player.model,
                endpoint=player.endpoint,
                response_time=response_time,
                player_id=player.player_id,
                error_type="timeout",
                error_message="Request timed out",
            )
            return game.actions[0], response_time, prompt, "[TIMEOUT]"

        except Exception as e:
            response_time = time.time() - start_time
            self.metrics.log_error(
                model=player.model,
                endpoint=player.endpoint,
                response_time=response_time,
                player_id=player.player_id,
                error_type="exception",
                error_message=str(e),
            )
            return game.actions[0], response_time, prompt, f"[EXCEPTION] {str(e)}"


class GameRunner:
    """Executes game series with specified configuration."""

    def __init__(self, game: GameDefinition, metrics: Optional[MetricsTracker] = None):
        """Initialize the runner.

        Args:
            game: The game definition.
            metrics: Optional MetricsTracker for logging.
        """
        self.game = game
        self.metrics = metrics or MetricsTracker()
        self.client = OllamaClient(self.metrics)

    async def play_round(
        self,
        session: aiohttp.ClientSession,
        players: List[PlayerConfig],
        history: List[Tuple[str, ...]],
    ) -> Tuple[Tuple[str, ...], Tuple[int, ...], Tuple[float, ...], Tuple[str, ...], Tuple[str, ...]]:
        """Play a single round of the game.

        Args:
            session: The aiohttp session.
            players: List of player configurations.
            history: List of previous round actions.

        Returns:
            Tuple of (actions, payoffs, response_times, prompts, raw_responses).
        """
        # Get actions from all players concurrently
        tasks = [
            self.client.get_action(session, player, self.game, history)
            for player in players
        ]
        results = await asyncio.gather(*tasks)

        # Unpack actions, response times, prompts, and raw responses
        actions = tuple(r[0] for r in results)
        response_times = tuple(r[1] for r in results)
        prompts = tuple(r[2] for r in results)
        raw_responses = tuple(r[3] for r in results)

        # Calculate payoffs
        payoffs = self.game.payoff_matrix.get(actions, tuple(0 for _ in players))

        return actions, payoffs, response_times, prompts, raw_responses

    async def run_series(
        self,
        players: List[PlayerConfig],
        num_rounds: int,
        progress_callback=None,
    ) -> List[dict]:
        """Run a series of game rounds.

        Args:
            players: List of player configurations.
            num_rounds: Number of rounds to play.
            progress_callback: Optional callback for progress updates.

        Returns:
            List of game results including response times.
        """
        results = []
        history: List[Tuple[str, ...]] = []

        # Log expected requests
        self.metrics.log_request_start(num_rounds * len(players))

        async with aiohttp.ClientSession() as session:
            for round_num in range(num_rounds):
                actions, payoffs, response_times = await self.play_round(session, players, history)

                result = {
                    "game_number": round_num + 1,
                    "actions": actions,
                    "payoffs": payoffs,
                    "response_times": response_times,
                }

                # Add player-specific fields
                for i, player in enumerate(players):
                    result[f"player{i+1}_action"] = actions[i]
                    result[f"player{i+1}_payoff"] = payoffs[i]
                    result[f"player{i+1}_model"] = player.model
                    result[f"player{i+1}_endpoint"] = player.endpoint
                    result[f"player{i+1}_response_time"] = response_times[i]

                results.append(result)
                history.append(actions)

                if progress_callback:
                    progress_callback(round_num + 1, num_rounds)

        return results
