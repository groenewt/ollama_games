"""Async client for Ollama API interactions."""

import asyncio
import time
import aiohttp
from functools import lru_cache
from typing import List, Tuple, Optional

from ..core.types import GameDefinition, PlayerConfig
from ..core.config import DEFAULT_TIMEOUT
from ..metrics.tracker import MetricsTracker


@lru_cache(maxsize=256)
def _normalize_action(action: str) -> Tuple[str, ...]:
    """Generate normalized variants of an action for semantic matching (cached).

    Args:
        action: The action string (e.g., "free_ride", "cooperate").

    Returns:
        Tuple of variants to match against (tuple for hashability).
    """
    action_lower = action.lower()
    variants = {action_lower}

    # Generate variants with different separators
    # "free_ride" -> "free-ride", "free ride", "freeride"
    if "_" in action_lower:
        variants.update([
            action_lower.replace("_", "-"),
            action_lower.replace("_", " "),
            action_lower.replace("_", ""),
        ])
    if "-" in action_lower:
        variants.update([
            action_lower.replace("-", "_"),
            action_lower.replace("-", " "),
            action_lower.replace("-", ""),
        ])
    if " " in action_lower:
        variants.update([
            action_lower.replace(" ", "_"),
            action_lower.replace(" ", "-"),
            action_lower.replace(" ", ""),
        ])

    return tuple(variants)


def _match_action(response_text: str, actions: List[str]) -> Tuple[Optional[str], bool]:
    """Match response text against valid actions with semantic variants.

    Args:
        response_text: The LLM response text (already lowercased).
        actions: List of valid actions.

    Returns:
        Tuple of (matched_action, was_parsed). Returns (None, False) if no match.
    """
    for action in actions:
        variants = _normalize_action(action)
        for variant in variants:
            if variant in response_text:
                return action, True
    return None, False


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
        payoff_display: str = "full",
        history_payoffs: Optional[List[Tuple[int, ...]]] = None,
        cumulative_payoffs: Optional[Tuple[int, ...]] = None,
        is_repeated: bool = True,
    ) -> str:
        """Generate contextual prompt for the game.

        Args:
            game: The game definition.
            history: List of previous round actions.
            player_id: The player's ID (1-indexed).
            payoff_display: "full" (all payoffs), "player" (player's only), "none".
            history_payoffs: List of payoff tuples from previous rounds.
            cumulative_payoffs: Running total of payoffs per player.
            is_repeated: True for repeated/sequential games, False for one-off.

        Returns:
            The formatted prompt string.
        """
        game_type = "repeated" if is_repeated else "one-shot"
        sections = [f"You are playing a {game_type} {game.name} game."]

        # Build payoff matrix section
        if payoff_display != "none":
            payoff_lines = []
            if game.num_players == 2:
                # 2-player format: cleaner presentation
                if payoff_display == "full":
                    payoff_lines.append("\nPayoff Matrix (Your payoff, Opponent's payoff):")
                    for action_combo, payoffs in game.payoff_matrix.items():
                        if player_id == 1:
                            your_action, opp_action = action_combo
                            your_payoff, opp_payoff = payoffs
                        else:
                            opp_action, your_action = action_combo
                            opp_payoff, your_payoff = payoffs
                        payoff_lines.append(
                            f"- If you {your_action} and opponent {opp_action}: ({your_payoff}, {opp_payoff})"
                        )
                else:  # player only
                    payoff_lines.append("\nYour Payoffs:")
                    for action_combo, payoffs in game.payoff_matrix.items():
                        if player_id == 1:
                            your_action, opp_action = action_combo
                            your_payoff = payoffs[0]
                        else:
                            opp_action, your_action = action_combo
                            your_payoff = payoffs[1]
                        payoff_lines.append(
                            f"- If you {your_action} and opponent {opp_action}: {your_payoff}"
                        )
            else:
                # N-player format (3+ players)
                if payoff_display == "full":
                    payoff_lines.append(f"\nPayoff Matrix ({game.num_players} players):")
                    payoff_lines.append(f"Format: (P1 payoff, P2 payoff, ..., P{game.num_players} payoff)")
                    payoff_lines.append(f"You are Player {player_id}.")
                    payoff_lines.append("")
                    for action_combo, payoffs in game.payoff_matrix.items():
                        actions_str = ", ".join([f"P{i+1}={a}" for i, a in enumerate(action_combo)])
                        payoffs_str = ", ".join(str(p) for p in payoffs)
                        your_payoff = payoffs[player_id - 1]
                        payoff_lines.append(f"- {actions_str}")
                        payoff_lines.append(f"  Payoffs: ({payoffs_str}) | Your payoff: {your_payoff}")
                else:  # player only
                    payoff_lines.append(f"\nYour Payoffs (as Player {player_id}):")
                    for action_combo, payoffs in game.payoff_matrix.items():
                        actions_str = ", ".join([f"P{i+1}={a}" for i, a in enumerate(action_combo)])
                        your_payoff = payoffs[player_id - 1]
                        payoff_lines.append(f"- {actions_str}: {your_payoff}")
            sections.append("\n".join(payoff_lines))

        # Format history with payoffs (only for repeated games)
        if is_repeated:
            if history:
                history_lines = ["\nPrevious rounds:"]
                for i, actions in enumerate(history):
                    if game.num_players == 2:
                        line = f"Round {i+1}: Player 1 chose {actions[0]}, Player 2 chose {actions[1]}"
                        # Add payoffs if available
                        if history_payoffs and i < len(history_payoffs):
                            line += f" → Payoffs: {history_payoffs[i]}"
                    else:
                        line = f"Round {i+1}: " + ", ".join([
                            f"Player {j+1} chose {action}"
                            for j, action in enumerate(actions)
                        ])
                        if history_payoffs and i < len(history_payoffs):
                            line += f" → Payoffs: {history_payoffs[i]}"
                    history_lines.append(line)

                # Add cumulative payoffs (works for any number of players)
                if cumulative_payoffs:
                    cum_lines = ["Current cumulative payoffs:"]
                    for p_idx in range(game.num_players):
                        p_num = p_idx + 1
                        if p_num == player_id:
                            cum_lines.append(f"  You (P{p_num}): {cumulative_payoffs[p_idx]}")
                        else:
                            cum_lines.append(f"  Player {p_num}: {cumulative_payoffs[p_idx]}")
                    history_lines.extend(cum_lines)

                sections.append("\n".join(history_lines))
            else:
                sections.append("\nPrevious rounds:\nNo previous rounds.")

        actions_str = ", ".join([f'"{a}"' for a in game.actions])
        sections.append(
            f"\nYou are Player {player_id}. Your available choices are: {actions_str}.\n"
            f"What is your choice for this round? Respond with only one word from the available choices."
        )

        return "\n".join(sections)

    async def get_action(
        self,
        session: aiohttp.ClientSession,
        player: PlayerConfig,
        game: GameDefinition,
        history: List[Tuple[str, ...]],
        timeout: float = DEFAULT_TIMEOUT,
        payoff_display: str = "full",
        history_payoffs: Optional[List[Tuple[int, ...]]] = None,
        cumulative_payoffs: Optional[Tuple[int, ...]] = None,
        is_repeated: bool = True,
    ) -> Tuple[str, float, str, str, bool]:
        """Query model for action decision.

        Args:
            session: The aiohttp session.
            player: Player configuration with model and endpoint.
            game: The game definition.
            history: List of previous round actions.
            timeout: Request timeout in seconds.
            payoff_display: "full", "player", or "none" for payoff info in prompt.
            history_payoffs: List of payoff tuples from previous rounds.
            cumulative_payoffs: Running total of payoffs per player.
            is_repeated: True for repeated/sequential games, False for one-off.

        Returns:
            Tuple of (action, response_time, prompt, raw_response, was_parsed).
            was_parsed is True if action was found in response, False if defaulted.
        """
        prompt = self._build_prompt(
            game, history, player.player_id,
            payoff_display=payoff_display,
            history_payoffs=history_payoffs,
            cumulative_payoffs=cumulative_payoffs,
            is_repeated=is_repeated,
        )
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

                    # Extract action from response with semantic matching
                    matched, was_parsed = _match_action(response_text, game.actions)
                    action = matched if matched else game.actions[0]

                    self.metrics.log_success(
                        model=player.model,
                        endpoint=player.endpoint,
                        response_time=response_time,
                        player_id=player.player_id,
                        response=response_text,
                    )
                    return action, response_time, prompt, response_text, was_parsed
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
                    return game.actions[0], response_time, prompt, f"[ERROR {response.status}] {error_text}", False

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
            return game.actions[0], response_time, prompt, "[TIMEOUT]", False

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
            return game.actions[0], response_time, prompt, f"[EXCEPTION] {str(e)}", False


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
        payoff_display: str = "full",
        history_payoffs: Optional[List[Tuple[int, ...]]] = None,
        cumulative_payoffs: Optional[Tuple[int, ...]] = None,
        is_repeated: bool = True,
    ) -> Tuple[Tuple[str, ...], Tuple[int, ...], Tuple[float, ...], Tuple[str, ...], Tuple[str, ...], Tuple[bool, ...]]:
        """Play a single round of the game.

        Args:
            session: The aiohttp session.
            players: List of player configurations.
            history: List of previous round actions.
            payoff_display: "full", "player", or "none" for payoff info in prompt.
            history_payoffs: List of payoff tuples from previous rounds.
            cumulative_payoffs: Running total of payoffs per player.
            is_repeated: True for repeated/sequential games, False for one-off.

        Returns:
            Tuple of (actions, payoffs, response_times, prompts, raw_responses, was_parsed).
        """
        # Get actions from all players concurrently
        tasks = [
            self.client.get_action(
                session, player, self.game, history,
                payoff_display=payoff_display,
                history_payoffs=history_payoffs,
                cumulative_payoffs=cumulative_payoffs,
                is_repeated=is_repeated,
            )
            for player in players
        ]
        results = await asyncio.gather(*tasks)

        # Unpack actions, response times, prompts, raw responses, and parsed flags
        actions = tuple(r[0] for r in results)
        response_times = tuple(r[1] for r in results)
        prompts = tuple(r[2] for r in results)
        raw_responses = tuple(r[3] for r in results)
        was_parsed = tuple(r[4] for r in results)

        # Calculate payoffs
        payoffs = self.game.payoff_matrix.get(actions, tuple(0 for _ in players))

        return actions, payoffs, response_times, prompts, raw_responses, was_parsed

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
