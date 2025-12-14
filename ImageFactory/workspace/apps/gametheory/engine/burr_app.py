"""Burr-based game engine for complex game state management.

This module provides a Burr state machine implementation for running
game theory experiments with full state persistence and observability.
"""

from typing import List, Dict, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass
import asyncio
import aiohttp
import json
import os
import re
import time

from burr.core import ApplicationBuilder, State, action, default, expr, Application
from burr.core.action import Action

from ..core.types import PlayerConfig, LLMResponse
from .discrete_space import DiscreteActionSpace
from .permutation_space import PermutationSpace, SumoCoachSpace


@dataclass
class ActionResult:
    """Result of getting an action from an LLM, with token metrics."""
    action: Any
    response_time: float
    prompt: str
    raw_response: str
    was_parsed: bool
    was_normalized: bool
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    # Decision reasoning capture (experimental)
    reasoning_trace: Optional[str] = None
    alternatives_considered: Optional[List[str]] = None
    confidence_score: Optional[float] = None


@dataclass
class AllocationSpace:
    """Action space for resource allocation games like Colonel Blotto."""
    num_fields: int  # Number of battlefields/targets
    budget: float    # Total resources to allocate

    def validate(self, allocation: List[float]) -> Tuple[bool, str]:
        """Validate an allocation against constraints."""
        if len(allocation) != self.num_fields:
            return False, f"Expected {self.num_fields} values, got {len(allocation)}"
        if any(a < 0 for a in allocation):
            return False, "Allocations cannot be negative"
        if abs(sum(allocation) - self.budget) > 0.01:
            return False, f"Allocations must sum to {self.budget}, got {sum(allocation)}"
        return True, ""

    def prompt_instructions(self) -> str:
        """Generate prompt instructions for this action space."""
        # Generate example that sums to budget
        example_allocs = [0] * self.num_fields
        example_allocs[0] = self.budget
        example_json = ', '.join(str(int(a)) for a in example_allocs)

        return f"""You have {self.budget} resources to allocate across {self.num_fields} targets.

CRITICAL FORMAT REQUIREMENTS:
- Respond with ONLY a JSON object
- "allocations" must be a simple array of {self.num_fields} NUMBERS (not objects)
- Numbers must sum to EXACTLY {int(self.budget)}

EXAMPLE (concentrating all on target 1):
{{"allocations": [{example_json}], "reasoning": "concentrate forces"}}

YOUR RESPONSE MUST FOLLOW THIS EXACT FORMAT."""


@dataclass
class BurrGameDefinition:
    """Game definition for Burr-based games with dynamic payoff functions.

    Supports allocation games (Colonel Blotto), discrete games (PD, Chicken),
    and permutation games (Tennis Coach, Sumo Coach).
    """
    id: str
    name: str
    description: str
    action_space: Union[AllocationSpace, DiscreteActionSpace, PermutationSpace, SumoCoachSpace]
    payoff_fn: Callable[[Tuple[Any, ...]], Tuple[float, ...]]
    num_players: int = 2

    @property
    def is_discrete(self) -> bool:
        """Check if this game uses discrete actions."""
        return isinstance(self.action_space, DiscreteActionSpace)

    @property
    def is_permutation(self) -> bool:
        """Check if this game uses permutation actions."""
        return isinstance(self.action_space, (PermutationSpace, SumoCoachSpace))

    @property
    def is_allocation(self) -> bool:
        """Check if this game uses allocation actions (has budget)."""
        return isinstance(self.action_space, AllocationSpace)


# --- Burr Actions ---

@action(reads=["round_num", "max_rounds", "history", "results"], writes=["round_num", "history", "results"])
def play_game_round(
    state: State,
    actions: List[Any],
    payoffs: Tuple[float, ...],
    player_configs: List[PlayerConfig],
    response_times: List[float],
    prompts: List[str],
    raw_responses: List[str],
    was_parsed: List[bool],
    was_normalized: List[bool],
    is_discrete: bool,
) -> State:
    """Execute one round of ANY game type (discrete, allocation, or permutation).

    This action is called externally with the player actions already collected.
    """
    # Convert actions to appropriate tuple format
    if is_discrete:
        actions_tuple = tuple(actions)  # strings stay as strings
    else:
        # Allocation or permutation: convert lists to tuples
        actions_tuple = tuple(tuple(a) if isinstance(a, (list, tuple)) else a for a in actions)

    round_num = state["round_num"] + 1

    # Build result in same format as GameRunner for compatibility
    result = {
        "game_number": round_num,
        "actions": actions_tuple,
        "payoffs": payoffs,
        "response_times": tuple(response_times),
        "runtime_mode": state.get("runtime_mode", "repeated"),
        "uses_custom_payoffs": False,
    }

    # Add player-specific fields
    for i, player in enumerate(player_configs):
        p_num = i + 1
        action = actions[i]
        # Format action for display
        if is_discrete:
            action_display = str(action)
        elif hasattr(action, '__iter__') and not isinstance(action, str):
            action_display = f"[{', '.join(f'{a:.0f}' for a in action)}]"
        else:
            action_display = str(action)

        result[f"player{p_num}_action"] = action_display
        if not is_discrete and hasattr(action, '__iter__'):
            result[f"player{p_num}_allocation"] = str(list(action))
        result[f"player{p_num}_payoff"] = payoffs[i]
        result[f"player{p_num}_model"] = player.model
        result[f"player{p_num}_endpoint"] = player.endpoint
        result[f"player{p_num}_response_time"] = response_times[i]
        result[f"player{p_num}_was_parsed"] = was_parsed[i]
        result[f"player{p_num}_temperature"] = player.temperature
        result[f"player{p_num}_top_p"] = player.top_p

    return state.update(
        round_num=round_num,
        history=state["history"] + [actions_tuple],
        results=state["results"] + [result],
    )


# Keep old action for backwards compatibility
@action(reads=["round_num", "max_rounds", "history", "results"], writes=["round_num", "history", "results"])
def play_allocation_round(
    state: State,
    player_allocations: List[List[float]],
    payoff_fn: Callable,
    player_configs: List[PlayerConfig],
    response_times: List[float],
) -> State:
    """Execute one round of an allocation game (DEPRECATED - use play_game_round)."""
    allocations_tuple = tuple(tuple(a) for a in player_allocations)
    payoffs = payoff_fn(allocations_tuple)

    round_num = state["round_num"] + 1

    result = {
        "game_number": round_num,
        "actions": allocations_tuple,
        "payoffs": payoffs,
        "response_times": tuple(response_times),
    }

    for i, player in enumerate(player_configs):
        result[f"player{i+1}_action"] = player_allocations[i]
        result[f"player{i+1}_payoff"] = payoffs[i]
        result[f"player{i+1}_model"] = player.model
        result[f"player{i+1}_endpoint"] = player.endpoint
        result[f"player{i+1}_response_time"] = response_times[i]

    return state.update(
        round_num=round_num,
        history=state["history"] + [allocations_tuple],
        results=state["results"] + [result],
    )


@action(reads=["round_num", "max_rounds"], writes=[])
def check_game_complete(state: State) -> State:
    """Check if the game series is complete."""
    return state


@action(reads=["results"], writes=[])
def game_end(state: State) -> State:
    """Terminal action - game series complete."""
    return state


def validate_tracking_setup(storage_dir: str) -> Tuple[bool, str]:
    """Validate Burr tracking can work before starting games.

    Args:
        storage_dir: Path to storage directory (can include ~)

    Returns:
        (is_valid, error_message) - error_message is empty if valid
    """
    # Expand path
    expanded = os.path.expanduser(storage_dir)

    # Check/create directory
    if not os.path.exists(expanded):
        try:
            os.makedirs(expanded, exist_ok=True)
        except OSError as e:
            return False, f"Cannot create storage dir: {e}"

    # Check writable
    if not os.access(expanded, os.W_OK):
        return False, f"Storage dir not writable: {expanded}"

    return True, ""


def build_game_app(
    game: BurrGameDefinition,
    num_rounds: int,
    runtime_mode: str = "repeated",
    app_id: Optional[str] = None,
    enable_tracking: bool = True,
) -> Application:
    """Build a Burr application for ANY game type (discrete, allocation, permutation).

    Args:
        game: The game definition
        num_rounds: Number of rounds to play
        runtime_mode: "one_off", "repeated", "sequential", "multi_player"
        app_id: Optional application ID for persistence
        enable_tracking: Whether to enable Burr's tracking UI

    Returns:
        A configured Burr Application
    """
    builder = (
        ApplicationBuilder()
        .with_actions(
            play_round=play_game_round,
            check_complete=check_game_complete,
            end=game_end,
        )
        .with_transitions(
            ("play_round", "check_complete", default),
            ("check_complete", "play_round", expr("round_num < max_rounds")),
            ("check_complete", "end", default),
        )
        .with_state(
            game_id=game.id,
            game_name=game.name,
            is_discrete=game.is_discrete,
            runtime_mode=runtime_mode,
            history=[],
            round_num=0,
            max_rounds=num_rounds,
            results=[],
        )
        .with_entrypoint("play_round")
    )

    if app_id:
        builder = builder.with_identifiers(app_id=app_id)

    if enable_tracking:
        storage_dir = os.path.expanduser(
            os.environ.get("BURR_STORAGE_DIR", "~/.burr")
        )
        builder = builder.with_tracker(
            "local",
            project=os.environ.get("BURR_PROJECT", "gametheory"),
            params={"storage_dir": storage_dir}
        )

    return builder.build()


# Backwards compatibility alias
def build_allocation_game_app(
    game: BurrGameDefinition,
    num_rounds: int,
    app_id: Optional[str] = None,
    enable_tracking: bool = True,
) -> Application:
    """DEPRECATED: Use build_game_app instead."""
    return build_game_app(game, num_rounds, "repeated", app_id, enable_tracking)


class BurrGameRunner:
    """Burr-based game runner for allocation games.

    Drop-in replacement for GameRunner that uses Burr for state management.
    Currently supports allocation games (like Colonel Blotto).
    """

    def __init__(
        self,
        game: BurrGameDefinition,
        enable_tracking: bool = True,
    ):
        self.game = game
        self.enable_tracking = enable_tracking

    async def _get_action(
        self,
        session: aiohttp.ClientSession,
        player: PlayerConfig,
        history: List[Tuple[Any, ...]],
        history_payoffs: Optional[List[Tuple[float, ...]]] = None,
        cumulative_payoffs: Optional[Tuple[float, ...]] = None,
        is_repeated: bool = True,
    ) -> ActionResult:
        """Get action from a player's model.

        Works for both allocation games (returns List[float]) and
        discrete games (returns str).

        Returns:
            ActionResult containing action, timing, prompt/response, parsing status,
            and token counts (if available from Ollama).
        """
        # Build prompt
        prompt = self._build_prompt(
            player, history, history_payoffs, cumulative_payoffs, is_repeated
        )

        # Query model
        start_time = time.time()
        raw_response = ""
        was_parsed = False
        was_normalized = False
        prompt_tokens: Optional[int] = None
        completion_tokens: Optional[int] = None

        try:
            async with session.post(
                f"{player.endpoint}/api/generate",
                json={
                    "model": player.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": player.temperature,
                        "top_p": player.top_p,
                        "top_k": player.top_k,
                        "repeat_penalty": player.repeat_penalty,
                    }
                },
                timeout=aiohttp.ClientTimeout(total=60),
            ) as response:
                response_time = time.time() - start_time
                if response.status == 200:
                    data = await response.json()
                    raw_response = data.get("response", "")

                    # Capture token counts from Ollama response
                    prompt_tokens = data.get("prompt_eval_count")
                    completion_tokens = data.get("eval_count")

                    if self.game.is_discrete:
                        action, was_parsed = self._parse_discrete_action(raw_response)
                        # Extract reasoning (experimental)
                        reasoning = self._extract_reasoning(raw_response)
                        alternatives = self._extract_alternatives(raw_response)
                        confidence = self._estimate_confidence(
                            raw_response, was_parsed, response_time
                        )
                        return ActionResult(
                            action=action,
                            response_time=response_time,
                            prompt=prompt,
                            raw_response=raw_response,
                            was_parsed=was_parsed,
                            was_normalized=False,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            reasoning_trace=reasoning,
                            alternatives_considered=alternatives,
                            confidence_score=confidence,
                        )
                    elif self.game.is_permutation:
                        action, was_parsed = self._parse_permutation(raw_response)
                        # Extract reasoning (experimental)
                        reasoning = self._extract_reasoning(raw_response)
                        alternatives = self._extract_alternatives(raw_response)
                        confidence = self._estimate_confidence(
                            raw_response, was_parsed, response_time
                        )
                        return ActionResult(
                            action=action,
                            response_time=response_time,
                            prompt=prompt,
                            raw_response=raw_response,
                            was_parsed=was_parsed,
                            was_normalized=False,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            reasoning_trace=reasoning,
                            alternatives_considered=alternatives,
                            confidence_score=confidence,
                        )
                    else:
                        allocation, was_parsed, was_normalized = self._parse_allocation(raw_response)
                        # Extract reasoning (experimental)
                        reasoning = self._extract_reasoning(raw_response)
                        alternatives = self._extract_alternatives(raw_response)
                        confidence = self._estimate_confidence(
                            raw_response, was_parsed, response_time
                        )
                        return ActionResult(
                            action=allocation,
                            response_time=response_time,
                            prompt=prompt,
                            raw_response=raw_response,
                            was_parsed=was_parsed,
                            was_normalized=was_normalized,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            reasoning_trace=reasoning,
                            alternatives_considered=alternatives,
                            confidence_score=confidence,
                        )
                else:
                    # Return default action on error
                    return ActionResult(
                        action=self._default_action(),
                        response_time=time.time() - start_time,
                        prompt=prompt,
                        raw_response=raw_response,
                        was_parsed=False,
                        was_normalized=False,
                    )
        except Exception as e:
            return ActionResult(
                action=self._default_action(),
                response_time=time.time() - start_time,
                prompt=prompt,
                raw_response=str(e),
                was_parsed=False,
                was_normalized=False,
            )

    # Keep old name for backwards compatibility
    async def _get_allocation(
        self,
        session: aiohttp.ClientSession,
        player: PlayerConfig,
        history: List[Tuple[Any, ...]],
        history_payoffs: Optional[List[Tuple[float, ...]]] = None,
        cumulative_payoffs: Optional[Tuple[float, ...]] = None,
        is_repeated: bool = True,
    ) -> ActionResult:
        """Alias for _get_action for backwards compatibility."""
        return await self._get_action(
            session, player, history, history_payoffs, cumulative_payoffs, is_repeated
        )

    def _build_prompt(
        self,
        player: PlayerConfig,
        history: List[Tuple[Any, ...]],
        history_payoffs: Optional[List[Tuple[float, ...]]] = None,
        cumulative_payoffs: Optional[Tuple[float, ...]] = None,
        is_repeated: bool = True,
    ) -> str:
        """Build the prompt for the model."""
        sections = []

        # System prompt with prominent framing to encourage model to follow it
        if player.system_prompt:
            sections.append("=== YOUR STRATEGIC GUIDANCE ===")
            sections.append(player.system_prompt)
            sections.append("=== APPLY THIS GUIDANCE IN YOUR STRATEGY ===")
            sections.append("")

        # Game description - strip default hints if custom provided
        game_desc = self.game.description
        if player.strategy_hints:
            # Remove "Strategic considerations:" section from description
            if "Strategic considerations:" in game_desc:
                game_desc = game_desc.split("Strategic considerations:")[0].strip()

        sections.append(f"Game: {self.game.name}")
        sections.append(game_desc)

        # Add custom strategy hints if provided
        if player.strategy_hints:
            sections.append("")
            sections.append("Strategic considerations:")
            sections.append(player.strategy_hints)

        sections.append(self.game.action_space.prompt_instructions())

        # History (only for repeated games)
        if is_repeated and history:
            sections.append("\nPrevious rounds:")
            # Show last 5 rounds with payoffs if available
            start_idx = max(0, len(history) - 5)
            for i, round_allocs in enumerate(history[start_idx:], start_idx + 1):
                round_str = f"  Round {i}: Allocations={round_allocs}"
                if history_payoffs and i <= len(history_payoffs):
                    round_str += f", Payoffs={history_payoffs[i-1]}"
                sections.append(round_str)

            # Show cumulative payoffs if available
            if cumulative_payoffs:
                sections.append(f"  Cumulative payoffs so far: {cumulative_payoffs}")

        # Reminder to apply strategic guidance
        if player.system_prompt:
            sections.append("")
            sections.append("IMPORTANT: Apply the strategic guidance above when choosing your allocation.")

        return "\n".join(sections)

    def _parse_allocation(self, response: str) -> Tuple[List[float], bool, bool]:
        """Parse allocation from model response.

        Returns:
            Tuple of (allocation, was_parsed, was_normalized) where:
            - was_parsed: True if we extracted numbers from the response
            - was_normalized: True if allocations were adjusted to sum to budget
        """
        import json
        import re

        budget = self.game.action_space.budget

        # Try JSON first - use a more robust pattern for multi-line JSON
        try:
            # Find JSON object - handle multi-line by using DOTALL
            match = re.search(r'\{[^{}]*"allocations"\s*:\s*\[[^\]]+\][^{}]*\}', response, re.DOTALL)
            if match:
                data = json.loads(match.group())
                if "allocations" in data:
                    raw_allocs = data["allocations"]
                    # Handle case where allocations might be dicts or other non-numeric types
                    allocs = []
                    for x in raw_allocs:
                        if isinstance(x, (int, float)):
                            allocs.append(float(x))
                        elif isinstance(x, str) and x.replace('.', '').isdigit():
                            allocs.append(float(x))
                        elif isinstance(x, dict):
                            # Try common keys for nested allocations
                            for key in ['value', 'amount', 'allocation', 'troops']:
                                if key in x and isinstance(x[key], (int, float)):
                                    allocs.append(float(x[key]))
                                    break
                            else:
                                raise ValueError(f"Cannot extract number from dict: {x}")
                        else:
                            raise ValueError(f"Invalid allocation type: {type(x)}")

                    if len(allocs) == self.game.action_space.num_fields:
                        valid, _ = self.game.action_space.validate(allocs)
                        if valid:
                            return allocs, True, False  # Valid, parsed, not normalized

                        # JSON found but invalid sum - normalize it
                        total = sum(allocs)
                        if total > 0:
                            scale = budget / total
                            allocs = [a * scale for a in allocs]
                            # was_parsed=True only if adjustment was minor (<5%)
                            was_parsed = abs(1.0 - scale) < 0.05
                            return allocs, was_parsed, True  # Normalized
        except (json.JSONDecodeError, ValueError, KeyError, TypeError):
            pass

        # Fallback: extract numbers from response
        numbers = re.findall(r'\d+\.?\d*', response)
        if len(numbers) >= self.game.action_space.num_fields:
            allocs = [float(n) for n in numbers[:self.game.action_space.num_fields]]
            total = sum(allocs)
            if total > 0:
                # Check if normalization is needed
                needs_normalize = abs(total - budget) > 0.01
                if needs_normalize:
                    scale = budget / total
                    allocs = [a * scale for a in allocs]
                    # was_parsed=True only if adjustment was minor (<5%)
                    was_parsed = abs(1.0 - scale) < 0.05
                    return allocs, was_parsed, True  # Normalized
                return allocs, True, False  # Valid without normalization

        return self._default_allocation(), False, False  # Default allocation

    def _parse_discrete_action(self, response: str) -> Tuple[str, bool]:
        """Parse discrete action from model response.

        Args:
            response: The raw LLM response text.

        Returns:
            Tuple of (action, was_parsed) where:
            - action: The matched action or default if not parsed
            - was_parsed: True if action was found in response
        """
        # Try JSON first
        try:
            match = re.search(r'\{[^{}]*"action"\s*:\s*"([^"]+)"[^{}]*\}', response, re.DOTALL)
            if match:
                action_str = match.group(1)
                # Validate against action space
                matched = self.game.action_space._normalize_action(action_str)
                if matched:
                    return matched, True
        except Exception:
            pass

        # Fallback: use action space's match_action
        matched, was_parsed = self.game.action_space.match_action(response)
        if matched:
            return matched, was_parsed

        # Default to first action
        return self.game.action_space.actions[0], False

    def _parse_permutation(self, response: str) -> Tuple[List[int], bool]:
        """Parse permutation assignment from model response.

        Args:
            response: The raw LLM response text.

        Returns:
            Tuple of (assignment, was_parsed) where:
            - assignment: List of integers (skill levels or indices)
            - was_parsed: True if valid permutation was extracted
        """
        # Try JSON first - look for "assignment" key
        try:
            match = re.search(
                r'\{[^{}]*"assignment"\s*:\s*\[([^\]]+)\][^{}]*\}',
                response,
                re.DOTALL,
            )
            if match:
                # Parse the array content
                array_content = match.group(1)
                # Extract integers from the array
                values = [
                    int(x.strip())
                    for x in array_content.split(",")
                    if x.strip().lstrip("-").isdigit()
                ]

                if len(values) == self.game.action_space.num_positions:
                    valid, _ = self.game.action_space.validate(values)
                    if valid:
                        return values, True
        except (json.JSONDecodeError, ValueError, AttributeError):
            pass

        # Fallback: extract integers from brackets like [4, 1, 3, 2] or (4, 1, 3, 2)
        bracket_match = re.search(r"[\[\(]([^\]\)]+)[\]\)]", response)
        if bracket_match:
            try:
                values = [
                    int(x.strip())
                    for x in bracket_match.group(1).split(",")
                    if x.strip().lstrip("-").isdigit()
                ]
                if len(values) == self.game.action_space.num_positions:
                    valid, _ = self.game.action_space.validate(values)
                    if valid:
                        return values, True
            except ValueError:
                pass

        # Last resort: extract any sequence of integers from text
        numbers = re.findall(r"\b\d+\b", response)
        if len(numbers) >= self.game.action_space.num_positions:
            values = [int(n) for n in numbers[: self.game.action_space.num_positions]]
            valid, _ = self.game.action_space.validate(values)
            if valid:
                return values, True

        # Return default action
        return self._default_permutation(), False

    def _default_permutation(self) -> List[int]:
        """Return default permutation action."""
        if hasattr(self.game.action_space, "default_action"):
            return self.game.action_space.default_action()
        # Fallback: sequential indices
        return list(range(self.game.action_space.num_positions))

    def _default_action(self) -> Any:
        """Return default action based on action space type."""
        if self.game.is_discrete:
            return self.game.action_space.actions[0]
        elif self.game.is_permutation:
            return self._default_permutation()
        else:
            return self._default_allocation()

    def _default_allocation(self) -> List[float]:
        """Return uniform allocation as default."""
        per_field = self.game.action_space.budget / self.game.action_space.num_fields
        return [per_field] * self.game.action_space.num_fields

    def _extract_reasoning(self, response: str) -> Optional[str]:
        """Extract reasoning/explanation from LLM response.

        Looks for common patterns indicating decision reasoning:
        - JSON "reasoning" field
        - "because", "therefore", "since" explanations
        - Numbered lists of considerations

        Args:
            response: Raw LLM response text.

        Returns:
            Extracted reasoning string or None if not found.
        """
        # Try JSON reasoning field first
        try:
            match = re.search(
                r'"reasoning"\s*:\s*"([^"]*)"',
                response,
                re.IGNORECASE | re.DOTALL
            )
            if match and match.group(1).strip():
                return match.group(1).strip()

            # Also try with escaped quotes
            match = re.search(
                r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"',
                response,
                re.IGNORECASE | re.DOTALL
            )
            if match and match.group(1).strip():
                return match.group(1).strip().replace('\\"', '"').replace('\\n', '\n')
        except Exception:
            pass

        # Look for reasoning keywords with context
        reasoning_patterns = [
            r'(?:because|since|therefore|thus|given that)[:\s]+([^.!?\n]+[.!?]?)',
            r'(?:my reasoning|strategy|approach)[:\s]+([^.!?\n]+[.!?]?)',
            r'(?:I (?:choose|selected|decided|picked|went with))[^.]*(?:because|since)[:\s]+([^.!?\n]+[.!?]?)',
        ]

        for pattern in reasoning_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match and len(match.group(1).strip()) > 10:
                return match.group(1).strip()

        # Look for numbered reasoning (1. First... 2. Second...)
        numbered_match = re.findall(r'\d+\.\s+([^.!?\n]+[.!?]?)', response)
        if len(numbered_match) >= 2:
            return '; '.join(numbered_match[:3])

        return None

    def _extract_alternatives(self, response: str) -> Optional[List[str]]:
        """Extract alternative options considered from LLM response.

        Looks for patterns like "I considered X but chose Y" or
        lists of options evaluated.

        Args:
            response: Raw LLM response text.

        Returns:
            List of alternatives considered, or None.
        """
        alternatives = []

        # Pattern: "I considered X" or "alternative: X"
        patterns = [
            r'(?:I )?(?:considered|evaluated|thought about|weighed)\s+([^.,]+)',
            r'(?:alternative|option)[:\s]+([^.,]+)',
            r'(?:could have|might have)\s+(?:chosen|picked|gone with)\s+([^.,]+)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            alternatives.extend([m.strip() for m in matches if len(m.strip()) > 3])

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for alt in alternatives:
            if alt.lower() not in seen:
                seen.add(alt.lower())
                unique.append(alt)

        return unique[:5] if unique else None  # Limit to 5 alternatives

    def _estimate_confidence(
        self,
        raw_response: str,
        was_parsed: bool,
        response_time: float,
    ) -> float:
        """Estimate confidence of the LLM's decision.

        Uses heuristics based on:
        - Whether the response was successfully parsed
        - Presence of hedging language
        - Response length and structure
        - Response time (very fast or slow may indicate issues)

        Args:
            raw_response: The raw LLM response text.
            was_parsed: Whether the action was successfully parsed.
            response_time: Time taken to generate response.

        Returns:
            Confidence score from 0.0 to 1.0.
        """
        if not was_parsed:
            return 0.3  # Low confidence if parsing failed

        score = 0.7  # Base confidence for parsed response

        # Hedging language reduces confidence
        hedging_patterns = [
            r'\b(?:maybe|perhaps|possibly|might|could be|not sure|uncertain)\b',
            r'\b(?:i think|i guess|i suppose|probably)\b',
            r'\b(?:hard to say|difficult to decide|unclear)\b',
        ]

        hedging_count = 0
        for pattern in hedging_patterns:
            hedging_count += len(re.findall(pattern, raw_response, re.IGNORECASE))

        score -= min(0.2, hedging_count * 0.05)

        # Confident language increases score
        confident_patterns = [
            r'\b(?:definitely|certainly|clearly|obviously|best choice)\b',
            r'\b(?:optimal|strategic|calculated)\b',
        ]

        confident_count = 0
        for pattern in confident_patterns:
            confident_count += len(re.findall(pattern, raw_response, re.IGNORECASE))

        score += min(0.15, confident_count * 0.05)

        # Reasoning presence increases confidence
        if self._extract_reasoning(raw_response):
            score += 0.1

        # Response time heuristics (normalized)
        if response_time < 0.5:
            score -= 0.05  # Very fast might be cached/default
        elif response_time > 30:
            score -= 0.05  # Very slow might indicate confusion

        # JSON structure increases confidence
        if re.search(r'\{[^}]+\}', raw_response):
            score += 0.05

        return max(0.0, min(1.0, score))

    async def play_round(
        self,
        session: aiohttp.ClientSession,
        players: List[PlayerConfig],
        history: List[Tuple[Any, ...]],
        payoff_display: str = "full",
        history_payoffs: Optional[List[Tuple[float, ...]]] = None,
        cumulative_payoffs: Optional[Tuple[float, ...]] = None,
        is_repeated: bool = True,
    ) -> Tuple[
        Tuple[Any, ...],              # actions (allocations or discrete)
        Tuple[float, ...],            # payoffs
        Tuple[float, ...],            # response_times
        Tuple[str, ...],              # prompts
        Tuple[str, ...],              # raw_responses
        Tuple[bool, ...],             # was_parsed
        Tuple[bool, ...],             # was_normalized
        Tuple[Optional[int], ...],    # prompt_tokens
        Tuple[Optional[int], ...],    # completion_tokens
        Tuple[Optional[str], ...],    # reasoning_traces (experimental)
        Tuple[Optional[List[str]], ...],  # alternatives_considered (experimental)
        Tuple[Optional[float], ...],  # confidence_scores (experimental)
    ]:
        """Play a single round of a Burr game.

        Compatible with GameRunner.play_round() interface. Supports both
        allocation games and discrete action games.

        Args:
            session: aiohttp session for API calls
            players: List of player configurations
            history: Previous round actions (for repeated games)
            payoff_display: How to display payoffs (unused for Burr games)
            history_payoffs: Payoffs from previous rounds
            cumulative_payoffs: Running total of payoffs per player
            is_repeated: Whether this is a repeated game (affects history in prompt)

        Returns:
            Tuple of (actions, payoffs, response_times, prompts, raw_responses,
                     was_parsed, was_normalized, prompt_tokens, completion_tokens)
        """
        # Gather actions from all players in parallel
        tasks = [
            self._get_action(
                session, player, history,
                history_payoffs, cumulative_payoffs, is_repeated
            )
            for player in players
        ]
        action_results: List[ActionResult] = await asyncio.gather(*tasks)

        # Unpack ActionResult objects
        actions = tuple(r.action for r in action_results)
        response_times = tuple(r.response_time for r in action_results)
        prompts = tuple(r.prompt for r in action_results)
        raw_responses = tuple(r.raw_response for r in action_results)
        was_parsed = tuple(r.was_parsed for r in action_results)
        was_normalized = tuple(r.was_normalized for r in action_results)
        prompt_tokens = tuple(r.prompt_tokens for r in action_results)
        completion_tokens = tuple(r.completion_tokens for r in action_results)
        # Experimental reasoning capture fields
        reasoning_traces = tuple(r.reasoning_trace for r in action_results)
        alternatives = tuple(r.alternatives_considered for r in action_results)
        confidence_scores = tuple(r.confidence_score for r in action_results)

        # Calculate payoffs using the game's payoff function
        if self.game.is_discrete:
            # Discrete actions are strings, pass as tuple
            actions_for_payoff = actions
        elif self.game.is_permutation:
            # Permutation actions are lists of ints, convert to tuple of tuples
            actions_for_payoff = tuple(tuple(a) for a in actions)
        else:
            # Allocation actions are lists of floats, convert to tuple of tuples
            actions_for_payoff = tuple(tuple(a) for a in actions)

        payoffs = self.game.payoff_fn(actions_for_payoff)

        return (
            actions, payoffs, response_times, prompts, raw_responses,
            was_parsed, was_normalized, prompt_tokens, completion_tokens,
            reasoning_traces, alternatives, confidence_scores
        )

    async def run_series(
        self,
        players: List[PlayerConfig],
        num_rounds: int,
        session: Optional[aiohttp.ClientSession] = None,
        payoff_display: str = "full",
        runtime_mode: str = "repeated",
        callbacks: Optional[Any] = None,
        progress_callback=None,
    ) -> List[dict]:
        """Run a series of game rounds using Burr for state management.

        Uses Burr's state machine for persistence and tracking while
        keeping async LLM calls external to the action system.

        Args:
            players: List of player configurations
            num_rounds: Number of rounds to play
            session: Optional aiohttp session (creates one if not provided)
            payoff_display: "full", "player", or "none"
            runtime_mode: "one_off", "repeated", "sequential", "multi_player"
            callbacks: Optional GameRunnerCallbacks for UI updates
            progress_callback: Legacy progress callback (deprecated)

        Returns:
            List of result dicts compatible with GameRunnerService
        """
        is_repeated = runtime_mode in ("repeated", "sequential")

        # Pre-flight validation for Burr tracking
        tracking_enabled = self.enable_tracking
        if tracking_enabled:
            storage_dir = os.environ.get("BURR_STORAGE_DIR", "~/.burr")
            is_valid, error = validate_tracking_setup(storage_dir)
            if not is_valid:
                tracking_enabled = False
                if callbacks and hasattr(callbacks, 'on_activity'):
                    from datetime import datetime as dt
                    callbacks.on_activity({
                        "timestamp": dt.now().isoformat(),
                        "level": "WARN",
                        "message": f"Burr tracking disabled: {error}"
                    })

        # Generate unique app_id for Burr tracking
        import uuid
        from datetime import datetime
        app_id = f"{self.game.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        app = build_game_app(
            self.game,
            num_rounds,
            runtime_mode=runtime_mode,
            app_id=app_id,
            enable_tracking=tracking_enabled,
        )

        async def _run_with_session(aio_session: aiohttp.ClientSession) -> List[dict]:
            while app.state["round_num"] < app.state["max_rounds"]:
                round_num = app.state["round_num"]
                history = app.state["history"]
                history_payoffs = self._extract_payoffs_from_results(app.state["results"])
                cumulative = self._compute_cumulative(history_payoffs)

                # Async LLM calls OUTSIDE Burr
                tasks = [
                    self._get_action(
                        aio_session, player, history if is_repeated else [],
                        history_payoffs if is_repeated else None,
                        cumulative if is_repeated else None,
                        is_repeated
                    )
                    for player in players
                ]
                action_results: List[ActionResult] = await asyncio.gather(*tasks)

                # Extract data from action results
                actions = [r.action for r in action_results]
                response_times = [r.response_time for r in action_results]
                prompts = [r.prompt for r in action_results]
                raw_responses = [r.raw_response for r in action_results]
                was_parsed = [r.was_parsed for r in action_results]
                was_normalized = [r.was_normalized for r in action_results]

                # Calculate payoffs
                if self.game.is_discrete:
                    actions_for_payoff = tuple(actions)
                else:
                    actions_for_payoff = tuple(tuple(a) if isinstance(a, (list, tuple)) else a for a in actions)
                payoffs = self.game.payoff_fn(actions_for_payoff)

                # Invoke callbacks if provided
                if callbacks:
                    from datetime import datetime as dt

                    # Log prompts
                    for p_idx, player in enumerate(players):
                        if hasattr(callbacks, 'on_prompt'):
                            callbacks.on_prompt({
                                "round": round_num + 1,
                                "player": p_idx + 1,
                                "model": player.model,
                                "prompt": prompts[p_idx],
                                "response": raw_responses[p_idx],
                                "system_prompt": player.system_prompt,
                                "was_parsed": was_parsed[p_idx],
                            })

                    # Log API responses
                    for p_idx, player in enumerate(players):
                        if hasattr(callbacks, 'on_activity'):
                            callbacks.on_activity({
                                "timestamp": dt.now().isoformat(),
                                "level": "SUCCESS",
                                "message": f"API P{p_idx+1} {player.model}: {response_times[p_idx]:.2f}s"
                            })

                    # Log round status (matches GameRunnerService._run_game_loop behavior)
                    if hasattr(callbacks, 'on_activity'):
                        status_parts = []
                        for p_idx in range(len(players)):
                            if not self.game.is_discrete and was_normalized[p_idx]:
                                status = "NORMALIZED"
                            elif not was_parsed[p_idx]:
                                status = "DEFAULT"
                            else:
                                status = "parsed"

                            # Format action for display
                            action = actions[p_idx]
                            if self.game.is_discrete:
                                action_display = str(action)
                            elif hasattr(action, '__iter__') and not isinstance(action, str):
                                action_display = f"[{', '.join(f'{a:.0f}' for a in action)}]"
                            else:
                                action_display = str(action)

                            payoff_val = (
                                f"{payoffs[p_idx]:.1f}"
                                if not self.game.is_discrete
                                else str(payoffs[p_idx])
                            )
                            status_parts.append(f"P{p_idx+1}->{action_display} [{status}] ({payoff_val})")

                        any_issue = (
                            any(not p for p in was_parsed) or
                            (not self.game.is_discrete and any(was_normalized))
                        )
                        callbacks.on_activity({
                            "timestamp": dt.now().isoformat(),
                            "level": "WARN" if any_issue else "INFO",
                            "message": f"Round {round_num+1}: " + ", ".join(status_parts)
                        })

                # Feed results INTO Burr action with error handling
                tracking_failed = False
                try:
                    app.step(
                        inputs={
                            "actions": actions,
                            "payoffs": payoffs,
                            "player_configs": players,
                            "response_times": response_times,
                            "prompts": prompts,
                            "raw_responses": raw_responses,
                            "was_parsed": was_parsed,
                            "was_normalized": was_normalized,
                            "is_discrete": self.game.is_discrete,
                        }
                    )

                    # Step through check_complete transition
                    app.step()

                    # Validate state consistency
                    expected_round = round_num + 1
                    actual_round = app.state.get("round_num", -1)
                    if actual_round != expected_round:
                        if callbacks and hasattr(callbacks, 'on_activity'):
                            callbacks.on_activity({
                                "timestamp": dt.now().isoformat(),
                                "level": "WARN",
                                "message": f"Burr state mismatch: expected round {expected_round}, got {actual_round}"
                            })

                except Exception as e:
                    tracking_failed = True
                    if callbacks and hasattr(callbacks, 'on_activity'):
                        callbacks.on_activity({
                            "timestamp": dt.now().isoformat(),
                            "level": "ERROR",
                            "message": f"Burr tracking failed at round {round_num + 1}: {e}"
                        })
                    # Continue without tracking - game results still valid

                # Get the latest result from state
                latest_result = app.state["results"][-1] if app.state["results"] else None

                # Invoke on_round_complete callback
                if callbacks and latest_result and hasattr(callbacks, 'on_round_complete'):
                    callbacks.on_round_complete(latest_result)

                # Legacy progress callback
                if progress_callback:
                    progress_callback(app.state["round_num"], num_rounds)

            return app.state["results"]

        # Run with provided session or create new one
        if session is not None:
            results = await _run_with_session(session)
        else:
            async with aiohttp.ClientSession() as new_session:
                results = await _run_with_session(new_session)

        # Post-game verification
        if tracking_enabled and app_id:
            verification = self.verify_tracking(app_id, num_rounds)
            if callbacks and hasattr(callbacks, 'on_activity'):
                from datetime import datetime as dt
                if not verification["valid"]:
                    callbacks.on_activity({
                        "timestamp": dt.now().isoformat(),
                        "level": "WARN",
                        "message": f"Burr tracking verification failed: {verification['error']}"
                    })
                else:
                    callbacks.on_activity({
                        "timestamp": dt.now().isoformat(),
                        "level": "SUCCESS",
                        "message": f"Burr tracking verified: {verification['rounds_found']} rounds saved to {app_id}"
                    })

        return results

    def _extract_payoffs_from_results(
        self, results: List[dict]
    ) -> List[Tuple[float, ...]]:
        """Extract payoff history from Burr results state."""
        return [r.get("payoffs", ()) for r in results]

    def _compute_cumulative(
        self, history_payoffs: List[Tuple[float, ...]]
    ) -> Tuple[float, ...]:
        """Compute cumulative payoffs from history."""
        if not history_payoffs:
            return ()
        num_players = len(history_payoffs[0])
        totals = [
            sum(p[i] for p in history_payoffs) for i in range(num_players)
        ]
        return tuple(totals)

    def verify_tracking(self, app_id: str, expected_rounds: int) -> Dict[str, Any]:
        """Verify tracking files exist and contain expected data.

        Args:
            app_id: The Burr app ID used for this run
            expected_rounds: Number of rounds that should be tracked

        Returns:
            {"valid": bool, "rounds_found": int, "error": str, "log_path": str}
        """
        storage_dir = os.path.expanduser(
            os.environ.get("BURR_STORAGE_DIR", "~/.burr")
        )
        project = os.environ.get("BURR_PROJECT", "gametheory")

        log_path = os.path.join(storage_dir, project, app_id, "log.jsonl")

        if not os.path.exists(log_path):
            return {
                "valid": False,
                "rounds_found": 0,
                "error": "log.jsonl not found",
                "log_path": log_path
            }

        # Count play_game_round actions
        rounds_found = 0
        try:
            with open(log_path, 'r') as f:
                for line in f:
                    if "play_game_round" in line:
                        rounds_found += 1
        except IOError as e:
            return {
                "valid": False,
                "rounds_found": 0,
                "error": f"Failed to read log: {e}",
                "log_path": log_path
            }

        if rounds_found != expected_rounds:
            return {
                "valid": False,
                "rounds_found": rounds_found,
                "error": f"Expected {expected_rounds} rounds, found {rounds_found}",
                "log_path": log_path
            }

        return {
            "valid": True,
            "rounds_found": rounds_found,
            "error": "",
            "log_path": log_path
        }
