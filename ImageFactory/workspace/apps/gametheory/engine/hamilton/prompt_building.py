"""Hamilton DAG for prompt construction."""
from typing import Any, List, Optional, Tuple, Union


def system_prompt_section(player_system_prompt: Optional[str]) -> str:
    """Build system prompt framing section."""
    if not player_system_prompt:
        return ""
    return f"""=== YOUR STRATEGIC GUIDANCE ===
{player_system_prompt}
=== APPLY THIS GUIDANCE IN YOUR STRATEGY ===
"""


def game_description_section(
    game_name: str,
    game_description: str,
    player_strategy_hints: Optional[str],
) -> str:
    """Build game description, strip default hints if custom provided."""
    desc = game_description
    if player_strategy_hints and "Strategic considerations:" in desc:
        desc = desc.split("Strategic considerations:")[0].strip()

    result = f"Game: {game_name}\n{desc}"

    if player_strategy_hints:
        result += f"\n\nStrategic considerations:\n{player_strategy_hints}"

    return result


def action_space_instructions(action_space: Any) -> str:
    """Delegate to action space's prompt_instructions()."""
    return action_space.prompt_instructions()


def history_section(
    history: List[Tuple[Any, ...]],
    history_payoffs: Optional[List[Tuple[float, ...]]],
    cumulative_payoffs: Optional[Tuple[float, ...]],
    is_repeated: bool,
    max_history_display: int = 5,
) -> str:
    """Format history for prompt."""
    if not is_repeated or not history:
        return ""

    lines = ["\nPrevious rounds:"]
    start_idx = max(0, len(history) - max_history_display)

    for i, round_allocs in enumerate(history[start_idx:], start_idx + 1):
        round_str = f"  Round {i}: Actions={round_allocs}"
        if history_payoffs and i <= len(history_payoffs):
            round_str += f", Payoffs={history_payoffs[i-1]}"
        lines.append(round_str)

    if cumulative_payoffs:
        lines.append(f"  Cumulative payoffs so far: {cumulative_payoffs}")

    return "\n".join(lines)


def reminder_section(player_system_prompt: Optional[str]) -> str:
    """Reminder to apply strategic guidance."""
    if not player_system_prompt:
        return ""
    return "\nIMPORTANT: Apply the strategic guidance above when choosing your action."


def full_prompt(
    system_prompt_section: str,
    game_description_section: str,
    action_space_instructions: str,
    history_section: str,
    reminder_section: str,
) -> str:
    """Compose full prompt from sections."""
    sections = [
        system_prompt_section,
        game_description_section,
        action_space_instructions,
        history_section,
        reminder_section,
    ]
    return "\n".join(s for s in sections if s)
