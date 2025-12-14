"""Discrete action space for classic game theory games.

This module provides the DiscreteActionSpace class for games like
Prisoner's Dilemma, Chicken, Stag Hunt, etc.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Tuple, Optional


@lru_cache(maxsize=256)
def _normalize_action_variants(action: str) -> Tuple[str, ...]:
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


@dataclass
class DiscreteActionSpace:
    """Action space for discrete choice games (PD, Chicken, etc.).

    This class provides validation and prompt generation for games
    where players choose from a finite set of actions.
    """

    actions: List[str]

    def validate(self, action: str) -> Tuple[bool, str]:
        """Validate an action against the allowed choices.

        Args:
            action: The action string to validate.

        Returns:
            Tuple of (is_valid, error_message). error_message is empty if valid.
        """
        normalized = self._normalize_action(action)
        if normalized:
            return True, ""
        return False, f"Invalid action: '{action}'. Choose from: {self.actions}"

    def _normalize_action(self, action: str) -> Optional[str]:
        """Normalize an action string to match one of the valid actions.

        Handles variants like "cooperate" vs "cooperation" vs "co-operate".

        Args:
            action: The action string from the LLM response.

        Returns:
            The canonical action string if matched, None otherwise.
        """
        action_lower = action.lower().strip()

        # Direct match first
        for valid_action in self.actions:
            if action_lower == valid_action.lower():
                return valid_action

        # Check variants
        for valid_action in self.actions:
            variants = _normalize_action_variants(valid_action)
            for variant in variants:
                if variant in action_lower:
                    return valid_action

        return None

    def match_action(self, response_text: str) -> Tuple[Optional[str], bool]:
        """Match response text against valid actions.

        Args:
            response_text: The LLM response text.

        Returns:
            Tuple of (matched_action, was_parsed). Returns (None, False) if no match.
        """
        text_lower = response_text.lower().strip()

        # Try each valid action
        for action in self.actions:
            variants = _normalize_action_variants(action)
            for variant in variants:
                if variant in text_lower:
                    return action, True

        return None, False

    def prompt_instructions(self) -> str:
        """Generate prompt instructions for this action space.

        Returns:
            String with formatting instructions for the LLM response.
        """
        actions_str = ", ".join([f'"{a}"' for a in self.actions])

        return f"""You must choose exactly ONE action from: {actions_str}

CRITICAL FORMAT REQUIREMENTS:
- Respond with ONLY a JSON object
- Include your chosen action and brief reasoning

EXAMPLE RESPONSE FORMAT:
{{"action": "{self.actions[0]}", "reasoning": "brief explanation of your choice"}}

YOUR RESPONSE MUST FOLLOW THIS EXACT FORMAT."""
