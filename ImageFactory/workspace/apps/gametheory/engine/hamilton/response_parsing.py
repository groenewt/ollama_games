"""Hamilton DAG for response parsing."""
from typing import Any, List, Optional, Tuple, Union
import json
import re


def parse_discrete_action(
    response_text: str,
    action_space: Any,
) -> Tuple[str, bool]:
    """Parse discrete action from response."""
    # Try JSON first
    try:
        match = re.search(r'\{[^{}]*"action"\s*:\s*"([^"]+)"[^{}]*\}', response_text, re.DOTALL)
        if match:
            action_str = match.group(1)
            matched = action_space._normalize_action(action_str)
            if matched:
                return matched, True
    except Exception:
        pass

    # Fallback: use action space's match_action
    matched, was_parsed = action_space.match_action(response_text)
    if matched:
        return matched, was_parsed

    return action_space.actions[0], False


def parse_allocation(
    response_text: str,
    action_space: Any,
) -> Tuple[List[float], bool, bool]:
    """Parse allocation from response."""
    budget = action_space.budget
    num_fields = action_space.num_fields

    # Try JSON first
    try:
        match = re.search(r'\{[^{}]*"allocations"\s*:\s*\[[^\]]+\][^{}]*\}', response_text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            if "allocations" in data:
                raw_allocs = data["allocations"]
                allocs = []
                for x in raw_allocs:
                    if isinstance(x, (int, float)):
                        allocs.append(float(x))
                    elif isinstance(x, str) and x.replace('.', '').isdigit():
                        allocs.append(float(x))

                if len(allocs) == num_fields:
                    valid, _ = action_space.validate(allocs)
                    if valid:
                        return allocs, True, False

                    # Normalize
                    total = sum(allocs)
                    if total > 0:
                        scale = budget / total
                        allocs = [a * scale for a in allocs]
                        was_parsed = abs(1.0 - scale) < 0.05
                        return allocs, was_parsed, True
    except Exception:
        pass

    # Fallback: extract numbers
    numbers = re.findall(r'\d+\.?\d*', response_text)
    if len(numbers) >= num_fields:
        allocs = [float(n) for n in numbers[:num_fields]]
        total = sum(allocs)
        if total > 0:
            if abs(total - budget) > 0.01:
                scale = budget / total
                allocs = [a * scale for a in allocs]
                was_parsed = abs(1.0 - scale) < 0.05
                return allocs, was_parsed, True
            return allocs, True, False

    # Default uniform
    per_field = budget / num_fields
    return [per_field] * num_fields, False, False


def parse_permutation(
    response_text: str,
    action_space: Any,
) -> Tuple[List[int], bool]:
    """Parse permutation from response."""
    num_positions = action_space.num_positions

    # Try JSON
    try:
        match = re.search(r'\{[^{}]*"assignment"\s*:\s*\[([^\]]+)\][^{}]*\}', response_text, re.DOTALL)
        if match:
            values = [int(x.strip()) for x in match.group(1).split(",") if x.strip().lstrip("-").isdigit()]
            if len(values) == num_positions:
                valid, _ = action_space.validate(values)
                if valid:
                    return values, True
    except Exception:
        pass

    # Fallback: brackets
    bracket_match = re.search(r"[\[\(]([^\]\)]+)[\]\)]", response_text)
    if bracket_match:
        try:
            values = [int(x.strip()) for x in bracket_match.group(1).split(",") if x.strip().lstrip("-").isdigit()]
            if len(values) == num_positions:
                valid, _ = action_space.validate(values)
                if valid:
                    return values, True
        except Exception:
            pass

    # Default
    if hasattr(action_space, "default_action"):
        return action_space.default_action(), False
    return list(range(num_positions)), False


def reasoning_trace(response_text: str) -> Optional[str]:
    """Extract reasoning from response."""
    try:
        match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', response_text, re.IGNORECASE | re.DOTALL)
        if match and match.group(1).strip():
            return match.group(1).strip()
    except Exception:
        pass

    # Look for reasoning keywords
    patterns = [
        r'(?:because|since|therefore)[:\s]+([^.!?\n]+[.!?]?)',
        r'(?:my reasoning|strategy)[:\s]+([^.!?\n]+[.!?]?)',
    ]
    for pattern in patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match and len(match.group(1).strip()) > 10:
            return match.group(1).strip()

    return None
