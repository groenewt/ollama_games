"""Custom game creation logic for arena."""

from typing import List, Tuple, Dict, Optional
from itertools import product as itertools_product


def parse_actions_list(actions_text: str) -> List[str]:
    """Parse comma-separated actions string into list.

    Args:
        actions_text: Comma-separated action names

    Returns:
        List of trimmed action strings
    """
    return [a.strip() for a in actions_text.split(",") if a.strip()]


def create_custom_game(
    name: str,
    actions: List[str],
    num_players: int,
    GameDefinition,
    description: str = "User-created custom game",
) -> Optional['GameDefinition']:
    """Create a custom GameDefinition with default payoffs.

    Args:
        name: Display name for the game
        actions: List of available actions
        num_players: Number of players (2 or 3)
        GameDefinition: GameDefinition class to instantiate
        description: Optional game description

    Returns:
        GameDefinition instance or None if invalid inputs
    """
    if not name:
        name = "Custom Game"

    if not actions or len(actions) < 2:
        return None

    if num_players not in (2, 3):
        return None

    # Generate payoff matrix with default values (0)
    payoff_matrix: Dict[Tuple[str, ...], Tuple[int, ...]] = {}
    for combo in itertools_product(actions, repeat=num_players):
        payoff_matrix[combo] = tuple(0 for _ in range(num_players))

    game = GameDefinition(
        id="custom_" + name.lower().replace(" ", "_"),
        name=name,
        description=description,
        payoff_matrix=payoff_matrix,
        actions=actions,
        num_players=num_players,
    )
    return game
