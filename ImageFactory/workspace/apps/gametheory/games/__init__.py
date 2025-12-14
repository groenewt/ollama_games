"""Game definitions registry.

All games now use BurrGameDefinition with either DiscreteActionSpace
or AllocationSpace for unified behavior through BurrGameRunner.
"""

from typing import Dict, List

from ..engine.burr_app import BurrGameDefinition
from .matrix_factory import create_matrix_game

# Import all games (all now BurrGameDefinition)
from .prisoners_dilemma import GAME as PRISONERS_DILEMMA
from .chicken import GAME as CHICKEN
from .stag_hunt import GAME as STAG_HUNT
from .battle_of_sexes import GAME as BATTLE_OF_SEXES
from .matching_pennies import GAME as MATCHING_PENNIES
from .trust_game import GAME as TRUST_GAME
from .public_goods import GAME as PUBLIC_GOODS
from .coordination import GAME as COORDINATION
from .three_player_public import GAME as THREE_PLAYER_PUBLIC_GOOD
from .iterated_pd import GAME as ITERATED_PRISONERS_DILEMMA
from .punishment_public import GAME as PUNISHMENT_PUBLIC_GOODS
from .colonel_blotto import BLOTTO_GAMES, COLONEL_BLOTTO_5
from .tennis_coach import TENNIS_GAMES, TENNIS_COACH_4
from .sumo_coach import SUMO_GAMES, SUMO_COACH_4

# Unified game registry - ALL games use BurrGameDefinition now
GAME_REGISTRY: Dict[str, BurrGameDefinition] = {
    # Discrete games (2-player)
    "prisoners_dilemma": PRISONERS_DILEMMA,
    "chicken": CHICKEN,
    "stag_hunt": STAG_HUNT,
    "battle_of_the_sexes": BATTLE_OF_SEXES,
    "matching_pennies": MATCHING_PENNIES,
    "trust_game": TRUST_GAME,
    "public_goods": PUBLIC_GOODS,
    "coordination_game": COORDINATION,
    "iterated_prisoners_dilemma": ITERATED_PRISONERS_DILEMMA,
    "punishment_public_goods": PUNISHMENT_PUBLIC_GOODS,
    # Discrete games (3-player)
    "three_player_public_good": THREE_PLAYER_PUBLIC_GOOD,
    # Allocation games
    **BLOTTO_GAMES,
    **TENNIS_GAMES,
    **SUMO_GAMES,
}

# Backwards compatibility alias
BURR_GAME_REGISTRY = GAME_REGISTRY

# Categorized game lists
DISCRETE_GAMES = [
    "prisoners_dilemma",
    "chicken",
    "stag_hunt",
    "battle_of_the_sexes",
    "matching_pennies",
    "trust_game",
    "public_goods",
    "coordination_game",
    "iterated_prisoners_dilemma",
    "punishment_public_goods",
    "three_player_public_good",
]

TWO_PLAYER_GAMES = [
    "prisoners_dilemma",
    "chicken",
    "stag_hunt",
    "battle_of_the_sexes",
    "matching_pennies",
    "trust_game",
    "public_goods",
    "coordination_game",
    "iterated_prisoners_dilemma",
    "punishment_public_goods",
]

MULTI_PLAYER_GAMES = [
    "three_player_public_good",
]

SEQUENTIAL_GAMES = [
    "iterated_prisoners_dilemma",
]

ALLOCATION_GAMES = list(BLOTTO_GAMES.keys())
PERMUTATION_GAMES = list(TENNIS_GAMES.keys()) + list(SUMO_GAMES.keys())


def get_game(game_id: str) -> BurrGameDefinition:
    """Retrieve game by ID.

    Args:
        game_id: The unique identifier of the game.

    Returns:
        The BurrGameDefinition for the requested game.

    Raises:
        KeyError: If the game_id is not found.
    """
    if game_id in GAME_REGISTRY:
        return GAME_REGISTRY[game_id]
    available = ", ".join(GAME_REGISTRY.keys())
    raise KeyError(f"Game '{game_id}' not found. Available games: {available}")


def list_games() -> List[str]:
    """List all available game IDs."""
    return list(GAME_REGISTRY.keys())


def get_game_names() -> Dict[str, str]:
    """Get a mapping of game_id to display name for UI dropdowns."""
    return {game_id: game.name for game_id, game in GAME_REGISTRY.items()}


def is_discrete_game(game_id: str) -> bool:
    """Check if a game uses discrete actions."""
    if game_id not in GAME_REGISTRY:
        return False
    return GAME_REGISTRY[game_id].is_discrete


def is_allocation_game(game_id: str) -> bool:
    """Check if a game uses allocation/permutation actions."""
    return game_id in ALLOCATION_GAMES or game_id in PERMUTATION_GAMES


# Backwards compatibility - is_burr_game now always returns True
def is_burr_game(game_id: str) -> bool:
    """Check if a game uses the Burr engine. Always True now."""
    return game_id in GAME_REGISTRY


__all__ = [
    "GAME_REGISTRY",
    "BURR_GAME_REGISTRY",
    "get_game",
    "list_games",
    "get_game_names",
    "is_burr_game",
    "is_discrete_game",
    "is_allocation_game",
    "create_matrix_game",
    "DISCRETE_GAMES",
    "ALLOCATION_GAMES",
    "PERMUTATION_GAMES",
    "TWO_PLAYER_GAMES",
    "MULTI_PLAYER_GAMES",
]
