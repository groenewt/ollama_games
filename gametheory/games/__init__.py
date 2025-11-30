"""Game definitions registry."""

from typing import Dict, List
from ..core.types import GameDefinition

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

# Game registry mapping game_id to GameDefinition
GAME_REGISTRY: Dict[str, GameDefinition] = {
    "prisoners_dilemma": PRISONERS_DILEMMA,
    "chicken": CHICKEN,
    "stag_hunt": STAG_HUNT,
    "battle_of_the_sexes": BATTLE_OF_SEXES,
    "matching_pennies": MATCHING_PENNIES,
    "trust_game": TRUST_GAME,
    "public_goods": PUBLIC_GOODS,
    "coordination_game": COORDINATION,
    "three_player_public_good": THREE_PLAYER_PUBLIC_GOOD,
    "iterated_prisoners_dilemma": ITERATED_PRISONERS_DILEMMA,
    "punishment_public_goods": PUNISHMENT_PUBLIC_GOODS,
}

# Categorized game lists
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


def get_game(game_id: str) -> GameDefinition:
    """Retrieve game by ID.

    Args:
        game_id: The unique identifier of the game.

    Returns:
        The GameDefinition for the requested game.

    Raises:
        KeyError: If the game_id is not found in the registry.
    """
    if game_id not in GAME_REGISTRY:
        available = ", ".join(GAME_REGISTRY.keys())
        raise KeyError(f"Game '{game_id}' not found. Available games: {available}")
    return GAME_REGISTRY[game_id]


def list_games() -> List[str]:
    """List all available game IDs."""
    return list(GAME_REGISTRY.keys())


def list_games_by_players(num_players: int) -> List[str]:
    """List games that support a specific number of players."""
    return [
        game_id
        for game_id, game in GAME_REGISTRY.items()
        if game.num_players == num_players
    ]


def get_game_names() -> Dict[str, str]:
    """Get a mapping of game_id to display name for UI dropdowns."""
    return {game_id: game.name for game_id, game in GAME_REGISTRY.items()}


__all__ = [
    "GAME_REGISTRY",
    "TWO_PLAYER_GAMES",
    "MULTI_PLAYER_GAMES",
    "SEQUENTIAL_GAMES",
    "get_game",
    "list_games",
    "list_games_by_players",
    "get_game_names",
    "PRISONERS_DILEMMA",
    "CHICKEN",
    "STAG_HUNT",
    "BATTLE_OF_SEXES",
    "MATCHING_PENNIES",
    "TRUST_GAME",
    "PUBLIC_GOODS",
    "COORDINATION",
    "THREE_PLAYER_PUBLIC_GOOD",
    "ITERATED_PRISONERS_DILEMMA",
    "PUNISHMENT_PUBLIC_GOODS",
]
