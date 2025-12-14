"""Role configuration for player identities."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, TYPE_CHECKING
import uuid

if TYPE_CHECKING:
    from .types import PlayerConfig


def extract_strategic_considerations(description: str) -> str:
    """Extract the strategic considerations section from a game description.

    Args:
        description: Full game description text

    Returns:
        Just the strategic considerations portion, or empty string if not found
    """
    marker = "Strategic considerations:"
    if marker in description:
        return description.split(marker, 1)[1].strip()
    return ""


@dataclass
class RoleConfig:
    """Configuration for a player role/identity.

    Roles are persistent player identities that can be selected in the Play tab.
    They encapsulate model configuration, system prompts, and game constraints.
    """

    # Unique identifier (auto-generated if not provided)
    role_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    # Display name (must be unique)
    name: str = ""

    # Optional description of the role's purpose/strategy
    description: str = ""

    # Model configuration
    endpoint: str = ""  # Ollama API URL
    model: str = ""  # LLM model name

    # System prompt - custom instructions for this role
    system_prompt: str = ""

    # Game-specific instructions - mapping of game_id -> strategy instructions
    # Empty dict means all games allowed with default instructions
    game_instructions: Dict[str, str] = field(default_factory=dict)

    # Default parameters
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True  # Soft delete support

    @property
    def allowed_games(self) -> List[str]:
        """Get list of allowed game IDs (backwards compatibility).

        Returns:
            List of game_ids from game_instructions keys
        """
        return list(self.game_instructions.keys())

    def to_player_config(
        self,
        player_id: int,
        game_id: Optional[str] = None,
        strategy_hints: Optional[str] = None,
    ) -> "PlayerConfig":
        """Convert role to PlayerConfig for game execution.

        Args:
            player_id: The player number (1, 2, 3...)
            game_id: Optional game ID to get game-specific instructions
            strategy_hints: Optional session-specific strategy override (takes precedence)

        Returns:
            PlayerConfig ready for game execution
        """
        from .types import PlayerConfig

        # Priority: session override > game-specific > None
        hints = strategy_hints
        if not hints and game_id and game_id in self.game_instructions:
            hints = self.game_instructions[game_id] or None

        return PlayerConfig(
            player_id=player_id,
            model=self.model,
            endpoint=self.endpoint,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repeat_penalty=self.repeat_penalty,
            system_prompt=self.system_prompt or None,
            strategy_hints=hints,
        )

    def can_play_game(self, game_id: str) -> bool:
        """Check if this role is allowed to play a specific game.

        Args:
            game_id: The game type ID to check

        Returns:
            True if the role can play this game
        """
        if not self.game_instructions:
            return True  # Empty dict = all games allowed
        return game_id in self.game_instructions

    def get_game_instructions(self, game_id: str) -> str:
        """Get the custom instructions for a specific game.

        Args:
            game_id: The game type ID

        Returns:
            The custom strategy instructions, or empty string if not set
        """
        return self.game_instructions.get(game_id, "")

    def validate(self) -> List[str]:
        """Validate role configuration.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        if not self.name:
            errors.append("Role name is required")
        if not self.endpoint:
            errors.append("Endpoint is required")
        if not self.model:
            errors.append("Model is required")
        if self.temperature < 0 or self.temperature > 2:
            errors.append("Temperature must be between 0 and 2")
        if self.top_p < 0 or self.top_p > 1:
            errors.append("Top P must be between 0 and 1")
        if self.top_k < 1:
            errors.append("Top K must be at least 1")
        if self.repeat_penalty < 1:
            errors.append("Repeat penalty must be at least 1")
        return errors

    def __str__(self) -> str:
        """String representation for display."""
        games = (
            ", ".join(self.allowed_games[:3]) if self.allowed_games else "All games"
        )
        if len(self.allowed_games) > 3:
            games += f" (+{len(self.allowed_games) - 3})"
        return f"{self.name} ({self.model} @ {self.endpoint.split('/')[-1]}) - {games}"
