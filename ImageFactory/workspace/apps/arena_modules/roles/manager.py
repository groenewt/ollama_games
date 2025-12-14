"""Role management business logic for arena."""

from typing import List, Dict, Any, Optional, Tuple


class RolesManager:
    """Manages role CRUD operations and related analytics.

    Provides a facade over RoleRepository and AnalyticsService
    for role-related operations in the arena.
    """

    def __init__(self, role_repository, analytics_service=None):
        """Initialize manager with repository and optional analytics.

        Args:
            role_repository: RoleRepository instance
            analytics_service: Optional AnalyticsService for model stats
        """
        self.role_repository = role_repository
        self.analytics_service = analytics_service

    def list_all(self) -> List[Any]:
        """Get all roles.

        Returns:
            List of RoleConfig objects
        """
        return self.role_repository.list_all()

    def list_with_stats(self) -> List[Tuple[Any, Dict[str, Any]]]:
        """Get all roles with their performance stats.

        Returns:
            List of (RoleConfig, stats_dict) tuples
        """
        roles = self.role_repository.list_all()
        model_stats = self.get_model_stats()

        results = []
        for role in roles:
            stats = model_stats.get(role.model, {})
            results.append((role, stats))

        return results

    def get_model_stats(self) -> Dict[str, Any]:
        """Get model statistics from analytics service.

        Returns:
            Dict of model -> stats, empty if no analytics service
        """
        if not self.analytics_service:
            return {}
        return self.analytics_service.get_model_stats_dict()

    def get_by_id(self, role_id: str) -> Optional[Any]:
        """Get a role by ID.

        Args:
            role_id: Role identifier

        Returns:
            RoleConfig or None if not found
        """
        return self.role_repository.get_by_id(role_id)

    def create(self, role) -> None:
        """Create a new role.

        Args:
            role: RoleConfig to create

        Raises:
            ValueError: If role already exists or validation fails
        """
        self.role_repository.create(role)

    def update(self, role) -> None:
        """Update an existing role.

        Args:
            role: RoleConfig to update

        Raises:
            ValueError: If role doesn't exist or validation fails
        """
        self.role_repository.update(role)

    def delete(self, role_id: str) -> None:
        """Delete a role.

        Args:
            role_id: Role identifier to delete
        """
        self.role_repository.delete(role_id)

    def get_roles_for_game(self, game_id: str) -> List[Any]:
        """Get roles that are allowed to play a specific game.

        Args:
            game_id: Game identifier

        Returns:
            List of RoleConfig objects allowed for the game
        """
        roles = self.role_repository.list_all()
        return [
            role for role in roles
            if not role.allowed_games or game_id in role.allowed_games
        ]

    def build_game_descriptions(self, game_registry: Dict[str, Any]) -> Dict[str, str]:
        """Build game_id -> description mapping.

        Args:
            game_registry: Dict of game_id -> GameDefinition

        Returns:
            Dict of game_id -> description
        """
        return {
            game_id: game.description
            for game_id, game in game_registry.items()
        }

    def build_game_prompt_instructions(
        self,
        game_registry: Dict[str, Any]
    ) -> Dict[str, str]:
        """Build game_id -> prompt instructions mapping.

        Args:
            game_registry: Dict of game_id -> GameDefinition

        Returns:
            Dict of game_id -> prompt_instructions
        """
        return {
            game_id: game.action_space.prompt_instructions()
            for game_id, game in game_registry.items()
        }
