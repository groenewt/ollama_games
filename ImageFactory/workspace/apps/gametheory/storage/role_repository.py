"""DuckDB-backed repository for Role CRUD operations."""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import duckdb

from ..core.role import RoleConfig
from .schema import ROLES_TABLE_DDL


class RoleRepository:
    """Repository for managing Role entities in DuckDB."""

    def __init__(self, db_path: str = "data/arena.duckdb"):
        """Initialize repository with database path.

        Args:
            db_path: Path to DuckDB database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        """Get a database connection."""
        return duckdb.connect(str(self.db_path))

    def _init_schema(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.execute(ROLES_TABLE_DDL)

    def create(self, role: RoleConfig) -> RoleConfig:
        """Create a new role.

        Args:
            role: RoleConfig to create

        Returns:
            Created RoleConfig with generated ID

        Raises:
            ValueError: If role name already exists
        """
        with self._get_connection() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO roles (
                        role_id, name, description, endpoint, model,
                        system_prompt, game_instructions, temperature, top_p,
                        top_k, repeat_penalty, created_at, updated_at, is_active
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    [
                        role.role_id,
                        role.name,
                        role.description,
                        role.endpoint,
                        role.model,
                        role.system_prompt,
                        json.dumps(role.game_instructions),
                        role.temperature,
                        role.top_p,
                        role.top_k,
                        role.repeat_penalty,
                        role.created_at,
                        role.updated_at,
                        role.is_active,
                    ],
                )
            except duckdb.ConstraintException:
                raise ValueError(f"Role name '{role.name}' already exists")
        return role

    def get_by_id(self, role_id: str) -> Optional[RoleConfig]:
        """Get a role by ID."""
        with self._get_connection() as conn:
            result = conn.execute(
                "SELECT * FROM roles WHERE role_id = ? AND is_active = TRUE", [role_id]
            ).fetchone()
            if result:
                return self._row_to_role(result, conn)
        return None

    def get_by_name(self, name: str) -> Optional[RoleConfig]:
        """Get a role by unique name."""
        with self._get_connection() as conn:
            result = conn.execute(
                "SELECT * FROM roles WHERE name = ? AND is_active = TRUE", [name]
            ).fetchone()
            if result:
                return self._row_to_role(result, conn)
        return None

    def list_all(self, include_inactive: bool = False) -> List[RoleConfig]:
        """List all roles.

        Args:
            include_inactive: Whether to include soft-deleted roles
        """
        with self._get_connection() as conn:
            query = "SELECT * FROM roles"
            if not include_inactive:
                query += " WHERE is_active = TRUE"
            query += " ORDER BY name"
            results = conn.execute(query).fetchall()
            return [self._row_to_role(r, conn) for r in results]

    def list_for_game(self, game_id: str) -> List[RoleConfig]:
        """List roles that can play a specific game.

        Args:
            game_id: The game type ID
        """
        all_roles = self.list_all()
        return [r for r in all_roles if r.can_play_game(game_id)]

    def update(self, role: RoleConfig) -> RoleConfig:
        """Update an existing role.

        Args:
            role: RoleConfig with updated fields

        Returns:
            Updated RoleConfig
        """
        role.updated_at = datetime.utcnow()
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE roles SET
                    name = ?, description = ?, endpoint = ?, model = ?,
                    system_prompt = ?, game_instructions = ?, temperature = ?,
                    top_p = ?, top_k = ?, repeat_penalty = ?, updated_at = ?
                WHERE role_id = ?
            """,
                [
                    role.name,
                    role.description,
                    role.endpoint,
                    role.model,
                    role.system_prompt,
                    json.dumps(role.game_instructions),
                    role.temperature,
                    role.top_p,
                    role.top_k,
                    role.repeat_penalty,
                    role.updated_at,
                    role.role_id,
                ],
            )
        return role

    def delete(self, role_id: str, hard: bool = False):
        """Delete a role.

        Args:
            role_id: ID of role to delete
            hard: If True, permanently delete. If False, soft delete.
        """
        with self._get_connection() as conn:
            if hard:
                conn.execute("DELETE FROM roles WHERE role_id = ?", [role_id])
            else:
                conn.execute(
                    "UPDATE roles SET is_active = FALSE, updated_at = ? WHERE role_id = ?",
                    [datetime.utcnow(), role_id],
                )

    def _row_to_role(self, row, conn: duckdb.DuckDBPyConnection) -> RoleConfig:
        """Convert database row to RoleConfig."""
        # Get column names from the connection
        columns = [desc[0] for desc in conn.description]
        row_dict = dict(zip(columns, row))

        return RoleConfig(
            role_id=row_dict["role_id"],
            name=row_dict["name"],
            description=row_dict.get("description") or "",
            endpoint=row_dict["endpoint"],
            model=row_dict["model"],
            system_prompt=row_dict.get("system_prompt") or "",
            game_instructions=(
                json.loads(row_dict["game_instructions"])
                if row_dict.get("game_instructions")
                else {}
            ),
            temperature=row_dict.get("temperature", 0.7),
            top_p=row_dict.get("top_p", 0.9),
            top_k=row_dict.get("top_k", 40),
            repeat_penalty=row_dict.get("repeat_penalty", 1.1),
            created_at=row_dict.get("created_at", datetime.utcnow()),
            updated_at=row_dict.get("updated_at", datetime.utcnow()),
            is_active=row_dict.get("is_active", True),
        )
