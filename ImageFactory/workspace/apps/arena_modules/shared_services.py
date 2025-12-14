"""Service initialization for arena modules.

Provides a centralized service container for arena.py to reduce boilerplate
and make service dependencies explicit.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class ArenaServices:
    """Container for shared arena services.

    Groups all core services needed by the arena application:
    - metrics: MetricsTracker for real-time stats
    - session_manager: SessionManager for persistent storage
    - role_repository: RoleRepository for role CRUD
    - analytics_service: AnalyticsService for cross-game analytics
    - role_analytics_service: RoleAnalyticsService for per-role analytics
    """
    metrics: 'MetricsTracker'
    session_manager: 'SessionManager'
    role_repository: 'RoleRepository'
    analytics_service: Optional['AnalyticsService'] = None
    role_analytics_service: Optional['RoleAnalyticsService'] = None


def init_services(
    app_path: Path,
    MetricsTracker,
    SessionManager,
    RoleRepository,
    AnalyticsService=None,
    RoleAnalyticsService=None,
) -> ArenaServices:
    """Initialize all arena services from app path.

    This function centralizes service initialization logic that was
    previously scattered across multiple Marimo cells.

    Args:
        app_path: Path to the arena.py file (__file__)
        MetricsTracker: MetricsTracker class
        SessionManager: SessionManager class
        RoleRepository: RoleRepository class
        AnalyticsService: Optional AnalyticsService class
        RoleAnalyticsService: Optional RoleAnalyticsService class

    Returns:
        ArenaServices container with initialized services

    Example:
        from arena_modules import init_services

        services = init_services(
            Path(__file__),
            MetricsTracker,
            SessionManager,
            RoleRepository,
        )
        metrics = services.metrics
        session_manager = services.session_manager
    """
    data_dir = app_path.parent.parent / "data"

    metrics = MetricsTracker()
    session_manager = SessionManager(str(data_dir / "sessions"))
    role_repository = RoleRepository(str(data_dir / "arena.duckdb"))

    analytics_service = None
    if AnalyticsService is not None:
        analytics_service = AnalyticsService(session_manager, metrics)

    role_analytics_service = None
    if RoleAnalyticsService is not None:
        role_analytics_service = RoleAnalyticsService(session_manager, role_repository)

    return ArenaServices(
        metrics=metrics,
        session_manager=session_manager,
        role_repository=role_repository,
        analytics_service=analytics_service,
        role_analytics_service=role_analytics_service,
    )
