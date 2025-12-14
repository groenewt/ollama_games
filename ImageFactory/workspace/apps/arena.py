"""Game Theory LLM Arena - Main tabbed Marimo application."""
import sys
import marimo
from pathlib import Path


__generated_with = "0.17.8"
app = marimo.App(width="full")


@app.cell
def _(mo):
    # Session storage with reactive state for custom payoffs
    session_custom_games = {}
    activity_log = []  # Verbose activity logging

    # Reactive state for custom payoffs - enables UI updates when payoffs change
    get_custom_payoffs, set_custom_payoffs = mo.state({})

    return (
        activity_log,
        session_custom_games,
        get_custom_payoffs,
        set_custom_payoffs,
    )


@app.cell
def _():
    import marimo as mo
    import asyncio
    import polars as pl
    import altair as alt
    import time
    import sys as _sys
    from datetime import datetime
    from pathlib import Path

    # Use absolute path - __file__ is unreliable in marimo (points to /tmp/)
    _apps_dir = Path("/proj/gametheory/ImageFactory/workspace/apps")
    if str(_apps_dir) not in _sys.path:
        _sys.path.insert(0, str(_apps_dir))

    _project_root = _apps_dir.parent
    if str(_project_root) not in _sys.path:
        _sys.path.insert(0, str(_project_root))

    from itertools import product as itertools_product

    # Import game theory package
    from gametheory import (
        OLLAMA_MODELS,
        OLLAMA_ENDPOINTS,
        GAME_REGISTRY,
        get_game,
        list_games,
        get_game_names,
        is_burr_game,
        is_discrete_game,
        PlayerConfig,
        BurrGameRunner,
        MetricsTracker,
        SessionManager,
        CrossGameAnalyzer,
        AnalyticsService,
    )
    # Import new analytics and experiment modules
    from gametheory.analytics import (
        HyperparameterSensitivityAnalyzer,
        MetaStrategyAnalyzer,
        ModelPersonalityProfiler,
        CrossGameComparativeAnalyzer,
    )
    from gametheory.experiments import (
        TournamentConfig,
        TournamentRunner,
        EcosystemSimulator,
        PayoffSensitivityAnalyzer,
        create_quick_tournament,
    )
    from gametheory.core.config import (
        discover_all_available,
        DEFAULT_OLLAMA_MODELS,
        DEFAULT_OLLAMA_ENDPOINTS,
        DISCOVERY_TIMEOUT,
        SERIES_TIMEOUT,
        MAX_NUM_GAMES,
    )
    from gametheory.core.types import GameDefinition
    from gametheory.visualization import (
        create_cumulative_payoff_chart,
        create_action_distribution_chart,
        create_payoff_comparison_chart,
        create_avg_payoff_chart,
    )
    from gametheory.ui import (
        AnalyticsPanelBuilder,  # Deprecated
        RoleAnalyticsPanelBuilder,
        GameAnalyticsPanelBuilder,
        RolesTabBuilder,
        RoleAnalyticsFilterBuilder,
        RoleAnalyticsEmptyStates,
        ImprovedRoleAnalyticsPanelBuilder,
    )
    from arena_modules.queue import QueuedGame, QueueExecutionResult, QueueUIBuilder
    from gametheory.analytics import (
        RoleAnalyticsService,
        RoleFilterParams,
    )
    from gametheory.core.utils import detect_num_players
    from gametheory import RoleConfig, RoleRepository

    return (
        AnalyticsPanelBuilder,
        AnalyticsService,
        DEFAULT_OLLAMA_ENDPOINTS,
        DEFAULT_OLLAMA_MODELS,
        DISCOVERY_TIMEOUT,
        GAME_REGISTRY,
        GameDefinition,
        OLLAMA_ENDPOINTS,
        OLLAMA_MODELS,
        CrossGameAnalyzer,
        BurrGameRunner,
        MetricsTracker,
        Path,
        PlayerConfig,
        SessionManager,
        alt,
        asyncio,
        create_action_distribution_chart,
        create_avg_payoff_chart,
        create_cumulative_payoff_chart,
        create_payoff_comparison_chart,
        datetime,
        discover_all_available,
        get_game,
        get_game_names,
        is_burr_game,
        is_discrete_game,
        itertools_product,
        list_games,
        mo,
        pl,
        time,
        SERIES_TIMEOUT,
        MAX_NUM_GAMES,
        detect_num_players,
        # New analytics/experiments
        HyperparameterSensitivityAnalyzer,
        MetaStrategyAnalyzer,
        ModelPersonalityProfiler,
        CrossGameComparativeAnalyzer,
        TournamentConfig,
        TournamentRunner,
        EcosystemSimulator,
        PayoffSensitivityAnalyzer,
        create_quick_tournament,
        # Roles
        RolesTabBuilder,
        RoleConfig,
        RoleRepository,
        # Split Analytics builders
        RoleAnalyticsPanelBuilder,
        GameAnalyticsPanelBuilder,
        # New Role Analytics
        RoleAnalyticsFilterBuilder,
        RoleAnalyticsEmptyStates,
        ImprovedRoleAnalyticsPanelBuilder,
        RoleAnalyticsService,
        RoleFilterParams,
        # Queue imports
        QueuedGame,
        QueueExecutionResult,
        QueueUIBuilder,
    )


@app.cell
def _(mo, Path):
    import sys as _sys
    _apps_dir = Path("/proj/gametheory/ImageFactory/workspace/apps")
    if str(_apps_dir) not in _sys.path:
        _sys.path.insert(0, str(_apps_dir))
    from arena_modules import format_activity_log as _fmt

    def format_activity_log(log_entries):
        """Format activity log as terminal-style output."""
        return _fmt(log_entries, mo)
    return (format_activity_log,)


@app.cell
def _(mo, Path):
    import sys as _sys
    _apps_dir = Path("/proj/gametheory/ImageFactory/workspace/apps")
    if str(_apps_dir) not in _sys.path:
        _sys.path.insert(0, str(_apps_dir))
    from arena_modules import format_prompt_log as _fmt

    def format_prompt_log(prompt_entries):
        """Format prompt log showing full prompts and responses."""
        return _fmt(prompt_entries, mo)
    return (format_prompt_log,)


@app.cell
async def _(DEFAULT_OLLAMA_ENDPOINTS, DEFAULT_OLLAMA_MODELS, DISCOVERY_TIMEOUT, discover_all_available, mo):
    # Discover available Ollama models and endpoints at session start
    mo.output.append(mo.md("_Discovering Ollama endpoints..._"))

    discovered_models, discovered_endpoints, endpoint_models = await discover_all_available(
        endpoints=DEFAULT_OLLAMA_ENDPOINTS,
        timeout=DISCOVERY_TIMEOUT,
    )

    # Use discovered values or fall back to defaults
    if discovered_models:
        available_models = discovered_models
        discovery_status = f"Found {len(discovered_models)} models across {len(discovered_endpoints)} endpoints"
    else:
        available_models = DEFAULT_OLLAMA_MODELS
        discovery_status = "Discovery failed - using default models"

    if discovered_endpoints:
        available_endpoints = discovered_endpoints
    else:
        available_endpoints = DEFAULT_OLLAMA_ENDPOINTS

    return available_endpoints, available_models, discovery_status, endpoint_models


@app.cell
def _(available_endpoints, get_game_names, mo, MAX_NUM_GAMES):
    # Global UI Elements - Endpoint selection (models depend on this)
    game_names = get_game_names()

    # Game type selector - {display_name: game_id} shows names, returns game_id
    game_type_selector = mo.ui.dropdown(
        options={v: k for k, v in game_names.items()},
        label="Select Game Type",
        value="Prisoner's Dilemma",
    )

    # Runtime mode selector
    runtime_selector = mo.ui.dropdown(
        options={
            "One-off": "one_off",
            "Repeated": "repeated",
            "Sequential": "sequential",
            "Multiplayer": "multi_player",
        },
        label="Runtime Mode",
        value="One-off",
    )

    # Number of games
    num_games = mo.ui.slider(1, MAX_NUM_GAMES, value=10, label="Number of Games")

    # Endpoint selectors (models filtered by these in next cell)
    endpoint_p1 = mo.ui.dropdown(
        options=available_endpoints,
        label="Player 1 Endpoint",
        value=available_endpoints[0] if available_endpoints else None,
    )
    endpoint_p2 = mo.ui.dropdown(
        options=available_endpoints,
        label="Player 2 Endpoint",
        value=available_endpoints[1] if len(available_endpoints) > 1 else (available_endpoints[0] if available_endpoints else None),
    )
    endpoint_p3 = mo.ui.dropdown(
        options=available_endpoints,
        label="Player 3 Endpoint",
        value=available_endpoints[2] if len(available_endpoints) > 2 else (available_endpoints[0] if available_endpoints else None),
    )

    # Payoff display in prompt toggle
    payoff_display = mo.ui.dropdown(
        options={"Full Matrix": "full", "Player Only": "player", "None": "none"},
        label="Payoff Info in Prompt",
        value="Full Matrix",
    )

    # Run button
    run_button = mo.ui.run_button(label="Run Game Series")

    return (
        endpoint_p1,
        endpoint_p2,
        endpoint_p3,
        game_names,
        game_type_selector,
        num_games,
        payoff_display,
        run_button,
        runtime_selector,
    )


@app.cell
def _(available_models, endpoint_models, endpoint_p1, endpoint_p2, endpoint_p3, mo):
    # Model selectors - filtered by selected endpoint
    def get_models_for_endpoint(endpoint):
        """Get models available at the selected endpoint."""
        if endpoint and endpoint in endpoint_models:
            return endpoint_models[endpoint]
        return available_models  # Fallback to all models

    # Player 1 model (filtered by endpoint_p1)
    models_p1 = get_models_for_endpoint(endpoint_p1.value)
    model_p1 = mo.ui.dropdown(
        options=models_p1,
        label="Player 1 Model",
        value=models_p1[0] if models_p1 else None,
    )
    # Player 1 advanced settings
    temp_p1 = mo.ui.slider(0.0, 2.0, value=0.7, step=0.1, label="Temperature")
    top_p_p1 = mo.ui.slider(0.0, 1.0, value=0.9, step=0.05, label="Top P")
    top_k_p1 = mo.ui.slider(1, 100, value=40, step=1, label="Top K")
    repeat_penalty_p1 = mo.ui.slider(1.0, 2.0, value=1.1, step=0.05, label="Repeat Penalty")
    system_p1 = mo.ui.text_area(label="System Prompt", placeholder="Optional: Custom instructions for this player...")
    strategy_p1 = mo.ui.text_area(label="Strategy Hints", placeholder="Optional: Override default strategic hints (for Burr games)...")

    # Player 2 model (filtered by endpoint_p2)
    models_p2 = get_models_for_endpoint(endpoint_p2.value)
    model_p2 = mo.ui.dropdown(
        options=models_p2,
        label="Player 2 Model",
        value=models_p2[1] if len(models_p2) > 1 else (models_p2[0] if models_p2 else None),
    )
    # Player 2 advanced settings
    temp_p2 = mo.ui.slider(0.0, 2.0, value=0.7, step=0.1, label="Temperature")
    top_p_p2 = mo.ui.slider(0.0, 1.0, value=0.9, step=0.05, label="Top P")
    top_k_p2 = mo.ui.slider(1, 100, value=40, step=1, label="Top K")
    repeat_penalty_p2 = mo.ui.slider(1.0, 2.0, value=1.1, step=0.05, label="Repeat Penalty")
    system_p2 = mo.ui.text_area(label="System Prompt", placeholder="Optional: Custom instructions for this player...")
    strategy_p2 = mo.ui.text_area(label="Strategy Hints", placeholder="Optional: Override default strategic hints (for Burr games)...")

    # Player 3 model (filtered by endpoint_p3)
    models_p3 = get_models_for_endpoint(endpoint_p3.value)
    model_p3 = mo.ui.dropdown(
        options=models_p3,
        label="Player 3 Model",
        value=models_p3[2] if len(models_p3) > 2 else (models_p3[0] if models_p3 else None),
    )
    # Player 3 advanced settings
    temp_p3 = mo.ui.slider(0.0, 2.0, value=0.7, step=0.1, label="Temperature")
    top_p_p3 = mo.ui.slider(0.0, 1.0, value=0.9, step=0.05, label="Top P")
    top_k_p3 = mo.ui.slider(1, 100, value=40, step=1, label="Top K")
    repeat_penalty_p3 = mo.ui.slider(1.0, 2.0, value=1.1, step=0.05, label="Repeat Penalty")
    system_p3 = mo.ui.text_area(label="System Prompt", placeholder="Optional: Custom instructions for this player...")
    strategy_p3 = mo.ui.text_area(label="Strategy Hints", placeholder="Optional: Override default strategic hints (for Burr games)...")

    return (
        model_p1,
        model_p2,
        model_p3,
        temp_p1,
        temp_p2,
        temp_p3,
        top_p_p1,
        top_p_p2,
        top_p_p3,
        top_k_p1,
        top_k_p2,
        top_k_p3,
        repeat_penalty_p1,
        repeat_penalty_p2,
        repeat_penalty_p3,
        system_p1,
        system_p2,
        system_p3,
        strategy_p1,
        strategy_p2,
        strategy_p3,
    )


@app.cell
def _(MetricsTracker, Path, SessionManager, RoleRepository):
    # Initialize shared state
    metrics = MetricsTracker()

    # Use absolute path - __file__ is unreliable in marimo (points to /tmp/)
    _data_dir = Path("/proj/gametheory/ImageFactory/workspace/data")
    session_manager = SessionManager(str(_data_dir / "sessions"))

    # Initialize role repository
    role_repository = RoleRepository(str(_data_dir / "arena.duckdb"))

    return metrics, session_manager, role_repository


@app.cell
def _(mo):
    # Roles tab state management
    get_roles_refresh, set_roles_refresh = mo.state(0)  # Counter to trigger refresh
    get_editing_role, set_editing_role = mo.state(None)  # Role being edited (or None)
    get_role_form_visible, set_role_form_visible = mo.state(False)  # Show/hide form

    return (
        get_roles_refresh,
        set_roles_refresh,
        get_editing_role,
        set_editing_role,
        get_role_form_visible,
        set_role_form_visible,
    )


@app.cell
def _(mo):
    # Game queue state management
    get_game_queue, set_game_queue = mo.state([])  # List[QueuedGame]
    get_queue_executing, set_queue_executing = mo.state(False)
    get_current_queue_idx, set_current_queue_idx = mo.state(-1)
    get_queue_results, set_queue_results = mo.state({})

    return (
        get_game_queue,
        set_game_queue,
        get_queue_executing,
        set_queue_executing,
        get_current_queue_idx,
        set_current_queue_idx,
        get_queue_results,
        set_queue_results,
    )


@app.cell
def _(
    role_repository,
    get_roles_refresh,
    get_editing_role,
    get_role_form_visible,
    set_roles_refresh,
    set_editing_role,
    set_role_form_visible,
    available_endpoints,
    endpoint_models,
    game_names,
    GAME_REGISTRY,
    mo,
    RolesTabBuilder,
    RoleConfig,
    analytics_service,
):
    # Trigger refresh on state change
    _ = get_roles_refresh()

    # Load all roles
    all_roles = role_repository.list_all()

    # Get cross-game model stats (independent of game filter)
    model_stats = analytics_service.get_model_stats_dict()

    # Build roles table with performance stats
    roles_table = RolesTabBuilder.build_roles_table(all_roles, model_stats=model_stats)

    # Action buttons
    create_role_btn = mo.ui.button(
        label="Create New Role",
        on_click=lambda _: (set_role_form_visible(True), set_editing_role(None)),
    )
    refresh_roles_btn = mo.ui.button(
        label="Refresh",
        on_click=lambda _: set_roles_refresh(get_roles_refresh() + 1),
    )

    # Build game_descriptions from registry for default strategy text
    game_descriptions = {game_id: game.description for game_id, game in GAME_REGISTRY.items()}

    # Build game_prompt_instructions from registry - shows what LLM receives
    game_prompt_instructions = {
        game_id: game.action_space.prompt_instructions()
        for game_id, game in GAME_REGISTRY.items()
    }

    # Build form elements
    editing_role = get_editing_role()
    form_elements = RolesTabBuilder.build_role_form(
        available_endpoints=available_endpoints,
        endpoint_models=endpoint_models,
        game_names=game_names,
        game_descriptions=game_descriptions,
        game_prompt_instructions=game_prompt_instructions,
        existing_role=editing_role,
    )

    # Return multiselect separately for proper marimo reactivity
    allowed_games_selector = form_elements["allowed_games"]

    return (
        all_roles,
        roles_table,
        create_role_btn,
        refresh_roles_btn,
        form_elements,
        game_descriptions,
        game_prompt_instructions,
        allowed_games_selector,
    )


@app.cell
def _(form_elements, RolesTabBuilder, allowed_games_selector, game_names):
    # Build form layout reactively - updates when selected games change
    # Depend on allowed_games_selector directly to trigger re-run on selection change
    selected_games = list(allowed_games_selector.value or [])

    # Build layout with game instruction text areas for selected games
    # Pass game_names for reverse mapping (multiselect returns display names, not game_ids)
    form_layout = RolesTabBuilder.build_form_layout(
        form_elements=form_elements,
        selected_games=selected_games,
        game_names=game_names,
    )

    return (form_layout,)


@app.cell
def _(
    roles_table,
    all_roles,
    game_names,
    set_editing_role,
    set_role_form_visible,
    mo,
    RolesTabBuilder,
):
    # Selected role details (when a role is selected in the table)
    selected_role = None
    role_detail_card = mo.md("")

    if hasattr(roles_table, 'value') and roles_table.value and len(roles_table.value) > 0:
        selected_row = roles_table.value[0]
        selected_role_id = selected_row.get('_role_id') if isinstance(selected_row, dict) else None
        if selected_role_id:
            selected_role = next((r for r in all_roles if r.role_id == selected_role_id), None)
            if selected_role:
                role_detail_card = RolesTabBuilder.build_role_detail_card(selected_role, game_names)

    # Edit/Delete buttons for selected role
    edit_role_btn = mo.ui.button(
        label="Edit",
        on_click=lambda _: (set_editing_role(selected_role), set_role_form_visible(True)) if selected_role else None,
        disabled=selected_role is None,
    )

    return selected_role, role_detail_card, edit_role_btn


@app.cell
def _(
    selected_role,
    role_repository,
    set_roles_refresh,
    get_roles_refresh,
    mo,
):
    # Delete role button with confirmation
    def handle_delete(_):
        if selected_role:
            role_repository.delete(selected_role.role_id)
            set_roles_refresh(get_roles_refresh() + 1)

    delete_role_btn = mo.ui.button(
        label="Delete",
        on_click=handle_delete,
        disabled=selected_role is None,
    )

    return (delete_role_btn,)


@app.cell
def _(
    form_elements,
    get_editing_role,
    get_role_form_visible,
    set_role_form_visible,
    set_roles_refresh,
    get_roles_refresh,
    role_repository,
    mo,
    RolesTabBuilder,
    RoleConfig,
):
    # Form submission handling - use list to allow mutation in nested function
    _form_messages = {"error": "", "success": ""}

    def handle_save(_):
        editing = get_editing_role()
        role, errors = RolesTabBuilder.extract_role_from_form(
            form_elements,
            existing_role_id=editing.role_id if editing else None,
        )
        if errors:
            _form_messages["error"] = "; ".join(errors)
            return

        try:
            if editing:
                role_repository.update(role)
                _form_messages["success"] = f"Role '{role.name}' updated successfully!"
            else:
                role_repository.create(role)
                _form_messages["success"] = f"Role '{role.name}' created successfully!"
            set_role_form_visible(False)
            set_roles_refresh(get_roles_refresh() + 1)
        except ValueError as e:
            _form_messages["error"] = str(e)

    save_role_btn = mo.ui.button(label="Save Role", on_click=handle_save)
    cancel_form_btn = mo.ui.button(
        label="Cancel",
        on_click=lambda _: set_role_form_visible(False),
    )

    return save_role_btn, cancel_form_btn


@app.cell
def _(
    create_role_btn,
    refresh_roles_btn,
    roles_table,
    role_detail_card,
    edit_role_btn,
    delete_role_btn,
    form_layout,
    save_role_btn,
    cancel_form_btn,
    get_role_form_visible,
    get_editing_role,
    selected_role,
    mo,
):
    # Build the complete Roles tab content
    roles_header = mo.hstack([create_role_btn, refresh_roles_btn], justify="start")

    # Role detail section (shown when a role is selected)
    role_detail_section = mo.vstack([
        mo.md("---"),
        mo.md("### Selected Role"),
        role_detail_card,
        mo.hstack([edit_role_btn, delete_role_btn], justify="start"),
    ]) if selected_role else mo.md("")

    # Form section (shown when creating/editing)
    is_editing = get_editing_role() is not None
    form_title = f"### {'Edit' if is_editing else 'Create'} Role"
    form_section = mo.vstack([
        mo.md("---"),
        mo.md(form_title),
        form_layout,
        mo.hstack([save_role_btn, cancel_form_btn], justify="start"),
    ]) if get_role_form_visible() else mo.md("")

    roles_tab_content = mo.vstack([
        mo.md("# Role Management"),
        mo.md("_Create and manage player identities with predefined configurations._"),
        roles_header,
        roles_table,
        role_detail_section,
        form_section,
    ])

    return (roles_tab_content,)


@app.cell
def _(all_roles, game_type_selector, mo, RolesTabBuilder):
    # Role selectors for Play tab
    selected_game_id = game_type_selector.value

    # Build role dropdowns for each player
    role_p1 = RolesTabBuilder.build_player_role_selector(
        player_num=1,
        roles=all_roles,
        game_id=selected_game_id,
    )
    role_p2 = RolesTabBuilder.build_player_role_selector(
        player_num=2,
        roles=all_roles,
        game_id=selected_game_id,
    )
    role_p3 = RolesTabBuilder.build_player_role_selector(
        player_num=3,
        roles=all_roles,
        game_id=selected_game_id,
    )

    return role_p1, role_p2, role_p3


@app.cell
def _(role_p1, role_p2, role_p3, role_repository, mo):
    # Helper to get RoleConfig from dropdown value
    def get_role_for_player(role_dropdown):
        """Get RoleConfig from role dropdown value, or None if manual config."""
        role_id = role_dropdown.value
        if not role_id:  # Empty string means manual config
            return None
        return role_repository.get_by_id(role_id)

    # Get currently selected roles (or None if manual config)
    selected_role_p1 = get_role_for_player(role_p1)
    selected_role_p2 = get_role_for_player(role_p2)
    selected_role_p3 = get_role_for_player(role_p3)

    return selected_role_p1, selected_role_p2, selected_role_p3, get_role_for_player


@app.cell
def _(mo):
    # Queue action buttons - use run_button pattern (like Run Game Series)
    add_to_queue_btn = mo.ui.run_button(label="Add to Queue")
    clear_queue_btn = mo.ui.run_button(label="Clear Queue")
    execute_queue_btn = mo.ui.run_button(label="Execute All")

    return add_to_queue_btn, clear_queue_btn, execute_queue_btn


@app.cell
def _(
    mo,
    add_to_queue_btn,
    get_game_queue,
    set_game_queue,
    game_type_selector,
    runtime_selector,
    num_games,
    payoff_display,
    game_names,
    get_custom_payoffs,
    QueuedGame,
    get_game,
    # Player 1 settings
    model_p1,
    endpoint_p1,
    temp_p1,
    top_p_p1,
    top_k_p1,
    repeat_penalty_p1,
    system_p1,
    strategy_p1,
    selected_role_p1,
    # Player 2 settings
    model_p2,
    endpoint_p2,
    temp_p2,
    top_p_p2,
    top_k_p2,
    repeat_penalty_p2,
    system_p2,
    strategy_p2,
    selected_role_p2,
    # Player 3 settings
    model_p3,
    endpoint_p3,
    temp_p3,
    top_p_p3,
    top_k_p3,
    repeat_penalty_p3,
    system_p3,
    strategy_p3,
    selected_role_p3,
):
    # Handle Add to Queue - only runs when button clicked
    mo.stop(not add_to_queue_btn.value)

    _game_id = game_type_selector.value

    # Get game to determine num_players
    try:
        _game = get_game(_game_id)
        _num_players = _game.num_players
    except KeyError:
        _num_players = 2  # Default fallback

    # Build player settings (same pattern as Run Game Series)
    def _get_settings(role, model, endpoint, temp, top_p, top_k, repeat, system, strategy):
        if role:
            return {
                "model": role.model,
                "endpoint": role.endpoint,
                "temperature": role.temperature,
                "top_p": role.top_p,
                "top_k": role.top_k,
                "repeat_penalty": role.repeat_penalty,
                "system_prompt": role.system_prompt,
                "strategy_hints": strategy.value if strategy.value else role.get_game_instructions(_game_id),
                "role_name": role.name,
            }
        return {
            "model": model.value,
            "endpoint": endpoint.value,
            "temperature": temp.value,
            "top_p": top_p.value,
            "top_k": top_k.value,
            "repeat_penalty": repeat.value,
            "system_prompt": system.value if system.value else None,
            "strategy_hints": strategy.value if strategy.value else None,
            "role_name": None,
        }

    _all_settings = [
        _get_settings(selected_role_p1, model_p1, endpoint_p1, temp_p1, top_p_p1, top_k_p1, repeat_penalty_p1, system_p1, strategy_p1),
        _get_settings(selected_role_p2, model_p2, endpoint_p2, temp_p2, top_p_p2, top_k_p2, repeat_penalty_p2, system_p2, strategy_p2),
        _get_settings(selected_role_p3, model_p3, endpoint_p3, temp_p3, top_p_p3, top_k_p3, repeat_penalty_p3, system_p3, strategy_p3),
    ]

    new_game = QueuedGame(
        game_type=_game_id,
        game_name=game_names.get(_game_id, _game_id),
        runtime_mode=runtime_selector.value,
        num_games=num_games.value,
        payoff_display=payoff_display.value,
        players=_all_settings[:_num_players],  # Capture player configs!
        custom_payoffs=get_custom_payoffs().get(_game_id, {}),
    )
    set_game_queue(get_game_queue() + [new_game])
    return ()


@app.cell
def _(mo, clear_queue_btn, set_game_queue, set_queue_results):
    # Handle Clear Queue - only runs when button clicked
    mo.stop(not clear_queue_btn.value)

    set_game_queue([])
    set_queue_results({})
    return ()


@app.cell
async def _(
    mo,
    execute_queue_btn,
    get_game_queue,
    set_game_queue,
    set_queue_results,
    get_game,
    BurrGameRunner,
    PlayerConfig,
    time,
):
    # Handle Execute All - only runs when button clicked
    mo.stop(not execute_queue_btn.value)

    _exec_queue = get_game_queue()
    mo.stop(not _exec_queue)  # Stop if queue is empty

    _results = {}
    _updated_queue = []

    # Run games with progress bar (matching existing pattern from Run Game Series)
    import aiohttp as _aiohttp

    _connector = _aiohttp.TCPConnector(limit=20, keepalive_timeout=30)
    async with _aiohttp.ClientSession(connector=_connector) as _session:
        for _queued_game in mo.status.progress_bar(
            _exec_queue,
            title="Executing Queue",
            subtitle=f"{len(_exec_queue)} games queued"
        ):
            _queued_game.status = "running"

            # Get game definition using get_game()
            try:
                _game = get_game(_queued_game.game_type)
            except KeyError:
                _queued_game.status = "failed"
                _queued_game.error_message = f"Unknown game: {_queued_game.game_type}"
                _updated_queue.append(_queued_game)
                continue

            # Build PlayerConfig list from queued player settings
            _players = [
                PlayerConfig(
                    player_id=_pn + 1,
                    model=_queued_game.players[_pn].get("model", "llama3.2"),
                    endpoint=_queued_game.players[_pn].get("endpoint", "http://localhost:11434"),
                    temperature=_queued_game.players[_pn].get("temperature", 0.7),
                    top_p=_queued_game.players[_pn].get("top_p", 0.9),
                    top_k=_queued_game.players[_pn].get("top_k", 40),
                    repeat_penalty=_queued_game.players[_pn].get("repeat_penalty", 1.1),
                    system_prompt=_queued_game.players[_pn].get("system_prompt"),
                    strategy_hints=_queued_game.players[_pn].get("strategy_hints"),
                )
                for _pn in range(_game.num_players)
            ]

            # Create runner with tracking enabled and use run_series for Burr tracking
            _runner = BurrGameRunner(_game, enable_tracking=True)
            _start = time.time()

            try:
                # Run using run_series() which uses Burr state machine with tracking
                _results_list = await _runner.run_series(
                    _players,
                    _queued_game.num_games,
                    session=_session,
                    runtime_mode=_queued_game.runtime_mode,
                    payoff_display=_queued_game.payoff_display,
                )

                _queued_game.status = "completed"
                _queued_game.session_id = _game.id[:8]
                _results[_queued_game.queue_id] = {
                    "rounds": len(_results_list),
                    "elapsed": time.time() - _start,
                    "session_id": _queued_game.session_id,
                }
            except Exception as _e:
                _queued_game.status = "failed"
                _queued_game.error_message = str(_e)

            _updated_queue.append(_queued_game)

    set_game_queue(_updated_queue)
    set_queue_results(_results)

    # Show confirmation
    _completed = sum(1 for g in _updated_queue if g.status == "completed")
    _failed = sum(1 for g in _updated_queue if g.status == "failed")
    mo.output.replace(mo.callout(
        mo.md(f"**Queue executed:** {_completed} completed, {_failed} failed"),
        kind="success" if _failed == 0 else "warn"
    ))


@app.cell
def _(
    mo,
    QueueUIBuilder,
    get_game_queue,
    get_queue_executing,
    get_current_queue_idx,
    get_queue_results,
    add_to_queue_btn,
    execute_queue_btn,
    clear_queue_btn,
):
    # Build queue display UI
    _queue = get_game_queue()
    _is_executing = get_queue_executing()
    _current_idx = get_current_queue_idx()
    _queue_results = get_queue_results()

    queue_table = QueueUIBuilder.build_queue_table(_queue, _current_idx, mo)
    _progress_display = QueueUIBuilder.build_execution_progress(_queue, _current_idx, _is_executing, mo)
    _results_summary = QueueUIBuilder.build_queue_results_summary(_queue, _queue_results, mo)

    queue_section = mo.vstack([
        mo.md("### Game Queue"),
        mo.hstack([add_to_queue_btn, execute_queue_btn, clear_queue_btn], justify="start"),
        mo.md(f"**{len(_queue)} game(s) in queue**") if _queue else mo.md(""),
        queue_table,
        _progress_display,
        _results_summary,
    ])

    return (
        queue_table,
        queue_section,
    )


@app.cell
def _(
    endpoint_p1,
    endpoint_p2,
    endpoint_p3,
    get_game,
    game_type_selector,
    mo,
    model_p1,
    model_p2,
    model_p3,
    num_games,
    payoff_display,
    run_button,
    runtime_selector,
    temp_p1,
    temp_p2,
    temp_p3,
    top_p_p1,
    top_p_p2,
    top_p_p3,
    top_k_p1,
    top_k_p2,
    top_k_p3,
    repeat_penalty_p1,
    repeat_penalty_p2,
    repeat_penalty_p3,
    system_p1,
    system_p2,
    system_p3,
    strategy_p1,
    strategy_p2,
    strategy_p3,
    role_p1,
    role_p2,
    role_p3,
    selected_role_p1,
    selected_role_p2,
    selected_role_p3,
):
    # Build configuration UI
    def _build_config_ui():
        # Helper to build player section with role or manual config
        def build_player_section(player_num, role_dropdown, selected_role,
                                  model_dd, endpoint_dd,
                                  temp_slider, top_p_slider, top_k_slider, repeat_slider,
                                  system_ta, strategy_ta):
            # Manual config accordion
            manual_advanced = mo.accordion({
                "Advanced Settings": mo.vstack([
                    mo.hstack([temp_slider, top_p_slider]),
                    mo.hstack([top_k_slider, repeat_slider]),
                    system_ta,
                    strategy_ta,
                ])
            })
            manual_config = mo.vstack([
                mo.hstack([model_dd, endpoint_dd]),
                manual_advanced,
            ])

            # Role summary (when role is selected)
            if selected_role:
                role_summary = mo.callout(
                    mo.md(f"**{selected_role.name}**  \n{selected_role.model} @ {selected_role.endpoint.split('/')[-1]}"),
                    kind="success",
                )
                # Only show strategy hints override for roles
                role_config = mo.vstack([
                    role_summary,
                    mo.accordion({
                        "Session Override": mo.vstack([strategy_ta]),
                    }),
                ])
                player_content = role_config
            else:
                player_content = manual_config

            return mo.vstack([
                mo.md(f"**Player {player_num}**"),
                role_dropdown,
                player_content,
            ])

        player1_section = build_player_section(
            1, role_p1, selected_role_p1,
            model_p1, endpoint_p1,
            temp_p1, top_p_p1, top_k_p1, repeat_penalty_p1,
            system_p1, strategy_p1
        )
        player2_section = build_player_section(
            2, role_p2, selected_role_p2,
            model_p2, endpoint_p2,
            temp_p2, top_p_p2, top_k_p2, repeat_penalty_p2,
            system_p2, strategy_p2
        )

        rows = [
            mo.md("## Configuration"),
            mo.hstack([game_type_selector, runtime_selector, payoff_display]),
            num_games,
            mo.md("### Players"),
            mo.md("_Select a role or use manual configuration for each player._"),
            player1_section,
            player2_section,
        ]

        # Show Player 3 for multiplayer mode OR when selected game has 3+ players
        try:
            selected_game = get_game(game_type_selector.value)
        except KeyError:
            selected_game = None
        game_needs_3_players = selected_game and selected_game.num_players >= 3
        if runtime_selector.value == "multi_player" or game_needs_3_players:
            player3_section = build_player_section(
                3, role_p3, selected_role_p3,
                model_p3, endpoint_p3,
                temp_p3, top_p_p3, top_k_p3, repeat_penalty_p3,
                system_p3, strategy_p3
            )
            rows.append(player3_section)

        rows.append(run_button)
        return mo.vstack(rows)

    config_ui = _build_config_ui()
    return (config_ui,)


# Legacy cell for Player 3 - now handled inline
@app.cell
def _(get_game, game_type_selector, runtime_selector):
    # Determine if Player 3 is needed
    try:
        _selected_game = get_game(game_type_selector.value)
    except KeyError:
        _selected_game = None
    game_needs_3_players = _selected_game and _selected_game.num_players >= 3
    show_player3 = runtime_selector.value == "multi_player" or game_needs_3_players
    return (show_player3,)


@app.cell
def _(active_game, get_custom_payoffs, has_custom_payoffs, custom_payoffs_changes, game_type_selector, mo, pl, is_burr_game):
    # Display current game info - depends on get_custom_payoffs for reactivity when payoffs change
    def _get_game_info():
        game = active_game
        game_id = game_type_selector.value

        if not game:
            return mo.md("Select a game to see details.")

        # Build elements list
        elements = [mo.md(f"### {game.name}")]

        # Check if this is a Burr game by checking the actual object
        # This is more robust than string-based is_burr_game(game_id) check
        if not hasattr(game, 'payoff_matrix'):
            # Burr games have action_space instead of payoff_matrix
            action_space_type = type(game.action_space).__name__
            elements.extend([
                mo.md(game.description),
                mo.md(f"**Players:** {game.num_players} | **Action Space:** {action_space_type}"),
                mo.md("#### Action Space Details"),
                mo.md(f"```\n{game.action_space.prompt_instructions()}\n```"),
                mo.callout(
                    mo.md("*Burr engine game - payoffs are computed dynamically based on player actions.*"),
                    kind="info"
                ),
            ])
            return mo.vstack(elements)

        # Discrete game - build payoff matrix display
        if game.num_players == 2:
            rows = [
                {
                    "P1 Action": a1,
                    "P2 Action": a2,
                    "P1 Payoff": p1,
                    "P2 Payoff": p2,
                }
                for (a1, a2), (p1, p2) in game.payoff_matrix.items()
            ]
        else:
            rows = []
            for actions, payoffs in game.payoff_matrix.items():
                row = {}
                for i, action in enumerate(actions):
                    row[f"P{i+1} Action"] = action
                for i, payoff in enumerate(payoffs):
                    row[f"P{i+1} Payoff"] = payoff
                rows.append(row)

        matrix_df = pl.DataFrame(rows)

        # Add custom payoffs badge if active
        if has_custom_payoffs:
            elements.append(
                mo.callout(
                    mo.md("**Custom Payoffs Active** - Using modified payoff values"),
                    kind="warn",
                )
            )

            # Show what changed
            if custom_payoffs_changes:
                changes_text = ", ".join([
                    f"{c['action']} P{c['player']}: {c['default']}->{c['custom']}"
                    for c in custom_payoffs_changes
                ])
                elements.append(mo.md(f"_Changes: {changes_text}_"))

        # game.actions only exists for discrete games (defensive check)
        actions_str = ', '.join(game.actions) if hasattr(game, 'actions') else "N/A"
        elements.extend([
            mo.md(game.description),
            mo.md(f"**Players:** {game.num_players} | **Actions:** {actions_str}"),
            mo.md("#### Payoff Matrix"),
            matrix_df,
        ])

        return mo.vstack(elements)

    game_info = _get_game_info()
    return (game_info,)


@app.cell
def _(GAME_REGISTRY, GameDefinition, get_custom_payoffs, set_custom_payoffs, game_type_selector, mo, is_burr_game, get_game):
    # Custom Payoff Editor with reactive state
    game_id = game_type_selector.value

    # Check if this is a Burr game (Colonel Blotto, Tennis Coach, Sumo Coach)
    if is_burr_game(game_id):
        # Burr games don't have editable payoff matrices
        burr_game = get_game(game_id)
        active_game = burr_game
        payoff_editor_ui = mo.vstack([
            mo.md(f"**{burr_game.name}** *(Burr Engine)*"),
            mo.md(burr_game.description),
            mo.md("---"),
            mo.md(f"**Action Space:** `{type(burr_game.action_space).__name__}`"),
            mo.md(burr_game.action_space.prompt_instructions()),
            mo.callout(
                mo.md("*Payoff editing is not available for allocation/permutation games.*"),
                kind="info"
            ),
        ])
        payoff_inputs = {}
        has_custom_payoffs = False
        custom_payoffs_changes = []
    else:
        # Discrete game - use GAME_REGISTRY
        base_game = GAME_REGISTRY.get(game_id)

        if not base_game:
            payoff_editor_ui = mo.md("Select a game to customize payoffs.")
            active_game = None
            payoff_inputs = {}
            has_custom_payoffs = False
            custom_payoffs_changes = []
        else:
            # Check if we have custom payoffs for this game (using reactive getter)
            all_custom_payoffs = get_custom_payoffs()
            current_custom_payoffs = all_custom_payoffs.get(game_id, {})
            has_custom_payoffs = bool(current_custom_payoffs)

            # Build payoff inputs and track changes
            payoff_inputs = {}
            custom_payoffs_changes = []

            for actions, payoffs in base_game.payoff_matrix.items():
                action_key = "_".join(actions)
                for i, default_val in enumerate(payoffs):
                    input_key = f"{action_key}_p{i+1}"
                    current_val = current_custom_payoffs.get(input_key, default_val)

                    # Track changes for visual indicator
                    if current_val != default_val:
                        custom_payoffs_changes.append({
                            "action": " vs ".join(actions),
                            "player": i + 1,
                            "default": default_val,
                            "custom": current_val,
                        })

                    payoff_inputs[input_key] = mo.ui.number(
                        value=current_val,
                        label=f"P{i+1}",
                        start=-100,
                        stop=100,
                    )

            # Group inputs by action combination
            payoff_rows = []
            for actions in base_game.payoff_matrix.keys():
                action_key = "_".join(actions)
                action_label = " vs ".join(actions)
                row_inputs = [payoff_inputs[f"{action_key}_p{_pi+1}"] for _pi in range(base_game.num_players)]
                payoff_rows.append(mo.hstack([mo.md(f"**{action_label}:**")] + row_inputs))

            def _apply_custom_payoffs():
                new_payoffs = {}
                for key, inp in payoff_inputs.items():
                    new_payoffs[key] = inp.value
                # Use reactive setter - triggers dependent cell updates
                current = get_custom_payoffs()
                set_custom_payoffs({**current, game_id: new_payoffs})

            apply_button = mo.ui.button(label="Apply Custom Payoffs", on_click=lambda _: _apply_custom_payoffs())

            def _reset_payoffs():
                current = get_custom_payoffs()
                # Remove this game's custom payoffs using reactive setter
                set_custom_payoffs({k: v for k, v in current.items() if k != game_id})

            reset_button = mo.ui.button(label="Reset to Default", on_click=lambda _: _reset_payoffs())

            payoff_editor_ui = mo.accordion({
                "Custom Payoffs": mo.vstack([
                    mo.md("Edit payoff values for each action combination:"),
                    *payoff_rows,
                    mo.hstack([apply_button, reset_button]),
                ])
            })

            # Build active game with custom payoffs if any
            if current_custom_payoffs:
                new_matrix = {}
                for actions, payoffs in base_game.payoff_matrix.items():
                    action_key = "_".join(actions)
                    new_matrix[actions] = tuple(
                        current_custom_payoffs.get(f"{action_key}_p{_pidx+1}", payoffs[_pidx])
                        for _pidx in range(base_game.num_players)
                    )
                active_game = GameDefinition(
                    id=base_game.id,
                    name=base_game.name + " (Custom)",
                    description=base_game.description,
                    payoff_matrix=new_matrix,
                    actions=base_game.actions,
                    num_players=base_game.num_players,
                    is_sequential=base_game.is_sequential,
                    memory_depth=base_game.memory_depth,
                )
            else:
                active_game = base_game

    return active_game, payoff_editor_ui, payoff_inputs, has_custom_payoffs, custom_payoffs_changes


@app.cell
def _(GameDefinition, session_custom_games, mo, Path):
    import sys as _sys
    _apps_dir = Path("/proj/gametheory/ImageFactory/workspace/apps")
    if str(_apps_dir) not in _sys.path:
        _sys.path.insert(0, str(_apps_dir))
    from arena_modules import parse_actions_list, create_custom_game as _create_custom_game

    # Custom Game Creator
    custom_game_name = mo.ui.text(label="Game Name", placeholder="My Custom Game")
    custom_game_actions = mo.ui.text(label="Actions (comma-separated)", placeholder="cooperate, defect")
    custom_game_num_players = mo.ui.dropdown(
        options={"2 Players": 2, "3 Players": 3},
        label="Number of Players",
        value="2 Players",
    )

    def _add_custom_game(_):
        name = custom_game_name.value or "Custom Game"
        actions = parse_actions_list(custom_game_actions.value or "")
        n_players = custom_game_num_players.value
        game = _create_custom_game(name, actions, n_players, GameDefinition)
        if game:
            session_custom_games[game.id] = game

    create_game_button = mo.ui.button(label="Create Custom Game", on_click=_add_custom_game)

    # List current custom games
    custom_game_list = list(session_custom_games.keys())

    def _remove_custom_game(game_id):
        if game_id in session_custom_games:
            del session_custom_games[game_id]

    custom_game_creator_ui = mo.accordion({
        "Create Custom Game": mo.vstack([
            mo.md("Create a new game with custom actions:"),
            custom_game_name,
            custom_game_actions,
            custom_game_num_players,
            create_game_button,
            mo.md(f"**Custom games in session:** {', '.join(custom_game_list) if custom_game_list else 'None'}"),
        ])
    })

    return custom_game_creator_ui, custom_game_name, custom_game_actions, custom_game_num_players


@app.cell
async def _(
    BurrGameRunner,
    GameRunner,
    PlayerConfig,
    SERIES_TIMEOUT,
    active_game,
    activity_log,
    asyncio,
    create_action_distribution_chart,
    create_avg_payoff_chart,
    create_cumulative_payoff_chart,
    create_payoff_comparison_chart,
    datetime,
    endpoint_p1,
    endpoint_p2,
    endpoint_p3,
    format_activity_log,
    format_prompt_log,
    game_type_selector,
    get_custom_payoffs,
    has_custom_payoffs,
    metrics,
    mo,
    model_p1,
    model_p2,
    model_p3,
    num_games,
    payoff_display,
    pl,
    run_button,
    runtime_selector,
    session_manager,
    temp_p1,
    temp_p2,
    temp_p3,
    top_p_p1,
    top_p_p2,
    top_p_p3,
    top_k_p1,
    top_k_p2,
    top_k_p3,
    repeat_penalty_p1,
    repeat_penalty_p2,
    repeat_penalty_p3,
    system_p1,
    system_p2,
    system_p3,
    strategy_p1,
    strategy_p2,
    strategy_p3,
    time,
    selected_role_p1,
    selected_role_p2,
    selected_role_p3,
):
    # Game execution cell - N-player support
    mo.stop(not run_button.value)

    game = active_game  # Get game early
    # All games now use BurrGameDefinition - check if it's discrete or allocation
    is_discrete = game.is_discrete if hasattr(game, 'is_discrete') else False

    # Helper to format actions for display
    def format_action(action, game):
        """Format action for display based on game type."""
        if game.is_discrete:
            return str(action)  # Discrete actions are strings
        elif hasattr(game.action_space, 'num_fields'):  # AllocationSpace (Blotto)
            return f"[{', '.join(f'{a:.0f}' for a in action)}]"
        return str(action)  # Permutation or other

    # Clear and initialize logs
    activity_log.clear()
    prompt_log = []  # Store prompts separately
    activity_log.append({
        "timestamp": datetime.now().isoformat(),
        "level": "INFO",
        "message": f"Starting game series..."
    })

    # Soft reset metrics (keeps cumulative totals)
    metrics.soft_reset()

    # game already set above (for Burr check), includes custom payoffs
    num_players = game.num_players

    activity_log.append({
        "timestamp": datetime.now().isoformat(),
        "level": "INFO",
        "message": f"Game: {game.name} | Players: {num_players} | Rounds: {num_games.value}"
    })

    # Helper to build player settings from role or manual config
    def get_player_settings(role, model_dd, endpoint_dd, temp_sl, top_p_sl, top_k_sl, repeat_sl, system_ta, strategy_ta, game_id):
        """Get player settings from role or manual configuration."""
        if role:
            # Use role settings
            # Priority for strategy_hints: session override > game-specific instructions > None
            strategy_hints = strategy_ta.value if strategy_ta.value else role.get_game_instructions(game_id) or None
            return {
                "model": role.model,
                "endpoint": role.endpoint,
                "temperature": role.temperature,
                "top_p": role.top_p,
                "top_k": role.top_k,
                "repeat_penalty": role.repeat_penalty,
                "system_prompt": role.system_prompt or None,
                "strategy_hints": strategy_hints,
                "role_id": role.role_id,
                "role_name": role.name,
            }
        else:
            # Use manual configuration
            return {
                "model": model_dd.value,
                "endpoint": endpoint_dd.value,
                "temperature": temp_sl.value,
                "top_p": top_p_sl.value,
                "top_k": top_k_sl.value,
                "repeat_penalty": repeat_sl.value,
                "system_prompt": system_ta.value if system_ta.value else None,
                "strategy_hints": strategy_ta.value if strategy_ta.value else None,
                "role_id": None,
                "role_name": None,
            }

    # Selected roles (can be None for manual config)
    selected_roles = [selected_role_p1, selected_role_p2, selected_role_p3]
    current_game_id = game_type_selector.value

    # Gather all player settings into a list
    all_player_settings = [
        get_player_settings(selected_roles[0], model_p1, endpoint_p1, temp_p1, top_p_p1, top_k_p1, repeat_penalty_p1, system_p1, strategy_p1, current_game_id),
        get_player_settings(selected_roles[1], model_p2, endpoint_p2, temp_p2, top_p_p2, top_k_p2, repeat_penalty_p2, system_p2, strategy_p2, current_game_id),
        get_player_settings(selected_roles[2], model_p3, endpoint_p3, temp_p3, top_p_p3, top_k_p3, repeat_penalty_p3, system_p3, strategy_p3, current_game_id),
    ]

    # Create players based on game.num_players
    players = [
        PlayerConfig(
            player_id=_pn + 1,
            model=all_player_settings[_pn]["model"],
            endpoint=all_player_settings[_pn]["endpoint"],
            temperature=all_player_settings[_pn]["temperature"],
            top_p=all_player_settings[_pn]["top_p"],
            top_k=all_player_settings[_pn]["top_k"],
            repeat_penalty=all_player_settings[_pn]["repeat_penalty"],
            system_prompt=all_player_settings[_pn]["system_prompt"],
            strategy_hints=all_player_settings[_pn]["strategy_hints"],
        )
        for _pn in range(num_players)
    ]

    # Log player configurations with role/system prompt status
    for _pi, player in enumerate(players):
        role_info = f"[Role: {all_player_settings[_pi]['role_name']}]" if all_player_settings[_pi].get('role_name') else "[Manual]"
        activity_log.append({
            "timestamp": datetime.now().isoformat(),
            "level": "INFO",
            "message": f"P{_pi+1}: {player.model} @ {player.endpoint} {role_info} (temp={player.temperature}, sys_prompt={'YES' if player.system_prompt else 'NO'})"
        })

    # Create session
    session = session_manager.create_session(
        game_type=game_type_selector.value,
        players=players,
        num_rounds=num_games.value,
    )

    # Store custom payoffs, player settings, and full payoff matrix in session config
    player_settings = {f"p{_idx+1}": {"temperature": all_player_settings[_idx]["temperature"],
                                    "top_p": all_player_settings[_idx]["top_p"]}
                       for _idx in range(num_players)}

    # Handle Burr games (allocation/permutation) vs discrete games
    if hasattr(game, 'payoff_matrix'):
        # Discrete game - serialize the payoff matrix for storage
        serialized_matrix = {
            "_".join(actions): list(payoffs)
            for actions, payoffs in game.payoff_matrix.items()
        }
        game_actions = game.actions
    else:
        # Burr game - no payoff matrix, store action space info instead
        serialized_matrix = None
        game_actions = f"ActionSpace:{type(game.action_space).__name__}"

    session.config = {
        "custom_payoffs": get_custom_payoffs().get(game_type_selector.value, {}),
        "player_settings": player_settings,
        "uses_custom_payoffs": has_custom_payoffs,
        "payoff_matrix": serialized_matrix,
        "game_actions": game_actions,
        "game_name": game.name,
        "num_players": game.num_players,
        "is_discrete_game": is_discrete,
    }

    activity_log.append({
        "timestamp": datetime.now().isoformat(),
        "level": "INFO",
        "message": f"Session {session.session_id} started"
    })

    # Run game series - all games now use BurrGameRunner with Burr tracking
    runner = BurrGameRunner(game, enable_tracking=True)

    start_time = time.time()

    # Create callbacks for logging during game execution
    class ArenaCallbacks:
        def on_activity(self, entry):
            activity_log.append(entry)

        def on_prompt(self, entry):
            prompt_log.append(entry)

        def on_round_complete(self, result):
            pass  # Results collected by run_series

    callbacks = ArenaCallbacks()
    metrics.log_request_start(num_games.value * num_players)

    # Execute using run_series with Burr state machine tracking
    import aiohttp
    connector = aiohttp.TCPConnector(limit=20, keepalive_timeout=30)
    async with aiohttp.ClientSession(connector=connector) as aio_session:
        try:
            results = await asyncio.wait_for(
                runner.run_series(
                    players,
                    num_games.value,
                    session=aio_session,
                    runtime_mode=runtime_selector.value,
                    payoff_display=payoff_display.value,
                    callbacks=callbacks,
                ),
                timeout=SERIES_TIMEOUT
            )
        except asyncio.TimeoutError:
            activity_log.append({
                "timestamp": datetime.now().isoformat(),
                "level": "ERROR",
                "message": f"Game series timed out after {SERIES_TIMEOUT}s"
            })
            results = []

    elapsed = time.time() - start_time

    # Log completion
    activity_log.append({
        "timestamp": datetime.now().isoformat(),
        "level": "SUCCESS",
        "message": f"Completed {len(results)} games in {elapsed:.2f}s"
    })

    # Save results
    session_manager.save_session_metadata(session)
    session_manager.save_results(session.session_id, results, game_type_selector.value)

    activity_log.append({
        "timestamp": datetime.now().isoformat(),
        "level": "INFO",
        "message": f"Results saved for session {session.session_id}"
    })

    # Convert to DataFrame
    results_df = pl.DataFrame(results)

    # Only add cumulative columns and calculate stats if we have results
    if not results_df.is_empty():
        # Add cumulative payoffs dynamically for all players
        cumsum_cols = [
            pl.col(f"player{p+1}_payoff").cum_sum().alias(f"cumulative_payoff_player{p+1}")
            for p in range(num_players)
        ]
        results_df = results_df.with_columns(cumsum_cols)

        # Calculate summary stats dynamically
        totals = []
        avgs = []
        for p in range(num_players):
            col = f"player{p+1}_payoff"
            totals.append(results_df[col].sum())
            avgs.append(results_df[col].mean())
    else:
        # Handle empty results - initialize with zeros
        totals = [0.0] * num_players
        avgs = [0.0] * num_players

    # Get API metrics
    metrics_data = metrics.to_dict()

    # Build error callout if there were failures
    error_callout = None
    if metrics_data["failed_requests"] > 0:
        error_callout = mo.callout(
            mo.md(f"**Warning:** {metrics_data['failed_requests']} API call(s) failed"),
            kind="warn",
        )

    # Build stat display dynamically
    stat_elements = []
    for p in range(num_players):
        stat_elements.append(mo.stat(label=f"P{p+1} Total", value=str(totals[p])))
    for p in range(num_players):
        stat_elements.append(mo.stat(label=f"P{p+1} Avg", value=f"{avgs[p]:.2f}"))
    stat_elements.extend([
        mo.stat(label="API Calls", value=str(metrics_data["total_requests"])),
        mo.stat(label="Success", value=f"{metrics_data['success_rate']:.0f}%"),
        mo.stat(label="Avg Resp", value=f"{metrics_data['avg_response_time']:.2f}s"),
    ])

    # Display results - simplified summary (charts moved to Analytics tab)
    result_elements = [
        mo.md("## Results"),
        mo.md(f"**Session:** {session.session_id} | **Games:** {len(results)} | **Players:** {num_players} | **Time:** {elapsed:.2f}s"),
    ]

    # Add error callout if present
    if error_callout:
        result_elements.append(error_callout)

    result_elements.extend([
        # Inline metrics - game stats and API stats combined (dynamic for N players)
        mo.hstack(stat_elements, justify="start"),
        mo.md("---"),
        # Activity Log - always visible (not in accordion)
        mo.md("### Activity Log"),
        format_activity_log(activity_log),
        mo.md("---"),
        # Prompt Log - separate section showing full prompts/responses
        mo.md("### Prompt Log"),
        format_prompt_log(prompt_log),
        # Game Log in accordion
        mo.accordion({
            "Game Log": results_df
        }),
        mo.callout(
            mo.md("View detailed charts and cross-session analysis in the **Analytics** tab."),
            kind="info",
        ),
    ])

    mo.vstack(result_elements)
    return


@app.cell
def _(AnalyticsService, metrics, session_manager):
    # Create analytics service
    analytics_service = AnalyticsService(session_manager, metrics)
    return (analytics_service,)


@app.cell
def _(CrossGameComparativeAnalyzer, ModelPersonalityProfiler, all_roles, mo, session_manager):
    # Advanced Analysis UI Elements

    # Role selector for personality profiling (uses roles, not raw models)
    if all_roles:
        role_options = {f"{r.name} ({r.model})": r.role_id for r in all_roles}
        default_role = list(role_options.keys())[0] if role_options else None
    else:
        role_options = {"No roles available": None}
        default_role = "No roles available"

    profile_role_selector = mo.ui.dropdown(
        options=role_options,
        label="Select Role to Profile",
        value=default_role,
    )

    # Create mapping from role_id to role for easy lookup
    role_lookup = {r.role_id: r for r in all_roles} if all_roles else {}

    # Create analyzers
    personality_profiler = ModelPersonalityProfiler(session_manager)
    cross_game_analyzer = CrossGameComparativeAnalyzer(session_manager)

    return (
        profile_role_selector,
        role_lookup,
        personality_profiler,
        cross_game_analyzer,
    )


@app.cell
def _(mo, game_names):
    # Analytics filter UI elements - INDEPENDENT of Play tab's game selector
    # Game filter defaults to "All Games" (None)
    analytics_game_options = {"All Games": None, **{v: k for k, v in game_names.items()}}
    analytics_game_filter = mo.ui.dropdown(
        options=analytics_game_options,
        label="Game",
        value="All Games",
    )

    custom_payoffs_filter = mo.ui.dropdown(
        options={"All Sessions": None, "With Custom Payoffs": True, "Without Custom Payoffs": False},
        value=None,
        label="Payoffs",
    )

    runtime_mode_filter = mo.ui.dropdown(
        options={"All Modes": None, "One-off": "one-off", "Repeated": "repeated", "Sequential": "sequential"},
        value=None,
        label="Runtime",
    )

    return analytics_game_filter, custom_payoffs_filter, runtime_mode_filter


@app.cell
def _(all_roles, session_manager, role_repository, RoleAnalyticsFilterBuilder, RoleAnalyticsService):
    """Role Analytics: Create role selector and analytics service."""
    role_analytics_service = RoleAnalyticsService(session_manager, role_repository)
    role_analytics_selector = RoleAnalyticsFilterBuilder.build_role_selector(all_roles)
    return (role_analytics_service, role_analytics_selector)


@app.cell
def _(role_analytics_selector, role_analytics_service, game_names, RoleAnalyticsFilterBuilder):
    """Role Analytics: Create game and session filter chips based on selected role."""
    _selected_role_id = role_analytics_selector.value if role_analytics_selector.value else None

    if _selected_role_id:
        _available_games = role_analytics_service.get_available_games_for_role(_selected_role_id)
        _available_sessions = role_analytics_service.get_available_sessions_for_role(_selected_role_id)
        role_game_chips = RoleAnalyticsFilterBuilder.build_game_type_chips(_available_games, game_names)
        role_session_chips = RoleAnalyticsFilterBuilder.build_session_chips(_available_sessions)
    else:
        role_game_chips = None
        role_session_chips = None

    return (role_game_chips, role_session_chips)


@app.cell
def _(
    mo,
    role_analytics_selector,
    role_game_chips,
    role_session_chips,
    role_analytics_service,
    role_repository,
    personality_profiler,
    session_manager,
    RoleAnalyticsFilterBuilder,
    RoleAnalyticsEmptyStates,
    ImprovedRoleAnalyticsPanelBuilder,
    RoleFilterParams,
):
    """Role Analytics: Build the complete panel with filters and content."""
    _selected_role_id = role_analytics_selector.value if role_analytics_selector.value else None

    if not _selected_role_id:
        # No role selected - show empty state
        _role_analytics_panel = RoleAnalyticsEmptyStates.no_role_selected()
    else:
        # Get selected filters from chips
        _selected_games = list(role_game_chips.value) if role_game_chips is not None and role_game_chips.value else None
        _selected_sessions = list(role_session_chips.value) if role_session_chips is not None and role_session_chips.value else None

        # Create filter params
        _filters = RoleFilterParams(
            role_id=_selected_role_id,
            game_types=_selected_games,
            session_ids=_selected_sessions,
        )

        # Get role and stats
        _role = role_repository.get_by_id(_selected_role_id)
        _stats = role_analytics_service.get_role_statistics(_filters)

        if _stats.total_sessions == 0:
            if _selected_games or _selected_sessions:
                _role_analytics_panel = RoleAnalyticsEmptyStates.no_data_for_filters(
                    _role.name, _selected_games or [], _selected_sessions or []
                )
            else:
                _role_analytics_panel = RoleAnalyticsEmptyStates.no_data_for_role(_role.name)
        else:
            # Get all the data needed for the panel
            _timeline = role_analytics_service.get_session_timeline(_filters)
            _breakdown = role_analytics_service.get_game_breakdown(_filters)
            _round_data = role_analytics_service.get_round_level_data(_filters)
            _sessions = role_analytics_service.get_available_sessions_for_role(
                _selected_role_id,
                game_type=_selected_games[0] if _selected_games and len(_selected_games) == 1 else None
            )

            # Build the complete panel
            _role_analytics_panel = ImprovedRoleAnalyticsPanelBuilder.build_complete_panel(
                role=_role,
                stats=_stats,
                timeline=_timeline,
                breakdown=_breakdown,
                round_data=_round_data,
                sessions=_sessions,
                profiler=personality_profiler,
                session_manager=session_manager,
            )

    # Build the filter bar
    _filter_bar = RoleAnalyticsFilterBuilder.build_filter_bar(
        role_analytics_selector, role_game_chips, role_session_chips
    )

    # Combine filter bar and content
    role_analytics_full_content = mo.vstack([
        _filter_bar,
        mo.md("---"),
        _role_analytics_panel,
    ])

    return (role_analytics_full_content,)


@app.cell
def _(GameAnalyticsPanelBuilder, analytics_service, cross_game_analyzer, custom_payoffs_filter, analytics_game_filter, mo, runtime_mode_filter, is_burr_game, get_game, pl):
    # Game Analytics tab content - uses separate analytics_game_filter (independent of Play tab)
    builder = GameAnalyticsPanelBuilder
    selected_game = analytics_game_filter.value  # None = "All Games"
    custom_payoff_val = custom_payoffs_filter.value
    runtime_mode_val = runtime_mode_filter.value

    data = analytics_service.get_dashboard_data(
        game_type=selected_game,  # None = all games
        uses_custom_payoffs=custom_payoff_val,
        runtime_mode=runtime_mode_val,
    )

    # Build filter bar with game filter
    filter_bar = mo.hstack([
        mo.md("**Filters:**"),
        analytics_game_filter,
        custom_payoffs_filter,
        runtime_mode_filter,
    ], justify="start", gap=1)

    # Build active filters summary
    active_filters = []
    if custom_payoff_val is not None:
        active_filters.append("Custom Payoffs: " + ("Yes" if custom_payoff_val else "No"))
    if runtime_mode_val:
        active_filters.append(f"Runtime: {runtime_mode_val}")

    filter_summary = f" | Filters: {', '.join(active_filters)}" if active_filters else ""
    game_display = selected_game if selected_game else "All Games"

    if not data["has_data"]:
        analytics_content = mo.vstack([
            mo.md("## Analytics Dashboard"),
            filter_bar,
            mo.md(f"**Showing:** {game_display}{filter_summary}"),
            mo.callout(
                mo.md("No session data available for these filters. Try adjusting filters or run some games in the **Play** tab!"),
                kind="info",
            ),
        ])
    else:
        # Build dashboard elements dynamically
        dashboard_elements = [
            mo.md("## Analytics Dashboard"),
            mo.md("**Please note you might have to refresh the page**"),
            filter_bar,
            mo.md(f"**Showing:** {game_display} | **Sessions:** {data['sessions_count']}{filter_summary}"),
            builder.build_metrics_section(data["cumulative"]),
            mo.md("---"),
            builder.build_response_time_section(analytics_service.get_response_times()),
            mo.md("---"),
            builder.build_leaderboard_section(data["leaderboard"]),
            mo.md("---"),
            builder.build_heatmap_section(analytics_service.get_model_comparison_data(game_type=selected_game)),
            mo.md("---"),
            builder.build_model_game_specialization_section(cross_game_analyzer),
            mo.md("---"),
            builder.build_cooperation_section(analytics_service.get_cooperation_rates(game_type=selected_game)),
        ]

        # Add allocation analytics section if filtering to an allocation game
        is_allocation_filtered = selected_game and is_burr_game(selected_game)
        if is_allocation_filtered:
            # Get all results for this game type
            all_data = analytics_service.get_all_session_data()
            if not all_data.is_empty() and "game_type" in all_data.columns:
                filtered_data = all_data.filter(pl.col("game_type") == selected_game)
                if not filtered_data.is_empty():
                    # Convert to list of dicts for allocation analysis
                    all_results = filtered_data.to_dicts()

                    # Get budget from game definition
                    try:
                        _alloc_game = get_game(selected_game)
                        budget = _alloc_game.action_space.budget if hasattr(_alloc_game.action_space, 'budget') else 100.0
                        game_name = _alloc_game.name
                    except Exception:
                        budget = 100.0
                        game_name = game_display

                    dashboard_elements.extend([
                        mo.md("---"),
                        builder.build_allocation_aggregate_section(
                            all_results,
                            num_players=2,
                            budget=budget,
                            game_name=game_name
                        ),
                    ])

        # Add remaining sections
        dashboard_elements.extend([
            mo.md("---"),
            builder.build_game_summary_section(data["game_summary"]),
            mo.md("---"),
            builder.build_sessions_section(data["sessions"]),
        ])

        analytics_content = mo.vstack(dashboard_elements)

    return (analytics_content,)


@app.cell
def _(analytics_service, analytics_game_filter, mo):
    # Session selector for detail view - uses separate analytics filter (None = all games)
    sessions_list = analytics_service.get_dashboard_data(game_type=analytics_game_filter.value)["sessions"]

    if sessions_list:
        session_options = {
            f"{s.get('session_id', 'Unknown')} ({s.get('game_type', 'Unknown')})": s.get('session_id')
            for s in sessions_list
        }
        session_options = {"Select a session...": None, **session_options}
        default_value = "Select a session..."
    else:
        session_options = {"No sessions available": None}
        default_value = "No sessions available"

    session_selector = mo.ui.dropdown(
        options=session_options,
        label="View Session Details",
        value=default_value,
    )
    return (session_selector,)


@app.cell
def _(GameAnalyticsPanelBuilder, RoleAnalyticsPanelBuilder, analytics_service, detect_num_players, mo, session_manager, session_selector, get_game, is_burr_game):
    # Session detail views for both Game Analytics and Role Analytics
    selected_session_id = session_selector.value

    if selected_session_id:
        session_results = analytics_service.get_session_results(selected_session_id)
        if session_results is not None and not session_results.is_empty():
            # Load session metadata to get stored payoff matrix
            session_metadata = session_manager.load_session_metadata(selected_session_id)

            # Build standard session detail charts (Game Analytics)
            game_session_detail = GameAnalyticsPanelBuilder.build_session_detail_section(
                session_results, selected_session_id
            )

            # Build payoff matrix section from stored session config (Game Analytics)
            payoff_matrix_section = GameAnalyticsPanelBuilder.build_payoff_matrix_section(
                session_metadata
            )

            # Extract player models for labeling
            _num_players = detect_num_players(tuple(session_results.columns))
            player_models = {}
            for _pnum in range(1, _num_players + 1):
                if f"player{_pnum}_model" in session_results.columns:
                    models = session_results[f"player{_pnum}_model"].unique().to_list()
                    player_models[_pnum] = models[0] if models else f"Player {_pnum}"

            # Build strategy analysis (Role Analytics)
            strategy_section = RoleAnalyticsPanelBuilder.build_strategy_section(
                session_results, player_models
            )

            # Build learning curve analysis (Role Analytics)
            learning_section = RoleAnalyticsPanelBuilder.build_learning_curve_section(
                session_results, player_models
            )

            # Build equilibrium analysis - use stored payoffs if available, fallback to registry
            game_type = session_results["game_type"].unique().to_list()[0] if "game_type" in session_results.columns else None

            # Try to reconstruct game from stored session config (with custom payoffs)
            game_def = GameAnalyticsPanelBuilder.reconstruct_game_from_config(session_metadata, game_type)

            # Check if this is a Burr/allocation game
            _is_burr = is_burr_game(game_type) if game_type else False

            # Fallback to registry if no stored config (use get_game for both discrete and Burr games)
            if game_def is None and game_type:
                try:
                    game_def = get_game(game_type)
                    # Burr games don't have payoff_matrix for equilibrium analysis
                    if _is_burr:
                        game_def = None  # Skip equilibrium analysis for Burr games
                except KeyError:
                    game_def = None

            # Build appropriate analysis based on game type
            if _is_burr:
                # Allocation game: show allocation analytics instead of equilibrium
                _burr_game = get_game(game_type) if game_type else None
                _budget = _burr_game.action_space.budget if _burr_game and hasattr(_burr_game.action_space, 'budget') else 100.0
                _game_name = _burr_game.name if _burr_game else "Allocation Game"

                # Convert polars DataFrame to list of dicts for allocation analyzer
                _results_list = session_results.to_dicts()

                allocation_section = GameAnalyticsPanelBuilder.build_allocation_session_detail(
                    _results_list, _num_players, _budget, _game_name
                )

                # Game Analytics: session detail + allocation analysis
                game_session_detail_content = mo.vstack([
                    game_session_detail,
                    mo.md("---"),
                    allocation_section,
                ])

                # Role Analytics: strategy + learning for Burr games
                role_session_detail_content = mo.vstack([
                    mo.md(f"### Session: {selected_session_id}"),
                    mo.md("## Player Behavior Analysis"),
                    mo.accordion({
                        "Strategy Detection": strategy_section,
                        "Learning Curves": learning_section,
                    }),
                ])
            else:
                # Discrete game: standard equilibrium analysis
                equilibrium_section = GameAnalyticsPanelBuilder.build_equilibrium_section(
                    game_def, session_results
                )

                # Game Analytics: session detail + payoff matrix + equilibrium
                game_session_detail_content = mo.vstack([
                    game_session_detail,
                    mo.md("---"),
                    payoff_matrix_section,
                    mo.md("---"),
                    mo.md("## Game Theory Analysis"),
                    mo.accordion({
                        "Equilibrium Analysis": equilibrium_section,
                    }),
                ])

                # Role Analytics: strategy + learning
                role_session_detail_content = mo.vstack([
                    mo.md(f"### Session: {selected_session_id}"),
                    mo.md("## Player Behavior Analysis"),
                    mo.accordion({
                        "Strategy Detection": strategy_section,
                        "Learning Curves": learning_section,
                    }),
                ])
        else:
            game_session_detail_content = mo.callout(
                mo.md(f"Could not load data for session {selected_session_id}"),
                kind="warn",
            )
            role_session_detail_content = mo.callout(
                mo.md(f"Could not load data for session {selected_session_id}"),
                kind="warn",
            )
    else:
        game_session_detail_content = mo.md("_Select a session above to view detailed charts._")
        role_session_detail_content = mo.md("_Select a session above to view player behavior analysis._")

    # Keep backward compatibility - session_detail_content is used by old code
    session_detail_content = game_session_detail_content

    return (session_detail_content, game_session_detail_content, role_session_detail_content)


@app.cell
def _(
    RoleAnalyticsPanelBuilder,
    GameAnalyticsPanelBuilder,
    MetaStrategyAnalyzer,
    analytics_service,
    cross_game_analyzer,
    game_type_selector,
    mo,
    personality_profiler,
    profile_role_selector,
    role_lookup,
    session_selector,
):
    # Advanced Analysis Sections - split between Role Analytics and Game Analytics
    _role_builder = RoleAnalyticsPanelBuilder
    _game_builder = GameAnalyticsPanelBuilder
    _selected_game = game_type_selector.value
    _selected_session_id = session_selector.value

    # --- Meta-Learning Analysis Section (Role Analytics) ---
    def _build_meta_learning_accordion():
        if not _selected_session_id:
            return mo.callout(
                mo.md("Select a session above to see meta-learning analysis."),
                kind="neutral",
            )

        _session_results = analytics_service.get_session_results(_selected_session_id)
        if _session_results is None or _session_results.is_empty():
            return mo.callout(mo.md("No data for meta-learning analysis."), kind="neutral")

        # Convert to list of dicts for MetaStrategyAnalyzer
        _results_list = _session_results.to_dicts()
        _num_players = 2  # Default, could detect from columns
        for _col in _session_results.columns:
            if _col.startswith("player") and _col.endswith("_action"):
                _player_num = int(_col.replace("player", "").replace("_action", ""))
                _num_players = max(_num_players, _player_num)

        return _role_builder.build_meta_learning_section(_results_list, _num_players)

    _meta_learning_section = _build_meta_learning_accordion()

    # --- Role Personality Section (Role Analytics) ---
    def _build_personality_accordion():
        selected_role_id = profile_role_selector.value
        if not selected_role_id or selected_role_id not in role_lookup:
            return mo.callout(mo.md("Select a role to see personality profile."), kind="neutral")

        # Get the model from the selected role
        selected_role = role_lookup[selected_role_id]
        return _role_builder.build_personality_section(personality_profiler, selected_role.model)

    _personality_section = mo.vstack([
        mo.md("**Select role to analyze:**"),
        profile_role_selector,
        _build_personality_accordion(),
    ])

    # --- Intelligence Leaderboard Section (Game Analytics) ---
    def _build_intelligence_accordion():
        return _game_builder.build_intelligence_leaderboard_section(cross_game_analyzer)

    _intelligence_section = _build_intelligence_accordion()

    # Role Analytics advanced analysis accordion (role/player focused)
    role_advanced_accordion = mo.accordion({
        "Meta-Strategy Learning": _meta_learning_section,
        "Role Personality Profiling": _personality_section,
    })

    # Game Analytics advanced analysis accordion (game/outcome focused)
    game_advanced_accordion = mo.accordion({
        "Intelligence Leaderboard": _intelligence_section,
    })

    # Keep backward compatibility
    advanced_analysis_accordion = mo.accordion({
        "Meta-Strategy Learning": _meta_learning_section,
        "Role Personality Profiling": _personality_section,
        "Intelligence Leaderboard": _intelligence_section,
    })

    return (advanced_analysis_accordion, role_advanced_accordion, game_advanced_accordion)


@app.cell
def _(analytics_content, analytics_game_filter, config_ui, custom_game_creator_ui, game_info, mo, payoff_editor_ui, game_session_detail_content, role_session_detail_content, session_selector, role_advanced_accordion, game_advanced_accordion, roles_tab_content, role_analytics_full_content, queue_section):
    # Build Role Analytics tab content - now uses new smart filtering!
    role_analytics_tab_content = role_analytics_full_content

    # Build Game Analytics tab content (game outcomes and performance focused)
    game_analytics_tab_content = mo.vstack([
        analytics_content,
        mo.md("---"),
        mo.md("### Session Game Analysis"),
        session_selector,
        game_session_detail_content,
        mo.md("---"),
        mo.md("### Advanced Game Analysis"),
        mo.md("_Intelligence rankings and cross-game performance._"),
        game_advanced_accordion,
    ])

    # Main tabbed interface with split analytics
    tabs = mo.ui.tabs({
        "Roles": roles_tab_content,
        "Play": mo.vstack([
            mo.md("# Game Theory LLM Arena"),
            mo.hstack([
                mo.vstack([config_ui], align="start"),
                mo.vstack([
                    game_info,
                    mo.md("### Game Customization"),
                    payoff_editor_ui,
                    custom_game_creator_ui,
                ], align="start"),
            ]),
            mo.md("---"),
            queue_section,
        ]),
        "Role Analytics": role_analytics_tab_content,
        "Game Analytics": game_analytics_tab_content,
        "About": mo.md("""
# About Game Theory LLM Arena

This application tests how AI language models make strategic decisions in game-theoretic scenarios.

## Available Games

- **Prisoner's Dilemma**: Classic cooperation vs defection
- **Chicken Game**: High-stakes confrontation
- **Stag Hunt**: Coordination with risk
- **Battle of the Sexes**: Preference coordination
- **Matching Pennies**: Zero-sum matching
- **Trust Game**: Trust and betrayal
- **Public Goods**: Contribution dilemma
- **Coordination Game**: Pure coordination
- **Three-Player Public Good**: Multiplayer dynamics
- **Iterated Prisoner's Dilemma**: Repeated play with memory
- **Punishment Public Goods**: Enforcement mechanisms

## How It Works

1. Select a game type and configure players
2. Choose Ollama models and endpoints for each player
3. Optionally customize payoffs or create custom games
4. Run the game series
5. Analyze results and compare model behaviors

## Dynamic Configuration

- **Model Parameters**: Adjust temperature and top_p per player
- **Custom Payoffs**: Modify payoff values for any game
- **Custom Games**: Create new games with custom actions

## Data Persistence

Results are automatically saved to Parquet files for cross-game analysis.
View the Analytics tab for aggregated statistics and model comparisons.

## Observability

**Burr Tracking UI**: View game execution states and debug at [localhost:7241](http://localhost:7241)

The Burr server provides visibility into:
- Step-by-step state transitions
- Action inputs and outputs
- Complete execution history
        """),
    })

    tabs
    return (tabs,)


if __name__ == "__main__":
    app.run()
