"""Game Theory LLM Arena - Main tabbed Marimo application."""

import nest_asyncio
nest_asyncio.apply()

import marimo

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
    import nest_asyncio
    from pathlib import Path
    from datetime import datetime
    import sys

    # Add parent directory to path for local development
    _project_root = Path(__file__).parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

    from itertools import product as itertools_product

    # Import game theory package
    from gametheory import (
        OLLAMA_MODELS,
        OLLAMA_ENDPOINTS,
        GAME_REGISTRY,
        get_game,
        list_games,
        get_game_names,
        PlayerConfig,
        GameRunner,
        MetricsTracker,
        SessionManager,
        CrossGameAnalyzer,
        AnalyticsService,
    )
    from gametheory.core.config import (
        discover_all_available,
        DEFAULT_OLLAMA_MODELS,
        DEFAULT_OLLAMA_ENDPOINTS,
    )
    from gametheory.core.types import GameDefinition
    from gametheory.visualization import (
        create_cumulative_payoff_chart,
        create_action_distribution_chart,
        create_payoff_comparison_chart,
        create_avg_payoff_chart,
    )
    from gametheory.ui import AnalyticsPanelBuilder

    return (
        AnalyticsPanelBuilder,
        AnalyticsService,
        DEFAULT_OLLAMA_ENDPOINTS,
        DEFAULT_OLLAMA_MODELS,
        GAME_REGISTRY,
        GameDefinition,
        OLLAMA_ENDPOINTS,
        OLLAMA_MODELS,
        CrossGameAnalyzer,
        GameRunner,
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
        itertools_product,
        list_games,
        mo,
        nest_asyncio,
        pl,
        time,
    )


@app.cell
def _(mo):
    def format_activity_log(log_entries):
        """Format activity log as terminal-style output."""
        if not log_entries:
            return mo.md("_No activity yet. Run a game series to see logs._")

        log_lines = []
        for entry in log_entries[-200:]:  # Last 200 entries
            timestamp = entry.get("timestamp", "")[:19]  # Trim microseconds
            msg = entry.get("message", "")
            level = entry.get("level", "INFO")

            # Color code by level
            if level == "ERROR":
                line = f"<span style='color: #ff6b6b'>[{timestamp}] {msg}</span>"
            elif level == "WARN":
                line = f"<span style='color: #ffd43b'>[{timestamp}] ⚠ {msg}</span>"
            elif level == "SUCCESS":
                line = f"<span style='color: #69db7c'>[{timestamp}] {msg}</span>"
            else:
                line = f"<span style='color: #868e96'>[{timestamp}]</span> {msg}"
            log_lines.append(line)

        return mo.Html(f"""
            <div style="font-family: monospace; font-size: 12px;
                        background: #1a1a2e; color: #eee; padding: 12px;
                        border-radius: 6px; max-height: 500px; overflow-y: auto;">
                {"<br>".join(log_lines)}
            </div>
        """)

    return (format_activity_log,)


@app.cell
def _(mo):
    def format_prompt_log(prompt_entries):
        """Format prompt log showing full prompts and responses."""
        if not prompt_entries:
            return mo.md("_No prompts yet. Run a game series to see prompts._")

        sections = []
        for entry in prompt_entries[-50:]:  # Last 50 prompt exchanges
            round_num = entry.get("round", "?")
            player = entry.get("player", "?")
            model = entry.get("model", "unknown")
            prompt = entry.get("prompt", "")
            response = entry.get("response", "")
            system_prompt = entry.get("system_prompt")
            was_parsed = entry.get("was_parsed", True)

            # Parse status indicator
            parse_badge = '<span style="color: #69db7c;">✓ parsed</span>' if was_parsed else '<span style="color: #ff6b6b; font-weight: bold;">⚠ DEFAULT</span>'

            # System prompt section (only show if present)
            sys_prompt_html = ""
            if system_prompt:
                sys_prompt_html = f'''
<div style="color: #ffd43b; font-size: 11px; margin-bottom: 4px;">SYSTEM PROMPT:</div>
<pre style="background: #2d2d0d; color: #ffd43b; padding: 8px; font-size: 11px; white-space: pre-wrap; border-radius: 4px; margin: 0 0 8px 0; border: 1px solid #ffd43b;">{system_prompt}</pre>'''

            section = f"""
<details style="margin-bottom: 8px; background: #1a1a2e; padding: 8px; border-radius: 4px;">
<summary style="cursor: pointer; color: #69db7c; font-family: monospace;">
Round {round_num} | P{player} ({model}) {parse_badge}
</summary>
<div style="margin-top: 8px;">
{sys_prompt_html}<div style="color: #868e96; font-size: 11px; margin-bottom: 4px;">PROMPT:</div>
<pre style="background: #0d0d1a; color: #eee; padding: 8px; font-size: 11px; white-space: pre-wrap; border-radius: 4px; margin: 0 0 8px 0;">{prompt}</pre>
<div style="color: #868e96; font-size: 11px; margin-bottom: 4px;">RESPONSE:</div>
<pre style="background: #0d0d1a; color: #69db7c; padding: 8px; font-size: 11px; white-space: pre-wrap; border-radius: 4px; margin: 0;">{response}</pre>
</div>
</details>"""
            sections.append(section)

        return mo.Html(f"""
            <div style="max-height: 400px; overflow-y: auto;">
                {"".join(sections)}
            </div>
        """)

    return (format_prompt_log,)


@app.cell
async def _(DEFAULT_OLLAMA_ENDPOINTS, DEFAULT_OLLAMA_MODELS, discover_all_available, mo):
    # Discover available Ollama models and endpoints at session start
    mo.output.append(mo.md("_Discovering Ollama endpoints..._"))

    discovered_models, discovered_endpoints, endpoint_models = await discover_all_available(
        endpoints=DEFAULT_OLLAMA_ENDPOINTS,
        timeout=3.0,
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
def _(available_endpoints, get_game_names, mo):
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
    num_games = mo.ui.slider(1, 100, value=10, label="Number of Games")

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
    system_p1 = mo.ui.text_area(label="System Prompt", placeholder="Optional: Custom instructions for this player...")

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
    system_p2 = mo.ui.text_area(label="System Prompt", placeholder="Optional: Custom instructions for this player...")

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
    system_p3 = mo.ui.text_area(label="System Prompt", placeholder="Optional: Custom instructions for this player...")

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
        system_p1,
        system_p2,
        system_p3,
    )


@app.cell
def _(MetricsTracker, Path, SessionManager):
    # Initialize shared state
    metrics = MetricsTracker()
    session_manager = SessionManager(str(Path(__file__).parent.parent / "data" / "sessions"))

    return metrics, session_manager


@app.cell
def _(
    endpoint_p1,
    endpoint_p2,
    endpoint_p3,
    GAME_REGISTRY,
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
    system_p1,
    system_p2,
    system_p3,
):
    # Build configuration UI
    def _build_config_ui():
        # Player 1 section with advanced settings
        player1_advanced = mo.accordion({
            "Advanced Settings": mo.vstack([
                mo.hstack([temp_p1, top_p_p1]),
                system_p1,
            ])
        })
        player1_section = mo.vstack([
            mo.md("**Player 1**"),
            mo.hstack([model_p1, endpoint_p1]),
            player1_advanced,
        ])

        # Player 2 section with advanced settings
        player2_advanced = mo.accordion({
            "Advanced Settings": mo.vstack([
                mo.hstack([temp_p2, top_p_p2]),
                system_p2,
            ])
        })
        player2_section = mo.vstack([
            mo.md("**Player 2**"),
            mo.hstack([model_p2, endpoint_p2]),
            player2_advanced,
        ])

        rows = [
            mo.md("## Configuration"),
            mo.hstack([game_type_selector, runtime_selector, payoff_display]),
            num_games,
            mo.md("### Players"),
            player1_section,
            player2_section,
        ]

        # Show Player 3 for multiplayer mode OR when selected game has 3+ players
        selected_game = GAME_REGISTRY.get(game_type_selector.value)
        game_needs_3_players = selected_game and selected_game.num_players >= 3
        if runtime_selector.value == "multi_player" or game_needs_3_players:
            player3_advanced = mo.accordion({
                "Advanced Settings": mo.vstack([
                    mo.hstack([temp_p3, top_p_p3]),
                    system_p3,
                ])
            })
            player3_section = mo.vstack([
                mo.md("**Player 3**"),
                mo.hstack([model_p3, endpoint_p3]),
                player3_advanced,
            ])
            rows.append(player3_section)

        rows.append(run_button)
        return mo.vstack(rows)

    config_ui = _build_config_ui()
    return (config_ui,)


@app.cell
def _(active_game, get_custom_payoffs, has_custom_payoffs, custom_payoffs_changes, game_type_selector, mo, pl):
    # Display current game info - depends on get_custom_payoffs for reactivity when payoffs change
    def _get_game_info():
        game = active_game
        game_id = game_type_selector.value

        if not game:
            return mo.md("Select a game to see details.")

        # Build payoff matrix display
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

        # Build elements list
        elements = [mo.md(f"### {game.name}")]

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

        elements.extend([
            mo.md(game.description),
            mo.md(f"**Players:** {game.num_players} | **Actions:** {', '.join(game.actions)}"),
            mo.md("#### Payoff Matrix"),
            matrix_df,
        ])

        return mo.vstack(elements)

    game_info = _get_game_info()
    return (game_info,)


@app.cell
def _(GAME_REGISTRY, GameDefinition, get_custom_payoffs, set_custom_payoffs, game_type_selector, mo):
    # Custom Payoff Editor with reactive state
    game_id = game_type_selector.value
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
def _(GameDefinition, session_custom_games, itertools_product, mo):
    # Custom Game Creator
    custom_game_name = mo.ui.text(label="Game Name", placeholder="My Custom Game")
    custom_game_actions = mo.ui.text(label="Actions (comma-separated)", placeholder="cooperate, defect")
    custom_game_num_players = mo.ui.dropdown(
        options={"2 Players": 2, "3 Players": 3},
        label="Number of Players",
        value="2 Players",
    )

    def _get_actions_list():
        text = custom_game_actions.value or ""
        return [a.strip() for a in text.split(",") if a.strip()]

    def _create_custom_game():
        name = custom_game_name.value or "Custom Game"
        actions = _get_actions_list()
        n_players = custom_game_num_players.value

        if not actions or len(actions) < 2:
            return None

        # Generate payoff matrix with default values (0)
        payoff_matrix = {}
        for combo in itertools_product(actions, repeat=n_players):
            payoff_matrix[combo] = tuple(0 for _ in range(n_players))

        game = GameDefinition(
            id="custom_" + name.lower().replace(" ", "_"),
            name=name,
            description="User-created custom game",
            payoff_matrix=payoff_matrix,
            actions=actions,
            num_players=n_players,
        )
        return game

    def _add_custom_game(_):
        game = _create_custom_game()
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
def _(
    GameRunner,
    PlayerConfig,
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
    system_p1,
    system_p2,
    system_p3,
    time,
):
    # Game execution cell - N-player support
    mo.stop(not run_button.value)

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

    game = active_game  # Use active_game which includes custom payoffs
    num_players = game.num_players

    activity_log.append({
        "timestamp": datetime.now().isoformat(),
        "level": "INFO",
        "message": f"Game: {game.name} | Players: {num_players} | Rounds: {num_games.value}"
    })

    # Gather all player settings into a list
    all_player_settings = [
        {"model": model_p1.value, "endpoint": endpoint_p1.value, "temperature": temp_p1.value,
         "top_p": top_p_p1.value, "system_prompt": system_p1.value if system_p1.value else None},
        {"model": model_p2.value, "endpoint": endpoint_p2.value, "temperature": temp_p2.value,
         "top_p": top_p_p2.value, "system_prompt": system_p2.value if system_p2.value else None},
        {"model": model_p3.value, "endpoint": endpoint_p3.value, "temperature": temp_p3.value,
         "top_p": top_p_p3.value, "system_prompt": system_p3.value if system_p3.value else None},
    ]

    # Create players based on game.num_players
    players = [
        PlayerConfig(
            player_id=_pn + 1,
            model=all_player_settings[_pn]["model"],
            endpoint=all_player_settings[_pn]["endpoint"],
            temperature=all_player_settings[_pn]["temperature"],
            top_p=all_player_settings[_pn]["top_p"],
            system_prompt=all_player_settings[_pn]["system_prompt"],
        )
        for _pn in range(num_players)
    ]

    # Log player configurations with system prompt status
    for _pi, player in enumerate(players):
        activity_log.append({
            "timestamp": datetime.now().isoformat(),
            "level": "INFO",
            "message": f"P{_pi+1}: {player.model} @ {player.endpoint} (temp={player.temperature}, sys_prompt={'YES' if player.system_prompt else 'NO'})"
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

    # Serialize the payoff matrix for storage (tuple keys -> string keys)
    serialized_matrix = {
        "_".join(actions): list(payoffs)
        for actions, payoffs in game.payoff_matrix.items()
    }

    session.config = {
        "custom_payoffs": get_custom_payoffs().get(game_type_selector.value, {}),
        "player_settings": player_settings,
        "uses_custom_payoffs": has_custom_payoffs,
        "payoff_matrix": serialized_matrix,
        "game_actions": game.actions,
        "game_name": game.name,
        "num_players": game.num_players,
    }

    activity_log.append({
        "timestamp": datetime.now().isoformat(),
        "level": "INFO",
        "message": f"Session {session.session_id} started"
    })

    # Run game series
    runner = GameRunner(game, metrics)

    start_time = time.time()

    # Run with progress bar
    async def run_with_progress():
        results = []
        history = []
        history_payoffs = []  # Track payoffs from each round
        cumulative_payoffs = [0] * num_players  # Running total for each player
        # One-off mode: each game is independent (no history)
        is_repeated = runtime_selector.value in ("repeated", "sequential")
        metrics.log_request_start(num_games.value * num_players)

        import aiohttp

        # Build subtitle for progress bar
        model_names = [p.model for p in players]
        subtitle = " vs ".join(model_names)

        async with aiohttp.ClientSession() as aio_session:
            for round_num in mo.status.progress_bar(
                range(num_games.value),
                title="Running Games",
                subtitle=subtitle,
            ):
                # For one-off mode, don't pass history to prompt
                actions, payoffs, response_times, prompts, raw_responses, was_parsed = await runner.play_round(
                    aio_session, players,
                    history if is_repeated else [],
                    payoff_display=payoff_display.value,
                    history_payoffs=history_payoffs if is_repeated else None,
                    cumulative_payoffs=tuple(cumulative_payoffs) if is_repeated else None,
                    is_repeated=is_repeated,
                )

                # Store prompts for Prompt Log display (all players)
                for p_idx, player in enumerate(players):
                    prompt_log.append({
                        "round": round_num + 1,
                        "player": p_idx + 1,
                        "model": player.model,
                        "prompt": prompts[p_idx],
                        "response": raw_responses[p_idx],
                        "system_prompt": player.system_prompt,
                        "was_parsed": was_parsed[p_idx],
                    })

                # Log API responses for all players
                for p_idx, player in enumerate(players):
                    activity_log.append({
                        "timestamp": datetime.now().isoformat(),
                        "level": "SUCCESS",
                        "message": f"API P{p_idx+1} {player.model}: {response_times[p_idx]:.2f}s"
                    })

                # Build status message dynamically
                status_parts = []
                for p_idx in range(num_players):
                    status = "parsed" if was_parsed[p_idx] else "DEFAULT"
                    status_parts.append(f"P{p_idx+1}->{actions[p_idx]} [{status}] ({payoffs[p_idx]})")

                activity_log.append({
                    "timestamp": datetime.now().isoformat(),
                    "level": "INFO" if all(was_parsed) else "WARN",
                    "message": f"Round {round_num+1}: " + ", ".join(status_parts)
                })

                # Build result dictionary dynamically
                result = {"game_number": round_num + 1}
                for p_idx, player in enumerate(players):
                    p_num = p_idx + 1
                    result[f"player{p_num}_action"] = actions[p_idx]
                    result[f"player{p_num}_payoff"] = payoffs[p_idx]
                    result[f"player{p_num}_model"] = player.model
                    result[f"player{p_num}_endpoint"] = player.endpoint
                    result[f"player{p_num}_response_time"] = response_times[p_idx]
                    result[f"player{p_num}_was_parsed"] = was_parsed[p_idx]
                    result[f"player{p_num}_temperature"] = player.temperature
                    result[f"player{p_num}_top_p"] = player.top_p

                # Add metadata
                result["uses_custom_payoffs"] = has_custom_payoffs
                result["runtime_mode"] = runtime_selector.value

                results.append(result)
                history.append(actions)
                history_payoffs.append(payoffs)

                # Update cumulative payoffs
                for p_idx in range(num_players):
                    cumulative_payoffs[p_idx] += payoffs[p_idx]

        return results

    # Execute - use existing event loop (nest_asyncio is applied at module level)
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(run_with_progress())

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
def _(mo):
    # Analytics filter UI elements
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

    return custom_payoffs_filter, runtime_mode_filter


@app.cell
def _(AnalyticsPanelBuilder, analytics_service, custom_payoffs_filter, game_type_selector, mo, runtime_mode_filter):
    # Analytics tab content using new services - filtered by selected game and filters
    builder = AnalyticsPanelBuilder
    selected_game = game_type_selector.value
    custom_payoff_val = custom_payoffs_filter.value
    runtime_mode_val = runtime_mode_filter.value

    data = analytics_service.get_dashboard_data(
        game_type=selected_game,
        uses_custom_payoffs=custom_payoff_val,
        runtime_mode=runtime_mode_val,
    )

    # Build filter bar
    filter_bar = mo.hstack([
        mo.md("**Filters:**"),
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

    if not data["has_data"]:
        analytics_content = mo.vstack([
            mo.md("## Analytics Dashboard"),
            filter_bar,
            mo.md(f"**Showing:** {selected_game}{filter_summary}"),
            mo.callout(
                mo.md("No session data available for these filters. Try adjusting filters or run some games in the **Play** tab!"),
                kind="info",
            ),
        ])
    else:
        analytics_content = mo.vstack([
            mo.md("## Analytics Dashboard"),
            filter_bar,
            mo.md(f"**Showing:** {selected_game} | **Sessions:** {data['sessions_count']}{filter_summary}"),
            builder.build_metrics_section(data["cumulative"]),
            mo.md("---"),
            builder.build_response_time_section(analytics_service.get_response_times()),
            mo.md("---"),
            builder.build_leaderboard_section(data["leaderboard"]),
            mo.md("---"),
            builder.build_heatmap_section(analytics_service.get_model_comparison_data(game_type=selected_game)),
            mo.md("---"),
            builder.build_cooperation_section(analytics_service.get_cooperation_rates(game_type=selected_game)),
            mo.md("---"),
            builder.build_game_summary_section(data["game_summary"]),
            mo.md("---"),
            builder.build_sessions_section(data["sessions"]),
        ])

    return (analytics_content,)


@app.cell
def _(analytics_service, game_type_selector, mo):
    # Session selector for detail view - filtered by selected game
    sessions_list = analytics_service.get_dashboard_data(game_type=game_type_selector.value)["sessions"]

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
def _(AnalyticsPanelBuilder, GAME_REGISTRY, analytics_service, mo, session_manager, session_selector):
    # Session detail view with educational analysis panels
    selected_session_id = session_selector.value

    if selected_session_id:
        session_results = analytics_service.get_session_results(selected_session_id)
        if session_results is not None and not session_results.is_empty():
            # Load session metadata to get stored payoff matrix
            session_metadata = session_manager.load_session_metadata(selected_session_id)

            # Build standard session detail charts
            session_detail_content = AnalyticsPanelBuilder.build_session_detail_section(
                session_results, selected_session_id
            )

            # Build payoff matrix section from stored session config
            payoff_matrix_section = AnalyticsPanelBuilder.build_payoff_matrix_section(
                session_metadata
            )

            # Extract player models for labeling
            player_models = {}
            _pnum = 1
            while f"player{_pnum}_model" in session_results.columns:
                models = session_results[f"player{_pnum}_model"].unique().to_list()
                player_models[_pnum] = models[0] if models else f"Player {_pnum}"
                _pnum += 1

            # Build strategy analysis
            strategy_section = AnalyticsPanelBuilder.build_strategy_section(
                session_results, player_models
            )

            # Build learning curve analysis
            learning_section = AnalyticsPanelBuilder.build_learning_curve_section(
                session_results, player_models
            )

            # Build equilibrium analysis - use stored payoffs if available, fallback to registry
            game_type = session_results["game_type"].unique().to_list()[0] if "game_type" in session_results.columns else None

            # Try to reconstruct game from stored session config (with custom payoffs)
            game_def = AnalyticsPanelBuilder.reconstruct_game_from_config(session_metadata, game_type)

            # Fallback to registry if no stored config
            if game_def is None and game_type:
                game_def = GAME_REGISTRY.get(game_type)

            equilibrium_section = AnalyticsPanelBuilder.build_equilibrium_section(
                game_def, session_results
            )

            # Combine all sections with educational analysis in accordion
            session_detail_content = mo.vstack([
                session_detail_content,
                mo.md("---"),
                payoff_matrix_section,
                mo.md("---"),
                mo.md("## Educational Analysis"),
                mo.accordion({
                    "Strategy Detection": strategy_section,
                    "Learning Curves": learning_section,
                    "Equilibrium Analysis": equilibrium_section,
                }),
            ])
        else:
            session_detail_content = mo.callout(
                mo.md(f"Could not load data for session {selected_session_id}"),
                kind="warn",
            )
    else:
        session_detail_content = mo.md("_Select a session above to view detailed charts._")

    return (session_detail_content,)


@app.cell
def _(analytics_content, config_ui, custom_game_creator_ui, game_info, mo, payoff_editor_ui, session_detail_content, session_selector):
    # Build analytics tab with session viewer
    analytics_tab_content = mo.vstack([
        analytics_content,
        mo.md("---"),
        mo.md("### Session Detail Viewer"),
        session_selector,
        session_detail_content,
    ])

    # Main tabbed interface
    tabs = mo.ui.tabs({
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
        ]),
        "Analytics": analytics_tab_content,
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
        """),
    })

    tabs
    return (tabs,)


if __name__ == "__main__":
    app.run()
