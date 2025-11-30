"""Game Theory LLM Arena - Main tabbed Marimo application."""

import marimo

__generated_with = "0.17.8"
app = marimo.App(width="full")


@app.cell
def _():
    # Session storage dicts - defined in cell, returned for other cells
    session_custom_games = {}
    session_custom_payoffs = {}
    activity_log = []  # Verbose activity logging
    return activity_log, session_custom_games, session_custom_payoffs


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
    from gametheory.core.types import GameDefinition
    from gametheory.visualization import (
        create_cumulative_payoff_chart,
        create_action_distribution_chart,
        create_payoff_comparison_chart,
        create_avg_payoff_chart,
    )
    from gametheory.ui import AnalyticsPanelBuilder

    nest_asyncio.apply()

    return (
        AnalyticsPanelBuilder,
        AnalyticsService,
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

            section = f"""
<details style="margin-bottom: 8px; background: #1a1a2e; padding: 8px; border-radius: 4px;">
<summary style="cursor: pointer; color: #69db7c; font-family: monospace;">
Round {round_num} | P{player} ({model})
</summary>
<div style="margin-top: 8px;">
<div style="color: #868e96; font-size: 11px; margin-bottom: 4px;">PROMPT:</div>
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
def _(OLLAMA_ENDPOINTS, OLLAMA_MODELS, get_game_names, mo):
    # Global UI Elements
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

    # Player 1 selectors
    model_p1 = mo.ui.dropdown(
        options=OLLAMA_MODELS,
        label="Player 1 Model",
        value=OLLAMA_MODELS[0],
    )
    endpoint_p1 = mo.ui.dropdown(
        options=OLLAMA_ENDPOINTS,
        label="Player 1 Endpoint",
        value=OLLAMA_ENDPOINTS[0],
    )
    # Player 1 advanced settings
    temp_p1 = mo.ui.slider(0.0, 2.0, value=0.7, step=0.1, label="Temperature")
    top_p_p1 = mo.ui.slider(0.0, 1.0, value=0.9, step=0.05, label="Top P")
    system_p1 = mo.ui.text_area(label="System Prompt", placeholder="Optional: Custom instructions for this player...")

    # Player 2 selectors
    model_p2 = mo.ui.dropdown(
        options=OLLAMA_MODELS,
        label="Player 2 Model",
        value=OLLAMA_MODELS[1] if len(OLLAMA_MODELS) > 1 else OLLAMA_MODELS[0],
    )
    endpoint_p2 = mo.ui.dropdown(
        options=OLLAMA_ENDPOINTS,
        label="Player 2 Endpoint",
        value=OLLAMA_ENDPOINTS[1] if len(OLLAMA_ENDPOINTS) > 1 else OLLAMA_ENDPOINTS[0],
    )
    # Player 2 advanced settings
    temp_p2 = mo.ui.slider(0.0, 2.0, value=0.7, step=0.1, label="Temperature")
    top_p_p2 = mo.ui.slider(0.0, 1.0, value=0.9, step=0.05, label="Top P")
    system_p2 = mo.ui.text_area(label="System Prompt", placeholder="Optional: Custom instructions for this player...")

    # Player 3 selectors (for multiplayer)
    model_p3 = mo.ui.dropdown(
        options=OLLAMA_MODELS,
        label="Player 3 Model",
        value=OLLAMA_MODELS[2] if len(OLLAMA_MODELS) > 2 else OLLAMA_MODELS[0],
    )
    endpoint_p3 = mo.ui.dropdown(
        options=OLLAMA_ENDPOINTS,
        label="Player 3 Endpoint",
        value=OLLAMA_ENDPOINTS[2] if len(OLLAMA_ENDPOINTS) > 2 else OLLAMA_ENDPOINTS[0],
    )
    # Player 3 advanced settings
    temp_p3 = mo.ui.slider(0.0, 2.0, value=0.7, step=0.1, label="Temperature")
    top_p_p3 = mo.ui.slider(0.0, 1.0, value=0.9, step=0.05, label="Top P")
    system_p3 = mo.ui.text_area(label="System Prompt", placeholder="Optional: Custom instructions for this player...")

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

        # Show Player 3 for multiplayer mode
        if runtime_selector.value == "multi_player":
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
def _(GAME_REGISTRY, game_type_selector, mo, pl):
    # Display current game info
    def _get_game_info():
        game_id = game_type_selector.value
        game = GAME_REGISTRY.get(game_id)

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

        return mo.vstack([
            mo.md(f"### {game.name}"),
            mo.md(game.description),
            mo.md(f"**Players:** {game.num_players} | **Actions:** {', '.join(game.actions)}"),
            mo.md("#### Payoff Matrix"),
            matrix_df,
        ])

    game_info = _get_game_info()
    return (game_info,)


@app.cell
def _(GAME_REGISTRY, GameDefinition, session_custom_payoffs, game_type_selector, mo):
    # Custom Payoff Editor
    game_id = game_type_selector.value
    base_game = GAME_REGISTRY.get(game_id)

    if not base_game:
        payoff_editor_ui = mo.md("Select a game to customize payoffs.")
        active_game = None
        payoff_inputs = {}
    else:
        # Check if we have custom payoffs for this game
        current_custom_payoffs = session_custom_payoffs.get(game_id, {})

        # Build payoff inputs
        payoff_inputs = {}
        for actions, payoffs in base_game.payoff_matrix.items():
            action_key = "_".join(actions)
            for i, p in enumerate(payoffs):
                input_key = f"{action_key}_p{i+1}"
                current_val = current_custom_payoffs.get(input_key, p)
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
            row_inputs = [payoff_inputs[f"{action_key}_p{i+1}"] for i in range(base_game.num_players)]
            payoff_rows.append(mo.hstack([mo.md(f"**{action_label}:**")] + row_inputs))

        def _apply_custom_payoffs():
            new_payoffs = {}
            for key, inp in payoff_inputs.items():
                new_payoffs[key] = inp.value
            session_custom_payoffs[game_id] = new_payoffs

        apply_button = mo.ui.button(label="Apply Custom Payoffs", on_click=lambda _: _apply_custom_payoffs())

        def _reset_payoffs():
            if game_id in session_custom_payoffs:
                del session_custom_payoffs[game_id]

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
                    current_custom_payoffs.get(f"{action_key}_p{i+1}", payoffs[i])
                    for i in range(base_game.num_players)
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

    return active_game, payoff_editor_ui, payoff_inputs


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
    format_activity_log,
    format_prompt_log,
    game_type_selector,
    metrics,
    mo,
    model_p1,
    model_p2,
    num_games,
    payoff_display,
    pl,
    run_button,
    runtime_selector,
    session_custom_payoffs,
    session_manager,
    temp_p1,
    temp_p2,
    top_p_p1,
    top_p_p2,
    system_p1,
    system_p2,
    time,
):
    # Game execution cell
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

    activity_log.append({
        "timestamp": datetime.now().isoformat(),
        "level": "INFO",
        "message": f"Game: {game.name} | Rounds: {num_games.value}"
    })

    # Create player configs with advanced settings
    players = [
        PlayerConfig(
            player_id=1,
            model=model_p1.value,
            endpoint=endpoint_p1.value,
            temperature=temp_p1.value,
            top_p=top_p_p1.value,
            system_prompt=system_p1.value if system_p1.value else None,
        ),
        PlayerConfig(
            player_id=2,
            model=model_p2.value,
            endpoint=endpoint_p2.value,
            temperature=temp_p2.value,
            top_p=top_p_p2.value,
            system_prompt=system_p2.value if system_p2.value else None,
        ),
    ]

    # Log player configurations
    activity_log.append({
        "timestamp": datetime.now().isoformat(),
        "level": "INFO",
        "message": f"P1: {model_p1.value} @ {endpoint_p1.value} (temp={temp_p1.value})"
    })
    activity_log.append({
        "timestamp": datetime.now().isoformat(),
        "level": "INFO",
        "message": f"P2: {model_p2.value} @ {endpoint_p2.value} (temp={temp_p2.value})"
    })

    # Create session
    session = session_manager.create_session(
        game_type=game_type_selector.value,
        players=players,
        num_rounds=num_games.value,
    )

    # Store custom payoffs in session config
    session.config = {
        "custom_payoffs": session_custom_payoffs.get(game_type_selector.value, {}),
        "player_settings": {
            "p1": {"temperature": temp_p1.value, "top_p": top_p_p1.value},
            "p2": {"temperature": temp_p2.value, "top_p": top_p_p2.value},
        },
    }

    activity_log.append({
        "timestamp": datetime.now().isoformat(),
        "level": "INFO",
        "message": f"Session {session.session_id} started"
    })
    activity_log.append({
        "timestamp": datetime.now().isoformat(),
        "level": "INFO",
        "message": f"Game: {game.name} | Rounds: {num_games.value}"
    })
    activity_log.append({
        "timestamp": datetime.now().isoformat(),
        "level": "INFO",
        "message": f"P1: {model_p1.value} @ {endpoint_p1.value}"
    })
    activity_log.append({
        "timestamp": datetime.now().isoformat(),
        "level": "INFO",
        "message": f"P2: {model_p2.value} @ {endpoint_p2.value}"
    })

    # Run game series
    runner = GameRunner(game, metrics)

    start_time = time.time()

    # Run with progress bar
    async def run_with_progress():
        results = []
        history = []
        history_payoffs = []  # Track payoffs from each round
        cumulative_payoffs = [0, 0]  # Running total for each player
        # One-off mode: each game is independent (no history)
        is_repeated = runtime_selector.value in ("repeated", "sequential")
        metrics.log_request_start(num_games.value * 2)

        import aiohttp

        async with aiohttp.ClientSession() as aio_session:
            for round_num in mo.status.progress_bar(
                range(num_games.value),
                title="Running Games",
                subtitle=f"{model_p1.value} vs {model_p2.value}",
            ):
                # For one-off mode, don't pass history to prompt
                actions, payoffs, response_times, prompts, raw_responses = await runner.play_round(
                    aio_session, players,
                    history if is_repeated else [],
                    payoff_display=payoff_display.value,
                    history_payoffs=history_payoffs if is_repeated else None,
                    cumulative_payoffs=tuple(cumulative_payoffs) if is_repeated else None,
                    is_repeated=is_repeated,
                )

                # Store prompts for Prompt Log display
                prompt_log.append({
                    "round": round_num + 1,
                    "player": 1,
                    "model": model_p1.value,
                    "prompt": prompts[0],
                    "response": raw_responses[0],
                })
                prompt_log.append({
                    "round": round_num + 1,
                    "player": 2,
                    "model": model_p2.value,
                    "prompt": prompts[1],
                    "response": raw_responses[1],
                })

                # Log round results with request-level detail
                activity_log.append({
                    "timestamp": datetime.now().isoformat(),
                    "level": "SUCCESS",
                    "message": f"API P1 {model_p1.value}: {response_times[0]:.2f}s"
                })
                activity_log.append({
                    "timestamp": datetime.now().isoformat(),
                    "level": "SUCCESS",
                    "message": f"API P2 {model_p2.value}: {response_times[1]:.2f}s"
                })
                activity_log.append({
                    "timestamp": datetime.now().isoformat(),
                    "level": "INFO",
                    "message": f"Round {round_num+1}: P1→{actions[0]} ({payoffs[0]}), P2→{actions[1]} ({payoffs[1]})"
                })

                result = {
                    "game_number": round_num + 1,
                    "player1_action": actions[0],
                    "player2_action": actions[1],
                    "player1_payoff": payoffs[0],
                    "player2_payoff": payoffs[1],
                    "player1_model": model_p1.value,
                    "player2_model": model_p2.value,
                    "player1_endpoint": endpoint_p1.value,
                    "player2_endpoint": endpoint_p2.value,
                    "player1_response_time": response_times[0],
                    "player2_response_time": response_times[1],
                }
                results.append(result)
                history.append(actions)
                history_payoffs.append(payoffs)
                cumulative_payoffs[0] += payoffs[0]
                cumulative_payoffs[1] += payoffs[1]

        return results

    # Execute
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
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

    # Add cumulative payoffs
    results_df = results_df.with_columns([
        pl.col("player1_payoff").cum_sum().alias("cumulative_payoff_player1"),
        pl.col("player2_payoff").cum_sum().alias("cumulative_payoff_player2"),
    ])

    # Calculate summary stats
    action_counts = results_df.group_by(["player1_action", "player2_action"]).agg(
        pl.len().alias("count")
    )

    total_p1 = results_df["player1_payoff"].sum()
    total_p2 = results_df["player2_payoff"].sum()
    avg_p1 = results_df["player1_payoff"].mean()
    avg_p2 = results_df["player2_payoff"].mean()

    # Get API metrics
    metrics_data = metrics.to_dict()

    # Build error callout if there were failures
    error_callout = None
    if metrics_data["failed_requests"] > 0:
        error_callout = mo.callout(
            mo.md(f"**Warning:** {metrics_data['failed_requests']} API call(s) failed"),
            kind="warn",
        )

    # Display results - simplified summary (charts moved to Analytics tab)
    result_elements = [
        mo.md("## Results"),
        mo.md(f"**Session:** {session.session_id} | **Games:** {len(results)} | **Time:** {elapsed:.2f}s"),
    ]

    # Add error callout if present
    if error_callout:
        result_elements.append(error_callout)

    result_elements.extend([
        # Inline metrics - game stats and API stats combined
        mo.hstack([
            mo.stat(label="P1 Total", value=str(total_p1)),
            mo.stat(label="P2 Total", value=str(total_p2)),
            mo.stat(label="P1 Avg", value=f"{avg_p1:.2f}"),
            mo.stat(label="P2 Avg", value=f"{avg_p2:.2f}"),
            mo.stat(label="API Calls", value=str(metrics_data["total_requests"])),
            mo.stat(label="Success", value=f"{metrics_data['success_rate']:.0f}%"),
            mo.stat(label="Avg Resp", value=f"{metrics_data['avg_response_time']:.2f}s"),
        ], justify="start"),
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
def _(AnalyticsPanelBuilder, analytics_service, game_type_selector, mo):
    # Analytics tab content using new services - filtered by selected game
    builder = AnalyticsPanelBuilder
    selected_game = game_type_selector.value
    data = analytics_service.get_dashboard_data(game_type=selected_game)

    if not data["has_data"]:
        analytics_content = mo.vstack([
            mo.md("## Analytics Dashboard"),
            mo.md(f"**Showing:** {selected_game}"),
            mo.callout(
                mo.md("No session data available for this game. Run some games in the **Play** tab!"),
                kind="info",
            ),
        ])
    else:
        analytics_content = mo.vstack([
            mo.md("## Analytics Dashboard"),
            mo.md(f"**Showing:** {selected_game} | **Sessions:** {data['sessions_count']}"),
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
            for s in sessions_list[:20]  # Limit to 20 most recent
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
def _(AnalyticsPanelBuilder, analytics_service, mo, session_selector):
    # Session detail view
    selected_session_id = session_selector.value

    if selected_session_id:
        session_results = analytics_service.get_session_results(selected_session_id)
        if session_results is not None and not session_results.is_empty():
            session_detail_content = AnalyticsPanelBuilder.build_session_detail_section(
                session_results, selected_session_id
            )
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
