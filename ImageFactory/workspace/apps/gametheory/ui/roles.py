"""Roles tab UI components for managing player identities."""

from typing import Dict, List, Optional, Any, Tuple
import marimo as mo

from ..core.role import RoleConfig, extract_strategic_considerations


class RolesTabBuilder:
    """Static factory methods for building Roles tab UI sections."""

    @staticmethod
    def build_role_form(
        available_endpoints: List[str],
        endpoint_models: Dict[str, List[str]],
        game_names: Dict[str, str],
        game_descriptions: Dict[str, str],
        game_prompt_instructions: Optional[Dict[str, str]] = None,
        existing_role: Optional[RoleConfig] = None,
    ) -> Dict[str, Any]:
        """Build role creation/edit form elements.

        Args:
            available_endpoints: List of discovered Ollama endpoints
            endpoint_models: Map of endpoint -> available models
            game_names: Map of game_id -> display name
            game_descriptions: Map of game_id -> full description (for extracting defaults)
            game_prompt_instructions: Map of game_id -> action_space.prompt_instructions()
            existing_role: If editing, the role to populate form with

        Returns:
            Dictionary of form UI elements
        """
        is_edit = existing_role is not None

        # Basic info
        name_input = mo.ui.text(
            label="Role Name *",
            value=existing_role.name if is_edit else "",
            placeholder="e.g., AggressiveBlottoPlayer",
        )

        description_input = mo.ui.text_area(
            label="Description",
            value=existing_role.description if is_edit else "",
            placeholder="Brief description of this role's strategy or purpose",
        )

        # Model configuration
        default_endpoint = (
            existing_role.endpoint
            if is_edit
            else (available_endpoints[0] if available_endpoints else "")
        )
        endpoint_dropdown = mo.ui.dropdown(
            options={ep: ep for ep in available_endpoints},
            label="Endpoint *",
            value=default_endpoint if default_endpoint in available_endpoints else None,
        )

        # Model dropdown - get models for current endpoint
        current_models = endpoint_models.get(default_endpoint, [])
        if not current_models:
            # Flatten all models if no specific endpoint models
            current_models = list(
                set(m for models in endpoint_models.values() for m in models)
            )
        default_model = (
            existing_role.model
            if is_edit and existing_role.model in current_models
            else (current_models[0] if current_models else "")
        )
        model_dropdown = mo.ui.dropdown(
            options={m: m for m in current_models},
            label="Model *",
            value=default_model if default_model in current_models else None,
        )

        # System prompt
        system_prompt_input = mo.ui.text_area(
            label="System Prompt",
            value=existing_role.system_prompt if is_edit else "",
            placeholder="Custom instructions for this role (e.g., 'You are an aggressive player who prefers concentrated attacks...')",
        )

        # Game constraints - multiselect
        games_multiselect = mo.ui.multiselect(
            options=game_names,
            label="Allowed Games (empty = all games)",
            value=(
                [g for g in existing_role.allowed_games if g in game_names]
                if is_edit
                else []
            ),
        )

        # Create text area for each game's strategy instructions
        # Pre-populate with existing instructions or extracted strategic considerations
        game_instruction_inputs = {}
        for game_id, game_name in game_names.items():
            # Get existing value or extract default from game description
            if is_edit and game_id in existing_role.game_instructions:
                default_value = existing_role.game_instructions[game_id]
            else:
                # Extract "Strategic considerations:" from game description
                default_value = extract_strategic_considerations(
                    game_descriptions.get(game_id, "")
                )

            game_instruction_inputs[game_id] = mo.ui.text_area(
                label=f"{game_name} Strategy",
                value=default_value,
                placeholder=f"Custom strategy instructions for {game_name}...",
            )

        # Default parameters
        temp_slider = mo.ui.slider(
            start=0.0,
            stop=2.0,
            value=existing_role.temperature if is_edit else 0.7,
            step=0.1,
            label="Temperature",
            show_value=True,
        )
        top_p_slider = mo.ui.slider(
            start=0.0,
            stop=1.0,
            value=existing_role.top_p if is_edit else 0.9,
            step=0.05,
            label="Top P",
            show_value=True,
        )
        top_k_slider = mo.ui.slider(
            start=1,
            stop=100,
            value=existing_role.top_k if is_edit else 40,
            step=1,
            label="Top K",
            show_value=True,
        )
        repeat_penalty_slider = mo.ui.slider(
            start=1.0,
            stop=2.0,
            value=existing_role.repeat_penalty if is_edit else 1.1,
            step=0.05,
            label="Repeat Penalty",
            show_value=True,
        )

        return {
            "name": name_input,
            "description": description_input,
            "endpoint": endpoint_dropdown,
            "model": model_dropdown,
            "system_prompt": system_prompt_input,
            "allowed_games": games_multiselect,
            "game_instructions": game_instruction_inputs,  # Dict[game_id, text_area]
            "game_prompt_instructions": game_prompt_instructions or {},  # Dict[game_id, str]
            "temperature": temp_slider,
            "top_p": top_p_slider,
            "top_k": top_k_slider,
            "repeat_penalty": repeat_penalty_slider,
        }

    @staticmethod
    def build_roles_table(
        roles: List[RoleConfig],
        model_stats: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> mo.Html:
        """Build the roles list as a selectable table with optional performance stats.

        Args:
            roles: List of RoleConfig objects
            model_stats: Optional dict of model -> stats (from analytics_service.get_model_stats_dict())
                         Keys: total_sessions, games_played, avg_payoff, coop_rate

        Returns:
            Marimo UI table element
        """
        if not roles:
            return mo.callout(
                mo.md("No roles defined yet. Create your first role below!"),
                kind="info",
            )

        model_stats = model_stats or {}
        rows = []
        for role in roles:
            games_str = (
                ", ".join(role.allowed_games[:2]) if role.allowed_games else "All"
            )
            if len(role.allowed_games) > 2:
                games_str += f" (+{len(role.allowed_games) - 2})"

            # Get stats for this role's model (cross-game, unfiltered)
            stats = model_stats.get(role.model, {})

            row = {
                "Name": role.name,
                "Model": role.model,
                "Games": games_str,
                "Sessions": stats.get("total_sessions", 0),
                "Avg Payoff": stats.get("avg_payoff", "-"),
                "Coop %": f"{stats.get('coop_rate', 0):.0f}" if stats.get("coop_rate") is not None else "-",
                "_role_id": role.role_id,  # Hidden for selection tracking
            }
            rows.append(row)

        return mo.ui.table(
            rows,
            selection="single",
            label="Available Roles (stats per model, all games)",
        )

    @staticmethod
    def build_role_detail_card(
        role: RoleConfig, game_names: Dict[str, str]
    ) -> mo.Html:
        """Build detailed view of a selected role.

        Args:
            role: The role to display
            game_names: Map of game_id -> display name

        Returns:
            Marimo vstack with role details
        """
        games_list = (
            [game_names.get(g, g) for g in role.allowed_games]
            if role.allowed_games
            else ["All games"]
        )

        prompt_display = (
            f"```\n{role.system_prompt}\n```"
            if role.system_prompt
            else "_No system prompt configured_"
        )

        return mo.vstack(
            [
                mo.md(f"### {role.name}"),
                (
                    mo.md(f"_{role.description}_")
                    if role.description
                    else mo.md("_No description_")
                ),
                mo.md("---"),
                mo.hstack(
                    [
                        mo.stat(label="Model", value=role.model),
                        mo.stat(label="Temperature", value=f"{role.temperature:.1f}"),
                        mo.stat(label="Top P", value=f"{role.top_p:.2f}"),
                        mo.stat(label="Top K", value=str(role.top_k)),
                    ],
                    justify="start",
                ),
                mo.md("---"),
                mo.md(f"**Endpoint:** `{role.endpoint}`"),
                mo.md("**System Prompt:**"),
                mo.md(prompt_display),
                mo.md("**Allowed Games:**"),
                mo.md(", ".join(games_list)),
            ]
        )

    @staticmethod
    def build_form_layout(
        form_elements: Dict[str, Any],
        selected_games: Optional[List[str]] = None,
        game_names: Optional[Dict[str, str]] = None,
    ) -> mo.Html:
        """Build the form layout from form elements.

        Args:
            form_elements: Dictionary of form UI elements from build_role_form
            selected_games: List of currently selected game display names (from multiselect)
            game_names: Dict of game_id -> display_name (for reverse mapping)

        Returns:
            Organized form layout
        """
        # Build game instruction boxes for selected games
        game_instructions_section = []
        prompt_instructions = form_elements.get("game_prompt_instructions", {})
        game_instructions_inputs = form_elements.get("game_instructions", {})

        # Create reverse mapping: display_name -> game_id
        # (marimo multiselect returns display names, but our dicts use game_ids as keys)
        name_to_id = {v: k for k, v in (game_names or {}).items()}

        if selected_games:
            game_instructions_section.append(mo.md("**Game-Specific Instructions**"))
            for display_name in selected_games:
                # Convert display name to game_id
                game_id = name_to_id.get(display_name, display_name)
                # Show prompt format callout (what LLM receives)
                if game_id in prompt_instructions:
                    game_instructions_section.append(
                        mo.callout(
                            mo.vstack([
                                mo.md(f"**{game_id} - LLM Prompt Format**"),
                                mo.md("_This is the format instruction the LLM will receive:_"),
                                mo.md(f"```\n{prompt_instructions[game_id]}\n```"),
                            ]),
                            kind="info",
                        )
                    )
                # Show strategy text area (editable)
                if game_id in game_instructions_inputs:
                    game_instructions_section.append(game_instructions_inputs[game_id])

        return mo.vstack(
            [
                mo.md("### Role Configuration"),
                # Basic info row
                mo.hstack(
                    [form_elements["name"], form_elements["description"]],
                    justify="start",
                    widths=[1, 2],
                ),
                mo.md("---"),
                # Model configuration row
                mo.md("**Model Configuration**"),
                mo.hstack(
                    [form_elements["endpoint"], form_elements["model"]],
                    justify="start",
                ),
                mo.md("---"),
                # System prompt
                mo.md("**System Prompt**"),
                form_elements["system_prompt"],
                mo.md("---"),
                # Game constraints
                mo.md("**Game Constraints**"),
                form_elements["allowed_games"],
                # Game-specific instructions (only shown for selected games)
                *game_instructions_section,
                mo.md("---"),
                # Parameters
                mo.md("**Default Parameters**"),
                mo.hstack(
                    [
                        form_elements["temperature"],
                        form_elements["top_p"],
                    ],
                    justify="start",
                ),
                mo.hstack(
                    [
                        form_elements["top_k"],
                        form_elements["repeat_penalty"],
                    ],
                    justify="start",
                ),
            ]
        )

    @staticmethod
    def extract_role_from_form(
        form_elements: Dict[str, Any],
        existing_role_id: Optional[str] = None,
    ) -> Tuple[Optional[RoleConfig], List[str]]:
        """Extract RoleConfig from form values.

        Args:
            form_elements: Dictionary of form UI elements
            existing_role_id: If editing, the existing role's ID

        Returns:
            Tuple of (RoleConfig or None if invalid, list of error messages)
        """
        errors = []

        name = form_elements["name"].value.strip() if form_elements["name"].value else ""
        endpoint = form_elements["endpoint"].value or ""
        model = form_elements["model"].value or ""

        if not name:
            errors.append("Role name is required")
        if not endpoint:
            errors.append("Endpoint is required")
        if not model:
            errors.append("Model is required")

        if errors:
            return None, errors

        # Build game_instructions dict from selected games and their text areas
        selected_games = list(form_elements["allowed_games"].value or [])
        game_instructions = {}
        if selected_games and "game_instructions" in form_elements:
            for game_id in selected_games:
                if game_id in form_elements["game_instructions"]:
                    text_area = form_elements["game_instructions"][game_id]
                    game_instructions[game_id] = (
                        text_area.value.strip() if text_area.value else ""
                    )

        role = RoleConfig(
            role_id=existing_role_id or RoleConfig().role_id,
            name=name,
            description=(
                form_elements["description"].value.strip()
                if form_elements["description"].value
                else ""
            ),
            endpoint=endpoint,
            model=model,
            system_prompt=(
                form_elements["system_prompt"].value.strip()
                if form_elements["system_prompt"].value
                else ""
            ),
            game_instructions=game_instructions,
            temperature=form_elements["temperature"].value,
            top_p=form_elements["top_p"].value,
            top_k=int(form_elements["top_k"].value),
            repeat_penalty=form_elements["repeat_penalty"].value,
        )

        validation_errors = role.validate()
        if validation_errors:
            return None, validation_errors

        return role, []

    @staticmethod
    def build_player_role_selector(
        player_num: int,
        roles: List[RoleConfig],
        game_id: Optional[str] = None,
        selected_role_id: Optional[str] = None,
    ) -> mo.ui.dropdown:
        """Build a role dropdown for player selection in Play tab.

        Args:
            player_num: Player number (1, 2, or 3)
            roles: List of available roles
            game_id: Optional game ID to filter roles
            selected_role_id: Optional pre-selected role ID

        Returns:
            Dropdown UI element for role selection
        """
        # Filter roles by game if specified
        if game_id:
            roles = [r for r in roles if r.can_play_game(game_id)]

        # Build options with manual config as first option
        options = {"-- Manual Configuration --": ""}
        for role in roles:
            options[f"{role.name} ({role.model})"] = role.role_id

        # Find default - marimo dropdown value expects option KEY (label), not VALUE
        default = "-- Manual Configuration --"
        if selected_role_id:
            for label, role_id in options.items():
                if role_id == selected_role_id:
                    default = label  # Use the label (key), not role_id (value)
                    break

        return mo.ui.dropdown(
            options=options,
            label=f"Player {player_num} Role",
            value=default,
        )
