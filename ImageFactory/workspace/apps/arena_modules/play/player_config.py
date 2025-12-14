"""Player configuration UI builder for arena."""

from typing import Optional, Any


def build_player_section(
    player_num: int,
    role_dropdown: Any,
    selected_role: Optional[Any],
    model_dropdown: Any,
    endpoint_dropdown: Any,
    temp_slider: Any,
    top_p_slider: Any,
    top_k_slider: Any,
    repeat_slider: Any,
    system_textarea: Any,
    strategy_textarea: Any,
    mo,
) -> 'mo.Html':
    """Build configuration section for one player.

    Supports both role-based (preset) and manual configuration.
    When a role is selected, shows role summary with override option.
    When manual, shows full configuration controls.

    Args:
        player_num: Player number (1, 2, or 3)
        role_dropdown: Marimo dropdown for role selection
        selected_role: RoleConfig if role is selected, None for manual
        model_dropdown: Marimo dropdown for model selection
        endpoint_dropdown: Marimo dropdown for endpoint selection
        temp_slider: Temperature slider
        top_p_slider: Top-p slider
        top_k_slider: Top-k slider
        repeat_slider: Repeat penalty slider
        system_textarea: System prompt textarea
        strategy_textarea: Strategy hints textarea
        mo: Marimo module

    Returns:
        Marimo vstack with player configuration UI
    """
    # Manual config accordion with advanced settings
    manual_advanced = mo.accordion({
        "Advanced Settings": mo.vstack([
            mo.hstack([temp_slider, top_p_slider]),
            mo.hstack([top_k_slider, repeat_slider]),
            system_textarea,
            strategy_textarea,
        ])
    })
    manual_config = mo.vstack([
        mo.hstack([model_dropdown, endpoint_dropdown]),
        manual_advanced,
    ])

    # Role summary (when role is selected)
    if selected_role:
        role_summary = mo.callout(
            mo.md(
                f"**{selected_role.name}**  \n"
                f"{selected_role.model} @ {selected_role.endpoint.split('/')[-1]}"
            ),
            kind="success",
        )
        # Only show strategy hints override for roles
        role_config = mo.vstack([
            role_summary,
            mo.accordion({
                "Session Override": mo.vstack([strategy_textarea]),
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
