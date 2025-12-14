"""Formatting utilities for arena activity and prompt logs."""

from typing import List, Dict, Any


def format_activity_log(log_entries: List[Dict[str, Any]], mo) -> 'mo.Html':
    """Format activity log as terminal-style output.

    Args:
        log_entries: List of log entry dicts with keys:
            - timestamp: ISO timestamp string
            - message: Log message
            - level: INFO, ERROR, WARN, SUCCESS
        mo: Marimo module for HTML rendering

    Returns:
        Marimo Html element with formatted log
    """
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


def format_prompt_log(prompt_entries: List[Dict[str, Any]], mo) -> 'mo.Html':
    """Format prompt log showing full prompts and responses.

    Args:
        prompt_entries: List of prompt entry dicts with keys:
            - round: Round number
            - player: Player number
            - model: Model name
            - prompt: The prompt text
            - response: The response text
            - system_prompt: Optional system prompt
            - was_parsed: Whether response was successfully parsed
        mo: Marimo module for HTML rendering

    Returns:
        Marimo Html element with expandable prompt sections
    """
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
        parse_badge = (
            '<span style="color: #69db7c;">✓ parsed</span>'
            if was_parsed
            else '<span style="color: #ff6b6b; font-weight: bold;">⚠ DEFAULT</span>'
        )

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
