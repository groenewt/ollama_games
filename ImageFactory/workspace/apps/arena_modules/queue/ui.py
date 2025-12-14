"""UI builder components for the game queue system."""
from typing import List, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .models import QueuedGame


class QueueUIBuilder:
    """Static factory methods for queue UI components."""

    @staticmethod
    def build_queue_controls(
        queue_length: int,
        is_executing: bool,
        mo,
    ):
        """Build Add to Queue and Execute All buttons."""
        add_btn = mo.ui.button(
            label="Add to Queue",
            disabled=is_executing,
            kind="neutral",
        )
        execute_btn = mo.ui.button(
            label=f"Execute All ({queue_length})" if queue_length > 0 else "Execute All",
            disabled=is_executing or queue_length == 0,
            kind="success",
        )
        clear_btn = mo.ui.button(
            label="Clear Queue",
            disabled=is_executing or queue_length == 0,
            kind="danger",
        )
        return add_btn, execute_btn, clear_btn

    @staticmethod
    def build_queue_table(
        queue: List["QueuedGame"],
        current_idx: int,
        mo,
    ):
        """Build queue display as interactive table."""
        if not queue:
            return mo.callout(
                mo.md("Queue is empty. Configure a game and click **Add to Queue**."),
                kind="info",
            )

        rows = []
        for i, game in enumerate(queue):
            status_icon = {
                "pending": "...",
                "running": ">>",
                "completed": "[OK]",
                "failed": "[X]",
            }.get(game.status, "?")

            players_summary = " vs ".join(
                p.get("role_name") or p.get("model", "?")[:12]
                for p in game.players[:2]
            )

            rows.append({
                "#": i + 1,
                "Status": status_icon,
                "Game": game.game_name[:20],
                "Mode": game.runtime_mode,
                "Rounds": game.num_games,
                "Players": players_summary[:30],
            })

        return mo.ui.table(rows, selection="single", label="Game Queue")

    @staticmethod
    def build_queue_item_actions(
        selected_idx: Optional[int],
        queue_length: int,
        is_executing: bool,
        mo,
    ):
        """Build move up/down/remove buttons for selected item."""
        can_move_up = selected_idx is not None and selected_idx > 0
        can_move_down = selected_idx is not None and selected_idx < queue_length - 1
        has_selection = selected_idx is not None

        move_up_btn = mo.ui.button(
            label="Move Up",
            disabled=not can_move_up or is_executing,
            kind="neutral",
        )
        move_down_btn = mo.ui.button(
            label="Move Down",
            disabled=not can_move_down or is_executing,
            kind="neutral",
        )
        remove_btn = mo.ui.button(
            label="Remove",
            disabled=not has_selection or is_executing,
            kind="danger",
        )
        return move_up_btn, move_down_btn, remove_btn

    @staticmethod
    def build_execution_progress(
        queue: List["QueuedGame"],
        current_idx: int,
        is_executing: bool,
        mo,
    ):
        """Build progress display during queue execution."""
        if not is_executing:
            return mo.md("")

        current_game = queue[current_idx] if 0 <= current_idx < len(queue) else None
        completed = sum(1 for g in queue if g.status == "completed")

        elements = [
            mo.md(f"### Executing Queue ({completed + 1}/{len(queue)})"),
        ]

        if current_game:
            elements.append(
                mo.callout(
                    mo.md(f"**Running:** {current_game.game_name} - {current_game.runtime_mode} ({current_game.num_games} rounds)"),
                    kind="info",
                )
            )

        return mo.vstack(elements)

    @staticmethod
    def build_queue_results_summary(
        queue: List["QueuedGame"],
        queue_results: Dict[str, Dict[str, Any]],
        mo,
    ):
        """Build aggregated results from queue execution."""
        completed = [g for g in queue if g.status == "completed"]
        if not completed:
            return mo.md("")

        total_rounds = sum(r.get("rounds", 0) for r in queue_results.values())
        total_time = sum(r.get("elapsed", 0) for r in queue_results.values())

        rows = []
        for game in completed:
            results = queue_results.get(game.queue_id, {})
            rows.append({
                "Game": game.game_name[:20],
                "Rounds": results.get("rounds", 0),
                "Session": (results.get("session_id") or "N/A")[:8],
                "Time": f"{results.get('elapsed', 0):.1f}s",
            })

        return mo.vstack([
            mo.md("### Queue Execution Results"),
            mo.hstack([
                mo.stat(label="Games", value=str(len(completed))),
                mo.stat(label="Total Rounds", value=str(total_rounds)),
                mo.stat(label="Total Time", value=f"{total_time:.1f}s"),
            ], justify="start"),
            mo.ui.table(rows, label="Completed Games"),
        ])
