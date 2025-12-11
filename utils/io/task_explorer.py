"""
TUI for exploring ARC tasks.

Usage:
    uv run python -m utils.io.task_explorer
"""

import subprocess
import sys
from pathlib import Path
from typing import cast

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, ScrollableContainer
from textual.coordinate import Coordinate
from textual.screen import Screen
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Static,
    Label,
)

from constants import DATA
from utils.io.loader import path_to_task
from arc.types import ColorGrid

# Path to problem.py relative to this file
PROBLEM_PY = Path(__file__).parent.parent.parent / "problem.py"

# ARC color palette (RGB values)
ARC_COLORS = {
    0: (0, 0, 0),  # Black
    1: (30, 147, 255),  # Blue
    2: (249, 60, 49),  # Red
    3: (79, 204, 48),  # Green
    4: (255, 220, 0),  # Yellow
    5: (153, 153, 153),  # Gray
    6: (229, 58, 163),  # Magenta
    7: (255, 133, 27),  # Orange
    8: (135, 216, 241),  # Cyan
    9: (146, 18, 49),  # Maroon
}


def grid_to_rich_text(grid: ColorGrid, cell_width: int = 2) -> Text:
    """Convert a ColorGrid to a Rich Text object with colored blocks."""
    text = Text()
    for row in grid:
        for cell in row:
            r, g, b = ARC_COLORS.get(cell, (128, 128, 128))
            text.append(" " * cell_width, style=f"on rgb({r},{g},{b})")
        text.append("\n")
    return text


def get_training_tasks() -> list[str]:
    """Get list of all training task filenames."""
    training_path = Path(DATA) / "training"
    if not training_path.exists():
        return []
    return sorted([f.name for f in training_path.iterdir() if f.suffix == ".json"])


class GridWidget(Static):
    """Widget to display a single ARC grid."""

    def __init__(self, grid: ColorGrid, title: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self.grid = grid
        self.title = title

    def compose(self) -> ComposeResult:
        if self.title:
            yield Label(self.title, classes="grid-title")
        yield Static(grid_to_rich_text(self.grid), classes="grid-display")


class ExampleWidget(Static):
    """Widget to display an input-output pair."""

    def __init__(
        self, input_grid: ColorGrid, output_grid: ColorGrid, index: int, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.input_grid = input_grid
        self.output_grid = output_grid
        self.index = index

    def compose(self) -> ComposeResult:
        yield Label(f"Example {self.index + 1}", classes="example-title")
        with Horizontal(classes="example-grids"):
            with Vertical(classes="grid-container"):
                yield Label("Input", classes="grid-label")
                yield Static(grid_to_rich_text(self.input_grid), classes="grid-display")
            with Vertical(classes="grid-container"):
                yield Label("Output", classes="grid-label")
                yield Static(
                    grid_to_rich_text(self.output_grid), classes="grid-display"
                )


class TaskDetailScreen(Screen):
    """Screen showing details of a single task."""

    BINDINGS = [
        Binding("escape", "pop_screen", "Back"),
        Binding("q", "pop_screen", "Back"),
        Binding("r", "run_solver", "Run Solver"),
    ]

    CSS = """
    TaskDetailScreen {
        background: $surface;
    }

    .task-title {
        text-align: center;
        text-style: bold;
        padding: 1;
        background: $primary;
        color: $text;
        width: 100%;
    }

    .example-title {
        text-style: bold;
        padding: 1 0;
        color: $secondary;
    }

    .example-grids {
        height: auto;
        padding: 0 2;
    }

    .grid-container {
        padding: 0 2;
        height: auto;
        width: auto;
    }

    .grid-label {
        text-style: italic;
        color: $text-muted;
    }

    .grid-display {
        padding: 0;
        height: auto;
        width: auto;
    }

    .test-section {
        margin-top: 2;
        border-top: solid $primary;
        padding-top: 1;
    }

    .test-title {
        text-style: bold;
        color: $warning;
        padding: 1 0;
    }

    #content {
        height: 100%;
        padding: 1 2;
    }
    """

    def __init__(self, task_name: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.task_name = task_name

    def compose(self) -> ComposeResult:
        yield Header()
        yield ScrollableContainer(id="content")
        yield Footer()

    def on_mount(self) -> None:
        """Load and display task data."""
        container = self.query_one("#content")

        try:
            task_data = path_to_task(f"training/{self.task_name}")
        except FileNotFoundError:
            container.mount(Label(f"Task not found: {self.task_name}"))
            return

        # Task title
        container.mount(Label(f"Task: {self.task_name}", classes="task-title"))

        # Training examples
        for i, example in enumerate(task_data.get("train", [])):
            container.mount(
                ExampleWidget(example["input"], example["output"], i, classes="example")
            )

        # Test examples
        test_examples = task_data.get("test", [])
        if test_examples:
            container.mount(Label("Test Cases", classes="test-title test-section"))
            for i, example in enumerate(test_examples):
                input_grid = example["input"]
                output_grid = example.get("output", [[]])
                container.mount(
                    ExampleWidget(
                        input_grid, output_grid, i, classes="example test-example"
                    )
                )

    def action_run_solver(self) -> None:
        """Run problem.py on this task."""
        app = cast("TaskExplorerApp", self.app)
        app.run_solver(self.task_name)


class TaskListScreen(Screen):
    """Main screen showing list of tasks."""

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("enter", "select_task", "View Task"),
        Binding("r", "run_solver", "Run Solver"),
        Binding("/", "focus_search", "Search"),
    ]

    CSS = """
    TaskListScreen {
        background: $surface;
    }

    #task-table {
        height: 100%;
    }

    DataTable {
        height: 100%;
    }

    DataTable > .datatable--header {
        text-style: bold;
        background: $primary;
    }

    DataTable > .datatable--cursor {
        background: $secondary;
    }
    """

    def compose(self) -> ComposeResult:
        yield Header()
        yield DataTable(id="task-table")
        yield Footer()

    def on_mount(self) -> None:
        """Populate the task table."""
        table = self.query_one(DataTable)
        table.cursor_type = "row"
        table.add_columns("Task ID", "Train Examples", "Test Examples")

        tasks = get_training_tasks()

        for task_name in tasks:
            try:
                task_data = path_to_task(f"training/{task_name}")
                train_count = len(task_data.get("train", []))
                test_count = len(task_data.get("test", []))
                table.add_row(
                    task_name.replace(".json", ""),
                    str(train_count),
                    str(test_count),
                    key=task_name,
                )
            except Exception:
                table.add_row(task_name.replace(".json", ""), "?", "?", key=task_name)

    def action_select_task(self) -> None:
        """Open the selected task."""
        table = self.query_one(DataTable)
        if table.cursor_row is not None:
            row_key = table.get_row_at(table.cursor_row)
            if row_key:
                coord = Coordinate(table.cursor_row, 0)
                task_name = str(table.get_cell_at(coord)) + ".json"
                self.app.push_screen(TaskDetailScreen(task_name))

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle double-click on a row."""
        if event.row_key:
            task_name = str(event.row_key.value)
            self.app.push_screen(TaskDetailScreen(task_name))

    def get_selected_task(self) -> str | None:
        """Get the currently selected task name."""
        table = self.query_one(DataTable)
        if table.cursor_row is not None:
            coord = Coordinate(table.cursor_row, 0)
            return str(table.get_cell_at(coord)) + ".json"
        return None

    def action_run_solver(self) -> None:
        """Run problem.py on the selected task."""
        task_name = self.get_selected_task()
        if task_name:
            app = cast("TaskExplorerApp", self.app)
            app.run_solver(task_name)


class TaskExplorerApp(App):
    """TUI application for exploring ARC tasks."""

    TITLE = "ARC Task Explorer"
    SUB_TITLE = "Browse and visualize ARC-AGI tasks"

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("d", "toggle_dark", "Toggle Dark Mode"),
    ]

    CSS = """
    Screen {
        background: $surface;
    }
    """

    def on_mount(self) -> None:
        """Push the main screen."""
        self.push_screen(TaskListScreen())

    def action_toggle_dark(self) -> None:
        """Toggle dark mode."""
        self.theme = "textual-light" if self.theme == "textual-dark" else "textual-dark"

    def run_solver(self, task_name: str) -> None:
        """Run problem.py on the given task, suspending the TUI."""
        with self.suspend():
            print(f"\n{'=' * 60}")
            print(f"Running solver on: {task_name}")
            print(f"{'=' * 60}\n")

            result = subprocess.run(
                [sys.executable, str(PROBLEM_PY), "--task", task_name],
                cwd=PROBLEM_PY.parent,
            )

            print(f"\n{'=' * 60}")
            print(f"Solver finished with exit code: {result.returncode}")
            print("Press Enter to return to the task explorer...")
            print(f"{'=' * 60}")
            input()


def main():
    """Run the task explorer."""
    app = TaskExplorerApp()
    app.run()


if __name__ == "__main__":
    main()
