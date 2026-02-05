#!/usr/bin/env python3
"""CLI tool to view model metadata."""

import json
from pathlib import Path

import click
from rich import box
from rich.console import Console
from rich.table import Table
from safetensors import safe_open

from style_search.config import MODELS_DIR


def format_value(key: str, value) -> str:
    """Format a value for display."""
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        if "accuracy" in key or "split" in key:
            return f"{value:.1%}"
        return f"{value:g}"
    if isinstance(value, list):
        if key == "datasets":
            return ", ".join(value)
        if len(value) <= 3:
            return ", ".join(str(v) for v in value)
        return f"[{len(value)} values]"
    return str(value)


def plot_loss(loss_history: list[float], console: Console, width: int) -> None:
    """Plot loss history using plotext integrated with Rich."""
    import plotext as plt
    from rich.align import Align
    from rich.ansi import AnsiDecoder
    from rich.console import Group
    from rich.panel import Panel

    plot_width = width - 4

    plt.clf()
    plt.plot(loss_history, marker="braille")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.plotsize(plot_width, 12)
    plt.theme("dark")

    canvas = plt.build()
    decoder = AnsiDecoder()
    rich_canvas = Group(*decoder.decode(canvas))

    panel = Panel(
        Align.center(rich_canvas),
        title="Loss",
        border_style="dim",
        width=width,
        box=box.ROUNDED,
    )
    console.print(panel)


def load_metadata(path: Path) -> dict:
    """Load metadata from a safetensors file and its JSON sidecar."""
    with safe_open(path, framework="pt") as f:
        meta = f.metadata() or {}

    json_path = path.with_suffix(".json")
    if json_path.exists():
        with open(json_path) as f:
            json_meta = json.load(f)
        for key, value in json_meta.items():
            meta[key] = value

    return meta


def display_model(path: Path, console: Console, as_json: bool = False) -> None:
    """Display model metadata and loss graph."""
    meta = load_metadata(path)

    if as_json:
        console.print(json.dumps(meta, indent=2))
        return

    table = Table(show_header=False, box=box.ROUNDED, padding=(0, 2))
    table.add_column("Key", style="cyan")
    table.add_column("Value")

    meta["file"] = str(path)

    groups = {
        "Model": ["file", "version", "datasets", "dataset", "dim", "timestamp"],
        "Training": [
            "epochs", "learning_rate", "margin", "batch_size",
            "l2_weight", "equality_weight", "balance_datasets",
        ],
        "Data": [
            "num_triplets", "num_train_triplets", "num_test_triplets",
            "num_equality_constraints", "test_split", "seed",
        ],
        "Results": ["train_accuracy", "test_accuracy", "baseline_accuracy"],
    }

    seen = set()

    for group_name, keys in groups.items():
        group_items = []
        for key in keys:
            if key in meta:
                group_items.append((key, meta[key]))
                seen.add(key)

        if group_items:
            table.add_row(f"[bold]{group_name}[/bold]", "")
            for key, value in group_items:
                table.add_row(f"  {key}", format_value(key, value))

    remaining = [(k, v) for k, v in meta.items() if k not in seen and k != "loss_history"]
    if remaining:
        table.add_row("[bold]Other[/bold]", "")
        for key, value in sorted(remaining):
            table.add_row(f"  {key}", format_value(key, value))

    table_width = console.measure(table).maximum

    console.print(table)

    if "loss_history" in meta and isinstance(meta["loss_history"], list):
        loss_history = meta["loss_history"]
        if loss_history:
            console.print()
            plot_loss(loss_history, console, table_width)


def discover_models() -> list[tuple[str, Path]]:
    """Discover all models in the models directory."""
    if not MODELS_DIR.exists():
        return []

    models = []
    for path in sorted(MODELS_DIR.glob("**/*.safetensors")):
        rel_path = path.relative_to(MODELS_DIR)
        name = str(rel_path.with_suffix(""))
        models.append((name, path))

    return models


def run_browser() -> None:
    """Run the interactive TUI browser."""
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical, VerticalScroll
    from textual.widgets import Footer, Header, ListItem, ListView, Static

    from rich.panel import Panel
    from rich.table import Table

    class ModelDetails(Static):
        """Widget to display model details."""

        def __init__(self, id: str | None = None) -> None:
            super().__init__("", id=id)
            self.current_path: Path | None = None

        def show_model(self, path: Path) -> None:
            """Display the given model's details."""
            self.current_path = path
            meta = load_metadata(path)

            # Build a table for the details
            table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
            table.add_column("Key", style="cyan")
            table.add_column("Value")

            groups = {
                "Model": ["version", "datasets", "dataset", "dim", "timestamp"],
                "Training": [
                    "epochs", "learning_rate", "margin", "batch_size",
                    "l2_weight", "equality_weight", "balance_datasets",
                ],
                "Data": [
                    "num_triplets", "num_train_triplets", "num_test_triplets",
                    "num_equality_constraints", "test_split", "seed",
                ],
                "Results": ["train_accuracy", "test_accuracy", "baseline_accuracy"],
            }

            seen = set()
            for group_name, keys in groups.items():
                group_items = []
                for key in keys:
                    if key in meta:
                        group_items.append((key, meta[key]))
                        seen.add(key)

                if group_items:
                    table.add_row(f"[bold]{group_name}[/bold]", "")
                    for key, value in group_items:
                        table.add_row(f"  {key}", format_value(key, value))

            # Show loss summary instead of graph
            if "loss_history" in meta and isinstance(meta["loss_history"], list):
                loss = meta["loss_history"]
                if loss:
                    table.add_row("[bold]Loss[/bold]", "")
                    table.add_row("  start", f"{loss[0]:.4f}")
                    table.add_row("  end", f"{loss[-1]:.4f}")
                    table.add_row("  reduction", f"{(1 - loss[-1]/loss[0]):.1%}")

            self.update(Panel(table, title=path.name, border_style="green"))

    class ModelBrowser(App):
        """Two-pane model browser."""

        CSS = """
        Horizontal {
            height: 100%;
        }
        #model-list {
            width: 40;
            border: solid green;
        }
        #model-details {
            width: 1fr;
        }
        ListView {
            height: 100%;
        }
        ListItem {
            padding: 0 1;
        }
        ListItem.--highlight {
            background: $accent;
        }
        """

        BINDINGS = [
            Binding("q", "quit", "Quit"),
            Binding("j", "cursor_down", "Down", show=False),
            Binding("k", "cursor_up", "Up", show=False),
        ]

        def __init__(self) -> None:
            super().__init__()
            self.models = discover_models()

        def compose(self) -> ComposeResult:
            yield Header()
            with Horizontal():
                with Vertical(id="model-list"):
                    items = []
                    for name, path in self.models:
                        # Get accuracy for display
                        try:
                            meta = load_metadata(path)
                            acc = meta.get("test_accuracy") or meta.get("train_accuracy")
                            acc_str = f" ({acc:.0%})" if acc else ""
                        except Exception:
                            acc_str = ""

                        # Shorten name for display
                        display = name
                        if len(display) > 35:
                            display = "..." + display[-32:]

                        item = ListItem(Static(f"{display}{acc_str}"))
                        item.data = path  # Store path for later
                        items.append(item)

                    yield ListView(*items, id="list")
                yield ModelDetails(id="model-details")
            yield Footer()

        def on_mount(self) -> None:
            """Select first model on mount."""
            if self.models:
                self.query_one(ModelDetails).show_model(self.models[0][1])

        def on_list_view_selected(self, event: ListView.Selected) -> None:
            """Handle model selection."""
            path = event.item.data
            if path:
                self.query_one(ModelDetails).show_model(path)

        def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
            """Handle model highlight (cursor movement)."""
            if event.item and hasattr(event.item, "data"):
                path = event.item.data
                if path:
                    self.query_one(ModelDetails).show_model(path)

        def action_cursor_down(self) -> None:
            self.query_one(ListView).action_cursor_down()

        def action_cursor_up(self) -> None:
            self.query_one(ListView).action_cursor_up()

    app = ModelBrowser()
    app.run()


@click.command()
@click.argument("model_path", type=click.Path(exists=True), required=False)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def main(model_path: str | None, as_json: bool) -> None:
    """View metadata for a safetensors model file.

    If MODEL_PATH is provided, display that model's info.
    If no path is given, launch an interactive model browser.
    """
    console = Console()

    if model_path is None:
        run_browser()
        return

    path = Path(model_path)

    if path.is_dir():
        versions = sorted(path.glob("v*.safetensors"))
        if not versions:
            raise click.ClickException(f"No model versions found in {path}")
        path = versions[-1]

    display_model(path, console, as_json=as_json)


if __name__ == "__main__":
    main()
