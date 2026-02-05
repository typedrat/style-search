#!/usr/bin/env python3
"""CLI tool to view model metadata."""

import json
from pathlib import Path

import click
from rich import box
from rich.console import Console
from rich.table import Table
from safetensors import safe_open


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
        # Show short lists inline, summarize long ones
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

    # Account for panel borders (2) and padding (2)
    plot_width = width - 4

    plt.clf()
    plt.plot(loss_history, marker="braille")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.plotsize(plot_width, 12)
    plt.theme("dark")

    # Build the plot as ANSI string and decode for Rich
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


@click.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def main(model_path: str, as_json: bool) -> None:
    """View metadata for a safetensors model file.

    MODEL_PATH can be a .safetensors file or a directory containing versioned models.
    """
    path = Path(model_path)

    # If directory, find latest version
    if path.is_dir():
        versions = sorted(path.glob("v*.safetensors"))
        if not versions:
            raise click.ClickException(f"No model versions found in {path}")
        path = versions[-1]

    # Read safetensors metadata
    with safe_open(path, framework="pt") as f:
        meta = f.metadata() or {}

    # Check for JSON sidecar with additional metadata (preferred - has proper types)
    json_path = path.with_suffix(".json")
    if json_path.exists():
        with open(json_path) as f:
            json_meta = json.load(f)
        # Prefer JSON values over safetensors (JSON preserves types)
        for key, value in json_meta.items():
            meta[key] = value

    if as_json:
        output = Console()
        output.print(json.dumps(meta, indent=2))
        return

    # Build nice table
    table = Table(show_header=False, box=box.ROUNDED, padding=(0, 2))
    table.add_column("Key", style="cyan")
    table.add_column("Value")

    # Add file path to meta
    meta["file"] = str(path)

    # Group and order keys
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
    output = Console()

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

    # Any remaining keys (except loss_history, we'll show that separately)
    remaining = [(k, v) for k, v in meta.items() if k not in seen and k != "loss_history"]
    if remaining:
        table.add_row("[bold]Other[/bold]", "")
        for key, value in sorted(remaining):
            table.add_row(f"  {key}", format_value(key, value))

    # Measure table width for consistent sizing
    table_width = output.measure(table).maximum

    output.print(table)

    # Show loss graph if available
    if "loss_history" in meta and isinstance(meta["loss_history"], list):
        loss_history = meta["loss_history"]
        if loss_history:
            output.print()
            plot_loss(loss_history, output, table_width)


if __name__ == "__main__":
    main()
