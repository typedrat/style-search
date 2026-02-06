#!/usr/bin/env python3
"""Generate EVA02-CLIP embeddings and store in ChromaDB."""

import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import ClassVar

import chromadb
import click
import torch
from PIL import Image
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

console = Console()
cancelled = False

DEFAULT_WORKERS = os.cpu_count() or 4


def handle_sigint(signum, frame):
    global cancelled
    cancelled = True


class EVA02EmbeddingFunction:
    """Custom embedding function for EVA02-CLIP using open_clip."""

    # Map friendly names to open_clip model names and pretrained weights
    MODELS: ClassVar[dict[str, tuple[str, str]]] = {
        "EVA02-CLIP-L-14": ("EVA02-L-14", "merged2b_s4b_b131k"),
        "EVA02-CLIP-bigE-14": ("EVA02-E-14", "laion2b_s4b_b115k"),
        "EVA02-CLIP-bigE-14-plus": ("EVA02-E-14-plus", "laion2b_s9b_b144k"),
    }

    def __init__(
        self,
        model_name: str = "EVA02-CLIP-bigE-14-plus",
        device: str = "cuda",
    ):
        import open_clip

        self.device = device
        self.model_name = model_name

        model_arch, pretrained = self.MODELS[model_name]
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_arch,
            pretrained=pretrained,
            device=device,
            precision="fp16",
        )
        self.model.eval()

    def __call__(self, images: list[Image.Image]) -> list[list[float]]:
        with torch.no_grad():
            image_tensors = (
                torch.stack([self.preprocess(img) for img in images])  # pyright: ignore[reportCallIssue, reportArgumentType]
                .to(self.device)
                .half()
            )
            features = self.model.encode_image(image_tensors)  # pyright: ignore[reportCallIssue]
            features = features / features.norm(dim=-1, keepdim=True)
            return features.float().cpu().tolist()


@click.command()
@click.argument(
    "input_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "-d", "--db-path",
    type=click.Path(path_type=Path),
    help="ChromaDB path (default: INPUT_DIR/chroma)",
)
@click.option(
    "-m", "--model",
    type=click.Choice(["large", "big", "big-plus"]),
    default="big-plus",
    help="Model variant",
)
@click.option(
    "-b", "--batch-size", default=16,
    help="Batch size for inference",
)
@click.option("--reset", is_flag=True, help="Reset existing collection")
def embed(
    input_dir: Path,
    db_path: Path | None,
    model: str,
    batch_size: int,
    reset: bool,
):
    """Generate EVA02-CLIP embeddings for images and store in ChromaDB."""
    global cancelled
    signal.signal(signal.SIGINT, handle_sigint)

    db_path = db_path or input_dir / "chroma"
    metadata_path = input_dir / "metadata.json"

    if not metadata_path.exists():
        console.print(f"[red]Error: {metadata_path} not found[/red]")
        sys.exit(1)

    with open(metadata_path) as f:
        metadata = json.load(f)

    artists = metadata.get("artists", {})
    settings = metadata.get("settings", {})

    # Map model choice to full name
    model_names = {
        "large": "EVA02-CLIP-L-14",
        "big": "EVA02-CLIP-bigE-14",
        "big-plus": "EVA02-CLIP-bigE-14-plus",
    }
    model_name = model_names[model]

    console.print(Panel(
        f"[bold]EVA02-CLIP Embedding Generator[/bold]\n"
        f"Model: {model_name}\n"
        f"Artists: {len(artists)}\n"
        f"Batch size: {batch_size}",
        subtitle=f"DB: {db_path}"
    ))

    # Find images for each artist
    image_files = []
    for artist, data in artists.items():
        for ext in ('.jpg', '.jpeg', '.png', '.gif', '.webp'):
            safe_name = "".join(c if c not in '<>:"/\\|?*' else '_' for c in artist)
            img_path = input_dir / f"{safe_name}{ext}"
            if img_path.exists():
                image_files.append((artist, img_path, data))
                break

    console.print(f"[green]Found {len(image_files)} images[/green]\n")

    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=str(db_path))
    collection_name = input_dir.name

    if reset:
        try:
            client.delete_collection(collection_name)
            console.print(f"[yellow]Reset collection '{collection_name}'[/yellow]")
        except ValueError:
            pass

    # Check existing entries
    try:
        collection = client.get_collection(collection_name)
        existing_ids = set(collection.get()["ids"])
    except Exception:
        existing_ids = set()

    # Filter to only new images
    to_embed = [(a, p, d) for a, p, d in image_files if a not in existing_ids]

    if not to_embed:
        console.print("[green]All images already embedded![/green]")
        return

    n_skip = len(image_files) - len(to_embed)
    console.print(f"[dim]Skipping {n_skip} already embedded[/dim]")

    # Load model
    with console.status("[bold cyan]Loading EVA02-CLIP model..."):
        embed_fn = EVA02EmbeddingFunction(model_name)

    console.print("[green]Model loaded[/green]\n")

    # Create/get collection
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={
            "model": model_name,
            "source": str(input_dir),
            **{f"setting_{k}": v for k, v in settings.items()},
        }
    )

    # Process images in batches
    start_time = time.time()
    embedded_count = 0

    def format_speed(ips: float) -> str:
        return f"{ips:.1f} img/s"

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TextColumn("ETA"),
        TimeRemainingColumn(),
        TextColumn("•"),
        TextColumn("{task.fields[speed]}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            "[cyan]Generating embeddings...",
            total=len(to_embed),
            speed="0 img/s",
        )

        for i in range(0, len(to_embed), batch_size):
            if cancelled:
                console.print("\n[yellow]Cancelled. Partial results saved.[/yellow]")
                break

            batch = to_embed[i:i + batch_size]
            images = []
            batch_ids = []
            batch_metadata = []
            batch_uris = []

            for artist, img_path, data in batch:
                try:
                    img = Image.open(img_path).convert("RGB")
                    images.append(img)
                    batch_ids.append(artist)
                    batch_uris.append(img_path.name)
                    # Store artist metadata (excluding image_url which is in uri)
                    batch_metadata.append({
                        k: v for k, v in data.items()
                        if k != "image_url" and isinstance(v, (str, int, float, bool))
                    })
                except Exception as e:
                    console.print(f"[red]Failed to load {img_path}: {e}[/red]")

            if not images:
                continue

            # Generate embeddings
            embeddings = embed_fn(images)

            # Add to collection
            collection.add(
                ids=batch_ids,
                embeddings=embeddings,  # pyright: ignore[reportArgumentType]
                metadatas=batch_metadata,
                uris=batch_uris,
            )

            embedded_count += len(images)
            elapsed = time.time() - start_time
            speed = embedded_count / elapsed if elapsed > 0 else 0
            progress.update(task, advance=len(batch), speed=format_speed(speed))

    # Summary
    console.print()
    elapsed = time.time() - start_time
    total_in_db = collection.count()

    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="bold")
    summary.add_column()
    summary.add_row("[green]Embedded:", f"{embedded_count} images")
    summary.add_row("[dim]Total in DB:", str(total_in_db))
    summary.add_row("[dim]Time:", f"{elapsed:.1f}s")
    if elapsed > 0:
        summary.add_row("[dim]Speed:", f"{embedded_count / elapsed:.1f} img/s")
    summary.add_row("[blue]Database:", str(db_path))
    console.print(Panel(summary, title="Complete"))
