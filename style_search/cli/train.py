"""CLI command for training similarity models."""

import json
import random
from datetime import UTC, datetime
from pathlib import Path

import click
import numpy as np
import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader

from style_search.config import MODELS_DIR, TRIPLETS_DB
from style_search.training import (
    EqualityDataset,
    TripletDataset,
    WeightedDistance,
    evaluate,
    load_embeddings,
    load_equality_constraints,
    load_multi_dataset,
    load_triplets,
    train,
)


def _is_path(output: str) -> bool:
    """Check if output is a path (contains / or \\) vs a basename."""
    return "/" in output or "\\" in output or output.endswith(".safetensors")


def _get_series_prefix(datasets: tuple[str, ...], epochs: int) -> str:
    """Get the series prefix for auto-naming: {datasets}_{epochs}."""
    datasets_part = "_".join(datasets)
    return f"{datasets_part}_{epochs}"


def _get_next_series_version(prefix: str) -> int:
    """Find the next version number for a series prefix.

    Looks for files matching {prefix}_v{NNN}.safetensors and returns
    the smallest number larger than all existing versions.
    """
    if not MODELS_DIR.exists():
        return 1
    versions = []
    for f in MODELS_DIR.glob(f"{prefix}_v*.safetensors"):
        # Extract version from "{prefix}_v{NNN}.safetensors"
        stem = f.stem  # e.g., "art_100_v003"
        suffix = stem[len(prefix) + 2:]  # Skip "{prefix}_v" to get "003"
        try:
            version = int(suffix)
            versions.append(version)
        except ValueError:
            continue
    return (max(versions) if versions else 0) + 1


def _init_dataset_model(
    dataset: str,
    weights_path: Path,
    meta_path: Path,
) -> None:
    """Initialize a dataset's model directory with trained weights.

    Backs up any existing directory and copies the trained model as v001.
    This gives similarity.py a fresh starting point for live training.
    """
    import shutil

    dataset_dir = MODELS_DIR / dataset

    # Backup existing directory if present
    if dataset_dir.exists():
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        backup_dir = MODELS_DIR / f"{dataset}.backup.{timestamp}"
        shutil.move(dataset_dir, backup_dir)
        print(f"Backed up existing model directory to {backup_dir}")

    # Create fresh directory with v001
    dataset_dir.mkdir(parents=True, exist_ok=True)
    v001_weights = dataset_dir / "v001.safetensors"
    v001_meta = dataset_dir / "v001.json"

    shutil.copy(weights_path, v001_weights)
    shutil.copy(meta_path, v001_meta)

    # Update version in the copied metadata
    with open(v001_meta) as f:
        meta = json.load(f)
    meta["version"] = 1
    with open(v001_meta, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Initialized {dataset_dir} with v001")


@click.command()
@click.argument("datasets", nargs=-1, required=True)
@click.option("--epochs", default=100, help="Number of training epochs")
@click.option("--lr", default=0.01, help="Learning rate")
@click.option("--margin", default=0.2, help="Triplet margin")
@click.option("--batch-size", default=32, help="Batch size")
@click.option("--equality-weight", default=0.5, help="Weight for equality constraints")
@click.option(
    "--output", "-o", default=None,
    help="Output path or basename for weights",
)
@click.option(
    "--init",
    is_flag=True,
    help="Initialize dataset model dirs (backup existing, copy as v001)",
)
@click.option(
    "--balance-datasets",
    is_flag=True,
    help="Use inverse frequency weighting so each dataset contributes equally",
)
@click.option(
    "--test-split",
    default=0.0,
    help="Fraction of triplets to hold out for testing (0.0-0.5)",
)
@click.option(
    "--seed",
    default=42,
    help="Random seed for train/test split",
)
@click.option(
    "--l2-weight",
    default=0.01,
    help="L2 regularization weight (pulls weights toward uniform)",
)
def train_cmd(
    datasets: tuple[str, ...],
    epochs: int,
    lr: float,
    margin: float,
    batch_size: int,
    equality_weight: float,
    output: str | None,
    init: bool,
    balance_datasets: bool,
    test_split: float,
    seed: int,
    l2_weight: float,
):
    """Train similarity weights from triplet judgments.

    Accepts one or more dataset names. When multiple datasets are provided,
    triplets from each are combined for training while keeping embeddings
    scoped to prevent cross-dataset comparisons.
    """
    db_path = TRIPLETS_DB
    if not db_path.exists():
        raise FileNotFoundError(f"Triplets database not found: {db_path}")

    # Load data from all datasets
    triplet_weights = None
    if len(datasets) == 1:
        # Single dataset - use original logic (no prefixing)
        dataset = datasets[0]
        print(f"Loading triplets for dataset '{dataset}'...")
        triplets = load_triplets(db_path, dataset)
        print(f"Found {len(triplets)} triplet judgments")

        equality_constraints = load_equality_constraints(db_path, dataset)
        n_eq = len(equality_constraints)
        print(f"Found {n_eq} equality constraints"
              " (too_similar/anchor_outlier skips)")

        print("Loading embeddings...")
        embeddings = load_embeddings(dataset)
        print(f"Loaded {len(embeddings)} embeddings")
    else:
        # Multiple datasets - combine with prefixed IDs
        print(f"Loading {len(datasets)} datasets: {', '.join(datasets)}")
        (triplets, equality_constraints,
         embeddings, triplet_weights) = load_multi_dataset(
            db_path, list(datasets),
            balance_datasets=balance_datasets,
        )
        print(
            f"\nCombined: {len(triplets)} triplets,"
            f" {len(equality_constraints)} equality"
            f" constraints, {len(embeddings)} embeddings"
        )

    if len(triplets) == 0:
        print("No triplets found. Collect some training data first!")
        return

    # Train/test split
    test_triplets = []
    if test_split > 0:
        random.seed(seed)
        n_test = int(len(triplets) * test_split)
        indices = list(range(len(triplets)))
        random.shuffle(indices)
        test_indices = set(indices[:n_test])
        train_indices = indices[n_test:]

        test_triplets = [triplets[i] for i in test_indices]
        triplets = [triplets[i] for i in train_indices]
        if triplet_weights:
            triplet_weights = [triplet_weights[i] for i in train_indices]
        print(
            f"\nTrain/test split: {len(triplets)} train,"
            f" {len(test_triplets)} test ({test_split:.0%})"
        )

    # Create datasets and dataloaders
    train_dataset = TripletDataset(triplets, embeddings, weights=triplet_weights)
    triplet_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    equality_loader = None
    if equality_constraints:
        equality_dataset = EqualityDataset(equality_constraints, embeddings)
        equality_loader = DataLoader(
            equality_dataset, batch_size=batch_size, shuffle=True,
        )

    # Initialize model
    dim = len(next(iter(embeddings.values())))
    print(f"Embedding dimension: {dim}")
    model = WeightedDistance(dim)

    # Evaluate baseline (uniform weights)
    baseline_acc = evaluate(model, triplets, embeddings)
    baseline_test_acc = (
        evaluate(model, test_triplets, embeddings)
        if test_triplets else None
    )
    if baseline_test_acc is not None:
        print(
            f"Baseline accuracy (uniform weights):"
            f" train={baseline_acc:.1%},"
            f" test={baseline_test_acc:.1%}"
        )
    else:
        print(f"Baseline accuracy (uniform weights): {baseline_acc:.1%}")

    # Train
    print(f"\nTraining for {epochs} epochs...")
    loss_history = train(
        model,
        triplet_loader,
        equality_loader=equality_loader,
        epochs=epochs,
        lr=lr,
        margin=margin,
        equality_weight=equality_weight,
        l2_weight=l2_weight,
    )

    # Evaluate trained model
    trained_acc = evaluate(model, triplets, embeddings)
    test_acc = (
        evaluate(model, test_triplets, embeddings)
        if test_triplets else None
    )
    if test_acc is not None and baseline_test_acc is not None:
        improvement = trained_acc - baseline_acc
        test_improvement = test_acc - baseline_test_acc
        print(
            f"\nTrain accuracy: {trained_acc:.1%}"
            f" (improvement: {improvement:+.1%})"
        )
        print(
            f"Test accuracy:  {test_acc:.1%}"
            f" (improvement: {test_improvement:+.1%})"
        )
        if test_acc < baseline_test_acc:
            print("  ⚠ Test accuracy decreased — possible overfitting")
    else:
        print(f"\nTrained accuracy: {trained_acc:.1%}")
        print(f"Improvement: {trained_acc - baseline_acc:+.1%}")

    # Determine output path and version
    # Three cases:
    # 1. -o with path (contains / or ends with .safetensors): use exact path
    # 2. -o with basename: save to data/models/{basename}.safetensors
    # 3. No -o: auto-name as data/models/{datasets}_{epochs}_v{N}.safetensors
    if output:
        if _is_path(output):
            output_path = Path(output)
            if not output_path.suffix:
                output_path = output_path.with_suffix(".safetensors")
        else:
            # Just a basename
            output_path = MODELS_DIR / f"{output}.safetensors"
        meta_path = output_path.with_suffix(".json")
        version = None
    else:
        # Auto-generate name based on datasets and epochs
        prefix = _get_series_prefix(datasets, epochs)
        version = _get_next_series_version(prefix)
        output_path = MODELS_DIR / f"{prefix}_v{version:03d}.safetensors"
        meta_path = MODELS_DIR / f"{prefix}_v{version:03d}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    weights = torch.nn.functional.softplus(model.weights).detach().numpy()

    # Build full metadata for JSON file
    full_metadata = {
        "version": version,
        "timestamp": datetime.now(UTC).isoformat(),
        "datasets": list(datasets),
        "dim": dim,
        "train_accuracy": trained_acc,
        "test_accuracy": test_acc,
        "baseline_accuracy": baseline_acc,
        "test_split": test_split if test_split > 0 else None,
        "seed": seed if test_split > 0 else None,
        "epochs": epochs,
        "learning_rate": lr,
        "margin": margin,
        "batch_size": batch_size,
        "num_train_triplets": len(triplets),
        "num_test_triplets": len(test_triplets) if test_triplets else 0,
        "num_triplets": (
            len(triplets)
            + (len(test_triplets) if test_triplets else 0)
        ),
        "num_equality_constraints": (
            len(equality_constraints)
            if equality_constraints else 0
        ),
        "equality_weight": equality_weight if equality_constraints else None,
        "balance_datasets": balance_datasets,
        "l2_weight": l2_weight if l2_weight > 0 else None,
        "loss_history": loss_history,
    }

    # Save with safetensors (metadata must be strings)
    tensors = {"weights": model.weights.data}
    safetensors_metadata = {
        "dim": str(dim),
        "train_accuracy": str(trained_acc),
        "baseline_accuracy": str(baseline_acc),
        "epochs": str(epochs),
        "learning_rate": str(lr),
        "margin": str(margin),
        "batch_size": str(batch_size),
        "l2_weight": str(l2_weight),
        "num_triplets": str(len(triplets)),
        "datasets": ",".join(datasets),
        "balance_datasets": str(balance_datasets),
    }
    if test_acc is not None:
        safetensors_metadata["test_accuracy"] = str(test_acc)
        safetensors_metadata["test_split"] = str(test_split)
    if equality_constraints:
        safetensors_metadata["equality_weight"] = str(equality_weight)
    if version is not None:
        safetensors_metadata["version"] = str(version)
    save_file(tensors, output_path, metadata=safetensors_metadata)

    # Save JSON metadata
    with open(meta_path, "w") as f:
        json.dump(full_metadata, f, indent=2)

    print(f"\nSaved weights to {output_path}")
    print(f"Saved metadata to {meta_path}")

    # Initialize per-dataset model directories if requested
    if init:
        print("\nInitializing dataset model directories...")
        for dataset in datasets:
            _init_dataset_model(dataset, output_path, meta_path)

    # Print top weighted dimensions
    print("\nTop 10 most important dimensions:")
    top_indices = np.argsort(weights)[::-1][:10]
    for i, idx in enumerate(top_indices):
        print(f"  {i+1}. dim {idx}: {weights[idx]:.3f}")
