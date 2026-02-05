#!/usr/bin/env python3
"""Train a weighted similarity function from triplet judgments."""

import json
import random
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import chromadb
import click
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset

# Models directory
MODELS_DIR = Path("data/models")


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


class WeightedDistance(nn.Module):
    """Learnable weighted Euclidean distance."""

    def __init__(self, dim: int = 1024):
        super().__init__()
        # Initialize weights to 1 (equivalent to standard Euclidean)
        self.weights = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute weighted distance between x and y."""
        # Softplus to keep weights positive
        w = torch.nn.functional.softplus(self.weights)
        diff = x - y
        return torch.sqrt(torch.sum(w * diff * diff, dim=-1) + 1e-8)


class TripletDataset(Dataset):
    """Dataset of triplet judgments with optional sample weights."""

    def __init__(
        self,
        triplets: list[tuple[str, str, str]],
        embeddings: dict[str, np.ndarray],
        weights: list[float] | None = None,
    ):
        # Filter to triplets where all artists have embeddings
        valid_indices = [
            i for i, t in enumerate(triplets)
            if t[0] in embeddings and t[1] in embeddings and t[2] in embeddings
        ]
        self.triplets = [triplets[i] for i in valid_indices]
        self.embeddings = embeddings
        self.weights = (
            [weights[i] for i in valid_indices] if weights else [1.0] * len(self.triplets)
        )
        print(f"Loaded {len(self.triplets)} valid triplets (of {len(triplets)} total)")

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor, positive, negative = self.triplets[idx]
        return (
            torch.tensor(self.embeddings[anchor], dtype=torch.float32),
            torch.tensor(self.embeddings[positive], dtype=torch.float32),
            torch.tensor(self.embeddings[negative], dtype=torch.float32),
            torch.tensor(self.weights[idx], dtype=torch.float32),
        )


class EqualityDataset(Dataset):
    """Dataset of equality constraints (skipped triplets where A ≈ B)."""

    def __init__(
        self,
        pairs: list[tuple[str, str, str]],  # (anchor, option_a, option_b)
        embeddings: dict[str, np.ndarray],
    ):
        self.pairs = [
            p for p in pairs
            if p[0] in embeddings and p[1] in embeddings and p[2] in embeddings
        ]
        self.embeddings = embeddings
        print(f"Loaded {len(self.pairs)} equality constraints (of {len(pairs)} total)")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor, option_a, option_b = self.pairs[idx]
        return (
            torch.tensor(self.embeddings[anchor], dtype=torch.float32),
            torch.tensor(self.embeddings[option_a], dtype=torch.float32),
            torch.tensor(self.embeddings[option_b], dtype=torch.float32),
        )


def load_triplets(db_path: Path, dataset: str) -> list[tuple[str, str, str]]:
    """Load triplets from SQLite database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        """
        SELECT anchor, option_a, option_b, choice
        FROM triplets
        WHERE dataset = ? AND choice IS NOT NULL
        """,
        (dataset,),
    ).fetchall()
    conn.close()

    triplets = []
    for row in rows:
        anchor = row["anchor"]
        if row["choice"] == "A":
            positive, negative = row["option_a"], row["option_b"]
        else:
            positive, negative = row["option_b"], row["option_a"]
        triplets.append((anchor, positive, negative))

    return triplets


def load_equality_constraints(
    db_path: Path, dataset: str
) -> list[tuple[str, str, str]]:
    """Load equality constraints from skipped triplets.

    Returns triplets where the user indicated A and B are equally similar to anchor.
    Only includes 'too_similar' and 'anchor_outlier' skips, not 'unknown'.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        """
        SELECT anchor, option_a, option_b
        FROM triplets
        WHERE dataset = ? AND choice IS NULL AND skip_reason IN ('too_similar', 'anchor_outlier')
        """,
        (dataset,),
    ).fetchall()
    conn.close()

    return [(row["anchor"], row["option_a"], row["option_b"]) for row in rows]


def load_embeddings(dataset: str) -> dict[str, np.ndarray]:
    """Load embeddings from ChromaDB."""
    db_path = Path(f"data/{dataset}/chroma")
    if not db_path.exists():
        raise FileNotFoundError(f"Dataset not found: {db_path}")

    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_collection(dataset)
    results = collection.get(include=["embeddings"])

    return {
        id_: np.array(emb)
        for id_, emb in zip(results["ids"], results["embeddings"])
    }


def train(
    model: WeightedDistance,
    triplet_loader: DataLoader,
    equality_loader: DataLoader | None = None,
    epochs: int = 100,
    lr: float = 0.01,
    margin: float = 0.2,
    equality_weight: float = 0.5,
    l2_weight: float = 0.0,
) -> list[float]:
    """Train the model with triplet margin loss and optional equality constraints.

    Args:
        model: The WeightedDistance model to train
        triplet_loader: DataLoader for triplet data
        equality_loader: Optional DataLoader for equality constraints
        epochs: Number of training epochs
        lr: Learning rate
        margin: Triplet margin
        equality_weight: Weight for equality constraint loss
        l2_weight: L2 regularization weight (pulls weights toward uniform)
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Use reduction='none' to get per-sample losses for weighting
    triplet_loss_fn = nn.TripletMarginWithDistanceLoss(
        distance_function=model,
        margin=margin,
        reduction="none",
    )

    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0

        # Triplet loss
        for anchor, positive, negative, sample_weights in triplet_loader:
            optimizer.zero_grad()
            per_sample_loss = triplet_loss_fn(anchor, positive, negative)
            # Apply sample weights and reduce
            loss = torch.mean(per_sample_loss * sample_weights)

            # Add equality constraint loss if available
            if equality_loader is not None and len(equality_loader) > 0:
                # Sample one batch from equality loader
                eq_batch = next(iter(equality_loader))
                eq_anchor, eq_a, eq_b = eq_batch
                d_a = model(eq_anchor, eq_a)
                d_b = model(eq_anchor, eq_b)
                # L1 loss: distances should be equal
                eq_loss = torch.mean(torch.abs(d_a - d_b))
                loss = loss + equality_weight * eq_loss

            # L2 regularization: penalize deviation from uniform weights
            # softplus(0) ≈ 0.693, so we pull toward that (uniform weighting)
            if l2_weight > 0:
                l2_loss = l2_weight * torch.mean(model.weights ** 2)
                loss = loss + l2_loss

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    return losses


def evaluate(
    model: WeightedDistance,
    triplets: list[tuple[str, str, str]],
    embeddings: dict[str, np.ndarray],
) -> float:
    """Evaluate accuracy on triplets."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for anchor, positive, negative in triplets:
            if anchor not in embeddings or positive not in embeddings or negative not in embeddings:
                continue

            a = torch.tensor(embeddings[anchor], dtype=torch.float32).unsqueeze(0)
            p = torch.tensor(embeddings[positive], dtype=torch.float32).unsqueeze(0)
            n = torch.tensor(embeddings[negative], dtype=torch.float32).unsqueeze(0)

            d_pos = model(a, p).item()
            d_neg = model(a, n).item()

            if d_pos < d_neg:
                correct += 1
            total += 1

    return correct / total if total > 0 else 0.0


def load_multi_dataset(
    db_path: Path, datasets: list[str], balance_datasets: bool = False
) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]], dict[str, np.ndarray], list[float] | None]:
    """Load triplets and embeddings from multiple datasets.

    Each dataset's IDs are prefixed with the dataset name to avoid collisions.
    This allows training on combined triplets while keeping embeddings scoped.

    Args:
        db_path: Path to the triplets database
        datasets: List of dataset names to load
        balance_datasets: If True, compute inverse frequency weights so each
            dataset contributes equally to the loss regardless of size

    Returns:
        triplets: Combined list of (anchor, positive, negative) with prefixed IDs
        equality_constraints: Combined list of equality constraints with prefixed IDs
        embeddings: Combined dict mapping prefixed IDs to embeddings
        weights: Per-triplet weights if balance_datasets=True, else None
    """
    all_triplets = []
    all_equality = []
    all_embeddings = {}
    triplet_counts = {}  # Track count per dataset for weighting

    for dataset in datasets:
        print(f"\nLoading dataset '{dataset}'...")

        # Load triplets and prefix IDs
        triplets = load_triplets(db_path, dataset)
        prefixed_triplets = [
            (f"{dataset}:{a}", f"{dataset}:{p}", f"{dataset}:{n}")
            for a, p, n in triplets
        ]
        triplet_counts[dataset] = len(prefixed_triplets)
        all_triplets.extend(prefixed_triplets)
        print(f"  {len(triplets)} triplet judgments")

        # Load equality constraints and prefix IDs
        equality = load_equality_constraints(db_path, dataset)
        prefixed_equality = [
            (f"{dataset}:{a}", f"{dataset}:{oa}", f"{dataset}:{ob}")
            for a, oa, ob in equality
        ]
        all_equality.extend(prefixed_equality)
        print(f"  {len(equality)} equality constraints")

        # Load embeddings and prefix IDs
        embeddings = load_embeddings(dataset)
        for artist_id, emb in embeddings.items():
            all_embeddings[f"{dataset}:{artist_id}"] = emb
        print(f"  {len(embeddings)} embeddings")

    # Compute inverse frequency weights if requested
    weights = None
    if balance_datasets and len(datasets) > 1:
        # Weight = 1 / (num_datasets * dataset_count) so each dataset sums to 1/num_datasets
        # This normalizes so total weight per dataset is equal
        num_datasets = len(datasets)
        weights = []
        for dataset in datasets:
            count = triplet_counts[dataset]
            w = 1.0 / (num_datasets * count) if count > 0 else 0.0
            weights.extend([w] * count)
        # Scale weights so mean = 1 (preserves loss magnitude)
        mean_weight = sum(weights) / len(weights) if weights else 1.0
        weights = [w / mean_weight for w in weights]
        print(f"\nDataset balancing enabled:")
        for dataset in datasets:
            count = triplet_counts[dataset]
            w = 1.0 / (num_datasets * count) / mean_weight if count > 0 else 0.0
            print(f"  {dataset}: {count} triplets, weight={w:.4f}")

    return all_triplets, all_equality, all_embeddings, weights


@click.command()
@click.argument("datasets", nargs=-1, required=True)
@click.option("--epochs", default=100, help="Number of training epochs")
@click.option("--lr", default=0.01, help="Learning rate")
@click.option("--margin", default=0.2, help="Triplet margin")
@click.option("--batch-size", default=32, help="Batch size")
@click.option("--equality-weight", default=0.5, help="Weight for equality constraints")
@click.option("--output", "-o", default=None, help="Output path or basename for weights")
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
def main(
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
    db_path = Path("data/triplets.db")
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
        print(f"Found {len(equality_constraints)} equality constraints (too_similar/anchor_outlier skips)")

        print("Loading embeddings...")
        embeddings = load_embeddings(dataset)
        print(f"Loaded {len(embeddings)} embeddings")
    else:
        # Multiple datasets - combine with prefixed IDs
        print(f"Loading {len(datasets)} datasets: {', '.join(datasets)}")
        triplets, equality_constraints, embeddings, triplet_weights = load_multi_dataset(
            db_path, list(datasets), balance_datasets=balance_datasets
        )
        print(f"\nCombined: {len(triplets)} triplets, {len(equality_constraints)} equality constraints, {len(embeddings)} embeddings")

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
        print(f"\nTrain/test split: {len(triplets)} train, {len(test_triplets)} test ({test_split:.0%})")

    # Create datasets and dataloaders
    train_dataset = TripletDataset(triplets, embeddings, weights=triplet_weights)
    triplet_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    equality_loader = None
    if equality_constraints:
        equality_dataset = EqualityDataset(equality_constraints, embeddings)
        equality_loader = DataLoader(equality_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    dim = len(next(iter(embeddings.values())))
    print(f"Embedding dimension: {dim}")
    model = WeightedDistance(dim)

    # Evaluate baseline (uniform weights)
    baseline_acc = evaluate(model, triplets, embeddings)
    baseline_test_acc = evaluate(model, test_triplets, embeddings) if test_triplets else None
    if baseline_test_acc is not None:
        print(f"Baseline accuracy (uniform weights): train={baseline_acc:.1%}, test={baseline_test_acc:.1%}")
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
    test_acc = evaluate(model, test_triplets, embeddings) if test_triplets else None
    if test_acc is not None:
        print(f"\nTrain accuracy: {trained_acc:.1%} (improvement: {trained_acc - baseline_acc:+.1%})")
        print(f"Test accuracy:  {test_acc:.1%} (improvement: {test_acc - baseline_test_acc:+.1%})")
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
        "num_triplets": len(triplets) + (len(test_triplets) if test_triplets else 0),  # total
        "num_equality_constraints": len(equality_constraints) if equality_constraints else 0,
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


if __name__ == "__main__":
    main()
