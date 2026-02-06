"""Shared training utilities: model classes, data loading, and training functions."""

import sqlite3
from pathlib import Path

import chromadb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from style_search.config import dataset_chroma_path


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
            [weights[i] for i in valid_indices]
            if weights
            else [1.0] * len(self.triplets)
        )
        print(f"Loaded {len(self.triplets)} valid triplets (of {len(triplets)} total)")

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        WHERE dataset = ?
            AND choice IS NULL
            AND skip_reason IN ('too_similar', 'anchor_outlier')
        """,
        (dataset,),
    ).fetchall()
    conn.close()

    return [(row["anchor"], row["option_a"], row["option_b"]) for row in rows]


def load_embeddings(dataset: str) -> dict[str, np.ndarray]:
    """Load embeddings from ChromaDB."""
    chroma_path = dataset_chroma_path(dataset)
    if not chroma_path.exists():
        raise FileNotFoundError(f"Dataset not found: {chroma_path}")

    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_collection(dataset)
    results = collection.get(include=["embeddings"])
    embeddings = results["embeddings"]
    assert embeddings is not None

    return {
        id_: np.array(emb)
        for id_, emb in zip(results["ids"], embeddings, strict=True)
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
            if (
                anchor not in embeddings
                or positive not in embeddings
                or negative not in embeddings
            ):
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
) -> tuple[
    list[tuple[str, str, str]],
    list[tuple[str, str, str]],
    dict[str, np.ndarray],
    list[float] | None,
]:
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
        # Weight = 1 / (num_datasets * dataset_count)
        # so each dataset sums to 1/num_datasets
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
        print("\nDataset balancing enabled:")
        for dataset in datasets:
            count = triplet_counts[dataset]
            w = 1.0 / (num_datasets * count) / mean_weight if count > 0 else 0.0
            print(f"  {dataset}: {count} triplets, weight={w:.4f}")

    return all_triplets, all_equality, all_embeddings, weights
