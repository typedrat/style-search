#!/usr/bin/env python3
"""Train a weighted similarity function from triplet judgments."""

import sqlite3
from pathlib import Path

import click
import chromadb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


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
    """Dataset of triplet judgments."""

    def __init__(
        self,
        triplets: list[tuple[str, str, str]],
        embeddings: dict[str, np.ndarray],
    ):
        # Filter to triplets where all artists have embeddings
        self.triplets = [
            t for t in triplets
            if t[0] in embeddings and t[1] in embeddings and t[2] in embeddings
        ]
        self.embeddings = embeddings
        print(f"Loaded {len(self.triplets)} valid triplets (of {len(triplets)} total)")

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor, positive, negative = self.triplets[idx]
        return (
            torch.tensor(self.embeddings[anchor], dtype=torch.float32),
            torch.tensor(self.embeddings[positive], dtype=torch.float32),
            torch.tensor(self.embeddings[negative], dtype=torch.float32),
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
) -> list[float]:
    """Train the model with triplet margin loss and optional equality constraints."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    triplet_loss_fn = nn.TripletMarginWithDistanceLoss(
        distance_function=model,
        margin=margin,
    )

    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0

        # Triplet loss
        for anchor, positive, negative in triplet_loader:
            optimizer.zero_grad()
            loss = triplet_loss_fn(anchor, positive, negative)

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


@click.command()
@click.argument("dataset")
@click.option("--epochs", default=100, help="Number of training epochs")
@click.option("--lr", default=0.01, help="Learning rate")
@click.option("--margin", default=0.2, help="Triplet margin")
@click.option("--batch-size", default=32, help="Batch size")
@click.option("--equality-weight", default=0.5, help="Weight for equality constraints")
@click.option("--output", "-o", default=None, help="Output path for weights")
def main(
    dataset: str,
    epochs: int,
    lr: float,
    margin: float,
    batch_size: int,
    equality_weight: float,
    output: str | None,
):
    """Train similarity weights from triplet judgments."""
    db_path = Path("data/triplets.db")
    if not db_path.exists():
        raise FileNotFoundError(f"Triplets database not found: {db_path}")

    # Load data
    print(f"Loading triplets for dataset '{dataset}'...")
    triplets = load_triplets(db_path, dataset)
    print(f"Found {len(triplets)} triplet judgments")

    equality_constraints = load_equality_constraints(db_path, dataset)
    print(f"Found {len(equality_constraints)} equality constraints (too_similar/anchor_outlier skips)")

    if len(triplets) == 0:
        print("No triplets found. Collect some training data first!")
        return

    print("Loading embeddings...")
    embeddings = load_embeddings(dataset)
    print(f"Loaded {len(embeddings)} embeddings")

    # Create datasets and dataloaders
    train_dataset = TripletDataset(triplets, embeddings)
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
    print(f"Baseline accuracy (uniform weights): {baseline_acc:.1%}")

    # Train
    print(f"\nTraining for {epochs} epochs...")
    train(
        model,
        triplet_loader,
        equality_loader=equality_loader,
        epochs=epochs,
        lr=lr,
        margin=margin,
        equality_weight=equality_weight,
    )

    # Evaluate trained model
    trained_acc = evaluate(model, triplets, embeddings)
    print(f"\nTrained accuracy: {trained_acc:.1%}")
    print(f"Improvement: {trained_acc - baseline_acc:+.1%}")

    # Save weights
    output_path = Path(output) if output else Path(f"data/{dataset}/similarity_weights.pt")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    weights = torch.nn.functional.softplus(model.weights).detach().numpy()
    torch.save({
        "weights": model.state_dict(),
        "weights_numpy": weights,
        "dim": dim,
        "train_accuracy": trained_acc,
        "baseline_accuracy": baseline_acc,
        "epochs": epochs,
        "num_triplets": len(triplets),
    }, output_path)
    print(f"\nSaved weights to {output_path}")

    # Print top weighted dimensions
    print("\nTop 10 most important dimensions:")
    top_indices = np.argsort(weights)[::-1][:10]
    for i, idx in enumerate(top_indices):
        print(f"  {i+1}. dim {idx}: {weights[idx]:.3f}")


if __name__ == "__main__":
    main()
