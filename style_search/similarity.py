#!/usr/bin/env python3
"""Central module for model management and active learning."""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import chromadb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from safetensors.torch import load_file, save_file
from sklearn.cluster import KMeans

if TYPE_CHECKING:
    from numpy.typing import NDArray

from style_search.train_similarity import (
    TripletDataset,
    WeightedDistance,
    evaluate,
    load_embeddings,
    load_triplets,
    train,
)

# In-memory state (per dataset)
_models: dict[str, WeightedDistance] = {}
_embeddings: dict[str, dict[str, NDArray[np.floating]]] = {}
_clusters: dict[str, NDArray[np.integer]] = {}  # artist_id -> cluster_id mapping as array
_cluster_labels: dict[str, dict[str, int]] = {}  # dataset -> {artist_id: cluster_id}
_anchor_counts: dict[str, dict[str, int]] = {}  # dataset -> {artist_id: count}
_triplet_cache: dict[str, list[tuple[str, str, str]]] = {}  # recent triplets for warm updates

# Lock for thread-safe model updates
_model_locks: dict[str, threading.Lock] = {}

# Constants
WEIGHTS_FILENAME = "similarity_weights.safetensors"
N_CLUSTERS = 30
WARM_UPDATE_STEPS = 5
WARM_UPDATE_BATCH_SIZE = 10
FULL_RETRAIN_INTERVAL = 20


@dataclass
class SuggestedTriplet:
    """A suggested triplet with scoring info."""

    anchor: str
    option_a: str
    option_b: str
    uncertainty_score: float
    diversity_score: float


@dataclass
class ModelStatus:
    """Status of the similarity model for a dataset."""

    loaded: bool
    dim: int | None
    num_triplets: int
    train_accuracy: float | None
    weights_path: str | None
    weights_exist: bool


def _get_lock(dataset: str) -> threading.Lock:
    """Get or create a lock for a dataset."""
    if dataset not in _model_locks:
        _model_locks[dataset] = threading.Lock()
    return _model_locks[dataset]


def _weights_path(dataset: str) -> Path:
    """Get the path to the weights file for a dataset."""
    return Path(f"data/{dataset}/{WEIGHTS_FILENAME}")


def get_model(dataset: str) -> WeightedDistance:
    """Get or load the model for a dataset."""
    if dataset not in _models:
        load_weights(dataset)
    return _models[dataset]


def get_embeddings(dataset: str) -> dict[str, NDArray[np.floating]]:
    """Get or load embeddings for a dataset."""
    if dataset not in _embeddings:
        _embeddings[dataset] = load_embeddings(dataset)
    return _embeddings[dataset]


def get_clusters(dataset: str) -> dict[str, int]:
    """Get or compute cluster assignments for a dataset."""
    if dataset not in _cluster_labels:
        embeddings = get_embeddings(dataset)
        ids = list(embeddings.keys())
        X = np.array([embeddings[id_] for id_ in ids])

        # Fit KMeans
        n_clusters = min(N_CLUSTERS, len(ids))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        _cluster_labels[dataset] = {id_: int(label) for id_, label in zip(ids, labels)}

    return _cluster_labels[dataset]


def get_anchor_counts(dataset: str) -> dict[str, int]:
    """Get anchor usage counts from the database."""
    if dataset not in _anchor_counts:
        _refresh_anchor_counts(dataset)
    return _anchor_counts[dataset]


def _refresh_anchor_counts(dataset: str) -> None:
    """Refresh anchor counts from the database."""
    db_path = Path("data/triplets.db")
    if not db_path.exists():
        _anchor_counts[dataset] = {}
        return

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT anchor, COUNT(*) as count FROM triplets WHERE dataset = ? GROUP BY anchor",
        (dataset,),
    ).fetchall()
    conn.close()

    _anchor_counts[dataset] = {row[0]: row[1] for row in rows}


def load_weights(dataset: str) -> None:
    """Load weights from disk or initialize a new model."""
    embeddings = get_embeddings(dataset)
    dim = len(next(iter(embeddings.values())))

    path = _weights_path(dataset)
    if path.exists():
        tensors = load_file(path)
        model = WeightedDistance(dim)
        model.weights.data = tensors["weights"]
        _models[dataset] = model
        print(f"Loaded weights from {path}")
    else:
        _models[dataset] = WeightedDistance(dim)
        print(f"Initialized new model for {dataset} (dim={dim})")


def save_weights(dataset: str, metadata: dict | None = None) -> None:
    """Save weights to disk in safetensors format."""
    if dataset not in _models:
        raise ValueError(f"No model loaded for dataset {dataset}")

    model = _models[dataset]
    path = _weights_path(dataset)
    path.parent.mkdir(parents=True, exist_ok=True)

    tensors = {"weights": model.weights.data}

    # Build metadata dict (safetensors requires string values)
    meta = {}
    if metadata:
        for key, value in metadata.items():
            meta[key] = str(value)

    save_file(tensors, path, metadata=meta)
    print(f"Saved weights to {path}")


def warm_update(dataset: str, triplet: tuple[str, str, str]) -> None:
    """Perform a warm update after a new triplet judgment.

    Args:
        dataset: The dataset name
        triplet: (anchor, positive, negative) - positive is the chosen one
    """
    lock = _get_lock(dataset)
    with lock:
        # Add to recent triplets cache
        if dataset not in _triplet_cache:
            _triplet_cache[dataset] = []
        _triplet_cache[dataset].append(triplet)

        # Keep only last WARM_UPDATE_BATCH_SIZE triplets
        if len(_triplet_cache[dataset]) > WARM_UPDATE_BATCH_SIZE:
            _triplet_cache[dataset] = _triplet_cache[dataset][-WARM_UPDATE_BATCH_SIZE:]

        # Update anchor counts
        if dataset not in _anchor_counts:
            _anchor_counts[dataset] = {}
        anchor = triplet[0]
        _anchor_counts[dataset][anchor] = _anchor_counts[dataset].get(anchor, 0) + 1

        # Get model and embeddings
        model = get_model(dataset)
        embeddings = get_embeddings(dataset)

        # Create dataset from recent triplets
        valid_triplets = [
            t
            for t in _triplet_cache[dataset]
            if t[0] in embeddings and t[1] in embeddings and t[2] in embeddings
        ]

        if not valid_triplets:
            return

        # Prepare tensors
        anchors = torch.stack(
            [torch.tensor(embeddings[t[0]], dtype=torch.float32) for t in valid_triplets]
        )
        positives = torch.stack(
            [torch.tensor(embeddings[t[1]], dtype=torch.float32) for t in valid_triplets]
        )
        negatives = torch.stack(
            [torch.tensor(embeddings[t[2]], dtype=torch.float32) for t in valid_triplets]
        )

        # Run a few gradient steps
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.TripletMarginWithDistanceLoss(distance_function=model, margin=0.2)

        model.train()
        for _ in range(WARM_UPDATE_STEPS):
            optimizer.zero_grad()
            loss = loss_fn(anchors, positives, negatives)
            loss.backward()
            optimizer.step()


def full_retrain(dataset: str) -> dict:
    """Perform a full retrain of the model.

    Returns metrics dict with train_accuracy, baseline_accuracy, etc.
    """
    lock = _get_lock(dataset)
    with lock:
        db_path = Path("data/triplets.db")
        triplets = load_triplets(db_path, dataset)
        embeddings = get_embeddings(dataset)

        if not triplets:
            return {"error": "No triplets found", "num_triplets": 0}

        # Initialize fresh model
        dim = len(next(iter(embeddings.values())))
        model = WeightedDistance(dim)

        # Baseline accuracy
        baseline_acc = evaluate(model, triplets, embeddings)

        # Create dataset and dataloader
        from torch.utils.data import DataLoader

        train_dataset = TripletDataset(triplets, embeddings)
        triplet_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # Train
        train(model, triplet_loader, epochs=100, lr=0.01, margin=0.2)

        # Evaluate
        train_acc = evaluate(model, triplets, embeddings)

        # Update in-memory model
        _models[dataset] = model

        # Save weights
        save_weights(
            dataset,
            metadata={
                "dim": dim,
                "train_accuracy": train_acc,
                "baseline_accuracy": baseline_acc,
                "num_triplets": len(triplets),
            },
        )

        # Clear triplet cache
        _triplet_cache[dataset] = []

        return {
            "train_accuracy": train_acc,
            "baseline_accuracy": baseline_acc,
            "improvement": train_acc - baseline_acc,
            "num_triplets": len(triplets),
            "dim": dim,
        }


def suggest_triplet(dataset: str, n_candidates: int = 100) -> SuggestedTriplet:
    """Suggest a triplet using diversity + uncertainty sampling.

    Args:
        dataset: The dataset name
        n_candidates: Number of candidate triplets to sample

    Returns:
        SuggestedTriplet with the best scoring triplet
    """
    model = get_model(dataset)
    embeddings = get_embeddings(dataset)
    clusters = get_clusters(dataset)
    anchor_counts = get_anchor_counts(dataset)

    ids = list(embeddings.keys())
    max_anchor_count = max(anchor_counts.values()) if anchor_counts else 1

    # Sample candidate triplets
    candidates = _sample_candidate_triplets(ids, n_candidates)

    best_score = float("-inf")
    best_triplet = None
    best_uncertainty = 0.0
    best_diversity = 0.0

    model.eval()
    with torch.no_grad():
        for anchor, a, b in candidates:
            # Get embeddings as tensors
            e_anchor = torch.tensor(embeddings[anchor], dtype=torch.float32).unsqueeze(0)
            e_a = torch.tensor(embeddings[a], dtype=torch.float32).unsqueeze(0)
            e_b = torch.tensor(embeddings[b], dtype=torch.float32).unsqueeze(0)

            # Uncertainty: how close are d(anchor,a) and d(anchor,b)?
            d_a = model(e_anchor, e_a).item()
            d_b = model(e_anchor, e_b).item()
            uncertainty = 1.0 / (1.0 + abs(d_a - d_b))

            # Diversity: penalize overused anchors, reward cluster coverage
            anchor_penalty = anchor_counts.get(anchor, 0) / max(max_anchor_count, 1)
            cluster_bonus = _cluster_diversity(anchor, a, b, clusters)
            diversity = cluster_bonus - 0.5 * anchor_penalty

            # Combined score (tune weights as needed)
            score = 0.6 * uncertainty + 0.4 * diversity

            if score > best_score:
                best_score = score
                best_triplet = (anchor, a, b)
                best_uncertainty = uncertainty
                best_diversity = diversity

    if best_triplet is None:
        # Fallback to random
        import random

        anchor = random.choice(ids)
        a = random.choice([x for x in ids if x != anchor])
        b = random.choice([x for x in ids if x != anchor and x != a])
        return SuggestedTriplet(
            anchor=anchor,
            option_a=a,
            option_b=b,
            uncertainty_score=0.0,
            diversity_score=0.0,
        )

    return SuggestedTriplet(
        anchor=best_triplet[0],
        option_a=best_triplet[1],
        option_b=best_triplet[2],
        uncertainty_score=best_uncertainty,
        diversity_score=best_diversity,
    )


def _sample_candidate_triplets(
    ids: list[str], n: int
) -> list[tuple[str, str, str]]:
    """Sample n candidate triplets."""
    import random

    candidates = []
    for _ in range(n):
        anchor = random.choice(ids)
        a = random.choice([x for x in ids if x != anchor])
        b = random.choice([x for x in ids if x != anchor and x != a])
        candidates.append((anchor, a, b))
    return candidates


def _cluster_diversity(
    anchor: str, a: str, b: str, clusters: dict[str, int]
) -> float:
    """Compute cluster diversity bonus.

    Higher bonus when options come from different clusters,
    and when anchor is in a different cluster than both options.
    """
    c_anchor = clusters.get(anchor, 0)
    c_a = clusters.get(a, 0)
    c_b = clusters.get(b, 0)

    # Bonus for options being in different clusters
    options_diff = 1.0 if c_a != c_b else 0.0

    # Bonus for anchor being different from at least one option
    anchor_diff = 1.0 if (c_anchor != c_a or c_anchor != c_b) else 0.0

    return 0.5 * options_diff + 0.5 * anchor_diff


def get_model_status(dataset: str) -> ModelStatus:
    """Get the status of the model for a dataset."""
    path = _weights_path(dataset)
    weights_exist = path.exists()

    # Count triplets
    db_path = Path("data/triplets.db")
    num_triplets = 0
    if db_path.exists():
        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT COUNT(*) FROM triplets WHERE dataset = ? AND choice IS NOT NULL",
            (dataset,),
        ).fetchone()
        num_triplets = row[0] if row else 0
        conn.close()

    # Check if model is loaded
    loaded = dataset in _models
    dim = None
    train_accuracy = None

    if loaded:
        dim = _models[dataset].weights.shape[0]

    if weights_exist:
        # Read metadata from safetensors
        try:
            from safetensors import safe_open

            with safe_open(path, framework="pt") as f:
                metadata = f.metadata()
                if metadata:
                    dim = int(metadata.get("dim", 0)) or dim
                    train_accuracy = float(metadata.get("train_accuracy", 0)) or None
        except Exception:
            pass

    return ModelStatus(
        loaded=loaded,
        dim=dim,
        num_triplets=num_triplets,
        train_accuracy=train_accuracy,
        weights_path=str(path) if weights_exist else None,
        weights_exist=weights_exist,
    )


def trigger_background_retrain(dataset: str) -> None:
    """Trigger a full retrain in a background thread."""
    thread = threading.Thread(target=full_retrain, args=(dataset,), daemon=True)
    thread.start()
    print(f"Started background retrain for {dataset}")
