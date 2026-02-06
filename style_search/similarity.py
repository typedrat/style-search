#!/usr/bin/env python3
"""Central module for model management and active learning."""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from safetensors.torch import load_file, save_file
from sklearn.cluster import KMeans

if TYPE_CHECKING:
    from numpy.typing import NDArray

from style_search.config import LOG_DIR, MODELS_DIR, TRIPLETS_DB, dataset_dir
from style_search.training import (
    TripletDataset,
    WeightedDistance,
    evaluate,
    load_embeddings,
    load_triplets,
    train,
)

# Configure per-user logging
LOG_DIR.mkdir(parents=True, exist_ok=True)

_LOG_FORMATTER = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_user_loggers: dict[str | None, logging.Logger] = {}
_user_names: dict[str, str] = {}  # token -> name cache


def _resolve_user_name(user_id: str) -> str:
    """Look up a user's name from their token."""
    if user_id in _user_names:
        return _user_names[user_id]
    try:
        conn = sqlite3.connect(TRIPLETS_DB)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT name FROM users WHERE token = ?", (user_id,),
        ).fetchone()
        conn.close()
        name = row["name"] if row else user_id[:8]
    except Exception:
        name = user_id[:8]
    _user_names[user_id] = name
    return name


def get_user_logger(user_id: str | None = None) -> logging.Logger:
    """Get a per-user logger that writes to data/logs/{user_name}.log."""
    if user_id in _user_loggers:
        return _user_loggers[user_id]

    if user_id is None:
        log_file = LOG_DIR / "anonymous.log"
        logger_name = f"{__name__}.anonymous"
    else:
        name = _resolve_user_name(user_id)
        log_file = LOG_DIR / f"{name}.log"
        logger_name = f"{__name__}.user.{user_id}"

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(log_file, mode="a")
        handler.setLevel(logging.INFO)
        handler.setFormatter(_LOG_FORMATTER)
        logger.addHandler(handler)

    _user_loggers[user_id] = logger
    return logger

# In-memory state (per dataset)
_models: dict[str, WeightedDistance] = {}
_embeddings: dict[str, dict[str, NDArray[np.floating]]] = {}
_clusters: dict[str, NDArray[np.integer]] = {}  # cluster_id mapping
_cluster_labels: dict[str, dict[str, int]] = {}  # dataset -> {artist_id: cluster_id}
_anchor_counts: dict[str, dict[str, int]] = {}  # dataset -> {artist_id: count}
_triplet_cache: dict[str, list[tuple[str, str, str]]] = {}  # warm updates

# Lock for thread-safe model updates
_model_locks: dict[str, threading.Lock] = {}

# Constants
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


def _state_key(dataset: str, user_id: str | None) -> str:
    """Return a dict key scoping state by dataset and optional user."""
    if user_id is None:
        return dataset
    return f"{dataset}:{user_id}"


def _get_lock(dataset: str, user_id: str | None = None) -> threading.Lock:
    """Get or create a lock for a dataset (and optional user)."""
    key = _state_key(dataset, user_id)
    if key not in _model_locks:
        _model_locks[key] = threading.Lock()
    return _model_locks[key]


def _models_dir(dataset: str, user_id: str | None = None) -> Path:
    """Get the models directory for a dataset (and optional user)."""
    if user_id is not None:
        return MODELS_DIR / dataset / "users" / user_id
    return MODELS_DIR / dataset


def _get_versions(dataset: str, user_id: str | None = None) -> list[int]:
    """Get all existing version numbers for a dataset, sorted ascending."""
    models_dir = _models_dir(dataset, user_id)
    if not models_dir.exists():
        return []
    versions = []
    for f in models_dir.glob("v*.safetensors"):
        try:
            version = int(f.stem[1:])  # Extract number from "v001"
            versions.append(version)
        except ValueError:
            continue
    return sorted(versions)


def _get_latest_version(
    dataset: str, user_id: str | None = None,
) -> int | None:
    """Get the latest version number for a dataset, or None if no versions exist."""
    versions = _get_versions(dataset, user_id)
    return versions[-1] if versions else None


def _get_next_version(dataset: str, user_id: str | None = None) -> int:
    """Get the next version number for a dataset."""
    latest = _get_latest_version(dataset, user_id)
    return (latest or 0) + 1


def _version_path(
    dataset: str, version: int, user_id: str | None = None,
) -> Path:
    """Get the path to a specific version's weights file."""
    return _models_dir(dataset, user_id) / f"v{version:03d}.safetensors"


def _metadata_path(
    dataset: str, version: int, user_id: str | None = None,
) -> Path:
    """Get the path to a specific version's metadata file."""
    return _models_dir(dataset, user_id) / f"v{version:03d}.json"


def _legacy_weights_path(dataset: str) -> Path:
    """Get the legacy weights path for migration."""
    return dataset_dir(dataset) / "similarity_weights.safetensors"


def get_model(dataset: str, user_id: str | None = None) -> WeightedDistance:
    """Get or load the model for a dataset (and optional user)."""
    key = _state_key(dataset, user_id)
    if key not in _models:
        load_weights(dataset, user_id)
    return _models[key]


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
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)  # pyright: ignore[reportArgumentType]
        labels = kmeans.fit_predict(X)

        _cluster_labels[dataset] = {
            id_: int(label)
            for id_, label in zip(ids, labels, strict=True)
        }

    return _cluster_labels[dataset]


def get_anchor_counts(
    dataset: str, user_id: str | None = None,
) -> dict[str, int]:
    """Get anchor usage counts from the database."""
    key = _state_key(dataset, user_id)
    if key not in _anchor_counts:
        _refresh_anchor_counts(dataset, user_id)
    return _anchor_counts[key]


def _refresh_anchor_counts(
    dataset: str, user_id: str | None = None,
) -> None:
    """Refresh anchor counts from the database."""
    key = _state_key(dataset, user_id)
    db_path = TRIPLETS_DB
    if not db_path.exists():
        _anchor_counts[key] = {}
        return

    conn = sqlite3.connect(db_path)
    query = (
        "SELECT anchor, COUNT(*) as count FROM triplets"
        " WHERE dataset = ?"
    )
    params: list[str] = [dataset]
    if user_id is not None:
        query += " AND user_id = ?"
        params.append(user_id)
    query += " GROUP BY anchor"
    rows = conn.execute(query, params).fetchall()
    conn.close()

    _anchor_counts[key] = {row[0]: row[1] for row in rows}


def load_weights(dataset: str, user_id: str | None = None) -> None:
    """Load weights from disk or initialize a new model.

    For global models (user_id=None): loads from data/models/{dataset}/,
    falls back to legacy location.
    For per-user models: loads from data/models/{dataset}/users/{user_id}/,
    falls back to fresh model (no global fallback).
    """
    key = _state_key(dataset, user_id)
    embeddings = get_embeddings(dataset)
    dim = len(next(iter(embeddings.values())))

    # Check for versioned weights
    latest_version = _get_latest_version(dataset, user_id)
    if latest_version is not None:
        path = _version_path(dataset, latest_version, user_id)
        tensors = load_file(path)
        model = WeightedDistance(dim)
        model.weights.data = tensors["weights"]
        _models[key] = model
        print(f"Loaded weights from {path} (v{latest_version:03d})")
        return

    # Check legacy location for migration (global models only)
    if user_id is None:
        legacy_path = _legacy_weights_path(dataset)
        if legacy_path.exists():
            tensors = load_file(legacy_path)
            model = WeightedDistance(dim)
            model.weights.data = tensors["weights"]
            _models[key] = model
            print(f"Loaded weights from legacy path {legacy_path}")
            return

    # No weights found, initialize new model
    _models[key] = WeightedDistance(dim)
    print(f"Initialized new model for {key} (dim={dim})")


def save_weights(
    dataset: str, user_id: str | None = None, metadata: dict | None = None,
) -> int:
    """Save weights to disk as a new version.

    Saves to data/models/{dataset}/v{NNN}.safetensors (or per-user subdir)
    with accompanying v{NNN}.json metadata file.

    Returns the version number of the saved model.
    """
    key = _state_key(dataset, user_id)
    if key not in _models:
        raise ValueError(f"No model loaded for {key}")

    model = _models[key]
    version = _get_next_version(dataset, user_id)
    weights_path = _version_path(dataset, version, user_id)
    meta_path = _metadata_path(dataset, version, user_id)

    # Ensure directory exists
    weights_path.parent.mkdir(parents=True, exist_ok=True)

    # Build full metadata for JSON file
    full_metadata = {
        "version": version,
        "timestamp": datetime.now(UTC).isoformat(),
        "dataset": dataset,
    }
    if metadata:
        full_metadata.update(metadata)

    # Save safetensors with basic metadata (string values only, skip lists/dicts)
    tensors = {"weights": model.weights.data}
    safetensors_meta = {}
    if metadata:
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                safetensors_meta[key] = str(value)
    safetensors_meta["version"] = str(version)
    save_file(tensors, weights_path, metadata=safetensors_meta)

    # Save JSON metadata
    with open(meta_path, "w") as f:
        json.dump(full_metadata, f, indent=2)

    print(f"Saved weights to {weights_path} (v{version:03d})")
    return version


def warm_update(
    dataset: str, triplet: tuple[str, str, str],
    user_id: str | None = None,
) -> None:
    """Perform a warm update after a new triplet judgment.

    Args:
        dataset: The dataset name
        triplet: (anchor, positive, negative) - positive is the chosen one
        user_id: Optional user to scope the update to
    """
    key = _state_key(dataset, user_id)
    lock = _get_lock(dataset, user_id)
    with lock:
        # Add to recent triplets cache
        if key not in _triplet_cache:
            _triplet_cache[key] = []
        _triplet_cache[key].append(triplet)

        # Keep only last WARM_UPDATE_BATCH_SIZE triplets
        if len(_triplet_cache[key]) > WARM_UPDATE_BATCH_SIZE:
            _triplet_cache[key] = _triplet_cache[key][-WARM_UPDATE_BATCH_SIZE:]

        # Update anchor counts
        if key not in _anchor_counts:
            _anchor_counts[key] = {}
        anchor = triplet[0]
        _anchor_counts[key][anchor] = _anchor_counts[key].get(anchor, 0) + 1

        # Get model and embeddings
        model = get_model(dataset, user_id)
        embeddings = get_embeddings(dataset)

        # Create dataset from recent triplets
        valid_triplets = [
            t
            for t in _triplet_cache[key]
            if t[0] in embeddings and t[1] in embeddings and t[2] in embeddings
        ]

        if not valid_triplets:
            get_user_logger(user_id).debug(
                f"Warm update [{key}]: no valid triplets in cache"
            )
            return

        # Prepare tensors
        def _to_tensor(idx: int) -> torch.Tensor:
            return torch.stack([
                torch.tensor(embeddings[t[idx]], dtype=torch.float32)
                for t in valid_triplets
            ])

        anchors = _to_tensor(0)
        positives = _to_tensor(1)
        negatives = _to_tensor(2)

        # Run a few gradient steps
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.TripletMarginWithDistanceLoss(distance_function=model, margin=0.2)

        model.train()
        initial_loss = None
        final_loss = None
        for step in range(WARM_UPDATE_STEPS):
            optimizer.zero_grad()
            loss = loss_fn(anchors, positives, negatives)
            if step == 0:
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()
            final_loss = loss.item()

        get_user_logger(user_id).info(
            f"Warm update [{key}]: {len(valid_triplets)} triplets, "
            f"{WARM_UPDATE_STEPS} steps, loss {initial_loss:.4f} -> {final_loss:.4f}"
        )


def full_retrain(dataset: str, user_id: str | None = None) -> dict:
    """Perform a full retrain of the model.

    Returns metrics dict with train_accuracy, baseline_accuracy, etc.
    """
    key = _state_key(dataset, user_id)
    ulog = get_user_logger(user_id)
    ulog.info(f"Full retrain [{key}]: starting")
    lock = _get_lock(dataset, user_id)
    with lock:
        db_path = TRIPLETS_DB
        triplets = load_triplets(db_path, dataset, user_id=user_id)
        embeddings = get_embeddings(dataset)

        if not triplets:
            ulog.warning(f"Full retrain [{key}]: no triplets found")
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
        ulog.info(
            f"Full retrain [{key}]: training on {len(triplets)} triplets"
        )
        loss_history = train(
            model, triplet_loader,
            epochs=100, lr=0.01, margin=0.2,
            l2_weight=0.01,
        )

        # Evaluate
        train_acc = evaluate(model, triplets, embeddings)

        # Update in-memory model
        _models[key] = model

        # Save weights with training metadata
        version = save_weights(
            dataset,
            user_id=user_id,
            metadata={
                "dim": dim,
                "train_accuracy": train_acc,
                "baseline_accuracy": baseline_acc,
                "num_triplets": len(triplets),
                "epochs": 100,
                "learning_rate": 0.01,
                "margin": 0.2,
                "batch_size": 32,
                "l2_weight": 0.01,
                "loss_history": loss_history,
            },
        )

        # Clear triplet cache
        _triplet_cache[key] = []

        ulog.info(
            f"Full retrain [{key}]: complete v{version:03d} | "
            f"accuracy {baseline_acc:.1%} -> {train_acc:.1%}"
            f" (+{train_acc - baseline_acc:.1%})"
        )

        return {
            "version": version,
            "train_accuracy": train_acc,
            "baseline_accuracy": baseline_acc,
            "improvement": train_acc - baseline_acc,
            "num_triplets": len(triplets),
            "dim": dim,
        }


def suggest_triplet(
    dataset: str, user_id: str | None = None, n_candidates: int = 100,
) -> SuggestedTriplet:
    """Suggest a triplet using diversity + uncertainty sampling.

    Args:
        dataset: The dataset name
        user_id: Optional user to scope the suggestion to
        n_candidates: Number of candidate triplets to sample

    Returns:
        SuggestedTriplet with the best scoring triplet
    """
    key = _state_key(dataset, user_id)
    model = get_model(dataset, user_id)
    embeddings = get_embeddings(dataset)
    clusters = get_clusters(dataset)
    anchor_counts = get_anchor_counts(dataset, user_id)

    ids = list(embeddings.keys())
    max_anchor_count = max(anchor_counts.values()) if anchor_counts else 1

    # Sample candidate triplets
    candidates = _sample_candidate_triplets(ids, n_candidates)

    best_score = float("-inf")
    best_triplet = None
    best_uncertainty = 0.0
    best_diversity = 0.0

    # Track all scores for logging
    all_uncertainties = []
    all_diversities = []
    all_scores = []

    model.eval()
    with torch.no_grad():
        for anchor, a, b in candidates:
            # Get embeddings as tensors
            e_anchor = torch.tensor(
                embeddings[anchor], dtype=torch.float32,
            ).unsqueeze(0)
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

            all_uncertainties.append(uncertainty)
            all_diversities.append(diversity)
            all_scores.append(score)

            if score > best_score:
                best_score = score
                best_triplet = (anchor, a, b)
                best_uncertainty = uncertainty
                best_diversity = diversity

    # Log active learning statistics
    ulog = get_user_logger(user_id)
    if all_scores:
        uncertainties = np.array(all_uncertainties)
        diversities = np.array(all_diversities)
        scores = np.array(all_scores)

        ulog.info(
            f"Active learning [{key}]: sampled {n_candidates} candidates | "
            f"uncertainty: mean={uncertainties.mean():.3f},"
            f" max={uncertainties.max():.3f} | "
            f"diversity: mean={diversities.mean():.3f},"
            f" max={diversities.max():.3f} | "
            f"selected: unc={best_uncertainty:.3f},"
            f" div={best_diversity:.3f},"
            f" score={best_score:.3f}"
            f" (rank={int((scores < best_score).sum()) + 1}"
            f"/{len(scores)})"
        )

    if best_triplet is None:
        # Fallback to random
        import random

        ulog.warning(
            f"Active learning [{key}]:"
            " no valid candidates, falling back to random"
        )
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


def get_model_status(
    dataset: str, user_id: str | None = None,
) -> ModelStatus:
    """Get the status of the model for a dataset (and optional user)."""
    key = _state_key(dataset, user_id)
    latest_version = _get_latest_version(dataset, user_id)
    weights_exist = latest_version is not None
    path = (
        _version_path(dataset, latest_version, user_id)
        if latest_version else None
    )

    # Also check legacy location (global models only)
    if user_id is None:
        legacy_path = _legacy_weights_path(dataset)
        if not weights_exist and legacy_path.exists():
            weights_exist = True
            path = legacy_path

    # Count triplets (scoped to user if provided)
    db_path = TRIPLETS_DB
    num_triplets = 0
    if db_path.exists():
        conn = sqlite3.connect(db_path)
        query = (
            "SELECT COUNT(*) FROM triplets"
            " WHERE dataset = ? AND choice IS NOT NULL"
        )
        params: list[str] = [dataset]
        if user_id is not None:
            query += " AND user_id = ?"
            params.append(user_id)
        row = conn.execute(query, params).fetchone()
        num_triplets = row[0] if row else 0
        conn.close()

    # Check if model is loaded
    loaded = key in _models
    dim = None
    train_accuracy = None

    if loaded:
        dim = _models[key].weights.shape[0]

    if weights_exist and path:
        # Try reading JSON metadata first (for versioned models)
        if latest_version is not None:
            meta_path = _metadata_path(dataset, latest_version, user_id)
            if meta_path.exists():
                try:
                    with open(meta_path) as f:
                        metadata = json.load(f)
                        dim = metadata.get("dim") or dim
                        train_accuracy = metadata.get("train_accuracy")
                except Exception:
                    pass

        # Fall back to safetensors metadata
        if dim is None or train_accuracy is None:
            try:
                from safetensors import safe_open

                with safe_open(path, framework="pt") as f:
                    metadata = f.metadata()
                    if metadata:
                        dim = int(metadata.get("dim", 0)) or dim
                        train_accuracy = (
                            float(metadata.get("train_accuracy", 0))
                            or None
                        )
            except Exception:
                pass

    return ModelStatus(
        loaded=loaded,
        dim=dim,
        num_triplets=num_triplets,
        train_accuracy=train_accuracy,
        weights_path=str(path) if weights_exist and path else None,
        weights_exist=weights_exist,
    )


def trigger_background_retrain(
    dataset: str, user_id: str | None = None,
) -> None:
    """Trigger a full retrain in a background thread."""
    key = _state_key(dataset, user_id)
    thread = threading.Thread(
        target=full_retrain, args=(dataset,),
        kwargs={"user_id": user_id}, daemon=True,
    )
    thread.start()
    print(f"Started background retrain for {key}")
