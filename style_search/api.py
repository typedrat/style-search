#!/usr/bin/env python3
"""FastAPI backend for style-search visualization."""

import sqlite3
from contextlib import contextmanager
from pathlib import Path

import click
import chromadb
import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from style_search import similarity

app = FastAPI(title="Style Search API")

# SQLite database for triplets
TRIPLETS_DB = Path("data/triplets.db")


@contextmanager
def get_db():
    """Get a database connection."""
    TRIPLETS_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(TRIPLETS_DB)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    """Initialize the triplets database."""
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS triplets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset TEXT NOT NULL,
                anchor TEXT NOT NULL,
                option_a TEXT NOT NULL,
                option_b TEXT NOT NULL,
                choice TEXT,  -- 'A', 'B', or NULL for skip
                skip_reason TEXT,  -- 'too_similar', 'anchor_outlier', 'unknown', or NULL
                timestamp INTEGER NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_triplets_dataset ON triplets(dataset)
        """)
        # Migration: add skip_reason column if it doesn't exist
        cursor = conn.execute("PRAGMA table_info(triplets)")
        columns = [row[1] for row in cursor.fetchall()]
        if "skip_reason" not in columns:
            conn.execute("ALTER TABLE triplets ADD COLUMN skip_reason TEXT")
            # Migrate legacy skips to 'unknown'
            conn.execute("UPDATE triplets SET skip_reason = 'unknown' WHERE choice IS NULL AND skip_reason IS NULL")
        conn.commit()


# Initialize database on startup
init_db()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state for loaded collections
_clients: dict[str, chromadb.PersistentClient] = {}
_collections: dict[str, chromadb.Collection] = {}


class ArtistResult(BaseModel):
    id: str
    distance: float | None = None
    metadata: dict
    uri: str | None = None


class SimilarityQuery(BaseModel):
    artist_id: str
    n_results: int = 10


class CollectionInfo(BaseModel):
    name: str
    count: int
    metadata: dict


class TripletCreate(BaseModel):
    dataset: str
    anchor: str
    option_a: str
    option_b: str
    choice: str | None  # 'A', 'B', or None for skip
    skip_reason: str | None = None  # 'too_similar', 'anchor_outlier', 'unknown'
    timestamp: int


class TripletResponse(BaseModel):
    id: int
    dataset: str
    anchor: str
    option_a: str
    option_b: str
    choice: str | None
    skip_reason: str | None
    timestamp: int


class SuggestedTripletResponse(BaseModel):
    anchor: str
    option_a: str
    option_b: str
    uncertainty_score: float
    diversity_score: float


class ModelStatusResponse(BaseModel):
    loaded: bool
    dim: int | None
    num_triplets: int
    train_accuracy: float | None
    weights_path: str | None
    weights_exist: bool


class RetrainResponse(BaseModel):
    train_accuracy: float | None = None
    baseline_accuracy: float | None = None
    improvement: float | None = None
    num_triplets: int
    dim: int | None = None
    error: str | None = None


def get_collection(dataset: str) -> chromadb.Collection:
    """Get or load a ChromaDB collection."""
    if dataset not in _collections:
        db_path = Path(f"data/{dataset}/chroma")
        if not db_path.exists():
            raise HTTPException(404, f"Dataset '{dataset}' not found")
        _clients[dataset] = chromadb.PersistentClient(path=str(db_path))
        _collections[dataset] = _clients[dataset].get_collection(dataset)
    return _collections[dataset]


@app.get("/api/datasets")
def list_datasets() -> list[str]:
    """List available datasets."""
    data_dir = Path("data")
    if not data_dir.exists():
        return []
    return [
        d.name for d in data_dir.iterdir()
        if d.is_dir() and (d / "chroma").exists()
    ]


@app.get("/api/datasets/{dataset}")
def get_dataset_info(dataset: str) -> CollectionInfo:
    """Get info about a dataset."""
    collection = get_collection(dataset)
    return CollectionInfo(
        name=collection.name,
        count=collection.count(),
        metadata=collection.metadata or {},
    )


@app.get("/api/datasets/{dataset}/artists")
def get_all_artists(dataset: str) -> list[ArtistResult]:
    """Get all artists in a dataset with their metadata."""
    collection = get_collection(dataset)
    results = collection.get(include=["metadatas", "uris"])

    artists = []
    for i, id_ in enumerate(results["ids"]):
        artists.append(ArtistResult(
            id=id_,
            metadata=results["metadatas"][i] if results["metadatas"] else {},
            uri=results["uris"][i] if results["uris"] else None,
        ))
    return artists


@app.get("/api/datasets/{dataset}/embeddings")
def get_all_embeddings(dataset: str) -> dict[str, list[float]]:
    """Get all embeddings for UMAP projection."""
    collection = get_collection(dataset)
    results = collection.get(include=["embeddings"])
    return {
        id_: emb
        for id_, emb in zip(results["ids"], results["embeddings"])
    }


@app.post("/api/datasets/{dataset}/similar")
def find_similar(dataset: str, query: SimilarityQuery) -> list[ArtistResult]:
    """Find artists similar to a given artist."""
    collection = get_collection(dataset)

    # Get the embedding for the query artist
    query_result = collection.get(ids=[query.artist_id], include=["embeddings"])
    if not query_result["ids"]:
        raise HTTPException(404, f"Artist '{query.artist_id}' not found")

    query_embedding = query_result["embeddings"][0]

    # Query for similar
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=query.n_results + 1,  # +1 because it includes itself
        include=["metadatas", "uris", "distances"],
    )

    artists = []
    for i, id_ in enumerate(results["ids"][0]):
        if id_ == query.artist_id:
            continue  # Skip self
        artists.append(ArtistResult(
            id=id_,
            distance=results["distances"][0][i] if results["distances"] else None,
            metadata=results["metadatas"][0][i] if results["metadatas"] else {},
            uri=results["uris"][0][i] if results["uris"] else None,
        ))
    return artists[:query.n_results]


@app.get("/api/datasets/{dataset}/distances/{artist_id}")
def get_distances_from(dataset: str, artist_id: str) -> dict[str, float]:
    """Get distances from one artist to all others."""
    collection = get_collection(dataset)

    # Get the embedding for the query artist
    query_result = collection.get(ids=[artist_id], include=["embeddings"])
    if not query_result["ids"]:
        raise HTTPException(404, f"Artist '{artist_id}' not found")

    query_embedding = query_result["embeddings"][0]

    # Query for all artists
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=collection.count(),
        include=["distances"],
    )

    return {
        id_: dist
        for id_, dist in zip(results["ids"][0], results["distances"][0])
    }


@app.get("/api/datasets/{dataset}/umap")
def get_umap_projection(dataset: str, n_neighbors: int = 15, min_dist: float = 0.1) -> dict[str, list[float]]:
    """Get 2D UMAP projection of embeddings."""
    import numpy as np
    from umap import UMAP

    collection = get_collection(dataset)
    results = collection.get(include=["embeddings"])

    ids = results["ids"]
    embeddings = np.array(results["embeddings"])

    reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2)
    coords = reducer.fit_transform(embeddings)

    return {
        id_: coord.tolist()
        for id_, coord in zip(ids, coords)
    }


@app.get("/api/datasets/{dataset}/images/{artist_id:path}")
def get_artist_image(dataset: str, artist_id: str):
    """Serve an artist's image file."""
    collection = get_collection(dataset)
    result = collection.get(ids=[artist_id], include=["uris"])

    if not result["ids"]:
        raise HTTPException(404, f"Artist '{artist_id}' not found")

    uri = result["uris"][0] if result["uris"] else None
    if not uri:
        raise HTTPException(404, f"No image for artist '{artist_id}'")

    # URI should be a file path
    image_path = Path(uri)
    if not image_path.exists():
        raise HTTPException(404, f"Image file not found: {uri}")

    return FileResponse(image_path)


@app.post("/api/triplets")
def create_triplet(triplet: TripletCreate, background_tasks: BackgroundTasks) -> TripletResponse:
    """Store a new triplet judgment."""
    with get_db() as conn:
        cursor = conn.execute(
            """
            INSERT INTO triplets (dataset, anchor, option_a, option_b, choice, skip_reason, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (triplet.dataset, triplet.anchor, triplet.option_a, triplet.option_b, triplet.choice, triplet.skip_reason, triplet.timestamp),
        )
        conn.commit()
        triplet_id = cursor.lastrowid

        # Count triplets for this dataset to check if we need a full retrain
        count_row = conn.execute(
            "SELECT COUNT(*) FROM triplets WHERE dataset = ? AND choice IS NOT NULL",
            (triplet.dataset,),
        ).fetchone()
        triplet_count = count_row[0] if count_row else 0

    # Trigger warm update if this is a real choice (not a skip)
    if triplet.choice is not None:
        # Convert choice to (anchor, positive, negative) format
        if triplet.choice == "A":
            positive, negative = triplet.option_a, triplet.option_b
        else:
            positive, negative = triplet.option_b, triplet.option_a

        try:
            similarity.warm_update(
                triplet.dataset, (triplet.anchor, positive, negative)
            )

            # Trigger full retrain every FULL_RETRAIN_INTERVAL triplets
            if triplet_count > 0 and triplet_count % similarity.FULL_RETRAIN_INTERVAL == 0:
                background_tasks.add_task(similarity.full_retrain, triplet.dataset)
                print(f"Queued background retrain for {triplet.dataset} (triplet #{triplet_count})")
        except Exception as e:
            # Log but don't fail the request
            print(f"Warm update failed: {e}")

    return TripletResponse(
        id=triplet_id,
        dataset=triplet.dataset,
        anchor=triplet.anchor,
        option_a=triplet.option_a,
        option_b=triplet.option_b,
        choice=triplet.choice,
        skip_reason=triplet.skip_reason,
        timestamp=triplet.timestamp,
    )


@app.get("/api/triplets")
def get_triplets(dataset: str | None = None) -> list[TripletResponse]:
    """Get all triplets, optionally filtered by dataset."""
    with get_db() as conn:
        if dataset:
            rows = conn.execute(
                "SELECT * FROM triplets WHERE dataset = ? ORDER BY id",
                (dataset,),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM triplets ORDER BY id").fetchall()
        return [
            TripletResponse(
                id=row["id"],
                dataset=row["dataset"],
                anchor=row["anchor"],
                option_a=row["option_a"],
                option_b=row["option_b"],
                choice=row["choice"],
                skip_reason=row["skip_reason"],
                timestamp=row["timestamp"],
            )
            for row in rows
        ]


class TripletUpdate(BaseModel):
    choice: str | None = None  # 'A', 'B', or None for skip
    skip_reason: str | None = None  # 'too_similar', 'anchor_outlier', 'unknown'


@app.patch("/api/triplets/{triplet_id}")
def update_triplet(triplet_id: int, update: TripletUpdate) -> TripletResponse:
    """Update a triplet's choice and/or skip_reason."""
    with get_db() as conn:
        # Check if triplet exists
        row = conn.execute("SELECT * FROM triplets WHERE id = ?", (triplet_id,)).fetchone()
        if not row:
            raise HTTPException(404, f"Triplet {triplet_id} not found")

        conn.execute(
            "UPDATE triplets SET choice = ?, skip_reason = ? WHERE id = ?",
            (update.choice, update.skip_reason, triplet_id),
        )
        conn.commit()

        row = conn.execute("SELECT * FROM triplets WHERE id = ?", (triplet_id,)).fetchone()
        return TripletResponse(
            id=row["id"],
            dataset=row["dataset"],
            anchor=row["anchor"],
            option_a=row["option_a"],
            option_b=row["option_b"],
            choice=row["choice"],
            skip_reason=row["skip_reason"],
            timestamp=row["timestamp"],
        )


@app.delete("/api/triplets/{triplet_id}")
def delete_triplet(triplet_id: int):
    """Delete a triplet."""
    with get_db() as conn:
        cursor = conn.execute("DELETE FROM triplets WHERE id = ?", (triplet_id,))
        conn.commit()
        if cursor.rowcount == 0:
            raise HTTPException(404, f"Triplet {triplet_id} not found")
        return {"deleted": triplet_id}


@app.get("/api/datasets/{dataset}/suggest-triplet")
def suggest_triplet(dataset: str) -> SuggestedTripletResponse:
    """Suggest a triplet using active learning (uncertainty + diversity sampling)."""
    # Verify dataset exists
    get_collection(dataset)

    try:
        suggested = similarity.suggest_triplet(dataset)
        return SuggestedTripletResponse(
            anchor=suggested.anchor,
            option_a=suggested.option_a,
            option_b=suggested.option_b,
            uncertainty_score=suggested.uncertainty_score,
            diversity_score=suggested.diversity_score,
        )
    except Exception as e:
        raise HTTPException(500, f"Failed to suggest triplet: {e}")


@app.post("/api/datasets/{dataset}/retrain")
def retrain_model(dataset: str, background_tasks: BackgroundTasks) -> RetrainResponse:
    """Trigger a full model retrain."""
    # Verify dataset exists
    get_collection(dataset)

    try:
        # Run synchronously for now to return metrics
        result = similarity.full_retrain(dataset)
        return RetrainResponse(**result)
    except Exception as e:
        raise HTTPException(500, f"Failed to retrain model: {e}")


@app.get("/api/datasets/{dataset}/model-status")
def get_model_status(dataset: str) -> ModelStatusResponse:
    """Get the status of the similarity model for a dataset."""
    # Verify dataset exists
    get_collection(dataset)

    try:
        status = similarity.get_model_status(dataset)
        return ModelStatusResponse(
            loaded=status.loaded,
            dim=status.dim,
            num_triplets=status.num_triplets,
            train_accuracy=status.train_accuracy,
            weights_path=status.weights_path,
            weights_exist=status.weights_exist,
        )
    except Exception as e:
        raise HTTPException(500, f"Failed to get model status: {e}")


@click.command()
@click.option("-h", "--host", default="127.0.0.1", help="Host to bind to")
@click.option("-p", "--port", default=8000, help="Port to bind to")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
def main(host: str, port: int, reload: bool):
    """Run the style-search API server."""
    uvicorn.run(
        "style_search.api:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    main()
