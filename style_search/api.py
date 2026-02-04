#!/usr/bin/env python3
"""FastAPI backend for style-search visualization."""

from pathlib import Path

import click
import chromadb
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Style Search API")

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
