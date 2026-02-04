# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Style Search is a full-stack application for exploring visual style similarity in image collections. It uses CLIP embeddings to map images into a semantic space, provides interactive visualization, and supports training personalized similarity functions from triplet judgments.

**Stack:** Python (FastAPI) backend + React (TypeScript) frontend

## Development Commands

### Running the App

```bash
# Combined (recommended)
process-compose up

# Or separately:
style-api --reload          # API on :8000
cd web && npm run dev       # Frontend on :5173
```

### Backend

```bash
# Generate embeddings for a dataset
embed-images <dataset_dir>

# Train similarity model
python -m style_search.train_similarity <dataset> --epochs 100

# Multi-dataset training
python -m style_search.train_similarity dataset1 dataset2 -o combined.safetensors

# Linting and type checking
uv run ruff check style_search/
uv run basedpyright style_search/
```

### Frontend

```bash
cd web
npm run dev      # Dev server
npm run build    # Production build
npm run lint     # ESLint
```

## Architecture

### Backend (`style_search/`)

- `api.py` — FastAPI server with REST endpoints for datasets, embeddings, similarity search, triplets, and model management
- `similarity.py` — Model loading, warm updates (few gradient steps after each judgment), full retraining, and active learning triplet suggestions
- `train_similarity.py` — WeightedDistance model (1024 learnable dimension weights), triplet margin loss training, multi-dataset support
- `embed_images.py` — EVA02-CLIP embedding generation into ChromaDB

### Frontend (`web/src/`)

- `routes/` — TanStack Router file-based routing:
  - `index.tsx` — Dataset listing
  - `$dataset.tsx` — Main exploration view with UMAP scatterplot and similarity search
  - `train.$dataset/index.tsx` — Triplet training interface with keyboard shortcuts (A/B for choice, S/D for skip reasons)
- `components/` — React components including D3 scatterplot visualization
- `api.ts` — TypeScript API client

### Data Storage

```
data/
├── {dataset}/
│   ├── chroma/                        # ChromaDB vector store
│   └── similarity_weights.safetensors # Trained model weights
└── triplets.db                        # SQLite triplet judgments
```

## Key Concepts

**Weighted Euclidean Distance:** The model learns 1024 dimension weights. Training uses triplet margin loss — given an anchor, push the user's preferred item closer than the alternative.

**Active Learning:** Suggests triplets using uncertainty (model confusion) + diversity (cluster coverage) sampling. Warm updates run after each judgment; full retrain every 20 judgments.

**Skip Constraints:** "Too similar" or "anchor outlier" skips become equality constraints during training (d(anchor,A) ≈ d(anchor,B)).
