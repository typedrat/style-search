# Style Search

An interactive tool for exploring visual style similarity in image collections. Uses CLIP embeddings to map images into a semantic space, then lets you explore relationships through visualization and train personalized similarity functions from your own judgments.

## Features

- **Embedding Generation** — Extract 1024-dimensional style embeddings using EVA02-CLIP
- **Interactive Visualization** — Explore the embedding space with UMAP projections and force-directed layouts
- **Similarity Search** — Find visually similar items using cosine distance
- **Triplet Training** — Collect comparison judgments to learn a personalized similarity metric
- **Weighted Distance Learning** — Train dimension weights that reflect your preferences

## How It Works

### Embedding Space

Each image is encoded into a 1024-dimensional vector using [EVA02-CLIP](https://github.com/baaivision/EVA/tree/master/EVA-CLIP), a vision-language model trained on large-scale image-text pairs. These embeddings capture semantic and stylistic properties—images with similar visual styles tend to cluster together in this space.

### Visualization

The web interface projects embeddings to 2D using [UMAP](https://umap-learn.readthedocs.io/) for an overview of the collection. When you select an item, the view switches to a force-directed layout that positions other items by their distance to your selection, making it easy to explore neighborhoods.

### Learning Your Preferences

Raw embedding distance captures general visual similarity, but your notion of "similar style" may weight certain features differently. The triplet training interface presents you with comparisons:

> Given anchor A, is option X or option Y more similar?

From these judgments, we learn a weighted Euclidean distance:

```
d(x, y) = sqrt(Σ wᵢ(xᵢ - yᵢ)²)
```

where `w` is a learned 1024-dimensional weight vector. Dimensions important to your preferences get higher weights; irrelevant dimensions get down-weighted.

Training uses triplet margin loss:

```
L = max(0, d(anchor, positive) - d(anchor, negative) + margin)
```

This pushes the model to rank your preferred item closer to the anchor.

### Skip Reasons as Constraints

When you can't decide between two options, you can indicate why:

- **Too similar** — Both options are equally close to the anchor
- **Anchor too different** — The anchor is an outlier relative to both options

These skips become equality constraints during training, encouraging `d(anchor, A) ≈ d(anchor, B)` for those pairs.

## Installation

```bash
# Clone and install
git clone <repo-url>
cd style-search
pip install -e .

# Install frontend dependencies
cd web
npm install
```

## Usage

### 1. Prepare Your Data

Place images in a dataset directory:

```
data/
└── my-dataset/
    ├── image1.jpg
    ├── image2.png
    └── ...
```

### 2. Generate Embeddings

```bash
embed-images my-dataset --model EVA02-CLIP-L-14
```

This creates a ChromaDB collection at `data/my-dataset/chroma/` with embeddings and image URIs.

### 3. Run the Server

```bash
# Start the API server
style-api --reload

# In another terminal, start the frontend
cd web
npm run dev
```

Open http://localhost:5173 to explore your collection.

### 4. Train a Similarity Function (Optional)

Navigate to `/train/my-dataset` to collect triplet judgments. Keyboard shortcuts:

| Key | Action |
|-----|--------|
| A / ← | Choose left option |
| B / → | Choose right option |
| S | Skip: too similar |
| D | Skip: anchor too different |
| Space | Skip: don't know |

Once you have ~100+ judgments, train weights:

```bash
python -m style_search.train_similarity my-dataset --epochs 100
```

This saves learned weights to `data/my-dataset/similarity_weights.pt`.

## API

The FastAPI backend exposes:

| Endpoint | Description |
|----------|-------------|
| `GET /api/datasets` | List available datasets |
| `GET /api/datasets/{name}/artists` | Get all items with metadata |
| `GET /api/datasets/{name}/umap` | Get 2D UMAP coordinates |
| `POST /api/datasets/{name}/similar` | Find similar items |
| `GET /api/datasets/{name}/distances/{id}` | Get distances from one item to all others |
| `POST /api/triplets` | Record a triplet judgment |
| `GET /api/triplets?dataset=X` | Get triplets for a dataset |

## Project Structure

```
style-search/
├── style_search/
│   ├── api.py              # FastAPI backend
│   ├── embed_images.py     # Embedding generation CLI
│   └── train_similarity.py # Weighted distance training
├── web/
│   └── src/
│       ├── routes/         # TanStack Router pages
│       ├── components/     # React components
│       └── api.ts          # API client
├── data/
│   ├── {dataset}/          # Per-dataset storage
│   │   ├── chroma/         # ChromaDB embeddings
│   │   └── similarity_weights.pt
│   └── triplets.db         # SQLite triplet storage
└── TRAINING.md             # Detailed training methodology docs
```

## Tech Stack

**Backend:** Python, FastAPI, ChromaDB, PyTorch, EVA02-CLIP, UMAP

**Frontend:** React, TypeScript, TanStack Router, D3.js, shadcn/ui, Tailwind CSS

## License

MIT
