"""Central configuration for style-search data paths."""

import os
from pathlib import Path

# Base data directory - configurable via environment variable
DATA_DIR = Path(os.environ.get("STYLE_SEARCH_DATA_DIR", "data"))

# Derived paths
TRIPLETS_DB = DATA_DIR / "triplets.db"
MODELS_DIR = DATA_DIR / "models"
LOG_DIR = DATA_DIR / "logs"


def dataset_chroma_path(dataset: str) -> Path:
    """Get the ChromaDB path for a dataset."""
    return DATA_DIR / dataset / "chroma"


def dataset_dir(dataset: str) -> Path:
    """Get the base directory for a dataset."""
    return DATA_DIR / dataset
