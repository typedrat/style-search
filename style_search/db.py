"""Database access for the triplets/users SQLite database."""

import sqlite3
from contextlib import contextmanager

from style_search.config import TRIPLETS_DB


@contextmanager
def get_db():
    """Get a database connection with WAL mode for better concurrency."""
    TRIPLETS_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(TRIPLETS_DB)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    """Initialize the triplets database."""
    with get_db() as conn:
        # Users table for allowlist
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                token TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS triplets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset TEXT NOT NULL,
                anchor TEXT NOT NULL,
                option_a TEXT NOT NULL,
                option_b TEXT NOT NULL,
                choice TEXT,  -- 'A', 'B', or NULL for skip
                -- 'too_similar', 'anchor_outlier', 'unknown', or NULL
                skip_reason TEXT,
                user_id TEXT,  -- references users(token)
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
            conn.execute(
                "UPDATE triplets SET skip_reason = 'unknown'"
                " WHERE choice IS NULL AND skip_reason IS NULL"
            )
        # Migration: add user_id column if it doesn't exist
        if "user_id" not in columns:
            conn.execute("ALTER TABLE triplets ADD COLUMN user_id TEXT")
        # Create index on user_id (after migration ensures column exists)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_triplets_user ON triplets(user_id)
        """)
        conn.commit()
