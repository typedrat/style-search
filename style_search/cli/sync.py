"""Sync triplets from a remote Style Search server."""

import json
import urllib.request

import click

from style_search.db import get_db


@click.command()
@click.argument("server_url")
@click.option("--dataset", "-d", required=True, help="Dataset to sync triplets for")
@click.option("--user", "-u", default=None, help="User token to filter by")
def sync(server_url: str, dataset: str, user: str | None):
    """Pull triplets from a remote server into the local database."""
    server_url = server_url.rstrip("/")
    url = f"{server_url}/api/triplets?dataset={dataset}"
    if user:
        url += f"&user={user}"

    click.echo(f"Fetching triplets from {url}")
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        triplets = json.loads(resp.read())

    click.echo(f"Received {len(triplets)} triplets")

    new = 0
    skipped = 0
    with get_db() as conn:
        for t in triplets:
            # Check for existing triplet by natural key
            existing = conn.execute(
                """SELECT 1 FROM triplets
                   WHERE dataset = ? AND anchor = ? AND option_a = ?
                     AND option_b = ? AND user_id IS ? AND timestamp = ?
                   LIMIT 1""",
                (
                    t["dataset"],
                    t["anchor"],
                    t["option_a"],
                    t["option_b"],
                    t["user_id"],
                    t["timestamp"],
                ),
            ).fetchone()
            if existing:
                skipped += 1
                continue
            conn.execute(
                """INSERT INTO triplets
                   (dataset, anchor, option_a, option_b,
                    choice, skip_reason, user_id, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    t["dataset"],
                    t["anchor"],
                    t["option_a"],
                    t["option_b"],
                    t.get("choice"),
                    t.get("skip_reason"),
                    t.get("user_id"),
                    t["timestamp"],
                ),
            )
            new += 1
        conn.commit()

    click.echo(f"Done: {new} new, {skipped} skipped (already exist)")
