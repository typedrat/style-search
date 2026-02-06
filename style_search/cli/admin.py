"""User management CLI commands."""

import os

import click

from style_search.db import get_db


@click.command()
@click.argument("name")
def add_user(name: str):
    """Add a user to the allowlist. Generates a random token."""
    import hashlib
    import secrets

    token = hashlib.sha256(secrets.token_bytes(32)).hexdigest()

    with get_db() as conn:
        conn.execute("INSERT INTO users (token, name) VALUES (?, ?)", (token, name))
        conn.commit()

    domain = os.environ.get("STYLE_SEARCH_DOMAIN")
    protocol = os.environ.get("STYLE_SEARCH_PROTOCOL", "https")

    print(f"Created user '{name}'")
    print(f"Token: {token}")
    if domain:
        print(f"Share URL: {protocol}://{domain}/?user={token}")


@click.command()
def list_users():
    """List all users in the allowlist with triplet stats."""
    with get_db() as conn:
        rows = conn.execute("""
            SELECT u.token, u.name, u.created_at,
                   COUNT(t.id) as total_triplets,
                   SUM(CASE WHEN t.choice IS NOT NULL THEN 1 ELSE 0 END) as judgments,
                   SUM(CASE WHEN t.choice IS NULL THEN 1 ELSE 0 END) as skips
            FROM users u
            LEFT JOIN triplets t ON u.token = t.user_id
            GROUP BY u.token
            ORDER BY u.created_at
        """).fetchall()
        if not rows:
            print("No users found.")
            return
        for row in rows:
            stats = f"{row['total_triplets']} triplets "
            stats += f"({row['judgments']} judgments, {row['skips']} skips)"
            print(
                " | ".join(
                    [
                        f"{row['name']}: {row['token'][:16]}...",
                        stats,
                        f"created {row['created_at']}",
                    ]
                )
            )


@click.command()
@click.argument("name")
@click.option("--show-token", is_flag=True, help="Show the full token")
def get_user(name: str, show_token: bool):
    """Show details for a user by name."""
    with get_db() as conn:
        row = conn.execute(
            """
            SELECT u.token, u.name, u.created_at,
                   COUNT(t.id) as total_triplets,
                   SUM(CASE WHEN t.choice IS NOT NULL THEN 1 ELSE 0 END) as judgments,
                   SUM(CASE WHEN t.choice IS NULL THEN 1 ELSE 0 END) as skips
            FROM users u
            LEFT JOIN triplets t ON u.token = t.user_id
            WHERE u.name = ?
            GROUP BY u.token
        """,
            (name,),
        ).fetchone()
        if not row:
            print(f"User not found: {name}")
            raise SystemExit(1)
        token = row["token"] if show_token else f"{row['token'][:16]}..."
        stats = f"{row['total_triplets']} triplets "
        stats += f"({row['judgments']} judgments, {row['skips']} skips)"
        print(
            " | ".join(
                [
                    f"{row['name']}: {token}",
                    stats,
                    f"created {row['created_at']}",
                ]
            )
        )


@click.command()
@click.argument("token")
def remove_user(token: str):
    """Remove a user from the allowlist."""
    with get_db() as conn:
        cursor = conn.execute("DELETE FROM users WHERE token = ?", (token,))
        conn.commit()
        if cursor.rowcount == 0:
            print(f"User not found: {token}")
        else:
            print(f"Removed user with token: {token[:16]}...")
