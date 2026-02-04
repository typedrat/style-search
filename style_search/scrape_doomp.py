#!/usr/bin/env python3
"""Scrape all images from a rentry.org doomp page (all A-Z subpages)."""

import json
import os
import re
import signal
import string
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

# Global flag for cancellation
cancelled = False


def handle_sigint(signum, frame):
    global cancelled
    cancelled = True

DEFAULT_WORKERS = os.cpu_count() or 4

import click
import requests
from bs4 import BeautifulSoup
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.panel import Panel

console = Console()


def get_all_page_urls(base_url: str) -> list[str]:
    """Generate URLs for all alphabetical subpages (A-Z)."""
    urls = [base_url]  # A is the main page
    for letter in string.ascii_uppercase[1:]:  # B through Z
        urls.append(f"{base_url}-{letter}")
    return urls


def fetch_page(url: str) -> str:
    """Fetch the HTML content of a page."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.text


def extract_settings_table(html: str) -> dict:
    """Extract the generation settings table from the page."""
    soup = BeautifulSoup(html, "html.parser")
    settings = {}

    table = soup.find("table")
    if not table:
        return settings

    for row in table.find_all("tr"):
        cells = row.find_all(["td", "th"])
        if len(cells) >= 2:
            key = cells[0].get_text(strip=True).lower()
            value = cells[1].get_text(strip=True)
            if key and value:
                settings[key] = value

    return settings


def extract_artist_data(html: str) -> dict[str, dict]:
    """Extract artist data from HTML."""
    soup = BeautifulSoup(html, "html.parser")
    artists = {}

    for strong in soup.find_all("strong"):
        artist_name = strong.get_text(strip=True)
        if not artist_name:
            continue

        entry = {"name": artist_name}

        prev = strong.find_previous("a")
        while prev:
            if prev.find("img") and prev.get("href"):
                href = prev["href"]
                if any(href.lower().endswith(ext) for ext in ('.jpg', '.jpeg', '.png', '.gif', '.webp')):
                    entry["image_url"] = href
                    break
            prev = prev.find_previous("a")

        if "image_url" not in entry:
            continue

        e621_link = strong.find_next("a")
        if e621_link and "e621" in e621_link.get("href", ""):
            entry["e621_url"] = e621_link["href"]

        parent = strong.parent
        if parent:
            text = parent.get_text()
            posts_match = re.search(r'Posts:\s*(\d+)', text)
            score_match = re.search(r'Avg Score:\s*(\d+)', text)
            if posts_match:
                entry["posts"] = int(posts_match.group(1))
            if score_match:
                entry["avg_score"] = int(score_match.group(1))

        artists[artist_name] = entry

    return artists


def get_extension(url: str) -> str:
    """Extract file extension from URL."""
    parsed = urlparse(url)
    path = parsed.path.lower()
    for ext in ('.jpg', '.jpeg', '.png', '.gif', '.webp'):
        if path.endswith(ext):
            return ext
    return '.jpg'


def download_image(artist: str, url: str, output_dir: Path) -> tuple[str, bool | None, str, int]:
    """Download a single image. Returns (artist, success, message, bytes)."""
    try:
        safe_name = re.sub(r'[<>:"/\\|?*]', '_', artist)
        ext = get_extension(url)
        filename = f"{safe_name}{ext}"
        filepath = output_dir / filename

        if filepath.exists():
            return (artist, None, filename, filepath.stat().st_size)  # None = skipped

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        content = response.content
        filepath.write_bytes(content)
        return (artist, True, filename, len(content))

    except Exception as e:
        return (artist, False, str(e), 0)


@click.command()
@click.argument("url")
@click.option("-o", "--output", default="data/doomp", help="Output directory")
@click.option("-w", "--workers", default=DEFAULT_WORKERS, show_default=True, help="Parallel workers")
def main(url: str, output: str, workers: int):
    """Scrape images from a rentry.org doomp (A-Z subpages)."""
    global cancelled
    signal.signal(signal.SIGINT, handle_sigint)

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(Panel(f"[bold]Doomp Scraper[/bold]\n{url}", subtitle=f"Output: {output_dir.absolute()}"))

    # Fetch all pages in parallel
    page_urls = get_all_page_urls(url)
    all_artists = {}
    settings = {}

    def fetch_and_parse(page_url: str) -> tuple[str, dict | None, dict | None, str | None]:
        """Fetch a page and extract data. Returns (letter, settings, artists, error)."""
        letter = page_url.split("-")[-1] if "-" in page_url.split("/")[-1] else "A"
        if letter == "doomp":
            letter = "A"
        try:
            html = fetch_page(page_url)
            page_settings = extract_settings_table(html) if letter == "A" else None
            page_artists = extract_artist_data(html)
            return (letter, page_settings, page_artists, None)
        except requests.RequestException as e:
            return (letter, None, None, str(e))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TextColumn("ETA"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        fetch_task = progress.add_task("[cyan]Fetching pages...", total=len(page_urls))

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(fetch_and_parse, url): url for url in page_urls}

            for future in as_completed(futures):
                if cancelled:
                    for f in futures:
                        f.cancel()
                    console.print("\n[yellow]Cancelled.[/yellow]")
                    sys.exit(130)

                letter, page_settings, page_artists, error = future.result()

                if error:
                    console.print(f"  [red]Failed page {letter}: {error}[/red]")
                else:
                    if page_settings:
                        settings = page_settings
                    if page_artists:
                        all_artists.update(page_artists)

                progress.advance(fetch_task)

    console.print(f"\n[green]Found {len(all_artists)} artists across {len(page_urls)} pages[/green]")

    # Show settings table
    if settings:
        table = Table(title="Generation Settings", show_header=True)
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="white", max_width=80)
        for key, value in settings.items():
            table.add_row(key, value[:80] + "..." if len(value) > 80 else value)
        console.print(table)

    # Save metadata
    metadata = {"settings": settings, "artists": all_artists}
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    console.print(f"\n[dim]Metadata saved to {metadata_path}[/dim]")

    # Download images
    downloaded = 0
    skipped = 0
    failed = 0
    total_bytes = 0

    start_time = time.time()

    def format_size(b: int) -> str:
        for unit in ("B", "KB", "MB", "GB"):
            if b < 1024:
                return f"{b:.1f} {unit}"
            b /= 1024
        return f"{b:.1f} TB"

    def format_speed(bps: float) -> str:
        return f"{format_size(bps)}/s"

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TextColumn("ETA"),
        TimeRemainingColumn(),
        TextColumn("•"),
        TextColumn("{task.fields[size]}"),
        TextColumn("{task.fields[speed]}"),
        console=console,
    ) as progress:
        download_task = progress.add_task(
            "[cyan]Downloading...",
            total=len(all_artists),
            size="0 B",
            speed="0 B/s",
        )

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(download_image, artist, data["image_url"], output_dir): artist
                for artist, data in all_artists.items()
            }

            for future in as_completed(futures):
                if cancelled:
                    for f in futures:
                        f.cancel()
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

                artist, success, message, size = future.result()

                if success is None:
                    skipped += 1
                elif success:
                    downloaded += 1
                else:
                    failed += 1
                    console.print(f"  [red]Failed {artist}: {message}[/red]")

                total_bytes += size
                elapsed = time.time() - start_time
                speed = total_bytes / elapsed if elapsed > 0 else 0

                progress.update(
                    download_task,
                    advance=1,
                    size=format_size(total_bytes),
                    speed=format_speed(speed),
                )

    if cancelled:
        console.print("\n[yellow]Cancelled. Partial results saved.[/yellow]")
        sys.exit(130)

    # Summary
    console.print()
    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="bold")
    summary.add_column()
    summary.add_row("[green]Downloaded:", str(downloaded))
    summary.add_row("[yellow]Skipped:", str(skipped))
    summary.add_row("[red]Failed:", str(failed))
    summary.add_row("[dim]Total size:", f"{total_bytes / 1024 / 1024:.1f} MB")
    summary.add_row("[blue]Output:", str(output_dir.absolute()))
    console.print(Panel(summary, title="Complete"))


if __name__ == "__main__":
    main()
