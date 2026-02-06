#!/usr/bin/env python3
"""
CLI tool for managing JutulGPT documentation cache.

Usage:
    python scripts/manage_cache.py status    # Show cache status
    python scripts/manage_cache.py clear     # Clear all caches
    python scripts/manage_cache.py update    # Force update all caches
    python scripts/manage_cache.py extract   # Extract function docs
"""

import argparse
import shutil
from pathlib import Path

from rich.console import Console
from rich.table import Table

# Setup
console = Console()


def get_cache_root() -> Path:
    """Get the cache root directory."""
    from jutulgpt.configuration import PROJECT_ROOT

    return PROJECT_ROOT.parent / ".cache"


def show_status():
    """Show cache status."""
    from jutulgpt.rag.package_path import get_package_root, get_package_version

    cache_root = get_cache_root()

    table = Table(title="JutulGPT Cache Status")
    table.add_column("Cache Type", style="cyan")
    table.add_column("Location", style="magenta")
    table.add_column("Size", style="green")
    table.add_column("Exists", style="yellow")

    # Check various cache locations
    caches = {
        "Metadata": cache_root / "jutuldarcy",
        "Vector Stores": cache_root / "retriever_store",
        "Loaded Docs": cache_root / "loaded_store",
        "Function Docs": cache_root / "jutuldarcy_function_docs.pkl",
    }

    total_size = 0
    for cache_name, cache_path in caches.items():
        if cache_path.exists():
            if cache_path.is_dir():
                size = sum(f.stat().st_size for f in cache_path.rglob("*") if f.is_file())
            else:
                size = cache_path.stat().st_size

            size_mb = size / (1024 * 1024)
            total_size += size
            table.add_row(
                cache_name,
                str(cache_path.relative_to(cache_root.parent)),
                f"{size_mb:.2f} MB",
                "✓",
            )
        else:
            table.add_row(cache_name, str(cache_path.relative_to(cache_root.parent)), "-", "✗")

    console.print(table)
    console.print(f"\n[bold]Total cache size:[/bold] {total_size / (1024 * 1024):.2f} MB")

    # Show package info
    root = get_package_root()
    console.print("\n[bold]JutulDarcy Package:[/bold]")
    if root is not None:
        version = get_package_version(root)
        console.print(f"  Package Path: {root}")
        console.print(f"  Package Version: {version or 'N/A'}")
    else:
        console.print("  [yellow]Not installed[/yellow]")

    # Show function docs info
    func_docs_path = cache_root / "jutuldarcy_function_docs.pkl"
    if func_docs_path.exists():
        import pickle
        with open(func_docs_path, "rb") as f:
            func_docs = pickle.load(f)
        console.print("\n[bold]Function Documentation:[/bold]")
        console.print(f"  Total functions: {len(func_docs)}")
        console.print(f"  Location: {func_docs_path.relative_to(cache_root.parent)}")


def clear_cache():
    """Clear all caches."""
    cache_root = get_cache_root()

    if not cache_root.exists():
        console.print("[yellow]No cache directory found.[/yellow]")
        return

    # Confirm
    console.print(f"[bold red]This will delete all cached data in {cache_root}[/bold red]")
    response = input("Are you sure? (yes/no): ")

    if response.lower() != "yes":
        console.print("[yellow]Cancelled.[/yellow]")
        return

    # Remove cache
    shutil.rmtree(cache_root)
    console.print("[green]✓ Cache cleared successfully![/green]")


def update_cache():
    """Force update: clear derived caches, re-resolve package path, and extract function docs."""
    from jutulgpt.rag.package_path import _clear_derived_caches, get_package_root

    console.print("[cyan]Clearing derived caches...[/cyan]")
    _clear_derived_caches()

    console.print("[cyan]Resolving JutulDarcy package path...[/cyan]")
    root = get_package_root()
    if root is None:
        console.print("[red]JutulDarcy is not installed. Cannot update.[/red]")
        return
    console.print(f"[green]✓ Package root: {root}[/green]")

    # Also update function docs
    console.print("[cyan]Extracting function documentation...[/cyan]")
    from jutulgpt.julia.extract_docs import extract_jutuldarcy_documentation

    docs = extract_jutuldarcy_documentation(force_refresh=True)
    console.print(f"[green]✓ Extracted {len(docs)} function documentations[/green]")

    cache_root = get_cache_root()
    cache_path = cache_root / "jutuldarcy_function_docs.pkl"
    console.print(f"[dim]Saved to: {cache_path}[/dim]")


def extract_function_docs():
    """Extract function documentation."""
    console.print("[cyan]Extracting function documentation from JutulDarcy...[/cyan]")

    from jutulgpt.julia.extract_docs import extract_jutuldarcy_documentation

    docs = extract_jutuldarcy_documentation(force_refresh=True)
    console.print(f"[green]✓ Extracted {len(docs)} function documentations[/green]")

    # Show sample
    if docs:
        console.print("\n[bold]Sample functions:[/bold]")
        for i, func_name in enumerate(list(docs.keys())[:5]):
            console.print(f"  - {func_name}")


def main():
    parser = argparse.ArgumentParser(description="Manage JutulGPT documentation cache")
    parser.add_argument(
        "command",
        choices=["status", "clear", "update", "extract"],
        help="Command to execute",
    )

    args = parser.parse_args()

    if args.command == "status":
        show_status()
    elif args.command == "clear":
        clear_cache()
    elif args.command == "update":
        update_cache()
    elif args.command == "extract":
        extract_function_docs()


if __name__ == "__main__":
    main()
