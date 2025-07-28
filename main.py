#!/usr/bin/env python3
"""
Legacy entry point for backward compatibility.

This file provides backward compatibility for the old main.py entry point.
New installations should use 'stt' command or 'python -m stt.cli'.
"""

import sys
import warnings
from pathlib import Path

# Add src directory to path so we can import the new structure
sys.path.insert(0, str(Path(__file__).parent / "src"))


def main():
    """Legacy main function that delegates to the new CLI."""
    warnings.warn(
        "Using main.py directly is deprecated. Use 'stt' command or 'python -m stt.cli' instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    try:
        from stt.cli import main as cli_main

        cli_main()
    except ImportError as e:
        print(f"❌ Failed to import new STT modules: {e}")
        print("Make sure you've installed the package properly.")
        sys.exit(1)


if __name__ == "__main__":
    main()
