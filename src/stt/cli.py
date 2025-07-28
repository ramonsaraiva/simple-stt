"""Command-line interface for the STT system."""

import argparse
import logging
import sys
from pathlib import Path

from .config.manager import ConfigError
from .services.orchestrator import STTError, STTOrchestrator


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration.

    Args:
        verbose: Enable debug logging if True
    """
    level = logging.DEBUG if verbose else logging.INFO

    # Configure logging
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("stt.log"),
        ],
    )


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Speech-to-Text System with LLM Enhancement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  stt                           # Run with default settings
  stt --profile todo            # Use 'todo' LLM profile
  stt --no-llm                  # Skip LLM refinement
  stt --tune                    # Tune silence threshold
  stt --list-profiles           # Show available LLM profiles
  stt --config /path/config.yaml # Use custom config file
        """,
    )

    parser.add_argument(
        "--tune",
        action="store_true",
        help="Tune the silence threshold for your microphone and environment",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "--profile",
        "-p",
        type=str,
        help="Use a specific LLM profile (e.g., slack, email, todo, obsidian)",
    )
    parser.add_argument(
        "--list-profiles", action="store_true", help="List all available LLM profiles"
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM refinement and use raw transcription",
    )
    parser.add_argument("--config", "-c", type=Path, help="Path to configuration file")

    args = parser.parse_args()

    # Set up logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    try:
        # Initialize orchestrator
        orchestrator = STTOrchestrator(args.config)

        # Handle different commands
        if args.list_profiles:
            orchestrator.list_profiles()
        elif args.tune:
            orchestrator.tune_threshold()
        else:
            # Run transcription workflow
            orchestrator.run_transcription(args.profile, no_llm=args.no_llm)

    except ConfigError as e:
        logger.error(f"Configuration error: {e}")
        print(f"❌ Configuration error: {e}")
        print("Check your config file and ensure all required settings are present.")
        sys.exit(1)

    except STTError as e:
        logger.error(f"STT system error: {e}")
        print(f"❌ STT error: {e}")
        sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        print("\n🛑 Interrupted by user")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"❌ Unexpected error: {e}")
        print("Check the log file (stt.log) for more details.")
        sys.exit(1)

    finally:
        try:
            if "orchestrator" in locals():
                orchestrator.cleanup()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


if __name__ == "__main__":
    main()
