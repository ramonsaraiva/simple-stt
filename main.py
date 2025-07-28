#!/usr/bin/env python3

import argparse
import logging
import sys

from audio_recorder import AudioRecorder
from clipboard_manager import ClipboardManager
from config import Config
from llm_refiner import LLMRefiner
from stt_processor import STTProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("stt.log")],
)
logger = logging.getLogger(__name__)


def tune_threshold():
    """Tune the silence threshold and update config"""
    try:
        logger.info("Starting silence threshold tuning")
        config = Config()
        audio_recorder = AudioRecorder(config)

        # Run tuning
        optimal_threshold = audio_recorder.tune_silence_threshold()
        if optimal_threshold is None:
            logger.error("Tuning failed")
            return

        # Update config
        config.config["audio"]["silence_threshold"] = optimal_threshold
        config.save()

        logger.info(f"Updated config with new threshold: {optimal_threshold:.1f}")
        print(f"✅ Updated config with new threshold: {optimal_threshold:.1f}")
        print("You can now use the STT system with the optimized settings!")

    except Exception as e:
        logger.error(f"Failed to tune threshold: {e}")
        print(f"❌ Failed to tune threshold: {e}")
        sys.exit(1)


def run_stt(profile=None):
    """Run the speech-to-text process"""
    try:
        logger.info("Starting STT process")

        # Initialize components
        config = Config()
        audio_recorder = AudioRecorder(config)
        stt_processor = STTProcessor(config)
        llm_refiner = LLMRefiner(config)
        clipboard_manager = ClipboardManager(config)

        # Start loading the Whisper model in the background
        print("🚀 Starting model loading and recording...")
        stt_processor.start_loading_async()

        # Record until silence (happens in parallel with model loading)
        audio_file = audio_recorder.record_until_silence()
        if not audio_file:
            logger.warning("No audio recorded")
            print("❌ No audio recorded")
            return

        try:
            # Wait for model to finish loading (if it hasn't already)
            print("⏳ Ensuring model is ready...")
            timeout = config.get("whisper.load_timeout", 60)
            if not stt_processor.wait_for_model(timeout):
                print("❌ Failed to load Whisper model")
                return

            # Transcribe audio
            logger.info("Transcribing audio")
            print("🔄 Transcribing audio...")
            text = stt_processor.transcribe(audio_file)
            if not text:
                logger.warning("No speech detected in audio")
                print("❌ No speech detected")
                return

            logger.info(f"Transcribed text: {text}")

            # Refine text with LLM
            if profile:
                logger.info(f"Refining text with LLM using profile: {profile}")
                print(f"🔄 Refining text using '{profile}' profile...")
            else:
                logger.info("Refining text with LLM using default profile")
                print("🔄 Refining text...")

            refined_text = llm_refiner.refine_text(text, profile)
            if not refined_text:
                refined_text = text

            logger.info(f"Refined text: {refined_text}")

            # Handle clipboard/paste
            if config.get("clipboard.auto_paste", False):
                clipboard_manager.paste_text(refined_text)
                logger.info("Text auto-pasted to active window")
            else:
                clipboard_manager.copy_to_clipboard(refined_text)
                logger.info("Text copied to clipboard")

            print("✅ Processing complete!")
            logger.info("STT process completed successfully")

        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            print(f"❌ Error processing audio: {e}")
        finally:
            # Cleanup temporary audio file
            audio_recorder.cleanup()

    except Exception as e:
        logger.error(f"Failed to start STT: {e}")
        print(f"❌ Failed to start STT: {e}")
        sys.exit(1)


def list_profiles():
    """List all available LLM profiles"""
    try:
        config = Config()
        from llm_refiner import LLMRefiner

        llm_refiner = LLMRefiner(config)
        profiles = llm_refiner.list_profiles()

        print("📝 Available LLM profiles:")
        for profile_id, profile_data in profiles.items():
            name = profile_data.get("name", profile_id)
            print(f"  • {profile_id}: {name}")

        default_profile = config.get("llm.default_profile", "general")
        print(f"\n🎯 Default profile: {default_profile}")

    except Exception as e:
        logger.error(f"Failed to list profiles: {e}")
        print(f"❌ Failed to list profiles: {e}")


def main():
    parser = argparse.ArgumentParser(description="Speech-to-Text System")
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

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.list_profiles:
        list_profiles()
    elif args.tune:
        tune_threshold()
    else:
        run_stt(args.profile)


if __name__ == "__main__":
    main()
