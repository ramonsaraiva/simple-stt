#!/usr/bin/env python3

import argparse
import logging
import signal
import sys

from audio_recorder import AudioRecorder
from clipboard_manager import ClipboardManager
from config import Config
from llm_refiner import LLMRefiner
from stt_processor import STTProcessor
from ui_overlay import UIManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("stt.log")],
)
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
_shutdown_requested = False

def signal_handler(signum, frame):
    """Handle SIGINT (Ctrl+C) to terminate immediately"""
    global _shutdown_requested
    _shutdown_requested = True
    logger.info("Termination signal received, shutting down immediately")
    print("\n🛑 Terminating process...")
    sys.exit(0)


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
    ui_manager = None
    try:
        logger.info("Starting STT process")
        
        # Set up signal handler for Ctrl+C
        signal.signal(signal.SIGINT, signal_handler)

        # Initialize components
        config = Config()
        audio_recorder = AudioRecorder(config)
        stt_processor = STTProcessor(config)
        llm_refiner = LLMRefiner(config)
        clipboard_manager = ClipboardManager(config)

        # Initialize and start UI
        ui_manager = UIManager(config)
        ui_manager.start_ui()

        # Start loading the Whisper model in the background
        print("🚀 Starting model loading and recording...")
        stt_processor.start_loading_async()

        # Start recording and update UI
        ui_manager.start_recording(profile or "general")

        # Start a background thread to check model loading status
        def check_model_loading():
            import time

            while ui_manager.overlay and ui_manager.overlay.is_recording:
                if stt_processor._model_loaded.is_set() and stt_processor.model:
                    ui_manager.set_model_ready(True)
                    break
                time.sleep(1.0)  # Check less frequently to reduce interference

        import threading

        threading.Thread(target=check_model_loading, daemon=True).start()

        # Record until silence (happens in parallel with model loading)
        audio_file = audio_recorder.record_until_silence()

        # Update UI - recording stopped
        ui_manager.stop_recording()

        if not audio_file:
            logger.warning("No audio recorded")
            print("❌ No audio recorded")
            ui_manager.set_status("❌ No audio recorded", "#ff4444")
            return

        try:
            # Wait for model to finish loading (if it hasn't already)
            print("⏳ Ensuring model is ready...")
            ui_manager.set_status("⏳ Waiting for model...", "#ffaa00")
            timeout = config.get("whisper.load_timeout", 60)
            if not stt_processor.wait_for_model(timeout):
                print("❌ Failed to load Whisper model")
                ui_manager.set_status("❌ Model loading failed", "#ff4444")
                return

            # Update UI - model is ready
            ui_manager.set_model_ready(True)

            # Transcribe audio
            logger.info("Transcribing audio")
            print("🔄 Transcribing audio...")
            ui_manager.set_status("🔄 Transcribing audio...", "#00aaff")
            text = stt_processor.transcribe(audio_file)
            if not text:
                logger.warning("No speech detected in audio")
                print("❌ No speech detected")
                ui_manager.set_status("❌ No speech detected", "#ff4444")
                return

            logger.info(f"Transcribed text: {text}")

            # Refine text with LLM
            if profile:
                logger.info(f"Refining text with LLM using profile: {profile}")
                print(f"🔄 Refining text using '{profile}' profile...")
                ui_manager.set_status(f"🔄 Refining ({profile})...", "#00aaff")
            else:
                logger.info("Refining text with LLM using default profile")
                print("🔄 Refining text...")
                ui_manager.set_status("🔄 Refining text...", "#00aaff")

            refined_text = llm_refiner.refine_text(text, profile)
            if not refined_text:
                refined_text = text

            logger.info(f"Refined text: {refined_text}")

            # Handle clipboard/paste
            if config.get("clipboard.auto_paste", False):
                clipboard_manager.paste_text(refined_text)
                logger.info("Text auto-pasted to active window")
                ui_manager.set_status("✅ Auto-pasted!", "#00ff00")
            else:
                clipboard_manager.copy_to_clipboard(refined_text)
                logger.info("Text copied to clipboard")
                ui_manager.set_status("✅ Copied to clipboard!", "#00ff00")

            print("✅ Processing complete!")
            logger.info("STT process completed successfully")

            # Wait a bit for UI to auto-hide if enabled
            if (
                ui_manager.enabled
                and ui_manager.overlay
                and ui_manager.overlay.auto_hide_delay > 0
            ):
                import time

                time.sleep(min(ui_manager.overlay.auto_hide_delay + 0.5, 5.0))

        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            print(f"❌ Error processing audio: {e}")
        finally:
            # Cleanup temporary audio file
            audio_recorder.cleanup()

    except Exception as e:
        logger.error(f"Failed to start STT: {e}")
        print(f"❌ Failed to start STT: {e}")
        if ui_manager:
            ui_manager.set_status("❌ System error", "#ff4444")
        sys.exit(1)
    finally:
        # Cleanup UI resources
        if ui_manager:
            ui_manager.cleanup()


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
