"""Main orchestrator service for the STT system."""

import logging
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from ..audio.recorder import AudioError, AudioRecorder
from ..config.manager import ConfigManager
from ..services.clipboard import ClipboardError, ClipboardService
from ..speech.enhancer import EnhancementError, TextEnhancer
from ..speech.transcriber import SpeechTranscriber, TranscriptionError
from ..ui.overlay import THEME, UIManager

logger = logging.getLogger(__name__)


class STTError(Exception):
    """Base exception for STT system errors."""


class STTOrchestrator:
    """Main service that orchestrates the STT workflow."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize STT orchestrator.

        Args:
            config_path: Optional path to configuration file
        """
        self.config = ConfigManager(config_path)
        self.audio_recorder = AudioRecorder(self.config)
        self.transcriber = SpeechTranscriber(self.config)
        self.enhancer = TextEnhancer(self.config)
        self.clipboard = ClipboardService(self.config)
        self.ui_manager = UIManager(self.config)

        self._shutdown_requested = False
        self._setup_signal_handlers()

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""

        def signal_handler(signum, frame):
            self._shutdown_requested = True
            logger.info("Termination signal received, shutting down immediately")
            print("\n🛑 Terminating process...")
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

    def tune_threshold(self) -> None:
        """Tune the silence threshold for optimal voice detection."""
        try:
            logger.info("Starting silence threshold tuning")

            # Run tuning
            optimal_threshold = self.audio_recorder.tune_silence_threshold()
            if optimal_threshold is None:
                logger.error("Tuning failed")
                return

            # Update config
            self.config.set("audio.silence_threshold", optimal_threshold)
            self.config.save()

            logger.info(f"Updated config with new threshold: {optimal_threshold:.1f}")
            print(f"✅ Updated config with new threshold: {optimal_threshold:.1f}")
            print("You can now use the STT system with the optimized settings!")

        except Exception as e:
            logger.error(f"Failed to tune threshold: {e}")
            print(f"❌ Failed to tune threshold: {e}")
            raise STTError(f"Threshold tuning failed: {e}") from e

    def run_transcription(
        self, profile: Optional[str] = None, no_llm: bool = False
    ) -> None:
        """Run a complete transcription workflow.

        Args:
            profile: LLM profile to use for enhancement
            no_llm: Skip LLM enhancement if True
        """
        try:
            logger.info("Starting STT process")

            # Initialize and start UI
            self.ui_manager.start_ui()
            
            # Set up device name display
            device_name = self.audio_recorder.get_selected_device_name()
            self.ui_manager.set_device_name(device_name)
            
            # Set up volume monitoring callback
            self.audio_recorder.set_volume_callback(self.ui_manager.set_volume_level)

            # Start loading the Whisper model in the background
            print("🚀 Starting model loading and recording...")
            self.transcriber.start_loading_async()

            # Start recording and update UI
            self.ui_manager.start_recording(profile or "general")

            # Start a background thread to check model loading status
            self._start_model_loading_monitor()

            # Record until silence (happens in parallel with model loading)
            audio_file = self.audio_recorder.record_until_silence(
                voice_detected_callback=self.ui_manager.voice_detected
            )

            # Update UI - recording stopped
            self.ui_manager.stop_recording()

            if not audio_file:
                logger.warning("No audio recorded")
                print("❌ No audio recorded")
                self.ui_manager.set_status("❌ No audio recorded", THEME["accent_red"])
                return

            # Process the audio
            self._process_audio(audio_file, profile, no_llm)

        except (AudioError, TranscriptionError, EnhancementError, ClipboardError) as e:
            logger.error(f"STT workflow error: {e}")
            print(f"❌ Error: {e}")
            self.ui_manager.set_status(f"❌ {str(e)[:50]}...", THEME["accent_red"])
            raise STTError(f"STT workflow failed: {e}") from e

        except Exception as e:
            logger.error(f"Unexpected error in STT workflow: {e}")
            print(f"❌ Unexpected error: {e}")
            self.ui_manager.set_status("❌ System error", THEME["accent_red"])
            raise STTError(f"Unexpected error: {e}") from e

        finally:
            # Cleanup temporary audio file
            self.audio_recorder.cleanup()

            # Wait a bit for UI to auto-hide if enabled
            if (
                self.ui_manager.enabled
                and self.ui_manager.overlay
                and self.ui_manager.overlay.auto_hide_delay > 0
            ):
                time.sleep(min(self.ui_manager.overlay.auto_hide_delay + 0.5, 5.0))

    def _start_model_loading_monitor(self) -> None:
        """Start background thread to monitor model loading."""

        def check_model_loading():
            while (
                self.ui_manager.overlay
                and self.ui_manager.overlay.is_recording
                and not self._shutdown_requested
            ):
                if self.transcriber._model_loaded.is_set() and self.transcriber.model:
                    self.ui_manager.set_model_ready(True)
                    break
                time.sleep(1.0)  # Check less frequently to reduce interference

        threading.Thread(target=check_model_loading, daemon=True).start()

    def _process_audio(
        self, audio_file: Path, profile: Optional[str], no_llm: bool
    ) -> None:
        """Process recorded audio through transcription and enhancement.

        Args:
            audio_file: Path to recorded audio file
            profile: LLM profile to use
            no_llm: Skip LLM enhancement
        """
        try:
            # Wait for model to finish loading (if it hasn't already)
            print("⏳ Ensuring model is ready...")
            self.ui_manager.set_status(
                "⏳ Waiting for model...", THEME["accent_yellow"]
            )
            timeout = self.config.get("whisper.load_timeout", 60)
            if not self.transcriber.wait_for_model(timeout):
                print("❌ Failed to load Whisper model")
                self.ui_manager.set_status(
                    "❌ Model loading failed", THEME["accent_red"]
                )
                return

            # Update UI - model is ready
            self.ui_manager.set_model_ready(True)

            # Transcribe audio
            logger.info("Transcribing audio")
            print("🔄 Transcribing audio...")
            self.ui_manager.set_status("🔄 Transcribing audio...", THEME["accent_blue"])
            text = self.transcriber.transcribe(audio_file)
            if not text:
                logger.warning("No speech detected in audio")
                print("❌ No speech detected")
                self.ui_manager.set_status("❌ No speech detected", THEME["accent_red"])
                return

            logger.info(f"Transcribed text: {text}")

            # Check if LLM refinement is enabled
            llm_enabled = self.config.get("llm.enabled", True) and not no_llm

            if llm_enabled:
                # Enhance text with LLM
                if profile:
                    logger.info(f"Enhancing text with LLM using profile: {profile}")
                    print(f"🔄 Refining text using '{profile}' profile...")
                    self.ui_manager.set_status(
                        f"🔄 Refining ({profile})...", THEME["accent_purple"]
                    )
                else:
                    logger.info("Enhancing text with LLM using default profile")
                    print("🔄 Refining text...")
                    self.ui_manager.set_status(
                        "🔄 Refining text...", THEME["accent_purple"]
                    )

                enhanced_text = self.enhancer.enhance_text(text, profile)
                if not enhanced_text:
                    enhanced_text = text

                logger.info(f"Enhanced text: {enhanced_text}")
                final_text = enhanced_text
            else:
                # Skip LLM refinement, use raw transcription
                logger.info("LLM refinement disabled, using raw transcription")
                print("⚡ Using raw transcription (LLM disabled)")
                self.ui_manager.set_status("⚡ Raw transcription", THEME["accent_blue"])
                final_text = text

            # Handle clipboard/paste
            if self.config.get("clipboard.auto_paste", False):
                self.clipboard.paste_text(final_text)
                logger.info("Text auto-pasted to active window")
                self.ui_manager.set_status("✅ Auto-pasted!", THEME["accent_green"])
            else:
                self.clipboard.copy_to_clipboard(final_text)
                logger.info("Text copied to clipboard")
                self.ui_manager.set_status(
                    "✅ Copied to clipboard!", THEME["accent_green"]
                )

            print("✅ Processing complete!")
            logger.info("STT process completed successfully")

        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            print(f"❌ Error processing audio: {e}")
            raise

    def list_profiles(self) -> None:
        """List all available LLM profiles."""
        try:
            profiles = self.enhancer.list_profiles()

            print("📝 Available LLM profiles:")
            for profile_id, profile_data in profiles.items():
                name = profile_data.get("name", profile_id)
                print(f"  • {profile_id}: {name}")

            default_profile = self.config.get("llm.default_profile", "general")
            print(f"\n🎯 Default profile: {default_profile}")

        except Exception as e:
            logger.error(f"Failed to list profiles: {e}")
            print(f"❌ Failed to list profiles: {e}")
            raise STTError(f"Failed to list profiles: {e}") from e

    def list_audio_devices(self) -> None:
        """List all available audio input devices."""
        try:
            devices = self.audio_recorder.list_input_devices()
            current_device_index = self.config.get("audio.device_index", None)
            current_device_name = self.audio_recorder.get_selected_device_name()

            print("🎤 Available audio input devices:")
            print()
            
            for device in devices:
                index = device['index']
                name = device['name']
                channels = device['channels']
                sample_rate = int(device['default_sample_rate'])
                
                # Mark current device
                marker = "👈 CURRENT" if index == current_device_index else ""
                if current_device_index is None and index == 0:
                    # If no device is configured, assume index 0 is default
                    try:
                        default_device = self.audio_recorder.audio.get_default_input_device_info()
                        if default_device['index'] == index:
                            marker = "👈 DEFAULT"
                    except:
                        pass
                
                print(f"  [{index:2d}] {name}")
                print(f"       Channels: {channels}, Sample Rate: {sample_rate} Hz {marker}")
                print()

            print(f"🎯 Currently selected: {current_device_name}")
            print()
            print("💡 To change device, add to your config.yaml:")
            print("   audio:")
            print("     device_index: <index_number>")

        except Exception as e:
            logger.error(f"Failed to list audio devices: {e}")
            print(f"❌ Failed to list audio devices: {e}")
            raise STTError(f"Failed to list audio devices: {e}") from e

    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            self.audio_recorder.cleanup()
            self.ui_manager.cleanup()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def __del__(self):
        """Cleanup on destruction."""
        self.cleanup()
