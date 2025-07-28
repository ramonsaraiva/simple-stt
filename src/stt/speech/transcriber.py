"""Speech transcription using Whisper."""

import logging
import threading
from pathlib import Path
from typing import Optional

import torch
import whisper

from ..config.manager import ConfigManager

logger = logging.getLogger(__name__)


class TranscriptionError(Exception):
    """Raised when transcription fails."""


class SpeechTranscriber:
    """Handles speech-to-text transcription using OpenAI Whisper."""

    def __init__(self, config: ConfigManager):
        """Initialize speech transcriber.

        Args:
            config: Configuration manager instance
        """
        self.config = config
        self.model: Optional[whisper.Whisper] = None
        self.device: Optional[str] = None
        self._model_loaded = threading.Event()
        self._load_thread: Optional[threading.Thread] = None

    def start_loading_async(self) -> None:
        """Start loading the Whisper model in a background thread."""
        if self._load_thread is None or not self._load_thread.is_alive():
            logger.info("Starting asynchronous model loading...")
            self._model_loaded.clear()
            self._load_thread = threading.Thread(target=self._load_model, daemon=True)
            self._load_thread.start()

    def wait_for_model(self, timeout: int = 30) -> bool:
        """Wait for the model to finish loading.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if model loaded successfully, False otherwise
        """
        logger.info("Waiting for model to finish loading...")
        if self._model_loaded.wait(timeout):
            if self.model is not None:
                logger.info("Model loaded successfully!")
                return True
            else:
                logger.error("Model loading failed")
                return False
        else:
            logger.error(f"Model loading timed out after {timeout} seconds")
            return False

    def transcribe(self, audio_file_path: Path) -> Optional[str]:
        """Transcribe audio file to text.

        Args:
            audio_file_path: Path to audio file

        Returns:
            Transcribed text, or None if transcription failed
        """
        if not self.model:
            raise TranscriptionError("Whisper model not loaded")

        if not audio_file_path.exists():
            raise TranscriptionError(f"Audio file not found: {audio_file_path}")

        try:
            language = self.config.get("whisper.language", "auto")
            if language == "auto":
                language = None

            logger.debug(f"Starting transcription of {audio_file_path}")

            # Configure transcription options
            transcribe_options = {
                "language": language,
                "fp16": self.device == "cuda",  # Use fp16 on GPU for better performance
            }

            # Add compute_type if specified in config
            compute_type = self.config.get("whisper.compute_type")
            if compute_type:
                transcribe_options["fp16"] = compute_type == "float16"

            result = self.model.transcribe(str(audio_file_path), **transcribe_options)

            text = result["text"].strip()
            if text:
                logger.info(f"Transcription successful: {text}")
                return text
            else:
                logger.warning("No speech detected in audio")
                return None

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise TranscriptionError(f"Transcription failed: {e}") from e

    def _load_model(self) -> None:
        """Load the Whisper model (runs in background thread)."""
        model_name = self.config.get("whisper.model", "base")
        device = self.config.get("whisper.device", "auto")

        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                logger.info("CUDA is available, using GPU")
            else:
                device = "cpu"
                logger.info("CUDA not available, using CPU")
        elif device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = "cpu"

        try:
            self.model = whisper.load_model(model_name, device=device)
            self.device = device
            logger.info(f"Loaded Whisper model: {model_name} on device: {device}")

            # Log GPU info if using CUDA
            if device == "cuda":
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info(f"Using GPU: {gpu_name} ({gpu_memory:.1f}GB)")

        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            self.model = None
            self.device = None
        finally:
            # Signal that loading is complete (success or failure)
            self._model_loaded.set()
