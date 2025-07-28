"""Audio recording with voice activity detection."""

import logging
import os
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pyaudio

from ..config.manager import ConfigManager

logger = logging.getLogger(__name__)


class AudioError(Exception):
    """Raised when audio operations fail."""


class AudioRecorder:
    """Records audio with voice activity detection and silence timeout."""
    
    def __init__(self, config: ConfigManager):
        """Initialize audio recorder.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config
        self.audio: Optional[pyaudio.PyAudio] = None
        self.is_recording = False
        self.frames: list[bytes] = []
        self.stream = None
        self.temp_file: Optional[Path] = None
        self.silence_start_time: Optional[float] = None
        
        self._initialize_audio()
    
    def _initialize_audio(self) -> None:
        """Initialize PyAudio system."""
        # Suppress ALSA warnings
        os.environ.setdefault("ALSA_PCM_CARD", "0")
        os.environ.setdefault("ALSA_PCM_DEVICE", "0")
        
        try:
            self.audio = pyaudio.PyAudio()
            logger.debug("Audio system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize audio system: {e}")
            raise AudioError("Failed to initialize audio system. Check audio drivers.") from e
    
    def record_until_silence(self, voice_detected_callback: Optional[Callable[[], None]] = None) -> Optional[Path]:
        """Record audio until silence is detected.
        
        Args:
            voice_detected_callback: Called when voice activity is first detected
            
        Returns:
            Path to recorded audio file, or None if no audio recorded
        """
        if not self.audio:
            raise AudioError("Audio system not initialized")
        
        self.frames = []
        self.silence_start_time = None
        
        # Get configuration
        sample_rate = self.config.get("audio.sample_rate", 16000)
        channels = self.config.get("audio.channels", 1)
        chunk_size = self.config.get("audio.chunk_size", 1024)
        audio_format = self._get_audio_format()
        
        silence_threshold = self.config.get("audio.silence_threshold", 500)
        voice_threshold = max(silence_threshold * 2.0, 50)  # More conservative for voice detection
        silence_duration = self.config.get("audio.silence_duration", 2.0)
        max_recording_time = self.config.get("audio.max_recording_time", 120.0)
        
        try:
            stream = self.audio.open(
                format=audio_format,
                channels=channels,
                rate=sample_rate,
                input=True,
                frames_per_buffer=chunk_size,
            )
            
            logger.info("Recording started, waiting for voice activity")
            print("🎤 Waiting for voice... (start speaking to begin recording)")
            
            start_time = time.time()
            voice_detected = False
            
            while True:
                try:
                    data = stream.read(chunk_size, exception_on_overflow=False)
                    self.frames.append(data)
                except Exception as e:
                    logger.warning(f"Audio read warning: {e}")
                    continue
                
                # Calculate volume for this chunk
                audio_data = np.frombuffer(data, dtype=np.int16)
                volume = np.sqrt(np.mean(audio_data**2))
                
                if not voice_detected:
                    # Phase 1: Waiting for voice activity
                    if volume >= voice_threshold:
                        voice_detected = True
                        logger.info("Voice activity detected, starting silence detection")
                        print("🗣️ Voice detected! Recording... (will stop after silence)")
                        
                        # Notify UI that voice was detected
                        if voice_detected_callback:
                            voice_detected_callback()
                else:
                    # Phase 2: Recording with silence detection
                    if volume < silence_threshold:
                        if self.silence_start_time is None:
                            self.silence_start_time = time.time()
                        elif time.time() - self.silence_start_time > silence_duration:
                            logger.info("Silence detected, stopping recording")
                            print("🔇 Silence detected, stopping recording...")
                            break
                    else:
                        self.silence_start_time = None
                
                # Safety timeout
                if time.time() - start_time > max_recording_time:
                    logger.warning("Maximum recording time reached, stopping")
                    print("⏰ Max recording time reached, stopping...")
                    break
            
            stream.stop_stream()
            stream.close()
            
            if not self.frames:
                return None
            
            # Save to temporary file
            return self._save_to_temp_file(sample_rate, channels, audio_format)
            
        except Exception as e:
            logger.error(f"Failed to record audio: {e}")
            raise AudioError(f"Recording failed: {e}") from e
    
    def tune_silence_threshold(self, duration: int = 12) -> Optional[float]:
        """Tune the silence threshold by analyzing speech vs silence.
        
        Args:
            duration: Total recording duration in seconds
            
        Returns:
            Optimal threshold value, or None if tuning failed
        """
        if not self.audio:
            raise AudioError("Audio system not initialized")
        
        print("🎯 Starting silence threshold tuning...")
        print(f"This will record for {duration} seconds. Follow the prompts below:")
        print()
        
        # Get configuration
        sample_rate = self.config.get("audio.sample_rate", 16000)
        channels = self.config.get("audio.channels", 1)
        chunk_size = self.config.get("audio.chunk_size", 1024)
        audio_format = self._get_audio_format()
        
        try:
            stream = self.audio.open(
                format=audio_format,
                channels=channels,
                rate=sample_rate,
                input=True,
                frames_per_buffer=chunk_size,
            )
            
            volumes = []
            speech_volumes = []
            silence_volumes = []
            
            start_time = time.time()
            silence_time = 3  # 3 seconds of silence
            speech_time = duration - silence_time  # Remaining time for speech
            
            print("🔇 First, stay SILENT for 3 seconds...")
            
            while time.time() - start_time < duration:
                data = stream.read(chunk_size)
                audio_data = np.frombuffer(data, dtype=np.int16)
                volume = np.sqrt(np.mean(audio_data**2))
                volumes.append(volume)
                
                elapsed = time.time() - start_time
                
                if elapsed < silence_time:
                    # First part - silence
                    silence_volumes.append(volume)
                    if len(silence_volumes) % 15 == 0:  # Every ~0.375 seconds
                        remaining = silence_time - elapsed
                        print(f"🔇 Stay silent... {remaining:.1f}s remaining")
                else:
                    # Second part - speech
                    if len(speech_volumes) == 0:
                        print(f"🗣️  Now SPEAK CLEARLY for {speech_time:.0f} seconds... (say anything, read text, etc.)")
                    speech_volumes.append(volume)
                    if len(speech_volumes) % 30 == 0:  # Every ~0.75 seconds
                        remaining = duration - elapsed
                        print(f"🗣️  Keep talking... {remaining:.1f}s remaining")
            
            stream.stop_stream()
            stream.close()
            
            if not silence_volumes or not speech_volumes:
                print("❌ Not enough data collected")
                return None
            
            # Calculate optimal threshold
            return self._calculate_optimal_threshold(silence_volumes, speech_volumes)
            
        except Exception as e:
            logger.error(f"Failed to tune threshold: {e}")
            print(f"❌ Failed to tune threshold: {e}")
            return None
    
    def _calculate_optimal_threshold(self, silence_volumes: list[float], speech_volumes: list[float]) -> float:
        """Calculate optimal threshold from recorded volumes."""
        avg_silence = np.mean(silence_volumes)
        max_silence = np.max(silence_volumes)
        p95_silence = np.percentile(silence_volumes, 95)
        avg_speech = np.mean(speech_volumes)
        min_speech = np.min(speech_volumes)
        p10_speech = np.percentile(speech_volumes, 10)
        
        print("\n📊 Analysis Results:")
        print(f"   Average silence volume: {avg_silence:.1f}")
        print(f"   Maximum silence volume: {max_silence:.1f}")
        print(f"   95th percentile silence: {p95_silence:.1f}")
        print(f"   Average speech volume: {avg_speech:.1f}")
        print(f"   Minimum speech volume: {min_speech:.1f}")
        print(f"   10th percentile speech: {p10_speech:.1f}")
        
        # Conservative threshold calculation
        if p95_silence < p10_speech:
            # Clear separation - use midpoint with bias toward silence side
            raw_midpoint = (p95_silence + p10_speech) / 2
            optimal_threshold = raw_midpoint * 0.8 + p95_silence * 0.2
        else:
            # Overlap detected - use 95th percentile of silence + small margin
            optimal_threshold = p95_silence * 1.2
        
        # Safety bounds
        optimal_threshold = max(optimal_threshold, avg_silence * 1.5)  # At least 1.5x avg silence
        optimal_threshold = min(optimal_threshold, avg_speech * 0.3)  # At most 30% of avg speech
        
        print(f"🎯 Recommended threshold: {optimal_threshold:.1f}")
        return optimal_threshold
    
    def _get_audio_format(self) -> int:
        """Get PyAudio format constant from config."""
        format_str = self.config.get("audio.format", "int16")
        format_bits = format_str.replace("int", "")
        return getattr(pyaudio, f"paInt{format_bits}")
    
    def _save_to_temp_file(self, sample_rate: int, channels: int, audio_format: int) -> Path:
        """Save recorded frames to temporary WAV file."""
        temp_fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(temp_fd)
        temp_path_obj = Path(temp_path)
        
        with wave.open(str(temp_path_obj), "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(self.audio.get_sample_size(audio_format))
            wf.setframerate(sample_rate)
            wf.writeframes(b"".join(self.frames))
        
        self.temp_file = temp_path_obj
        return temp_path_obj
    
    def cleanup(self) -> None:
        """Clean up temporary files and resources."""
        if self.temp_file and self.temp_file.exists():
            self.temp_file.unlink()
            self.temp_file = None
        
        if self.audio:
            self.audio.terminate()
            self.audio = None
    
    def __del__(self):
        """Cleanup on destruction."""
        self.cleanup()