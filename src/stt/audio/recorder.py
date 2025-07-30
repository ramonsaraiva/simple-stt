"""Audio recording with voice activity detection."""

import logging
import os
import tempfile
import time
import wave
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pyaudio
import subprocess
import select
import fcntl

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
        self.frames: List[bytes] = []
        self.stream = None
        self.temp_file: Optional[Path] = None
        self.silence_start_time: Optional[float] = None
        self.current_volume: float = 0.0
        self.volume_callback: Optional[Callable[[float], None]] = None
        self.waveform_callback: Optional[Callable[[List[float]], None]] = None

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
            raise AudioError(
                "Failed to initialize audio system. Check audio drivers."
            ) from e

    def list_input_devices(self) -> List[Dict[str, any]]:
        """List available input audio devices from both PyAudio and PulseAudio.
        
        Returns:
            List of device info dictionaries
        """
        if not self.audio:
            raise AudioError("Audio system not initialized")
            
        devices = []
        device_count = self.audio.get_device_count()
        
        # Add PyAudio devices
        for i in range(device_count):
            device_info = self.audio.get_device_info_by_index(i)
            # Only include input devices
            if device_info['maxInputChannels'] > 0:
                devices.append({
                    'index': i,
                    'name': device_info['name'],
                    'channels': device_info['maxInputChannels'],
                    'default_sample_rate': device_info['defaultSampleRate'],
                    'backend': 'pyaudio'
                })
        
        # Add PulseAudio sources that PyAudio might miss
        pulse_sources = self._get_pulse_input_sources()
        for source in pulse_sources:
            # Check if this source is already in PyAudio devices
            already_exists = any(
                source['name'].replace('alsa_input.', '').replace('usb-', '').replace('-00', '').lower() 
                in device['name'].lower().replace(' ', '').replace(':', '').replace('_', '')
                for device in devices
            )
            
            if not already_exists:
                devices.append({
                    'index': f"pulse:{source['pulse_index']}",
                    'name': f"{source['name']} (PulseAudio)",
                    'channels': source['channels'],
                    'default_sample_rate': source['sample_rate'],
                    'backend': 'pulseaudio'
                })
        
        return devices
    
    def get_selected_device_name(self) -> str:
        """Get the name of the currently selected audio device.
        
        Returns:
            Name of the selected device
        """
        device_index = self.config.get("audio.device_index", None)
        
        if device_index is None:
            # Using default device - try to get default input device info
            try:
                if self.audio:
                    default_device = self.audio.get_default_input_device_info()
                    return default_device['name']
            except Exception:
                pass
            return "Default"
        
        # Get specific device name
        try:
            if self.audio:
                device_info = self.audio.get_device_info_by_index(device_index)
                return device_info['name']
        except Exception:
            pass
        
        return f"Device {device_index}"
    
    def find_preferred_device(self) -> Optional[int]:
        """Find the preferred microphone device based on priority.
        
        Priority order:
        1. Virtual/shared sources (pulse, virtual, shared)
        2. Headset/gaming mics (arctis, corsair, razer, etc.)
        3. USB mics (blue yeti, audio-technica, etc.)
        4. Webcam mics (logitech, etc.)
        
        Returns:
            Device index if found, None otherwise
        """
        if not self.audio:
            return None
            
        try:
            devices = self.list_input_devices()
            
            # Priority lists for device matching
            virtual_keywords = ['pulse', 'shared', 'virtual', 'remap']
            headset_keywords = ['arctis', 'corsair', 'razer', 'hyperx', 'steelseries']
            usb_mic_keywords = ['yeti', 'audio-technica', 'rode', 'samson', 'shure']
            webcam_keywords = ['logitech', 'brio', 'webcam', 'camera']
            
            def device_priority(device_name: str) -> int:
                name_lower = device_name.lower()
                if any(kw in name_lower for kw in virtual_keywords):
                    return 1
                elif any(kw in name_lower for kw in headset_keywords):
                    return 2
                elif any(kw in name_lower for kw in usb_mic_keywords):
                    return 3
                elif any(kw in name_lower for kw in webcam_keywords):
                    return 4
                else:
                    return 5
            
            # Sort devices by priority
            sorted_devices = sorted(devices, key=lambda d: device_priority(d['name']))
            
            if sorted_devices:
                preferred = sorted_devices[0]
                logger.info(f"Auto-selected preferred device: {preferred['name']} (index {preferred['index']})")
                return preferred['index']
                
        except Exception as e:
            logger.warning(f"Could not find preferred device: {e}")
        
        return None
    
    def _get_pulse_input_sources(self) -> List[Dict[str, any]]:
        """Get PulseAudio input sources that might not be visible to PyAudio."""
        try:
            # Run pactl to get source list
            result = subprocess.run(
                ['pactl', 'list', 'sources', 'short'],
                capture_output=True, text=True, check=True
            )
            
            sources = []
            for line in result.stdout.strip().split('\n'):
                if not line.strip():
                    continue
                    
                parts = line.split('\t')
                if len(parts) >= 4:
                    pulse_index = int(parts[0])
                    source_name = parts[1]
                    
                    # Skip monitor sources (they're outputs, not inputs)
                    if '.monitor' in source_name:
                        continue
                        
                    # Parse sample format for channels and sample rate
                    format_str = parts[3] if len(parts) > 3 else "s16le 1ch 48000Hz"
                    channels = 1
                    sample_rate = 48000
                    
                    # Extract channels and sample rate from format string
                    if 'ch' in format_str:
                        try:
                            ch_part = format_str.split('ch')[0].split()[-1]
                            channels = int(ch_part)
                        except:
                            pass
                    
                    if 'Hz' in format_str:
                        try:
                            hz_part = format_str.split('Hz')[0].split()[-1]
                            sample_rate = int(hz_part)
                        except:
                            pass
                    
                    # Create a friendly name for known devices
                    friendly_name = source_name
                    if 'Arctis' in source_name or 'arctis' in source_name:
                        if 'shared' in source_name:
                            friendly_name = "Arctis Nova Shared"
                        else:
                            friendly_name = "Arctis Nova Pro Wireless"
                    elif 'shared' in source_name:
                        friendly_name = f"Shared Audio ({source_name})"
                    
                    sources.append({
                        'pulse_index': pulse_index,
                        'name': friendly_name,
                        'raw_name': source_name,
                        'channels': channels,
                        'sample_rate': sample_rate
                    })
            
            return sources
            
        except Exception as e:
            logger.warning(f"Failed to get PulseAudio sources: {e}")
            return []
    
    def _record_with_pulseaudio(
        self, 
        pulse_device_index: str, 
        voice_detected_callback: Optional[Callable[[], None]],
        sample_rate: int,
        channels: int, 
        chunk_size: int,
        silence_threshold: float,
        voice_threshold: float,
        silence_duration: float,
        max_recording_time: float
    ) -> Optional[Path]:
        """Record audio using PulseAudio (parec) for devices not accessible via PyAudio."""
        
        # Extract pulse source index
        pulse_index = pulse_device_index.replace("pulse:", "")
        
        logger.info(f"Recording with PulseAudio source {pulse_index}")
        logger.info(f"Voice detection thresholds: silence={silence_threshold}, voice={voice_threshold}")
        print("🎤 Waiting for voice... (using PulseAudio backend)")
        
        try:
            # Use parec to record from specific PulseAudio source
            # We'll record continuously and process in chunks
            cmd = [
                'parec',
                '--device', pulse_index,
                '--format', 's16le',
                '--rate', str(sample_rate),
                '--channels', str(channels),
            ]
            
            # Start parec process
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            
            # Make stdout non-blocking to prevent UI freezing
            fd = process.stdout.fileno()
            fl = fcntl.fcntl(fd, fcntl.F_GETFL)
            fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
            
            self.frames = []
            self.silence_start_time = None
            voice_detected = False
            start_time = time.time()
            chunk_count = 0  # For debug logging
            
            # Read chunks from parec
            chunk_bytes = chunk_size * channels * 2  # 2 bytes per sample (s16le)
            
            while True:
                # Check if process is still running
                if process.poll() is not None:
                    logger.error("parec process terminated unexpectedly")
                    break
                
                # Read chunk from process (non-blocking)
                try:
                    # Use select to check if data is available
                    ready, _, _ = select.select([process.stdout], [], [], 0.1)  # 100ms timeout
                    if not ready:
                        continue  # No data available, continue loop
                        
                    data = process.stdout.read(chunk_bytes)
                    if not data:
                        break
                        
                    self.frames.append(data)
                    
                    # Calculate volume for this chunk
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    if len(audio_data) > 0:
                        mean_squared = np.mean(audio_data.astype(np.float64)**2)
                        volume = np.sqrt(mean_squared) if mean_squared >= 0 else 0.0
                        # Ensure volume is not NaN or infinite
                        if np.isnan(volume) or np.isinf(volume):
                            volume = 0.0
                    else:
                        volume = 0.0
                    self.current_volume = volume
                    
                    # Call volume callback if set (throttled to ~10 FPS)
                    current_time = time.time()
                    if self.volume_callback and (not hasattr(self, '_last_volume_update') or current_time - self._last_volume_update > 0.1):
                        self._last_volume_update = current_time
                        self.volume_callback(volume)
                    
                    # Call waveform callback if set (throttled to ~15 FPS and reduced samples)
                    if self.waveform_callback and (not hasattr(self, '_last_waveform_update') or current_time - self._last_waveform_update > 0.067):
                        self._last_waveform_update = current_time
                        normalized_samples = audio_data.astype(np.float32) / 32768.0
                        downsample_factor = max(1, len(normalized_samples) // 100)  # Reduced from 200 to 100 points
                        waveform_data = normalized_samples[::downsample_factor].tolist()
                        self.waveform_callback(waveform_data)

                    # Voice activity detection logic (same as PyAudio version)
                    chunk_count += 1
                    if not voice_detected:
                        # Log volume levels every 10 chunks to see what's happening
                        if chunk_count % 10 == 0:
                            logger.info(f"Waiting for voice: volume={volume:.1f}, threshold={voice_threshold:.1f}")
                        if volume >= voice_threshold:
                            voice_detected = True
                            logger.info(f"Voice activity detected! volume={volume:.1f} >= threshold={voice_threshold}")
                            print("🗣️ Voice detected! Recording... (will stop after silence)")
                            if voice_detected_callback:
                                voice_detected_callback()
                            else:
                                logger.warning("voice_detected_callback is None!")
                    else:
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
                        
                except (BlockingIOError, OSError):
                    # No data available right now, continue
                    continue
                except Exception as e:
                    logger.warning(f"Error reading from parec: {e}")
                    break
            
            # Clean up process
            try:
                process.terminate()
                process.wait(timeout=2)
            except:
                process.kill()
            
            if not self.frames:
                return None

            # Save to temporary file (same format as PyAudio version)
            return self._save_to_temp_file(sample_rate, channels, pyaudio.paInt16)
            
        except Exception as e:
            logger.error(f"Failed to record with PulseAudio: {e}")
            raise AudioError(f"PulseAudio recording failed: {e}") from e
    
    def set_volume_callback(self, callback: Callable[[float], None]) -> None:
        """Set callback to receive volume updates during recording.
        
        Args:
            callback: Function to call with volume level (0.0 to ~1000+)
        """
        self.volume_callback = callback
    
    def set_waveform_callback(self, callback: Callable[[List[float]], None]) -> None:
        """Set callback to receive waveform data for visualization.
        
        Args:
            callback: Function to call with normalized audio samples (-1.0 to 1.0)
        """
        self.waveform_callback = callback

    def record_until_silence(
        self, voice_detected_callback: Optional[Callable[[], None]] = None
    ) -> Optional[Path]:
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
        voice_threshold = max(
            silence_threshold * 2.0, 50
        )  # More conservative for voice detection
        silence_duration = self.config.get("audio.silence_duration", 2.0)
        max_recording_time = self.config.get("audio.max_recording_time", 120.0)
        device_index = self.config.get("audio.device_index", None)
        
        # Convert string device indices to proper types
        if isinstance(device_index, str):
            if device_index.isdigit():
                device_index = int(device_index)
            # else keep as string for PulseAudio devices (pulse:XX)
        
        # Try to use preferred device if no specific device configured
        if device_index is None:
            preferred_device = self.find_preferred_device()
            if preferred_device is not None:
                device_index = preferred_device
                logger.info(f"Using auto-selected device at index {device_index}")

        # For PulseAudio devices, use more sensitive thresholds
        if isinstance(device_index, str) and device_index.startswith("pulse:"):
            logger.info(f"Using PulseAudio device - adjusting thresholds for sensitivity")
            voice_threshold = max(silence_threshold * 1.2, 15)  # Even more sensitive for PulseAudio
            logger.info(f"Adjusted thresholds: silence={silence_threshold}, voice={voice_threshold}")

        try:
            # Check if this is a PulseAudio device
            if isinstance(device_index, str) and device_index.startswith("pulse:"):
                return self._record_with_pulseaudio(
                    device_index, voice_detected_callback, 
                    sample_rate, channels, chunk_size, 
                    silence_threshold, voice_threshold, 
                    silence_duration, max_recording_time
                )
                
            stream_params = {
                "format": audio_format,
                "channels": channels,
                "rate": sample_rate,
                "input": True,
                "frames_per_buffer": chunk_size,
            }
            
            if device_index is not None:
                stream_params["input_device_index"] = device_index
                
            stream = self.audio.open(**stream_params)

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
                self.current_volume = volume
                
                # Call volume callback if set (throttled to ~10 FPS)
                current_time = time.time()
                if self.volume_callback and (not hasattr(self, '_last_volume_update') or current_time - self._last_volume_update > 0.1):
                    self._last_volume_update = current_time
                    self.volume_callback(volume)
                
                # Call waveform callback if set (throttled to ~15 FPS and reduced samples)
                if self.waveform_callback and (not hasattr(self, '_last_waveform_update') or current_time - self._last_waveform_update > 0.067):
                    self._last_waveform_update = current_time
                    # Normalize int16 data to float range -1.0 to 1.0
                    normalized_samples = audio_data.astype(np.float32) / 32768.0
                    # Downsample for visualization (take every Nth sample for performance)
                    downsample_factor = max(1, len(normalized_samples) // 100)  # Reduced from 200 to 100 points
                    waveform_data = normalized_samples[::downsample_factor].tolist()
                    self.waveform_callback(waveform_data)

                if not voice_detected:
                    # Phase 1: Waiting for voice activity
                    if volume >= voice_threshold:
                        voice_detected = True
                        logger.info(
                            "Voice activity detected, starting silence detection"
                        )
                        print(
                            "🗣️ Voice detected! Recording... (will stop after silence)"
                        )

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
        device_index = self.config.get("audio.device_index", None)

        try:
            stream_params = {
                "format": audio_format,
                "channels": channels,
                "rate": sample_rate,
                "input": True,
                "frames_per_buffer": chunk_size,
            }
            
            if device_index is not None:
                stream_params["input_device_index"] = device_index
                
            stream = self.audio.open(**stream_params)

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
                        print(
                            f"🗣️  Now SPEAK CLEARLY for {speech_time:.0f} seconds... (say anything, read text, etc.)"
                        )
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

    def _calculate_optimal_threshold(
        self, silence_volumes: List[float], speech_volumes: List[float]
    ) -> float:
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
        optimal_threshold = max(
            optimal_threshold, avg_silence * 1.5
        )  # At least 1.5x avg silence
        optimal_threshold = min(
            optimal_threshold, avg_speech * 0.3
        )  # At most 30% of avg speech

        print(f"🎯 Recommended threshold: {optimal_threshold:.1f}")
        return optimal_threshold

    def _get_audio_format(self) -> int:
        """Get PyAudio format constant from config."""
        format_str = self.config.get("audio.format", "int16")
        format_bits = format_str.replace("int", "")
        return getattr(pyaudio, f"paInt{format_bits}")

    def _save_to_temp_file(
        self, sample_rate: int, channels: int, audio_format: int
    ) -> Path:
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
