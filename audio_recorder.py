import pyaudio
import wave
import threading
import tempfile
import os
import sys
import time
import numpy as np
import logging
from typing import Optional

# Suppress ALSA warnings
import os
os.environ['ALSA_PCM_CARD'] = '0'
os.environ['ALSA_PCM_DEVICE'] = '0'

logger = logging.getLogger(__name__)

class AudioRecorder:
    def __init__(self, config):
        self.config = config
        try:
            self.audio = pyaudio.PyAudio()
            logger.debug("Audio system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize audio system: {e}")
            logger.error("Make sure your audio drivers are properly installed")
            sys.exit(1)
        
        self.is_recording = False
        self.frames = []
        self.stream = None
        self.temp_file = None
        self.silence_start_time = None
        
    def record_until_silence(self) -> Optional[str]:
        """Record audio until silence is detected"""
        self.frames = []
        self.silence_start_time = None
        
        try:
            stream = self.audio.open(
                format=getattr(pyaudio, f"paInt{self.config.get('audio.format', 'int16')[3:]}"),
                channels=self.config.get('audio.channels', 1),
                rate=self.config.get('audio.sample_rate', 16000),
                input=True,
                frames_per_buffer=self.config.get('audio.chunk_size', 1024)
            )
            
            logger.info("Recording started, will stop after silence")
            print("🎤 Recording started... (speak now, will stop after silence)")
            
            silence_threshold = self.config.get('audio.silence_threshold', 500)
            silence_duration = self.config.get('audio.silence_duration', 2.0)
            max_recording_time = self.config.get('audio.max_recording_time', 30.0)
            
            start_time = time.time()
            
            while True:
                data = stream.read(self.config.get('audio.chunk_size', 1024))
                self.frames.append(data)
                
                # Check for silence
                audio_data = np.frombuffer(data, dtype=np.int16)
                volume = np.sqrt(np.mean(audio_data**2))
                
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
            
            # Create temporary file
            temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
            os.close(temp_fd)
            
            # Save audio to temporary file
            with wave.open(temp_path, 'wb') as wf:
                wf.setnchannels(self.config.get('audio.channels', 1))
                wf.setsampwidth(self.audio.get_sample_size(
                    getattr(pyaudio, f"paInt{self.config.get('audio.format', 'int16')[3:]}")
                ))
                wf.setframerate(self.config.get('audio.sample_rate', 16000))
                wf.writeframes(b''.join(self.frames))
            
            self.temp_file = temp_path
            return temp_path
            
        except Exception as e:
            logger.error(f"Failed to record audio: {e}")
            return None
    
    def tune_silence_threshold(self, duration=12) -> float:
        """Tune the silence threshold by analyzing speech vs silence"""
        print("🎯 Starting silence threshold tuning...")
        print(f"This will record for {duration} seconds. Follow the prompts below:")
        print()
        
        try:
            stream = self.audio.open(
                format=getattr(pyaudio, f"paInt{self.config.get('audio.format', 'int16')[3:]}"),
                channels=self.config.get('audio.channels', 1),
                rate=self.config.get('audio.sample_rate', 16000),
                input=True,
                frames_per_buffer=self.config.get('audio.chunk_size', 1024)
            )
            
            volumes = []
            speech_volumes = []
            silence_volumes = []
            
            start_time = time.time()
            silence_time = 3  # 3 seconds of silence
            speech_time = duration - silence_time  # 9 seconds of speech
            
            print("🔇 First, stay SILENT for 3 seconds...")
            
            while time.time() - start_time < duration:
                data = stream.read(self.config.get('audio.chunk_size', 1024))
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
                    # Second part - speech (much longer)
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
            
            # Calculate statistics
            avg_silence = np.mean(silence_volumes)
            max_silence = np.max(silence_volumes)
            p95_silence = np.percentile(silence_volumes, 95)  # 95th percentile of silence
            avg_speech = np.mean(speech_volumes)
            min_speech = np.min(speech_volumes)
            p10_speech = np.percentile(speech_volumes, 10)  # 10th percentile of speech
            
            print(f"\n📊 Analysis Results:")
            print(f"   Average silence volume: {avg_silence:.1f}")
            print(f"   Maximum silence volume: {max_silence:.1f}")
            print(f"   95th percentile silence: {p95_silence:.1f}")
            print(f"   Average speech volume: {avg_speech:.1f}")
            print(f"   Minimum speech volume: {min_speech:.1f}")
            print(f"   10th percentile speech: {p10_speech:.1f}")
            
            # More conservative threshold calculation
            # Use 95th percentile of silence + 20% margin, or midpoint if there's clear separation
            if p95_silence < p10_speech:
                # Clear separation - use midpoint but with bias toward avoiding false silence detection
                raw_midpoint = (p95_silence + p10_speech) / 2
                optimal_threshold = raw_midpoint * 0.8 + p95_silence * 0.2  # Bias toward silence side
            else:
                # Overlap detected - use 95th percentile of silence + small margin
                optimal_threshold = p95_silence * 1.2
            
            # Safety bounds
            optimal_threshold = max(optimal_threshold, avg_silence * 1.5)  # At least 1.5x avg silence
            optimal_threshold = min(optimal_threshold, avg_speech * 0.3)   # At most 30% of avg speech
            
            print(f"🎯 Recommended threshold: {optimal_threshold:.1f}")
            return optimal_threshold
            
        except Exception as e:
            print(f"❌ Failed to tune threshold: {e}")
            return None
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        if self.is_recording:
            self.frames.append(in_data)
        return (in_data, pyaudio.paContinue)
    
    def stop_recording(self) -> Optional[str]:
        if not self.is_recording:
            return None
            
        self.is_recording = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        if not self.frames:
            return None
            
        # Create temporary file
        temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
        os.close(temp_fd)
        
        # Save audio to temporary file
        with wave.open(temp_path, 'wb') as wf:
            wf.setnchannels(self.config.get('audio.channels', 1))
            wf.setsampwidth(self.audio.get_sample_size(
                getattr(pyaudio, f"paInt{self.config.get('audio.format', 'int16')[3:]}")
            ))
            wf.setframerate(self.config.get('audio.sample_rate', 16000))
            wf.writeframes(b''.join(self.frames))
        
        self.temp_file = temp_path
        return temp_path
    
    def cleanup(self):
        if self.temp_file and os.path.exists(self.temp_file):
            os.unlink(self.temp_file)
            self.temp_file = None
    
    def __del__(self):
        self.cleanup()
        if hasattr(self, 'audio'):
            self.audio.terminate()