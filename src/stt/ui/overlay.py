"""UI overlay for STT system status."""

import logging
import os
import threading
import time
from collections import deque
from datetime import datetime
from typing import Deque, Dict, List, Optional

from ..config.manager import ConfigManager

logger = logging.getLogger(__name__)

# Try to import tkinter, fall back to no-op mode if not available
try:
    import tkinter as tk

    TKINTER_AVAILABLE = True
except ImportError:
    logger.warning("tkinter not available - UI overlay will be disabled")
    TKINTER_AVAILABLE = False
    tk = None


# Tokyo Night color palette
THEME = {
    "bg_primary": "#1a1b26",  # Background
    "text_primary": "#c0caf5",  # Primary text (light blue)
    "text_secondary": "#565f89",  # Secondary text (muted blue-gray)
    "accent_blue": "#7aa2f7",  # Blue accent
    "accent_green": "#9ece6a",  # Green accent
    "accent_yellow": "#e0af68",  # Yellow accent
    "accent_red": "#f7768e",  # Red accent
    "accent_purple": "#bb9af7",  # Purple accent
}


class UIError(Exception):
    """Raised when UI operations fail."""


class STTOverlay:
    """Tkinter overlay showing STT system status."""

    def __init__(self, config: ConfigManager):
        """Initialize STT overlay.

        Args:
            config: Configuration manager instance
        """
        self.config = config
        self.root: Optional[tk.Tk] = None
        self.labels: Dict[str, tk.Label] = {}
        self.start_time: Optional[datetime] = None
        self.is_recording = False
        self.is_waiting_for_voice = False
        self.model_ready = False
        self.current_profile = "general"
        self.current_volume = 0.0
        self.device_name = "Default"
        self.waveform_buffer: Deque[float] = deque(maxlen=300)  # Store last 300 samples
        self.canvas = None
        self._last_waveform_update = 0.0  # Throttle waveform updates
        self._mainloop_running = False

        # UI update thread control
        self._ui_thread: Optional[threading.Thread] = None
        self._stop_ui = threading.Event()

        # Configuration
        self.enabled = config.get("ui.enabled", True) and TKINTER_AVAILABLE
        if not TKINTER_AVAILABLE and config.get("ui.enabled", True):
            logger.info("UI overlay disabled - tkinter not available")

        self.position_x = config.get("ui.position_x", 50)
        self.position_y = config.get("ui.position_y", 50)
        self.auto_hide_delay = config.get("ui.auto_hide_delay", 3.0)

    def create_overlay(self) -> None:
        """Create the overlay window."""
        if not self.enabled:
            return

        try:
            self.root = tk.Tk()
            self.root.title("STT Status")

            # Configure window
            self.root.attributes("-topmost", True)  # Always on top
            self.root.attributes("-alpha", 0.85)  # More transparency
            self.root.resizable(False, False)

            # Set theme
            self.root.configure(bg=THEME["bg_primary"])
            self._create_widgets()

            # Set minimum window size (increased for larger waveform)
            self.root.minsize(320, 200)

            # Update geometry to calculate size
            self.root.update_idletasks()

            # Calculate center position with dynamic sizing
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            window_width = max(self.root.winfo_reqwidth(), 320)
            window_height = max(self.root.winfo_reqheight(), 200)

            center_x = (screen_width - window_width) // 2
            center_y = (screen_height - window_height) // 2

            # Position window at center
            self.root.geometry(f"{window_width}x{window_height}+{center_x}+{center_y}")

            # Bind events
            self.root.bind("<Configure>", self._on_window_configure)
            self.root.protocol("WM_DELETE_WINDOW", self._on_window_close)

            self._update_display()
            logger.info("STT overlay created and centered")

        except Exception as e:
            logger.error(f"Failed to create overlay: {e}")
            self.enabled = False
            raise UIError(f"Failed to create overlay: {e}") from e

    def _create_widgets(self) -> None:
        """Create the UI widgets."""
        # Main frame
        main_frame = tk.Frame(self.root, bg=THEME["bg_primary"], padx=12, pady=10)
        main_frame.pack(fill="both", expand=True)

        # Status line - with text wrapping
        self.labels["status"] = tk.Label(
            main_frame,
            text="⏹️ Idle",
            bg=THEME["bg_primary"],
            fg=THEME["text_primary"],
            font=("Arial", 12, "bold"),
            wraplength=400,  # Wrap text at 400 pixels
            justify="center",
        )
        self.labels["status"].pack(fill="x", pady=(0, 2))

        # Timer line - centered
        self.labels["timer"] = tk.Label(
            main_frame,
            text="00:00",
            bg=THEME["bg_primary"],
            fg=THEME["text_secondary"],
            font=("Monaco", 10),
        )
        self.labels["timer"].pack(pady=(0, 8))

        # Profile and model status line - combined frame
        profile_model_frame = tk.Frame(main_frame, bg=THEME["bg_primary"])
        profile_model_frame.pack(fill="x", pady=(0, 8))

        # Profile line - left side
        self.labels["profile"] = tk.Label(
            profile_model_frame,
            text="📝 general",
            bg=THEME["bg_primary"],
            fg=THEME["accent_blue"],
            font=("Arial", 9),
        )
        self.labels["profile"].pack(side="left")

        # Model status line - right side
        self.labels["model"] = tk.Label(
            profile_model_frame,
            text="⏳ Loading model...",
            bg=THEME["bg_primary"],
            fg=THEME["accent_yellow"],
            font=("Arial", 9),
            wraplength=200,
            justify="right",
        )
        self.labels["model"].pack(side="right")

        # Waveform canvas (centered) - increased height for better visibility
        self.canvas = tk.Canvas(
            main_frame,
            width=300,
            height=120,
            bg=THEME["bg_primary"],
            highlightthickness=0,
            bd=0,
        )
        self.canvas.pack(pady=(5, 8), fill="x")

        # Level and device line - combined frame below waveform
        level_device_frame = tk.Frame(main_frame, bg=THEME["bg_primary"])
        level_device_frame.pack(fill="x", pady=(0, 2))

        # Volume line - left side
        self.labels["volume"] = tk.Label(
            level_device_frame,
            text="🔊 Level: 0",
            bg=THEME["bg_primary"],
            fg=THEME["text_secondary"],
            font=("Arial", 8),
        )
        self.labels["volume"].pack(side="left")

        # Device line - right side
        self.labels["device"] = tk.Label(
            level_device_frame,
            text="🎧 Default",
            bg=THEME["bg_primary"],
            fg=THEME["text_secondary"],
            font=("Arial", 8),
            wraplength=200,
            justify="right",
        )
        self.labels["device"].pack(side="right")
        
        # Initialize empty waveform
        self._draw_waveform()

    def start_recording(self, profile: str = "general") -> None:
        """Signal that recording has started."""
        if not self.enabled:
            return

        self.is_recording = True
        self.is_waiting_for_voice = True  # Start in waiting for voice mode
        self.start_time = datetime.now()
        self.current_profile = profile

        # Create overlay if it doesn't exist
        if self.root is None:
            self.create_overlay()

        # Start UI update thread
        if self._ui_thread is None or not self._ui_thread.is_alive():
            self._stop_ui.clear()
            self._ui_thread = threading.Thread(target=self._ui_update_loop, daemon=True)
            self._ui_thread.start()

        # Update display
        self._schedule_update()
        logger.info("Recording started - UI updated")

    def voice_detected(self) -> None:
        """Signal that voice activity has been detected."""
        if not self.enabled:
            return

        self.is_waiting_for_voice = False
        self._schedule_update()
        logger.debug("Voice detected - UI updated")

    def stop_recording(self) -> None:
        """Signal that recording has stopped."""
        if not self.enabled:
            return

        self.is_recording = False
        self.is_waiting_for_voice = False
        
        # Clear waveform buffer when stopping
        self.waveform_buffer.clear()
        
        self._schedule_update()
        self._schedule_waveform_update()

        # Auto-hide after delay
        if self.auto_hide_delay > 0:
            threading.Timer(self.auto_hide_delay, self.hide).start()

        logger.info("Recording stopped - UI updated")

    def set_model_ready(self, ready: bool = True) -> None:
        """Update model loading status."""
        if not self.enabled:
            return

        self.model_ready = ready
        self._schedule_update()
        logger.debug(f"Model status updated: {'ready' if ready else 'loading'}")

    def set_device_name(self, device_name: str) -> None:
        """Update the device name display."""
        if not self.enabled:
            return

        self.device_name = device_name
        self._schedule_update()

    def set_volume_level(self, volume: float) -> None:
        """Update the volume level display."""
        if not self.enabled:
            return

        self.current_volume = volume
        self._schedule_update()

    def update_waveform(self, samples: List[float]) -> None:
        """Update the waveform display with new audio samples."""
        if not self.enabled or not self.canvas:
            return

        # Add new samples to buffer
        self.waveform_buffer.extend(samples)
        
        # Schedule waveform redraw
        self._schedule_waveform_update()

    def _schedule_waveform_update(self) -> None:
        """Schedule a waveform update on the main thread (throttled)."""
        if self.root is not None:
            try:
                # Throttle updates to ~30 FPS for performance
                current_time = time.time()
                if current_time - self._last_waveform_update > 0.033:  # ~30 FPS
                    self._last_waveform_update = current_time
                    self.root.after(0, self._draw_waveform)
            except Exception as e:
                logger.error(f"Failed to schedule waveform update: {e}")

    def _draw_waveform(self) -> None:
        """Draw the waveform on the canvas."""
        if not self.enabled or not self.canvas:
            return

        try:
            # Clear canvas
            self.canvas.delete("all")
            
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            # Handle case when canvas isn't properly initialized yet
            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width = 300
                canvas_height = 120
            
            # Draw center line
            center_y = canvas_height // 2
            self.canvas.create_line(
                0, center_y, canvas_width, center_y,
                fill=THEME["text_secondary"], width=1
            )
            
            # Draw waveform if we have samples
            if len(self.waveform_buffer) < 2:
                return
                
            samples = list(self.waveform_buffer)
            
            # Calculate x step size
            x_step = canvas_width / len(samples) if samples else 1
            
            # Draw waveform lines with significantly increased amplitude
            points = []
            amplitude_scale = 4.5  # Much higher amplitude for better visibility
            for i, sample in enumerate(samples):
                x = i * x_step
                # Clamp sample to prevent extreme values
                clamped_sample = max(-1.0, min(1.0, sample))
                # Apply amplitude scaling and use almost full canvas height
                y = center_y - (clamped_sample * amplitude_scale * (canvas_height // 2 - 5))
                points.extend([x, y])
            
            # Draw the waveform
            if len(points) >= 4:  # Need at least 2 points (4 coordinates)
                # Choose color based on recording state
                if self.is_recording and not self.is_waiting_for_voice:
                    waveform_color = THEME["accent_green"]
                elif self.is_recording and self.is_waiting_for_voice:
                    waveform_color = THEME["accent_yellow"]
                else:
                    waveform_color = THEME["accent_blue"]
                
                self.canvas.create_line(
                    points,
                    fill=waveform_color,
                    width=4,  # Further increased line width for better visibility
                    smooth=True,
                    capstyle="round",
                    joinstyle="round"
                )
                
        except Exception as e:
            logger.error(f"Failed to draw waveform: {e}")

    def set_status(self, status_text: str, color: str = THEME["text_primary"]) -> None:
        """Set custom status text."""
        if not self.enabled or self.root is None:
            return

        try:

            def update_status():
                self.labels["status"].config(text=status_text, fg=color)
                # Force window to recalculate size after text change
                self.root.update_idletasks()

            self.root.after(0, update_status)
        except Exception as e:
            logger.error(f"Failed to update status: {e}")

    def _schedule_update(self) -> None:
        """Schedule a UI update on the main thread."""
        if self.root is not None:
            try:
                self.root.after(0, self._update_display)
            except Exception as e:
                logger.error(f"Failed to schedule UI update: {e}")

    def _update_display(self) -> None:
        """Update the display (must run on main thread)."""
        if not self.enabled or self.root is None:
            return

        try:
            # Update status
            if self.is_recording:
                if self.is_waiting_for_voice:
                    self.labels["status"].config(
                        text="⏳ Waiting for voice...", fg=THEME["accent_yellow"]
                    )
                else:
                    self.labels["status"].config(
                        text="🎤 Recording", fg=THEME["accent_red"]
                    )
            else:
                self.labels["status"].config(
                    text="⏹️ Processing", fg=THEME["accent_yellow"]
                )

            # Update timer
            if self.start_time and self.is_recording:
                elapsed = datetime.now() - self.start_time
                minutes = int(elapsed.total_seconds() // 60)
                seconds = int(elapsed.total_seconds() % 60)
                timer_text = f"{minutes:02d}:{seconds:02d}"
                self.labels["timer"].config(text=timer_text, fg=THEME["text_primary"])
            else:
                self.labels["timer"].config(text="--:--", fg=THEME["text_secondary"])

            # Update model status
            if self.model_ready:
                self.labels["model"].config(
                    text="✅ Model ready", fg=THEME["accent_green"]
                )
            else:
                self.labels["model"].config(
                    text="⏳ Loading model...", fg=THEME["accent_yellow"]
                )

            # Update profile
            self.labels["profile"].config(
                text=f"📝 {self.current_profile}", fg=THEME["accent_blue"]
            )

            # Update device
            device_text = self.device_name
            if len(device_text) > 35:  # Truncate long device names
                device_text = device_text[:32] + "..."
            self.labels["device"].config(
                text=f"🎧 {device_text}", fg=THEME["text_secondary"]
            )

            # Update volume
            volume_level = int(self.current_volume)
            volume_color = THEME["text_secondary"]
            if self.is_recording and not self.is_waiting_for_voice:
                # Show volume level with color coding during active recording
                if volume_level > 100:
                    volume_color = THEME["accent_green"]
                elif volume_level > 50:
                    volume_color = THEME["accent_yellow"]
                else:
                    volume_color = THEME["accent_red"]
            
            self.labels["volume"].config(
                text=f"🔊 Level: {volume_level}", fg=volume_color
            )

        except Exception as e:
            logger.error(f"Failed to update display: {e}")

    def _on_window_configure(self, event) -> None:
        """Handle window resize events."""
        # Only handle resize events for the main window, not child widgets
        if event.widget == self.root:
            # Update window position to stay centered if it grew
            pass  # For now, just let it resize naturally

    def _on_window_close(self) -> None:
        """Handle window close event by terminating the entire process."""
        logger.info("UI window closed, terminating process immediately")
        print("\n🛑 Window closed, terminating process...")
        os._exit(0)  # Force immediate termination

    def _ui_update_loop(self) -> None:
        """Background thread to update timer while recording."""
        while not self._stop_ui.is_set() and self.is_recording:
            self._schedule_update()
            time.sleep(1.0)  # Update every 1000ms to reduce interference

    def show(self) -> None:
        """Show the overlay."""
        if not self.enabled:
            return

        if self.root is None:
            self.create_overlay()

        try:
            self.root.deiconify()
            self.root.lift()
            self.root.attributes("-topmost", True)
        except Exception as e:
            logger.error(f"Failed to show overlay: {e}")

    def hide(self) -> None:
        """Hide the overlay."""
        if not self.enabled or self.root is None:
            return

        try:
            self._stop_ui.set()  # Stop update thread
            self.root.withdraw()
            # Only quit if mainloop is still running
            if self._mainloop_running:
                self.root.quit()
            logger.info("Overlay hidden")
        except Exception as e:
            logger.error(f"Failed to hide overlay: {e}")

    def destroy(self) -> None:
        """Destroy the overlay."""
        if not self.enabled:
            return

        try:
            self._stop_ui.set()  # Stop update thread
            if self.root is not None:
                # Only quit if mainloop is still running
                if self._mainloop_running:
                    self.root.quit()  # Exit mainloop first
                self.root.destroy()
                self.root = None
            logger.info("Overlay destroyed")
        except Exception as e:
            logger.error(f"Failed to destroy overlay: {e}")

    def run_main_loop(self) -> None:
        """Run the tkinter main loop (blocks)."""
        if not self.enabled or self.root is None:
            return

        try:
            self._mainloop_running = True
            self.root.mainloop()
        except Exception as e:
            logger.error(f"UI main loop error: {e}")
        finally:
            self._mainloop_running = False


class UIManager:
    """Thread-safe wrapper for managing the UI overlay."""

    def __init__(self, config: ConfigManager):
        """Initialize UI manager.

        Args:
            config: Configuration manager instance
        """
        self.config = config
        self.overlay: Optional[STTOverlay] = None
        self._ui_thread: Optional[threading.Thread] = None
        self.enabled = config.get("ui.enabled", True) and TKINTER_AVAILABLE

    def start_ui(self) -> None:
        """Start the UI in a separate thread."""
        if not self.enabled:
            return

        if self._ui_thread is None or not self._ui_thread.is_alive():
            self.overlay = STTOverlay(self.config)
            self._ui_thread = threading.Thread(target=self._run_ui, daemon=True)
            self._ui_thread.start()

            # Give UI time to initialize
            time.sleep(0.1)

    def _run_ui(self) -> None:
        """Run the UI (blocking)."""
        try:
            if self.overlay:
                self.overlay.create_overlay()
                if self.overlay.root is not None:
                    self.overlay.run_main_loop()
        except Exception as e:
            logger.error(f"UI thread error: {e}")

    def start_recording(self, profile: str = "general") -> None:
        """Signal recording start."""
        if self.enabled and self.overlay:
            self.overlay.start_recording(profile)

    def voice_detected(self) -> None:
        """Signal voice activity detected."""
        if self.enabled and self.overlay:
            self.overlay.voice_detected()

    def stop_recording(self) -> None:
        """Signal recording stop."""
        if self.enabled and self.overlay:
            self.overlay.stop_recording()

    def set_model_ready(self, ready: bool = True) -> None:
        """Update model status."""
        if self.enabled and self.overlay:
            self.overlay.set_model_ready(ready)

    def set_status(self, status_text: str, color: str = THEME["text_primary"]) -> None:
        """Set custom status."""
        if self.enabled and self.overlay:
            self.overlay.set_status(status_text, color)

    def set_device_name(self, device_name: str) -> None:
        """Set device name display."""
        if self.enabled and self.overlay:
            self.overlay.set_device_name(device_name)

    def set_volume_level(self, volume: float) -> None:
        """Set volume level display."""
        if self.enabled and self.overlay:
            self.overlay.set_volume_level(volume)

    def update_waveform(self, samples: List[float]) -> None:
        """Update waveform display."""
        if self.enabled and self.overlay:
            self.overlay.update_waveform(samples)

    def cleanup(self) -> None:
        """Clean up UI resources."""
        if self.enabled and self.overlay:
            self.overlay.destroy()
            # Wait for UI thread to finish (with timeout)
            if self._ui_thread and self._ui_thread.is_alive():
                self._ui_thread.join(timeout=2.0)
