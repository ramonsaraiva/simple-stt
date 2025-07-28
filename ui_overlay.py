import logging
import os
import threading
import time
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import tkinter, fall back to no-op mode if not available
try:
    import tkinter as tk

    TKINTER_AVAILABLE = True
except ImportError:
    logger.warning("tkinter not available - UI overlay will be disabled")
    TKINTER_AVAILABLE = False
    tk = None


class STTOverlay:
    def __init__(self, config):
        self.config = config
        self.root = None
        self.labels = {}
        self.start_time = None
        self.is_recording = False
        self.model_ready = False
        self.current_profile = "general"
        self._mainloop_running = False

        # UI update thread control
        self._ui_thread = None
        self._stop_ui = threading.Event()

        # Configuration - disable if tkinter not available
        self.enabled = config.get("ui.enabled", True) and TKINTER_AVAILABLE
        if not TKINTER_AVAILABLE and config.get("ui.enabled", True):
            logger.info("UI overlay disabled - tkinter not available")

        self.position_x = config.get("ui.position_x", 50)
        self.position_y = config.get("ui.position_y", 50)
        self.auto_hide_delay = config.get("ui.auto_hide_delay", 3.0)

    def create_overlay(self):
        """Create the overlay window"""
        if not self.enabled:
            return

        try:
            self.root = tk.Tk()
            self.root.title("STT Status")

            # Configure window
            self.root.attributes("-topmost", True)  # Always on top
            self.root.attributes("-alpha", 0.9)  # Slight transparency
            self.root.resizable(False, False)

            # Create widgets first to get accurate window size
            self.root.configure(bg="#2b2b2b")
            self._create_widgets()

            # Update geometry to calculate size
            self.root.update_idletasks()

            # Calculate center position
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            window_width = self.root.winfo_reqwidth()
            window_height = self.root.winfo_reqheight()
            
            center_x = (screen_width - window_width) // 2
            center_y = (screen_height - window_height) // 2

            # Position window at center
            self.root.geometry(f"{window_width}x{window_height}+{center_x}+{center_y}")

            # Handle window close event to terminate the process
            self.root.protocol("WM_DELETE_WINDOW", self._on_window_close)

            self._update_display()

            logger.info("STT overlay created and centered")

        except Exception as e:
            logger.error(f"Failed to create overlay: {e}")
            self.enabled = False

    def _create_widgets(self):
        """Create the UI widgets"""
        # Main frame
        main_frame = tk.Frame(self.root, bg="#2b2b2b", padx=10, pady=8)
        main_frame.pack()

        # Status line
        self.labels["status"] = tk.Label(
            main_frame,
            text="⏹️ Idle",
            bg="#2b2b2b",
            fg="#ffffff",
            font=("Arial", 12, "bold"),
        )
        self.labels["status"].pack(anchor="w")

        # Timer line
        self.labels["timer"] = tk.Label(
            main_frame, text="00:00", bg="#2b2b2b", fg="#888888", font=("Monaco", 10)
        )
        self.labels["timer"].pack(anchor="w")

        # Model status line
        self.labels["model"] = tk.Label(
            main_frame,
            text="⏳ Loading model...",
            bg="#2b2b2b",
            fg="#ffaa00",
            font=("Arial", 9),
        )
        self.labels["model"].pack(anchor="w")

        # Profile line
        self.labels["profile"] = tk.Label(
            main_frame, text="📝 general", bg="#2b2b2b", fg="#00aaff", font=("Arial", 9)
        )
        self.labels["profile"].pack(anchor="w")

    def start_recording(self, profile="general"):
        """Signal that recording has started"""
        if not self.enabled:
            return

        self.is_recording = True
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

    def stop_recording(self):
        """Signal that recording has stopped"""
        if not self.enabled:
            return

        self.is_recording = False
        self._schedule_update()

        # Auto-hide after delay
        if self.auto_hide_delay > 0:
            threading.Timer(self.auto_hide_delay, self.hide).start()

        logger.info("Recording stopped - UI updated")

    def set_model_ready(self, ready=True):
        """Update model loading status"""
        if not self.enabled:
            return

        self.model_ready = ready
        self._schedule_update()

        logger.debug(f"Model status updated: {'ready' if ready else 'loading'}")

    def set_status(self, status_text, color="#ffffff"):
        """Set custom status text"""
        if not self.enabled or self.root is None:
            return

        try:
            self.root.after(
                0, lambda: self.labels["status"].config(text=status_text, fg=color)
            )
        except Exception as e:
            logger.error(f"Failed to update status: {e}")

    def _schedule_update(self):
        """Schedule a UI update on the main thread"""
        if self.root is not None:
            try:
                self.root.after(0, self._update_display)
            except Exception as e:
                logger.error(f"Failed to schedule UI update: {e}")

    def _update_display(self):
        """Update the display (must run on main thread)"""
        if not self.enabled or self.root is None:
            return

        try:
            # Update status
            if self.is_recording:
                self.labels["status"].config(text="🎤 Recording", fg="#ff4444")
            else:
                self.labels["status"].config(text="⏹️ Processing", fg="#ffaa00")

            # Update timer
            if self.start_time and self.is_recording:
                elapsed = datetime.now() - self.start_time
                minutes = int(elapsed.total_seconds() // 60)
                seconds = int(elapsed.total_seconds() % 60)
                timer_text = f"{minutes:02d}:{seconds:02d}"
                self.labels["timer"].config(text=timer_text, fg="#ffffff")
            else:
                self.labels["timer"].config(text="--:--", fg="#888888")

            # Update model status
            if self.model_ready:
                self.labels["model"].config(text="✅ Model ready", fg="#00ff00")
            else:
                self.labels["model"].config(text="⏳ Loading model...", fg="#ffaa00")

            # Update profile
            self.labels["profile"].config(text=f"📝 {self.current_profile}")

        except Exception as e:
            logger.error(f"Failed to update display: {e}")

    def _on_window_close(self):
        """Handle window close event by terminating the entire process"""
        logger.info("UI window closed, terminating process immediately")
        print("\n🛑 Window closed, terminating process...")
        os._exit(0)  # Force immediate termination

    def _ui_update_loop(self):
        """Background thread to update timer while recording"""
        while not self._stop_ui.is_set() and self.is_recording:
            self._schedule_update()
            time.sleep(1.0)  # Update every 1000ms to reduce interference

    def show(self):
        """Show the overlay"""
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

    def hide(self):
        """Hide the overlay"""
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

    def destroy(self):
        """Destroy the overlay"""
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

    def run_main_loop(self):
        """Run the tkinter main loop (blocks)"""
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
    """Thread-safe wrapper for managing the UI overlay"""

    def __init__(self, config):
        self.config = config
        self.overlay = None
        self._ui_thread = None
        self.enabled = config.get("ui.enabled", True) and TKINTER_AVAILABLE

    def start_ui(self):
        """Start the UI in a separate thread"""
        if not self.enabled:
            return

        if self._ui_thread is None or not self._ui_thread.is_alive():
            self.overlay = STTOverlay(self.config)
            self._ui_thread = threading.Thread(target=self._run_ui, daemon=True)
            self._ui_thread.start()

            # Give UI time to initialize
            time.sleep(0.1)

    def _run_ui(self):
        """Run the UI (blocking)"""
        try:
            self.overlay.create_overlay()
            if self.overlay.root is not None:
                self.overlay.run_main_loop()
        except Exception as e:
            logger.error(f"UI thread error: {e}")

    def start_recording(self, profile="general"):
        """Signal recording start"""
        if self.enabled and self.overlay:
            self.overlay.start_recording(profile)

    def stop_recording(self):
        """Signal recording stop"""
        if self.enabled and self.overlay:
            self.overlay.stop_recording()

    def set_model_ready(self, ready=True):
        """Update model status"""
        if self.enabled and self.overlay:
            self.overlay.set_model_ready(ready)

    def set_status(self, status_text, color="#ffffff"):
        """Set custom status"""
        if self.enabled and self.overlay:
            self.overlay.set_status(status_text, color)

    def cleanup(self):
        """Clean up UI resources"""
        if self.enabled and self.overlay:
            self.overlay.destroy()
            # Wait for UI thread to finish (with timeout)
            if self._ui_thread and self._ui_thread.is_alive():
                self._ui_thread.join(timeout=2.0)
