"""Clipboard operations and text pasting."""

import logging
import subprocess
import time

from ..config.manager import ConfigManager

logger = logging.getLogger(__name__)


class ClipboardError(Exception):
    """Raised when clipboard operations fail."""


class ClipboardService:
    """Handles clipboard operations and text pasting."""

    def __init__(self, config: ConfigManager):
        """Initialize clipboard service.

        Args:
            config: Configuration manager instance
        """
        self.config = config

    def copy_to_clipboard(self, text: str) -> None:
        """Copy text to system clipboard.

        Args:
            text: Text to copy to clipboard
        """
        if not text.strip():
            logger.warning("Attempted to copy empty text to clipboard")
            return

        try:
            import pyperclip

            pyperclip.copy(text)
            logger.debug(f"Copied to clipboard: {text[:50]}...")

        except ImportError:
            logger.warning("pyperclip not available, trying system clipboard")
            self._copy_system_clipboard(text)
        except Exception as e:
            logger.error(f"Failed to copy to clipboard: {e}")
            raise ClipboardError(f"Failed to copy to clipboard: {e}") from e

    def paste_text(self, text: str) -> None:
        """Paste text to the active window.

        Args:
            text: Text to paste
        """
        if not text.strip():
            logger.warning("Attempted to paste empty text")
            return

        try:
            # First copy to clipboard
            self.copy_to_clipboard(text)

            # Wait a bit for clipboard to be set
            paste_delay = self.config.get("clipboard.paste_delay", 0.1)
            time.sleep(paste_delay)

            # Paste using xdotool (Linux/X11)
            self._paste_with_xdotool()

            logger.debug(f"Pasted text: {text[:50]}...")

        except Exception as e:
            logger.error(f"Failed to paste text: {e}")
            raise ClipboardError(f"Failed to paste text: {e}") from e

    def _copy_system_clipboard(self, text: str) -> None:
        """Copy text using system clipboard utilities."""
        try:
            # Try xclip first (most common on Linux)
            result = subprocess.run(
                ["xclip", "-selection", "clipboard"],
                input=text.encode("utf-8"),
                capture_output=True,
                timeout=5,
            )
            if result.returncode == 0:
                return

        except (
            subprocess.TimeoutExpired,
            subprocess.CalledProcessError,
            FileNotFoundError,
        ):
            pass

        try:
            # Try xsel as fallback
            result = subprocess.run(
                ["xsel", "--clipboard", "--input"],
                input=text.encode("utf-8"),
                capture_output=True,
                timeout=5,
            )
            if result.returncode == 0:
                return

        except (
            subprocess.TimeoutExpired,
            subprocess.CalledProcessError,
            FileNotFoundError,
        ):
            pass

        raise ClipboardError("No system clipboard utility available (xclip or xsel)")

    def _paste_with_xdotool(self) -> None:
        """Paste using xdotool."""
        try:
            # Use Ctrl+V to paste
            result = subprocess.run(
                ["xdotool", "key", "ctrl+v"], capture_output=True, timeout=5
            )
            if result.returncode != 0:
                raise ClipboardError(f"xdotool failed: {result.stderr.decode()}")

        except subprocess.TimeoutExpired:
            raise ClipboardError("xdotool paste timed out") from None
        except subprocess.CalledProcessError as e:
            raise ClipboardError(f"xdotool paste failed: {e}") from e
        except FileNotFoundError:
            raise ClipboardError(
                "xdotool not found - install xdotool for auto-paste"
            ) from None
