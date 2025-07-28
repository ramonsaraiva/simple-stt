import pyperclip
import subprocess
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class ClipboardManager:
    def __init__(self, config):
        self.config = config
        
    def copy_to_clipboard(self, text: str) -> bool:
        try:
            pyperclip.copy(text)
            logger.info("Text copied to clipboard successfully")
            print(f"Copied to clipboard: {text}")
            return True
        except Exception as e:
            logger.error(f"Failed to copy to clipboard: {e}")
            return False
    
    def paste_text(self, text: str) -> bool:
        if not self.config.get('clipboard.auto_paste', False):
            return False
            
        try:
            # First copy to clipboard
            if not self.copy_to_clipboard(text):
                return False
            
            # Small delay before pasting
            delay = self.config.get('clipboard.paste_delay', 0.1)
            time.sleep(delay)
            
            # Use xdotool to paste (Ctrl+V) into the active window
            subprocess.run(['xdotool', 'key', 'ctrl+v'], check=True)
            logger.info("Text auto-pasted to active window")
            print("Auto-pasted text to active window")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to auto-paste (xdotool error): {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to auto-paste: {e}")
            return False
    
    def get_active_window_info(self) -> Optional[dict]:
        try:
            # Get active window ID
            result = subprocess.run(['xdotool', 'getactivewindow'], 
                                  capture_output=True, text=True, check=True)
            window_id = result.stdout.strip()
            
            # Get window name
            result = subprocess.run(['xdotool', 'getwindowname', window_id], 
                                  capture_output=True, text=True, check=True)
            window_name = result.stdout.strip()
            
            return {
                'id': window_id,
                'name': window_name
            }
        except Exception as e:
            logger.error(f"Failed to get active window info: {e}")
            return None