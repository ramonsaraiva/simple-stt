"""Configuration management for the STT system."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Raised when configuration is invalid or cannot be loaded."""


class ConfigManager:
    """Manages application configuration with validation and defaults."""
    
    DEFAULT_CONFIG = {
        "audio": {
            "sample_rate": 16000,
            "channels": 1,
            "chunk_size": 2048,
            "format": "int16",
            "silence_threshold": 20.0,
            "silence_duration": 2.0,
            "max_recording_time": 120.0,
        },
        "whisper": {
            "model": "turbo",
            "language": "en",
            "device": "cuda",
            "compute_type": "float16",
            "load_timeout": 60,
        },
        "ui": {
            "enabled": True,
            "position_x": 50,
            "position_y": 50,
            "auto_hide_delay": 3.0,
        },
        "llm": {
            "enabled": True,
            "provider": "openai",
            "model": "gpt-3.5-turbo",
            "max_tokens": 500,
            "default_profile": "general",
            "profiles": {
                "general": {
                    "name": "General Text Cleanup",
                    "prompt": (
                        "Please clean up and format this transcribed text, "
                        "fixing any grammar issues and making it more readable."
                        "It is extremely important to maintain the original meaning "
                        "and not add any additional information:"
                    ),
                },
                "todo": {
                    "name": "Todo/Task",
                    "prompt": (
                        "Convert this speech into a clear, actionable todo item or task description. "
                        "Make it specific, concise, and action-oriented. "
                        "Use bullet points (markdown format) if multiple tasks are mentioned:"
                    ),
                },
            },
        },
        "clipboard": {
            "auto_paste": False,
            "paste_delay": 0.1,
        },
        "api_keys": {
            "openai_api_key": "",
            "anthropic_api_key": "",
        },
    }
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file. If None, uses default locations.
        """
        # Load environment variables from .env file first
        self._load_dotenv()
        
        self.config_path = self._resolve_config_path(config_path)
        self._config = self.DEFAULT_CONFIG.copy()
        self.load()
    
    def _resolve_config_path(self, config_path: Optional[Union[str, Path]]) -> Path:
        """Resolve the configuration file path."""
        if config_path:
            return Path(config_path)
        
        # Try default locations
        candidates = [
            Path("config.yaml"),
            Path.home() / ".stt" / "config.yaml",
            Path("/etc/stt/config.yaml"),
        ]
        
        for candidate in candidates:
            if candidate.exists():
                return candidate
        
        # Return first candidate as default
        return candidates[0]
    
    def load(self) -> None:
        """Load configuration from file."""
        if not self.config_path.exists():
            logger.info(f"Config file not found at {self.config_path}, using defaults")
            return
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f) or {}
            
            # Deep merge with defaults
            self._config = self._deep_merge(self.DEFAULT_CONFIG, user_config)
            
            # Load environment variables for API keys
            self._load_env_vars()
            
            logger.info(f"Configuration loaded from {self.config_path}")
            
        except Exception as e:
            raise ConfigError(f"Failed to load configuration: {e}") from e
    
    def save(self) -> None:
        """Save current configuration to file."""
        try:
            # Ensure directory exists
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Don't save API keys to file
            config_to_save = self._config.copy()
            if "api_keys" in config_to_save:
                config_to_save["api_keys"] = {"openai_api_key": "", "anthropic_api_key": ""}
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_to_save, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to {self.config_path}")
            
        except Exception as e:
            raise ConfigError(f"Failed to save configuration: {e}") from e
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation (e.g., 'audio.sample_rate')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        # Navigate to parent dict
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
    
    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _load_dotenv(self) -> None:
        """Load environment variables from .env file."""
        # Look for .env file in these locations
        env_candidates = [
            Path(".env"),
            Path.cwd() / ".env", 
            Path.home() / ".stt" / ".env",
        ]
        
        for env_path in env_candidates:
            if env_path.exists():
                logger.info(f"Loading environment variables from {env_path}")
                load_dotenv(env_path)
                break
        else:
            # Also try to load from current directory without specific path
            load_dotenv()
    
    def _load_env_vars(self) -> None:
        """Load API keys from environment variables."""
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            self.set("api_keys.openai_api_key", openai_key)
            logger.debug("Loaded OpenAI API key from environment")
        
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key:
            self.set("api_keys.anthropic_api_key", anthropic_key) 
            logger.debug("Loaded Anthropic API key from environment")
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get the full configuration dictionary."""
        return self._config.copy()