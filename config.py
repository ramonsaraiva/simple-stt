import yaml
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class Config:
    def __init__(self, config_path="config.yaml"):
        # Load environment variables from .env file
        load_dotenv()
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        logger.debug(f"Configuration loaded from {self.config_path}")
    
    def _load_config(self):
        if not self.config_path.exists():
            logger.error(f"Configuration file not found: {self.config_path}")
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise
        
        # Override with environment variables if they exist
        if os.getenv('OPENAI_API_KEY'):
            config['api_keys']['openai_api_key'] = os.getenv('OPENAI_API_KEY')
            logger.debug("Using OPENAI_API_KEY from environment")
        if os.getenv('ANTHROPIC_API_KEY'):
            config['api_keys']['anthropic_api_key'] = os.getenv('ANTHROPIC_API_KEY')
            logger.debug("Using ANTHROPIC_API_KEY from environment")
        
        return config
    
    def get(self, key, default=None):
        keys = key.split('.')
        value = self.config
        for k in keys:
            value = value.get(k, default)
            if value is None:
                return default
        return value
    
    def save(self):
        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            logger.debug(f"Configuration saved to {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise