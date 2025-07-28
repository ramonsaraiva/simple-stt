"""Text enhancement using LLM services."""

import logging
from typing import Dict, Optional

from ..config.manager import ConfigManager

logger = logging.getLogger(__name__)


class EnhancementError(Exception):
    """Raised when text enhancement fails."""


class TextEnhancer:
    """Enhances transcribed text using LLM services."""
    
    def __init__(self, config: ConfigManager):
        """Initialize text enhancer.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config
        self._client = None
    
    def enhance_text(self, text: str, profile: Optional[str] = None) -> Optional[str]:
        """Enhance text using specified LLM profile.
        
        Args:
            text: Text to enhance
            profile: LLM profile to use (defaults to configured default)
            
        Returns:
            Enhanced text, or None if enhancement failed
        """
        if not self.config.get("llm.enabled", True):
            logger.info("LLM enhancement disabled")
            return text
        
        if not text.strip():
            return text
        
        profile = profile or self.config.get("llm.default_profile", "general")
        
        try:
            prompt = self._get_profile_prompt(profile)
            if not prompt:
                logger.warning(f"Profile '{profile}' not found, using text as-is")
                return text
            
            logger.debug(f"Enhancing text with profile: {profile}")
            return self._call_llm(prompt, text)
        
        except Exception as e:
            logger.error(f"Text enhancement failed: {e}")
            return text  # Return original text on failure
    
    def list_profiles(self) -> Dict[str, Dict[str, str]]:
        """List available LLM profiles.
        
        Returns:
            Dictionary of profile_id -> profile_data
        """
        return self.config.get("llm.profiles", {})
    
    def _get_profile_prompt(self, profile: str) -> Optional[str]:
        """Get prompt for specified profile."""
        profiles = self.config.get("llm.profiles", {})
        if profile not in profiles:
            return None
        return profiles[profile].get("prompt", "")
    
    def _call_llm(self, prompt: str, text: str) -> Optional[str]:
        """Call LLM service to enhance text."""
        provider = self.config.get("llm.provider", "openai")
        
        if provider == "openai":
            return self._call_openai(prompt, text)
        elif provider == "anthropic":
            return self._call_anthropic(prompt, text)
        else:
            raise EnhancementError(f"Unsupported LLM provider: {provider}")
    
    def _call_openai(self, prompt: str, text: str) -> Optional[str]:
        """Call OpenAI API for text enhancement."""
        try:
            import openai
            
            api_key = self.config.get("api_keys.openai_api_key")
            if not api_key:
                logger.error("OpenAI API key not found. Check .env file or environment variables.")
                raise EnhancementError("OpenAI API key not configured")
            
            client = openai.OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model=self.config.get("llm.model", "gpt-3.5-turbo"),
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": text}
                ],
                max_tokens=self.config.get("llm.max_tokens", 500),
                temperature=0.3,
            )
            
            enhanced_text = response.choices[0].message.content
            if enhanced_text:
                logger.debug(f"OpenAI enhancement successful")
                return enhanced_text.strip()
            
            return None
        
        except ImportError:
            raise EnhancementError("OpenAI library not installed") from None
        except Exception as e:
            raise EnhancementError(f"OpenAI API call failed: {e}") from e
    
    def _call_anthropic(self, prompt: str, text: str) -> Optional[str]:
        """Call Anthropic API for text enhancement."""
        try:
            import anthropic
            
            api_key = self.config.get("api_keys.anthropic_api_key")
            if not api_key:
                logger.error("Anthropic API key not found. Check .env file or environment variables.")
                raise EnhancementError("Anthropic API key not configured")
            
            client = anthropic.Anthropic(api_key=api_key)
            
            response = client.messages.create(
                model=self.config.get("llm.model", "claude-3-haiku-20240307"),
                max_tokens=self.config.get("llm.max_tokens", 500),
                messages=[
                    {
                        "role": "user",
                        "content": f"{prompt}\n\nText to enhance: {text}"
                    }
                ],
            )
            
            enhanced_text = response.content[0].text
            if enhanced_text:
                logger.debug(f"Anthropic enhancement successful")
                return enhanced_text.strip()
            
            return None
        
        except ImportError:
            raise EnhancementError("Anthropic library not installed") from None
        except Exception as e:
            raise EnhancementError(f"Anthropic API call failed: {e}") from e