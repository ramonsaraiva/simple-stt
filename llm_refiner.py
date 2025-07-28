import openai
import anthropic
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class LLMRefiner:
    def __init__(self, config):
        self.config = config
        self.provider = config.get('llm.provider', 'openai')
        
        if self.provider == 'openai':
            api_key = config.get('api_keys.openai_api_key')
            if api_key:
                self.client = openai.OpenAI(api_key=api_key)
                logger.info("OpenAI client initialized")
            else:
                logger.warning("OpenAI API key not found")
                self.client = None
        elif self.provider == 'anthropic':
            api_key = config.get('api_keys.anthropic_api_key')
            if api_key:
                self.client = anthropic.Anthropic(api_key=api_key)
                logger.info("Anthropic client initialized")
            else:
                logger.warning("Anthropic API key not found")
                self.client = None
        else:
            logger.error(f"Unsupported LLM provider: {self.provider}")
            self.client = None
    
    def refine_text(self, text: str) -> Optional[str]:
        if not self.client or not text.strip():
            return text
        
        prompt = self.config.get('llm.refinement_prompt', 
            "Please clean up and format this transcribed text, fixing any grammar issues and making it more readable while preserving the original meaning:")
        
        try:
            if self.provider == 'openai':
                return self._refine_with_openai(text, prompt)
            elif self.provider == 'anthropic':
                return self._refine_with_anthropic(text, prompt)
        except Exception as e:
            logger.error(f"LLM refinement failed: {e}")
            return text
    
    def _refine_with_openai(self, text: str, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.config.get('llm.model', 'gpt-3.5-turbo'),
            messages=[
                {"role": "user", "content": f"{prompt}\n\nText to refine: {text}"}
            ],
            max_tokens=self.config.get('llm.max_tokens', 500),
            temperature=0.3
        )
        
        refined_text = response.choices[0].message.content.strip()
        logger.debug(f"OpenAI refined text: {refined_text}")
        return refined_text
    
    def _refine_with_anthropic(self, text: str, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.config.get('llm.model', 'claude-3-haiku-20240307'),
            max_tokens=self.config.get('llm.max_tokens', 500),
            messages=[
                {"role": "user", "content": f"{prompt}\n\nText to refine: {text}"}
            ]
        )
        
        refined_text = response.content[0].text.strip()
        logger.debug(f"Anthropic refined text: {refined_text}")
        return refined_text