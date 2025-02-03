"""Base agent implementation for HIC."""
import json
import time
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class AgentError(Exception):
    """Base exception for agent-related errors."""
    pass

class RetryError(AgentError):
    """Raised when max retries are exceeded."""
    pass

class BaseAgent:
    """Base agent class implementing core Claude interaction functionality.
    
    This class provides the foundation for Claude-powered agents with:
    - CHOFF-aware system prompts
    - Retry logic with exponential backoff
    - Token usage monitoring
    - Response caching for tests
    
    Args:
        client: Claude API client instance
        system_prompt: System prompt for the agent
        cache_dir: Directory for caching responses (optional)
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Base delay in seconds for exponential backoff (default: 1.0)
    """
    
    def __init__(
        self,
        client: Any,
        system_prompt: str,
        cache_dir: Optional[Path] = None,
        max_retries: int = 3,
        base_delay: float = 1.0
    ):
        self.client = client
        self.system_prompt = system_prompt
        self.cache_dir = cache_dir
        self.max_retries = max_retries
        self.base_delay = base_delay
        
        # Token usage tracking
        self.total_tokens = 0
        self.input_tokens = 0
        self.output_tokens = 0
        
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(self, prompt: str) -> Optional[Path]:
        """Get cache file path for a given prompt."""
        if not self.cache_dir:
            return None
            
        # Create deterministic filename from prompt
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        return self.cache_dir / f"{prompt_hash}.json"
    
    def _load_from_cache(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Load response from cache if available."""
        cache_path = self._get_cache_path(prompt)
        if not cache_path or not cache_path.exists():
            return None
            
        try:
            with cache_path.open() as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return None
    
    def _save_to_cache(self, prompt: str, response: Dict[str, Any]) -> None:
        """Save response to cache."""
        cache_path = self._get_cache_path(prompt)
        if not cache_path:
            return
            
        try:
            with cache_path.open('w') as f:
                json.dump(response, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def _update_token_usage(self, response: Dict[str, Any]) -> None:
        """Update token usage statistics from response."""
        usage = response.get('usage', {})
        self.input_tokens += usage.get('input_tokens', 0)
        self.output_tokens += usage.get('output_tokens', 0)
        self.total_tokens = self.input_tokens + self.output_tokens
    
    def send_message(
        self,
        prompt: str,
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """Send a message to Claude with retry logic and caching.
        
        Args:
            prompt: The message to send
            force_refresh: If True, bypass cache and force new API call
            
        Returns:
            Dict containing response data including content and usage stats
            
        Raises:
            RetryError: If max retries are exceeded
            AgentError: For other agent-related errors
        """
        # Check cache first unless force refresh
        if not force_refresh:
            cached = self._load_from_cache(prompt)
            if cached:
                self._update_token_usage(cached)
                return cached
        
        retry_count = 0
        while retry_count < self.max_retries:
            try:
                response = self.client.create_message(
                    model="claude-3-5-sonnet-20240620",
                    system=self.system_prompt,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                # Update usage stats
                self._update_token_usage(response)
                
                # Cache successful response
                self._save_to_cache(prompt, response)
                
                return response
                
            except Exception as e:
                retry_count += 1
                if retry_count == self.max_retries:
                    raise RetryError(f"Max retries ({self.max_retries}) exceeded") from e
                    
                # Exponential backoff
                delay = self.base_delay * (2 ** (retry_count - 1))
                logger.warning(f"Retry {retry_count}/{self.max_retries} after {delay}s delay")
                time.sleep(delay)