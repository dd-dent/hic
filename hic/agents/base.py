"""Base agent implementation for HIC."""
import json
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional
import logging
import asyncio
import attr

logger = logging.getLogger(__name__)

class AgentError(Exception):
    """Base exception for agent-related errors."""
    pass

class RetryError(AgentError):
    """Raised when max retries are exceeded."""
    pass

class NonRetryableError(AgentError):
    """Error that should not trigger retries."""
    pass

@attr.s(auto_attribs=True, slots=True)
class TokenUsage:
    """Track token usage with validation."""
    input_tokens: int = 0
    output_tokens: int = 0
    
    @property
    def total_tokens(self) -> int:
        """Calculate total tokens used."""
        return self.input_tokens + self.output_tokens
    
    def update(self, usage_data: Dict[str, Any]) -> None:
        """Update token counts from usage data."""
        if not isinstance(usage_data, dict):
            logger.warning("Invalid usage data format")
            return
            
        self.input_tokens += usage_data.get('input_tokens', 0)
        self.output_tokens += usage_data.get('output_tokens', 0)

class BaseAgent:
    """Base agent class implementing core Claude interaction functionality.
    
    This class provides the foundation for Claude-powered agents with:
    - CHOFF-aware system prompts
    - Async support with asyncio
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
        self.usage = TokenUsage()
        
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(self, prompt: str) -> Optional[Path]:
        """Get cache file path for a given prompt."""
        if not self.cache_dir:
            return None
            
        # Create deterministic filename from prompt
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        return self.cache_dir / f"{prompt_hash}.json"
    
    async def _load_from_cache(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Load response from cache if available."""
        cache_path = self._get_cache_path(prompt)
        if not cache_path or not cache_path.exists():
            return None
            
        try:
            # Use asyncio.to_thread for file I/O
            response = await asyncio.to_thread(
                lambda: json.loads(cache_path.read_text())
            )
            return response
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to load cache: {e}")
            return None
    
    async def _save_to_cache(self, prompt: str, response: Dict[str, Any]) -> None:
        """Save response to cache atomically."""
        cache_path = self._get_cache_path(prompt)
        if not cache_path:
            return
            
        temp_path = cache_path.with_suffix('.tmp')
        try:
            # Write to temporary file first
            content = json.dumps(response)
            
            # Write to temp file
            await asyncio.to_thread(
                lambda: temp_path.write_text(content)
            )
            # Atomic rename
            await asyncio.to_thread(
                lambda: temp_path.replace(cache_path)
            )
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
            # Clean up temp file if it exists
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
    
    def _update_token_usage(self, response: Dict[str, Any]) -> None:
        """Update token usage statistics from response."""
        usage = response.get('usage', {})
        if not isinstance(usage, dict):
            logger.warning("Invalid usage data format")
            return
        self.usage.update(usage)
    
    async def _retry_with_backoff(self, operation):
        """Execute operation with retry logic and exponential backoff."""
        retry_count = 0
        last_error = None
        
        while retry_count < self.max_retries:  # Changed <= to <
            try:
                return await operation()
            except NonRetryableError:
                raise
            except Exception as e:
                last_error = e
                retry_count += 1
                if retry_count == self.max_retries:
                    raise RetryError(f"Max retries ({self.max_retries}) exceeded") from last_error
                
                delay = self.base_delay * (2 ** (retry_count - 1))
                logger.warning(f"Retry {retry_count}/{self.max_retries} after {delay}s delay")
                await asyncio.sleep(delay)
        
        # This should never be reached due to the raise in the loop
        raise RetryError(f"Max retries ({self.max_retries}) exceeded") from last_error
    
    async def send_message(
        self,
        prompt: str,
        force_refresh: bool = False,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """Send a message to Claude with retry logic and caching.
        
        Args:
            prompt: The message to send
            force_refresh: If True, bypass cache and force new API call
            timeout: Optional timeout in seconds
            
        Returns:
            Dict containing response data including content and usage stats
            
        Raises:
            RetryError: If max retries are exceeded
            NonRetryableError: For errors that shouldn't be retried
            AgentError: For other agent-related errors
            asyncio.TimeoutError: If timeout is exceeded
        """
        # Check cache first unless force refresh
        if not force_refresh:
            cached = await self._load_from_cache(prompt)
            if cached:
                self._update_token_usage(cached)
                return cached

        async def _send():
            try:
                response = await self.client.create_message(
                    model="claude-3-5-sonnet-latest",
                    system=self.system_prompt,
                    messages=[{"role": "user", "content": prompt}]
                )
            except ValueError as e:
                # Invalid input shouldn't be retried
                raise NonRetryableError(str(e)) from e
            
            # Update usage stats
            self._update_token_usage(response)
            
            # Cache successful response
            await self._save_to_cache(prompt, response)
            
            return response

        # Use optional timeout
        if timeout is not None:
            try:
                return await asyncio.wait_for(
                    self._retry_with_backoff(_send),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                raise asyncio.TimeoutError("Message send timed out")
        else:
            return await self._retry_with_backoff(_send)