"""CHOFF-aware message retrieval agent."""
import json
from pathlib import Path
from typing import List, Optional, Dict
import attr
import logging
import asyncio

from .base import BaseAgent, AgentError, NonRetryableError
from ..message_store import Message, MessageStore

logger = logging.getLogger(__name__)

class RetrievalError(AgentError):
    """Raised when message retrieval fails."""
    pass

class InvalidQueryError(NonRetryableError):
    """Raised when query parameters are invalid."""
    pass

@attr.s(auto_attribs=True, frozen=True)
class ScoredMessage:
    """A message with its relevance score and matching patterns."""
    message: Message
    score: float
    matched_patterns: List[str]

class RetrieverAgent(BaseAgent):
    """Agent for intelligent message retrieval with CHOFF awareness.
    
    This agent specializes in finding relevant messages by analyzing:
    - CHOFF state transitions and patterns
    - Context markers and their relationships
    - Semantic relevance of content
    
    It supports both tag-based and semantic retrieval, with the ability to
    understand CHOFF patterns and state transitions over time.
    
    Args:
        client: Claude API client instance
        message_store: MessageStore instance to retrieve messages from
        cache_dir: Directory for caching responses (optional)
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Base delay in seconds for exponential backoff (default: 1.0)
        timeout: Timeout for retrieval operations (default: None)
    """
    
    def __init__(
        self,
        client: any,
        message_store: MessageStore,
        cache_dir: Optional[Path] = None,
        max_retries: int = 3,
        base_delay: float = 1.0,
        timeout: Optional[float] = None
    ):
        super().__init__(
            client=client,
            system_prompt=self._get_system_prompt(),
            cache_dir=cache_dir,
            max_retries=max_retries,
            base_delay=base_delay
        )
        self._store = message_store
        self.timeout = timeout
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for retrieval."""
        return """You are a CHOFF-aware retrieval agent.

Your task is to find and score relevant messages by analyzing:
1. CHOFF state markers {state:type[weight]}
2. Context definitions [context:type]
3. Pattern recognition markers &pattern:type|flow|

Guidelines:
1. Consider state transitions and their significance
2. Track context changes and relationships
3. Identify recurring patterns and their evolution
4. Score relevance based on both CHOFF markers and content
5. Explain matching patterns and their importance

Format your analysis with appropriate CHOFF markup reflecting the
retrieval context and identified patterns."""
    
    async def find_relevant(
        self,
        query: str,
        choff_tags: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> List[ScoredMessage]:
        """Find messages relevant to the query with CHOFF awareness.
        
        Args:
            query: Search query or description
            choff_tags: Optional CHOFF tags to filter by
            limit: Maximum number of results
            
        Returns:
            List[ScoredMessage]: Scored and sorted relevant messages
            
        Raises:
            InvalidQueryError: If query parameters are invalid
            RetrievalError: If retrieval fails
            asyncio.TimeoutError: If operation times out
        """
        if not query.strip():
            return []
            
        try:
            # First get candidate messages by CHOFF tags
            candidates = []
            seen_ids = set()
            
            if choff_tags:
                for tag in choff_tags:
                    for msg in self._store.find_by_choff_tag(tag):
                        msg_id = id(msg)
                        if msg_id not in seen_ids:
                            candidates.append(msg)
                            seen_ids.add(msg_id)
            
            if not candidates:
                # If no tags specified or no matches, analyze all messages
                # This would need to be implemented in MessageStore
                return []
            
            # Prepare prompt for relevance analysis
            prompt = f"""Analyze these messages for relevance to: {query}

Consider both content relevance and CHOFF patterns.

Messages to analyze:
{chr(10).join(f'- {msg.content} (CHOFF: {", ".join(msg.choff_tags)})' for msg in candidates)}

For each relevant message, provide:
1. Relevance score (0-1)
2. Matching CHOFF patterns
3. Brief explanation

Format: One message per line as JSON:
{{"message_idx": 0, "score": 0.95, "patterns": ["state:analytical", "context:technical"]}}"""

            # Get relevance analysis from Claude
            response = await self.send_message(prompt, timeout=self.timeout)
            analysis = response["content"][0]["text"]
            
            # Parse analysis and create scored messages
            scored_messages = []
            for line in analysis.strip().split("\n"):
                try:
                    result = json.loads(line)
                    msg_idx = result["message_idx"]
                    if 0 <= msg_idx < len(candidates):
                        scored_messages.append(ScoredMessage(
                            message=candidates[msg_idx],
                            score=result["score"],
                            matched_patterns=result["patterns"]
                        ))
                except (json.JSONDecodeError, KeyError, IndexError) as e:
                    logger.warning(f"Failed to parse analysis line: {e}")
                    continue
            
            # Sort by score and apply limit
            scored_messages.sort(key=lambda x: x.score, reverse=True)
            if limit is not None:
                scored_messages = scored_messages[:limit]
            
            return scored_messages
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise InvalidQueryError(str(e)) from e
            raise RetrievalError(f"Failed to find relevant messages: {e}") from e
    
    async def analyze_choff_patterns(
        self,
        messages: List[Message]
    ) -> Dict[str, List[str]]:
        """Analyze CHOFF patterns and relationships in messages.
        
        Args:
            messages: List of messages to analyze
            
        Returns:
            Dict mapping pattern types to lists of identified patterns
            
        Raises:
            RetrievalError: If analysis fails
            asyncio.TimeoutError: If operation times out
        """
        if not messages:
            return {}
            
        prompt = f"""Analyze CHOFF patterns in these messages:

{chr(10).join(f'- {msg.content} (CHOFF: {", ".join(msg.choff_tags)})' for msg in messages)}

Identify:
1. State transitions and their significance
2. Context relationships and changes
3. Recurring patterns and their evolution

Format results as JSON with pattern categories and lists."""

        try:
            response = await self.send_message(prompt, timeout=self.timeout)
            return json.loads(response["content"][0]["text"])
        except json.JSONDecodeError as e:
            raise RetrievalError(f"Failed to parse pattern analysis: {e}") from e
        except Exception as e:
            raise RetrievalError(f"Failed to analyze CHOFF patterns: {e}") from e