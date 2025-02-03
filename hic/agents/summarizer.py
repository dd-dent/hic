"""CHOFF-aware message summarization agent."""
import re
from pathlib import Path
from typing import List, Optional
from difflib import SequenceMatcher

from .base import BaseAgent, AgentError

class SummaryError(AgentError):
    """Raised when summarization fails."""
    pass

class SummarizerAgent(BaseAgent):
    """Agent for summarizing message conversations with CHOFF awareness.
    
    This agent specializes in creating concise summaries while preserving CHOFF
    state transitions and context markers. It supports batch processing of messages
    and includes relevance scoring of generated summaries.
    
    Args:
        client: Claude API client instance
        cache_dir: Directory for caching responses (optional)
        batch_size: Number of messages to summarize at once (default: 5)
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Base delay in seconds for exponential backoff (default: 1.0)
    """
    
    def __init__(
        self,
        client: any,
        cache_dir: Optional[Path] = None,
        batch_size: int = 5,
        max_retries: int = 3,
        base_delay: float = 1.0
    ):
        if batch_size < 1:
            raise ValueError("Batch size must be positive")
            
        super().__init__(
            client=client,
            system_prompt=self._get_system_prompt(),
            cache_dir=cache_dir,
            max_retries=max_retries,
            base_delay=base_delay
        )
        self.batch_size = batch_size
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for summarization."""
        return """You are a CHOFF-aware summarization agent.

Your task is to create concise summaries of message conversations while preserving:
- CHOFF state markers {state:type}
- Context definitions [context:type]
- Pattern recognition markers &pattern:type|flow|

Guidelines:
1. Preserve all CHOFF states present in the messages
2. Maintain context transitions and flow
3. Focus on key points and decisions
4. Use bullet points for clarity when appropriate
5. Include relevant pattern markers

Format your summary with appropriate CHOFF markup reflecting the overall
conversation flow and state transitions."""
    
    def _validate_choff_markup(self, messages: List[str]) -> None:
        """Validate that messages contain CHOFF markup."""
        choff_pattern = r'\{state:[^}]+\}|\[context:[^\]]+\]|&pattern:[^|]+\|'
        
        for msg in messages:
            if not re.search(choff_pattern, msg):
                raise SummaryError("Missing CHOFF markup in messages")
    
    def _extract_choff_states(self, text: str) -> List[str]:
        """Extract CHOFF state markers from text."""
        state_pattern = r'\{state:[^}]+\}'
        return re.findall(state_pattern, text)
    
    def summarize_messages(self, messages: List[str]) -> str:
        """Summarize a list of messages while preserving CHOFF markup.
        
        Args:
            messages: List of messages to summarize
            
        Returns:
            A summary string with preserved CHOFF markup
            
        Raises:
            SummaryError: If summarization fails or messages lack CHOFF markup
        """
        if not messages:
            raise SummaryError("No messages to summarize")
            
        self._validate_choff_markup(messages)
        
        # Process in batches
        summaries = []
        for i in range(0, len(messages), self.batch_size):
            batch = messages[i:i + self.batch_size]
            
            # Extract all CHOFF states from batch
            states = set()
            for msg in batch:
                states.update(self._extract_choff_states(msg))
            
            # Create prompt for batch
            prompt = f"""Summarize these {len(batch)} messages while preserving CHOFF markup:

Present CHOFF states: {', '.join(sorted(states))}

Messages:
{chr(10).join(f'- {msg}' for msg in batch)}"""

            response = self.send_message(prompt)
            summaries.append(response["content"][0]["text"])
        
        # Combine batch summaries if needed
        if len(summaries) == 1:
            return summaries[0]
            
        # Create a meta-summary for multiple batches
        meta_prompt = f"""Create a unified summary of these batch summaries while preserving CHOFF markup:

{chr(10).join(f'Batch {i+1}:{chr(10)}{summary}' for i, summary in enumerate(summaries))}"""
        
        response = self.send_message(meta_prompt)
        return response["content"][0]["text"]
    
    def score_summary(self, summary: str, original_messages: List[str]) -> float:
        """Score the relevance of a summary compared to original messages.
        
        The score is based on:
        1. CHOFF state preservation
        2. Key content similarity
        3. Length efficiency
        
        Args:
            summary: Generated summary
            original_messages: Original messages that were summarized
            
        Returns:
            Float between 0 and 1 indicating relevance score
        """
        # Check CHOFF state preservation
        original_states = set()
        for msg in original_messages:
            original_states.update(self._extract_choff_states(msg))
        summary_states = set(self._extract_choff_states(summary))
        state_score = len(summary_states & original_states) / len(original_states) if original_states else 0
        
        # Content similarity using difflib
        combined_original = ' '.join(original_messages)
        similarity = SequenceMatcher(None, summary, combined_original).ratio()
        
        # Length efficiency (penalize if summary is longer than combined original)
        length_ratio = len(summary) / len(combined_original)
        length_score = min(1.0, 1.0 / length_ratio)
        
        # Weighted combination
        return (0.4 * state_score + 0.4 * similarity + 0.2 * length_score)