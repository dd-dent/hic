"""CHOFF-aware message summarization agent."""
import re
import asyncio
from pathlib import Path
from typing import List, Set, Dict
from difflib import SequenceMatcher
from dataclasses import dataclass, field

from .base import BaseAgent, AgentError

class SummaryError(AgentError):
    """Raised when summarization fails."""
    pass

@dataclass
class ChoffState:
    """Represents a CHOFF state with weights and transitions."""
    state_type: str
    weight: float = 1.0
    transitions: List['ChoffState'] = field(default_factory=list)
    
    @classmethod
    def from_tag(cls, tag: str) -> 'ChoffState':
        """Parse a CHOFF state tag."""
        # For backward compatibility, use the original regex parser
        match = re.match(r'\{state:([^}\[]+)(?:\[([0-9.]+)\])?}', tag)
        if not match:
            raise ValueError(f"Invalid CHOFF state tag: {tag}")
        state_type, weight = match.groups()
        return cls(
            state_type=state_type.strip(),
            weight=float(weight) if weight else 1.0
        )

class SummarizerAgent(BaseAgent):
    """Agent for summarizing message conversations with CHOFF awareness.
    
    This agent specializes in creating concise summaries while preserving CHOFF
    state transitions and context markers. It supports concurrent batch processing
    of messages and includes relevance scoring of generated summaries.
    
    Args:
        client: Claude API client instance
        cache_dir: Directory for caching responses (optional)
        batch_size: Number of messages to summarize at once (default: 5)
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Base delay in seconds for exponential backoff (default: 1.0)
        timeout: Timeout for each summarization operation (default: None)
    """
    
    def __init__(
        self,
        client: any,
        cache_dir: Path = None,
        batch_size: int = 5,
        max_retries: int = 3,
        base_delay: float = 1.0,
        timeout: float = None
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
        self.timeout = timeout
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for summarization."""
        return """You are a CHOFF-aware summarization agent.

Your task is to create concise summaries of message conversations while preserving:
- CHOFF state markers {state:type[weight]}
- Context definitions [context:type]
- Pattern recognition markers &pattern:type|flow|

Guidelines:
1. Preserve all CHOFF states present in the messages
2. Maintain state transitions and weights
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
    
    def _extract_choff_states(self, text: str) -> List[ChoffState]:
        """Extract CHOFF state markers from text."""
        state_pattern = r'\{state:[^}]+\}'
        tags = re.findall(state_pattern, text)
        states = []
        for tag in tags:
            try:
                states.append(ChoffState.from_tag(tag))
            except ValueError:
                # Skip invalid tags but don't fail
                continue
        return states
    
    def _merge_states(self, states: List[ChoffState]) -> List[ChoffState]:
        """Merge similar states and calculate weights."""
        state_map: Dict[str, float] = {}
        for state in states:
            if state.state_type in state_map:
                state_map[state.state_type] += state.weight
            else:
                state_map[state.state_type] = state.weight
                
        # Normalize weights
        total_weight = sum(state_map.values())
        if total_weight > 0:
            for state_type in state_map:
                state_map[state_type] /= total_weight
                
        return [
            ChoffState(state_type=state_type, weight=weight)
            for state_type, weight in state_map.items()
        ]
    
    async def _process_batch(
        self,
        batch: List[str],
        states: Set[str]
    ) -> str:
        """Process a batch of messages concurrently."""
        # Create prompt for batch
        prompt = f"""Summarize these {len(batch)} messages while preserving CHOFF markup:

Present CHOFF states: {', '.join(sorted(states))}

Messages:
{chr(10).join(f'- {msg}' for msg in batch)}"""

        response = await self.send_message(prompt, timeout=self.timeout)
        return response["content"][0]["text"]
    
    async def summarize_messages(self, messages: List[str]) -> str:
        """Summarize a list of messages while preserving CHOFF markup.
        
        Args:
            messages: List of messages to summarize
            
        Returns:
            A summary string with preserved CHOFF markup
            
        Raises:
            SummaryError: If summarization fails or messages lack CHOFF markup
            asyncio.TimeoutError: If operation times out
        """
        if not messages:
            raise SummaryError("No messages to summarize")
            
        self._validate_choff_markup(messages)
        
        # Process in batches concurrently
        summaries = []
        
        async def process_batch(batch: List[str], states: Set[str]) -> None:
            try:
                summary = await self._process_batch(batch, states)
                summaries.append(summary)
            except asyncio.TimeoutError:
                # Propagate timeout error
                raise
            except Exception as e:
                # Wrap other errors
                raise SummaryError(f"Batch processing failed: {e}") from e
        
        # Process batches with timeout
        try:
            tasks = []
            for i in range(0, len(messages), self.batch_size):
                batch = messages[i:i + self.batch_size]
                
                # Extract all CHOFF states from batch
                states = set()
                for msg in batch:
                    states.update(state.state_type for state in self._extract_choff_states(msg))
                
                # Process batch
                tasks.append(asyncio.create_task(process_batch(batch, states)))
            
            # Wait for all tasks to complete with timeout
            if self.timeout:
                await asyncio.wait_for(asyncio.gather(*tasks), timeout=self.timeout)
            else:
                await asyncio.gather(*tasks)
        except asyncio.TimeoutError:
            # Check if we timed out
            if not summaries:
                raise asyncio.TimeoutError("Message processing timed out")
        
        # Combine batch summaries if needed
        if len(summaries) == 1:
            return summaries[0]
            
        # Create a meta-summary for multiple batches
        meta_prompt = f"""Create a unified summary of these batch summaries while preserving CHOFF markup:

{chr(10).join(f'Batch {i+1}:{chr(10)}{summary}' for i, summary in enumerate(summaries))}"""
        
        response = await self.send_message(meta_prompt, timeout=self.timeout)
        return response["content"][0]["text"]
    
    async def score_summary(self, summary: str, original_messages: List[str]) -> float:
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
        original_states = []
        for msg in original_messages:
            original_states.extend(self._extract_choff_states(msg))
        summary_states = self._extract_choff_states(summary)
        
        # Merge states and compare
        merged_original = self._merge_states(original_states)
        merged_summary = self._merge_states(summary_states)
        
        # Calculate state preservation score
        state_types_original = {s.state_type for s in merged_original}
        state_types_summary = {s.state_type for s in merged_summary}
        state_score = len(state_types_summary & state_types_original) / len(state_types_original) if state_types_original else 0
        
        # Content similarity using difflib
        combined_original = ' '.join(original_messages)
        similarity = SequenceMatcher(None, summary, combined_original).ratio()
        
        # Length efficiency (penalize if summary is longer than combined original)
        length_ratio = len(summary) / len(combined_original)
        length_score = min(1.0, 1.0 / length_ratio)
        
        # Weighted combination
        return (0.4 * state_score + 0.4 * similarity + 0.2 * length_score)