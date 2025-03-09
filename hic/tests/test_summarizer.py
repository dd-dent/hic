"""Tests for the SummarizerAgent."""
import re
import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock
from hypothesis import given, strategies as st, settings, HealthCheck
from hypothesis.stateful import RuleBasedStateMachine, rule, Bundle, initialize
from typing import List

from hic.agents.summarizer import SummarizerAgent, SummaryError

# Strategy for generating valid CHOFF states
choff_states = st.lists(
    st.builds(
        lambda name, weight: f"{{state:{name}[{weight:.1f}]}}",
        name=st.text(
            alphabet=st.characters(
                blacklist_categories=('Cs',),
                blacklist_characters=['[', ']', '{', '}', ':', ',']
            ),
            min_size=1,
            max_size=10
        ),
        weight=st.floats(min_value=0.1, max_value=1.0)
    ),
    min_size=1,
    max_size=3
)

# Strategy for generating messages with CHOFF states
message_strategy = st.builds(
    lambda states, text: " ".join(states + [text]),
    states=choff_states,
    text=st.text(min_size=1, max_size=200)
)

def create_mock_response(states: List[str], text: str) -> dict:
    """Create a mock response that preserves CHOFF states."""
    return {
        "content": [
            {
                "text": f"""Summary of messages:
{' '.join(states)}
{text}
[context:technical]"""
            }
        ],
        "usage": {"input_tokens": 150, "output_tokens": 50}
    }

@pytest.fixture
def mock_client():
    """Create a mock Claude client."""
    client = AsyncMock()
    # Response will be set in individual tests
    return client

@pytest.fixture
def summarizer(mock_client, tmp_path):
    """Create a SummarizerAgent instance with mocked client."""
    cache_dir = tmp_path / "test_cache"
    cache_dir.mkdir(exist_ok=True)
    return SummarizerAgent(
        client=mock_client,
        cache_dir=cache_dir,
        batch_size=2,
        timeout=5.0  # Add timeout for tests
    )

@pytest.mark.asyncio
async def test_summarize_batch(summarizer, mock_client):
    """Test basic batch summarization functionality."""
    messages = [
        "{state:analytical[0.8]} Let's discuss database scaling options.",
        "{state:analytical[0.7]} I suggest we implement cache invalidation."
    ]
    
    mock_client.create_message.return_value = create_mock_response(
        ["{state:analytical[0.75]}"],
        "Discussion on database scaling and cache invalidation"
    )
    
    summary = await summarizer.summarize_messages(messages)
    
    assert "{state:analytical" in summary
    assert "[context:technical]" in summary
    assert "database scaling" in summary.lower()
    assert "cache invalidation" in summary.lower()

@pytest.mark.asyncio
async def test_empty_messages(summarizer):
    """Test handling of empty message list."""
    with pytest.raises(SummaryError, match="No messages to summarize"):
        await summarizer.summarize_messages([])

def test_batch_size_validation():
    """Test batch size validation on initialization."""
    with pytest.raises(ValueError, match="Batch size must be positive"):
        SummarizerAgent(Mock(), batch_size=0)

@pytest.mark.asyncio
@given(messages=st.lists(message_strategy, min_size=1, max_size=5))
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
async def test_choff_state_preservation(tmp_path, messages):
    """Test that CHOFF states are preserved in summaries."""
    cache_dir = tmp_path / "test_cache"
    cache_dir.mkdir(exist_ok=True)
    
    # Extract states from messages for mock response
    all_states = set()
    for msg in messages:
        states = re.findall(r'\{state:[^}]+\}', msg)
        all_states.update(states)
    
    client = AsyncMock()
    client.create_message.return_value = create_mock_response(
        list(all_states),
        "Test summary content"
    )
    
    summarizer = SummarizerAgent(
        client=client,
        cache_dir=cache_dir,
        batch_size=2,
        timeout=5.0
    )
    
    summary = await summarizer.summarize_messages(messages)
    
    # Verify states in summary
    summary_states = set(state.state_type for state in summarizer._extract_choff_states(summary))
    original_states = set()
    for msg in messages:
        original_states.update(state.state_type for state in summarizer._extract_choff_states(msg))
    
    assert summary_states.intersection(original_states), "Summary should preserve at least one original state"

@pytest.mark.asyncio
@given(messages=st.lists(message_strategy, min_size=1, max_size=5))
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
async def test_summary_scoring(tmp_path, messages):
    """Test summary relevance scoring with property-based testing."""
    cache_dir = tmp_path / "test_cache"
    cache_dir.mkdir(exist_ok=True)
    
    client = AsyncMock()
    client.create_message.return_value = create_mock_response(
        ["{state:test[1.0]}"],
        "Test summary"
    )
    
    summarizer = SummarizerAgent(
        client=client,
        cache_dir=cache_dir,
        batch_size=2,
        timeout=5.0
    )
    
    summary = await summarizer.summarize_messages(messages)
    score = await summarizer.score_summary(summary, messages)
    
    assert 0 <= score <= 1.0
    assert isinstance(score, float)

@pytest.mark.asyncio
@given(messages=st.lists(message_strategy, min_size=3, max_size=10))
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
async def test_large_batch_processing(tmp_path, messages):
    """Test processing of messages larger than batch size."""
    cache_dir = tmp_path / "test_cache"
    cache_dir.mkdir(exist_ok=True)
    
    client = AsyncMock()
    client.create_message.return_value = create_mock_response(
        ["{state:test[1.0]}"],
        "Test batch summary"
    )
    
    summarizer = SummarizerAgent(
        client=client,
        cache_dir=cache_dir,
        batch_size=2,
        timeout=5.0
    )
    
    summary = await summarizer.summarize_messages(messages)
    assert "Summary" in summary
    assert len(summary.split('\n')) > 1

@pytest.mark.asyncio
async def test_invalid_message_format(summarizer):
    """Test handling of messages without CHOFF markup."""
    messages = ["Plain message without CHOFF", "Another plain message"]
    
    with pytest.raises(SummaryError, match="Missing CHOFF markup"):
        await summarizer.summarize_messages(messages)

@pytest.mark.asyncio
async def test_timeout_handling(tmp_path):
    """Test timeout handling in summarization."""
    cache_dir = tmp_path / "test_cache"
    cache_dir.mkdir(exist_ok=True)
    
    client = AsyncMock()
    # Simulate slow response
    async def slow_response(*args, **kwargs):
        await asyncio.sleep(2.0)
        return create_mock_response(["{state:test[1.0]}"], "Test summary")
    client.create_message.side_effect = slow_response
    
    summarizer = SummarizerAgent(
        client=client,
        cache_dir=cache_dir,
        batch_size=2,
        timeout=1.0  # Short timeout
    )
    
    messages = ["{state:test[1.0]} Test message"]
    with pytest.raises(asyncio.TimeoutError):
        await summarizer.summarize_messages(messages)

class SummarizerStateMachine(RuleBasedStateMachine):
    """State machine for testing summarizer behavior."""
    
    # Bundles for managing test state
    messages = Bundle('messages')
    summaries = Bundle('summaries')
    
    def __init__(self):
        super().__init__()
        # Setup only non-test state in __init__
        self.cache_dir = Path("test_cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.client = AsyncMock()
        self.client.create_message.return_value = create_mock_response(
            ["{state:test[1.0]}"],
            "Test summary"
        )
        self.summarizer = SummarizerAgent(
            client=self.client,
            cache_dir=self.cache_dir,
            batch_size=2,
            timeout=5.0
        )
    
    def teardown(self):
        """Clean up temporary files."""
        try:
            self.cache_dir.rmdir()
        except OSError:
            pass
        super().teardown()
    
    @initialize(target=messages)
    def init_messages(self):
        """Initialize empty message list."""
        return []
    
    @initialize(target=summaries)
    def init_summaries(self):
        """Initialize summary list."""
        return []
    
    @rule(target=messages, message=message_strategy)
    def add_message(self, message):
        """Add a message to the list."""
        return [message]
    
    @rule(target=summaries, messages=messages)
    def summarize(self, messages):
        """Try to summarize current messages."""
        if not messages:
            with pytest.raises(SummaryError):
                asyncio.run(self.summarizer.summarize_messages(messages))
            return []
            
        summary = asyncio.run(self.summarizer.summarize_messages(messages))
        assert isinstance(summary, str)
        assert len(summary) > 0
        return [summary]

# Simple TestCase definition as recommended
TestSummarizer = SummarizerStateMachine.TestCase