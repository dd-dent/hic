"""Tests for the RetrieverAgent."""
import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock, MagicMock
from hypothesis import given, strategies as st, settings, HealthCheck
from hypothesis.stateful import RuleBasedStateMachine, rule, Bundle, initialize
from typing import List
import json
from datetime import timezone, datetime

from hic.agents.retriever import RetrieverAgent, ScoredMessage, RetrievalError
from hic.message_store import Message, Speaker

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

# Strategy for generating messages
message_strategy = st.builds(
    Message,
    speaker=st.sampled_from(list(Speaker)),
    content=st.text(min_size=1, max_size=200),
    choff_tags=choff_states,
    timestamp=st.datetimes().map(lambda dt: dt.replace(tzinfo=timezone.utc))
)

def create_mock_response(score: float = 0.95, patterns: List[str] = None) -> dict:
    """Create a mock response for retrieval analysis."""
    patterns = patterns or ["{state:analytical}", "[context:technical]"]
    return {
        "content": [{
            "text": json.dumps({
                "message_idx": 0,
                "score": score,
                "patterns": patterns
            })
        }],
        "usage": {"input_tokens": 150, "output_tokens": 50}
    }

@pytest.fixture
def mock_client():
    """Create a mock Claude client."""
    client = AsyncMock()
    # Response will be set in individual tests
    return client

@pytest.fixture
def mock_store():
    """Create a mock MessageStore."""
    store = MagicMock()
    store.find_by_choff_tag = Mock(return_value=[])  # Default to empty list
    return store

@pytest.fixture
def retriever(mock_client, mock_store, tmp_path):
    """Create a RetrieverAgent instance with mocked dependencies."""
    cache_dir = tmp_path / "test_cache"
    cache_dir.mkdir(exist_ok=True)
    return RetrieverAgent(
        client=mock_client,
        message_store=mock_store,
        cache_dir=cache_dir,
        timeout=5.0
    )

@pytest.mark.asyncio
async def test_find_relevant(retriever, mock_client, mock_store):
    """Test basic retrieval functionality."""
    message = Message(
        speaker=Speaker.USER,
        content="Let's discuss recursion",
        choff_tags=["{state:curious[0.8]}", "[context:technical]"],
        timestamp=datetime.now(timezone.utc)
    )
    
    # Setup mock store to return our test message
    mock_store.find_by_choff_tag.return_value = [message]
    
    mock_client.create_message.return_value = create_mock_response()
    
    results = await retriever.find_relevant(
        query="recursion",
        choff_tags=["{state:curious}"]
    )
    
    assert len(results) > 0
    assert isinstance(results[0], ScoredMessage)
    assert 0 <= results[0].score <= 1
    assert len(results[0].matched_patterns) > 0
    mock_store.find_by_choff_tag.assert_called_with("{state:curious}")

@pytest.mark.asyncio
async def test_empty_query(retriever, mock_store):
    """Test handling of empty query."""
    results = await retriever.find_relevant("")
    assert len(results) == 0
    mock_store.find_by_choff_tag.assert_not_called()

@pytest.mark.asyncio
@given(messages=st.lists(message_strategy, min_size=1, max_size=5))
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
async def test_choff_pattern_analysis(tmp_path, messages):
    """Test CHOFF pattern analysis with property-based testing."""
    cache_dir = tmp_path / "test_cache"
    cache_dir.mkdir(exist_ok=True)
    
    client = AsyncMock()
    client.create_message.return_value = {
        "content": [{
            "text": json.dumps({
                "states": ["{state:test}"],
                "contexts": ["[context:technical]"],
                "patterns": ["&pattern:recursion|"]
            })
        }],
        "usage": {"input_tokens": 150, "output_tokens": 50}
    }
    
    store = MagicMock()
    store.find_by_choff_tag.return_value = messages
    
    retriever = RetrieverAgent(
        client=client,
        message_store=store,
        cache_dir=cache_dir,
        timeout=5.0
    )
    
    patterns = await retriever.analyze_choff_patterns(messages)
    assert isinstance(patterns, dict)
    assert all(isinstance(v, list) for v in patterns.values())

@pytest.mark.asyncio
async def test_timeout_handling(tmp_path):
    """Test timeout handling in retrieval."""
    cache_dir = tmp_path / "test_cache"
    cache_dir.mkdir(exist_ok=True)
    
    client = AsyncMock()
    # Simulate slow response
    async def slow_response(*args, **kwargs):
        await asyncio.sleep(2.0)
        return create_mock_response()
    client.create_message.side_effect = slow_response
    
    store = MagicMock()
    store.find_by_choff_tag.return_value = [
        Message(
            speaker=Speaker.USER,
            content="Test message",
            choff_tags=["{state:test}"],
            timestamp=datetime.now(timezone.utc)
        )
    ]
    
    retriever = RetrieverAgent(
        client=client,
        message_store=store,
        cache_dir=cache_dir,
        timeout=1.0  # Short timeout
    )
    
    with pytest.raises(RetrievalError) as exc_info:
        await retriever.find_relevant("test", ["{state:test}"])
    assert "Message send timed out" in str(exc_info.value)

class RetrieverStateMachine(RuleBasedStateMachine):
    """State machine for testing retriever behavior."""
    
    # Bundles for managing test state
    messages = Bundle('messages')
    queries = Bundle('queries')
    
    def __init__(self):
        super().__init__()
        # Setup only non-test state in __init__
        self.cache_dir = Path("test_cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.client = AsyncMock()
        self.client.create_message.return_value = create_mock_response()
        self.store = MagicMock()
        self.store.find_by_choff_tag.return_value = []
        self.retriever = RetrieverAgent(
            client=self.client,
            message_store=self.store,
            cache_dir=None,  # Disable caching for state machine tests
            timeout=5.0
        )
        # Create event loop for running async tasks
        self.loop = asyncio.new_event_loop()
    
    def teardown(self):
        """Clean up temporary files."""
        try:
            self.cache_dir.rmdir()
        except OSError:
            pass
        if hasattr(self, 'loop') and self.loop.is_running():
            self.loop.close()
        super().teardown()
    
    @initialize(target=messages)
    def init_messages(self):
        """Initialize empty message list."""
        return []
    
    @initialize(target=queries)
    def init_queries(self):
        """Initialize query list."""
        return []
    
    @rule(target=messages, message=message_strategy)
    def add_message(self, message):
        """Add a message to the store."""
        # Update mock store's behavior
        self.store.find_by_choff_tag.return_value = [message]
        return [message]
    
    @rule(target=queries, message=messages)
    def create_query(self, message):
        """Create a query from message tags."""
        if not message or not message[0].choff_tags:
            return []
        return [message[0].choff_tags[0]]
    
    @rule(query=queries, message=messages)
    def find_relevant(self, query, message):
        """Try to find relevant messages."""
        if not query or not message:
            return
        
        # Run the coroutine in the event loop
        results = self.loop.run_until_complete(
            self.retriever.find_relevant("test query", query)
        )
        
        assert isinstance(results, list)
        for result in results:
            assert isinstance(result, ScoredMessage)
            assert 0 <= result.score <= 1

# Simple TestCase definition as recommended
TestRetriever = RetrieverStateMachine.TestCase