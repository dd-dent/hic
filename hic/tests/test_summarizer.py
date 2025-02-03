"""Tests for the SummarizerAgent."""
import re
import pytest
from pathlib import Path
from unittest.mock import Mock
from hypothesis import given, strategies as st, settings, HealthCheck

from hic.agents.summarizer import SummarizerAgent, SummaryError

# Strategy for generating CHOFF states
choff_states = st.lists(
    st.text(
        alphabet=st.characters(blacklist_categories=('Cs',)),
        min_size=1
    ).map(lambda s: f"{{state:{s}}}"),
    min_size=1,  # At least one state required
    max_size=3
)

# Strategy for generating messages with CHOFF states
message_strategy = st.builds(
    lambda states, text: " ".join(states + [text]),
    states=choff_states,
    text=st.text(min_size=1, max_size=200)
)

def create_mock_response(states, text):
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
    client = Mock()
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
        batch_size=2
    )

def test_summarize_batch(summarizer, mock_client):
    """Test basic batch summarization functionality."""
    messages = [
        "{state:analytical} Let's discuss database scaling options.",
        "{state:analytical} I suggest we implement cache invalidation."
    ]
    
    mock_client.create_message.return_value = create_mock_response(
        ["{state:analytical}"],
        "Discussion on database scaling and cache invalidation"
    )
    
    summary = summarizer.summarize_messages(messages)
    
    assert "{state:analytical}" in summary
    assert "[context:technical]" in summary
    assert "database scaling" in summary.lower()
    assert "cache invalidation" in summary.lower()

def test_empty_messages(summarizer):
    """Test handling of empty message list."""
    with pytest.raises(SummaryError, match="No messages to summarize"):
        summarizer.summarize_messages([])

def test_batch_size_validation():
    """Test batch size validation on initialization."""
    with pytest.raises(ValueError, match="Batch size must be positive"):
        SummarizerAgent(Mock(), batch_size=0)

@given(messages=st.lists(message_strategy, min_size=1, max_size=5))
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_choff_state_preservation(tmp_path, messages):
    """Test that CHOFF states are preserved in summaries."""
    cache_dir = tmp_path / "test_cache"
    cache_dir.mkdir(exist_ok=True)
    
    # Extract states from messages for mock response
    all_states = set()
    for msg in messages:
        states = re.findall(r'\{state:[^}]+\}', msg)
        all_states.update(states)
    
    client = Mock()
    client.create_message.return_value = create_mock_response(
        list(all_states),
        "Test summary content"
    )
    
    summarizer = SummarizerAgent(
        client=client,
        cache_dir=cache_dir,
        batch_size=2
    )
    
    summary = summarizer.summarize_messages(messages)
    
    # Verify states in summary
    summary_states = set(summarizer._extract_choff_states(summary))
    assert summary_states.intersection(all_states), "Summary should preserve at least one original state"

@given(messages=st.lists(message_strategy, min_size=1, max_size=5))
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_summary_scoring(tmp_path, messages):
    """Test summary relevance scoring with property-based testing."""
    cache_dir = tmp_path / "test_cache"
    cache_dir.mkdir(exist_ok=True)
    
    client = Mock()
    client.create_message.return_value = create_mock_response(
        ["{state:test}"],
        "Test summary"
    )
    
    summarizer = SummarizerAgent(
        client=client,
        cache_dir=cache_dir,
        batch_size=2
    )
    
    summary = summarizer.summarize_messages(messages)
    score = summarizer.score_summary(summary, messages)
    
    assert 0 <= score <= 1.0
    assert isinstance(score, float)

@given(messages=st.lists(message_strategy, min_size=3, max_size=10))
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_large_batch_processing(tmp_path, messages):
    """Test processing of messages larger than batch size."""
    cache_dir = tmp_path / "test_cache"
    cache_dir.mkdir(exist_ok=True)
    
    client = Mock()
    client.create_message.return_value = create_mock_response(
        ["{state:test}"],
        "Test batch summary"
    )
    
    summarizer = SummarizerAgent(
        client=client,
        cache_dir=cache_dir,
        batch_size=2
    )
    
    summary = summarizer.summarize_messages(messages)
    assert "Summary" in summary
    assert len(summary.split('\n')) > 1

def test_invalid_message_format(summarizer):
    """Test handling of messages without CHOFF markup."""
    messages = ["Plain message without CHOFF", "Another plain message"]
    
    with pytest.raises(SummaryError, match="Missing CHOFF markup"):
        summarizer.summarize_messages(messages)

@given(batch_size=st.integers(min_value=1, max_value=10))
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_batch_size_property(tmp_path, batch_size):
    """Test that any positive batch size is valid."""
    cache_dir = tmp_path / "test_cache"
    cache_dir.mkdir(exist_ok=True)
    
    client = Mock()
    summarizer = SummarizerAgent(
        client=client,
        cache_dir=cache_dir,
        batch_size=batch_size
    )
    assert summarizer.batch_size == batch_size