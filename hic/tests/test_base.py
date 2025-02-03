"""Tests for the base agent implementation."""
import pytest
from unittest.mock import Mock
from pathlib import Path
from hypothesis import given, strategies as st, settings, HealthCheck

from ..agents.base import BaseAgent, RetryError

# Strategy for generating CHOFF states
choff_states = st.lists(
    st.text(
        alphabet=st.characters(blacklist_categories=('Cs',)),
        min_size=1
    ).map(lambda s: f"{{state:{s}}}"),
    min_size=0,
    max_size=3
)

# Strategy for generating prompts with CHOFF states
prompt_strategy = st.builds(
    lambda states, text: " ".join(states + [text]),
    states=choff_states,
    text=st.text(min_size=1, max_size=200)
)

@pytest.fixture
def mock_claude_client():
    """Mock Claude API client for testing."""
    client = Mock()
    client.create_message = Mock(return_value={
        "id": "test_msg_id",
        "content": "Test response",
        "usage": {"input_tokens": 10, "output_tokens": 20}
    })
    return client

@pytest.fixture
def test_cache_dir(tmp_path):
    """Create a temporary directory for response caching."""
    cache_dir = tmp_path / "test_cache"
    cache_dir.mkdir()
    return cache_dir

@pytest.fixture
def base_agent(mock_claude_client, test_cache_dir):
    """Create a BaseAgent instance for testing."""
    return BaseAgent(
        client=mock_claude_client,
        cache_dir=test_cache_dir,
        system_prompt="Test system prompt"
    )

def test_agent_initialization(base_agent):
    """Test basic agent initialization."""
    assert base_agent.system_prompt == "Test system prompt"
    assert base_agent.max_retries == 3
    assert base_agent.base_delay == 1.0

@given(prompt=prompt_strategy)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_choff_prompt_handling(mock_claude_client, test_cache_dir, prompt):
    """Test CHOFF state handling in prompts."""
    # Create fresh agent for each test case
    agent = BaseAgent(
        client=mock_claude_client,
        cache_dir=test_cache_dir,
        system_prompt="Test system prompt"
    )
    
    response = agent.send_message(prompt)
    
    # Verify the prompt was passed correctly
    assert len(mock_claude_client.create_message.call_args_list) == 1
    call_args = mock_claude_client.create_message.call_args[1]
    assert prompt in call_args["messages"][0]["content"]
    
    # Reset mock for next test case
    mock_claude_client.create_message.reset_mock()

@given(states=choff_states)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_choff_state_preservation(mock_claude_client, test_cache_dir, states):
    """Test that CHOFF states are preserved in system prompts."""
    system_prompt = " ".join(states + ["Base system prompt"])
    agent = BaseAgent(
        client=mock_claude_client,
        system_prompt=system_prompt,
        cache_dir=test_cache_dir
    )
    
    for state in states:
        assert state in agent.system_prompt

def test_retry_logic(mock_claude_client, base_agent):
    """Test retry logic with exponential backoff."""
    mock_claude_client.create_message.side_effect = [
        Exception("API Error"),
        Exception("API Error"),
        {"id": "test_msg_id", "content": "Success", "usage": {"input_tokens": 10, "output_tokens": 20}}
    ]
    
    response = base_agent.send_message("Test prompt")
    assert response["content"] == "Success"
    assert mock_claude_client.create_message.call_count == 3

def test_max_retries_exceeded(mock_claude_client, base_agent):
    """Test that RetryError is raised when max retries are exceeded."""
    mock_claude_client.create_message.side_effect = Exception("API Error")
    
    with pytest.raises(RetryError):
        base_agent.send_message("Test prompt")
    
    assert mock_claude_client.create_message.call_count == base_agent.max_retries

def test_token_usage_monitoring(base_agent):
    """Test token usage tracking."""
    response = base_agent.send_message("Test prompt")
    
    assert base_agent.total_tokens == 30  # 10 input + 20 output
    assert base_agent.input_tokens == 10
    assert base_agent.output_tokens == 20

@given(prompts=st.lists(prompt_strategy, min_size=1, max_size=5))
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_token_accumulation(mock_claude_client, test_cache_dir, prompts):
    """Test that token counts accumulate correctly across multiple messages."""
    # Create fresh agent for each test case
    agent = BaseAgent(
        client=mock_claude_client,
        cache_dir=test_cache_dir,
        system_prompt="Test system prompt"
    )
    
    expected_total = 0
    for prompt in prompts:
        response = agent.send_message(prompt)
        expected_total += 30  # 10 input + 20 output per message
    
    assert agent.total_tokens == expected_total
    assert agent.input_tokens == expected_total // 3  # 1/3 of total (10 per message)
    assert agent.output_tokens == 2 * expected_total // 3  # 2/3 of total (20 per message)

def test_response_caching(base_agent, test_cache_dir):
    """Test response caching for identical prompts."""
    prompt = "Test prompt for caching"
    
    # First call should hit the API
    response1 = base_agent.send_message(prompt)
    assert base_agent.client.create_message.call_count == 1
    
    # Second call should use cache
    response2 = base_agent.send_message(prompt)
    assert base_agent.client.create_message.call_count == 1
    assert response1 == response2

def test_cache_invalidation(base_agent):
    """Test cache invalidation when force_refresh is True."""
    prompt = "Test prompt for cache invalidation"
    
    # First call
    response1 = base_agent.send_message(prompt)
    
    # Second call with force_refresh
    response2 = base_agent.send_message(prompt, force_refresh=True)
    
    assert base_agent.client.create_message.call_count == 2

def test_cache_directory_creation(tmp_path):
    """Test that cache directory is created if it doesn't exist."""
    cache_dir = tmp_path / "new_cache"
    agent = BaseAgent(
        client=Mock(),
        system_prompt="Test",
        cache_dir=cache_dir
    )
    assert cache_dir.exists()

def test_failed_cache_operations(base_agent, monkeypatch):
    """Test graceful handling of cache failures."""
    def mock_open(*args, **kwargs):
        raise OSError("Mock file error")
    
    monkeypatch.setattr(Path, "open", mock_open)
    
    # Should not raise exception on cache failure
    response = base_agent.send_message("Test prompt")
    assert response is not None