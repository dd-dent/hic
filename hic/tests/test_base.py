"""Tests for the base agent implementation."""
import pytest
from unittest.mock import Mock, AsyncMock
from pathlib import Path
import trio
from hypothesis import given, strategies as st, settings, HealthCheck

from ..agents.base import BaseAgent, RetryError, NonRetryableError

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
    client.create_message = AsyncMock(return_value={
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
    assert base_agent.usage.total_tokens == 0

@pytest.mark.trio
async def test_choff_prompt_handling(mock_claude_client, test_cache_dir):
    """Test CHOFF state handling in prompts."""
    agent = BaseAgent(
        client=mock_claude_client,
        cache_dir=test_cache_dir,
        system_prompt="Test system prompt"
    )
    
    prompt = "{state:test} Hello world"
    response = await agent.send_message(prompt)
    
    # Verify the prompt was passed correctly
    assert len(mock_claude_client.create_message.call_args_list) == 1
    call_args = mock_claude_client.create_message.call_args[1]
    assert prompt in call_args["messages"][0]["content"]

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

@pytest.mark.trio
async def test_retry_logic(base_agent):
    """Test retry logic with exponential backoff."""
    base_agent.client.create_message.side_effect = [
        Exception("API Error"),
        Exception("API Error"),
        {"id": "test_msg_id", "content": "Success", "usage": {"input_tokens": 10, "output_tokens": 20}}
    ]
    
    response = await base_agent.send_message("Test prompt")
    assert response["content"] == "Success"
    assert base_agent.client.create_message.call_count == 3

@pytest.mark.trio
async def test_non_retryable_error(base_agent):
    """Test that NonRetryableError is not retried."""
    base_agent.client.create_message.side_effect = ValueError("Invalid input")
    
    with pytest.raises(NonRetryableError):
        await base_agent.send_message("Test prompt")
    
    assert base_agent.client.create_message.call_count == 1

@pytest.mark.trio
async def test_max_retries_exceeded(base_agent):
    """Test that RetryError is raised when max retries are exceeded."""
    base_agent.client.create_message.side_effect = Exception("API Error")
    
    with pytest.raises(RetryError):
        await base_agent.send_message("Test prompt")
    
    assert base_agent.client.create_message.call_count == base_agent.max_retries

@pytest.mark.trio
async def test_token_usage_monitoring(base_agent):
    """Test token usage tracking."""
    response = await base_agent.send_message("Test prompt")
    
    assert base_agent.usage.total_tokens == 30  # 10 input + 20 output
    assert base_agent.usage.input_tokens == 10
    assert base_agent.usage.output_tokens == 20

@pytest.mark.trio
@given(prompts=st.lists(prompt_strategy, min_size=1, max_size=5))
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
async def test_token_accumulation(mock_claude_client, test_cache_dir, prompts):
    """Test that token counts accumulate correctly across multiple messages."""
    agent = BaseAgent(
        client=mock_claude_client,
        cache_dir=test_cache_dir,
        system_prompt="Test system prompt"
    )
    
    expected_total = 0
    for prompt in prompts:
        await agent.send_message(prompt)
        expected_total += 30  # 10 input + 20 output per message
    
    assert agent.usage.total_tokens == expected_total
    assert agent.usage.input_tokens == expected_total // 3  # 1/3 of total (10 per message)
    assert agent.usage.output_tokens == 2 * expected_total // 3  # 2/3 of total (20 per message)

@pytest.mark.trio
async def test_response_caching(base_agent, test_cache_dir):
    """Test response caching for identical prompts."""
    prompt = "Test prompt for caching"
    
    # First call should hit the API
    response1 = await base_agent.send_message(prompt)
    assert base_agent.client.create_message.call_count == 1
    
    # Second call should use cache
    response2 = await base_agent.send_message(prompt)
    assert base_agent.client.create_message.call_count == 1
    assert response1 == response2

@pytest.mark.trio
async def test_cache_invalidation(base_agent):
    """Test cache invalidation when force_refresh is True."""
    prompt = "Test prompt for cache invalidation"
    
    # First call
    response1 = await base_agent.send_message(prompt)
    
    # Second call with force_refresh
    response2 = await base_agent.send_message(prompt, force_refresh=True)
    
    assert base_agent.client.create_message.call_count == 2

@pytest.mark.trio
async def test_timeout_handling(base_agent):
    """Test message timeout handling."""
    # Create a mock that sleeps longer than the timeout
    async def slow_response(*args, **kwargs):
        await trio.sleep(0.5)  # Sleep for 500ms
        return {"content": "too late", "usage": {"input_tokens": 10, "output_tokens": 20}}
    
    base_agent.client.create_message = AsyncMock(side_effect=slow_response)
    
    # Set a short timeout (100ms)
    with pytest.raises(trio.TooSlowError):
        await base_agent.send_message("Test prompt", timeout=0.1)

@pytest.mark.trio
async def test_atomic_cache_operations(base_agent, test_cache_dir, monkeypatch):
    """Test atomic cache write operations."""
    prompt = "Test atomic cache"
    
    # Mock write_text to fail after temp file creation
    orig_write_text = Path.write_text
    write_called = False
    
    def mock_write_text(self, content):
        nonlocal write_called
        if write_called:
            raise OSError("Write failed")
        write_called = True
        return orig_write_text(self, content)
    
    monkeypatch.setattr(Path, "write_text", mock_write_text)
    
    # Should handle failed write gracefully
    await base_agent.send_message(prompt)
    
    # Temp file should be cleaned up
    temp_files = list(test_cache_dir.glob("*.tmp"))
    assert len(temp_files) == 0

@pytest.mark.trio
async def test_invalid_usage_data(base_agent):
    """Test handling of invalid usage data."""
    base_agent.client.create_message.return_value = {
        "id": "test_msg_id",
        "content": "Test response",
        "usage": None  # Invalid usage data
    }
    
    await base_agent.send_message("Test prompt")
    assert base_agent.usage.total_tokens == 0  # Should handle gracefully

def test_cache_directory_creation(tmp_path):
    """Test that cache directory is created if it doesn't exist."""
    cache_dir = tmp_path / "new_cache"
    agent = BaseAgent(
        client=Mock(),
        system_prompt="Test",
        cache_dir=cache_dir
    )
    assert cache_dir.exists()