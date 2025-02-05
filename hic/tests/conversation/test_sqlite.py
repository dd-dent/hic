"""Tests for SQLite-based conversation manager implementation."""
import pytest
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import shutil
from typing import List

from hypothesis import given, strategies as st, settings, HealthCheck
from hypothesis.stateful import RuleBasedStateMachine, rule, initialize

from hic.conversation.bus import InMemoryEventBus
from hic.conversation.events import EventType, MessageEvent, StateEvent, SummaryEvent
from hic.conversation.sqlite import SQLiteConversationManager
from hic.events.store import EventStore
from hic.message_store import MessageStore
from hic.agents.summarizer import SummarizerAgent

# Test strategies
message_strategy = st.builds(
    dict,
    content=st.text(min_size=1, max_size=200),
    role=st.sampled_from(["user", "assistant"]),
    metadata=st.dictionaries(
        keys=st.text(min_size=1),
        values=st.text(),
        max_size=3
    )
)

choff_states = st.dictionaries(
    keys=st.just("state"),  # Make this deterministic
    values=st.text(min_size=1, max_size=10).map(lambda s: f"{{{s}}}"),
    min_size=1,
    max_size=1  # Only one state at a time
)

class MockSummarizer(SummarizerAgent):
    """Mock summarizer for testing."""
    def __init__(self):
        self.summaries = {}
    
    async def summarize_messages(self, messages: List[str]) -> str:
        summary = f"Summary of {len(messages)} messages"
        self.summaries[summary] = messages
        return summary

@pytest.fixture(autouse=True)
def clean_db():
    """Create and clean up test database for each test."""
    test_dir = tempfile.mkdtemp()
    yield test_dir
    shutil.rmtree(test_dir)

@pytest.fixture
def event_bus():
    """Create a fresh event bus."""
    return InMemoryEventBus()

@pytest.fixture
def event_store(clean_db):
    """Create a fresh event store."""
    return EventStore(str(Path(clean_db) / "events.db"))

@pytest.fixture
def message_store(clean_db):
    """Create a fresh message store."""
    return MessageStore(str(Path(clean_db) / "messages.db"))

@pytest.fixture
def summarizer():
    """Create a mock summarizer."""
    return MockSummarizer()

@pytest.fixture
def manager(event_bus, event_store, message_store, summarizer, clean_db):
    """Create a fresh SQLiteConversationManager."""
    manager = SQLiteConversationManager(
        event_bus=event_bus,
        event_store=event_store,
        message_store=message_store,
        summarizer=summarizer,
        db_path=clean_db
    )
    # Ensure clean state
    manager._clear_storage()
    return manager

def test_initialization(clean_db):
    """Test manager initialization with default and custom paths."""
    # Default initialization
    manager = SQLiteConversationManager(InMemoryEventBus())
    assert isinstance(manager.event_store, EventStore)
    assert isinstance(manager.message_store, MessageStore)
    
    # Custom path initialization
    manager = SQLiteConversationManager(InMemoryEventBus(), db_path=clean_db)
    assert str(Path(manager.event_store._db_path).parent) == clean_db
    assert str(Path(manager.message_store._db_path).parent) == clean_db

@given(message=message_strategy)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_add_message(clean_db, message):
    """Test adding a message with event sourcing."""
    manager = SQLiteConversationManager(InMemoryEventBus(), db_path=clean_db)
    events = []
    manager.event_bus.subscribe(lambda e: events.append(e))
    
    message_id = manager.add_message(**message)
    stored = manager.get_message(message_id)
    
    assert stored is not None
    assert stored["content"] == message["content"]
    assert stored["role"] == message["role"]
    
    # Verify event was published
    assert len(events) == 1
    assert isinstance(events[0], MessageEvent)
    assert events[0].type == EventType.MESSAGE_ADDED
    assert events[0].message_id == message_id

@given(message=message_strategy)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_delete_message(clean_db, message):
    """Test deleting a message with event sourcing."""
    manager = SQLiteConversationManager(InMemoryEventBus(), db_path=clean_db)
    events = []
    manager.event_bus.subscribe(lambda e: events.append(e))
    
    message_id = manager.add_message(**message)
    assert manager.delete_message(message_id)
    assert manager.get_message(message_id) is None
    
    # Verify deletion event was published
    assert len(events) == 2  # Add + Delete
    assert events[1].type == EventType.MESSAGE_DELETED
    assert events[1].message_id == message_id

@given(state=choff_states)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_choff_state_tracking(clean_db, state):
    """Test CHOFF state tracking with event sourcing."""
    manager = SQLiteConversationManager(InMemoryEventBus(), db_path=clean_db)
    events = []
    manager.event_bus.subscribe(lambda e: events.append(e))
    
    manager.update_choff_state(state, "test")
    assert manager.get_choff_state() == state
    
    transitions = manager.get_choff_transitions()
    assert len(transitions) == 1
    assert transitions[0]["new"] == state
    assert transitions[0]["type"] == "test"
    
    # Verify state event was published
    assert len(events) == 1
    assert isinstance(events[0], StateEvent)
    assert events[0].type == EventType.STATE_CHANGED
    assert events[0].new_state == state

@given(messages=st.lists(message_strategy, min_size=1, max_size=5))
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_batch_message_processing(clean_db, messages):
    """Test processing multiple messages with event sourcing."""
    manager = SQLiteConversationManager(InMemoryEventBus(), db_path=clean_db)
    events = []
    manager.event_bus.subscribe(lambda e: events.append(e))
    
    message_ids = []
    for message in messages:
        message_id = manager.add_message(**message)
        message_ids.append(message_id)
    
    stored_messages = manager.get_messages()
    assert len(stored_messages) == len(messages)
    
    # Verify all messages were stored correctly
    for msg_id, original in zip(message_ids, messages):
        stored = manager.get_message(msg_id)
        assert stored["content"] == original["content"]
        assert stored["role"] == original["role"]
    
    # Verify events were published
    assert len(events) == len(messages)
    for event in events:
        assert isinstance(event, MessageEvent)
        assert event.type == EventType.MESSAGE_ADDED

@pytest.mark.trio
async def test_summary_generation(manager):
    """Test summary generation with event sourcing."""
    events = []
    manager.event_bus.subscribe(lambda e: events.append(e))
    
    # Add some messages
    messages = [
        {"content": "Hello", "role": "user"},
        {"content": "Hi there", "role": "assistant"},
        {"content": "How are you?", "role": "user"}
    ]
    message_ids = [manager.add_message(**msg) for msg in messages]
    
    # Request summary
    summary_id = manager.request_summary(message_ids=message_ids)
    summary = await manager.get_summary(summary_id)
    
    assert summary is not None
    assert summary["status"] == "completed"
    assert "Summary of 3 messages" in summary["content"]
    
    # Verify summary events were published
    summary_events = [e for e in events if isinstance(e, SummaryEvent)]
    assert len(summary_events) == 2  # Request + Generated
    assert summary_events[0].type == EventType.SUMMARY_REQUESTED
    assert summary_events[0].summary_id == summary_id
    assert summary_events[1].type == EventType.SUMMARY_GENERATED

def test_message_filtering(manager):
    """Test message filtering by time and role."""
    # Add messages with different roles and times
    messages = [
        {"content": "User 1", "role": "user"},
        {"content": "Assistant 1", "role": "assistant"},
        {"content": "User 2", "role": "user"}
    ]
    for msg in messages:
        manager.add_message(**msg)
    
    # Test role filtering
    user_messages = manager.get_messages(role="user")
    assert len(user_messages) == 2
    assert all(msg["role"] == "user" for msg in user_messages)
    
    assistant_messages = manager.get_messages(role="assistant")
    assert len(assistant_messages) == 1
    assert all(msg["role"] == "assistant" for msg in assistant_messages)
    
    # Test time filtering
    now = datetime.now(timezone.utc)
    recent_messages = manager.get_messages(start_time=now)
    assert len(recent_messages) == 0  # All messages were added before now

def test_error_handling(manager):
    """Test error handling in various scenarios."""
    # Test invalid message ID
    assert manager.get_message("invalid_id") is None
    assert not manager.delete_message("invalid_id")
    
    # Test summary without messages
    with pytest.raises(ValueError):
        manager.request_summary()
    
    # Test summary without summarizer
    manager.summarizer = None
    message_id = manager.add_message("Test", "user")
    with pytest.raises(RuntimeError):
        manager.request_summary(message_ids=[message_id])

class SQLiteConversationStateMachine(RuleBasedStateMachine):
    """State machine for testing SQLiteConversationManager."""
    
    def __init__(self):
        super().__init__()
        self.test_dir = None
        self.event_bus = None
        self.manager = None
        self.messages = {}
        self.states = []
    
    @initialize()
    def init_state(self):
        """Initialize state for each test run."""
        self.test_dir = tempfile.mkdtemp()
        self.event_bus = InMemoryEventBus()
        self.manager = SQLiteConversationManager(
            event_bus=self.event_bus,
            db_path=self.test_dir
        )
        self.messages.clear()
        self.states.clear()
    
    def teardown(self):
        """Clean up resources."""
        if self.test_dir:
            shutil.rmtree(self.test_dir)
            self.test_dir = None
        self.manager = None
        self.event_bus = None
        self.messages.clear()
        self.states.clear()
    
    @rule(message=message_strategy)
    def add_message(self, message):
        """Add a message and verify it's stored correctly."""
        message_id = self.manager.add_message(**message)
        stored = self.manager.get_message(message_id)
        assert stored is not None
        assert stored["content"] == message["content"]
        assert stored["role"] == message["role"]
        self.messages[message_id] = message
    
    @rule(state=choff_states)
    def update_state(self, state):
        """Update CHOFF state and verify tracking."""
        self.manager.update_choff_state(state, "test")
        current_state = self.manager.get_choff_state()
        assert current_state == state
        self.states.append(state)
    
    @rule()
    def delete_random_message(self):
        """Delete a random message if any exist."""
        if not self.messages:
            return
            
        message_id = next(iter(self.messages))
        assert self.manager.delete_message(message_id)
        assert self.manager.get_message(message_id) is None
        del self.messages[message_id]
    
    @rule()
    def verify_message_count(self):
        """Verify message count matches our tracking."""
        stored_messages = self.manager.get_messages()
        assert len(stored_messages) == len(self.messages)
    
    @rule()
    def verify_state_consistency(self):
        """Verify CHOFF state is consistent."""
        if self.states:
            current_state = self.manager.get_choff_state()
            assert current_state == self.states[-1]

TestSQLiteConversationManager = SQLiteConversationStateMachine.TestCase