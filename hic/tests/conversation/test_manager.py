import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
from hypothesis.stateful import RuleBasedStateMachine, rule
from datetime import datetime, timezone
from typing import Dict, List, Optional
from contextlib import contextmanager

from hic.conversation.bus import EventBus, InMemoryEventBus
from hic.conversation.events import Event, EventType, MessageEvent, StateEvent, SummaryEvent
from hic.conversation.manager import ConversationManager

# Strategy for generating CHOFF states
choff_states = st.dictionaries(
    keys=st.text(min_size=1).map(lambda s: f"state"),
    values=st.text(min_size=1).map(lambda s: f"{{{s}}}"),
    min_size=1,
    max_size=3
)

# Strategy for generating messages
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

class MockConversationManager(ConversationManager):
    """Mock implementation for testing"""
    
    def __init__(self, event_bus: EventBus):
        super().__init__(event_bus)
        self._messages: Dict[str, Dict] = {}
        self._summaries: Dict[str, Dict] = {}
        self._message_counter = 0
        self._summary_counter = 0
    
    def add_message(self, content: str, role: str, metadata: Optional[Dict] = None) -> str:
        self._message_counter += 1
        message_id = f"msg_{self._message_counter}"
        
        message = {
            "id": message_id,
            "content": content,
            "role": role,
            "metadata": metadata or {},
            "timestamp": datetime.now(timezone.utc)
        }
        
        self._messages[message_id] = message
        
        event = MessageEvent(
            type=EventType.MESSAGE_ADDED,
            timestamp=datetime.now(timezone.utc),
            payload={},
            message_id=message_id,
            content=content,
            role=role
        )
        
        self.event_bus.publish(event)
        return message_id
    
    def get_message(self, message_id: str) -> Optional[Dict]:
        return self._messages.get(message_id)
    
    def delete_message(self, message_id: str) -> bool:
        if message_id in self._messages:
            del self._messages[message_id]
            return True
        return False
    
    def get_messages(self, 
                    start_time: Optional[datetime] = None,
                    end_time: Optional[datetime] = None,
                    role: Optional[str] = None) -> List[Dict]:
        messages = list(self._messages.values())
        
        if start_time:
            messages = [m for m in messages if m["timestamp"] >= start_time]
        if end_time:
            messages = [m for m in messages if m["timestamp"] <= end_time]
        if role:
            messages = [m for m in messages if m["role"] == role]
            
        return sorted(messages, key=lambda m: m["timestamp"])
    
    def request_summary(self,
                       message_ids: Optional[List[str]] = None,
                       start_time: Optional[datetime] = None,
                       end_time: Optional[datetime] = None) -> str:
        self._summary_counter += 1
        summary_id = f"sum_{self._summary_counter}"
        
        if message_ids is None:
            messages = self.get_messages(start_time, end_time)
            message_ids = [m["id"] for m in messages]
        
        summary = {
            "id": summary_id,
            "content": f"Summary of messages: {', '.join(message_ids)}",
            "message_ids": message_ids,
            "timestamp": datetime.now(timezone.utc)
        }
        
        self._summaries[summary_id] = summary
        
        event = SummaryEvent(
            type=EventType.SUMMARY_GENERATED,
            timestamp=datetime.now(timezone.utc),
            payload={},
            summary_id=summary_id,
            content=summary["content"],
            source_message_ids=message_ids
        )
        
        self.event_bus.publish(event)
        return summary_id
    
    def get_summary(self, summary_id: str) -> Optional[Dict]:
        return self._summaries.get(summary_id)

@contextmanager
def conversation_manager():
    """Create a fresh manager for testing."""
    event_bus = InMemoryEventBus()
    manager = MockConversationManager(event_bus)
    try:
        yield manager
    finally:
        # Clean up if needed
        pass

@given(message=message_strategy)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_add_message(message):
    """Test that we can add a message and retrieve it."""
    with conversation_manager() as manager:
        message_id = manager.add_message(**message)
        stored = manager.get_message(message_id)
        
        assert stored is not None
        assert stored["content"] == message["content"]
        assert stored["role"] == message["role"]
        assert len(manager._messages) == 1

@given(message=message_strategy)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_delete_message(message):
    """Test that we can delete a message."""
    with conversation_manager() as manager:
        message_id = manager.add_message(**message)
        assert manager.delete_message(message_id)
        assert manager.get_message(message_id) is None

@given(state=choff_states)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_choff_state_tracking(state):
    """Test CHOFF state tracking."""
    with conversation_manager() as manager:
        manager.update_choff_state(state, "test")
        assert manager.get_choff_state() == state
        
        transitions = manager.get_choff_transitions()
        assert len(transitions) == 1
        assert transitions[0]["new"] == state
        assert transitions[0]["type"] == "test"

@given(messages=st.lists(message_strategy, min_size=1, max_size=5))
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_batch_message_processing(messages):
    """Test processing multiple messages."""
    with conversation_manager() as manager:
        message_ids = []
        for message in messages:
            message_id = manager.add_message(**message)
            message_ids.append(message_id)
        
        stored_messages = manager.get_messages()
        assert len(stored_messages) == len(messages)
        
        for msg_id, original in zip(message_ids, messages):
            stored = manager.get_message(msg_id)
            assert stored["content"] == original["content"]
            assert stored["role"] == original["role"]

def test_event_publishing():
    """Test event publishing."""
    with conversation_manager() as manager:
        events = []
        
        def collect_events(event: Event):
            events.append(event)
        
        manager.event_bus.subscribe(collect_events)
        
        manager.add_message("Test", "user")
        assert len(events) == 1
        assert isinstance(events[0], MessageEvent)
        
        manager.update_choff_state({"state": "test"}, "test")
        assert len(events) == 2
        assert isinstance(events[1], StateEvent)

class ConversationStateMachine(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.event_bus = InMemoryEventBus()
        self.manager = MockConversationManager(self.event_bus)
        self.messages = {}  # Track added messages
        self.states = []  # Track state transitions
    
    def teardown(self):
        """Clean up resources."""
        self.messages.clear()
        self.states.clear()
    
    @rule(message=message_strategy)
    def add_message(self, message):
        """Add a message and verify it's stored correctly."""
        message_id = self.manager.add_message(**message)
        stored = self.manager.get_message(message_id)
        assert stored["content"] == message["content"]
        assert stored["role"] == message["role"]
        self.messages[message_id] = message
    
    @rule(state=choff_states)
    def update_state(self, state):
        """Update CHOFF state and verify tracking."""
        self.manager.update_choff_state(state, "test")
        assert self.manager.get_choff_state() == state
        self.states.append(state)
    
    @rule(data=st.data())
    def delete_message(self, data):
        """Delete a message if we have any."""
        if self.messages:
            message_id = data.draw(st.sampled_from(list(self.messages.keys())))
            self.manager.delete_message(message_id)
            del self.messages[message_id]
            assert self.manager.get_message(message_id) is None

TestConversationManager = ConversationStateMachine.TestCase