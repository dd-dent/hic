"""Property-based tests for the event store."""
import pytest
from pathlib import Path
from hypothesis import given, strategies as st, settings, HealthCheck
from hypothesis.stateful import RuleBasedStateMachine, rule
from contextlib import contextmanager
import tempfile
import os

from hic.events.schema import (
    EventType,
    ErrorSeverity,
    MessageReceivedEvent,
    CHOFFStateUpdatedEvent,
    ErrorOccurredEvent,
    CHOFFState
)
from hic.events.store import EventStore, EventStoreError

# Test data generation strategies
@st.composite
def choff_states(draw):
    """Generate valid CHOFF states."""
    return CHOFFState(
        notation=draw(st.text(min_size=1)),
        components=draw(st.dictionaries(
            keys=st.text(min_size=1),
            values=st.one_of(
                st.text(),
                st.integers(),
                st.floats(allow_nan=False, allow_infinity=False)
            )
        ))
    )

@st.composite
def message_events(draw):
    """Generate valid MessageReceivedEvents."""
    return MessageReceivedEvent.create(
        content=draw(st.text(min_size=1)),
        source=draw(st.text(min_size=1)),
        metadata=draw(st.dictionaries(
            keys=st.text(min_size=1),
            values=st.text()
        )),
        correlation_id=draw(st.none() | st.text(min_size=1))
    )

@st.composite
def choff_state_events(draw):
    """Generate valid CHOFFStateUpdatedEvents."""
    return CHOFFStateUpdatedEvent.create(
        new_state=draw(choff_states()),
        previous_state=draw(st.none() | choff_states()),
        transition_metadata=draw(st.dictionaries(
            keys=st.text(min_size=1),
            values=st.text()
        )),
        correlation_id=draw(st.none() | st.text(min_size=1))
    )

@st.composite
def error_events(draw):
    """Generate valid ErrorOccurredEvents."""
    return ErrorOccurredEvent.create(
        error_type=draw(st.text(min_size=1)),
        message=draw(st.text(min_size=1)),
        severity=draw(st.sampled_from(list(ErrorSeverity))),
        stack_trace=draw(st.none() | st.text()),
        context=draw(st.dictionaries(
            keys=st.text(min_size=1),
            values=st.text()
        )),
        correlation_id=draw(st.none() | st.text(min_size=1))
    )

@contextmanager
def temp_db():
    """Create a temporary database file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    try:
        yield EventStore(db_path)
    finally:
        try:
            os.unlink(db_path)
        except OSError:
            pass

@pytest.mark.trio
async def test_store_initialization():
    """Test store initialization creates required tables."""
    with temp_db() as store:
        event = MessageReceivedEvent.create(
            content="test",
            source="test"
        )
        await store.append(event)
        retrieved = await store.get_by_id(event.event_id)
        assert retrieved.content == "test"

@pytest.mark.trio
@given(message_events())
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
async def test_message_event_roundtrip(event):
    """Test MessageReceivedEvent serialization roundtrip."""
    with temp_db() as store:
        await store.append(event)
        retrieved = await store.get_by_id(event.event_id)
        
        assert retrieved.event_id == event.event_id
        assert retrieved.content == event.content
        assert retrieved.source == event.source
        assert retrieved.metadata == event.metadata
        assert retrieved.correlation_id == event.correlation_id

@pytest.mark.trio
@given(choff_state_events())
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
async def test_choff_state_event_roundtrip(event):
    """Test CHOFFStateUpdatedEvent serialization roundtrip."""
    with temp_db() as store:
        await store.append(event)
        retrieved = await store.get_by_id(event.event_id)
        
        assert retrieved.event_id == event.event_id
        assert retrieved.new_state.notation == event.new_state.notation
        assert retrieved.new_state.components == event.new_state.components
        if event.previous_state:
            assert retrieved.previous_state.notation == event.previous_state.notation
            assert retrieved.previous_state.components == event.previous_state.components
        assert retrieved.transition_metadata == event.transition_metadata
        assert retrieved.correlation_id == event.correlation_id

@pytest.mark.trio
@given(error_events())
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
async def test_error_event_roundtrip(event):
    """Test ErrorOccurredEvent serialization roundtrip."""
    with temp_db() as store:
        await store.append(event)
        retrieved = await store.get_by_id(event.event_id)
        
        assert retrieved.event_id == event.event_id
        assert retrieved.error_type == event.error_type
        assert retrieved.message == event.message
        assert retrieved.severity == event.severity
        assert retrieved.stack_trace == event.stack_trace
        assert retrieved.context == event.context
        assert retrieved.correlation_id == event.correlation_id

@pytest.mark.trio
async def test_invalid_event_id():
    """Test retrieving non-existent event."""
    with temp_db() as store:
        try:
            await store.get_by_id("nonexistent")
            pytest.fail("Expected EventStoreError")
        except ExceptionGroup as e:
            # Navigate through nested exception groups
            exc = e.exceptions[0]
            while isinstance(exc, ExceptionGroup):
                exc = exc.exceptions[0]
            assert isinstance(exc, EventStoreError)
            assert str(exc) == "Event not found: nonexistent"

@pytest.mark.trio
async def test_invalid_event_type():
    """Test handling invalid event types."""
    with temp_db() as store:
        with pytest.raises(ValueError):
            await store.get_by_type("invalid_type")

@pytest.mark.trio
async def test_correlation_id_filtering():
    """Test filtering events by correlation ID."""
    with temp_db() as store:
        events = [
            MessageReceivedEvent.create(
                content="test1",
                source="test",
                correlation_id="corr1"
            ),
            MessageReceivedEvent.create(
                content="test2",
                source="test",
                correlation_id="corr2"
            )
        ]
        
        for event in events:
            await store.append(event)
        
        filtered = await store.get_by_type(
            EventType.MESSAGE_RECEIVED,
            correlation_id="corr1"
        )
        assert len(filtered) == 1
        assert filtered[0].content == "test1"

@pytest.mark.trio
@given(st.lists(
    st.one_of(message_events(), choff_state_events(), error_events()),
    min_size=1,
    max_size=10
))
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
async def test_event_store_operations(events):
    """Test comprehensive event store operations."""
    with temp_db() as store:
        # Add all events
        event_map = {}
        for event in events:
            await store.append(event)
            event_map[event.event_id] = event

        # Verify each event can be retrieved
        for event_id, original in event_map.items():
            retrieved = await store.get_by_id(event_id)
            assert retrieved.event_id == original.event_id
            assert retrieved.event_type == original.event_type
            assert retrieved.version == original.version
            assert retrieved.correlation_id == original.correlation_id

        # Verify type-based queries
        for event_type in EventType:
            stored_events = await store.get_by_type(event_type)
            expected = [e for e in event_map.values() if e.event_type == event_type]
            assert len(stored_events) == len(expected)