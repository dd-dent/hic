"""Property-based tests for the CHOFF-aware event store."""
import pytest
from pathlib import Path
from hypothesis import given, strategies as st, settings, HealthCheck
from hypothesis.stateful import RuleBasedStateMachine, rule
from contextlib import contextmanager
import tempfile
import os
import uuid
from datetime import datetime, timezone

from hic.events.schema import (
    MessageEvent,
    StateEvent,
    ErrorEvent,
    ErrorSeverity,
    EventMetadata
)
from hic.events.store import EventStore, EventStoreError

# Test data generation strategies
@st.composite
def valid_timestamps(draw):
    """Generate valid timestamps within SQLite INTEGER bounds."""
    # SQLite INTEGER range is -2^63 to 2^63-1
    # We'll use Unix timestamps within reasonable bounds
    return draw(st.integers(
        min_value=int(datetime(2000, 1, 1).timestamp()),
        max_value=int(datetime(2100, 1, 1).timestamp())
    ))

@st.composite
def event_metadata(draw):
    """Generate valid event metadata."""
    return EventMetadata(
        event_id=str(uuid.uuid4()),
        timestamp=draw(valid_timestamps()),
        conversation_id=draw(st.text(min_size=1, max_size=36)),
        correlation_id=draw(st.none() | st.text(min_size=1, max_size=36)),
        version="1.0"
    )

@st.composite
def message_events(draw):
    """Generate valid MessageEvents."""
    return MessageEvent(
        metadata=draw(event_metadata()),
        content=draw(st.text(min_size=1, max_size=1000)),
        source=draw(st.none() | st.text(min_size=1, max_size=100))
    )

@st.composite
def state_events(draw):
    """Generate valid StateEvents."""
    state_type = draw(st.text(min_size=1, max_size=50))
    return StateEvent(
        metadata=draw(event_metadata()),
        state_expression=state_type,  # For basic states, just use the state type string
        expression_type="basic",
        context=draw(st.none() | st.text(min_size=1, max_size=100))
    )

@st.composite
def error_events(draw):
    """Generate valid ErrorEvents."""
    return ErrorEvent(
        metadata=draw(event_metadata()),
        error_type=draw(st.text(min_size=1, max_size=50)),
        message=draw(st.text(min_size=1, max_size=500)),
        severity=draw(st.sampled_from(list(ErrorSeverity))),
        stack_trace=draw(st.none() | st.text(max_size=1000)),
        context=draw(st.dictionaries(
            keys=st.text(min_size=1, max_size=50),
            values=st.text(max_size=100),
            max_size=10
        ))
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

@pytest.mark.asyncio
async def test_store_initialization():
    """Test store initialization creates required tables."""
    with temp_db() as store:
        event = MessageEvent.create(
            content="test",
            conversation_id="test-conv"
        )
        await store.append(event)
        retrieved = await store.get_by_conversation(event.metadata.conversation_id)
        assert len(retrieved) == 1
        assert retrieved[0].content == "test"

@pytest.mark.asyncio
@given(message_events())
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
async def test_message_event_roundtrip(event):
    """Test MessageEvent serialization roundtrip."""
    with temp_db() as store:
        await store.append(event)
        retrieved = await store.get_by_conversation(event.metadata.conversation_id)
        assert len(retrieved) == 1
        retrieved_event = retrieved[0]
        
        assert retrieved_event.metadata.event_id == event.metadata.event_id
        assert retrieved_event.metadata.conversation_id == event.metadata.conversation_id
        assert retrieved_event.content == event.content
        assert retrieved_event.source == event.source

@pytest.mark.asyncio
@given(state_events())
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
async def test_state_event_roundtrip(event):
    """Test StateEvent serialization roundtrip."""
    with temp_db() as store:
        await store.append(event)
        retrieved = await store.get_by_conversation(event.metadata.conversation_id)
        assert len(retrieved) == 1
        retrieved_event = retrieved[0]
        
        assert retrieved_event.metadata.event_id == event.metadata.event_id
        assert retrieved_event.metadata.conversation_id == event.metadata.conversation_id
        assert retrieved_event.state_type == event.state_type
        assert retrieved_event.intensity == event.intensity
        assert retrieved_event.context == event.context

@pytest.mark.asyncio
@given(error_events())
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
async def test_error_event_roundtrip(event):
    """Test ErrorEvent serialization roundtrip."""
    with temp_db() as store:
        await store.append(event)
        retrieved = await store.get_by_conversation(event.metadata.conversation_id)
        assert len(retrieved) == 1
        retrieved_event = retrieved[0]
        
        assert retrieved_event.metadata.event_id == event.metadata.event_id
        assert retrieved_event.metadata.conversation_id == event.metadata.conversation_id
        assert retrieved_event.error_type == event.error_type
        assert retrieved_event.message == event.message
        assert retrieved_event.severity == event.severity
        assert retrieved_event.stack_trace == event.stack_trace
        assert retrieved_event.context == event.context

@pytest.mark.asyncio
async def test_conversation_retrieval():
    """Test retrieving events by conversation ID."""
    with temp_db() as store:
        conv_id = str(uuid.uuid4())
        events = [
            MessageEvent.create(
                content=f"test{i}",
                conversation_id=conv_id
            ) for i in range(3)
        ]
        
        for event in events:
            await store.append(event)
        
        retrieved = await store.get_by_conversation(conv_id)
        assert len(retrieved) == 3
        assert all(e.metadata.conversation_id == conv_id for e in retrieved)
        assert [e.content for e in retrieved] == ["test0", "test1", "test2"]

@pytest.mark.asyncio
async def test_conversation_limit():
    """Test limiting conversation event retrieval."""
    with temp_db() as store:
        conv_id = str(uuid.uuid4())
        events = [
            MessageEvent.create(
                content=f"test{i}",
                conversation_id=conv_id
            ) for i in range(5)
        ]
        
        for event in events:
            await store.append(event)
        
        limited = await store.get_by_conversation(conv_id, limit=3)
        assert len(limited) == 3
        assert all(e.metadata.conversation_id == conv_id for e in limited)

@pytest.mark.asyncio
@given(st.lists(
    st.one_of(message_events(), state_events(), error_events()),
    min_size=1,
    max_size=5
))
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
async def test_mixed_event_types(events):
    """Test handling mixed event types in conversations."""
    with temp_db() as store:
        conv_id = str(uuid.uuid4())
        
        # Store all events under same conversation
        for event in events:
            new_event = type(event)(
                metadata=EventMetadata(
                    event_id=str(uuid.uuid4()),
                    timestamp=int(datetime.now(timezone.utc).timestamp()),
                    conversation_id=conv_id,
                    correlation_id=event.metadata.correlation_id,
                    version=event.metadata.version
                ),
                **{k: v for k, v in event.__dict__.items() if k != 'metadata'}
            )
            await store.append(new_event)
        
        retrieved = await store.get_by_conversation(conv_id)
        assert len(retrieved) == len(events)
        
        # Verify chronological order
        timestamps = [e.metadata.timestamp for e in retrieved]
        assert timestamps == sorted(timestamps)