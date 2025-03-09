import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
from hypothesis.stateful import RuleBasedStateMachine, rule, Bundle, initialize
from datetime import datetime, timezone
import tempfile
import os
from pathlib import Path
from contextlib import contextmanager
import asyncio

from hic.message_store import MessageStore, Message, Speaker

# Strategy for generating valid CHOFF tags
choff_tags = st.lists(
    st.text(
        alphabet=st.characters(blacklist_categories=('Cs',)), 
        min_size=1
    ).map(lambda s: f"{{{s}}}"),
    min_size=0,
    max_size=5
)

# Strategy for generating valid messages
message_strategy = st.builds(
    Message,
    speaker=st.sampled_from(list(Speaker)),
    content=st.text(min_size=1, max_size=1000),
    choff_tags=choff_tags,
    # Generate naive datetime first, then add timezone
    timestamp=st.datetimes(
        min_value=datetime(2024, 1, 1),
        max_value=datetime(2025, 12, 31)
    ).map(lambda dt: dt.replace(tzinfo=timezone.utc))
)

@contextmanager
def temp_db():
    """Create a temporary database file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    try:
        yield db_path
    finally:
        try:
            os.unlink(db_path)
        except OSError:
            pass

@pytest.mark.asyncio
async def test_store_creation():
    """Test that we can create an empty message store."""
    with temp_db() as db_path:
        store = MessageStore(db_path)
        assert await store.__len__() == 0

@pytest.mark.asyncio
@given(message=message_strategy)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
async def test_add_message(message: Message):
    """Test that we can add a message and retrieve it."""
    with temp_db() as db_path:
        store = MessageStore(db_path)
        message_id = await store.add(message)
        stored_msg = await store.get(message_id)
        # Message should be equal but not the same object
        assert stored_msg == message
        assert stored_msg is not message
        assert stored_msg.choff_tags == message.choff_tags
        assert stored_msg.choff_tags is not message.choff_tags
        assert await store.__len__() == 1

@pytest.mark.asyncio
@given(messages=st.lists(message_strategy, min_size=1, max_size=10))
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
async def test_multiple_messages(messages):
    """Test adding and retrieving multiple messages."""
    with temp_db() as db_path:
        store = MessageStore(db_path)
        ids = []
        for message in messages:
            msg_id = await store.add(message)
            ids.append(msg_id)
        
        assert await store.__len__() == len(messages)
        for msg_id, original_msg in zip(ids, messages):
            stored_msg = await store.get(msg_id)
            assert stored_msg == original_msg
            assert stored_msg is not original_msg
            assert stored_msg.choff_tags == original_msg.choff_tags
            assert stored_msg.choff_tags is not original_msg.choff_tags

@pytest.mark.asyncio
@given(message=message_strategy)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
async def test_delete_message(message: Message):
    """Test that we can delete a message."""
    with temp_db() as db_path:
        store = MessageStore(db_path)
        message_id = await store.add(message)
        assert await store.__len__() == 1
        await store.delete(message_id)
        assert await store.__len__() == 0
        with pytest.raises(KeyError):
            await store.get(message_id)

@pytest.mark.asyncio
@given(
    messages=st.lists(message_strategy, min_size=1, max_size=10),
    choff_tag=st.text(min_size=1).map(lambda s: f"{{{s}}}")
)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
async def test_find_by_choff_tag(messages, choff_tag):
    """Test finding messages by CHOFF tag."""
    with temp_db() as db_path:
        store = MessageStore(db_path)
        
        # First, create clean messages without the test tag
        messages_to_store = []
        tagged_messages = []
        
        for i, msg in enumerate(messages):
            # Create a new message with clean choff_tags
            new_msg = Message(
                speaker=msg.speaker,
                content=msg.content,
                choff_tags=[t for t in msg.choff_tags if t != choff_tag],  # Remove test tag if present
                timestamp=msg.timestamp
            )
            
            # Tag every other message
            if i % 2 == 0:
                new_msg.choff_tags.append(choff_tag)
                tagged_messages.append(new_msg)
            
            messages_to_store.append(new_msg)
            await store.add(new_msg)
        
        found_messages = await store.find_by_choff_tag(choff_tag)
        assert len(found_messages) == len(tagged_messages)
        
        # Messages should be returned in order of ID
        for found, expected in zip(found_messages, tagged_messages):
            assert found == expected
            assert found is not expected
            assert found.choff_tags == expected.choff_tags
            assert found.choff_tags is not expected.choff_tags
            assert choff_tag in found.choff_tags

@pytest.mark.asyncio
async def test_db_persistence():
    """Test that data persists between store instances."""
    with temp_db() as db_path:
        # Create a store and add a message
        store1 = MessageStore(db_path)
        message = Message(
            speaker=Speaker.USER,
            content="Test message",
            choff_tags=["{test}"],
            timestamp=datetime.now(timezone.utc)
        )
        message_id = await store1.add(message)
        
        # Create a new store instance with the same db
        store2 = MessageStore(db_path)
        # Message should still be accessible
        stored_msg = await store2.get(message_id)
        assert stored_msg == message
        assert await store2.__len__() == 1

class MessageStoreStateMachine(RuleBasedStateMachine):
    """State machine for testing message store behavior."""
    
    messages = Bundle('messages')
    stored_messages = Bundle('stored_messages')
    
    def __init__(self):
        super().__init__()
        self.db_file = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.store = MessageStore(self.db_file.name)
        self.messages = {}  # Keep track of added messages
    
    def teardown(self):
        """Clean up the temporary database."""
        self.db_file.close()
        try:
            os.unlink(self.db_file.name)
        except OSError:
            pass
        super().teardown()
    
    @initialize(target=messages)
    def init_messages(self):
        """Initialize empty message list."""
        return []
    
    @initialize(target=stored_messages)
    def init_stored_messages(self):
        """Initialize stored messages list."""
        return []
    
    @rule(target=messages, message=message_strategy)
    def add_message_to_list(self, message):
        """Add a message to the list."""
        return [message]
    
    @rule(target=stored_messages, message=messages)
    def store_message(self, message):
        """Store a message and verify it."""
        if not message:
            return []
        
        message_id = asyncio.run(self.store.add(message[0]))
        stored_msg = asyncio.run(self.store.get(message_id))
        
        assert stored_msg == message[0]
        assert stored_msg is not message[0]
        self.messages[message_id] = message[0]
        return [(message_id, message[0])]
    
    @rule(data=st.data())
    def delete_message(self, data):
        """Delete a message if we have any."""
        if self.messages:
            message_id = data.draw(st.sampled_from(list(self.messages.keys())))
            asyncio.run(self.store.delete(message_id))
            del self.messages[message_id]
            with pytest.raises(KeyError):
                asyncio.run(self.store.get(message_id))

TestMessageStore = MessageStoreStateMachine.TestCase