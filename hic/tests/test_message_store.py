import pytest
from hypothesis import given, strategies as st
from hypothesis.stateful import RuleBasedStateMachine, rule
from datetime import datetime, timezone

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

def test_store_creation():
    """Test that we can create an empty message store."""
    store = MessageStore()
    assert len(store) == 0

@given(message=message_strategy)
def test_add_message(message: Message):
    """Test that we can add a message and retrieve it."""
    store = MessageStore()
    message_id = store.add(message)
    stored_msg = store.get(message_id)
    # Message should be equal but not the same object
    assert stored_msg == message
    assert stored_msg is not message
    assert stored_msg.choff_tags == message.choff_tags
    assert stored_msg.choff_tags is not message.choff_tags
    assert len(store) == 1

@given(messages=st.lists(message_strategy, min_size=1, max_size=10))
def test_multiple_messages(messages):
    """Test adding and retrieving multiple messages."""
    store = MessageStore()
    ids = []
    for message in messages:
        msg_id = store.add(message)
        ids.append(msg_id)
    
    assert len(store) == len(messages)
    for msg_id, original_msg in zip(ids, messages):
        stored_msg = store.get(msg_id)
        assert stored_msg == original_msg
        assert stored_msg is not original_msg
        assert stored_msg.choff_tags == original_msg.choff_tags
        assert stored_msg.choff_tags is not original_msg.choff_tags

@given(message=message_strategy)
def test_delete_message(message: Message):
    """Test that we can delete a message."""
    store = MessageStore()
    message_id = store.add(message)
    assert len(store) == 1
    store.delete(message_id)
    assert len(store) == 0
    with pytest.raises(KeyError):
        store.get(message_id)

@given(
    messages=st.lists(message_strategy, min_size=1, max_size=10),
    choff_tag=st.text(min_size=1).map(lambda s: f"{{{s}}}")
)
def test_find_by_choff_tag(messages, choff_tag):
    """Test finding messages by CHOFF tag."""
    store = MessageStore()
    
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
        store.add(new_msg)
    
    found_messages = store.find_by_choff_tag(choff_tag)
    assert len(found_messages) == len(tagged_messages)
    
    # Messages should be returned in order of ID
    for found, expected in zip(found_messages, tagged_messages):
        assert found == expected
        assert found is not expected
        assert found.choff_tags == expected.choff_tags
        assert found.choff_tags is not expected.choff_tags
        assert choff_tag in found.choff_tags

# Bonus: State machine testing for more complex interactions
class MessageStoreStateMachine(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.store = MessageStore()
        self.messages = {}  # Keep track of added messages

    @rule(message=message_strategy)
    def add_message(self, message):
        """Add a message and verify it's stored correctly."""
        message_id = self.store.add(message)
        stored_msg = self.store.get(message_id)
        assert stored_msg == message
        assert stored_msg is not message
        self.messages[message_id] = message

    @rule(data=st.data())
    def delete_message(self, data):
        """Delete a message if we have any."""
        if self.messages:
            message_id = data.draw(st.sampled_from(list(self.messages.keys())))
            self.store.delete(message_id)
            del self.messages[message_id]
            with pytest.raises(KeyError):
                self.store.get(message_id)

TestMessageStore = MessageStoreStateMachine.TestCase