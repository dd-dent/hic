from datetime import datetime
from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Dict, Set

class Speaker(Enum):
    USER = auto()
    ASSISTANT = auto()

@dataclass
class Message:
    """A chat message with CHOFF metadata."""
    speaker: Speaker
    content: str
    choff_tags: List[str]
    timestamp: datetime

class MessageStore:
    """In-memory storage for chat messages with CHOFF tag support."""
    
    def __init__(self):
        self._messages: Dict[int, Message] = {}
        self._next_id: int = 1
        # Index for faster CHOFF tag lookups
        self._tag_index: Dict[str, Set[int]] = {}
    
    def __len__(self) -> int:
        return len(self._messages)
    
    def add(self, message: Message) -> int:
        """Add a message and return its ID."""
        message_id = self._next_id
        # Store a copy of the message to prevent external modifications
        stored_message = Message(
            speaker=message.speaker,
            content=message.content,
            choff_tags=list(message.choff_tags),  # Create a new list
            timestamp=message.timestamp
        )
        self._messages[message_id] = stored_message
        self._next_id += 1
        
        # Update tag index - ensure we only index unique tags
        unique_tags = set(stored_message.choff_tags)
        for tag in unique_tags:
            if tag not in self._tag_index:
                self._tag_index[tag] = set()
            self._tag_index[tag].add(message_id)
        
        return message_id
    
    def get(self, message_id: int) -> Message:
        """Retrieve a message by ID."""
        if message_id not in self._messages:
            raise KeyError(f"Message {message_id} not found")
        # Return a copy to prevent external modifications
        msg = self._messages[message_id]
        return Message(
            speaker=msg.speaker,
            content=msg.content,
            choff_tags=list(msg.choff_tags),
            timestamp=msg.timestamp
        )
    
    def delete(self, message_id: int) -> None:
        """Delete a message by ID."""
        if message_id not in self._messages:
            raise KeyError(f"Message {message_id} not found")
        
        # Clean up tag index
        message = self._messages[message_id]
        unique_tags = set(message.choff_tags)  # Handle duplicate tags
        for tag in unique_tags:
            if tag in self._tag_index:  # Check if tag exists in index
                self._tag_index[tag].remove(message_id)
                if not self._tag_index[tag]:  # Clean up empty sets
                    del self._tag_index[tag]
        
        del self._messages[message_id]
    
    def find_by_choff_tag(self, tag: str) -> List[Message]:
        """Find all messages with a specific CHOFF tag."""
        if tag not in self._tag_index:
            return []
        # Return messages in order of ID
        message_ids = sorted(self._tag_index[tag])
        # Return copies to prevent external modifications
        return [
            Message(
                speaker=msg.speaker,
                content=msg.content,
                choff_tags=list(msg.choff_tags),
                timestamp=msg.timestamp
            )
            for msg in (self._messages[msg_id] for msg_id in message_ids)
        ]