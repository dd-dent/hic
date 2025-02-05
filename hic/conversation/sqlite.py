"""SQLite-based implementation of conversation management."""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pathlib import Path
import uuid
import trio

from .manager import ConversationManager
from .bus import EventBus
from .events import EventType, MessageEvent, SummaryEvent
from ..events.store import EventStore
from ..message_store import MessageStore, Message, Speaker
from ..agents.summarizer import SummarizerAgent

class SQLiteConversationManager(ConversationManager):
    """SQLite-based implementation of conversation management with event sourcing."""
    
    def __init__(self, 
                event_bus: EventBus,
                event_store: Optional[EventStore] = None,
                message_store: Optional[MessageStore] = None,
                summarizer: Optional[SummarizerAgent] = None,
                db_path: Optional[str] = None):
        """Initialize the conversation manager."""
        super().__init__(event_bus)
        
        if db_path is None:
            db_path = str(Path.home() / ".hic")
        
        self.event_store = event_store or EventStore(f"{db_path}/events.db")
        self.message_store = message_store or MessageStore(f"{db_path}/messages.db")
        self.summarizer = summarizer
        self._summaries: Dict[str, Dict[str, Any]] = {}
        self._message_id_map: Dict[str, int] = {}  # Map UUID to SQLite IDs
        
        # Clear any existing data for clean state
        self._clear_storage()
    
    def _clear_storage(self):
        """Clear all storage for clean state."""
        # Clear message store
        with self.message_store._get_connection() as conn:
            conn.execute("DELETE FROM messages")
            conn.execute("DELETE FROM tags")
        
        # Clear event store
        with self.event_store._get_connection() as conn:
            conn.execute("DELETE FROM events")
        
        # Clear in-memory maps
        self._message_id_map.clear()
        self._summaries.clear()
        self._choff_state.clear()
        self._transitions.clear()
    
    def add_message(self, content: str, role: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add a new message to the conversation."""
        # Generate unique ID
        message_id = str(uuid.uuid4())
        
        # Create message with current CHOFF state
        choff_tags = []
        if self._choff_state:
            choff_tags = [f"{k}:{v}" for k, v in self._choff_state.items()]
        
        # Map role to speaker
        speaker = Speaker.USER if role.lower() == "user" else Speaker.ASSISTANT
        
        # Store message
        message = Message(
            speaker=speaker,
            content=content,
            choff_tags=choff_tags,
            timestamp=datetime.now(timezone.utc)
        )
        sqlite_id = self.message_store.add(message)
        self._message_id_map[message_id] = sqlite_id
        
        # Create and store event
        event = MessageEvent(
            type=EventType.MESSAGE_ADDED,
            timestamp=datetime.now(timezone.utc),
            payload=metadata or {},
            message_id=message_id,
            content=content,
            role=role
        )
        self.event_bus.publish(event)
        
        return message_id
    
    def get_message(self, message_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a message by ID."""
        try:
            sqlite_id = self._message_id_map.get(message_id)
            if sqlite_id is None:
                return None
                
            message = self.message_store.get(sqlite_id)
            return {
                'id': message_id,
                'content': message.content,
                'role': message.speaker.name.lower(),
                'timestamp': message.timestamp,
                'choff_tags': message.choff_tags
            }
        except KeyError:
            return None
    
    def delete_message(self, message_id: str) -> bool:
        """Delete a message from the conversation."""
        try:
            sqlite_id = self._message_id_map.get(message_id)
            if sqlite_id is None:
                return False
                
            self.message_store.delete(sqlite_id)
            del self._message_id_map[message_id]
            
            # Create and store deletion event
            event = MessageEvent(
                type=EventType.MESSAGE_DELETED,
                timestamp=datetime.now(timezone.utc),
                payload={},
                message_id=message_id,
                content="",
                role=""
            )
            self.event_bus.publish(event)
            
            return True
        except KeyError:
            return False
    
    def get_messages(self,
                    start_time: Optional[datetime] = None,
                    end_time: Optional[datetime] = None,
                    role: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve messages with optional filtering."""
        messages = []
        for uuid_id, sqlite_id in self._message_id_map.items():
            try:
                message = self.message_store.get(sqlite_id)
                
                # Apply filters
                if start_time and message.timestamp < start_time:
                    continue
                if end_time and message.timestamp > end_time:
                    continue
                if role and message.speaker.name.lower() != role.lower():
                    continue
                
                messages.append({
                    'id': uuid_id,
                    'content': message.content,
                    'role': message.speaker.name.lower(),
                    'timestamp': message.timestamp,
                    'choff_tags': message.choff_tags
                })
            except KeyError:
                continue
        
        return sorted(messages, key=lambda m: m['timestamp'])
    
    def request_summary(self,
                       message_ids: Optional[List[str]] = None,
                       start_time: Optional[datetime] = None,
                       end_time: Optional[datetime] = None) -> str:
        """Request a summary of specified messages."""
        if not self.summarizer:
            raise RuntimeError("Summarizer agent not configured")
        
        # Get messages to summarize
        if message_ids:
            messages = []
            for msg_id in message_ids:
                msg = self.get_message(msg_id)
                if msg:
                    messages.append(msg['content'])
        else:
            messages = [
                msg['content']
                for msg in self.get_messages(start_time, end_time)
            ]
        
        if not messages:
            raise ValueError("No messages found to summarize")
        
        # Generate unique summary ID
        summary_id = str(uuid.uuid4())
        
        # Store summary request event
        event = SummaryEvent(
            type=EventType.SUMMARY_REQUESTED,
            timestamp=datetime.now(timezone.utc),
            payload={
                'message_ids': message_ids,
                'start_time': start_time.isoformat() if start_time else None,
                'end_time': end_time.isoformat() if end_time else None
            },
            summary_id=summary_id,
            content="",
            source_message_ids=message_ids or []
        )
        self.event_bus.publish(event)
        
        # Store pending summary
        self._summaries[summary_id] = {
            'id': summary_id,
            'status': 'pending',
            'messages': messages,
            'timestamp': datetime.now(timezone.utc)
        }
        
        return summary_id
    
    async def get_summary(self, summary_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a generated summary."""
        summary = self._summaries.get(summary_id)
        if not summary:
            return None
        
        # If summary is pending, try to generate it
        if summary['status'] == 'pending' and self.summarizer:
            try:
                content = await self.summarizer.summarize_messages(summary['messages'])
                
                # Update summary
                summary['status'] = 'completed'
                summary['content'] = content
                
                # Store completion event
                event = SummaryEvent(
                    type=EventType.SUMMARY_GENERATED,
                    timestamp=datetime.now(timezone.utc),
                    payload={},
                    summary_id=summary_id,
                    content=content,
                    source_message_ids=[]
                )
                self.event_bus.publish(event)
            except Exception as e:
                summary['status'] = 'failed'
                summary['error'] = str(e)
        
        return summary