"""SQLite-based event store for CHOFF-aware conversation tracking."""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, TypeVar, Dict, Any
import asyncio

from .schema import (
    BaseEvent,
    MessageEvent,
    StateEvent,
    ErrorEvent,
    ErrorSeverity,
    EventMetadata,
    validate_event
)

T = TypeVar('T', bound=BaseEvent)

class EventStoreError(Exception):
    """Base exception for event store errors."""
    pass

class EventStore:
    """
    Async SQLite-based event store with CHOFF metadata support.
    
    This store maintains an append-only log of conversation events,
    preserving CHOFF state transitions and cognitive flow markers.
    """
    
    def __init__(self, db_path: str = None):
        """Initialize the event store with optional custom database path."""
        if db_path is None:
            # Use a default path in the user's home directory
            db_path = str(Path.home() / ".hic" / "events.db")
        
        # Ensure the directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self._db_path = db_path
        self._init_db()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with proper settings."""
        conn = sqlite3.connect(
            self._db_path,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn
    
    def _init_db(self):
        """Initialize the database schema for CHOFF event storage."""
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    timestamp INTEGER NOT NULL,
                    conversation_id TEXT NOT NULL,
                    correlation_id TEXT,
                    version TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload JSON NOT NULL
                );
                
                CREATE INDEX IF NOT EXISTS idx_events_conversation 
                ON events(conversation_id);
                
                CREATE INDEX IF NOT EXISTS idx_events_timestamp 
                ON events(timestamp);
                
                CREATE INDEX IF NOT EXISTS idx_events_correlation 
                ON events(correlation_id);
            """)
    
    def _serialize_event(self, event: BaseEvent) -> Dict[str, Any]:
        """
        Convert an event to its JSON representation for storage.
        
        Handles CHOFF metadata and event-specific fields appropriately.
        """
        if isinstance(event, MessageEvent):
            return {
                'content': event.content,
                'source': event.source
            }
        elif isinstance(event, StateEvent):
            return {
                'state_type': event.state_type,
                'intensity': event.intensity,
                'context': event.context,
                'state_expression': event.state_expression,
                'expression_type': event.expression_type
            }
        elif isinstance(event, ErrorEvent):
            return {
                'error_type': event.error_type,
                'message': event.message,
                'severity': event.severity.name,
                'stack_trace': event.stack_trace,
                'context': event.context
            }
        else:
            raise ValueError(f"Unsupported event type: {type(event)}")
    
    def _deserialize_event(self, row: sqlite3.Row) -> BaseEvent:
        """
        Reconstruct an event from its stored form.
        
        Handles CHOFF metadata and event-specific deserialization.
        """
        payload = json.loads(row['payload'])
        metadata = EventMetadata(
            event_id=row['event_id'],
            timestamp=row['timestamp'],
            conversation_id=row['conversation_id'],
            correlation_id=row['correlation_id'],
            version=row['version']
        )
        
        event_type = row['event_type']
        if event_type == 'message':
            return MessageEvent(
                metadata=metadata,
                content=payload['content'],
                source=payload.get('source')
            )
        elif event_type == 'state':
            # Use state_expression if available, otherwise construct from state_type/intensity
            if 'state_expression' in payload:
                state_expression = payload['state_expression']
                expression_type = payload.get('expression_type', 'basic')
            else:
                # Backward compatibility: convert state_type/intensity to state_expression
                state_type = payload['state_type']
                state_expression = state_type
                expression_type = 'basic'
            
            return StateEvent(
                metadata=metadata,
                state_expression=state_expression,
                expression_type=expression_type,
                context=payload.get('context')
            )
        elif event_type == 'error':
            return ErrorEvent(
                metadata=metadata,
                error_type=payload['error_type'],
                message=payload['message'],
                severity=ErrorSeverity[payload['severity']],
                stack_trace=payload.get('stack_trace'),
                context=payload.get('context', {})
            )
        else:
            raise ValueError(f"Unknown event type: {event_type}")
    
    async def append(self, event: BaseEvent) -> None:
        """
        Append an event to the store.
        
        Args:
            event: The event to store, with CHOFF metadata
        """
        validate_event(event)
        payload = self._serialize_event(event)
        
        await asyncio.to_thread(
            lambda: self._append_sync(event, json.dumps(payload))
        )
    
    def _append_sync(self, event: BaseEvent, payload_json: str):
        """Synchronous event append operation."""
        event_type = (
            'message' if isinstance(event, MessageEvent)
            else 'state' if isinstance(event, StateEvent)
            else 'error' if isinstance(event, ErrorEvent)
            else 'unknown'
        )
        
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO events (
                    event_id, timestamp, conversation_id,
                    correlation_id, version, event_type, payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.metadata.event_id,
                    event.metadata.timestamp,
                    event.metadata.conversation_id,
                    event.metadata.correlation_id,
                    event.metadata.version,
                    event_type,
                    payload_json
                )
            )
    
    async def get_by_conversation(
        self,
        conversation_id: str,
        limit: Optional[int] = None
    ) -> List[BaseEvent]:
        """
        Retrieve events for a specific conversation.
        
        Args:
            conversation_id: The conversation to fetch events for
            limit: Optional maximum number of events to return
            
        Returns:
            List of events in chronological order
        """
        rows = await asyncio.to_thread(
            lambda: self._get_by_conversation_sync(conversation_id, limit)
        )
        return [self._deserialize_event(row) for row in rows]
    
    def _get_by_conversation_sync(
        self,
        conversation_id: str,
        limit: Optional[int]
    ) -> List[sqlite3.Row]:
        """Synchronous conversation event retrieval."""
        with self._get_connection() as conn:
            query = """
                SELECT * FROM events 
                WHERE conversation_id = ?
                ORDER BY timestamp
            """
            params = [conversation_id]
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            return conn.execute(query, params).fetchall()