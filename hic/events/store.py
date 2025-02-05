"""SQLite-based event store implementation."""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, TypeVar
import trio

from .schema import (
    BaseEvent,
    EventType,
    MessageReceivedEvent,
    CHOFFStateUpdatedEvent,
    ErrorOccurredEvent,
    CHOFFState,
    ErrorSeverity,
    validate_event
)

T = TypeVar('T', bound=BaseEvent)

class EventStoreError(Exception):
    """Base exception for event store errors."""
    pass

class EventStore:
    """Async SQLite-based event store with schema validation."""
    
    def __init__(self, db_path: str = None):
        """Initialize the event store with optional custom database path."""
        if db_path is None:
            # Use a default path in the user's home directory
            db_path = str(Path.home() / ".hic" / "events.db")
        
        # Ensure the directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self._db_path = db_path
        self._init_db()
        
        # Register adapters for datetime and enum handling
        sqlite3.register_adapter(datetime, self._adapt_datetime)
        sqlite3.register_converter("timestamp", self._convert_timestamp)
        sqlite3.register_adapter(EventType, lambda e: e.name)
        sqlite3.register_adapter(ErrorSeverity, lambda e: e.name)
    
    @staticmethod
    def _adapt_datetime(dt: datetime) -> str:
        """Convert datetime to string for SQLite storage."""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()
    
    @staticmethod
    def _convert_timestamp(val: bytes) -> datetime:
        """Convert SQLite timestamp string back to datetime."""
        dt = datetime.fromisoformat(val.decode())
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    
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
        """Initialize the database schema if it doesn't exist."""
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    timestamp timestamp NOT NULL,
                    event_type TEXT NOT NULL,
                    version TEXT NOT NULL,
                    correlation_id TEXT,
                    payload JSON NOT NULL
                );
                
                CREATE INDEX IF NOT EXISTS idx_events_type 
                ON events(event_type);
                
                CREATE INDEX IF NOT EXISTS idx_events_timestamp 
                ON events(timestamp);
                
                CREATE INDEX IF NOT EXISTS idx_events_correlation 
                ON events(correlation_id);
            """)
    
    def _serialize_event(self, event: BaseEvent) -> dict:
        """Serialize an event for storage."""
        if isinstance(event, MessageReceivedEvent):
            payload = {
                'content': event.content,
                'source': event.source,
                'metadata': event.metadata
            }
        elif isinstance(event, CHOFFStateUpdatedEvent):
            payload = {
                'previous_state': {
                    'notation': event.previous_state.notation,
                    'components': event.previous_state.components
                } if event.previous_state else None,
                'new_state': {
                    'notation': event.new_state.notation,
                    'components': event.new_state.components
                },
                'transition_metadata': event.transition_metadata
            }
        elif isinstance(event, ErrorOccurredEvent):
            payload = {
                'error_type': event.error_type,
                'message': event.message,
                'severity': event.severity.name,
                'stack_trace': event.stack_trace,
                'context': event.context
            }
        else:
            raise ValueError(f"Unsupported event type: {type(event)}")
        
        return payload
    
    def _deserialize_event(self, row: sqlite3.Row) -> BaseEvent:
        """Deserialize an event from storage."""
        event_type = EventType[row['event_type']]
        payload = json.loads(row['payload'])
        
        if event_type == EventType.MESSAGE_RECEIVED:
            return MessageReceivedEvent(
                event_id=row['event_id'],
                timestamp=row['timestamp'],
                event_type=event_type,
                version=row['version'],
                correlation_id=row['correlation_id'],
                content=payload['content'],
                source=payload['source'],
                metadata=payload['metadata']
            )
        elif event_type == EventType.CHOFF_STATE_UPDATED:
            previous_state = None
            if payload['previous_state']:
                previous_state = CHOFFState(
                    notation=payload['previous_state']['notation'],
                    components=payload['previous_state']['components']
                )
            new_state = CHOFFState(
                notation=payload['new_state']['notation'],
                components=payload['new_state']['components']
            )
            return CHOFFStateUpdatedEvent(
                event_id=row['event_id'],
                timestamp=row['timestamp'],
                event_type=event_type,
                version=row['version'],
                correlation_id=row['correlation_id'],
                previous_state=previous_state,
                new_state=new_state,
                transition_metadata=payload['transition_metadata']
            )
        elif event_type == EventType.ERROR_OCCURRED:
            return ErrorOccurredEvent(
                event_id=row['event_id'],
                timestamp=row['timestamp'],
                event_type=event_type,
                version=row['version'],
                correlation_id=row['correlation_id'],
                error_type=payload['error_type'],
                message=payload['message'],
                severity=ErrorSeverity[payload['severity']],
                stack_trace=payload['stack_trace'],
                context=payload['context']
            )
        else:
            raise ValueError(f"Unknown event type: {event_type}")
    
    async def append(self, event: BaseEvent) -> None:
        """Append an event to the store."""
        # Validate event schema
        validate_event(event)
        
        # Serialize event
        payload = self._serialize_event(event)
        
        # Store event
        async with trio.open_nursery() as nursery:
            await trio.to_thread.run_sync(
                lambda: self._append_sync(event, json.dumps(payload))
            )
    
    def _append_sync(self, event: BaseEvent, payload_json: str):
        """Synchronous event append operation."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO events (
                    event_id, timestamp, event_type, version,
                    correlation_id, payload
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.timestamp,
                    event.event_type.name,
                    event.version,
                    event.correlation_id,
                    payload_json
                )
            )
    
    async def get_by_id(self, event_id: str) -> BaseEvent:
        """Retrieve an event by ID."""
        async with trio.open_nursery() as nursery:
            row = await trio.to_thread.run_sync(
                lambda: self._get_by_id_sync(event_id)
            )
            if not row:
                raise EventStoreError(f"Event not found: {event_id}")
            return self._deserialize_event(row)
    
    def _get_by_id_sync(self, event_id: str) -> Optional[sqlite3.Row]:
        """Synchronous get by ID operation."""
        with self._get_connection() as conn:
            return conn.execute(
                "SELECT * FROM events WHERE event_id = ?",
                (event_id,)
            ).fetchone()
    
    async def get_by_type(
        self,
        event_type: EventType | str,
        limit: Optional[int] = None,
        correlation_id: Optional[str] = None
    ) -> List[BaseEvent]:
        """Retrieve events by type with optional correlation ID filter."""
        # Convert string to enum if needed
        if isinstance(event_type, str):
            try:
                event_type = EventType[event_type]
            except KeyError:
                raise ValueError(f"Invalid event type: {event_type}")

        async with trio.open_nursery() as nursery:
            rows = await trio.to_thread.run_sync(
                lambda: self._get_by_type_sync(event_type, limit, correlation_id)
            )
            return [self._deserialize_event(row) for row in rows]
    
    def _get_by_type_sync(
        self,
        event_type: EventType,
        limit: Optional[int],
        correlation_id: Optional[str]
    ) -> List[sqlite3.Row]:
        """Synchronous get by type operation."""
        with self._get_connection() as conn:
            query = "SELECT * FROM events WHERE event_type = ?"
            params = [event_type.name]
            
            if correlation_id:
                query += " AND correlation_id = ?"
                params.append(correlation_id)
            
            query += " ORDER BY timestamp DESC"
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            return conn.execute(query, params).fetchall()
    
    async def get_latest_by_type(self, event_type: EventType) -> Optional[BaseEvent]:
        """Get the most recent event of a given type."""
        events = await self.get_by_type(event_type, limit=1)
        return events[0] if events else None