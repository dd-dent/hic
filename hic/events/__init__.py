"""Event store for HIC."""
from .schema import (
    EventType,
    ErrorSeverity,
    EventMetadata,
    BaseEvent,
    MessageEvent,
    StateEvent,
    ErrorEvent,
    validate_event
)
from .store import EventStore, EventStoreError

__all__ = [
    'EventType',
    'ErrorSeverity',
    'EventMetadata',
    'BaseEvent',
    'MessageEvent',
    'StateEvent',
    'ErrorEvent',
    'validate_event',
    'EventStore',
    'EventStoreError'
]