"""Event store for HIC."""
from .schema import (
    EventType,
    ErrorSeverity,
    MessageReceivedEvent,
    CHOFFStateUpdatedEvent,
    ErrorOccurredEvent,
    CHOFFState,
    validate_event
)
from .store import EventStore, EventStoreError

__all__ = [
    'EventType',
    'ErrorSeverity',
    'MessageReceivedEvent', 
    'CHOFFStateUpdatedEvent',
    'ErrorOccurredEvent',
    'CHOFFState',
    'validate_event',
    'EventStore',
    'EventStoreError'
]