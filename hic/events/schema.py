"""Event schemas and validation for HIC."""
import uuid
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Optional, Dict, Any, List
import attr

class EventType(Enum):
    """Core event types."""
    MESSAGE_RECEIVED = auto()
    CHOFF_STATE_UPDATED = auto()
    ERROR_OCCURRED = auto()

class ErrorSeverity(Enum):
    """Error severity levels."""
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()
@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class BaseEvent:
    """Base event with common fields."""
    event_type: EventType
    event_id: str = attr.Factory(lambda: str(uuid.uuid4()))
    timestamp: datetime = attr.Factory(lambda: datetime.now(timezone.utc))
    version: str = "1.0"
    correlation_id: Optional[str] = None

@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class MessageReceivedEvent(BaseEvent):
    """Event for capturing new messages."""
    content: str
    source: str
    metadata: Dict[str, Any] = attr.Factory(dict)

    @classmethod
    def create(
        cls,
        content: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ) -> 'MessageReceivedEvent':
        """Create a new MessageReceivedEvent with defaults."""
        return cls(
            event_type=EventType.MESSAGE_RECEIVED,
            content=content,
            source=source,
            metadata=metadata or {},
            correlation_id=correlation_id
        )

@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class CHOFFState:
    """CHOFF state representation."""
    notation: str
    components: Dict[str, Any]

@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class CHOFFStateUpdatedEvent(BaseEvent):
    """Event for tracking CHOFF state changes."""
    previous_state: Optional[CHOFFState]
    new_state: CHOFFState
    transition_metadata: Dict[str, Any] = attr.Factory(dict)

    @classmethod
    def create(
        cls,
        new_state: CHOFFState,
        previous_state: Optional[CHOFFState] = None,
        transition_metadata: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ) -> 'CHOFFStateUpdatedEvent':
        """Create a new CHOFFStateUpdatedEvent with defaults."""
        return cls(
            event_type=EventType.CHOFF_STATE_UPDATED,
            previous_state=previous_state,
            new_state=new_state,
            transition_metadata=transition_metadata or {},
            correlation_id=correlation_id
        )

@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class ErrorOccurredEvent(BaseEvent):
    """Event for tracking errors and exceptions."""
    error_type: str
    message: str
    severity: ErrorSeverity
    stack_trace: Optional[str] = None
    context: Dict[str, Any] = attr.Factory(dict)

    @classmethod
    def create(
        cls,
        error_type: str,
        message: str,
        severity: ErrorSeverity,
        stack_trace: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ) -> 'ErrorOccurredEvent':
        """Create a new ErrorOccurredEvent with defaults."""
        return cls(
            event_type=EventType.ERROR_OCCURRED,
            error_type=error_type,
            message=message,
            severity=severity,
            stack_trace=stack_trace,
            context=context or {},
            correlation_id=correlation_id
        )

def validate_event(event: BaseEvent) -> None:
    """Validate an event's schema and data."""
    if not isinstance(event, BaseEvent):
        raise ValueError("Event must inherit from BaseEvent")

    if not event.event_id:
        raise ValueError("Event ID is required")

    if not event.timestamp:
        raise ValueError("Timestamp is required")

    if not event.event_type:
        raise ValueError("Event type is required")

    if not event.version:
        raise ValueError("Version is required")

    # Event-specific validation
    if isinstance(event, MessageReceivedEvent):
        if not event.content:
            raise ValueError("Message content is required")
        if not event.source:
            raise ValueError("Message source is required")

    elif isinstance(event, CHOFFStateUpdatedEvent):
        if not event.new_state:
            raise ValueError("New CHOFF state is required")
        if not event.new_state.notation:
            raise ValueError("CHOFF notation is required")

    elif isinstance(event, ErrorOccurredEvent):
        if not event.error_type:
            raise ValueError("Error type is required")
        if not event.message:
            raise ValueError("Error message is required")
        if not event.severity:
            raise ValueError("Error severity is required")