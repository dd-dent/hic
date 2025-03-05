"""Event schemas and validation for HIC."""
import uuid
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Optional, Dict, Any, List, Union
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
class EventMetadata:
    """Common metadata for all events."""
    event_id: str = attr.Factory(lambda: str(uuid.uuid4()))
    timestamp: int = attr.Factory(lambda: int(datetime.now(timezone.utc).timestamp()))
    conversation_id: str
    correlation_id: Optional[str] = None
    version: str = "1.0"

@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class BaseEvent:
    """Base event with common fields."""
    metadata: EventMetadata

@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class MessageEvent(BaseEvent):
    """Event for capturing messages."""
    content: str
    source: Optional[str] = None

    @classmethod
    def create(
        cls,
        content: str,
        conversation_id: str,
        source: Optional[str] = None,
        correlation_id: Optional[str] = None
    ) -> 'MessageEvent':
        """Create a new MessageEvent with defaults."""
        return cls(
            metadata=EventMetadata(
                conversation_id=conversation_id,
                correlation_id=correlation_id
            ),
            content=content,
            source=source
        )

@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class StateEvent(BaseEvent):
    """Event for tracking CHOFF state changes."""
    state_expression: Union[str, Dict[str, float]]
    expression_type: str = "basic"
    context: Optional[str] = None
    
    @property
    def state_type(self) -> str:
        """Get the primary state type for backward compatibility."""
        if isinstance(self.state_expression, str):
            return self.state_expression
        return next(iter(self.state_expression), "")
    
    @property
    def intensity(self) -> float:
        """Get the primary intensity for backward compatibility."""
        if isinstance(self.state_expression, str):
            return 1.0
        return next(iter(self.state_expression.values()), 1.0)

    @classmethod
    def create(
        cls,
        state_expression: Union[str, Dict[str, float]],
        conversation_id: str,
        expression_type: str = "basic",
        context: Optional[str] = None,
        correlation_id: Optional[str] = None
    ) -> 'StateEvent':
        """Create a new StateEvent with defaults."""
        return cls(
            metadata=EventMetadata(
                conversation_id=conversation_id,
                correlation_id=correlation_id
            ),
            state_expression=state_expression,
            expression_type=expression_type,
            context=context
        )
    
    @classmethod
    def from_choff_state(
        cls,
        choff_state: 'ChoffState',
        conversation_id: str,
        context: Optional[str] = None,
        correlation_id: Optional[str] = None
    ) -> 'StateEvent':
        """Create a StateEvent from a ChoffState object."""
        from hic.choff.parser import StateType
        
        # Convert components to a dictionary
        components = {comp.state_type: comp.value for comp in choff_state.components}
        
        # For basic states with a single component, use string representation
        if choff_state.expression_type == StateType.BASIC and len(components) == 1:
            state_expression = next(iter(components.keys()))
        else:
            state_expression = components
        
        return cls(
            metadata=EventMetadata(
                conversation_id=conversation_id,
                correlation_id=correlation_id
            ),
            state_expression=state_expression,
            expression_type=choff_state.expression_type.value,
            context=context
        )

@attr.s(auto_attribs=True, frozen=True, kw_only=True)
class ErrorEvent(BaseEvent):
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
        conversation_id: str,
        severity: ErrorSeverity,
        stack_trace: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ) -> 'ErrorEvent':
        """Create a new ErrorEvent with defaults."""
        return cls(
            metadata=EventMetadata(
                conversation_id=conversation_id,
                correlation_id=correlation_id
            ),
            error_type=error_type,
            message=message,
            severity=severity,
            stack_trace=stack_trace,
            context=context or {}
        )

def validate_event(event: BaseEvent) -> None:
    """Validate an event's schema and data."""
    if not isinstance(event, BaseEvent):
        raise ValueError("Event must inherit from BaseEvent")

    if not event.metadata:
        raise ValueError("Event metadata is required")

    if not event.metadata.event_id:
        raise ValueError("Event ID is required")

    if not event.metadata.timestamp:
        raise ValueError("Timestamp is required")

    if not event.metadata.conversation_id:
        raise ValueError("Conversation ID is required")

    # Event-specific validation
    if isinstance(event, MessageEvent):
        if not event.content:
            raise ValueError("Message content is required")

    elif isinstance(event, StateEvent):
        if not event.state_expression:
            raise ValueError("State expression is required")
        if not event.expression_type:
            raise ValueError("Expression type is required")
        
        if isinstance(event.state_expression, dict):
            for state_type, value in event.state_expression.items():
                if not state_type:
                    raise ValueError("State type is required")
                if not isinstance(value, (int, float)):
                    raise ValueError("State values must be numbers")
                if value < 0.0 or value > 1.0:
                    raise ValueError(f"State value must be between 0.0 and 1.0, got {value}")

    elif isinstance(event, ErrorEvent):
        if not event.error_type:
            raise ValueError("Error type is required")
        if not event.message:
            raise ValueError("Error message is required")
        if not event.severity:
            raise ValueError("Error severity is required")