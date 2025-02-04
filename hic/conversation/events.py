from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional

class EventType(Enum):
    """Core conversation event types"""
    MESSAGE_ADDED = auto()
    MESSAGE_DELETED = auto()
    SUMMARY_REQUESTED = auto()
    SUMMARY_GENERATED = auto()
    STATE_CHANGED = auto()

@dataclass
class Event:
    """Base event class for conversation events"""
    type: EventType
    timestamp: datetime
    payload: Dict[str, Any] = field(default_factory=dict)
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)

@dataclass
class MessageEvent:
    """Event specific to message operations"""
    type: EventType
    timestamp: datetime
    message_id: str
    content: str
    role: str
    payload: Dict[str, Any] = field(default_factory=dict)
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)

@dataclass
class SummaryEvent:
    """Event specific to summary operations"""
    type: EventType
    timestamp: datetime
    summary_id: str
    content: str
    source_message_ids: List[str]
    payload: Dict[str, Any] = field(default_factory=dict)
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)

@dataclass
class StateEvent:
    """Event for CHOFF state changes"""
    type: EventType
    timestamp: datetime
    previous_state: Dict[str, Any]
    new_state: Dict[str, Any]
    transition_type: str
    payload: Dict[str, Any] = field(default_factory=dict)
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)