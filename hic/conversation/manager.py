from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from .bus import EventBus
from .events import Event, EventType, MessageEvent, StateEvent, SummaryEvent

class ConversationManager(ABC):
    """Abstract base class for conversation management"""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self._choff_state: Dict[str, Any] = {}
        self._transitions: List[Dict[str, Any]] = []
    
    @abstractmethod
    def add_message(self, content: str, role: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a new message to the conversation
        
        Args:
            content: Message content
            role: Role of the message sender
            metadata: Optional metadata for the message
            
        Returns:
            str: Unique identifier for the message
        """
        pass
    
    @abstractmethod
    def get_message(self, message_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a message by its ID
        
        Args:
            message_id: Unique identifier of the message
            
        Returns:
            Optional[Dict[str, Any]]: Message data if found, None otherwise
        """
        pass
    
    @abstractmethod
    def delete_message(self, message_id: str) -> bool:
        """
        Delete a message from the conversation
        
        Args:
            message_id: Unique identifier of the message
            
        Returns:
            bool: True if message was deleted, False if not found
        """
        pass
    
    @abstractmethod
    def get_messages(self, 
                    start_time: Optional[datetime] = None,
                    end_time: Optional[datetime] = None,
                    role: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve messages with optional filtering
        
        Args:
            start_time: Optional start time filter
            end_time: Optional end time filter
            role: Optional role filter
            
        Returns:
            List[Dict[str, Any]]: List of matching messages
        """
        pass
    
    @abstractmethod
    def request_summary(self, 
                       message_ids: Optional[List[str]] = None,
                       start_time: Optional[datetime] = None,
                       end_time: Optional[datetime] = None) -> str:
        """
        Request a summary of specified messages
        
        Args:
            message_ids: Optional list of specific message IDs to summarize
            start_time: Optional start time for message range
            end_time: Optional end time for message range
            
        Returns:
            str: Unique identifier for the summary request
        """
        pass
    
    @abstractmethod
    def get_summary(self, summary_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a generated summary
        
        Args:
            summary_id: Unique identifier of the summary
            
        Returns:
            Optional[Dict[str, Any]]: Summary data if found, None otherwise
        """
        pass
    
    def update_choff_state(self, new_state: Dict[str, Any], transition_type: str) -> None:
        """
        Update the CHOFF state with transition tracking
        
        Args:
            new_state: New CHOFF state to set
            transition_type: Type of state transition
        """
        previous_state = self._choff_state.copy()
        self._choff_state = new_state.copy()
        
        self._transitions.append({
            'previous': previous_state,
            'new': new_state,
            'type': transition_type,
            'timestamp': datetime.now()
        })
        
        event = StateEvent(
            type=EventType.STATE_CHANGED,
            timestamp=datetime.now(),
            payload={},
            previous_state=previous_state,
            new_state=new_state,
            transition_type=transition_type
        )
        
        self.event_bus.publish(event)
    
    def get_choff_state(self) -> Dict[str, Any]:
        """Get current CHOFF state"""
        return self._choff_state.copy()
    
    def get_choff_transitions(self) -> List[Dict[str, Any]]:
        """Get list of CHOFF state transitions"""
        return self._transitions.copy()