from abc import ABC, abstractmethod
from typing import Any, Callable, List
from threading import Lock

from .events import Event

EventHandler = Callable[[Event], None]

class EventBus(ABC):
    """Abstract interface for event bus implementations"""
    
    @abstractmethod
    def publish(self, event: Event) -> None:
        """Publish an event to all subscribers"""
        pass
    
    @abstractmethod
    def subscribe(self, handler: EventHandler) -> None:
        """Subscribe a handler to receive events"""
        pass
    
    @abstractmethod
    def unsubscribe(self, handler: EventHandler) -> None:
        """Unsubscribe a handler from receiving events"""
        pass

class InMemoryEventBus(EventBus):
    """Simple in-memory implementation of EventBus"""
    
    def __init__(self):
        self._handlers: List[EventHandler] = []
        self._lock = Lock()
    
    def publish(self, event: Event) -> None:
        """Publish event to all subscribers"""
        with self._lock:
            handlers = self._handlers.copy()
        
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                # Log error but continue processing
                print(f"Error in event handler: {e}")
    
    def subscribe(self, handler: EventHandler) -> None:
        """Add a handler to subscribers"""
        with self._lock:
            if handler not in self._handlers:
                self._handlers.append(handler)
    
    def unsubscribe(self, handler: EventHandler) -> None:
        """Remove a handler from subscribers"""
        with self._lock:
            if handler in self._handlers:
                self._handlers.remove(handler)