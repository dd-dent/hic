"""
WebSocket manager for handling real-time connections and event broadcasting.
"""
from typing import Set, Dict, Any
from fastapi import WebSocket, WebSocketDisconnect

from hic.events.schema import BaseEvent, MessageEvent, StateEvent


class WebSocketManager:
    """
    Manages WebSocket connections and handles event broadcasting.
    
    Responsibilities:
    - Connection lifecycle management
    - Event broadcasting
    - Error handling
    """
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
    
    async def connect(self, websocket: WebSocket) -> None:
        """
        Accept a new WebSocket connection and add it to active connections.
        
        Args:
            websocket: The WebSocket connection to manage
        """
        await websocket.accept()
        self.active_connections.add(websocket)
    
    async def disconnect(self, websocket: WebSocket) -> None:
        """
        Remove a WebSocket connection from active connections.
        
        Args:
            websocket: The WebSocket connection to remove
        """
        self.active_connections.remove(websocket)
    
    def _event_to_json(self, event: BaseEvent) -> Dict[str, Any]:
        """
        Convert an event to its JSON representation for transmission.
        
        Args:
            event: The event to convert
            
        Returns:
            Dict containing the event data in a format suitable for JSON transmission
        """
        base_data = {
            "event_id": event.metadata.event_id,
            "timestamp": event.metadata.timestamp,
            "conversation_id": event.metadata.conversation_id,
            "correlation_id": event.metadata.correlation_id,
            "version": event.metadata.version
        }
        
        if isinstance(event, MessageEvent):
            return {
                **base_data,
                "type": "message",
                "content": event.content,
                "source": event.source
            }
        elif isinstance(event, StateEvent):
            return {
                **base_data,
                "type": "state",
                "state_type": event.state_type,
                "intensity": event.intensity,
                "context": event.context
            }
        else:
            raise ValueError(f"Unsupported event type: {type(event)}")
    
    async def broadcast(self, event: BaseEvent) -> None:
        """
        Broadcast an event to all connected clients.
        
        Args:
            event: The event to broadcast
            
        Note:
            Automatically handles disconnected clients by removing them
            from active connections if they fail to receive the message.
        """
        disconnected = set()
        event_data = self._event_to_json(event)
        
        for connection in self.active_connections:
            try:
                await connection.send_json(event_data)
            except WebSocketDisconnect:
                disconnected.add(connection)
            except Exception:
                # Log other errors but keep connection
                # TODO: Add proper logging
                pass
        
        # Remove disconnected clients
        for connection in disconnected:
            self.active_connections.remove(connection)