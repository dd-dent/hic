"""WebSocket manager tests for CHOFF-aware event broadcasting."""
import pytest
import asyncio
from fastapi import WebSocket, WebSocketDisconnect
from unittest.mock import MagicMock

from hic.api.websocket_manager import WebSocketManager
from hic.events.schema import MessageEvent, StateEvent, EventMetadata

class MockWebSocket:
    """Mock WebSocket with synchronous tracking for testing."""
    def __init__(self):
        self.client = MagicMock()
        self.client.host = "127.0.0.1"
        self.client.port = 8000
        self._accepted = False
        self._sent_messages = []
        self._closed = False
        
    def __eq__(self, other):
        return id(self) == id(other)
        
    def __hash__(self):
        return id(self)

    async def accept(self):
        await asyncio.sleep(0)  # Use asyncio sleep instead of trio
        self._accepted = True

    async def send_json(self, data: dict):
        await asyncio.sleep(0)  # Use asyncio sleep instead of trio
        if self._closed:
            raise WebSocketDisconnect()
        self._sent_messages.append(data)

    def get_sent_messages(self):
        return self._sent_messages

    def disconnect(self):
        self._closed = True

@pytest.fixture
async def manager():
    """Create a fresh manager for each test."""
    return WebSocketManager()

@pytest.fixture
async def websocket():
    """Create a fresh WebSocket mock for each test."""
    return MockWebSocket()

@pytest.mark.asyncio
async def test_connect_client(manager, websocket):
    """Test connecting a new WebSocket client"""
    await manager.connect(websocket)
    assert websocket._accepted
    assert websocket in manager.active_connections
    assert len(manager.active_connections) == 1

@pytest.mark.asyncio
async def test_disconnect_client(manager, websocket):
    """Test disconnecting a WebSocket client"""
    await manager.connect(websocket)
    await manager.disconnect(websocket)
    assert websocket not in manager.active_connections
    assert len(manager.active_connections) == 0

@pytest.mark.asyncio
async def test_broadcast_message(manager, websocket):
    """Test broadcasting a message event to all connected clients"""
    await manager.connect(websocket)
    
    message_event = MessageEvent.create(
        content="Test message",
        conversation_id="test-conv",
        source="test-source",
        correlation_id="test-correlation"
    )
    
    await manager.broadcast(message_event)
    messages = websocket.get_sent_messages()
    assert len(messages) == 1
    broadcast_data = messages[0]
    
    assert broadcast_data["type"] == "message"
    assert broadcast_data["content"] == "Test message"
    assert broadcast_data["conversation_id"] == "test-conv"
    assert broadcast_data["source"] == "test-source"
    assert broadcast_data["correlation_id"] == "test-correlation"
    assert "event_id" in broadcast_data
    assert "timestamp" in broadcast_data
    assert "version" in broadcast_data

@pytest.mark.asyncio
async def test_broadcast_state_change(manager, websocket):
    """Test broadcasting a CHOFF state change event"""
    await manager.connect(websocket)
    
    state_event = StateEvent.create(
        state_type="analytical",
        intensity=0.8,
        conversation_id="test-conv",
        context="test-context",
        correlation_id="test-correlation"
    )
    
    await manager.broadcast(state_event)
    messages = websocket.get_sent_messages()
    assert len(messages) == 1
    broadcast_data = messages[0]
    
    assert broadcast_data["type"] == "state"
    assert broadcast_data["state_type"] == "analytical"
    assert broadcast_data["intensity"] == 0.8
    assert broadcast_data["context"] == "test-context"
    assert broadcast_data["conversation_id"] == "test-conv"
    assert broadcast_data["correlation_id"] == "test-correlation"
    assert "event_id" in broadcast_data
    assert "timestamp" in broadcast_data
    assert "version" in broadcast_data

@pytest.mark.asyncio
async def test_handle_client_error(manager, websocket):
    """Test handling WebSocket client errors"""
    await manager.connect(websocket)
    websocket.disconnect()  # Simulate disconnection
    
    message_event = MessageEvent.create(
        content="Test message",
        conversation_id="test-conv"
    )
    
    # Should handle error and remove connection
    await manager.broadcast(message_event)
    assert websocket not in manager.active_connections

@pytest.mark.asyncio
async def test_multiple_clients(manager):
    """Test managing multiple WebSocket clients"""
    ws1 = MockWebSocket()
    ws2 = MockWebSocket()
    
    await manager.connect(ws1)
    await manager.connect(ws2)
    assert len(manager.active_connections) == 2
    
    message_event = MessageEvent.create(
        content="Test message",
        conversation_id="test-conv"
    )
    
    await manager.broadcast(message_event)
    assert len(ws1.get_sent_messages()) == 1
    assert len(ws2.get_sent_messages()) == 1

@pytest.mark.asyncio
async def test_client_equality(manager):
    """Test that WebSocket connections are properly tracked"""
    ws1 = MockWebSocket()
    ws2 = MockWebSocket()
    
    await manager.connect(ws1)
    await manager.connect(ws2)
    await manager.disconnect(ws1)
    
    assert ws1 not in manager.active_connections
    assert ws2 in manager.active_connections
    assert len(manager.active_connections) == 1