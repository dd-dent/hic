import sqlite3
from datetime import datetime, timezone
from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Dict, Set
import json
from pathlib import Path

class Speaker(Enum):
    USER = auto()
    ASSISTANT = auto()

@dataclass
class Message:
    """A chat message with CHOFF metadata."""
    speaker: Speaker
    content: str
    choff_tags: List[str]
    timestamp: datetime

class MessageStore:
    """SQLite-based storage for chat messages with CHOFF tag support."""
    
    def __init__(self, db_path: str = None):
        """Initialize the message store with optional custom database path."""
        if db_path is None:
            # Use a default path in the user's home directory
            db_path = str(Path.home() / ".hic" / "messages.db")
        
        # Ensure the directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self._db_path = db_path
        self._init_db()
        
        # Register adapters for datetime handling
        sqlite3.register_adapter(datetime, self._adapt_datetime)
        sqlite3.register_converter("timestamp", self._convert_timestamp)
    
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
        # Enable foreign key support
        conn.execute("PRAGMA foreign_keys = ON")
        return conn
    
    def _init_db(self):
        """Initialize the database schema if it doesn't exist."""
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    speaker TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp timestamp NOT NULL
                );
                
                CREATE TABLE IF NOT EXISTS tags (
                    message_id INTEGER,
                    tag TEXT NOT NULL,
                    tag_order INTEGER NOT NULL,
                    FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE,
                    PRIMARY KEY (message_id, tag, tag_order)
                );
                
                CREATE INDEX IF NOT EXISTS idx_tags_tag ON tags(tag);
            """)
    
    def __len__(self) -> int:
        with self._get_connection() as conn:
            return conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    
    def add(self, message: Message) -> int:
        """Add a message and return its ID."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "INSERT INTO messages (speaker, content, timestamp) VALUES (?, ?, ?)",
                (message.speaker.name, message.content, message.timestamp)
            )
            message_id = cursor.lastrowid
            
            # Insert tags preserving order and duplicates
            if message.choff_tags:
                conn.executemany(
                    "INSERT INTO tags (message_id, tag, tag_order) VALUES (?, ?, ?)",
                    [(message_id, tag, i) for i, tag in enumerate(message.choff_tags)]
                )
            
            return message_id
    
    def get(self, message_id: int) -> Message:
        """Retrieve a message by ID."""
        with self._get_connection() as conn:
            # Get message
            cursor = conn.execute(
                "SELECT speaker, content, timestamp FROM messages WHERE id = ?",
                (message_id,)
            )
            row = cursor.fetchone()
            if row is None:
                raise KeyError(f"Message {message_id} not found")
            
            # Get tags in original order
            tags = [
                tag for (tag,) in conn.execute(
                    "SELECT tag FROM tags WHERE message_id = ? ORDER BY tag_order",
                    (message_id,)
                )
            ]
            
            return Message(
                speaker=Speaker[row[0]],
                content=row[1],
                choff_tags=tags,
                timestamp=row[2]
            )
    
    def delete(self, message_id: int) -> None:
        """Delete a message by ID."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM messages WHERE id = ?",
                (message_id,)
            )
            if cursor.rowcount == 0:
                raise KeyError(f"Message {message_id} not found")
            # Tags are automatically deleted due to ON DELETE CASCADE
    
    def find_by_choff_tag(self, tag: str) -> List[Message]:
        """Find all messages with a specific CHOFF tag."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT DISTINCT m.id, m.speaker, m.content, m.timestamp
                FROM messages m
                JOIN tags t ON m.id = t.message_id
                WHERE t.message_id IN (
                    SELECT message_id FROM tags WHERE tag = ?
                )
                ORDER BY m.id
            """, (tag,))
            
            messages = []
            for row in cursor:
                # Get tags for this message in original order
                tags = [
                    tag for (tag,) in conn.execute(
                        "SELECT tag FROM tags WHERE message_id = ? ORDER BY tag_order",
                        (row[0],)
                    )
                ]
                
                messages.append(Message(
                    speaker=Speaker[row[1]],
                    content=row[2],
                    choff_tags=tags,
                    timestamp=row[3]
                ))
            
            return messages