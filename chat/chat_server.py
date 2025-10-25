import asyncio
import json
import logging
from typing import Dict, Set, List, Optional
from fastapi import WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
from datetime import datetime
import uuid

from Database.DBConfiguration.database import get_db
from Database.DBmodels.extended_database_models import ChatRoom, ChatMessage

logger = logging.getLogger(__name__)

class ChatServer:
    def __init__(self):
        # Store active connections: room_id -> set of websockets
        self.active_connections: Dict[int, Set[WebSocket]] = {}
        # Store user info for each connection: websocket -> user_info
        self.connection_users: Dict[WebSocket, Dict] = {}
        # Store user connections: user_id -> set of websockets
        self.user_connections: Dict[int, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, room_id: int, user_id: int, user_name: str):
        """Connect a user to a chat room's session"""
        await websocket.accept()

        # Initialize connections set for this room if not exists
        if room_id not in self.active_connections:
            self.active_connections[room_id] = set()

        # Add connection
        self.active_connections[room_id].add(websocket)
        self.connection_users[websocket] = {
            "user_id": user_id,
            "user_name": user_name,
            "room_id": room_id
        }

        # Add to user connections
        if user_id not in self.user_connections:
            self.user_connections[user_id] = set()
        self.user_connections[user_id].add(websocket)

        logger.info(f"User {user_name} (ID: {user_id}) connected to chat room {room_id}")

        # Notify other users about the new connection
        await self.broadcast_to_room(
            room_id,
            {
                "type": "user_joined",
                "user_id": user_id,
                "user_name": user_name,
                "timestamp": datetime.utcnow().isoformat()
            },
            exclude_websocket=websocket
        )

        # Send current users list to the new user
        current_users = [
            self.connection_users[ws]["user_name"]
            for ws in self.active_connections[room_id]
            if ws != websocket
        ]
        await websocket.send_json({
            "type": "users_online",
            "users": current_users
        })

    async def disconnect(self, websocket: WebSocket):
        """Disconnect a user from chat"""
        user_info = self.connection_users.get(websocket)
        if not user_info:
            return

        room_id = user_info["room_id"]
        user_id = user_info["user_id"]
        user_name = user_info["user_name"]

        # Remove connection
        if room_id in self.active_connections:
            self.active_connections[room_id].discard(websocket)

            # Clean up empty connection sets
            if not self.active_connections[room_id]:
                del self.active_connections[room_id]

        # Remove user info
        if websocket in self.connection_users:
            del self.connection_users[websocket]

        # Remove from user connections
        if user_id in self.user_connections:
            self.user_connections[user_id].discard(websocket)
            if not self.user_connections[user_id]:
                del self.user_connections[user_id]

        logger.info(f"User {user_name} (ID: {user_id}) disconnected from chat room {room_id}")

        # Notify other users about the disconnection
        await self.broadcast_to_room(
            room_id,
            {
                "type": "user_left",
                "user_id": user_id,
                "user_name": user_name,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

    async def handle_message(self, websocket: WebSocket, message: str):
        """Handle incoming WebSocket messages (for future extensions, e.g., typing indicators)"""
        try:
            data = json.loads(message)
            message_type = data.get("type")

            if message_type == "ping":
                await websocket.send_json({"type": "pong"})
            # Add other message types as needed (e.g., typing)

        except json.JSONDecodeError:
            logger.error("Invalid JSON received")
        except Exception as e:
            logger.error(f"Error handling message: {e}")

    async def broadcast_message(self, room_id: int, message_data: dict, exclude_websocket: Optional[WebSocket] = None):
        """Broadcast a new chat message to all users in a room"""
        if room_id not in self.active_connections:
            return

        disconnected = []
        for websocket in self.active_connections[room_id]:
            if websocket == exclude_websocket:
                continue

            try:
                await websocket.send_json(message_data)
            except Exception as e:
                logger.error(f"Error sending message to websocket: {e}")
                disconnected.append(websocket)

        # Clean up disconnected websockets
        for websocket in disconnected:
            await self.disconnect(websocket)

    async def get_online_users(self, room_id: int) -> List[str]:
        """Get list of online users for a room"""
        if room_id not in self.active_connections:
            return []

        return [
            self.connection_users[ws]["user_name"]
            for ws in self.active_connections[room_id]
        ]

# Global chat server instance
chat_server = ChatServer()

async def handle_chat_websocket(websocket: WebSocket, room_id: int, user_id: int, user_name: str):
    """WebSocket endpoint handler for chat"""
    await chat_server.connect(websocket, room_id, user_id, user_name)

    try:
        while True:
            message = await websocket.receive_text()
            await chat_server.handle_message(websocket, message)
    except WebSocketDisconnect:
        pass
    finally:
        await chat_server.disconnect(websocket)
