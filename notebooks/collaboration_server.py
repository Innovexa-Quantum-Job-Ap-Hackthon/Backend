import asyncio
import json
import logging
from typing import Dict, Set, List, Optional, Any
from fastapi import WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
from datetime import datetime
import uuid

from notebooks.notebook_service import NotebookService
from Database.DBConfiguration.database import get_db

logger = logging.getLogger(__name__)

class NotebookCollaborationServer:
    def __init__(self):
        # Store active connections: notebook_id -> set of websockets
        self.active_connections: Dict[int, Set[WebSocket]] = {}
        # Store user info for each connection: websocket -> user_info
        self.connection_users: Dict[WebSocket, Dict] = {}
        # Store notebook locks: notebook_id -> user_id (who has the lock)
        self.notebook_locks: Dict[int, int] = {}
        # Store user connections: user_id -> set of websockets
        self.user_connections: Dict[int, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, notebook_id: int, user_id: int, user_name: str):
        """Connect a user to a notebook's collaboration session"""
        await websocket.accept()

        # Initialize connections set for this notebook if not exists
        if notebook_id not in self.active_connections:
            self.active_connections[notebook_id] = set()

        # Add connection
        self.active_connections[notebook_id].add(websocket)
        self.connection_users[websocket] = {
            "user_id": user_id,
            "user_name": user_name,
            "notebook_id": notebook_id
        }

        # Add to user connections
        if user_id not in self.user_connections:
            self.user_connections[user_id] = set()
        self.user_connections[user_id].add(websocket)

        logger.info(f"User {user_name} (ID: {user_id}) connected to notebook {notebook_id}")

        # Notify other users about the new connection
        await self.broadcast_to_notebook(
            notebook_id,
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
            self.connection_users[ws]
            for ws in self.active_connections[notebook_id]
            if ws != websocket
        ]
        await websocket.send_json({
            "type": "users_online",
            "users": current_users
        })

    async def disconnect(self, websocket: WebSocket):
        """Disconnect a user from collaboration"""
        user_info = self.connection_users.get(websocket)
        if not user_info:
            return

        notebook_id = user_info["notebook_id"]
        user_id = user_info["user_id"]
        user_name = user_info["user_name"]

        # Remove connection
        if notebook_id in self.active_connections:
            self.active_connections[notebook_id].discard(websocket)

            # Clean up empty connection sets
            if not self.active_connections[notebook_id]:
                del self.active_connections[notebook_id]

        # Remove user info
        del self.connection_users[websocket]

        # Remove from user connections
        if user_id in self.user_connections:
            self.user_connections[user_id].discard(websocket)
            if not self.user_connections[user_id]:
                del self.user_connections[user_id]

        # Release lock if user had it
        if self.notebook_locks.get(notebook_id) == user_id:
            del self.notebook_locks[notebook_id]

        logger.info(f"User {user_name} (ID: {user_id}) disconnected from notebook {notebook_id}")

        # Notify other users about the disconnection
        await self.broadcast_to_notebook(
            notebook_id,
            {
                "type": "user_left",
                "user_id": user_id,
                "user_name": user_name,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

    async def handle_message(self, websocket: WebSocket, message: str):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)
            message_type = data.get("type")
            user_info = self.connection_users.get(websocket)

            if not user_info:
                return

            notebook_id = user_info["notebook_id"]
            user_id = user_info["user_id"]

            if message_type == "cell_update":
                await self.handle_cell_update(notebook_id, user_id, data, websocket)
            elif message_type == "cell_create":
                await self.handle_cell_create(notebook_id, user_id, data, websocket)
            elif message_type == "cell_delete":
                await self.handle_cell_delete(notebook_id, user_id, data, websocket)
            elif message_type == "cell_reorder":
                await self.handle_cell_reorder(notebook_id, user_id, data, websocket)
            elif message_type == "cursor_position":
                await self.handle_cursor_position(notebook_id, user_id, data, websocket)
            elif message_type == "request_lock":
                await self.handle_request_lock(notebook_id, user_id, data, websocket)
            elif message_type == "release_lock":
                await self.handle_release_lock(notebook_id, user_id, data, websocket)
            elif message_type == "cell_execute":
                await self.handle_cell_execute(notebook_id, user_id, data, websocket)
            elif message_type == "execution_result":
                await self.handle_execution_result(notebook_id, user_id, data, websocket)
            elif message_type == "output_update":
                await self.handle_output_update(notebook_id, user_id, data, websocket)
            elif message_type == "kernel_restart":
                await self.handle_kernel_restart(notebook_id, user_id, data, websocket)
            elif message_type == "kernel_interrupt":
                await self.handle_kernel_interrupt(notebook_id, user_id, data, websocket)
            elif message_type == "ping":
                await websocket.send_json({"type": "pong"})

        except json.JSONDecodeError:
            logger.error("Invalid JSON received")
        except Exception as e:
            logger.error(f"Error handling message: {e}")

    async def handle_cell_update(self, notebook_id: int, user_id: int, data: dict, websocket: WebSocket):
        """Handle cell content update"""
        cell_id = data.get("cell_id")
        content = data.get("content")

        if not cell_id or content is None:
            return

        # Check if user has lock or if no lock is required
        current_lock = self.notebook_locks.get(notebook_id)
        if current_lock and current_lock != user_id:
            await websocket.send_json({
                "type": "lock_denied",
                "cell_id": cell_id
            })
            return

        # Persist the cell update using NotebookService
        try:
            db = get_db()
            service = NotebookService(db)
            service.update_cell(cell_id, content, user_id)
        except Exception as e:
            logger.error(f"Failed to update cell {cell_id}: {e}")
            await websocket.send_json({
                "type": "error",
                "message": f"Failed to update cell {cell_id}"
            })
            return

        # Broadcast the update to other users
        await self.broadcast_to_notebook(
            notebook_id,
            {
                "type": "cell_updated",
                "cell_id": cell_id,
                "content": content,
                "user_id": user_id,
                "timestamp": datetime.utcnow().isoformat()
            },
            exclude_websocket=websocket
        )

    async def handle_cell_create(self, notebook_id: int, user_id: int, data: dict, websocket: WebSocket):
        """Handle cell creation"""
        cell_type = data.get("cell_type", "code")
        content = data.get("content", "")
        order = data.get("order", 0)

        # Persist the cell creation using NotebookService
        try:
            db = get_db()
            service = NotebookService(db)
            service.create_cell(notebook_id, cell_type, content, order, user_id)
        except Exception as e:
            logger.error(f"Failed to create cell in notebook {notebook_id}: {e}")
            await websocket.send_json({
                "type": "error",
                "message": f"Failed to create cell in notebook {notebook_id}"
            })
            return

        # Broadcast cell creation to other users
        await self.broadcast_to_notebook(
            notebook_id,
            {
                "type": "cell_created",
                "cell_type": cell_type,
                "content": content,
                "order": order,
                "user_id": user_id,
                "timestamp": datetime.utcnow().isoformat()
            },
            exclude_websocket=websocket
        )

    async def handle_cell_delete(self, notebook_id: int, user_id: int, data: dict, websocket: WebSocket):
        """Handle cell deletion"""
        cell_id = data.get("cell_id")

        if not cell_id:
            return

        # Persist the cell deletion using NotebookService
        try:
            db = get_db()
            service = NotebookService(db)
            service.delete_cell(cell_id, user_id)
        except Exception as e:
            logger.error(f"Failed to delete cell {cell_id}: {e}")
            await websocket.send_json({
                "type": "error",
                "message": f"Failed to delete cell {cell_id}"
            })
            return

        # Broadcast cell deletion to other users
        await self.broadcast_to_notebook(
            notebook_id,
            {
                "type": "cell_deleted",
                "cell_id": cell_id,
                "user_id": user_id,
                "timestamp": datetime.utcnow().isoformat()
            },
            exclude_websocket=websocket
        )

    async def handle_cell_reorder(self, notebook_id: int, user_id: int, data: dict, websocket: WebSocket):
        """Handle cell reordering"""
        cell_orders = data.get("cell_orders", [])

        # Persist the cell reordering using NotebookService
        try:
            db = get_db()
            service = NotebookService(db)
            service.reorder_cells(notebook_id, cell_orders, user_id)
        except Exception as e:
            logger.error(f"Failed to reorder cells in notebook {notebook_id}: {e}")
            await websocket.send_json({
                "type": "error",
                "message": f"Failed to reorder cells in notebook {notebook_id}"
            })
            return

        # Broadcast cell reordering to other users
        await self.broadcast_to_notebook(
            notebook_id,
            {
                "type": "cells_reordered",
                "cell_orders": cell_orders,
                "user_id": user_id,
                "timestamp": datetime.utcnow().isoformat()
            },
            exclude_websocket=websocket
        )

    async def handle_cursor_position(self, notebook_id: int, user_id: int, data: dict, websocket: WebSocket):
        """Handle cursor position updates for collaborative editing"""
        cell_id = data.get("cell_id")
        position = data.get("position")

        # Broadcast cursor position to other users
        await self.broadcast_to_notebook(
            notebook_id,
            {
                "type": "cursor_moved",
                "cell_id": cell_id,
                "position": position,
                "user_id": user_id,
                "user_name": self.connection_users[websocket]["user_name"]
            },
            exclude_websocket=websocket
        )

    async def handle_request_lock(self, notebook_id: int, user_id: int, data: dict, websocket: WebSocket):
        """Handle lock request for exclusive editing"""
        current_lock = self.notebook_locks.get(notebook_id)

        if current_lock is None or current_lock == user_id:
            # Grant lock
            self.notebook_locks[notebook_id] = user_id
            await websocket.send_json({
                "type": "lock_granted",
                "notebook_id": notebook_id
            })

            # Notify others that lock was taken
            await self.broadcast_to_notebook(
                notebook_id,
                {
                    "type": "lock_taken",
                    "user_id": user_id,
                    "user_name": self.connection_users[websocket]["user_name"]
                },
                exclude_websocket=websocket
            )
        else:
            # Deny lock
            await websocket.send_json({
                "type": "lock_denied",
                "notebook_id": notebook_id,
                "current_holder": current_lock
            })

    async def handle_release_lock(self, notebook_id: int, user_id: int, data: dict, websocket: WebSocket):
        """Handle lock release"""
        current_lock = self.notebook_locks.get(notebook_id)

        if current_lock == user_id:
            del self.notebook_locks[notebook_id]

            # Notify others that lock was released
            await self.broadcast_to_notebook(
                notebook_id,
                {
                    "type": "lock_released",
                    "user_id": user_id,
                    "user_name": self.connection_users[websocket]["user_name"]
                }
            )

    async def handle_cell_execute(self, notebook_id: int, user_id: int, data: dict, websocket: WebSocket):
        """Handle cell execution request"""
        cell_id = data.get("cell_id")

        if not cell_id:
            return

        # Broadcast execution start to other users
        await self.broadcast_to_notebook(
            notebook_id,
            {
                "type": "cell_executing",
                "cell_id": cell_id,
                "user_id": user_id,
                "user_name": self.connection_users[websocket]["user_name"],
                "timestamp": datetime.utcnow().isoformat()
            },
            exclude_websocket=websocket
        )

    async def handle_execution_result(self, notebook_id: int, user_id: int, data: dict, websocket: WebSocket):
        """Handle execution result from kernel"""
        cell_id = data.get("cell_id")
        execution_count = data.get("execution_count")
        outputs = data.get("outputs", [])
        success = data.get("success", True)
        error = data.get("error")

        if not cell_id:
            return

        # Persist execution result if needed (optional, for history)
        # For now, just broadcast to other users
        await self.broadcast_to_notebook(
            notebook_id,
            {
                "type": "execution_completed",
                "cell_id": cell_id,
                "execution_count": execution_count,
                "outputs": outputs,
                "success": success,
                "error": error,
                "user_id": user_id,
                "timestamp": datetime.utcnow().isoformat()
            },
            exclude_websocket=websocket
        )

    async def handle_output_update(self, notebook_id: int, user_id: int, data: dict, websocket: WebSocket):
        """Handle streaming output updates during execution"""
        cell_id = data.get("cell_id")
        output = data.get("output")
        output_type = data.get("output_type", "stream")

        if not cell_id or not output:
            return

        # Broadcast output update to other users
        await self.broadcast_to_notebook(
            notebook_id,
            {
                "type": "output_updated",
                "cell_id": cell_id,
                "output": output,
                "output_type": output_type,
                "user_id": user_id,
                "timestamp": datetime.utcnow().isoformat()
            },
            exclude_websocket=websocket
        )

    async def handle_kernel_restart(self, notebook_id: int, user_id: int, data: dict, websocket: WebSocket):
        """Handle kernel restart request"""
        # Broadcast kernel restart to all users
        await self.broadcast_to_notebook(
            notebook_id,
            {
                "type": "kernel_restarted",
                "user_id": user_id,
                "user_name": self.connection_users[websocket]["user_name"],
                "timestamp": datetime.utcnow().isoformat()
            }
        )

    async def handle_kernel_interrupt(self, notebook_id: int, user_id: int, data: dict, websocket: WebSocket):
        """Handle kernel interrupt request"""
        # Broadcast kernel interrupt to all users
        await self.broadcast_to_notebook(
            notebook_id,
            {
                "type": "kernel_interrupted",
                "user_id": user_id,
                "user_name": self.connection_users[websocket]["user_name"],
                "timestamp": datetime.utcnow().isoformat()
            }
        )

    async def broadcast_to_notebook(self, notebook_id: int, message: dict, exclude_websocket: WebSocket = None):
        """Broadcast a message to all users in a notebook"""
        if notebook_id not in self.active_connections:
            return

        disconnected = []
        for websocket in self.active_connections[notebook_id]:
            if websocket == exclude_websocket:
                continue

            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error sending message to websocket: {e}")
                disconnected.append(websocket)

        # Clean up disconnected websockets
        for websocket in disconnected:
            await self.disconnect(websocket)

    async def broadcast_to_team(self, team_id: int, message: dict, exclude_user_id: int = None):
        """Broadcast a message to all online users in a team"""
        from Database.DBmodels.database_models import TeamMember

        # Get a new db session
        db = get_db()
        try:
            # Get team members
            team_members = db.query(TeamMember).filter(TeamMember.team_id == team_id).all()
            team_user_ids = {tm.user_id for tm in team_members}

            for user_id in team_user_ids:
                if user_id == exclude_user_id:
                    continue
                if user_id in self.user_connections:
                    for ws in self.user_connections[user_id]:
                        try:
                            await ws.send_json(message)
                        except Exception as e:
                            logger.error(f"Error sending to user {user_id}: {e}")
        finally:
            db.close()

    async def get_online_users(self, notebook_id: int) -> List[Dict]:
        """Get list of online users for a notebook"""
        if notebook_id not in self.active_connections:
            return []

        return [
            self.connection_users[ws]
            for ws in self.active_connections[notebook_id]
        ]

# Global collaboration server instance
collaboration_server = NotebookCollaborationServer()

async def handle_notebook_websocket(websocket: WebSocket, notebook_id: int, user_id: int, user_name: str):
    """WebSocket endpoint handler for notebook collaboration"""
    await collaboration_server.connect(websocket, notebook_id, user_id, user_name)

    try:
        while True:
            message = await websocket.receive_text()
            await collaboration_server.handle_message(websocket, message)
    except WebSocketDisconnect:
        pass
    finally:
        await collaboration_server.disconnect(websocket)
