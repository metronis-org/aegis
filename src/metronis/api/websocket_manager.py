'''
WebSocket Manager - Real-time updates for traces and evaluations
'''

from typing import Dict, Set
from fastapi import WebSocket
import structlog
import json

logger = structlog.get_logger(__name__)


class ConnectionManager:
    '''Manage WebSocket connections for real-time updates.'''

    def __init__(self):
        # organization_id -> set of WebSocket connections
        self.active_connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, organization_id: str):
        '''Accept new WebSocket connection.'''
        await websocket.accept()

        if organization_id not in self.active_connections:
            self.active_connections[organization_id] = set()

        self.active_connections[organization_id].add(websocket)

        logger.info(
            'WebSocket connected',
            organization_id=organization_id,
            total_connections=len(self.active_connections[organization_id]),
        )

    def disconnect(self, websocket: WebSocket, organization_id: str):
        '''Remove disconnected WebSocket.'''
        if organization_id in self.active_connections:
            self.active_connections[organization_id].discard(websocket)

            if not self.active_connections[organization_id]:
                del self.active_connections[organization_id]

            logger.info(
                'WebSocket disconnected',
                organization_id=organization_id,
                remaining_connections=len(self.active_connections.get(organization_id, [])),
            )

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        '''Send message to specific WebSocket.'''
        await websocket.send_json(message)

    async def broadcast_to_organization(self, message: dict, organization_id: str):
        '''Broadcast message to all connections for an organization.'''
        if organization_id not in self.active_connections:
            return

        connections = list(self.active_connections[organization_id])
        for connection in connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(
                    'Failed to send message',
                    organization_id=organization_id,
                    error=str(e),
                )
                self.disconnect(connection, organization_id)

    async def broadcast_trace_update(self, trace_data: dict, organization_id: str):
        '''Broadcast trace update to organization.'''
        message = {
            'type': 'trace_update',
            'data': trace_data,
        }
        await self.broadcast_to_organization(message, organization_id)

    async def broadcast_evaluation_complete(self, evaluation_data: dict, organization_id: str):
        '''Broadcast evaluation completion.'''
        message = {
            'type': 'evaluation_complete',
            'data': evaluation_data,
        }
        await self.broadcast_to_organization(message, organization_id)


# Global manager instance
manager = ConnectionManager()
