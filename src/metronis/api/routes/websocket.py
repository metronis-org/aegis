'''
WebSocket API Routes
'''

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query
from sqlalchemy.orm import Session

from metronis.db.session import get_db
from metronis.db.models import OrganizationModel
from metronis.infrastructure.repositories.organization_repository import OrganizationRepository
from metronis.api.websocket_manager import manager
import structlog

logger = structlog.get_logger(__name__)

router = APIRouter()


@router.websocket('/ws/traces')
async def websocket_traces(
    websocket: WebSocket,
    api_key: str = Query(...),
    db: Session = Depends(get_db),
):
    '''WebSocket endpoint for real-time trace updates.'''

    # Authenticate via API key
    org_repo = OrganizationRepository(db)
    organization = org_repo.get_by_api_key(api_key)

    if not organization:
        await websocket.close(code=1008, reason='Invalid API key')
        return

    organization_id = str(organization.organization_id)
    await manager.connect(websocket, organization_id)

    try:
        # Keep connection alive and handle incoming messages
        while True:
            data = await websocket.receive_text()
            # Echo back (can handle commands here)
            await manager.send_personal_message(
                {'type': 'ping', 'message': 'pong'},
                websocket,
            )

    except WebSocketDisconnect:
        manager.disconnect(websocket, organization_id)
        logger.info('WebSocket disconnected', organization_id=organization_id)
