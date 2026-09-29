from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Request

from classification import ClassificationError, ClassificationRequest, ClassificationResponse
from config import settings
from routers.embeddings import EmbeddingsRoute

router = APIRouter(route_class=EmbeddingsRoute)


@router.post("/classify/dokbot", response_model=ClassificationResponse)
async def classify_dokbot(request_obj: Request, request: ClassificationRequest,
                         authorization: Optional[str] = Header(None)):
    if settings.API_TOKEN and authorization != f"Bearer {settings.API_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    client = getattr(request_obj.app.state, "jev_client", None)
    if client is None:
        raise HTTPException(status_code=503, detail={
            "code": "classification_not_configured", "message": "Jev is not configured"})
    try:
        return await client.classify(request.text)
    except ClassificationError as error:
        raise HTTPException(status_code=error.status_code,
                            detail={"code": error.code, "message": error.message}) from None
