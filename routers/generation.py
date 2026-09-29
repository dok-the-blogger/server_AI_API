from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Request

from completion_providers import CompletionError
from config import settings
from generation import GenerateRequest, GenerateResponse, generate
from routers.embeddings import EmbeddingsRoute

router = APIRouter(route_class=EmbeddingsRoute)


@router.post("/generate", response_model=GenerateResponse)
async def generate_text(request_obj: Request, request: GenerateRequest,
                        authorization: Optional[str] = Header(None)):
    if settings.API_TOKEN and authorization != f"Bearer {settings.API_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    client = getattr(request_obj.app.state, "summaries_client", None)
    if client is None:
        raise HTTPException(status_code=503, detail={
            "code": "generation_not_configured", "message": "Generation provider is not configured"})
    try:
        return await generate(client, request)
    except CompletionError as error:
        raise HTTPException(status_code=error.status_code,
                            detail={"code": error.code, "message": error.message}) from None
