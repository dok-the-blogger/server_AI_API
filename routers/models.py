from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel
from typing import Literal, Optional

from config import settings
from completion_providers import MODEL_PROFILES

router = APIRouter()

class ModelInfo(BaseModel):
    id: str
    provider: str
    owned_by: str

class ModelsResponse(BaseModel):
    models: list[ModelInfo]
    provider: str
    current_model: str

@router.get("/models", response_model=ModelsResponse)
async def get_models(
    request_obj: Request,
    authorization: Optional[str] = Header(None),
    provider: Literal["gigachat", "mimo"] = "gigachat",
):
    if settings.API_TOKEN:
        if not authorization or authorization != f"Bearer {settings.API_TOKEN}":
            raise HTTPException(status_code=401, detail="Unauthorized")

    if provider == "mimo":
        if getattr(request_obj.app.state, "mimo_client", None) is None:
            raise HTTPException(status_code=503, detail={
                "code": "chat_not_configured", "message": "MiMo provider is not configured"})
        return ModelsResponse(
            models=[ModelInfo(id=p.model, provider="mimo", owned_by="xiaomi")
                    for p in MODEL_PROFILES.values() if p.provider == "mimo"],
            provider="mimo", current_model=settings.MIMO_MODEL)

    client = getattr(request_obj.app.state, "gigachat_client", None)
    if client is None:
        raise HTTPException(status_code=500, detail="GigaChat client is not initialized")

    try:
        response = await client.aget_models()
        models = [
            ModelInfo(
                id=model.id_,
                provider="gigachat",
                owned_by=model.owned_by
            )
            for model in response.data
        ]

        return ModelsResponse(
            models=models,
            provider="gigachat",
            current_model=settings.GIGACHAT_MODEL
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
