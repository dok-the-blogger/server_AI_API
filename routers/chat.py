from pydantic import BaseModel
from typing import Optional
from fastapi import APIRouter, Header, HTTPException

from config import settings

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    user_id: int
    profile: Optional[str] = None
    session_id: Optional[str] = None
    context: Optional[dict] = None

class ChatResponse(BaseModel):
    response: str
    session_id: Optional[str] = None

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    authorization: Optional[str] = Header(None),
):
    if settings.API_TOKEN:
        if not authorization or authorization != f"Bearer {settings.API_TOKEN}":
            raise HTTPException(status_code=401, detail="Unauthorized")

    return ChatResponse(response="OK")
