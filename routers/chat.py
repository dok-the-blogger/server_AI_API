from pydantic import BaseModel
from typing import Optional
from fastapi import APIRouter, Header, HTTPException, Request
from gigachat.models import Chat

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
    request_obj: Request,
    request: ChatRequest,
    authorization: Optional[str] = Header(None),
):
    if settings.API_TOKEN:
        if not authorization or authorization != f"Bearer {settings.API_TOKEN}":
            raise HTTPException(status_code=401, detail="Unauthorized")

    if request.profile in (None, "", "default"):
        client = getattr(request_obj.app.state, "gigachat_client", None)
        if client is None:
            raise HTTPException(status_code=500, detail="GigaChat client is not initialized")

        messages = []
        if request.context:
            if "system" in request.context and isinstance(request.context["system"], str):
                messages.append({"role": "system", "content": request.context["system"]})
            if "history" in request.context and isinstance(request.context["history"], list):
                messages.extend(request.context["history"])

        messages.append({"role": "user", "content": request.message})

        try:
            chat_payload = Chat(messages=messages, model=settings.GIGACHAT_MODEL)
            response = await client.achat(chat_payload)
            content = response.choices[0].message.content
            return ChatResponse(response=content, session_id=request.session_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    raise HTTPException(status_code=400, detail=f"Unknown profile: {request.profile}")
