from pydantic import BaseModel
from typing import Optional
import json
from fastapi import APIRouter, Header, HTTPException, Request
from gigachat.models import Chat

from config import settings
from profiles import get_system_prompt, get_fallback_prompt

router = APIRouter()

def parse_json_reply(text: str) -> str:
    """Извлекает поле reply из JSON-ответа модели. При ошибке возвращает raw text."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        data = json.loads(text)
        return data.get("reply", text)
    except (json.JSONDecodeError, AttributeError):
        return text

class ChatRequest(BaseModel):
    message: str
    user_id: int
    profile: Optional[str] = None
    session_id: Optional[str] = None
    context: Optional[dict] = None

class ChatResponse(BaseModel):
    response: str
    session_id: Optional[str] = None
    filtered: bool = False

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request_obj: Request,
    request: ChatRequest,
    authorization: Optional[str] = Header(None),
):
    if settings.API_TOKEN:
        if not authorization or authorization != f"Bearer {settings.API_TOKEN}":
            raise HTTPException(status_code=401, detail="Unauthorized")

    client = getattr(request_obj.app.state, "gigachat_client", None)
    if client is None:
        raise HTTPException(status_code=500, detail="GigaChat client is not initialized")

    system_prompt = None

    if request.profile:
        system_prompt = get_system_prompt(request.profile)
        if system_prompt is None and request.profile not in (None, "", "default"):
            raise HTTPException(status_code=400, detail=f"Unknown profile: {request.profile}")

    if request.context and "system" in request.context and isinstance(request.context["system"], str):
        system_prompt = request.context["system"]

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if request.context and "history" in request.context and isinstance(request.context["history"], list):
        messages.extend(request.context["history"])

    messages.append({"role": "user", "content": request.message})

    try:
        chat_payload = Chat(messages=messages, model=settings.GIGACHAT_MODEL)
        response = await client.achat(chat_payload)

        if response.choices[0].finish_reason == "blacklist":
            if request.profile == "dokbot":
                fallback_prompt = get_fallback_prompt(request.profile)
                if fallback_prompt:
                    wrapped_messages = [
                        {"role": "system", "content": fallback_prompt},
                        {"role": "user", "content": f"Пользователь чата спрашивает: «{request.message}»\nСгенерируй ответ персонажа. Только JSON."},
                    ]
                    fallback_payload = Chat(messages=wrapped_messages, model=settings.GIGACHAT_MODEL)
                    fallback_response = await client.achat(fallback_payload)

                    if fallback_response.choices[0].finish_reason == "blacklist":
                        return ChatResponse(response="", session_id=request.session_id, filtered=True)

                    content = fallback_response.choices[0].message.content
                    reply = parse_json_reply(content)
                    return ChatResponse(response=reply, session_id=request.session_id)

            return ChatResponse(response="", session_id=request.session_id, filtered=True)

        content = response.choices[0].message.content
        return ChatResponse(response=content, session_id=request.session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
