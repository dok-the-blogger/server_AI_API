from pydantic import BaseModel
from typing import Optional
import json
from fastapi import APIRouter, Header, HTTPException, Request
from gigachat.models import Chat
import logging

from config import settings
from profiles import (
    get_system_prompt,
    get_fallback_prompt,
    get_grok_system_prompt,
    get_grok_few_shot,
    get_meta_system_prompt,
    get_provider,
    get_user_template,
)

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
    user_id: Optional[int] = None
    profile: Optional[str] = None
    session_id: Optional[str] = None
    context: Optional[dict] = None

class ChatResponse(BaseModel):
    response: str
    session_id: Optional[str] = None
    filtered: bool = False
    model: Optional[str] = None


async def _call_gigachat(request: ChatRequest, client, messages: list) -> ChatResponse:
    chat_payload = Chat(messages=messages, model=settings.GIGACHAT_MODEL)
    response = await client.achat(chat_payload)

    if response.choices[0].finish_reason == "blacklist":
        return ChatResponse(response="", session_id=request.session_id, filtered=True)

    content = response.choices[0].message.content
    return ChatResponse(response=content, session_id=request.session_id, model="gigachat")


async def _call_grok(request: ChatRequest, xai_client, system_prompt: Optional[str], fallback_mode: bool = False, user_content: Optional[str] = None) -> ChatResponse:
    grok_sys_prompt = system_prompt
    meta_sys_prompt = None

    if request.profile:
        profile_grok_sys = get_grok_system_prompt(request.profile)
        if profile_grok_sys:
            grok_sys_prompt = profile_grok_sys
        if fallback_mode:
            meta_sys_prompt = get_meta_system_prompt(request.profile)

    grok_messages = []
    if grok_sys_prompt:
        grok_messages.append({"role": "system", "content": grok_sys_prompt})

    if request.profile:
        few_shot_pairs = get_grok_few_shot(request.profile)
        for pair in few_shot_pairs:
            if "user" in pair and "assistant" in pair:
                grok_messages.append({"role": "user", "content": pair["user"]})
                grok_messages.append({"role": "assistant", "content": pair["assistant"]})

    if request.context and "history" in request.context and isinstance(request.context["history"], list):
        grok_messages.extend(request.context["history"])

    grok_messages.append({"role": "user", "content": user_content if user_content is not None else request.message})

    try:
        grok_response = await xai_client.chat.completions.create(
            model=settings.GROK_MODEL,
            messages=grok_messages,
            max_tokens=settings.GROK_MAX_TOKENS
        )
        grok_content = grok_response.choices[0].message.content

        if not grok_content:
            raise ValueError("Empty response")

        log_ident = f"profile={request.profile} user={request.user_id}"
        logging.info(f"grok call: {log_ident} mode=direct")
        return ChatResponse(response=grok_content, session_id=request.session_id, model="grok")
    except Exception as e:
        if fallback_mode and meta_sys_prompt:
            try:
                meta_messages = [
                    {"role": "system", "content": meta_sys_prompt},
                    {"role": "user", "content": request.message}
                ]
                meta_response = await xai_client.chat.completions.create(
                    model=settings.GROK_MODEL,
                    messages=meta_messages,
                    max_tokens=settings.GROK_MAX_TOKENS
                )
                meta_content = meta_response.choices[0].message.content

                if not meta_content:
                    raise ValueError("Empty response")

                log_ident = f"profile={request.profile} user={request.user_id}"
                logging.info(f"grok fallback: {log_ident} mode=meta")
                return ChatResponse(response=meta_content, session_id=request.session_id, model="grok")
            except Exception:
                pass

        if fallback_mode:
            raise e
        else:
            raise e


async def _handle_gigachat_fallback(request: ChatRequest, client, xai_client) -> ChatResponse:
    if request.profile:
        fallback_prompt = get_fallback_prompt(request.profile)
        if fallback_prompt:
            wrapped_messages = [
                {"role": "system", "content": fallback_prompt},
                {"role": "user", "content": f"Пользователь чата спрашивает: «{request.message}»\nСгенерируй ответ персонажа. Только JSON."},
            ]
            fallback_payload = Chat(messages=wrapped_messages, model=settings.GIGACHAT_MODEL)
            fallback_response = await client.achat(fallback_payload)

            if fallback_response.choices[0].finish_reason != "blacklist":
                content = fallback_response.choices[0].message.content
                reply = parse_json_reply(content)
                return ChatResponse(response=reply, session_id=request.session_id, model="gigachat")

    if xai_client is not None:
        try:
            return await _call_grok(request, xai_client, get_system_prompt(request.profile) if request.profile else None, fallback_mode=True, user_content=request.message)
        except Exception:
            pass

    return ChatResponse(response="", session_id=request.session_id, filtered=True)


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request_obj: Request,
    request: ChatRequest,
    authorization: Optional[str] = Header(None),
):
    if settings.API_TOKEN:
        if not authorization or authorization != f"Bearer {settings.API_TOKEN}":
            raise HTTPException(status_code=401, detail="Unauthorized")

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

    user_content = request.message
    if request.profile:
        user_tpl = get_user_template(request.profile)
        if user_tpl and "{message}" in user_tpl:
            user_content = user_tpl.replace("{message}", request.message)

    messages.append({"role": "user", "content": user_content})

    provider = get_provider(request.profile) if request.profile else "gigachat"

    try:
        if provider == "grok":
            xai_client = getattr(request_obj.app.state, "xai_client", None)
            if xai_client is None:
                raise HTTPException(status_code=500, detail="Grok client is not initialized")
            return await _call_grok(request, xai_client, system_prompt, fallback_mode=False, user_content=user_content)
        else:
            client = getattr(request_obj.app.state, "gigachat_client", None)
            if client is None:
                raise HTTPException(status_code=500, detail="GigaChat client is not initialized")

            result = await _call_gigachat(request, client, messages)
            if result.filtered:
                xai_client = getattr(request_obj.app.state, "xai_client", None)
                return await _handle_gigachat_fallback(request, client, xai_client)
            return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
