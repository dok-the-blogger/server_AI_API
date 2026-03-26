with open("routers/chat.py", "r") as f:
    content = f.read()

# 1. Add model field to ChatResponse
content = content.replace(
    "    filtered: bool = False",
    "    filtered: bool = False\n    model: Optional[str] = None"
)

# 2. Add xai fallback logic and model parameter
new_logic = """
        if response.choices[0].finish_reason == "blacklist":
            if request.profile == "dokbot":
                fallback_prompt = get_fallback_prompt(request.profile)
                if fallback_prompt:
                    wrapped_messages = [
                        {"role": "system", "content": fallback_prompt},
                        {"role": "user", "content": f"Пользователь чата спрашивает: «{request.message}»\\nСгенерируй ответ персонажа. Только JSON."},
                    ]
                    fallback_payload = Chat(messages=wrapped_messages, model=settings.GIGACHAT_MODEL)
                    fallback_response = await client.achat(fallback_payload)

                    if fallback_response.choices[0].finish_reason != "blacklist":
                        content = fallback_response.choices[0].message.content
                        reply = parse_json_reply(content)
                        return ChatResponse(response=reply, session_id=request.session_id, model="gigachat")

            xai_client = getattr(request_obj.app.state, "xai_client", None)
            if xai_client is not None:
                try:
                    grok_response = await xai_client.chat.completions.create(
                        model=settings.GROK_MODEL,
                        messages=messages
                    )
                    grok_content = grok_response.choices[0].message.content
                    return ChatResponse(response=grok_content, session_id=request.session_id, model="grok")
                except Exception:
                    # Ignore grok error and fallback to filtered=True
                    pass

            return ChatResponse(response="", session_id=request.session_id, filtered=True)

        content = response.choices[0].message.content
        return ChatResponse(response=content, session_id=request.session_id, model="gigachat")"""

old_logic = """
        if response.choices[0].finish_reason == "blacklist":
            if request.profile == "dokbot":
                fallback_prompt = get_fallback_prompt(request.profile)
                if fallback_prompt:
                    wrapped_messages = [
                        {"role": "system", "content": fallback_prompt},
                        {"role": "user", "content": f"Пользователь чата спрашивает: «{request.message}»\\nСгенерируй ответ персонажа. Только JSON."},
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
        return ChatResponse(response=content, session_id=request.session_id)"""

content = content.replace(old_logic.strip(), new_logic.strip())

with open("routers/chat.py", "w") as f:
    f.write(content)
