"""Stateless free-text generation using the existing completion transports."""
import hashlib
import json
import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from completion_providers import MODEL_PROFILES, CompletionError, ProviderName
from summaries import SummaryUsage

GenerationModel = Literal["glm-5.3-flash", "deepseek-v4.1-flash", "mimo-v2.6-flash", "mimo-v2.6-pro"]
MAX_GENERATION_BYTES = 256 * 1024


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)
    input: str = Field(min_length=1, max_length=131072)
    instruction: str = Field(min_length=1, max_length=16000)
    model: GenerationModel = "mimo-v2.6-flash"
    max_output_tokens: int = Field(default=2048, ge=64, le=8192)

    @field_validator("input", "instruction")
    @classmethod
    def valid_text(cls, value):
        if not value.strip():
            raise ValueError("Text must not be blank")
        value.encode("utf-8")
        return value

    @model_validator(mode="after")
    def bounded_payload(self):
        if len(json.dumps(self.model_dump(), ensure_ascii=False).encode()) > MAX_GENERATION_BYTES:
            raise ValueError("Generation request is too large")
        return self


class GenerateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    text: str = Field(min_length=1, max_length=131072)
    provider: ProviderName
    model: GenerationModel
    finish_reason: Literal["stop", "length"]
    truncated: bool
    prompt_version: Literal["generate-v1"] = "generate-v1"
    prompt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    elapsed_ms: int = Field(ge=0)
    usage: SummaryUsage

    @field_validator("text")
    @classmethod
    def valid_text(cls, value):
        if not value.strip():
            raise ValueError("Empty generation")
        value.encode("utf-8")
        return value


async def generate(client, request: GenerateRequest) -> GenerateResponse:
    started = time.monotonic()
    profile = MODEL_PROFILES[request.model]
    provider = client.providers.get(profile.provider)
    if provider is None:
        raise CompletionError(503, "generation_not_configured", "Generation provider is not configured")
    payload = {**profile.parameters(request.max_output_tokens), "messages": [
        {"role": "system", "content": request.instruction},
        {"role": "user", "content": request.input},
    ]}
    body = await provider.complete(payload, timeout=client.timeout, max_bytes=MAX_GENERATION_BYTES)
    try:
        result = json.loads(body)
        if (not isinstance(result, dict) or result["model"] != request.model
                or not isinstance(result["choices"], list) or len(result["choices"]) != 1):
            raise ValueError
        choice = result["choices"][0]
        if not isinstance(choice, dict) or choice.get("finish_reason") not in {"stop", "length"}:
            raise ValueError
        message = choice["message"]
        if (not isinstance(message, dict) or message.get("role") != "assistant"
                or message.get("tool_calls") or message.get("function_call") or message.get("refusal")):
            raise ValueError
        usage = SummaryUsage.model_validate(result["usage"])
        if (usage.total_tokens != usage.prompt_tokens + usage.completion_tokens
                or usage.completion_tokens > request.max_output_tokens):
            raise ValueError
        return GenerateResponse(
            text=message["content"], model=request.model, provider=profile.provider,
            finish_reason=choice["finish_reason"], truncated=choice["finish_reason"] == "length",
            prompt_hash=hashlib.sha256(request.instruction.encode()).hexdigest(), usage=usage,
            elapsed_ms=round((time.monotonic() - started) * 1000))
    except (KeyError, TypeError, ValueError, RecursionError):
        raise CompletionError(502, "invalid_provider_response", "Generation provider returned an invalid result") from None
