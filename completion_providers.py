"""Server-owned model profiles and bounded Chat Completions transport."""
import asyncio
from dataclasses import dataclass
from typing import Literal

import httpx

ProviderName = Literal["digitalocean", "mimo"]
SummaryModel = Literal["glm-5.3-flash", "deepseek-v4.1-flash", "mimo-v2.6-flash"]
MimoChatModel = Literal["mimo-v2.6-flash", "mimo-v2.6-pro"]
MAX_RESPONSE_BYTES = 64 * 1024


@dataclass(frozen=True)
class ModelProfile:
    model: str
    provider: ProviderName
    summary: bool = False
    max_completion_tokens: int = 1024
    reasoning_effort: str | None = None
    thinking: str | None = None

    def parameters(self, max_tokens: int | None = None) -> dict:
        result = {"model": self.model, "stream": False,
                  "max_completion_tokens": max_tokens or self.max_completion_tokens}
        if self.reasoning_effort is not None:
            result["reasoning_effort"] = self.reasoning_effort
        if self.thinking is not None:
            result["thinking"] = {"type": self.thinking}
        return result


MODEL_PROFILES = {
    "glm-5.3-flash": ModelProfile("glm-5.3-flash", "digitalocean", summary=True,
                                reasoning_effort="none"),
    "deepseek-v4.1-flash": ModelProfile("deepseek-v4.1-flash", "digitalocean", summary=True,
                                     max_completion_tokens=2048, reasoning_effort="low"),
    "mimo-v2.6-flash": ModelProfile("mimo-v2.6-flash", "mimo", summary=True,
                                 thinking="disabled"),
    "mimo-v2.6-pro": ModelProfile("mimo-v2.6-pro", "mimo", thinking="disabled"),
}


class CompletionError(Exception):
    def __init__(self, status_code, code, message):
        super().__init__(message)
        self.status_code, self.code, self.message = status_code, code, message


class ChatCompletionsProvider:
    """One credential and endpoint per provider; no redirects, proxies or retries."""

    def __init__(self, *, provider: ProviderName, api_key: str, base_url: str):
        self.provider = provider
        self.http = httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            headers={"Authorization": f"Bearer {api_key}"},
            follow_redirects=False, trust_env=False,
        )

    async def aclose(self):
        await self.http.aclose()

    async def complete(self, payload: dict, *, timeout: float) -> bytes:
        """Return bounded bytes; each operation validates its own output contract."""
        try:
            async with asyncio.timeout(timeout):
                async with self.http.stream(
                        "POST", "chat/completions", json=payload, timeout=timeout) as response:
                    self._check_status(response.status_code)
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        body.extend(chunk)
                        if len(body) > MAX_RESPONSE_BYTES:
                            raise CompletionError(502, "invalid_provider_response",
                                                  "Completion response is too large")
                    return bytes(body)
        except (TimeoutError, httpx.TimeoutException):
            raise CompletionError(504, "provider_timeout", "Completion provider timed out") from None
        except httpx.HTTPError:
            raise CompletionError(502, "provider_unavailable",
                                  "Completion provider could not be reached") from None

    def _check_status(self, status: int):
        if status == 200:
            return
        payment = ("DigitalOcean requires payment (HTTP 402). Check the Serverless Inference prepaid balance."
                   if self.provider == "digitalocean" else
                   "Xiaomi MiMo requires payment (HTTP 402). Check the MiMo account balance.")
        errors = {
            402: (502, "provider_payment_required", payment),
            429: (429, "provider_rate_limited", "Completion provider rate limit exceeded"),
            401: (502, "provider_authentication_failed", "Completion provider credentials were rejected"),
            403: (502, "provider_authentication_failed", "Completion provider credentials were rejected"),
            400: (422, "provider_input_rejected", "Completion provider rejected the input or model parameters"),
            413: (422, "provider_input_rejected", "Completion provider rejected the input size"),
            422: (422, "provider_input_rejected", "Completion provider rejected the input or model parameters"),
        }
        if self.provider == "mimo":
            errors.update({
                403: (502, "provider_access_denied", "Xiaomi MiMo denied access: region or API key restriction"),
                421: (422, "provider_content_filtered", "Xiaomi MiMo filtered the content"),
            })
        raise CompletionError(*errors.get(status, (
            502, "provider_error", "Completion provider returned an unsuccessful response")))
