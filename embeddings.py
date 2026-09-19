"""Qwen text preparation and the DigitalOcean embeddings HTTP boundary."""

import asyncio
from typing import Annotated, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, field_validator


MAX_INPUT_CHARS = 32768
MAX_BATCH_SIZE = 64
MAX_BATCH_CHARS = 131072
PREPROCESSING_VERSION = "qwen3-retrieval-v1"
QUERY_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
InputType = Literal["document", "query"]
TextInput = Annotated[str, Field(min_length=1, max_length=MAX_INPUT_CHARS)]


class EmbeddingsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    input: TextInput | Annotated[list[TextInput], Field(min_length=1, max_length=MAX_BATCH_SIZE)]
    input_type: InputType = "document"

    @field_validator("input")
    @classmethod
    def validate_text(cls, value):
        texts = [value] if isinstance(value, str) else value
        if any(not text.strip() for text in texts):
            raise ValueError("Each input must contain non-whitespace text")
        # JSON can contain lone UTF-16 surrogates, which cannot be sent as UTF-8.
        try:
            for text in texts:
                text.encode("utf-8")
        except UnicodeEncodeError:
            raise ValueError("Each input must be valid Unicode text") from None
        return value

    def prepared_inputs(self) -> list[str]:
        texts = [self.input] if isinstance(self.input, str) else self.input
        if self.input_type == "query":
            texts = [f"Instruct: {QUERY_INSTRUCTION}\nQuery:{text}" for text in texts]
        if any(len(text) > MAX_INPUT_CHARS for text in texts) or sum(map(len, texts)) > MAX_BATCH_CHARS:
            raise EmbeddingsError(422, "input_too_large", "Prepared input exceeds the API character limit")
        return texts


class EmbeddingItem(BaseModel):
    model_config = ConfigDict(strict=True)

    object: Literal["embedding"]
    index: int = Field(ge=0)
    embedding: list[FiniteFloat] = Field(min_length=1)


class EmbeddingUsage(BaseModel):
    model_config = ConfigDict(strict=True)

    prompt_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class ProviderResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    object: Literal["list"]
    model: str
    data: list[EmbeddingItem] = Field(min_length=1)
    usage: EmbeddingUsage


class EmbeddingsResponse(ProviderResponse):
    provider: Literal["digitalocean"] = "digitalocean"
    dimensions: int
    input_type: InputType
    preprocessing_version: Literal["qwen3-retrieval-v1"] = PREPROCESSING_VERSION


class EmbeddingsError(Exception):
    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


class DigitalOceanEmbeddings:
    """One HTTP request per API call; no retries, fallback model or text storage."""

    def __init__(self, *, api_key: str, base_url: str, model: str, dimensions: int, timeout: float):
        self.model = model
        self.dimensions = dimensions
        self.timeout = timeout
        self.http = httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
            follow_redirects=False,
        )

    async def aclose(self):
        await self.http.aclose()

    async def embed(self, texts: list[str], input_type: InputType) -> EmbeddingsResponse:
        try:
            # Bound the whole HTTP exchange, including slow responses.
            async with asyncio.timeout(self.timeout):
                response = await self.http.post(
                    "embeddings",
                    json={"model": self.model, "input": texts, "encoding_format": "float"},
                )
        except (TimeoutError, httpx.TimeoutException):
            raise EmbeddingsError(504, "provider_timeout", "Embeddings provider timed out") from None
        except httpx.HTTPError:
            raise EmbeddingsError(502, "provider_unavailable", "Embeddings provider could not be reached") from None

        if response.status_code == 402:
            raise EmbeddingsError(
                502, "provider_payment_required",
                "DigitalOcean requires payment (HTTP 402). Check and top up the Serverless Inference prepayment balance.",
            )
        if response.status_code == 429:
            raise EmbeddingsError(429, "provider_rate_limited", "Embeddings provider rate limit exceeded")
        if response.status_code in (400, 413, 422):
            raise EmbeddingsError(422, "provider_input_rejected", "Embeddings provider rejected the input; check its token limit")
        if response.status_code in (401, 403):
            raise EmbeddingsError(502, "provider_authentication_failed", "Embeddings provider credentials were rejected")
        if response.status_code != 200:
            raise EmbeddingsError(502, "provider_error", "Embeddings provider returned an unsuccessful response")

        try:
            result = ProviderResponse.model_validate(response.json())
            if result.model != self.model or len(result.data) != len(texts):
                raise ValueError("Unexpected model or row count")
            if sorted(item.index for item in result.data) != list(range(len(texts))):
                raise ValueError("Unexpected embedding indices")
            for item in result.data:
                if len(item.embedding) != self.dimensions or not any(item.embedding):
                    raise ValueError("Unexpected vector dimensions or zero vector")
            if result.usage.total_tokens < result.usage.prompt_tokens:
                raise ValueError("Inconsistent usage")
        except ValueError:
            raise EmbeddingsError(502, "invalid_provider_response", "Embeddings provider returned an invalid result") from None

        return EmbeddingsResponse(
            **result.model_dump(exclude={"data"}),
            data=sorted(result.data, key=lambda item: item.index),
            dimensions=self.dimensions,
            input_type=input_type,
        )
