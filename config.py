from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    API_TOKEN: str = ""        # Bearer-токен для авторизации
    GIGACHAT_CREDENTIALS: str = "" # Креды для GigaChat
    GIGACHAT_ID: str = ""
    GIGACHAT_MODEL: str = "GigaChat-2"
    GROK_API_KEY: str = ""
    GROK_MODEL: str = "grok-4-1-fast-non-reasoning"
    GROK_MAX_TOKENS: int = 256
    DIGITALOCEAN_API_KEY: str = ""
    EMBEDDINGS_BASE_URL: str = "https://inference.do-ai.run/v1"
    EMBEDDINGS_MODEL: str = "qwen3-embedding-0.6b"
    EMBEDDINGS_DIMENSIONS: int = Field(default=1024, gt=0)
    EMBEDDINGS_TIMEOUT_SECONDS: float = Field(default=30.0, gt=0, le=300)
    HOST: str = "127.0.0.1"
    PORT: int = 9000
    LOG_LEVEL: str = "info"

    class Config:
        env_file = ".env"
        env_prefix = "AI_API_"
        extra = "ignore"

settings = Settings()
