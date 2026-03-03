from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    API_TOKEN: str = ""        # Bearer-токен для авторизации
    GIGACHAT_CREDENTIALS: str = "" # Креды для GigaChat
    GIGACHAT_ID: str = ""
    HOST: str = "127.0.0.1"
    PORT: int = 9000
    LOG_LEVEL: str = "info"

    class Config:
        env_file = ".env"
        env_prefix = "AI_API_"

settings = Settings()
