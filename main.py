import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager
from gigachat import GigaChat

from config import settings
from routers import chat

@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.GIGACHAT_CREDENTIALS:
        async with GigaChat(credentials=settings.GIGACHAT_CREDENTIALS, verify_ssl_certs=False) as client:
            app.state.gigachat_client = client
            yield
    else:
        app.state.gigachat_client = None
        yield

app = FastAPI(title="AI API", docs_url=None, redoc_url=None, lifespan=lifespan)
app.include_router(chat.router)

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        log_level=settings.LOG_LEVEL,
    )
