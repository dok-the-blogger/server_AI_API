import uvicorn
from fastapi import FastAPI
from config import settings
from routers import chat

app = FastAPI(title="AI API", docs_url=None, redoc_url=None)
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
