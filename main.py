from routers.generation import router as generation_router
import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager
from gigachat import GigaChat
from openai import AsyncOpenAI


from config import settings
from routers import chat_router, models_router, embeddings_router, summaries_router
from profiles import load_profiles
from embeddings import DigitalOceanEmbeddings
from summaries import Summaries
from completion_providers import ChatCompletionsProvider
from classification import JevClassifier
from routers.classification import router as classification_router
from routers.research import router as research_router
from experiments.doknews_research.server_process import start_research, stop_research
from experiments.doknews_research.server_config import ResearchError

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_profiles()
    app.state.research_controller = None
    app.state.research_worker = None
    app.state.research_service_token = settings.RESEARCH_API_TOKEN or settings.API_TOKEN

    if settings.GROK_API_KEY:
        app.state.xai_client = AsyncOpenAI(
            api_key=settings.GROK_API_KEY,
            base_url="https://api.x.ai/v1",
            default_headers={"x-grok-conv-id": "51bae473-e626-4ac9-8765-82fcb3e3905c"},
        )
    else:
        app.state.xai_client = None

    app.state.embeddings_client = (
        DigitalOceanEmbeddings(
            api_key=settings.DIGITALOCEAN_API_KEY,
            base_url=settings.EMBEDDINGS_BASE_URL,
            model=settings.EMBEDDINGS_MODEL,
            dimensions=settings.EMBEDDINGS_DIMENSIONS,
            timeout=settings.EMBEDDINGS_TIMEOUT_SECONDS,
        )
        if settings.DIGITALOCEAN_API_KEY else None
    )

    try:
        try:
            app.state.research_controller, app.state.research_worker = start_research(
                settings.RESEARCH_CONFIG_PATH,
                service_token_configured=bool(app.state.research_service_token))
        except ResearchError:
            # Misconfigured research does not disable existing AI routes.
            pass
        app.state.jev_client = (
            JevClassifier(api_key=settings.DIGITALOCEAN_API_KEY,
                          base_url=settings.JEV_BASE_URL, model=settings.JEV_MODEL,
                          timeout=settings.JEV_TIMEOUT_SECONDS)
            if settings.DIGITALOCEAN_API_KEY else None
        )
        providers = {}
        if settings.DIGITALOCEAN_API_KEY:
            providers["digitalocean"] = ChatCompletionsProvider(
                provider="digitalocean", api_key=settings.DIGITALOCEAN_API_KEY,
                base_url=settings.SUMMARIES_BASE_URL)
        if settings.MIMO_API_KEY:
            providers["mimo"] = ChatCompletionsProvider(
                provider="mimo", api_key=settings.MIMO_API_KEY,
                base_url=settings.MIMO_BASE_URL)
        app.state.mimo_client = providers.get("mimo")
        app.state.summaries_client = Summaries(
            providers=providers, model=settings.SUMMARIES_MODEL,
            timeout=settings.SUMMARIES_TIMEOUT_SECONDS,
        )
        if settings.GIGACHAT_CREDENTIALS:
            async with GigaChat(credentials=settings.GIGACHAT_CREDENTIALS, verify_ssl_certs=False) as client:
                app.state.gigachat_client = client
                yield
        else:
            app.state.gigachat_client = None
            yield
    finally:
        await stop_research(app.state.research_worker)
        if getattr(app.state, "jev_client", None) is not None:
            await app.state.jev_client.aclose()
        if app.state.embeddings_client is not None:
            await app.state.embeddings_client.aclose()
        if getattr(app.state, "summaries_client", None) is not None:
            await app.state.summaries_client.aclose()

app = FastAPI(title="AI API", docs_url=None, redoc_url=None, lifespan=lifespan)
app.include_router(chat_router)
app.include_router(models_router)
app.include_router(embeddings_router)
app.include_router(summaries_router)
app.include_router(generation_router)
app.include_router(classification_router)
app.include_router(research_router)

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

