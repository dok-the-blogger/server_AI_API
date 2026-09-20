from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Request

from config import settings
from routers.embeddings import EmbeddingsRoute
from summaries import SummaryError, SummaryRequest, SummaryResponse

router = APIRouter(route_class=EmbeddingsRoute)


@router.post("/summaries", response_model=SummaryResponse)
async def summarize_article(request_obj: Request, request: SummaryRequest,
                            authorization: Optional[str] = Header(None)):
    if settings.API_TOKEN and authorization != f"Bearer {settings.API_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    client = getattr(request_obj.app.state, "summaries_client", None)
    if client is None:
        raise HTTPException(status_code=503, detail={
            "code": "summaries_not_configured", "message": "Summary provider is not configured"})
    try:
        return await client.summarize(request)
    except SummaryError as error:
        raise HTTPException(status_code=error.status_code,
                            detail={"code": error.code, "message": error.message}) from None
