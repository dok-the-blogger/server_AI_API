from typing import Optional
import json

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from starlette.responses import Response

from config import settings
from embeddings import EmbeddingsError, EmbeddingsRequest, EmbeddingsResponse


class EmbeddingsRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()

        async def handle(request):
            try:
                return await handler(request)
            except RequestValidationError as error:
                # Do not echo the text. ASCII JSON also represents invalid Unicode
                # in field locations without breaking the error response itself.
                detail = [
                    {key: item[key] for key in ("loc", "msg", "type")}
                    for item in error.errors()
                ]
                return Response(json.dumps({"detail": detail}), status_code=422, media_type="application/json")

        return handle


router = APIRouter(route_class=EmbeddingsRoute)


@router.post("/embeddings", response_model=EmbeddingsResponse)
async def embeddings(
    request_obj: Request,
    request: EmbeddingsRequest,
    authorization: Optional[str] = Header(None),
):
    if settings.API_TOKEN:
        if authorization != f"Bearer {settings.API_TOKEN}":
            raise HTTPException(status_code=401, detail="Unauthorized")

    client = getattr(request_obj.app.state, "embeddings_client", None)
    if client is None:
        raise HTTPException(
            status_code=503,
            detail={"code": "embeddings_not_configured", "message": "Embeddings provider is not configured"},
        )

    try:
        return await client.embed(request.prepared_inputs(), request.input_type)
    except EmbeddingsError as error:
        raise HTTPException(
            status_code=error.status_code,
            detail={"code": error.code, "message": error.message},
        ) from None
