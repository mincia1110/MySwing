"""Main API router aggregating all endpoint routers."""

from fastapi import APIRouter

from app.api.analyses import router as analyses_router
from app.api.history import router as history_router
from app.api.profile import router as profile_router
from app.api.upload import router as upload_router
from app.api.videos import router as videos_router

api_router = APIRouter()

api_router.include_router(upload_router)
api_router.include_router(profile_router)
api_router.include_router(videos_router)
api_router.include_router(analyses_router)
api_router.include_router(history_router)


@api_router.get("/status")
async def api_status() -> dict[str, str]:
    """API status endpoint."""
    return {"status": "operational"}
