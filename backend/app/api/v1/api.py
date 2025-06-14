from fastapi import APIRouter
from app.api.v1.endpoints import makeup

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(makeup.router, prefix="/makeup", tags=["makeup"]) 