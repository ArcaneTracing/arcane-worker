"""
Health check and root endpoints.
"""
from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def root():
    return {
        "message": "Arcane Eval Chat API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@router.get("/health")
async def health():
    return {"status": "healthy"}

