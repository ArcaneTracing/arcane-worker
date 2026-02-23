"""
API route handlers.
"""
# Import routes to make them available
from app.api.routes import chat as chat_router
from app.api.routes import health as health_router

__all__ = ["chat_router", "health_router"]
