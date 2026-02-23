from pydantic import BaseModel
from typing import Any

class RagasScore(BaseModel):
    score: Any
    metric: str
    id: str