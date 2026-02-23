"""
Base class for LLM model services.

Defines the interface that all provider-specific services must implement.
Uses the Template Method pattern to define the structure of service execution.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any
from app.models.schemas import LLMServiceRequestDto


class BaseModelService(ABC):
    """Base class for model services"""
    
    @abstractmethod
    async def execute(
        self,
        request: LLMServiceRequestDto,
    ) -> Dict[str, Any]:
        """
        Execute LLM request with full request DTO.
        
        Handles:
        - Template rendering
        - Message conversion (provider-specific)
        - Parameter extraction (provider-specific)
        - LLM execution
        
        Returns a dictionary matching LLMServiceResponseDto structure.
        """
        pass
    
    @abstractmethod
    def get_adapter_type(self) -> str:
        """Return the adapter type this service handles"""
        pass

