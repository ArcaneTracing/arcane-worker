"""
Base classes for RAGAS evaluation metrics.
Implements Strategy pattern for evaluation metrics.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Type
from pydantic import BaseModel
from ragas.llms import BaseRagasLLM
from ragas.embeddings.base import BaseRagasEmbedding

from app.services.evaluation.ragas.models.ragas_score import RagasScore


class BaseEvaluationMetric[T: BaseModel](ABC):
    """
    Abstract base class for evaluation metrics.
    
    Each metric implements this interface, allowing them to be used
    interchangeably by the RagasProcessor.
    """
    
    def __init__(
        self,
        metric_id: str,
        metric_name: str,
        input_class: Type[T],
        requires_llm: bool = False,
        requires_embeddings: bool = False
    ):
        """
        Initialize the metric.
        
        Args:
            metric_id: Unique identifier (UUID) for this metric
            metric_name: Human-readable name of the metric
            input_class: Pydantic model class for input validation
            requires_llm: Whether this metric requires an LLM instance
            requires_embeddings: Whether this metric requires an embeddings instance
        """
        self.metric_id = metric_id
        self.metric_name = metric_name
        self.input_class = input_class
        self.requires_llm = requires_llm
        self.requires_embeddings = requires_embeddings
    
    def validate_input(self, score_mapping: Dict[str, Any]) -> T:
        """
        Validate and parse input data.
        
        Args:
            score_mapping: Dictionary containing input data
            
        Returns:
            Validated input model instance
        """
        return self.input_class(**score_mapping)
    
    @abstractmethod
    async def evaluate(
        self,
        input_data: T,
        llm: Optional[BaseRagasLLM] = None,
        embeddings: Optional[BaseRagasEmbedding] = None
    ) -> RagasScore:
        """
        Evaluate the metric.
        
        Args:
            input_data: Validated input data
            llm: Optional LLM instance (required if requires_llm=True)
            embeddings: Optional embeddings instance (required if requires_embeddings=True)
            
        Returns:
            RagasScore containing evaluation results
        """
        pass


class FunctionBasedMetric[T: BaseModel](BaseEvaluationMetric[T]):
    """
    Wrapper for existing evaluation functions.
    
    This allows us to use existing functions with the Strategy pattern
    without requiring a full rewrite of all metric files.
    """
    
    def __init__(
        self,
        metric_id: str,
        metric_name: str,
        input_class: Type[T],
        evaluation_function,
        requires_llm: bool = False,
        requires_embeddings: bool = False
    ):
        """
        Initialize the metric wrapper.
        
        Args:
            metric_id: Unique identifier (UUID) for this metric
            metric_name: Human-readable name of the metric
            input_class: Pydantic model class for input validation
            evaluation_function: The async evaluation function to call
            requires_llm: Whether this metric requires an LLM instance
            requires_embeddings: Whether this metric requires an embeddings instance
        """
        super().__init__(metric_id, metric_name, input_class, requires_llm, requires_embeddings)
        self.evaluation_function = evaluation_function
    
    async def evaluate(
        self,
        input_data: T,
        llm: Optional[BaseRagasLLM] = None,
        embeddings: Optional[BaseRagasEmbedding] = None
    ) -> RagasScore:
        """
        Evaluate the metric by calling the wrapped function.
        
        Args:
            input_data: Validated input data
            llm: Optional LLM instance
            embeddings: Optional embeddings instance
            
        Returns:
            RagasScore containing evaluation results
        """
        # Call the function with appropriate arguments based on requirements
        if self.requires_embeddings:
            return await self.evaluation_function(input_data, llm=llm, embeddings=embeddings)
        elif self.requires_llm:
            return await self.evaluation_function(input_data, llm=llm)
        else:
            return await self.evaluation_function(input_data)

