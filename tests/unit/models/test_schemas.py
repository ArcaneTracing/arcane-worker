"""
Unit tests for message_id field in job and result schemas.
"""
import pytest
from app.models.schemas import (
    ExperimentJobDto,
    ExperimentResultDto,
    EvaluationJobDto,
    EvaluationResultDto,
)


class TestExperimentJobDtoMessageId:
    """Tests for message_id in ExperimentJobDto."""

    def test_accepts_message_id_snake_case(self):
        """Should accept message_id (snake_case)."""
        dto = ExperimentJobDto(
            experiment_id="exp-1",
            dataset_row_id="row-1",
            prompt_id="prompt-1",
            inputs={"q": "test"},
            message_id="exp-1-row-1-1700000000000",
        )
        assert dto.message_id == "exp-1-row-1-1700000000000"

    def test_accepts_message_id_camel_case_alias(self):
        """Should accept messageId (camelCase alias from Node)."""
        dto = ExperimentJobDto(
            experimentId="exp-1",
            datasetRowId="row-1",
            promptId="prompt-1",
            inputs={"q": "test"},
            messageId="exp-1-row-1-1700000000000",
        )
        assert dto.message_id == "exp-1-row-1-1700000000000"

    def test_message_id_optional(self):
        """Should allow message_id to be omitted."""
        dto = ExperimentJobDto(
            experiment_id="exp-1",
            dataset_row_id="row-1",
            prompt_id="prompt-1",
            inputs={"q": "test"},
        )
        assert dto.message_id is None


class TestExperimentResultDtoMessageId:
    """Tests for message_id in ExperimentResultDto."""

    def test_accepts_message_id_snake_case(self):
        """Should accept message_id (snake_case)."""
        dto = ExperimentResultDto(
            experiment_id="exp-1",
            dataset_row_id="row-1",
            result="ok",
            message_id="exp-1-row-1-1700000000000",
        )
        assert dto.message_id == "exp-1-row-1-1700000000000"

    def test_accepts_message_id_camel_case_alias(self):
        """Should accept messageId (camelCase alias from Node)."""
        dto = ExperimentResultDto(
            experimentId="exp-1",
            datasetRowId="row-1",
            result="ok",
            messageId="exp-1-row-1-1700000000000",
        )
        assert dto.message_id == "exp-1-row-1-1700000000000"

    def test_message_id_optional(self):
        """Should allow message_id to be omitted."""
        dto = ExperimentResultDto(
            experiment_id="exp-1",
            dataset_row_id="row-1",
            result="ok",
        )
        assert dto.message_id is None


class TestEvaluationJobDtoMessageId:
    """Tests for message_id in EvaluationJobDto."""

    def test_accepts_message_id_snake_case(self):
        """Should accept message_id (snake_case)."""
        dto = EvaluationJobDto(
            evaluation_id="eval-1",
            score_id="score-1",
            scoring_type="RAGAS",
            dataset_row_id="row-1",
            experiment_result_id="result-1",
            ragas_score_key="test-key",
            ragas_model_configuration_id="model-1",
            score_mapping={"test": "data"},
            message_id="eval-1-score-1-1700000000000",
        )
        assert dto.message_id == "eval-1-score-1-1700000000000"

    def test_accepts_message_id_camel_case_alias(self):
        """Should accept messageId (camelCase alias from Node)."""
        dto = EvaluationJobDto(
            evaluationId="eval-1",
            scoreId="score-1",
            scoringType="RAGAS",
            datasetRowId="row-1",
            experimentResultId="result-1",
            ragasScoreKey="test-key",
            ragasModelConfigurationId="model-1",
            scoreMapping={"test": "data"},
            messageId="eval-1-score-1-1700000000000",
        )
        assert dto.message_id == "eval-1-score-1-1700000000000"

    def test_message_id_optional(self):
        """Should allow message_id to be omitted."""
        dto = EvaluationJobDto(
            evaluation_id="eval-1",
            score_id="score-1",
            scoring_type="RAGAS",
            dataset_row_id="row-1",
            experiment_result_id="result-1",
            ragas_score_key="test-key",
            ragas_model_configuration_id="model-1",
            score_mapping={"test": "data"},
        )
        assert dto.message_id is None


class TestEvaluationResultDtoMessageId:
    """Tests for message_id in EvaluationResultDto."""

    def test_accepts_message_id_snake_case(self):
        """Should accept message_id (snake_case)."""
        dto = EvaluationResultDto(
            evaluation_id="eval-1",
            score_id="score-1",
            dataset_row_id="row-1",
            score="0.85",
            message_id="eval-1-score-1-1700000000000",
        )
        assert dto.message_id == "eval-1-score-1-1700000000000"

    def test_accepts_message_id_camel_case_alias(self):
        """Should accept messageId (camelCase alias from Node)."""
        dto = EvaluationResultDto(
            evaluationId="eval-1",
            scoreId="score-1",
            datasetRowId="row-1",
            score="0.85",
            messageId="eval-1-score-1-1700000000000",
        )
        assert dto.message_id == "eval-1-score-1-1700000000000"

    def test_message_id_optional(self):
        """Should allow message_id to be omitted."""
        dto = EvaluationResultDto(
            evaluation_id="eval-1",
            score_id="score-1",
            dataset_row_id="row-1",
            score="0.85",
        )
        assert dto.message_id is None
