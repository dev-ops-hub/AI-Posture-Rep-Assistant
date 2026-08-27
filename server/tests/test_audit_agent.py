import os
from unittest.mock import MagicMock, patch

import pytest

from agents.audit_agent import FormAuditAgent
from agents.models import PostureViolationEvent, PostureViolationMetrics


@pytest.fixture
def mock_violation_event():
    metrics = PostureViolationMetrics(
        spine_angle_deg=25.5,
        knee_angle_deg=120.0,
        violation_duration_sec=2.5,
    )
    return PostureViolationEvent(
        sender="Agent_1_Vision",
        recipient="Agent_2_FormAudit",
        event_type="POSTURE_VIOLATION",
        exercise="Squat",
        metrics=metrics,
        image_base64="data:image/jpeg;base64,test",
    )


def test_form_audit_agent_initialization_no_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    agent = FormAuditAgent()
    assert agent.enabled is False
    assert agent.client is None
    assert agent.model == "gpt-4o"


def test_form_audit_agent_initialization_with_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
    with patch("agents.audit_agent.OpenAI"):
        agent = FormAuditAgent()
        assert agent.enabled is True
        assert agent.model == "gpt-4o"


def test_form_audit_agent_initialization_disabled(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
    with patch("agents.audit_agent.OpenAI"):
        agent = FormAuditAgent(enabled=False)
        assert agent.enabled is False


def test_form_audit_agent_fallback_feedback_forward_lean(mock_violation_event):
    mock_violation_event.metrics.spine_angle_deg = 25.0
    result = FormAuditAgent._fallback_feedback(mock_violation_event)
    assert "torso is folding forward" in result
    assert "Brace your core" in result


def test_form_audit_agent_fallback_feedback_slight_drift(mock_violation_event):
    mock_violation_event.metrics.spine_angle_deg = 18.0
    result = FormAuditAgent._fallback_feedback(mock_violation_event)
    assert "drifting out of position" in result
    assert "spine neutral" in result


def test_form_audit_agent_audit_posture_fallback_no_api_key(
    mock_violation_event, monkeypatch
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    agent = FormAuditAgent()
    result = agent.audit_posture(mock_violation_event)
    assert isinstance(result, str)
    assert len(result) > 0


def test_form_audit_agent_audit_posture_fallback_no_image():
    metrics = PostureViolationMetrics(
        spine_angle_deg=25.5,
        knee_angle_deg=120.0,
        violation_duration_sec=2.5,
    )
    event = PostureViolationEvent(
        sender="Agent_1_Vision",
        recipient="Agent_2_FormAudit",
        event_type="POSTURE_VIOLATION",
        exercise="Squat",
        metrics=metrics,
        image_base64="",
    )

    monkeypatch_obj = pytest.MonkeyPatch()
    monkeypatch_obj.setenv("OPENAI_API_KEY", "test-key")
    with patch("agents.audit_agent.OpenAI"):
        agent = FormAuditAgent()
        agent.client = MagicMock()
        result = agent.audit_posture(event)
        assert isinstance(result, str)
        assert len(result) > 0
    monkeypatch_obj.undo()


def test_form_audit_agent_audit_posture_with_openai_success(
    mock_violation_event, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")

    mock_response = MagicMock()
    mock_response.output_text = "Keep your chest up and core tight."

    mock_client = MagicMock()
    mock_client.responses.create.return_value = mock_response

    with patch("agents.audit_agent.OpenAI", return_value=mock_client):
        agent = FormAuditAgent()
        result = agent.audit_posture(mock_violation_event)

        assert result == "Keep your chest up and core tight."
        mock_client.responses.create.assert_called_once()


def test_form_audit_agent_audit_posture_with_openai_empty_response(
    mock_violation_event, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")

    mock_response = MagicMock()
    mock_response.output_text = None

    mock_client = MagicMock()
    mock_client.responses.create.return_value = mock_response

    with patch("agents.audit_agent.OpenAI", return_value=mock_client):
        agent = FormAuditAgent()
        result = agent.audit_posture(mock_violation_event)

        assert isinstance(result, str)
        assert len(result) > 0
