from unittest.mock import MagicMock, patch

import pytest

from agents.coach_agent import FitnessCoachAgent
from agents.models import SessionSummary, SessionSummaryPayload, UserProfile


@pytest.fixture
def mock_session_payload():
    user_profile = UserProfile(weight_kg=70.0, goal="general fitness")
    session_summary = SessionSummary(
        exercise="Squat",
        total_reps=10,
        duration_seconds=120.0,
        met_value=5.0,
        calculated_calories=50.0,
        total_posture_faults=2,
    )
    return SessionSummaryPayload(
        sender="Orchestrator_Main",
        recipient="Agent_3_Coach",
        user_profile=user_profile,
        session_summary=session_summary,
        form_audit_diagnostics=["Keep chest up", "Brace core"],
    )


def test_fitness_coach_agent_initialization_no_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    agent = FitnessCoachAgent()
    assert agent.enabled is False
    assert agent.client is None
    assert agent.model == "gpt-4o-mini"


def test_fitness_coach_agent_initialization_with_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
    with patch("agents.coach_agent.OpenAI"):
        agent = FitnessCoachAgent()
        assert agent.enabled is True
        assert agent.model == "gpt-4o-mini"


def test_fitness_coach_agent_initialization_disabled(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
    with patch("agents.coach_agent.OpenAI"):
        agent = FitnessCoachAgent(enabled=False)
        assert agent.enabled is False


def test_fitness_coach_agent_calculate_calories():
    calories = FitnessCoachAgent.calculate_calories(5.0, 70.0, 3600.0)
    assert calories == pytest.approx(350.0, rel=0.01)


def test_fitness_coach_agent_calculate_calories_half_hour():
    calories = FitnessCoachAgent.calculate_calories(5.0, 70.0, 1800.0)
    assert calories == pytest.approx(175.0, rel=0.01)


def test_fitness_coach_agent_calculate_calories_negative_duration():
    calories = FitnessCoachAgent.calculate_calories(5.0, 70.0, -100.0)
    assert calories == 0.0


def test_fitness_coach_agent_calculate_calories_zero_duration():
    calories = FitnessCoachAgent.calculate_calories(5.0, 70.0, 0.0)
    assert calories == 0.0


def test_fitness_coach_agent_fallback_summary_with_diagnostics(mock_session_payload):
    result = FitnessCoachAgent._fallback_summary(mock_session_payload)
    assert "10 squat reps" in result
    assert "120 seconds" in result
    assert "50.00 kcal" in result
    assert "Posture faults recorded: 2" in result
    assert "Brace core" in result


def test_fitness_coach_agent_fallback_summary_no_diagnostics():
    user_profile = UserProfile(weight_kg=70.0, goal="general fitness")
    session_summary = SessionSummary(
        exercise="Squat",
        total_reps=10,
        duration_seconds=120.0,
        met_value=5.0,
        calculated_calories=50.0,
        total_posture_faults=0,
    )
    payload = SessionSummaryPayload(
        sender="Orchestrator_Main",
        recipient="Agent_3_Coach",
        user_profile=user_profile,
        session_summary=session_summary,
        form_audit_diagnostics=[],
    )
    result = FitnessCoachAgent._fallback_summary(payload)
    assert "No major posture faults" in result


def test_fitness_coach_agent_build_summary_fallback(mock_session_payload, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    agent = FitnessCoachAgent()
    result = agent.build_summary(mock_session_payload)
    assert isinstance(result, str)
    assert len(result) > 0
    assert "10 squat reps" in result


def test_fitness_coach_agent_build_summary_with_openai_success(
    mock_session_payload, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")

    mock_response = MagicMock()
    mock_response.output_text = "Great job! Keep up the form work."

    mock_client = MagicMock()
    mock_client.responses.create.return_value = mock_response

    with patch("agents.coach_agent.OpenAI", return_value=mock_client):
        agent = FitnessCoachAgent()
        result = agent.build_summary(mock_session_payload)

        assert result == "Great job! Keep up the form work."
        mock_client.responses.create.assert_called_once()


def test_fitness_coach_agent_build_summary_with_openai_empty_response(
    mock_session_payload, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")

    mock_response = MagicMock()
    mock_response.output_text = None

    mock_client = MagicMock()
    mock_client.responses.create.return_value = mock_response

    with patch("agents.coach_agent.OpenAI", return_value=mock_client):
        agent = FitnessCoachAgent()
        result = agent.build_summary(mock_session_payload)

        assert isinstance(result, str)
        assert len(result) > 0
