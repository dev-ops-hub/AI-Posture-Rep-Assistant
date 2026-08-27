from agents.models import (
    PostureViolationEvent,
    PostureViolationMetrics,
    SessionSummary,
    SessionSummaryPayload,
    UserProfile,
)


def test_posture_violation_metrics():
    metrics = PostureViolationMetrics(
        spine_angle_deg=25.5,
        knee_angle_deg=120.3,
        violation_duration_sec=2.5,
    )
    assert metrics.spine_angle_deg == 25.5
    assert metrics.knee_angle_deg == 120.3
    assert metrics.violation_duration_sec == 2.5


def test_posture_violation_event_to_dict():
    metrics = PostureViolationMetrics(
        spine_angle_deg=25.567,
        knee_angle_deg=120.345,
        violation_duration_sec=2.567,
    )
    event = PostureViolationEvent(
        sender="Agent_1_Vision",
        recipient="Agent_2_FormAudit",
        event_type="POSTURE_VIOLATION",
        exercise="Squat",
        metrics=metrics,
        image_base64="data:image/jpeg;base64,test",
    )

    result = event.to_dict()

    assert result["sender"] == "Agent_1_Vision"
    assert result["recipient"] == "Agent_2_FormAudit"
    assert result["event_type"] == "POSTURE_VIOLATION"
    assert result["exercise"] == "Squat"
    assert result["metrics"]["spine_angle_deg"] == 25.57
    assert result["metrics"]["knee_angle_deg"] == 120.34
    assert result["metrics"]["violation_duration_sec"] == 2.57
    assert result["image_base64"] == "data:image/jpeg;base64,test"


def test_user_profile():
    profile = UserProfile(weight_kg=70.0, goal="general fitness")
    assert profile.weight_kg == 70.0
    assert profile.goal == "general fitness"


def test_session_summary():
    summary = SessionSummary(
        exercise="Squat",
        total_reps=10,
        duration_seconds=120.5,
        met_value=5.0,
        calculated_calories=50.25,
        total_posture_faults=2,
    )
    assert summary.exercise == "Squat"
    assert summary.total_reps == 10
    assert summary.duration_seconds == 120.5
    assert summary.met_value == 5.0
    assert summary.calculated_calories == 50.25
    assert summary.total_posture_faults == 2


def test_session_summary_payload_to_dict():
    user_profile = UserProfile(weight_kg=70.0, goal="general fitness")
    session_summary = SessionSummary(
        exercise="Squat",
        total_reps=10,
        duration_seconds=120.567,
        met_value=5.0,
        calculated_calories=50.345,
        total_posture_faults=2,
    )
    payload = SessionSummaryPayload(
        sender="Orchestrator_Main",
        recipient="Agent_3_Coach",
        user_profile=user_profile,
        session_summary=session_summary,
        form_audit_diagnostics=["Fix posture", "Keep chest up"],
    )

    result = payload.to_dict()

    assert result["sender"] == "Orchestrator_Main"
    assert result["recipient"] == "Agent_3_Coach"
    assert result["user_profile"]["weight_kg"] == 70.0
    assert result["user_profile"]["goal"] == "general fitness"
    assert result["session_summary"]["exercise"] == "Squat"
    assert result["session_summary"]["total_reps"] == 10
    assert result["session_summary"]["duration_seconds"] == 120.57
    assert result["session_summary"]["met_value"] == 5.0
    assert result["session_summary"]["calculated_calories"] == 50.34
    assert result["session_summary"]["total_posture_faults"] == 2
    assert result["form_audit_diagnostics"] == ["Fix posture", "Keep chest up"]


def test_session_summary_payload_default_diagnostics():
    user_profile = UserProfile(weight_kg=70.0, goal="general fitness")
    session_summary = SessionSummary(
        exercise="Squat",
        total_reps=10,
        duration_seconds=120.5,
        met_value=5.0,
        calculated_calories=50.25,
        total_posture_faults=0,
    )
    payload = SessionSummaryPayload(
        sender="Orchestrator_Main",
        recipient="Agent_3_Coach",
        user_profile=user_profile,
        session_summary=session_summary,
    )

    assert payload.form_audit_diagnostics == []
    result = payload.to_dict()
    assert result["form_audit_diagnostics"] == []
