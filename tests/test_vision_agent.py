import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from agents.vision_agent import PoseFrameMetrics, SquatState, VisionAgent


def test_squat_state_enum():
    assert SquatState.STANDING.value == "standing"
    assert SquatState.DESCENDING.value == "descending"
    assert SquatState.BOTTOM.value == "bottom"
    assert SquatState.ASCENDING.value == "ascending"


def test_pose_frame_metrics_defaults():
    metrics = PoseFrameMetrics()
    assert metrics.reps == 0
    assert metrics.knee_angle_deg == 180.0
    assert metrics.spine_angle_deg == 0.0
    assert metrics.posture_fault_count == 0
    assert metrics.posture_violation_active is False
    assert metrics.posture_violation_duration == 0.0
    assert metrics.audit_ready is False
    assert metrics.state == "standing"
    assert metrics.landmarks_detected is False


def test_vision_agent_angle_calculation():
    point_a = np.array([0.0, 1.0], dtype=np.float32)
    point_b = np.array([0.0, 0.0], dtype=np.float32)
    point_c = np.array([1.0, 0.0], dtype=np.float32)
    angle = VisionAgent._angle(point_a, point_b, point_c)
    assert abs(angle - 90.0) < 1.0


def test_vision_agent_spine_angle_vertical():
    shoulder = np.array([0.5, 0.3], dtype=np.float32)
    hip = np.array([0.5, 0.5], dtype=np.float32)
    angle = VisionAgent._spine_angle(shoulder, hip)
    assert abs(angle - 0.0) < 1.0


def test_vision_agent_spine_angle_forward_lean():
    shoulder = np.array([0.4, 0.3], dtype=np.float32)
    hip = np.array([0.5, 0.5], dtype=np.float32)
    angle = VisionAgent._spine_angle(shoulder, hip)
    assert angle > 0.0


def test_vision_agent_initialization():
    with patch("agents.vision_agent.mp") as mock_mp:
        mock_mp.solutions.pose = MagicMock()
        mock_mp.solutions.drawing_utils = MagicMock()

        agent = VisionAgent(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
            posture_threshold_deg=20.0,
            violation_hold_seconds=3.0,
            max_audits_per_session=3,
        )
        assert agent.posture_threshold_deg == 20.0
        assert agent.violation_hold_seconds == 3.0
        assert agent.max_audits_per_session == 3
        assert agent.reps == 0
        assert agent.posture_fault_count == 0
        assert agent.state == SquatState.STANDING
        assert agent.violation_started_at is None
        assert agent.last_violation_notified_at is None
        assert agent.audit_count == 0


def test_vision_agent_update_rep_state_standing_to_descending():
    with patch("agents.vision_agent.mp") as mock_mp:
        mock_mp.solutions.pose = MagicMock()
        mock_mp.solutions.drawing_utils = MagicMock()

        agent = VisionAgent()
        agent.state = SquatState.STANDING
        agent._update_rep_state(150.0)
        assert agent.state == SquatState.DESCENDING


def test_vision_agent_update_rep_state_descending_to_bottom():
    with patch("agents.vision_agent.mp") as mock_mp:
        mock_mp.solutions.pose = MagicMock()
        mock_mp.solutions.drawing_utils = MagicMock()

        agent = VisionAgent()
        agent.state = SquatState.DESCENDING
        agent._update_rep_state(85.0)
        assert agent.state == SquatState.BOTTOM


def test_vision_agent_update_rep_state_bottom_to_ascending():
    with patch("agents.vision_agent.mp") as mock_mp:
        mock_mp.solutions.pose = MagicMock()
        mock_mp.solutions.drawing_utils = MagicMock()

        agent = VisionAgent()
        agent.state = SquatState.BOTTOM
        agent._update_rep_state(95.0)
        assert agent.state == SquatState.ASCENDING


def test_vision_agent_update_rep_state_ascending_to_standing_increments_reps():
    with patch("agents.vision_agent.mp") as mock_mp:
        mock_mp.solutions.pose = MagicMock()
        mock_mp.solutions.drawing_utils = MagicMock()

        agent = VisionAgent()
        agent.state = SquatState.ASCENDING
        initial_reps = agent.reps
        agent._update_rep_state(165.0)
        assert agent.state == SquatState.STANDING
        assert agent.reps == initial_reps + 1


def test_vision_agent_update_rep_state_bottom_to_standing_increments_reps():
    with patch("agents.vision_agent.mp") as mock_mp:
        mock_mp.solutions.pose = MagicMock()
        mock_mp.solutions.drawing_utils = MagicMock()

        agent = VisionAgent()
        agent.state = SquatState.BOTTOM
        initial_reps = agent.reps
        agent._update_rep_state(165.0)
        assert agent.state == SquatState.STANDING
        assert agent.reps == initial_reps + 1


def test_vision_agent_update_posture_state_no_violation():
    with patch("agents.vision_agent.mp") as mock_mp:
        mock_mp.solutions.pose = MagicMock()
        mock_mp.solutions.drawing_utils = MagicMock()

        agent = VisionAgent(posture_threshold_deg=15.0)
        duration, audit_ready = agent._update_posture_state(10.0)
        assert duration == 0.0
        assert audit_ready is False
        assert agent.violation_started_at is None


def test_vision_agent_update_posture_state_violation_start():
    with patch("agents.vision_agent.mp") as mock_mp:
        mock_mp.solutions.pose = MagicMock()
        mock_mp.solutions.drawing_utils = MagicMock()

        agent = VisionAgent(posture_threshold_deg=15.0)
        duration, audit_ready = agent._update_posture_state(20.0)
        assert duration == 0.0
        assert audit_ready is False
        assert agent.violation_started_at is not None


def test_vision_agent_update_posture_state_violation_held_triggers_audit():
    with patch("agents.vision_agent.mp") as mock_mp:
        mock_mp.solutions.pose = MagicMock()
        mock_mp.solutions.drawing_utils = MagicMock()

        agent = VisionAgent(posture_threshold_deg=15.0, violation_hold_seconds=0.01)
        import time

        agent._update_posture_state(20.0)
        time.sleep(0.02)
        duration, audit_ready = agent._update_posture_state(20.0)
        assert duration >= 0.01
        assert audit_ready is True
        assert agent.posture_fault_count == 1
        assert agent.audit_count == 1


def test_vision_agent_update_posture_state_max_audits_limit():
    with patch("agents.vision_agent.mp") as mock_mp:
        mock_mp.solutions.pose = MagicMock()
        mock_mp.solutions.drawing_utils = MagicMock()

        agent = VisionAgent(
            posture_threshold_deg=15.0,
            violation_hold_seconds=0.01,
            max_audits_per_session=2,
        )
        import time

        for i in range(3):
            agent.violation_started_at = None
            agent._update_posture_state(20.0)
            time.sleep(0.02)
            _, audit_ready = agent._update_posture_state(20.0)
            if i < 2:
                assert audit_ready is True
            else:
                assert audit_ready is False

        assert agent.audit_count == 2
        assert agent.posture_fault_count == 2


def test_vision_agent_process_frame_no_mediapipe(monkeypatch):
    monkeypatch.setattr("agents.vision_agent.mp", None)
    agent = VisionAgent()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    annotated, metrics, violation_event = agent.process_frame(frame)
    assert annotated.shape == frame.shape
    assert metrics.reps == 0
    assert violation_event is None


@patch("agents.vision_agent.mp")
def test_vision_agent_process_frame_no_landmarks(mock_mp):
    mock_pose_instance = MagicMock()
    mock_pose_instance.process.return_value = MagicMock(pose_landmarks=None)
    mock_mp.solutions.pose.Pose.return_value = mock_pose_instance
    mock_mp.solutions.drawing_utils = MagicMock()

    agent = VisionAgent()
    agent.pose = mock_pose_instance
    agent.mp_pose = mock_mp.solutions.pose
    agent.mp_drawing = mock_mp.solutions.drawing_utils

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    annotated, metrics, violation_event = agent.process_frame(frame)
    assert violation_event is None
    assert agent.violation_started_at is None


def test_vision_agent_build_violation_event():
    with patch("agents.vision_agent.mp") as mock_mp:
        mock_mp.solutions.pose = MagicMock()
        mock_mp.solutions.drawing_utils = MagicMock()

        agent = VisionAgent()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        event = agent._build_violation_event(frame, 120.5, 25.3, 2.5)

        assert event.sender == "Agent_1_Vision"
        assert event.recipient == "Agent_2_FormAudit"
        assert event.event_type == "POSTURE_VIOLATION"
        assert event.exercise == "Squat"
        assert event.metrics.knee_angle_deg == 120.5
        assert event.metrics.spine_angle_deg == 25.3
        assert event.metrics.violation_duration_sec == 2.5
        assert event.image_base64.startswith("data:image/jpeg;base64,")


def test_vision_agent_close():
    with patch("agents.vision_agent.mp") as mock_mp:
        mock_mp.solutions.pose = MagicMock()
        mock_mp.solutions.drawing_utils = MagicMock()

        agent = VisionAgent()
        if agent.pose is not None:
            agent.close()
