from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from webapp.session_manager import SessionConfig, SessionState, WorkoutSessionManager


@pytest.fixture
def mock_camera():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.read.return_value = (True, frame)
    return cap


def _fake_process_frame(reps: int, faults: int):
    metrics = MagicMock()
    metrics.reps = reps
    metrics.state = "standing"
    metrics.knee_angle_deg = 170.0
    metrics.spine_angle_deg = 10.0
    metrics.posture_fault_count = faults
    metrics.posture_violation_active = False

    def _process(frame):
        return frame, metrics, None

    return _process


def test_start_fails_when_camera_unavailable():
    cap = MagicMock()
    cap.isOpened.return_value = False
    with patch("webapp.session_manager.cv2.VideoCapture", return_value=cap):
        manager = WorkoutSessionManager()
        result = manager.start(SessionConfig())
        assert result["ok"] is False
        assert "webcam" in result["error"].lower()


def test_full_session_lifecycle(mock_camera):
    with patch("webapp.session_manager.cv2.VideoCapture", return_value=mock_camera):
        manager = WorkoutSessionManager()
        start_result = manager.start(SessionConfig(weight_kg=70, goal="test", met_value=5.0))
        assert start_result == {"ok": True, "state": "running"}

        manager._vision_agent.process_frame = _fake_process_frame(reps=8, faults=1)
        manager._vision_agent.reps = 8
        manager._vision_agent.posture_fault_count = 1
        time.sleep(0.3)

        status = manager.get_status()
        assert status["session_state"] == "running"

        pause_result = manager.pause()
        assert pause_result["ok"] is True
        assert manager.get_status()["session_state"] == "paused"

        resume_result = manager.resume()
        assert resume_result["ok"] is True
        assert manager.get_status()["session_state"] == "running"

        stop_result = manager.stop()
        assert stop_result["ok"] is True
        report = stop_result["report"]
        assert report["total_reps"] == 8
        assert report["total_posture_faults"] == 1
        assert "improvement_tips" in report
        assert isinstance(report["improvement_tips"], list)

        quit_result = manager.quit()
        assert quit_result["ok"] is True
        assert quit_result["state"] == "closed"


def test_pause_without_active_session_fails():
    manager = WorkoutSessionManager()
    result = manager.pause()
    assert result["ok"] is False


def test_stop_without_active_session_fails():
    manager = WorkoutSessionManager()
    result = manager.stop()
    assert result["ok"] is False


def test_toggle_pause_resumes_when_paused(mock_camera):
    with patch("webapp.session_manager.cv2.VideoCapture", return_value=mock_camera):
        manager = WorkoutSessionManager()
        manager.start(SessionConfig())
        manager._vision_agent.process_frame = _fake_process_frame(reps=0, faults=0)

        manager.toggle_pause()
        assert manager.get_status()["session_state"] == "paused"
        manager.toggle_pause()
        assert manager.get_status()["session_state"] == "running"

        manager.quit()


def test_report_tips_for_zero_reps(mock_camera):
    with patch("webapp.session_manager.cv2.VideoCapture", return_value=mock_camera):
        manager = WorkoutSessionManager()
        manager.start(SessionConfig())
        manager._vision_agent.process_frame = _fake_process_frame(reps=0, faults=0)
        manager._vision_agent.reps = 0
        manager._vision_agent.posture_fault_count = 0
        time.sleep(0.2)

        stop_result = manager.stop()
        tips = stop_result["report"]["improvement_tips"]
        assert any("no reps were detected" in tip.lower() for tip in tips)
