from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from main import draw_hud


def test_draw_hud_basic_metrics():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    metrics = MagicMock()
    metrics.reps = 10
    metrics.state = "standing"
    metrics.knee_angle_deg = 165.5
    metrics.spine_angle_deg = 5.3
    metrics.posture_fault_count = 2
    metrics.posture_violation_active = False

    draw_hud(frame, 30.0, metrics, "")

    assert frame is not None
    assert frame.shape == (480, 640, 3)


def test_draw_hud_with_violation_active():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    metrics = MagicMock()
    metrics.reps = 5
    metrics.state = "descending"
    metrics.knee_angle_deg = 120.0
    metrics.spine_angle_deg = 25.0
    metrics.posture_fault_count = 1
    metrics.posture_violation_active = True

    draw_hud(frame, 25.5, metrics, "")

    non_black_pixels = np.any(frame != 0, axis=2).sum()
    assert non_black_pixels > 0


def test_draw_hud_with_latest_note():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    metrics = MagicMock()
    metrics.reps = 8
    metrics.state = "ascending"
    metrics.knee_angle_deg = 100.0
    metrics.spine_angle_deg = 10.0
    metrics.posture_fault_count = 3
    metrics.posture_violation_active = False

    note = "Keep your chest up and core tight for better form"
    draw_hud(frame, 28.3, metrics, note)

    non_black_pixels = np.any(frame != 0, axis=2).sum()
    assert non_black_pixels > 0


def test_draw_hud_with_long_note():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    metrics = MagicMock()
    metrics.reps = 12
    metrics.state = "bottom"
    metrics.knee_angle_deg = 85.0
    metrics.spine_angle_deg = 8.0
    metrics.posture_fault_count = 0
    metrics.posture_violation_active = False

    long_note = "This is a very long note that should be truncated to fit within the display area" * 3
    draw_hud(frame, 30.0, metrics, long_note)

    assert frame is not None


def test_draw_hud_with_violation_and_note():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    metrics = MagicMock()
    metrics.reps = 3
    metrics.state = "standing"
    metrics.knee_angle_deg = 170.0
    metrics.spine_angle_deg = 20.0
    metrics.posture_fault_count = 4
    metrics.posture_violation_active = True

    note = "Brace your core harder"
    draw_hud(frame, 24.0, metrics, note)

    non_black_pixels = np.any(frame != 0, axis=2).sum()
    assert non_black_pixels > 0
