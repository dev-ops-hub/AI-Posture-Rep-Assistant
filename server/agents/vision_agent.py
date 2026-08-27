"""Agent 1: Vision & Tracking Agent.

Runs entirely locally (no network calls) using MediaPipe Pose, OpenCV, and
NumPy to turn raw webcam frames into:

- A squat rep count, driven by a finite state machine over the knee angle.
- Sustained forward-lean posture-fault detection, driven by the spine angle.
- An annotated frame (pose skeleton overlay) for display.
- ``PostureViolationEvent`` payloads for the Form Audit Agent when a fault
  has been sustained long enough to warrant an AI form check.

See the module-level ``VisionAgent`` docstring for details on the
rep-counting robustness measures (dual-leg averaging, visibility gating,
smoothing, and the minimum inter-rep cooldown).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import time
from typing import Any

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover - depends on local environment
    mp = None

from .models import PostureViolationEvent, PostureViolationMetrics


class SquatState(str, Enum):
    """Finite states of the squat rep-counting state machine.

    A rep is counted on the transition from ``ASCENDING`` (or, for fast reps
    sampled at a low frame rate, directly from ``BOTTOM``) back to
    ``STANDING``: ``STANDING -> DESCENDING -> BOTTOM -> ASCENDING -> STANDING``.
    """

    STANDING = "standing"
    DESCENDING = "descending"
    BOTTOM = "bottom"
    ASCENDING = "ascending"


@dataclass(slots=True)
class PoseFrameMetrics:
    """Per-frame output of :meth:`VisionAgent.process_frame`.

    Attributes:
        reps: Total reps counted so far this session.
        knee_angle_deg: Smoothed knee angle for this frame (degrees).
        spine_angle_deg: Spine lean angle for this frame (degrees from vertical).
        posture_fault_count: Total sustained posture faults detected so far.
        posture_violation_active: Whether a posture violation is currently in progress.
        posture_violation_duration: How long (seconds) the current violation has lasted.
        audit_ready: Whether this frame triggered a new ``PostureViolationEvent``.
        state: Current :class:`SquatState` value, as a string.
        landmarks_detected: Whether MediaPipe found a pose in this frame.
    """

    reps: int = 0
    knee_angle_deg: float = 180.0
    spine_angle_deg: float = 0.0
    posture_fault_count: int = 0
    posture_violation_active: bool = False
    posture_violation_duration: float = 0.0
    audit_ready: bool = False
    state: str = SquatState.STANDING.value
    landmarks_detected: bool = False


class VisionAgent:
    """Local pose-tracking agent: rep counting + posture-fault detection.

    Rep-counting robustness: the raw per-frame knee angle is never fed
    directly into the state machine. Instead, each frame:

    1. Both legs' knee angles are computed (from 3D landmark coordinates)
       and averaged when both are reliably visible, falling back to
       whichever single leg is visible, or holding the previous smoothed
       value if neither leg meets ``min_landmark_visibility``
       (:meth:`_estimate_knee_angle`).
    2. The combined reading is passed through an exponential moving average
       (:meth:`_smooth_knee_angle`, weight ``knee_angle_smoothing``) to filter
       out per-frame MediaPipe landmark jitter.
    3. Only the smoothed angle drives :meth:`_update_rep_state`, which also
       enforces a minimum cooldown (``min_rep_interval_sec``) between two
       counted reps as a defense-in-depth guard against double-counting.

    These measures address a class of accuracy issues where a single noisy
    frame (or a briefly occluded leg) could otherwise flip the rep state
    machine without any real movement having occurred.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        posture_threshold_deg: float = 15.0,
        violation_hold_seconds: float = 2.0,
        max_audits_per_session: int = 2,
        standing_angle_threshold_deg: float = 160.0,
        bottom_angle_threshold_deg: float = 90.0,
        knee_angle_smoothing: float = 0.4,
        min_rep_interval_sec: float = 0.3,
        min_landmark_visibility: float = 0.5,
    ) -> None:
        """Initializes the agent and (if MediaPipe is available) the pose model.

        Args:
            min_detection_confidence: MediaPipe Pose detection confidence threshold.
            min_tracking_confidence: MediaPipe Pose tracking confidence threshold.
            posture_threshold_deg: Spine angle (degrees) above which a forward-lean
                posture fault begins accumulating.
            violation_hold_seconds: How long a posture fault must persist before
                a ``PostureViolationEvent`` is emitted.
            max_audits_per_session: Maximum number of Form Audit events emitted
                per session (keeps OpenAI API usage bounded).
            standing_angle_threshold_deg: Knee angle (degrees) at/above which the
                user is considered fully standing.
            bottom_angle_threshold_deg: Knee angle (degrees) at/below which the
                user is considered at the bottom of the squat.
            knee_angle_smoothing: Exponential-moving-average weight (0-1) applied
                to each new raw knee-angle reading; lower values smooth out more
                jitter at the cost of a small amount of lag.
            min_rep_interval_sec: Minimum time that must elapse between two
                counted reps.
            min_landmark_visibility: Minimum MediaPipe landmark visibility score
                required before a leg's knee angle is trusted for a given frame.
        """
        self.posture_threshold_deg = posture_threshold_deg
        self.violation_hold_seconds = violation_hold_seconds
        self.max_audits_per_session = max_audits_per_session
        self.standing_angle_threshold_deg = standing_angle_threshold_deg
        self.bottom_angle_threshold_deg = bottom_angle_threshold_deg
        # Exponential-moving-average weight applied to each new raw knee-angle
        # reading. Lower values smooth out more per-frame landmark jitter
        # (which otherwise causes spurious rep counts) at the cost of a small
        # amount of lag.
        self.knee_angle_smoothing = knee_angle_smoothing
        # Minimum time that must elapse between two counted reps. Acts as a
        # defense-in-depth guard against double-counting a single rep if the
        # smoothed angle still oscillates around the standing threshold.
        self.min_rep_interval_sec = min_rep_interval_sec
        # Minimum MediaPipe landmark visibility score required before a leg's
        # knee angle is trusted; frames where both legs are occluded/out of
        # frame keep the last known smoothed angle instead of reacting to
        # unreliable coordinates.
        self.min_landmark_visibility = min_landmark_visibility
        self.reps = 0
        self.posture_fault_count = 0
        self.state = SquatState.STANDING
        self.violation_started_at: float | None = None
        self.last_violation_notified_at: float | None = None
        self.last_rep_completed_at: float | None = None
        self.audit_count = 0
        self._smoothed_knee_angle: float | None = None
        self.mp_pose = mp.solutions.pose if mp else None
        self.mp_drawing = mp.solutions.drawing_utils if mp else None
        self.pose = (
            self.mp_pose.Pose(
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            if self.mp_pose
            else None
        )

    def close(self) -> None:
        """Releases the underlying MediaPipe Pose model, if one was created."""
        if self.pose is not None:
            self.pose.close()

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, PoseFrameMetrics, PostureViolationEvent | None]:
        """Processes a single BGR video frame.

        Runs MediaPipe Pose on the frame, updates the rep-counting and
        posture-fault state machines, and draws the pose skeleton overlay
        onto a copy of the frame.

        Args:
            frame: A single BGR frame as a NumPy array (e.g. from ``cv2.VideoCapture``).

        Returns:
            A 3-tuple of:
                - the annotated frame (pose skeleton drawn on a copy of ``frame``),
                - a :class:`PoseFrameMetrics` snapshot for this frame, and
                - a :class:`PostureViolationEvent` if this frame triggered a new
                  Form Audit request, otherwise ``None``.
        """
        metrics = PoseFrameMetrics(reps=self.reps, posture_fault_count=self.posture_fault_count, state=self.state.value)
        violation_event = None

        if self.pose is None or self.mp_pose is None or self.mp_drawing is None:
            annotated = frame.copy()
            cv2.putText(
                annotated,
                "MediaPipe not installed",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
            return annotated, metrics, None

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        annotated = frame.copy()

        if not results.pose_landmarks:
            self.violation_started_at = None
            return annotated, metrics, None

        self.mp_drawing.draw_landmarks(
            annotated,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
        )

        metrics.landmarks_detected = True
        landmarks = results.pose_landmarks.landmark
        left_shoulder = self._landmark_xy(landmarks, self.mp_pose.PoseLandmark.LEFT_SHOULDER)
        left_hip_2d = self._landmark_xy(landmarks, self.mp_pose.PoseLandmark.LEFT_HIP)

        raw_knee_angle = self._estimate_knee_angle(landmarks)
        knee_angle = self._smooth_knee_angle(raw_knee_angle)
        spine_angle = self._spine_angle(left_shoulder, left_hip_2d)

        self._update_rep_state(knee_angle)
        violation_duration, audit_ready = self._update_posture_state(spine_angle)

        metrics.reps = self.reps
        metrics.knee_angle_deg = knee_angle
        metrics.spine_angle_deg = spine_angle
        metrics.posture_fault_count = self.posture_fault_count
        metrics.posture_violation_active = violation_duration > 0.0
        metrics.posture_violation_duration = violation_duration
        metrics.audit_ready = audit_ready
        metrics.state = self.state.value

        if audit_ready:
            violation_event = self._build_violation_event(frame, knee_angle, spine_angle, violation_duration)

        return annotated, metrics, violation_event

    def _build_violation_event(
        self,
        frame: np.ndarray,
        knee_angle: float,
        spine_angle: float,
        violation_duration: float,
    ) -> PostureViolationEvent:
        success, encoded = cv2.imencode(".jpg", frame)
        image_base64 = ""
        if success:
            import base64

            image_base64 = "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("utf-8")

        return PostureViolationEvent(
            sender="Agent_1_Vision",
            recipient="Agent_2_FormAudit",
            event_type="POSTURE_VIOLATION",
            exercise="Squat",
            metrics=PostureViolationMetrics(
                spine_angle_deg=spine_angle,
                knee_angle_deg=knee_angle,
                violation_duration_sec=violation_duration,
            ),
            image_base64=image_base64,
        )

    def _update_rep_state(self, knee_angle: float) -> None:
        standing = self.standing_angle_threshold_deg
        bottom = self.bottom_angle_threshold_deg

        if knee_angle > standing and self.state in {SquatState.ASCENDING, SquatState.BOTTOM}:
            now = time.monotonic()
            can_count = (
                self.last_rep_completed_at is None
                or (now - self.last_rep_completed_at) >= self.min_rep_interval_sec
            )
            if can_count:
                self.reps += 1
                self.last_rep_completed_at = now
            self.state = SquatState.STANDING
            return
        if knee_angle > standing:
            self.state = SquatState.STANDING
        elif knee_angle < bottom:
            self.state = SquatState.BOTTOM
        elif self.state == SquatState.BOTTOM and knee_angle >= bottom:
            self.state = SquatState.ASCENDING
        else:
            self.state = SquatState.DESCENDING

    def _update_posture_state(self, spine_angle: float) -> tuple[float, bool]:
        now = time.monotonic()
        if spine_angle <= self.posture_threshold_deg:
            self.violation_started_at = None
            return 0.0, False

        if self.violation_started_at is None:
            self.violation_started_at = now
            return 0.0, False

        duration = now - self.violation_started_at
        audit_ready = False
        can_notify = self.audit_count < self.max_audits_per_session
        already_notified = self.last_violation_notified_at == self.violation_started_at
        if duration >= self.violation_hold_seconds and can_notify and not already_notified:
            self.audit_count += 1
            self.posture_fault_count += 1
            self.last_violation_notified_at = self.violation_started_at
            audit_ready = True
        return duration, audit_ready

    def _estimate_knee_angle(self, landmarks: list[Any]) -> float:
        """Computes the knee angle from whichever leg(s) are reliably visible.

        Averaging both legs (when both are visible) cancels out a portion of
        per-landmark tracking noise, and falling back to a single visible leg
        makes the reading robust to one leg being briefly occluded or turned
        away from the camera. If neither leg meets the visibility threshold,
        the previous smoothed angle is reused instead of trusting a
        low-confidence (potentially wildly wrong) reading.
        """
        left_angle, left_visibility = self._leg_knee_angle(
            landmarks,
            self.mp_pose.PoseLandmark.LEFT_HIP,
            self.mp_pose.PoseLandmark.LEFT_KNEE,
            self.mp_pose.PoseLandmark.LEFT_ANKLE,
        )
        right_angle, right_visibility = self._leg_knee_angle(
            landmarks,
            self.mp_pose.PoseLandmark.RIGHT_HIP,
            self.mp_pose.PoseLandmark.RIGHT_KNEE,
            self.mp_pose.PoseLandmark.RIGHT_ANKLE,
        )

        angles = []
        if left_visibility >= self.min_landmark_visibility:
            angles.append(left_angle)
        if right_visibility >= self.min_landmark_visibility:
            angles.append(right_angle)

        if angles:
            return sum(angles) / len(angles)
        # Neither leg is reliably visible this frame; hold the last known
        # smoothed value rather than reacting to noisy/occluded landmarks.
        return self._smoothed_knee_angle if self._smoothed_knee_angle is not None else 180.0

    def _smooth_knee_angle(self, raw_angle: float) -> float:
        if self._smoothed_knee_angle is None:
            self._smoothed_knee_angle = raw_angle
        else:
            alpha = self.knee_angle_smoothing
            self._smoothed_knee_angle = alpha * raw_angle + (1 - alpha) * self._smoothed_knee_angle
        return self._smoothed_knee_angle

    def _leg_knee_angle(
        self, landmarks: list[Any], hip_id: Any, knee_id: Any, ankle_id: Any
    ) -> tuple[float, float]:
        hip = self._landmark_xyz(landmarks, hip_id)
        knee = self._landmark_xyz(landmarks, knee_id)
        ankle = self._landmark_xyz(landmarks, ankle_id)
        visibility = min(
            self._landmark_visibility(landmarks, hip_id),
            self._landmark_visibility(landmarks, knee_id),
            self._landmark_visibility(landmarks, ankle_id),
        )
        return self._angle(hip, knee, ankle), visibility

    @staticmethod
    def _landmark_xy(landmarks: list[Any], landmark_id: Any) -> np.ndarray:
        landmark = landmarks[landmark_id]
        return np.array([landmark.x, landmark.y], dtype=np.float32)

    @staticmethod
    def _landmark_xyz(landmarks: list[Any], landmark_id: Any) -> np.ndarray:
        landmark = landmarks[landmark_id]
        # Including MediaPipe's estimated depth (z) makes the knee angle far
        # less sensitive to the user's orientation relative to the camera
        # than a pure 2D (x, y) calculation, which otherwise over/under
        # reports the angle when the person isn't perfectly side-on.
        return np.array([landmark.x, landmark.y, landmark.z], dtype=np.float32)

    @staticmethod
    def _landmark_visibility(landmarks: list[Any], landmark_id: Any) -> float:
        return float(landmarks[landmark_id].visibility)

    @staticmethod
    def _angle(point_a: np.ndarray, point_b: np.ndarray, point_c: np.ndarray) -> float:
        vector_ba = point_a - point_b
        vector_bc = point_c - point_b
        denominator = np.linalg.norm(vector_ba) * np.linalg.norm(vector_bc)
        if denominator < 1e-9:
            return 180.0
        cosine = float(np.dot(vector_ba, vector_bc) / denominator)
        cosine = max(-1.0, min(1.0, cosine))
        return math.degrees(math.acos(cosine))

    @staticmethod
    def _spine_angle(shoulder: np.ndarray, hip: np.ndarray) -> float:
        torso_vector = shoulder - hip
        vertical_reference = np.array([0.0, -1.0], dtype=np.float32)
        cosine = float(np.dot(torso_vector, vertical_reference) / (np.linalg.norm(torso_vector) * np.linalg.norm(vertical_reference)))
        cosine = max(-1.0, min(1.0, cosine))
        return math.degrees(math.acos(cosine))
