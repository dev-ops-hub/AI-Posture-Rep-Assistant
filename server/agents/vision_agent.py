"""Agent 1: Vision & Tracking.

Runs MediaPipe Pose locally on webcam frames to count squat reps from knee-angle
transitions and detect sustained forward-lean posture faults from the spine angle.
When a posture fault holds long enough, it emits a `PostureViolationEvent` (with a
JPEG snapshot) for the Form Audit Agent to review.
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
    """Phase of a single squat rep, driven by the knee angle each frame.

    STANDING -> DESCENDING -> BOTTOM -> ASCENDING -> STANDING (rep counted).
    ASCENDING is sticky: once the knee angle leaves the bottom band while rising,
    the state stays ASCENDING (rather than reclassifying as DESCENDING) until it
    either completes the rep at the standing threshold or drops back into BOTTOM.
    See `VisionAgent._update_rep_state`.
    """

    STANDING = "standing"
    DESCENDING = "descending"
    BOTTOM = "bottom"
    ASCENDING = "ascending"


@dataclass(slots=True)
class PoseFrameMetrics:
    """Per-frame snapshot returned by `VisionAgent.process_frame` for the UI/telemetry."""

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
    """Local pose-tracking agent: counts reps and flags posture faults per frame.

    Wraps a MediaPipe `Pose` instance and tracks rep/posture state across calls to
    `process_frame`. If MediaPipe isn't installed, degrades gracefully: frames are
    annotated with a warning and no tracking is performed.
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
    ) -> None:
        """Configure detection thresholds for one workout session.

        Args:
            min_detection_confidence: MediaPipe Pose's minimum confidence to
                treat a frame as containing a person.
            min_tracking_confidence: MediaPipe Pose's minimum confidence to
                keep tracking landmarks between frames instead of re-detecting.
            posture_threshold_deg: Spine angle (degrees from vertical) above
                which the torso is considered to be leaning/folding forward.
            violation_hold_seconds: How long a posture fault must persist
                before it counts as a fault and is eligible for an audit.
            max_audits_per_session: Cap on how many posture snapshots are sent
                to the Form Audit Agent in one session.
            standing_angle_threshold_deg: Knee angle at/above which the user is
                considered fully standing (completes a rep).
            bottom_angle_threshold_deg: Knee angle at/below which the user is
                considered at the bottom of the squat.
        """
        self.posture_threshold_deg = posture_threshold_deg
        self.violation_hold_seconds = violation_hold_seconds
        self.max_audits_per_session = max_audits_per_session
        self.standing_angle_threshold_deg = standing_angle_threshold_deg
        self.bottom_angle_threshold_deg = bottom_angle_threshold_deg
        self.reps = 0
        self.posture_fault_count = 0
        self.state = SquatState.STANDING
        self.violation_started_at: float | None = None
        self.last_violation_notified_at: float | None = None
        self.audit_count = 0
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
        """Release the underlying MediaPipe `Pose` instance."""
        if self.pose is not None:
            self.pose.close()

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, PoseFrameMetrics, PostureViolationEvent | None]:
        """Run pose detection on one BGR frame and update rep/posture state.

        Args:
            frame: A single webcam frame in OpenCV's BGR format.

        Returns:
            A tuple of:
            - the frame annotated with pose landmarks (or an unmodified copy
              plus a warning overlay if MediaPipe is unavailable or no person
              was detected),
            - `PoseFrameMetrics` reflecting the state after this frame,
            - a `PostureViolationEvent` if this frame's posture fault just
              became eligible for an audit, otherwise `None`.
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
        left_hip = self._landmark_xy(landmarks, self.mp_pose.PoseLandmark.LEFT_HIP)
        left_knee = self._landmark_xy(landmarks, self.mp_pose.PoseLandmark.LEFT_KNEE)
        left_ankle = self._landmark_xy(landmarks, self.mp_pose.PoseLandmark.LEFT_ANKLE)
        left_shoulder = self._landmark_xy(landmarks, self.mp_pose.PoseLandmark.LEFT_SHOULDER)

        knee_angle = self._angle(left_hip, left_knee, left_ankle)
        spine_angle = self._spine_angle(left_shoulder, left_hip)

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
        """Package the current frame and metrics as an event for the Form Audit Agent."""
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
        """Advance the squat state machine and increment `self.reps` on completion.

        ASCENDING is treated as sticky: an in-between angle (neither standing nor
        bottom) only falls back to DESCENDING when the previous state was STANDING
        or DESCENDING itself. If the previous state was BOTTOM or ASCENDING, it
        stays/becomes ASCENDING. Without this, a real ascent that takes multiple
        frames to cross the standing threshold would flicker back to DESCENDING on
        every intermediate frame, and the rep-increment check below — which only
        fires from ASCENDING/BOTTOM — would never see the state it needs.
        """
        standing = self.standing_angle_threshold_deg
        bottom = self.bottom_angle_threshold_deg

        if knee_angle > standing:
            if self.state in {SquatState.ASCENDING, SquatState.BOTTOM}:
                self.reps += 1
            self.state = SquatState.STANDING
        elif knee_angle < bottom:
            self.state = SquatState.BOTTOM
        elif self.state in {SquatState.BOTTOM, SquatState.ASCENDING}:
            self.state = SquatState.ASCENDING
        else:
            self.state = SquatState.DESCENDING

    def _update_posture_state(self, spine_angle: float) -> tuple[float, bool]:
        """Track how long a posture fault has persisted and throttle audit triggers.

        A fault starts timing the first frame `spine_angle` exceeds
        `posture_threshold_deg`, and resets as soon as the angle recovers.
        Once a fault has held for `violation_hold_seconds`, it's counted once
        (`posture_fault_count`) and marked `audit_ready` once — subsequent
        frames of the same ongoing fault won't re-trigger — subject to the
        `max_audits_per_session` cap.

        Returns:
            A tuple of `(violation_duration_seconds, audit_ready)`, where
            `violation_duration_seconds` is 0.0 if there is no active fault.
        """
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

    @staticmethod
    def _landmark_xy(landmarks: list[Any], landmark_id: Any) -> np.ndarray:
        """Extract a MediaPipe landmark's normalized (x, y) as a float32 array."""
        landmark = landmarks[landmark_id]
        return np.array([landmark.x, landmark.y], dtype=np.float32)

    @staticmethod
    def _angle(point_a: np.ndarray, point_b: np.ndarray, point_c: np.ndarray) -> float:
        """Interior angle at vertex `point_b`, in degrees, formed by rays to `point_a` and `point_c`.

        Used for the knee angle, with hip/knee/ankle as a/b/c: ~180 degrees when
        the leg is straight (standing), decreasing toward ~90 degrees or below
        at the bottom of a squat.
        """
        angle = math.degrees(
            math.atan2(point_c[1] - point_b[1], point_c[0] - point_b[0])
            - math.atan2(point_a[1] - point_b[1], point_a[0] - point_b[0])
        )
        angle = abs(angle)
        return 360.0 - angle if angle > 180.0 else angle

    @staticmethod
    def _spine_angle(shoulder: np.ndarray, hip: np.ndarray) -> float:
        """Angle in degrees between the shoulder-to-hip vector and straight-up vertical.

        0 degrees is an upright torso; larger angles indicate forward lean and
        are compared against `posture_threshold_deg`.
        """
        torso_vector = shoulder - hip
        vertical_reference = np.array([0.0, -1.0], dtype=np.float32)
        cosine = float(np.dot(torso_vector, vertical_reference) / (np.linalg.norm(torso_vector) * np.linalg.norm(vertical_reference)))
        cosine = max(-1.0, min(1.0, cosine))
        return math.degrees(math.acos(cosine))