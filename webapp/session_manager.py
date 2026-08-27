from __future__ import annotations

import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import cv2
import numpy as np

from agents import FitnessCoachAgent, FormAuditAgent, SessionSummaryPayload, VisionAgent, VoiceAgent
from agents.models import SessionSummary, UserProfile


class SessionState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    CLOSED = "closed"


def _placeholder_frame(message: str) -> bytes:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(
        frame,
        message,
        (40, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (200, 200, 200),
        2,
        cv2.LINE_AA,
    )
    ok, encoded = cv2.imencode(".jpg", frame)
    return encoded.tobytes() if ok else b""


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, secs = divmod(seconds, 60)
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _generate_improvement_tips(
    total_reps: int,
    total_faults: int,
    duration_seconds: float,
    audit_notes: list[str],
) -> list[str]:
    tips: list[str] = []

    if total_reps == 0:
        tips.append(
            "No reps were detected. Make sure your full body (hips to ankles) is visible "
            "in the camera frame and the room is well lit."
        )
        return tips

    if total_faults == 0:
        tips.append("Great job! No sustained posture faults were detected during this session.")
    else:
        tips.append(
            f"{total_faults} posture fault(s) were detected. Focus on keeping your chest up "
            "and spine neutral, especially during the descent."
        )

    seen: set[str] = set()
    for note in reversed(audit_notes):
        cleaned = note.strip()
        if cleaned and cleaned not in seen:
            tips.append(f"AI form note: {cleaned}")
            seen.add(cleaned)
        if len(seen) >= 3:
            break

    if duration_seconds > 0:
        reps_per_minute = total_reps / (duration_seconds / 60.0)
        if reps_per_minute > 30:
            tips.append("Your pace was quite fast. Slow down each rep to maintain control and depth.")
        elif reps_per_minute < 4:
            tips.append("Your pace was slow. Try to keep a steady tempo between reps once form feels solid.")

    if total_reps < 10:
        tips.append("Aim for at least 10-12 reps per set once your form is consistent.")

    return tips


@dataclass(slots=True)
class LiveMetrics:
    reps: int = 0
    state: str = "standing"
    knee_angle_deg: float = 180.0
    spine_angle_deg: float = 0.0
    posture_fault_count: int = 0
    posture_violation_active: bool = False
    latest_note: str = ""
    elapsed_seconds: float = 0.0
    session_state: str = SessionState.IDLE.value
    error: str = ""


@dataclass(slots=True)
class SessionConfig:
    camera_index: int = 0
    weight_kg: float = 70.0
    goal: str = "general fitness"
    met_value: float = 5.0


class WorkoutSessionManager:
    """Thread-safe orchestrator that drives the camera + agents for the web frontend."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._state = SessionState.IDLE
        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        self._vision_agent: VisionAgent | None = None
        self._audit_agent = FormAuditAgent()
        self._coach_agent = FitnessCoachAgent()
        self._voice_agent: VoiceAgent | None = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._audit_future: Future[str] | None = None

        self._audit_notes: list[str] = []
        self._latest_note = ""
        self._last_rep_count = 0
        self._posture_alert_count = 0

        self._config = SessionConfig()
        self._started_at = 0.0
        self._paused_at = 0.0
        self._total_paused_seconds = 0.0
        self._error = ""

        self._frame_lock = threading.Lock()
        self._latest_jpeg = _placeholder_frame("Press Start to begin your workout")

        self._metrics = LiveMetrics()
        self._report: dict[str, Any] | None = None

    # ------------------------------------------------------------------ #
    # Public control API
    # ------------------------------------------------------------------ #
    def start(self, config: SessionConfig) -> dict[str, Any]:
        with self._lock:
            if self._state in (SessionState.RUNNING, SessionState.PAUSED):
                return {"ok": False, "error": "A session is already in progress."}

            self._config = config
            cap = cv2.VideoCapture(config.camera_index)
            if not cap.isOpened():
                cap.release()
                self._error = (
                    f"Could not open webcam at CAMERA_INDEX={config.camera_index}. "
                    "Check camera permissions and device availability."
                )
                return {"ok": False, "error": self._error}

            self._cap = cap
            self._vision_agent = VisionAgent(
                standing_angle_threshold_deg=float(os.getenv("SQUAT_STANDING_ANGLE_DEG", "160")),
                bottom_angle_threshold_deg=float(os.getenv("SQUAT_BOTTOM_ANGLE_DEG", "110")),
            )
            self._voice_agent = VoiceAgent()
            self._audit_notes = []
            self._latest_note = ""
            self._last_rep_count = 0
            self._posture_alert_count = 0
            self._audit_future = None
            self._report = None
            self._error = ""
            self._started_at = time.monotonic()
            self._total_paused_seconds = 0.0
            self._paused_at = 0.0

            self._stop_event.clear()
            self._state = SessionState.RUNNING
            self._thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._thread.start()

            self._voice_agent.announce_workout_start()
            return {"ok": True, "state": self._state.value}

    def pause(self) -> dict[str, Any]:
        with self._lock:
            if self._state != SessionState.RUNNING:
                return {"ok": False, "error": "Session is not running."}
            self._state = SessionState.PAUSED
            self._paused_at = time.monotonic()
            return {"ok": True, "state": self._state.value}

    def resume(self) -> dict[str, Any]:
        with self._lock:
            if self._state != SessionState.PAUSED:
                return {"ok": False, "error": "Session is not paused."}
            self._total_paused_seconds += time.monotonic() - self._paused_at
            self._state = SessionState.RUNNING
            return {"ok": True, "state": self._state.value}

    def toggle_pause(self) -> dict[str, Any]:
        with self._lock:
            if self._state == SessionState.RUNNING:
                return self.pause()
            if self._state == SessionState.PAUSED:
                return self.resume()
            return {"ok": False, "error": "Session is not active."}

    def stop(self) -> dict[str, Any]:
        """Ends the workout session and produces the final report."""
        with self._lock:
            if self._state not in (SessionState.RUNNING, SessionState.PAUSED):
                if self._report is not None:
                    return {"ok": True, "state": SessionState.STOPPED.value, "report": self._report}
                return {"ok": False, "error": "No active session to stop."}

            if self._state == SessionState.PAUSED:
                self._total_paused_seconds += time.monotonic() - self._paused_at

            duration_seconds = max(time.monotonic() - self._started_at - self._total_paused_seconds, 0.0)
            self._state = SessionState.STOPPED
            self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=3.0)

        report = self._build_report(duration_seconds)

        with self._lock:
            self._report = report
            self._release_resources()

        return {"ok": True, "state": SessionState.STOPPED.value, "report": report}

    def quit(self) -> dict[str, Any]:
        """Fully terminates the current session and releases every resource."""
        result: dict[str, Any] = {"ok": True, "state": SessionState.CLOSED.value}
        with self._lock:
            active = self._state in (SessionState.RUNNING, SessionState.PAUSED)

        if active:
            stop_result = self.stop()
            result["report"] = stop_result.get("report")

        with self._lock:
            self._state = SessionState.CLOSED
            self._latest_jpeg = _placeholder_frame("Session closed. You can close this window.")
            self._executor.shutdown(wait=False, cancel_futures=True)

        return result

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            elapsed = 0.0
            if self._state in (SessionState.RUNNING, SessionState.PAUSED) and self._started_at:
                paused = self._total_paused_seconds
                if self._state == SessionState.PAUSED:
                    paused += time.monotonic() - self._paused_at
                elapsed = max(time.monotonic() - self._started_at - paused, 0.0)

            metrics = self._metrics
            return {
                "session_state": self._state.value,
                "reps": metrics.reps,
                "state": metrics.state,
                "knee_angle_deg": round(metrics.knee_angle_deg, 1),
                "spine_angle_deg": round(metrics.spine_angle_deg, 1),
                "posture_fault_count": metrics.posture_fault_count,
                "posture_violation_active": metrics.posture_violation_active,
                "latest_note": metrics.latest_note,
                "elapsed_seconds": round(elapsed, 1),
                "elapsed_formatted": _format_duration(elapsed),
                "error": self._error,
                "report": self._report,
            }

    def get_frame(self) -> bytes:
        with self._frame_lock:
            return self._latest_jpeg

    # ------------------------------------------------------------------ #
    # Background worker
    # ------------------------------------------------------------------ #
    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            with self._lock:
                state = self._state
                cap = self._cap
                vision_agent = self._vision_agent
                voice_agent = self._voice_agent

            if cap is None or vision_agent is None:
                break

            if state == SessionState.PAUSED:
                time.sleep(0.15)
                continue

            success, frame = cap.read()
            if not success:
                time.sleep(0.05)
                continue

            annotated, metrics, violation_event = vision_agent.process_frame(frame)

            if metrics.reps > self._last_rep_count:
                self._last_rep_count = metrics.reps
                if voice_agent:
                    voice_agent.announce_rep(metrics.reps)

            if violation_event and self._audit_future is None:
                if voice_agent:
                    voice_agent.announce_posture_alert(self._posture_alert_count)
                self._posture_alert_count += 1
                self._audit_future = self._executor.submit(self._audit_agent.audit_posture, violation_event)

            if self._audit_future is not None and self._audit_future.done():
                try:
                    note = self._audit_future.result().strip()
                except Exception:
                    note = ""
                if note:
                    self._audit_notes.append(note)
                    self._latest_note = note
                    if voice_agent:
                        voice_agent.announce_form_note(note, len(self._audit_notes) == 1)
                self._audit_future = None

            self._draw_hud(annotated, metrics, self._latest_note)

            ok, encoded = cv2.imencode(".jpg", annotated)
            if ok:
                with self._frame_lock:
                    self._latest_jpeg = encoded.tobytes()

            self._metrics = LiveMetrics(
                reps=metrics.reps,
                state=metrics.state,
                knee_angle_deg=metrics.knee_angle_deg,
                spine_angle_deg=metrics.spine_angle_deg,
                posture_fault_count=metrics.posture_fault_count,
                posture_violation_active=metrics.posture_violation_active,
                latest_note=self._latest_note,
            )

            time.sleep(0.01)

    @staticmethod
    def _draw_hud(frame, metrics, latest_note: str) -> None:
        cv2.putText(frame, f"Reps: {metrics.reps}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"State: {metrics.state}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(
            frame,
            f"Knee angle: {metrics.knee_angle_deg:.1f}",
            (20, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Spine angle: {metrics.spine_angle_deg:.1f}",
            (20, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Faults: {metrics.posture_fault_count}",
            (20, 155),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 165, 255),
            2,
        )

        if metrics.posture_violation_active:
            cv2.rectangle(frame, (15, 175), (440, 220), (0, 0, 255), -1)
            cv2.putText(frame, "Posture alert: chest up, brace harder", (25, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        if latest_note:
            cv2.rectangle(frame, (15, 235), (620, 295), (30, 30, 30), -1)
            cv2.putText(frame, latest_note[:78], (25, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # ------------------------------------------------------------------ #
    # Report + cleanup helpers
    # ------------------------------------------------------------------ #
    def _build_report(self, duration_seconds: float) -> dict[str, Any]:
        vision_agent = self._vision_agent
        total_reps = vision_agent.reps if vision_agent else 0
        total_faults = vision_agent.posture_fault_count if vision_agent else 0
        calories = self._coach_agent.calculate_calories(self._config.met_value, self._config.weight_kg, duration_seconds)

        payload = SessionSummaryPayload(
            sender="Orchestrator_WebApp",
            recipient="Agent_3_Coach",
            user_profile=UserProfile(weight_kg=self._config.weight_kg, goal=self._config.goal),
            session_summary=SessionSummary(
                exercise="Squat",
                total_reps=total_reps,
                duration_seconds=duration_seconds,
                met_value=self._config.met_value,
                calculated_calories=calories,
                total_posture_faults=total_faults,
            ),
            form_audit_diagnostics=self._audit_notes,
        )
        coach_summary = self._coach_agent.build_summary(payload)
        improvement_tips = _generate_improvement_tips(total_reps, total_faults, duration_seconds, self._audit_notes)

        if self._voice_agent:
            self._voice_agent.announce_session_complete(total_reps, duration_seconds, total_faults)

        return {
            "exercise": "Squat",
            "total_reps": total_reps,
            "duration_seconds": round(duration_seconds, 1),
            "duration_formatted": _format_duration(duration_seconds),
            "calories": round(calories, 2),
            "total_posture_faults": total_faults,
            "coach_summary": coach_summary,
            "improvement_tips": improvement_tips,
            "form_audit_diagnostics": list(self._audit_notes),
        }

    def _release_resources(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._vision_agent is not None:
            self._vision_agent.close()
            self._vision_agent = None
        if self._voice_agent is not None:
            self._voice_agent.close()
            self._voice_agent = None
        self._latest_jpeg = _placeholder_frame("Session ended. Press Start to begin a new workout.")
