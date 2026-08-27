from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import os
import time

import cv2
from dotenv import load_dotenv

from agents import FitnessCoachAgent, FormAuditAgent, SessionSummaryPayload, VisionAgent, VoiceAgent
from agents.models import SessionSummary, UserProfile


def draw_hud(frame, fps: float, metrics, latest_note: str) -> None:
    cv2.putText(frame, f"Reps: {metrics.reps}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"State: {metrics.state}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Knee angle: {metrics.knee_angle_deg:.1f}", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Spine angle: {metrics.spine_angle_deg:.1f}", (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Faults: {metrics.posture_fault_count}", (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if metrics.posture_violation_active:
        cv2.rectangle(frame, (15, 205), (440, 250), (0, 0, 255), -1)
        cv2.putText(frame, "Posture alert: chest up, brace harder", (25, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if latest_note:
        cv2.rectangle(frame, (15, 265), (620, 330), (30, 30, 30), -1)
        cv2.putText(frame, latest_note[:78], (25, 305), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


def main() -> None:
    load_dotenv()

    weight_kg = float(os.getenv("USER_WEIGHT_KG", "70"))
    goal = os.getenv("FITNESS_GOAL", "general fitness")
    met_value = float(os.getenv("WORKOUT_MET", "5.0"))
    camera_index = int(os.getenv("CAMERA_INDEX", "0"))

    vision_agent = VisionAgent()
    audit_agent = FormAuditAgent()
    coach_agent = FitnessCoachAgent()
    voice_agent = VoiceAgent()
    executor = ThreadPoolExecutor(max_workers=1)
    audit_future: Future[str] | None = None
    audit_notes: list[str] = []
    latest_note = ""
    last_rep_count = 0
    posture_alert_count = 0

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open webcam at CAMERA_INDEX={camera_index}. Check macOS camera permissions and device availability."
        )

    started_at = time.monotonic()
    last_frame_at = started_at
    voice_agent.announce_workout_start()

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            now = time.monotonic()
            fps = 1.0 / max(now - last_frame_at, 1e-6)
            last_frame_at = now

            annotated, metrics, violation_event = vision_agent.process_frame(frame)

            if metrics.reps > last_rep_count:
                last_rep_count = metrics.reps
                voice_agent.announce_rep(metrics.reps)

            if violation_event and audit_future is None:
                voice_agent.announce_posture_alert(posture_alert_count)
                posture_alert_count += 1
                audit_future = executor.submit(audit_agent.audit_posture, violation_event)

            if audit_future is not None and audit_future.done():
                latest_note = audit_future.result().strip()
                if latest_note:
                    audit_notes.append(latest_note)
                    is_first_note = len(audit_notes) == 1
                    voice_agent.announce_form_note(latest_note, is_first_note)
                audit_future = None

            draw_hud(annotated, fps, metrics, latest_note)
            cv2.imshow("AI Posture Rep Assistant", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        duration_seconds = time.monotonic() - started_at
        cap.release()
        cv2.destroyAllWindows()
        vision_agent.close()
        voice_agent.close()
        executor.shutdown(wait=False, cancel_futures=True)

    calories = coach_agent.calculate_calories(met_value, weight_kg, duration_seconds)
    payload = SessionSummaryPayload(
        sender="Orchestrator_Main",
        recipient="Agent_3_Coach",
        user_profile=UserProfile(weight_kg=weight_kg, goal=goal),
        session_summary=SessionSummary(
            exercise="Squat",
            total_reps=vision_agent.reps,
            duration_seconds=duration_seconds,
            met_value=met_value,
            calculated_calories=calories,
            total_posture_faults=vision_agent.posture_fault_count,
        ),
        form_audit_diagnostics=audit_notes,
    )
    voice_agent.announce_session_complete(
        vision_agent.reps, duration_seconds, vision_agent.posture_fault_count
    )
    print(coach_agent.build_summary(payload))


if __name__ == "__main__":
    main()
