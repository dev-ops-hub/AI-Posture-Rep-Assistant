from __future__ import annotations

import os
import queue
import threading
from typing import Any

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None


class VoiceAgent:
    """Text-to-speech agent for real-time audio feedback during workouts."""

    def __init__(self, enabled: bool = True, rate: int = 150, volume: float = 1.0) -> None:
        """Initialize the voice agent.

        Args:
            enabled: Whether voice feedback is enabled (can be disabled via env var)
            rate: Speech rate in words per minute (default: 150, slower for clarity)
            volume: Volume level 0.0 to 1.0 (default: 1.0)
        """
        voice_enabled = os.getenv("VOICE_ENABLED", "true").lower() in ("true", "1", "yes")
        self.enabled = enabled and voice_enabled and pyttsx3 is not None

        if not self.enabled:
            self.engine = None
            self.speech_queue: queue.Queue[str | None] = queue.Queue()
            self.worker_thread = None
            return

        # Allow customization via environment variables
        try:
            rate = int(os.getenv("VOICE_RATE", str(rate)))
            volume = float(os.getenv("VOICE_VOLUME", str(volume)))
        except (ValueError, TypeError):
            pass  # Use defaults if env vars are invalid

        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", rate)
            self.engine.setProperty("volume", volume)

            voices = self.engine.getProperty("voices")
            if voices:
                for voice in voices:
                    if "english" in voice.name.lower():
                        self.engine.setProperty("voice", voice.id)
                        break
        except Exception:
            self.enabled = False
            self.engine = None
            self.speech_queue = queue.Queue()
            self.worker_thread = None
            return

        self.speech_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.worker_thread.start()

    def _speech_worker(self) -> None:
        """Background worker thread that processes speech requests."""
        while True:
            text = self.speech_queue.get()
            if text is None:
                break
            if self.engine:
                try:
                    self.engine.say(text)
                    self.engine.runAndWait()
                except Exception:
                    pass

    def speak(self, text: str, priority: bool = False) -> None:
        """Queue text to be spoken.

        Args:
            text: Text to speak
            priority: If True, clear queue and speak immediately
        """
        if not self.enabled or not text:
            return

        if priority:
            while not self.speech_queue.empty():
                try:
                    self.speech_queue.get_nowait()
                except queue.Empty:
                    break

        self.speech_queue.put(text)

    def announce_workout_start(self) -> None:
        """Announce the workout is starting."""
        self.speak("Workout starting. Get ready!", priority=True)

    def announce_rep(self, rep_count: int) -> None:
        """Announce a rep milestone.

        Args:
            rep_count: Current rep count
        """
        if rep_count % 5 == 0 and rep_count > 0:
            if rep_count == 5:
                self.speak("Good start! 5 reps completed")
            elif rep_count == 10:
                self.speak("Great progress! 10 reps done")
            elif rep_count % 10 == 0:
                self.speak(f"Excellent! {rep_count} reps completed")
            else:
                self.speak(f"Keep going! {rep_count} reps")

    def announce_posture_alert(self, alert_count: int = 0) -> None:
        """Announce a posture violation alert.

        Args:
            alert_count: Number of alerts so far this session
        """
        if alert_count == 0:
            message = "Form check. Keep your chest up and core braced"
        elif alert_count == 1:
            message = "Posture alert. Chest up, maintain neutral spine"
        else:
            message = "Watch your form. Stay upright"
        self.speak(message, priority=True)

    def announce_form_note(self, note: str, is_first: bool = True) -> None:
        """Announce a form correction note.

        Args:
            note: Form correction note from audit agent
            is_first: Whether this is the first correction this session
        """
        # Add context to make it clear this is coaching feedback
        prefix = "Coaching tip: " if is_first else "Remember: "
        # Simplify long technical feedback
        short_note = note[:80]
        message = f"{prefix}{short_note}"
        self.speak(message)

    def announce_session_complete(self, rep_count: int, duration_seconds: float, fault_count: int = 0) -> None:
        """Announce session completion.

        Args:
            rep_count: Total reps completed
            duration_seconds: Total duration in seconds
            fault_count: Number of posture faults detected
        """
        minutes = int(duration_seconds // 60)
        if minutes > 0:
            duration_text = f"{minutes} minute{'s' if minutes != 1 else ''}"
        else:
            duration_text = f"{int(duration_seconds)} seconds"

        # Build a more informative completion message
        if rep_count == 0:
            message = "Session ended. No reps detected. Try getting in frame and squatting lower."
        elif fault_count == 0:
            message = f"Perfect form! Workout complete. {rep_count} reps in {duration_text}. Excellent work!"
        elif fault_count <= 2:
            message = f"Great job! {rep_count} reps completed in {duration_text}. Minor form notes to review."
        else:
            message = f"Workout done. {rep_count} reps in {duration_text}. Focus on form corrections for next time."

        self.speak(message, priority=True)

    def close(self) -> None:
        """Stop the voice agent and clean up resources."""
        if not self.enabled:
            return

        self.speech_queue.put(None)
        if self.worker_thread:
            self.worker_thread.join(timeout=2.0)

        if self.engine:
            try:
                self.engine.stop()
            except Exception:
                pass
