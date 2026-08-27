"""The four agents that make up the AI Posture Rep Assistant multi-agent pipeline.

- ``VisionAgent`` (Agent 1): local pose tracking, rep counting, and posture-fault detection.
- ``FormAuditAgent`` (Agent 2): cloud vision-model form corrections on sustained faults.
- ``FitnessCoachAgent`` (Agent 3): calorie math and end-of-session coaching summary.
- ``VoiceAgent`` (Agent 4): local offline text-to-speech announcements.

Both orchestrators (``main.py`` for the desktop OpenCV window, and
``webapp/session_manager.py`` for the browser control panel) import and drive
these same agent classes.
"""

from .audit_agent import FormAuditAgent
from .coach_agent import FitnessCoachAgent
from .models import PostureViolationEvent, SessionSummaryPayload
from .vision_agent import PoseFrameMetrics, VisionAgent
from .voice_agent import VoiceAgent

__all__ = [
    "FormAuditAgent",
    "FitnessCoachAgent",
    "PoseFrameMetrics",
    "PostureViolationEvent",
    "SessionSummaryPayload",
    "VisionAgent",
    "VoiceAgent",
]
