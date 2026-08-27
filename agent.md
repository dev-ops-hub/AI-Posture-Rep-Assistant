# Multi-Agent Architecture & Specification (`agent.md`)

## 1. System Architecture Overview (Python Stack)
This application uses a hybrid edge-cloud **Python Multi-Agent Architecture**. Computer vision, landmark tracking, and vector geometry run locally on Python (`mediapipe` + `numpy` + `opencv`), maintaining zero-latency video processing, while generative AI agents leverage the `openai` Python SDK.

```
 [ OpenCV Camera Stream (30 FPS) ]
                 │
                 ▼
┌────────────────────────────────────────────────────────┐
│  Agent 1: Edge Vision & Tracking Agent (Python)        │
│  - Runs `mediapipe.solutions.pose` locally            │
│  - Computes joint angles via `numpy` vector math       │
│  - Executes Squat FSM state transitions                │
│  - Detects continuous posture deviation thresholds     │
└──────────────┬─────────────────────────────────────────┘
               │ (Emits Violation Triggers / Telemetry JSON)
               ├─────────────────────────────────────────┐
               ▼                                         ▼
┌───────────────────────────────┐       ┌───────────────────────────────┐
│ Agent 2: Form Audit Agent     │       │ Agent 3: Fitness Coach Agent  │
│ (Python - GPT-4o Vision)      │       │ (Python - GPT-4o-mini)        │
│ - Encodes JPEG base64 frame   │       │ - Aggregates MET calories     │
│ - Diagnoses visual error      │       │ - Produces session summary    │
└──────────────┬────────────────┘       └──────────────┬────────────────┘
               │                                       │
               └───────────────────┬───────────────────┘
                                   ▼
                   [ OpenCV HUD / Audio Chime Output ]
```

---

## 2. Agent Definitions & Specifications

### Agent 1: Edge Vision & Tracking Agent
* **Runtime Environment:** Local Python Interpreter (`vision_agent.py`).
* **Primary Libraries:** `mediapipe`, `numpy`, `opencv-python`.
* **Primary Responsibilities:**
  1. Capture video feed from camera at 30 FPS with 0 ms network delay.
  2. Extract 33 spatial keypoint coordinates via MediaPipe Pose.
  3. Execute squat state machine transitions (`STANDING` -> `BOTTOM` -> `ASCENDING` -> rep complete).
  4. Track slouching angle (`spine_angle_deg > 15` for more than `2.0s`).
* **Execution Trigger:** Continuous frame loop (`while cap.isOpened()`).
* **Cost:** **$0.00** (Runs locally on CPU).

---

### Agent 2: Form Audit Agent
* **Runtime Environment:** Cloud API via Python SDK (`audit_agent.py`).
* **Primary Libraries:** `openai`, `base64`, `cv2`.
* **Primary Responsibilities:**
  1. Convert current OpenCV NumPy frame to JPEG bytes and encode to Base64.
  2. Send snapshot to `gpt-4o` Vision model with biomechanical diagnostic prompt.
  3. Return a concise, 1-to-2 sentence corrective note.
* **Execution Trigger:** Event-driven (Triggered when `POSTURE_VIOLATION` lasts $>2.0	ext{s}$; max 2 calls per set).
* **Cost Estimate:** **~$0.0025 per snapshot**.

---

### Agent 3: Fitness Coach Agent
* **Runtime Environment:** Cloud API via Python SDK (`coach_agent.py`).
* **Primary Libraries:** `openai`, `json`.
* **Primary Responsibilities:**
    1. Aggregate total reps, exercise duration, posture fault counts, and computed MET calories:
      `calories_burned = MET * weight_kg * (duration_seconds / 3600)`
  2. Synthesize session metrics with Form Audit Agent notes.
  3. Generate a personalized post-workout report string printed to screen or UI.
* **Execution Trigger:** On-demand (Triggered when user presses 'q' to finish workout).
* **Cost Estimate:** **~$0.0001 per workout session**.

---

## 3. Agent Communication Protocols (Python Data Classes / Dicts)

### Schema A: Agent 1 -> Agent 2 (Posture Violation Event)
```python
{
    "sender": "Agent_1_Vision",
    "recipient": "Agent_2_FormAudit",
    "event_type": "POSTURE_VIOLATION",
    "exercise": "Squat",
    "metrics": {
        "spine_angle_deg": 22.4,
        "knee_angle_deg": 84.1,
        "violation_duration_sec": 2.1
    },
    "image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
}
```

### Schema B: Agent 1 & 2 -> Agent 3 (Session End Payload)
```python
{
    "sender": "Orchestrator_Main",
    "recipient": "Agent_3_Coach",
    "user_profile": {
        "weight_kg": 70,
        "goal": "weight loss"
    },
    "session_summary": {
        "exercise": "Squat",
        "total_reps": 20,
        "duration_seconds": 90,
        "met_value": 5.0,
        "calculated_calories": 8.75,
        "total_posture_faults": 2
    },
    "form_audit_diagnostics": [
        "Chest collapsing forward at the bottom of the movement."
    ]
}
```

---

## 4. Half-Day Python Implementation Workflow

1. **Implemented Agent 1:** `agents/vision_agent.py` runs MediaPipe locally, computes knee and spine angles, and tracks squat reps with a finite state machine.
2. **Implemented Agent 2:** `agents/audit_agent.py` sends posture snapshots to `gpt-4o` when `OPENAI_API_KEY` is available, otherwise falls back to deterministic local feedback.
3. **Implemented Agent 3:** `agents/coach_agent.py` computes MET calories and produces a session summary, with the same optional OpenAI fallback behavior.
4. **Implemented Orchestrator:** `main.py` owns the camera loop, HUD rendering, async audit dispatch, and final session summary output.

## 5. Runtime Notes

- Install dependencies with `uv pip install -r requirements.txt` after creating the environment with `uv venv`.
- Run the app with `uv run main.py`.
- Optional environment variables: `OPENAI_API_KEY`, `USER_WEIGHT_KG`, `FITNESS_GOAL`, `WORKOUT_MET`.
- Without an OpenAI API key, the application still runs locally and uses built-in coaching fallbacks.
