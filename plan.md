# Implementation Plan: Python Multi-Agent AI Workout Tracker

## 1. Vision & Architecture Overview
This project implements a multi-agent AI system for real-time workout tracking, posture enforcement, and adaptive AI coaching using a **Python-based Multi-Agent Architecture**. OpenCV and MediaPipe process real-time webcam streams on the edge, while Python OpenAI SDK agents deliver biomechanical form audits and post-workout fitness reports.

### Multi-Agent Flow
```
 ┌────────────────────────────────────────────────────────┐
 │ 1. Vision & Tracking Agent (Python / OpenCV Edge)     │
 │    - MediaPipe Pose (30 FPS local tracking)            │
 │    - NumPy trigonometric posture & rep state logic     │
 └──────────────────────────┬─────────────────────────────┘
                            │ Emits Telemetry Events & Snapshots
                            ▼
 ┌────────────────────────────────────────────────────────┐
 │ 2. Form Audit Agent (Cloud Vision - OpenAI GPT-4o)     │
 │    - Triggered on persistent posture breaks            │
 │    - Biomechanical visual diagnosis from frame         │
 └──────────────────────────┬─────────────────────────────┘
                            │ Diagnostic Notes
                            ▼
 ┌────────────────────────────────────────────────────────┐
 │ 3. Fitness Coach Agent (Cloud LLM - GPT-4o-mini)       │
 │    - Session MET calorie aggregation                   │
 │    - Synthesizes feedback & post-workout summary       │
 └────────────────────────────────────────────────────────┘
```

---

## 2. Updated Python Tech Stack Matrix

| Layer | Component | Technology / Library | Role & Purpose |
| :--- | :--- | :--- | :--- |
| **Core Runtime** | Programming Language | Python 3.10+ | Unified language for CV pipeline, logic, and LLM orchestration. |
| **GUI / Video Pipeline**| Display & Stream | `opencv-python` (`cv2`) | Handles webcam input (`cv2.VideoCapture`), canvas drawing, and frame output. |
| **Edge Vision AI** | Pose Landmark Detection | `mediapipe` | Native Python SDK for 33 3D body keypoint tracking at 30+ FPS. |
| **Math & Vector Logic** | Angle Geometry | `numpy` | Vector dot products (`np.arctan2`) for fast joint angle calculation. |
| **Generative AI** | Cloud LLM Coach | `openai` Python SDK | Multimodal form auditing (`gpt-4o`) and session summaries (`gpt-4o-mini`). |
| **Feedback Overlay** | HUD Alerts | `opencv-python` (`cv2.putText`) | Immediate on-screen posture warnings without platform-specific audio dependencies. |

---

## 3. Project File Structure
```text
AI-Posture-Rep-Assistant/
├── main.py              # Main OpenCV loop and orchestrator
├── requirements.txt     # Python dependencies
├── agents/
│   ├── __init__.py
│   ├── models.py        # Shared agent payload data classes
│   ├── vision_agent.py  # Agent 1: pose tracking, angles, squat FSM
│   ├── audit_agent.py   # Agent 2: OpenAI vision audit with fallback
│   └── coach_agent.py   # Agent 3: session summary with fallback
├── agent.md
├── plan.md
└── README.md
```

---

## 4. Half-Day Execution Roadmap (Python Sprint Checklist)

### Task 1: Python Environment & Vision Agent Setup
- [x] Define Python dependencies in `requirements.txt`.
- [x] Build `cv2.VideoCapture(0)` loop in `main.py`.
- [x] Initialize `mediapipe.solutions.pose` in `agents/vision_agent.py` and draw landmarks onto frames.

### Task 2: Rep Engine & Posture Detection
- [x] Implement angle math in `agents/vision_agent.py` for knee and spine calculations.
- [x] Implement squat finite state transitions based on knee-angle thresholds.
- [x] Track posture violation duration and surface a visible HUD alert when the threshold is exceeded.

### Task 3: Multi-Agent Handoff & OpenAI Integration
- [x] Write `agents/audit_agent.py` to encode posture events and call `gpt-4o` when configured.
- [x] Write `agents/coach_agent.py` to calculate MET calories and build the session summary.
- [x] Dispatch posture audits on a background thread from `main.py` to avoid blocking the video loop.

### Task 4: UI Overlay & Testing
- [x] Render an OpenCV HUD for reps, state, posture alerts, and the latest audit note.
- [x] Print a final summary report to the terminal on exit.
- [ ] Validate the full webcam flow on a machine with the required Python packages and camera access.

## 5. Remaining Validation Work

- Install the dependencies in a virtual environment.
- Run `uv run main.py` with webcam access.
- Optionally set `OPENAI_API_KEY` to exercise the cloud audit and coach agents.
