# Implementation Plan: Python Multi-Agent AI Workout Tracker

## 1. Vision & Architecture Overview
This project implements a multi-agent AI system for real-time workout tracking, posture enforcement, voice feedback, and adaptive AI coaching using a **Python-based Multi-Agent Architecture**. OpenCV and MediaPipe process real-time webcam streams on the edge, while Python OpenAI SDK agents deliver biomechanical form audits and post-workout fitness reports. Real-time voice feedback provides audio coaching during workouts.

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
 └──────────────────────────┬─────────────────────────────┘
                            │
                            ▼
 ┌────────────────────────────────────────────────────────┐
 │ 4. Voice Agent (Local TTS - pyttsx3)                   │
 │    - Real-time audio coaching and encouragement        │
 │    - Contextual posture alerts and form corrections    │
 │    - Performance-adaptive session summaries            │
 └────────────────────────────────────────────────────────┘
```

---

## 2. Updated Python Tech Stack Matrix

| Layer | Component | Technology / Library | Role & Purpose |
| :--- | :--- | :--- | :--- |
| **Core Runtime** | Programming Language | Python 3.11+ | Unified language for CV pipeline, logic, and LLM orchestration. |
| **GUI / Video Pipeline**| Display & Stream | `opencv-python` (`cv2`) | Handles webcam input (`cv2.VideoCapture`), canvas drawing, and frame output. |
| **Edge Vision AI** | Pose Landmark Detection | `mediapipe==0.10.9` | Native Python SDK for 33 3D body keypoint tracking at 30+ FPS. |
| **Math & Vector Logic** | Angle Geometry | `numpy` | Vector dot products (`np.arctan2`) for fast joint angle calculation. |
| **Generative AI** | Cloud LLM Coach | `openai` Python SDK | Multimodal form auditing (`gpt-4o`) and session summaries (`gpt-4o-mini`). |
| **Voice Feedback** | Text-to-Speech | `pyttsx3` | Local offline TTS for real-time audio coaching without API dependencies. |
| **Feedback Overlay** | HUD Alerts | `opencv-python` (`cv2.putText`) | On-screen posture warnings and metrics display. |
| **Testing** | Unit Testing | `pytest`, `pytest-cov`, `pytest-mock` | Comprehensive test suite with 74% code coverage. |

---

## 3. Project File Structure
```text
AI-Posture-Rep-Assistant/
├── main.py                # Main OpenCV loop and orchestrator (desktop mode, stays in root)
├── requirements.txt       # Python dependencies
├── pyproject.toml         # Pytest config + packaging (editable install of server/agents)
├── server/                # Backend agent code + verification script
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── models.py          # Shared agent payload data classes
│   │   ├── vision_agent.py    # Agent 1: pose tracking, angles, squat FSM
│   │   ├── audit_agent.py     # Agent 2: OpenAI vision audit with fallback
│   │   ├── coach_agent.py     # Agent 3: session summary with fallback
│   │   └── voice_agent.py     # Agent 4: real-time audio coaching
│   └── verify_setup.py    # Component verification script
├── tests/                 # Test suite (top-level, kept alongside main.py/webapp)
│   ├── __init__.py
│   ├── test_models.py     # Data model tests (6 tests)
│   ├── test_vision_agent.py # Vision/pose tests (19 tests)
│   ├── test_audit_agent.py  # Audit agent tests (9 tests)
│   ├── test_coach_agent.py  # Coach agent tests (12 tests)
│   ├── test_voice_agent.py  # Voice agent tests (23 tests)
│   ├── test_main.py       # HUD rendering tests (5 tests)
│   └── test_webapp_session_manager.py # Web session lifecycle tests (6 tests)
├── webapp/                # Browser control panel (unchanged location)
│   ├── __init__.py
│   ├── app.py             # Flask routes: start/pause/stop/quit, status, MJPEG stream
│   ├── session_manager.py # WorkoutSessionManager state machine wrapping the 4 agents
│   ├── templates/
│   │   └── index.html     # Start/Pause/Stop/Quit UI, live stats, report modal
│   └── static/
│       ├── style.css      # Dark theme, responsive video panel (object-fit: contain)
│       └── app.js         # Polling, button handlers, report rendering
├── .env.example           # Environment configuration template
├── .gitignore             # Git ignore patterns
├── agent.md               # Agent architecture documentation
├── plan.md                # This file - implementation plan
├── README.md              # Project overview and setup
├── TESTING.md              # Testing documentation
├── VOICE_FEATURES.md      # Voice feedback documentation
└── VOICE_IMPROVEMENTS.md  # Voice clarity improvements
```

---

## 4. Implementation Status (✅ Complete)

### Task 1: Python Environment & Vision Agent Setup
- [x] Define Python dependencies in `requirements.txt`.
- [x] Build `cv2.VideoCapture(0)` loop in `main.py`.
- [x] Initialize `mediapipe.solutions.pose` in `server/agents/vision_agent.py` and draw landmarks onto frames.

### Task 2: Rep Engine & Posture Detection
- [x] Implement angle math in `server/agents/vision_agent.py` for knee and spine calculations.
- [x] Implement squat finite state transitions based on knee-angle thresholds.
- [x] Track posture violation duration and surface a visible HUD alert when the threshold is exceeded.

### Task 3: Multi-Agent Handoff & OpenAI Integration
- [x] Write `server/agents/audit_agent.py` to encode posture events and call `gpt-4o` when configured.
- [x] Write `server/agents/coach_agent.py` to calculate MET calories and build the session summary.
- [x] Dispatch posture audits on a background thread from `main.py` to avoid blocking the video loop.

### Task 4: UI Overlay & Testing
- [x] Render an OpenCV HUD for reps, state, posture alerts, and the latest audit note.
- [x] Print a final summary report to the terminal on exit.
- [x] Validate the full webcam flow on a machine with the required Python packages and camera access.

### Task 5: Voice Feedback System (NEW)
- [x] Implement `server/agents/voice_agent.py` with background threading for non-blocking TTS.
- [x] Add contextual voice announcements for workout start, reps, posture alerts, and session end.
- [x] Integrate voice feedback into main application loop.
- [x] Add environment variables for voice control (`VOICE_ENABLED`, `VOICE_RATE`, `VOICE_VOLUME`).

### Task 6: Voice Clarity Improvements (NEW)
- [x] Reduce speech rate from 175 to 150 WPM for better clarity during exercise.
- [x] Increase volume from 0.9 to 1.0 for better audibility while moving.
- [x] Add contextual prefixes to all voice messages (purpose identification).
- [x] Implement adaptive messaging based on performance (first alert vs. repeated, perfect form vs. faults).
- [x] Add workout start announcement and performance-adaptive session summaries.

### Task 7: Comprehensive Testing Suite (NEW)
- [x] Create unit tests for all agent modules (74 tests total).
- [x] Achieve 74% overall code coverage.
- [x] Set up pytest configuration with coverage reporting.
- [x] Create verification script for quick component testing.
- [x] Document testing procedures and patterns.

### Task 8: Web Frontend & Session Orchestration (NEW)
- [x] Build `webapp/session_manager.py`: a thread-safe `WorkoutSessionManager` state machine
      (`idle → running ⇄ paused → stopped/closed`) that reuses `VisionAgent`, `FormAuditAgent`,
      `FitnessCoachAgent`, and `VoiceAgent` inside a background capture thread.
- [x] Build `webapp/app.py`: Flask routes for `/api/start`, `/api/pause` (toggle), `/api/stop`,
      `/api/quit`, `/api/status` (polling), and `/video_feed` (MJPEG stream of annotated frames).
- [x] Build the browser UI (`webapp/templates/index.html`, `webapp/static/app.js`,
      `webapp/static/style.css`) with **Start / Pause / Stop / Quit** buttons, live stats, and a
      report modal shown after Stop/Quit.
- [x] Generate a workout report on Stop/Quit: total reps, duration, calories, posture faults, an
      AI coach summary, and heuristic "things to improve" tips (pace, faults, low rep count, and
      recent AI form-audit notes).
- [x] Fix the video panel to display the **entire camera frame** (`object-fit: contain`) instead
      of cropping it.
- [x] Make **Quit** fully shut down the Flask server process (`os.kill(pid, SIGINT)` after
      releasing camera/voice resources), not just end the workout session.
- [x] Add `tests/test_webapp_session_manager.py` (6 tests) covering the full lifecycle with a
      mocked camera.

### Task 9: Backend Reorganization into `server/` (NEW)
- [x] Move `agents/`, `tests/`, and `verify_setup.py` into a new `server/` folder
      (`server/agents/`, `server/tests/`, `server/verify_setup.py`); `main.py` stays in the
      repo root and `webapp/` is left completely untouched.
- [x] Configure `pyproject.toml` with `[tool.setuptools.packages.find] where = ["server"]` so
      `uv pip install -e .` installs `agents` as an editable, globally importable package —
      keeping `from agents import ...` working unchanged in `main.py` and every file under
      `webapp/`.
- [x] Update `pyproject.toml` pytest config: `testpaths = ["server/tests"]` and
      `pythonpath = [".", "server"]` (redundant safety net alongside the editable install).
- [x] Verify `uv run pytest`, `uv run main.py`, `uv run python -m webapp.app`, and
      `uv run python server/verify_setup.py` all still work after the move.

### Task 10: Move `tests/` Back to the Repo Root (NEW)
- [x] Move `server/tests/` back to a top-level `tests/` folder (`agents/` and `verify_setup.py`
      remain under `server/`).
- [x] Update `pyproject.toml`: `testpaths = ["tests"]` (the `pythonpath = [".", "server"]`
      setting still applies, so `agents` and `webapp` both resolve correctly for the test suite).
- [x] Re-verify `uv run pytest`, `uv run main.py`, `uv run python server/verify_setup.py`, and
      `uv run python -m webapp.app` all continue to work.

### Task 11: Rep-Counting Accuracy Improvements (NEW)
- [x] Diagnosed root causes of inaccurate rep counts: single-frame MediaPipe landmark jitter
      crossing the 90°/160° thresholds without real movement, left-leg-only tracking being
      unreliable when that leg is turned away from the camera or briefly occluded, and a pure
      2D (x, y) angle calculation being sensitive to the user's orientation relative to the
      camera.
- [x] Added dual-leg averaging with landmark-visibility gating (`_estimate_knee_angle`,
      `_leg_knee_angle`): averages both knees when both are reliably visible, falls back to
      whichever leg is visible otherwise, and holds the last known angle if neither leg meets
      `min_landmark_visibility` (default `0.5`).
- [x] Switched `_angle()` to a 3D-capable vector dot-product/arccos formula using MediaPipe's
      `(x, y, z)` landmark coordinates, reducing sensitivity to camera-facing angle.
- [x] Added exponential-moving-average smoothing of the knee angle (`_smooth_knee_angle`,
      `knee_angle_smoothing=0.4` default) before it reaches the rep state machine, filtering
      per-frame jitter that previously caused spurious `BOTTOM`/`STANDING` transitions.
- [x] Added a minimum inter-rep cooldown (`min_rep_interval_sec=0.3s` default) inside
      `_update_rep_state()` as a defense-in-depth guard against double-counting.
- [x] Made `standing_angle_threshold_deg` (160°) and `bottom_angle_threshold_deg` (90°)
      configurable constructor parameters instead of hardcoded magic numbers.
- [x] Added 9 new unit tests (28 total for the Vision Agent, up from 19) covering smoothing,
      dual-leg fallback, visibility gating, and the rep-interval debounce; all 19 pre-existing
      tests continue to pass unchanged.

---

## 5. Testing Infrastructure

### Test Coverage Summary
- **Total Tests:** 89 passing
- **Overall Coverage:** 77% (`agents/` package, physically located at `server/agents/`, + `main.py`; `webapp/` has its own dedicated test file)
- **Test Files:** 7, located in `tests/` at the repo root (models, vision, audit, coach, voice, main, webapp session manager)

### Module Coverage Breakdown
| Module | Coverage | Tests | Status |
|--------|----------|-------|--------|
| server/agents/models.py | 100% | 6 | ✅ Complete |
| server/agents/audit_agent.py | 100% | 9 | ✅ Complete |
| server/agents/coach_agent.py | 100% | 12 | ✅ Complete |
| server/agents/__init__.py | 100% | - | ✅ Complete |
| server/agents/voice_agent.py | 79% | 23 | ✅ Complete |
| server/agents/vision_agent.py | 87% | 28 | ✅ Complete |
| main.py | 28% | 5 | ⚠️ Partial (webcam required) |
| webapp/session_manager.py | — | 6 | ✅ Complete (mocked camera) |

### Running Tests
```bash
# Run all tests (testpaths is set to tests in pyproject.toml)
uv run pytest -v

# Run with coverage
uv run pytest --cov=agents --cov=main

# Run only the web frontend tests
uv run pytest tests/test_webapp_session_manager.py -v

# Verify setup (no webcam needed)
uv run python server/verify_setup.py
```

---

## 6. Voice Feedback System

### Features
1. **Workout Start**: "Workout starting. Get ready!"
2. **Rep Milestones**: Encouraging messages every 5 reps
3. **Posture Alerts**: Context-aware warnings (varies by frequency)
4. **Form Corrections**: Prefixed coaching feedback
5. **Session Complete**: Performance-adaptive summary

### Configuration
```bash
# .env file settings
VOICE_ENABLED=true        # Enable/disable voice
VOICE_RATE=150            # Speech rate (WPM)
VOICE_VOLUME=1.0          # Volume (0.0-1.0)
```

### Architecture
- **Engine**: pyttsx3 (offline, cross-platform)
- **Threading**: Background worker prevents blocking
- **Priority System**: Critical alerts interrupt queued speech
- **Fallback**: Gracefully disables if TTS unavailable

---

## 6b. Web Frontend (NEW)

### Features
1. **Start / Pause / Resume / Stop / Quit controls** driven from the browser via `webapp/app.py`.
2. **Live video stream** (`/video_feed`, MJPEG) showing the annotated frame with rep/posture HUD, sized with `object-fit: contain` so the full camera frame is always visible (no cropping).
3. **Live stats panel** polled every ~800ms via `/api/status` (reps, elapsed time, posture faults, knee/spine angles, latest AI form note).
4. **End-of-session report modal** shown after Stop or Quit: total reps, duration, calories, posture faults, AI coach summary, and heuristic "things to improve" tips.
5. **Quit fully terminates the server process** (not just the workout session) via `os.kill(os.getpid(), signal.SIGINT)`, dispatched from a background thread shortly after the response is sent so the client still receives confirmation.

### Running
```bash
uv run python -m webapp.app
# open http://localhost:5000
```

### Architecture
- **Flask routes** (`webapp/app.py`): thin HTTP layer over `WorkoutSessionManager`.
- **`WorkoutSessionManager`** (`webapp/session_manager.py`): thread-safe state machine
  (`idle → running ⇄ paused → stopped/closed`) that owns a background capture thread driving
  `VisionAgent.process_frame()`, dispatches `FormAuditAgent` calls on a `ThreadPoolExecutor`,
  and triggers `VoiceAgent` announcements — mirroring the same agent pipeline as `main.py`.
- **Frontend** (`webapp/templates/index.html`, `webapp/static/{app.js,style.css}`): vanilla
  JS polling + fetch calls, no build step required.

---

## 7. Environment Configuration

### Required Variables
```bash
OPENAI_API_KEY=           # Optional: enables AI audit/coach
USER_WEIGHT_KG=70         # User weight for calorie calculation
FITNESS_GOAL=general fitness  # User fitness goal
WORKOUT_MET=5.0           # MET value for exercise intensity
CAMERA_INDEX=0            # Webcam device index
```

### Voice Variables (Optional)
```bash
VOICE_ENABLED=true        # Enable voice feedback
VOICE_RATE=150            # Speech rate (100-200 WPM)
VOICE_VOLUME=1.0          # Volume level (0.0-1.0)
```

### Web Frontend Variables (Optional, NEW)
```bash
WEB_PORT=5000              # Port for webapp/app.py (Flask dev server)
```

---

## 8. Documentation Files

| File | Purpose |
|------|---------|
| `README.md` | Project overview, setup instructions |
| `agent.md` | Multi-agent architecture specification |
| `plan.md` | This file - implementation roadmap |
| `TESTING.md` | Comprehensive testing guide |
| `VOICE_FEATURES.md` | Voice feedback documentation |
| `VOICE_IMPROVEMENTS.md` | Voice clarity improvements |

---

## 9. Installation & Usage

### Setup
```bash
# Create virtual environment
uv venv

# Activate environment
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows

# Install dependencies
uv pip install -r requirements.txt

# Editable install so `agents` (physically in server/agents/) is importable
# from main.py, webapp/, and the test suite without any path changes
uv pip install -e .

# Configure environment
cp .env.example .env
# Edit .env with your settings
```

### Run
```bash
# Verify setup (no webcam)
uv run python server/verify_setup.py

# Run desktop app with webcam (OpenCV window)
uv run main.py

# Run the web frontend (Start/Pause/Stop/Quit browser control panel)
uv run python -m webapp.app
# then open http://localhost:5000

# Run tests (testpaths is set to tests in pyproject.toml)
uv run pytest -v
```

---

## 10. Development Notes

### MediaPipe Version
- Pinned to `mediapipe==0.10.9` for `solutions` API compatibility
- Versions 1.0+ have breaking API changes

### Voice Quality
- Default rate (150 WPM) optimized for clarity during exercise
- Maximum volume (1.0) ensures audibility while moving
- Contextual messaging provides purpose identification

### Testing Philosophy
- Mock external dependencies (OpenAI, webcam, TTS)
- Verify core logic and integrations
- Main loop requires camera for full testing

### Cost Estimates
- **Vision Agent**: $0.00 (local)
- **Voice Agent**: $0.00 (local)
- **Audit Agent**: ~$0.0025 per snapshot
- **Coach Agent**: ~$0.0001 per session
- **Total per workout**: ~$0.0051 (with 2 audits)

---

## 11. Future Enhancements

- [ ] Support for additional exercises (deadlifts, bench press)
- [ ] Multi-user profiles with progress tracking
- [ ] Historical workout data visualization
- [ ] Voice commands for hands-free control
- [ ] Mobile app integration
- [ ] Real-time form comparison with reference videos
- [ ] Integration with fitness tracking platforms
