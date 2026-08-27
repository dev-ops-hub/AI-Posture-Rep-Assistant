# AI-Posture-Rep-Assistant

Python multi-agent workout tracker for squat rep counting, posture fault detection, and AI-generated coaching.

## Requirements

- **Python 3.11** (see `requires-python` in [pyproject.toml](pyproject.toml)). CPython 3.11.x is recommended, matching the pinned `mediapipe==0.10.9` dependency.
- A webcam for live tracking (not required to run the test suite).
- [`uv`](https://docs.astral.sh/uv/) for dependency management (or substitute `pip`/`venv` commands manually).

## Quick Start

<details open>
<summary><strong>macOS / Linux</strong></summary>

```bash
# 1. Install dependencies
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .              # makes `agents` (in server/agents) importable everywhere

# 2. Configure environment variables
cp .env.example .env
# edit .env with your webcam index, weight, goal, etc.

# 3. Launch the web app
uv run python -m webapp.app
```

</details>

<details>
<summary><strong>Windows (PowerShell)</strong></summary>

```powershell
# 1. Install dependencies
uv venv
.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
uv pip install -e .              # makes `agents` (in server/agents) importable everywhere

# 2. Configure environment variables
copy .env.example .env
# edit .env with your webcam index, weight, goal, etc.

# 3. Launch the web app
uv run python -m webapp.app
```

> If `Activate.ps1` is blocked, run PowerShell as Administrator once and execute
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then re-open your terminal.
> Using Command Prompt instead of PowerShell? Activate with `.venv\Scripts\activate.bat`
> and copy the env file with `copy .env.example .env` (same as above).

</details>

Then open **[http://localhost:5000](http://localhost:5000)** in your browser and press **Start**.
See [Run](#run) below for the desktop (OpenCV window) alternative and full details on each option.

## What It Does

- Tracks squat reps locally with OpenCV, MediaPipe, and NumPy.
- Detects sustained forward-lean posture faults from pose landmarks.
- **Provides real-time voice feedback** during your workout.
- Sends posture snapshots to an OpenAI vision agent when `OPENAI_API_KEY` is configured.
- Generates a post-workout coaching summary at session end.
- **Browser-based control panel** (Flask) with Start / Pause / Stop / Quit controls, a live
  annotated video stream, live stats, and an end-of-session report with improvement tips.

## Multi-Agent Architecture

The system uses four specialized agents working together:

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

See [agent.md](agent.md) for detailed agent architecture specifications.

## Technology Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Programming Language** | Python 3.11+ | Unified runtime for CV, logic, and LLM orchestration |
| **Video Pipeline** | `opencv-python` | Webcam input, canvas drawing, frame output |
| **Pose Detection** | `mediapipe==0.10.9` | 33 3D body keypoint tracking at 30+ FPS |
| **Math & Geometry** | `numpy` | Vector operations for joint angle calculation |
| **AI Coaching** | `openai` SDK | Multimodal form auditing and session summaries |
| **Voice Feedback** | `pyttsx3` | Local offline text-to-speech for audio coaching |
| **Web Frontend** | `Flask` | Browser control panel, REST API, and MJPEG video streaming |
| **Testing** | `pytest`, `pytest-cov` | Unit testing with 74%+ code coverage |

## Project Layout

```text
.
├── main.py                 # Desktop OpenCV entry point (stays in repo root)
├── requirements.txt
├── pyproject.toml           # Packaging + pytest config (makes `agents` importable everywhere)
├── server/                  # Backend agent code + verification script
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   ├── vision_agent.py
│   │   ├── audit_agent.py
│   │   ├── coach_agent.py
│   │   └── voice_agent.py
│   └── verify_setup.py
├── tests/                   # Test suite (top-level, alongside main.py/webapp)
├── webapp/                  # Browser control panel (unchanged location)
│   ├── app.py               # Flask server & REST API
│   ├── session_manager.py   # Start/pause/resume/stop/quit state machine
│   ├── templates/index.html
│   └── static/{style.css, app.js}
├── agent.md
├── plan.md
└── README.md
```

> **Note:** `server/agents` is installed in editable mode (`uv pip install -e .`) so that
> `from agents import ...` continues to work unchanged from `main.py`, `webapp/`, and the test
> suite — regardless of where the `agents` package physically lives.

## Setup

Requires Python 3.11 — see [Requirements](#requirements).

**macOS / Linux:**
```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Editable install so `agents` (under server/agents) is importable from
# main.py, webapp/, and the test suite without any path juggling
uv pip install -e .
```

**Windows (PowerShell):**
```powershell
uv venv
.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt

# Editable install so `agents` (under server/agents) is importable from
# main.py, webapp/, and the test suite without any path juggling
uv pip install -e .
```

Create a local `.env` file in the project root:

```bash
# macOS / Linux
cp .env.example .env
```

```powershell
# Windows
copy .env.example .env
```

Then edit `.env` as needed.

### Environment Variables

**Required Variables:**
```dotenv
USER_WEIGHT_KG=70              # User weight for calorie calculation
FITNESS_GOAL=general fitness   # User fitness goal
WORKOUT_MET=5.0                # MET value for exercise intensity
CAMERA_INDEX=0                 # Webcam device index (usually 0)
```

**Optional Variables:**
```dotenv
OPENAI_API_KEY=                # Enable AI audit/coach features
VOICE_ENABLED=true             # Enable voice feedback
VOICE_RATE=150                 # Speech rate in words per minute (100-200)
VOICE_VOLUME=1.0               # Volume level (0.0 to 1.0)
WEB_PORT=5000                  # Port for the web frontend (webapp/app.py)
```

### Voice Feedback

The application provides **clear, contextual** real-time audio coaching during your workout:

- **Workout start**: "Workout starting. Get ready!"
- **Rep milestones**: Encouraging messages every 5 reps
  - "Good start! 5 reps completed"
  - "Great progress! 10 reps done"
- **Posture alerts**: Context-aware warnings when form issues detected
  - "Form check. Keep your chest up and core braced"
- **Form corrections**: Prefixed coaching feedback
  - "Coaching tip: Your torso is folding forward..."
- **Session complete**: Performance-adaptive summary
  - Perfect form, minor issues, or areas to focus on

#### Customization

Adjust voice settings in your `.env` file:

```bash
# Slower/clearer speech (100-150 recommended)
VOICE_RATE=120

# Volume level (0.0 to 1.0)
VOICE_VOLUME=1.0

# Disable voice completely
VOICE_ENABLED=false
```

## Run

Make sure you've completed [Setup](#setup) first (dependencies installed and `.env` created).
The commands below (`uv run ...`) are identical on Windows, macOS, and Linux — just make sure
the virtual environment created in [Setup](#setup) is activated first.

### Option 1: Web Frontend (recommended)

A browser-based control panel with **Start**, **Pause**, **Stop**, and **Quit** buttons, a live
annotated video feed, live stats, and an end-of-session report.

**Step 1 — Start the server:**
```bash
uv run python -m webapp.app
```

**Step 2 — Open the app:** navigate to [http://localhost:5000](http://localhost:5000) in your
browser (the port can be changed with the `WEB_PORT` environment variable).

**Step 3 — Control your workout:**

| Button | Behavior |
| :--- | :--- |
| ▶ Start | Opens the webcam and begins rep/posture tracking |
| ⏸ Pause / ▶ Resume | Freezes tracking and the elapsed timer without ending the session (toggles) |
| ⏹ Stop | Ends the current workout, releases the camera, and shows the workout report |
| ⏻ Quit | Stops any active session, releases camera/voice resources, shows the report (if a session was active), and **shuts down the Flask server process** |

Optionally adjust camera index, body weight, fitness goal, and MET value from the
"Session settings" panel before pressing Start.

**Step 4 — Review your report:** after Stop/Quit, a report is shown with:
- Summary: total reps, duration, calories burned, posture faults
- **Things to improve**: heuristic tips (pace, posture faults, low rep count) plus the most
  recent AI form-audit notes
- AI coach summary (falls back to a deterministic summary if `OPENAI_API_KEY` is not set)

The live video feed automatically fits the full camera frame (letterboxed if the aspect ratio
doesn't match the panel) instead of cropping any part of the picture.

Note: pressing **Quit** terminates the Flask process (equivalent to `Ctrl+C` in the terminal).
Use **Stop** instead if you want to end a workout but keep the server running for another session.

### Option 2: Desktop (OpenCV window)

```bash
uv run main.py
```

Press `q` to end the workout session and print the final coaching summary in the terminal.

## Testing

This project includes comprehensive unit tests with **89 passing tests** and **77% overall code coverage** (agents + main.py; `webapp/` is covered by its own dedicated test file).

### Coverage by Module

| Module | Coverage | Tests |
|--------|----------|-------|
| `server/agents/models.py` | 100% | 6 |
| `server/agents/audit_agent.py` | 100% | 9 |
| `server/agents/coach_agent.py` | 100% | 12 |
| `server/agents/voice_agent.py` | 79% | 23 |
| `server/agents/vision_agent.py` | 87% | 28 |
| `main.py` | 28% | 5 |
| `webapp/session_manager.py` | — | 6 |

*Note: `main.py` requires webcam access for full testing coverage. `webapp/session_manager.py` tests mock the camera so they run without hardware. Test files live in `tests/`; `pyproject.toml` configures pytest's `testpaths` accordingly.*

### Quick Verification

Verify all components work without requiring a webcam:

```bash
uv run python server/verify_setup.py
```

### Run Tests

```bash
# Run all tests (testpaths is set to tests in pyproject.toml)
uv run pytest

# Run with verbose output
uv run pytest -v

# Run with coverage report
uv run pytest --cov=agents --cov=main
```

See [TESTING.md](TESTING.md) for detailed testing documentation.

## Notes

- The OpenAI-powered agents fall back to local deterministic feedback if `OPENAI_API_KEY` is not set.
- Voice feedback uses the system's text-to-speech engine (offline, no API needed).
- The webcam flow requires local camera access and the packages in `requirements.txt`.
- The web frontend (`webapp/app.py`) runs Flask's built-in development server, intended for
  local single-user use only; it is not hardened for production deployment.
- **Windows:** `pyttsx3` uses the built-in SAPI5 voices, so voice feedback works out of the box
  with no extra install. If the webcam doesn't open or is slow to start, try a different
  `CAMERA_INDEX` in `.env` — OpenCV on Windows enumerates capture devices via DirectShow, which
  can number them differently than on macOS/Linux.

### Rep-Counting Accuracy Tuning

Rep counting tracks the left knee angle (hip-knee-ankle) and left spine lean (shoulder-hip)
each frame; it doesn't average both legs or smooth readings across frames. If reps are still
under/over counted for your setup, these `VisionAgent` constructor parameters can be tuned:

| Parameter | Default | Effect |
| :--- | :--- | :--- |
| `standing_angle_threshold_deg` | 160.0 | Knee angle considered "fully standing" (completes a rep) |
| `bottom_angle_threshold_deg` | 90.0 | Knee angle considered "at the bottom" (lower = deeper squat required) |
| `min_detection_confidence` | 0.5 | MediaPipe's minimum confidence to detect a person in a frame |
| `min_tracking_confidence` | 0.5 | MediaPipe's minimum confidence to keep tracking landmarks between frames |

For best results, stand far enough back that your full body (hips to ankles) is visible and
side-on (or at a slight angle) to the camera, with your left side facing it.

### API Cost Estimates

When using OpenAI API:
- **Vision Agent**: Free (runs locally)
- **Voice Agent**: Free (runs locally)
- **Audit Agent**: ~$0.0025 per posture snapshot (GPT-4o vision)
- **Coach Agent**: ~$0.0001 per session summary (GPT-4o-mini)
- **Total per workout**: ~$0.0051 (assuming 2 audit snapshots)

## Future Enhancements

- Support for additional exercises (deadlifts, bench press)
- Multi-user profiles with progress tracking
- Historical workout data visualization
- Voice commands for hands-free control
- Mobile app integration
- Real-time form comparison with reference videos
- Integration with fitness tracking platforms

