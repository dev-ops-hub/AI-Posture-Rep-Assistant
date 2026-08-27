# Multi-Agent Architecture & Specification (`agent.md`)

## 1. System Architecture Overview (Python Stack)
This application uses a hybrid edge-cloud **Python Multi-Agent Architecture** with **real-time voice feedback**. Computer vision, landmark tracking, and vector geometry run locally on Python (`mediapipe` + `numpy` + `opencv`), maintaining zero-latency video processing. Text-to-speech audio coaching runs locally via `pyttsx3`, while generative AI agents leverage the `openai` Python SDK.

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
               ├─────────────────────┬───────────────────┐
               ▼                     ▼                   ▼
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│ Agent 2: Form Audit │  │ Agent 3: Fitness    │  │ Agent 4: Voice      │
│ (Cloud - GPT-4o)    │  │ Coach (GPT-4o-mini) │  │ (Local - pyttsx3)   │
│ - Encodes JPEG      │  │ - Aggregates MET    │  │ - Real-time TTS     │
│ - Visual diagnosis  │  │ - Session summary   │  │ - Contextual audio  │
└──────────┬──────────┘  └──────────┬──────────┘  └──────────┬──────────┘
           │                        │                         │
           └────────────────────────┴─────────────────────────┘
                                    ▼
                [ OpenCV HUD + Audio Coaching Output ]
```

---

## 2. Agent Definitions & Specifications

### Agent 1: Edge Vision & Tracking Agent
* **Runtime Environment:** Local Python Interpreter (`vision_agent.py`).
* **Primary Libraries:** `mediapipe==0.10.14`, `numpy`, `opencv-python`.
* **Primary Responsibilities:**
  1. Capture video feed from camera at 30 FPS with 0 ms network delay.
  2. Extract 33 spatial keypoint coordinates via MediaPipe Pose.
  3. Execute squat state machine transitions (`STANDING` -> `DESCENDING` -> `BOTTOM` -> `ASCENDING` -> rep complete).
  4. Track slouching angle (`spine_angle_deg > 15°` for more than `2.0s`).
  5. Emit posture violation events and telemetry to other agents.
* **Execution Trigger:** Continuous frame loop (`while cap.isOpened()`).
* **Cost:** **$0.00** (Runs locally on CPU).
* **Test Coverage:** 87% (28 tests covering angle math, rep counting, posture detection, and the rep-counting robustness improvements below).

#### Key Methods
- `process_frame()`: Main processing pipeline returning annotated frame, metrics, and violation events
- `_estimate_knee_angle()`: Combines both legs' knee angles (visibility-gated, see below) into one robust reading
- `_smooth_knee_angle()`: Exponential-moving-average filter applied before the reading reaches the rep FSM
- `_leg_knee_angle()`: Computes one leg's knee angle plus its minimum landmark visibility
- `_angle()`: Calculate the angle between three 2D/3D points via the vector dot-product formula
- `_spine_angle()`: Calculate spine deviation from vertical
- `_update_rep_state()`: Finite state machine for rep counting (now with a minimum inter-rep cooldown)
- `_update_posture_state()`: Track sustained posture violations

#### Rep-Counting Accuracy Improvements (NEW)
Real-world testing showed the original single-frame, left-leg-only, unsmoothed knee angle could
mis-count reps whenever MediaPipe's per-frame landmark jitter or a partially occluded leg pushed
the raw angle across the 90°/160° thresholds without an actual squat occurring. The following
changes were made to `vision_agent.py`, all covered by new unit tests and verified not to change
the previously-tested FSM transition behavior:

1. **Dual-leg averaging with visibility gating** (`_estimate_knee_angle` / `_leg_knee_angle`) — the
   knee angle is now computed from both legs (using MediaPipe's per-landmark `visibility` score,
   thresholded by `min_landmark_visibility`, default `0.5`) and averaged when both are reliable,
   falling back to whichever single leg is visible, or holding the last known smoothed angle if
   neither leg is trustworthy that frame (e.g. stepped out of frame, occluded by an arm).
2. **3D angle calculation** — `_angle()` now uses the vector dot-product/arccos formula on
   `(x, y, z)` landmark coordinates (MediaPipe's estimated depth) instead of a 2D-only
   `atan2`-based formula, making the reading less sensitive to the user's exact orientation
   relative to the camera.
3. **Exponential moving-average smoothing** (`_smooth_knee_angle`, weight `knee_angle_smoothing`,
   default `0.4`) — filters per-frame landmark jitter before it ever reaches the rep state
   machine, so a single noisy/occluded-landmark frame can no longer flip the FSM into `BOTTOM` or
   `STANDING` on its own.
4. **Minimum inter-rep cooldown** (`min_rep_interval_sec`, default `0.3s`) — a defense-in-depth
   guard inside `_update_rep_state()` that prevents two rep counts firing in rapid succession if
   the smoothed angle still oscillates right at the standing threshold.
5. **Configurable thresholds** — `standing_angle_threshold_deg` (default `160°`) and
   `bottom_angle_threshold_deg` (default `90°`) are now constructor parameters instead of magic
   numbers, so depth requirements can be tuned per user/exercise without editing the code.

---


### Agent 2: Form Audit Agent
* **Runtime Environment:** Cloud API via Python SDK (`audit_agent.py`).
* **Primary Libraries:** `openai`, `base64`, `cv2`.
* **Primary Responsibilities:**
  1. Convert current OpenCV NumPy frame to JPEG bytes and encode to Base64.
  2. Send snapshot to `gpt-4o` Vision model with biomechanical diagnostic prompt.
  3. Return a concise, 1-to-2 sentence corrective note.
  4. Fallback to deterministic local feedback if API key not configured.
* **Execution Trigger:** Event-driven (Triggered when `POSTURE_VIOLATION` lasts >2.0s; max 2 calls per session).
* **Cost Estimate:** **~$0.0025 per snapshot**.
* **Test Coverage:** 100% (9 tests covering initialization, API calls, fallback logic).

#### Fallback Logic
- Forward lean (>20°): "Your torso is folding forward. Brace your core and keep your chest stacked over your hips on the descent."
- Moderate drift (≤20°): "Your squat is drifting out of position. Slow the rep slightly and keep your spine neutral as you stand up."

---

### Agent 3: Fitness Coach Agent
* **Runtime Environment:** Cloud API via Python SDK (`coach_agent.py`).
* **Primary Libraries:** `openai`, `json`.
* **Primary Responsibilities:**
  1. Aggregate total reps, exercise duration, posture fault counts, and computed MET calories:
     ```python
     calories_burned = MET * weight_kg * (duration_seconds / 3600)
     ```
  2. Synthesize session metrics with Form Audit Agent notes.
  3. Generate a personalized post-workout report string.
  4. Fallback to deterministic local summary if API key not configured.
* **Execution Trigger:** On-demand (Triggered when user presses 'q' to finish workout).
* **Cost Estimate:** **~$0.0001 per workout session**.
* **Test Coverage:** 100% (12 tests covering calorie calculations, summaries, API integration).

#### Summary Format
```
You completed {total_reps} squat reps in {duration} seconds and burned about {calories:.2f} kcal.
Posture faults recorded: {fault_count}.
Focus next on this cue: {latest_diagnostic}
```

---

### Agent 4: Voice Agent (NEW)
* **Runtime Environment:** Local Python Interpreter (`voice_agent.py`).
* **Primary Libraries:** `pyttsx3`, `queue`, `threading`.
* **Primary Responsibilities:**
  1. Provide real-time audio coaching during workout sessions.
  2. Announce rep milestones with encouraging messages every 5 reps.
  3. Deliver contextual posture alerts with priority speech.
  4. Speak form correction notes with coaching prefixes.
  5. Generate performance-adaptive session completion summaries.
* **Execution Trigger:** Event-driven (triggered by vision agent events and user actions).
* **Cost:** **$0.00** (Runs locally with offline TTS engine).
* **Test Coverage:** 79% (23 tests covering initialization, speech queue, announcements).

#### Voice Features

**1. Workout Start**
```python
"Workout starting. Get ready!"
```

**2. Rep Milestones** (Every 5 reps)
```python
rep == 5:  "Good start! 5 reps completed"
rep == 10: "Great progress! 10 reps done"
rep % 10 == 0: "Excellent! {rep} reps completed"
other: "Keep going! {rep} reps"
```

**3. Posture Alerts** (Context-aware)
```python
alert_count == 0: "Form check. Keep your chest up and core braced"
alert_count == 1: "Posture alert. Chest up, maintain neutral spine"
alert_count > 1:  "Watch your form. Stay upright"
```

**4. Form Corrections** (Prefixed)
```python
first_note:  "Coaching tip: {feedback}"
later_notes: "Remember: {feedback}"
```

**5. Session Complete** (Adaptive)
```python
no_reps:      "Session ended. No reps detected. Try getting in frame..."
perfect_form: "Perfect form! Workout complete. {reps} reps in {time}..."
minor_faults: "Great job! {reps} reps completed. Minor form notes..."
many_faults:  "Workout done. {reps} reps. Focus on form corrections..."
```

#### Voice Configuration
```python
# Environment variables
VOICE_ENABLED=true    # Enable/disable voice
VOICE_RATE=150        # Speech rate in WPM (default: 150)
VOICE_VOLUME=1.0      # Volume 0.0-1.0 (default: 1.0)
```

#### Architecture Details
- **Threading**: Background worker thread prevents blocking main loop
- **Priority Queue**: Critical alerts can interrupt queued speech
- **Voice Selection**: Auto-selects English voice when available
- **Graceful Degradation**: Disables cleanly if pyttsx3 unavailable

---

## 3. Agent Communication Protocols (Python Data Classes)

### Schema A: Agent 1 -> Agent 2 (Posture Violation Event)
```python
@dataclass
class PostureViolationEvent:
    sender: str = "Agent_1_Vision"
    recipient: str = "Agent_2_FormAudit"
    event_type: str = "POSTURE_VIOLATION"
    exercise: str = "Squat"
    metrics: PostureViolationMetrics
    image_base64: str

# Example payload
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

### Schema B: Agent 1 -> Agent 4 (Voice Triggers)
```python
# Rep milestone
voice_agent.announce_rep(rep_count: int)

# Posture alert
voice_agent.announce_posture_alert(alert_count: int)

# Form note
voice_agent.announce_form_note(note: str, is_first: bool)

# Session complete
voice_agent.announce_session_complete(
    rep_count: int,
    duration_seconds: float,
    fault_count: int
)
```

### Schema C: Agents 1 & 2 -> Agent 3 (Session End Payload)
```python
@dataclass
class SessionSummaryPayload:
    sender: str = "Orchestrator_Main"
    recipient: str = "Agent_3_Coach"
    user_profile: UserProfile
    session_summary: SessionSummary
    form_audit_diagnostics: list[str]

# Example payload
{
    "sender": "Orchestrator_Main",
    "recipient": "Agent_3_Coach",
    "user_profile": {
        "weight_kg": 70,
        "goal": "general fitness"
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
        "Your torso is folding forward. Brace your core..."
    ]
}
```

---

## 4. Testing Infrastructure

### Test Suite Overview
- **Total Tests:** 89
- **Overall Coverage:** 77% (`agents/` package under `server/agents/`, plus `main.py`; `webapp/` has its own dedicated test file)
- **Test Framework:** pytest with pytest-cov and pytest-mock

### Per-Agent Test Coverage

| Agent | Tests | Coverage | Key Test Areas |
|-------|-------|----------|----------------|
| Models | 6 | 100% | Data classes, serialization, rounding |
| Vision Agent | 28 | 87% | Angle math, rep FSM, posture detection, dual-leg averaging, visibility gating, EMA smoothing, rep-interval debounce |
| Audit Agent | 9 | 100% | API integration, fallback logic |
| Coach Agent | 12 | 100% | Calorie calc, summaries, API integration |
| Voice Agent | 23 | 79% | TTS control, speech queue, announcements |
| Main HUD | 5 | 28% | HUD rendering (limited by webcam dependency) |
| Web Session Manager | 6 | — | Start/pause/resume/stop/quit lifecycle, report generation (mocked camera) |

### Running Tests
```bash
# All tests with coverage (testpaths is set to tests in pyproject.toml)
uv run pytest -v --cov=agents --cov=main

# Specific agent tests
uv run pytest tests/test_voice_agent.py -v

# Quick verification (no webcam)
uv run python server/verify_setup.py
```

### Test Patterns
- **Mocking**: OpenAI API, MediaPipe, pyttsx3, webcam
- **Fixtures**: Reusable test data and mocked objects
- **Coverage**: HTML reports in `htmlcov/`
- **CI-Ready**: No external dependencies required

---

## 5. Orchestrator Implementation (`main.py`)

### Main Loop Architecture
```python
# Initialize all agents
vision_agent = VisionAgent()
audit_agent = FormAuditAgent()
coach_agent = FitnessCoachAgent()
voice_agent = VoiceAgent()  # NEW

# Announce workout start
voice_agent.announce_workout_start()  # NEW

# Main camera loop
while cap.isOpened():
    # Process frame
    annotated, metrics, violation_event = vision_agent.process_frame(frame)
    
    # Announce rep milestones
    if metrics.reps > last_rep_count:
        voice_agent.announce_rep(metrics.reps)  # NEW
    
    # Handle posture violations
    if violation_event:
        voice_agent.announce_posture_alert(alert_count)  # NEW
        audit_future = executor.submit(audit_agent.audit_posture, violation_event)
    
    # Process audit results
    if audit_future.done():
        note = audit_future.result()
        voice_agent.announce_form_note(note, is_first)  # NEW
    
    # Render HUD and display
    draw_hud(annotated, fps, metrics, latest_note)
    cv2.imshow("AI Posture Rep Assistant", annotated)

# Session complete
voice_agent.announce_session_complete(reps, duration, faults)  # NEW
summary = coach_agent.build_summary(payload)
print(summary)
```

### Threading Model
- **Main Thread**: Camera capture, frame processing, UI rendering
- **Background Thread 1**: Audit agent API calls (ThreadPoolExecutor)
- **Background Thread 2**: Voice synthesis (pyttsx3 worker)

---

## 5b. Web Orchestrator Implementation (`webapp/`) (NEW)

In addition to the desktop OpenCV window (`main.py`), the same four agents can be driven from a
browser-based control panel built with **Flask**.

```
 [ Browser UI: Start / Pause / Stop / Quit ]
                  │  fetch() REST calls
                  ▼
 ┌────────────────────────────────────────────────────────┐
 │ webapp/app.py (Flask routes)                           │
 │  - /api/start, /api/pause, /api/stop, /api/quit        │
 │  - /api/status (polled every ~800ms)                   │
 │  - /video_feed (MJPEG stream of annotated frames)       │
 └──────────────────────┬──────────────────────────────────┘
                         ▼
 ┌────────────────────────────────────────────────────────┐
 │ webapp/session_manager.py: WorkoutSessionManager        │
 │  - State machine: idle → running ⇄ paused → stopped/closed │
 │  - Owns a background capture thread that calls          │
 │    VisionAgent.process_frame() every loop iteration      │
 │  - Dispatches FormAuditAgent calls on a ThreadPoolExecutor│
 │  - Drives VoiceAgent announcements                       │
 │  - Builds the end-of-session report via FitnessCoachAgent │
 └────────────────────────────────────────────────────────┘
```

### Session States
| State | Meaning |
| :--- | :--- |
| `idle` | No session yet, or after Quit |
| `running` | Camera loop active, reps/posture tracked |
| `paused` | Capture loop keeps the thread alive but skips frame processing; elapsed timer is frozen |
| `stopped` | Session ended via Stop; camera/voice released; report available |
| `closed` | Session ended via Quit; camera/voice released; Flask process is shutting down |

### Key Methods (`WorkoutSessionManager`)
- `start(config)`: Opens the webcam, instantiates fresh `VisionAgent`/`VoiceAgent` instances, and spawns the capture thread.
- `pause()` / `resume()` / `toggle_pause()`: Flip between `running` and `paused` without releasing the camera.
- `stop()`: Ends the capture loop, releases the camera/voice engine, and returns the workout report (reps, duration, calories, posture faults, AI coach summary, improvement tips).
- `quit()`: Calls `stop()` if a session is active, then fully releases the executor; `webapp/app.py` follows this with `os.kill(os.getpid(), signal.SIGINT)` on a short delay so the **Flask process itself terminates**, not just the workout session.
- `get_status()`: Thread-safe snapshot used by the `/api/status` polling endpoint.
- `get_frame()`: Returns the latest annotated JPEG frame for the MJPEG stream.

### Report Contents (`_build_report`)
- `total_reps`, `duration_seconds` / `duration_formatted`, `calories`, `total_posture_faults`
- `coach_summary`: generated by `FitnessCoachAgent.build_summary()` (falls back locally without `OPENAI_API_KEY`)
- `improvement_tips`: heuristic list combining posture-fault counts, rep pacing (reps/minute), low rep counts, and the most recent AI form-audit notes
- `form_audit_diagnostics`: raw list of audit notes captured during the session

### Frontend Behavior Notes
- The video `<img>` uses `object-fit: contain` so the **entire camera frame is always visible**
  (letterboxed if the aspect ratio doesn't match the panel), rather than cropping the picture.
- Pressing **Quit** in the browser calls `/api/quit`, which stops the session, shows the report if
  one was active, and shuts down the Flask server; the video feed and polling are stopped
  client-side once the server is confirmed gone.
- **Test Coverage:** `tests/test_webapp_session_manager.py` (6 tests) covers the full
  start/pause/resume/stop/quit lifecycle and report generation with a mocked camera.

---

## 6. Environment Configuration

### Required Variables
```bash
OPENAI_API_KEY=           # Optional: enables AI audit/coach
USER_WEIGHT_KG=70         # User weight for calorie calculation
FITNESS_GOAL=general fitness  # User fitness goal
WORKOUT_MET=5.0           # MET value for exercise intensity
CAMERA_INDEX=0            # Webcam device index
```

### Voice Variables (NEW)
```bash
VOICE_ENABLED=true        # Enable voice feedback
VOICE_RATE=150            # Speech rate (100-200 WPM)
VOICE_VOLUME=1.0          # Volume level (0.0-1.0)
```

### Web Frontend Variables (NEW)
```bash
WEB_PORT=5000             # Port for webapp/app.py (Flask dev server)
```

---

## 7. Performance Characteristics

### Agent Performance Profile

| Agent | Latency | Throughput | Cost | Resource |
|-------|---------|------------|------|----------|
| Vision Agent | <33ms | 30 FPS | $0 | CPU/GPU |
| Audit Agent | ~2-3s | Event-driven | $0.0025 | Network |
| Coach Agent | ~1-2s | End of session | $0.0001 | Network |
| Voice Agent | <50ms | Background | $0 | CPU |

### System Requirements
- **CPU**: Modern multi-core processor (M1/M2 or equivalent)
- **RAM**: 4GB minimum, 8GB recommended
- **Camera**: Webcam with 640x480+ resolution
- **Python**: 3.11+
- **OS**: macOS, Linux, or Windows

---

## 8. Error Handling & Fallbacks

### Vision Agent
- MediaPipe not installed → Display error message, disable pose detection
- No camera access → Exit with helpful error message

### Audit Agent
- No API key → Use deterministic local feedback
- API error → Use fallback feedback based on angle thresholds

### Coach Agent
- No API key → Generate structured local summary
- API error → Use template-based summary

### Voice Agent (NEW)
- pyttsx3 not installed → Disable voice, continue silently
- TTS engine error → Log error, continue without voice
- Invalid settings → Use default values

---

## 9. Cost Analysis (Per Workout Session)

### Typical 5-Minute Workout
- **Vision Agent**: $0.00 (local)
- **Voice Agent**: $0.00 (local)
- **Audit Agent**: $0.005 (2 violations @ $0.0025 each)
- **Coach Agent**: $0.0001 (1 summary)
- **Total Cost**: ~$0.0051 per session

### Monthly Cost (20 workouts)
- **20 sessions × $0.0051 = ~$0.10/month**

---

## 10. Documentation References

| Document | Purpose |
|----------|---------|
| `README.md` | Quick start guide |
| `agent.md` | This file - agent specs |
| `plan.md` | Implementation roadmap |
| `TESTING.md` | Test documentation |
| `VOICE_FEATURES.md` | Voice system guide |
| `VOICE_IMPROVEMENTS.md` | Voice clarity details |

---

## 11. Runtime Notes

### Installation
```bash
# Create environment
uv venv

# Install dependencies
uv pip install -r requirements.txt

# Editable install so `agents` (server/agents) is importable everywhere
uv pip install -e .

# Configure
cp .env.example .env
# Edit .env with your settings
```

### Execution
```bash
# Verify setup (no webcam)
uv run python server/verify_setup.py

# Run with webcam (desktop OpenCV window)
uv run main.py

# Run the web frontend (Start/Pause/Stop/Quit control panel)
uv run python -m webapp.app
# then open http://localhost:5000

# Run tests (testpaths is set to tests in pyproject.toml)
uv run pytest -v
```

### Platform-Specific Notes

**macOS:**
- Native TTS via NSSpeechSynthesizer (high quality)
- Metal-accelerated MediaPipe on M1/M2

**Linux:**
- Install espeak for TTS: `sudo apt-get install espeak`
- May need camera permissions configuration

**Windows:**
- TTS via SAPI5
- Ensure camera privacy settings allow access
