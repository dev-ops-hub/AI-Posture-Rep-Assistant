# AI-Posture-Rep-Assistant

Python multi-agent workout tracker for squat rep counting, posture fault detection, and AI-generated coaching.

## What It Does

- Tracks squat reps locally with OpenCV, MediaPipe, and NumPy.
- Detects sustained forward-lean posture faults from pose landmarks.
- **Provides real-time voice feedback** during your workout.
- Sends posture snapshots to an OpenAI vision agent when `OPENAI_API_KEY` is configured.
- Generates a post-workout coaching summary at session end.

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
| **Testing** | `pytest`, `pytest-cov` | Unit testing with 76% code coverage |

## Project Layout

```text
.
├── main.py
├── requirements.txt
├── agents/
│   ├── __init__.py
│   ├── models.py
│   ├── vision_agent.py
│   ├── audit_agent.py
│   ├── coach_agent.py
│   └── voice_agent.py
├── agent.md
├── plan.md
└── README.md
```

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Create a local `.env` file in the project root:

```bash
cp .env.example .env
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

See [VOICE_IMPROVEMENTS.md](VOICE_IMPROVEMENTS.md) for detailed customization options.

## Run

```bash
uv run main.py
```

Press `q` to end the workout session and print the final coaching summary.

## Testing

This project includes comprehensive unit tests with **74 passing tests** and **74% overall code coverage**.

### Coverage by Module

| Module | Coverage | Tests |
|--------|----------|-------|
| `agents/models.py` | 100% | 6 |
| `agents/audit_agent.py` | 100% | 9 |
| `agents/coach_agent.py` | 100% | 12 |
| `agents/voice_agent.py` | 79% | 23 |
| `agents/vision_agent.py` | 81% | 19 |
| `main.py` | 28% | 5 |

*Note: `main.py` requires webcam access for full testing coverage.*

### Quick Verification

Verify all components work without requiring a webcam:

```bash
uv run python verify_setup.py
```

### Run Tests

```bash
# Run all tests
uv run pytest tests/

# Run with verbose output
uv run pytest tests/ -v

# Run with coverage report
uv run pytest tests/ --cov=agents --cov=main
```

See [TESTING.md](TESTING.md) for detailed testing documentation.

## Notes

- The OpenAI-powered agents fall back to local deterministic feedback if `OPENAI_API_KEY` is not set.
- Voice feedback uses the system's text-to-speech engine (offline, no API needed).
- The webcam flow requires local camera access and the packages in `requirements.txt`.

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

