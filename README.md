# AI-Posture-Rep-Assistant

Python multi-agent workout tracker for squat rep counting, posture fault detection, and AI-generated coaching.

## What It Does

- Tracks squat reps locally with OpenCV, MediaPipe, and NumPy.
- Detects sustained forward-lean posture faults from pose landmarks.
- Sends posture snapshots to an OpenAI vision agent when `OPENAI_API_KEY` is configured.
- Generates a post-workout coaching summary at session end.

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
│   └── coach_agent.py
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

Then edit `.env` as needed. Supported variables:

```dotenv
OPENAI_API_KEY=
USER_WEIGHT_KG=70
FITNESS_GOAL=general fitness
WORKOUT_MET=5.0
CAMERA_INDEX=0
```

## Run

```bash
uv run main.py
```

Press `q` to end the workout session and print the final coaching summary.

## Testing

This project includes comprehensive unit tests with 76% code coverage.

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
- The webcam flow requires local camera access and the packages in `requirements.txt`.

