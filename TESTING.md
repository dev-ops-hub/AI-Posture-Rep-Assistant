# Test Summary

## Overview

Comprehensive unit tests have been added for all code modules with **76% overall code coverage** and **51 passing tests**.

## Running Tests

### Run all tests:
```bash
uv run pytest tests/
```

### Run tests with verbose output:
```bash
uv run pytest tests/ -v
```

### Run tests with coverage report:
```bash
uv run pytest tests/ --cov=agents --cov=main --cov-report=term-missing
```

### Run specific test file:
```bash
uv run pytest tests/test_vision_agent.py -v
```

### Run specific test:
```bash
uv run pytest tests/test_coach_agent.py::test_fitness_coach_agent_calculate_calories -v
```

## Test Coverage

| Module | Coverage | Notes |
|--------|----------|-------|
| agents/__init__.py | 100% | All exports verified |
| agents/audit_agent.py | 100% | Full coverage with mocked OpenAI |
| agents/coach_agent.py | 100% | Full coverage with mocked OpenAI |
| agents/models.py | 100% | All dataclasses and methods tested |
| agents/vision_agent.py | 81% | Core logic tested, MediaPipe interactions partially covered |
| main.py | 33% | HUD rendering tested, main loop requires webcam |

## Test Structure

```
tests/
├── __init__.py
├── test_models.py           # Data models and serialization
├── test_vision_agent.py     # Computer vision and pose detection
├── test_audit_agent.py      # Posture audit feedback
├── test_coach_agent.py      # Fitness coaching summaries
└── test_main.py             # HUD rendering functions
```

## Verification Script

Run the verification script to test all components without requiring a webcam:

```bash
uv run python verify_setup.py
```

This checks:
- ✓ All imports work
- ✓ VisionAgent initialization and cleanup
- ✓ FormAuditAgent initialization
- ✓ FitnessCoachAgent calorie calculations
- ✓ Summary generation
- ✓ Angle calculations (knee and spine)

## Key Test Features

### Mocked Dependencies
- **OpenAI API**: Tests work without API key, verify both API and fallback modes
- **MediaPipe**: Tests work with and without MediaPipe installed
- **Webcam**: Vision tests use synthetic data instead of camera

### Tested Scenarios

#### VisionAgent
- ✓ Angle calculations (knee, spine)
- ✓ Rep counting state machine (standing → descending → bottom → ascending)
- ✓ Posture violation detection
- ✓ Violation hold duration tracking
- ✓ Max audits per session limit

#### FormAuditAgent
- ✓ Initialization with/without API key
- ✓ Fallback feedback generation
- ✓ OpenAI API integration (mocked)
- ✓ Different posture fault types

#### FitnessCoachAgent
- ✓ Calorie calculations
- ✓ Session summary generation
- ✓ Fallback mode
- ✓ OpenAI API integration (mocked)

#### Models
- ✓ Dataclass initialization
- ✓ Rounding in serialization
- ✓ Default values
- ✓ to_dict() methods

## Continuous Integration Ready

Tests are configured for CI/CD with:
- `pyproject.toml` for pytest configuration
- Coverage reporting (HTML and terminal)
- No external dependencies required (camera, API keys)
- Fast execution (~2.5 seconds)

## Requirements

Test dependencies (already in requirements.txt):
- pytest>=8.0
- pytest-cov>=4.1
- pytest-mock>=3.12

## Notes

- Tests use mocking for external dependencies (OpenAI, webcam)
- MediaPipe version pinned to 0.10.9 for compatibility with solutions API
- All tests pass on macOS with Apple Silicon
- Coverage HTML report available in `htmlcov/index.html`
