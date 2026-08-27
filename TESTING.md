# Voice Feedback Testing

## Voice Agent Tests

The voice agent includes 19 comprehensive tests covering:

### Initialization
- ✓ Enabled/disabled states
- ✓ Environment variable control (`VOICE_ENABLED`)
- ✓ Custom speech rate and volume
- ✓ Voice selection (prefers English voices)
- ✓ Graceful fallback when pyttsx3 is unavailable

### Speech Queue Management
- ✓ Normal speech queuing
- ✓ Priority speech (clears queue)
- ✓ Empty text handling

### Announcements
- ✓ Rep milestones (every 5 reps)
- ✓ Posture alerts
- ✓ Form correction notes
- ✓ Long note truncation
- ✓ Session completion announcements
  - With minutes
  - With seconds only

### Resource Management
- ✓ Proper cleanup on close()
- ✓ Background worker thread

## Running Voice Tests

```bash
# Run voice agent tests only
uv run pytest server/tests/test_voice_agent.py -v

# Test voice functionality
uv run python server/verify_setup.py
```

## Voice Features in Action

During a workout session, the voice agent provides:

1. **Rep Counting**: Every 5 reps → "5 reps", "10 reps", etc.
2. **Posture Alerts**: Immediate warning → "Chest up, brace harder"
3. **Form Corrections**: Announces audit feedback from the AI
4. **Session Complete**: Final summary with rep count and duration

## Disabling Voice for Tests

Set environment variable in test fixtures:

```python
monkeypatch.setenv("VOICE_ENABLED", "false")
```

Or disable in your `.env` file:

```bash
VOICE_ENABLED=false
```

## Platform Support

The voice agent uses `pyttsx3` which provides cross-platform TTS:
- **macOS**: Uses NSSpeechSynthesizer
- **Linux**: Uses espeak or festival
- **Windows**: Uses SAPI5

All tests mock the TTS engine, so they run on any platform without audio hardware.
