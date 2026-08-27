# Voice Feedback Feature

## Overview

The AI Posture Rep Assistant now includes real-time voice feedback to provide audio coaching during your workouts. This allows you to focus on your form without needing to look at the screen.

## Features

### 1. Rep Count Announcements
Every 5 reps, the voice agent announces your progress:
- "5 reps"
- "10 reps"
- "15 reps"
- etc.

### 2. Posture Alerts
When a posture violation is detected, you immediately hear:
- **"Chest up, brace harder"**

This alert is spoken with priority, interrupting any other queued speech to provide immediate feedback.

### 3. Form Corrections
After the audit agent analyzes a posture violation, the detailed correction is spoken:
- Example: "Your torso is folding forward. Brace your core and keep your chest stacked over your hips on the descent."

Notes are truncated to 100 characters for concise audio feedback.

### 4. Session Complete
At the end of your workout, you hear a summary:
- Example: "Workout complete. 20 reps in 2 minutes. Great job!"

## Configuration

### Enable/Disable Voice

Set in your `.env` file:

```bash
# Enable voice feedback (default)
VOICE_ENABLED=true

# Disable voice feedback
VOICE_ENABLED=false
```

### Custom Voice Settings

You can customize the voice agent in `server/agents/voice_agent.py`:

```python
voice_agent = VoiceAgent(
    enabled=True,
    rate=175,      # Words per minute (default: 175)
    volume=0.9     # Volume 0.0 to 1.0 (default: 0.9)
)
```

## Technical Details

### Text-to-Speech Engine

The voice agent uses **pyttsx3**, which provides:
- ✓ Offline operation (no internet or API keys needed)
- ✓ Cross-platform support (macOS, Linux, Windows)
- ✓ Low latency for real-time feedback
- ✓ Multiple voice options

### Platform-Specific TTS

- **macOS**: Uses NSSpeechSynthesizer (native macOS voices)
- **Linux**: Uses espeak or festival
- **Windows**: Uses SAPI5 (Microsoft Speech API)

### Voice Selection

The agent automatically selects an English voice if available. On macOS, common voices include:
- "Alex" (male)
- "Samantha" (female)
- "Victoria" (female)

### Architecture

The voice agent uses a background worker thread to prevent blocking the main application:

```
Main Thread              Worker Thread
    |                         |
    | speak("5 reps")         |
    |---> Queue ------>       |
    |                    Read from queue
    |                    Call TTS engine
    | continue workout   Wait for speech
    |                    |
```

This ensures the camera feed and rep counting continue smoothly while speech is being processed.

### Priority Speech

Posture alerts use priority speech, which:
1. Clears any queued speech
2. Speaks immediately
3. Ensures critical feedback isn't delayed

## Testing

### Unit Tests

19 comprehensive tests cover all voice functionality:

```bash
uv run pytest tests/test_voice_agent.py -v
```

### Manual Testing

```bash
# Test voice without webcam
uv run python server/verify_setup.py

# Test voice during workout
uv run main.py
```

## Troubleshooting

### No Voice Output

1. Check `VOICE_ENABLED` in `.env` is set to `true`
2. Verify `pyttsx3` is installed: `uv pip list | grep pyttsx3`
3. Check system audio is not muted
4. On Linux, ensure espeak is installed: `sudo apt-get install espeak`

### Voice is Too Fast/Slow

Adjust the rate parameter in `server/agents/voice_agent.py`:

```python
voice_agent = VoiceAgent(rate=150)  # Slower
voice_agent = VoiceAgent(rate=200)  # Faster
```

### Voice is Too Quiet/Loud

Adjust the volume parameter:

```python
voice_agent = VoiceAgent(volume=0.5)  # Quieter
voice_agent = VoiceAgent(volume=1.0)  # Maximum
```

### Different Voice

To manually select a voice, modify the initialization in `server/agents/voice_agent.py`:

```python
voices = self.engine.getProperty("voices")
# Print available voices
for voice in voices:
    print(f"{voice.name}: {voice.id}")

# Set specific voice
self.engine.setProperty("voice", "specific_voice_id")
```

## Performance Impact

The voice agent has minimal performance impact:
- ✓ Speech processing runs in background thread
- ✓ Does not affect camera FPS or rep detection
- ✓ Queue prevents speech from overlapping
- ✓ Optional feature (can be disabled)

## Future Enhancements

Potential improvements:
- Voice commands (e.g., "pause", "stop")
- Customizable announcement frequency
- Different voice profiles for different workout types
- Configurable announcement phrases
- Multiple language support
