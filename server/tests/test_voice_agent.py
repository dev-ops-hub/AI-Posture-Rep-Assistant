from unittest.mock import MagicMock, patch
import queue

import pytest

from agents.voice_agent import VoiceAgent


def test_voice_agent_initialization_enabled():
    with patch("agents.voice_agent.pyttsx3") as mock_pyttsx3:
        mock_engine = MagicMock()
        mock_pyttsx3.init.return_value = mock_engine
        mock_engine.getProperty.return_value = []

        agent = VoiceAgent()

        assert agent.enabled is True
        mock_pyttsx3.init.assert_called_once()
        mock_engine.setProperty.assert_any_call("rate", 150)
        mock_engine.setProperty.assert_any_call("volume", 1.0)


def test_voice_agent_initialization_disabled():
    with patch("agents.voice_agent.pyttsx3") as mock_pyttsx3:
        mock_engine = MagicMock()
        mock_pyttsx3.init.return_value = mock_engine

        agent = VoiceAgent(enabled=False)

        assert agent.enabled is False
        assert agent.engine is None


def test_voice_agent_initialization_no_pyttsx3(monkeypatch):
    monkeypatch.setattr("agents.voice_agent.pyttsx3", None)

    agent = VoiceAgent()

    assert agent.enabled is False
    assert agent.engine is None


def test_voice_agent_initialization_with_env_disabled(monkeypatch):
    monkeypatch.setenv("VOICE_ENABLED", "false")

    with patch("agents.voice_agent.pyttsx3"):
        agent = VoiceAgent()

        assert agent.enabled is False


def test_voice_agent_initialization_custom_rate_volume():
    with patch("agents.voice_agent.pyttsx3") as mock_pyttsx3:
        mock_engine = MagicMock()
        mock_pyttsx3.init.return_value = mock_engine
        mock_engine.getProperty.return_value = []

        agent = VoiceAgent(rate=200, volume=0.8)

        mock_engine.setProperty.assert_any_call("rate", 200)
        mock_engine.setProperty.assert_any_call("volume", 0.8)


def test_voice_agent_speak_when_disabled():
    with patch("agents.voice_agent.pyttsx3"):
        agent = VoiceAgent(enabled=False)
        agent.speak("test message")


def test_voice_agent_speak_empty_text():
    with patch("agents.voice_agent.pyttsx3") as mock_pyttsx3:
        mock_engine = MagicMock()
        mock_pyttsx3.init.return_value = mock_engine
        mock_engine.getProperty.return_value = []

        agent = VoiceAgent()
        agent.speak("")

        assert agent.speech_queue.qsize() == 0


def test_voice_agent_speak_with_priority():
    with patch("agents.voice_agent.pyttsx3") as mock_pyttsx3:
        mock_engine = MagicMock()
        mock_pyttsx3.init.return_value = mock_engine
        mock_engine.getProperty.return_value = []

        agent = VoiceAgent()
        agent.speech_queue.put("old message 1")
        agent.speech_queue.put("old message 2")

        agent.speak("priority message", priority=True)

        assert agent.speech_queue.qsize() == 1


def test_voice_agent_announce_rep_milestone():
    with patch("agents.voice_agent.pyttsx3") as mock_pyttsx3:
        mock_engine = MagicMock()
        mock_pyttsx3.init.return_value = mock_engine
        mock_engine.getProperty.return_value = []

        agent = VoiceAgent()
        agent.announce_rep(5)

        assert agent.speech_queue.qsize() == 1


def test_voice_agent_announce_rep_not_milestone():
    with patch("agents.voice_agent.pyttsx3") as mock_pyttsx3:
        mock_engine = MagicMock()
        mock_pyttsx3.init.return_value = mock_engine
        mock_engine.getProperty.return_value = []

        agent = VoiceAgent()
        agent.announce_rep(3)

        assert agent.speech_queue.qsize() == 0


def test_voice_agent_announce_rep_zero():
    with patch("agents.voice_agent.pyttsx3") as mock_pyttsx3:
        mock_engine = MagicMock()
        mock_pyttsx3.init.return_value = mock_engine
        mock_engine.getProperty.return_value = []

        agent = VoiceAgent()
        agent.announce_rep(0)

        assert agent.speech_queue.qsize() == 0


def test_voice_agent_announce_posture_alert():
    with patch("agents.voice_agent.pyttsx3") as mock_pyttsx3:
        mock_engine = MagicMock()
        mock_pyttsx3.init.return_value = mock_engine
        mock_engine.getProperty.return_value = []

        agent = VoiceAgent()
        agent.announce_posture_alert(0)

        assert agent.speech_queue.qsize() == 1


def test_voice_agent_announce_posture_alert_subsequent():
    with patch("agents.voice_agent.pyttsx3") as mock_pyttsx3:
        mock_engine = MagicMock()
        mock_pyttsx3.init.return_value = mock_engine
        mock_engine.getProperty.return_value = []

        agent = VoiceAgent()
        agent.announce_posture_alert(2)

        assert agent.speech_queue.qsize() == 1


def test_voice_agent_announce_form_note():
    with patch("agents.voice_agent.pyttsx3") as mock_pyttsx3:
        mock_engine = MagicMock()
        mock_pyttsx3.init.return_value = mock_engine
        mock_engine.getProperty.return_value = []

        agent = VoiceAgent()
        agent.announce_form_note("Keep your chest up and core tight", is_first=True)

        assert agent.speech_queue.qsize() == 1


def test_voice_agent_announce_form_note_subsequent():
    with patch("agents.voice_agent.pyttsx3") as mock_pyttsx3:
        mock_engine = MagicMock()
        mock_pyttsx3.init.return_value = mock_engine
        mock_engine.getProperty.return_value = []

        agent = VoiceAgent()
        agent.announce_form_note("Keep your chest up", is_first=False)

        assert agent.speech_queue.qsize() == 1


def test_voice_agent_announce_form_note_truncated():
    with patch("agents.voice_agent.pyttsx3") as mock_pyttsx3:
        mock_engine = MagicMock()
        mock_pyttsx3.init.return_value = mock_engine
        mock_engine.getProperty.return_value = []

        agent = VoiceAgent()
        long_note = "This is a very long form correction note " * 10
        agent.announce_form_note(long_note)

        assert agent.speech_queue.qsize() == 1


def test_voice_agent_announce_session_complete_with_minutes():
    with patch("agents.voice_agent.pyttsx3") as mock_pyttsx3:
        mock_engine = MagicMock()
        mock_pyttsx3.init.return_value = mock_engine
        mock_engine.getProperty.return_value = []

        agent = VoiceAgent()
        agent.announce_session_complete(20, 125.0, 1)

        assert agent.speech_queue.qsize() == 1


def test_voice_agent_announce_session_complete_with_seconds():
    with patch("agents.voice_agent.pyttsx3") as mock_pyttsx3:
        mock_engine = MagicMock()
        mock_pyttsx3.init.return_value = mock_engine
        mock_engine.getProperty.return_value = []

        agent = VoiceAgent()
        agent.announce_session_complete(5, 45.0, 0)

        assert agent.speech_queue.qsize() == 1


def test_voice_agent_announce_session_complete_no_reps():
    with patch("agents.voice_agent.pyttsx3") as mock_pyttsx3:
        mock_engine = MagicMock()
        mock_pyttsx3.init.return_value = mock_engine
        mock_engine.getProperty.return_value = []

        agent = VoiceAgent()
        agent.announce_session_complete(0, 30.0, 0)

        assert agent.speech_queue.qsize() == 1


def test_voice_agent_close_when_enabled():
    with patch("agents.voice_agent.pyttsx3") as mock_pyttsx3:
        mock_engine = MagicMock()
        mock_pyttsx3.init.return_value = mock_engine
        mock_engine.getProperty.return_value = []

        agent = VoiceAgent()
        agent.close()

        mock_engine.stop.assert_called_once()


def test_voice_agent_close_when_disabled():
    with patch("agents.voice_agent.pyttsx3"):
        agent = VoiceAgent(enabled=False)
        agent.close()


def test_voice_agent_voice_selection():
    with patch("agents.voice_agent.pyttsx3") as mock_pyttsx3:
        mock_engine = MagicMock()
        mock_pyttsx3.init.return_value = mock_engine

        mock_voice1 = MagicMock()
        mock_voice1.name = "Spanish Voice"
        mock_voice1.id = "es_voice"

        mock_voice2 = MagicMock()
        mock_voice2.name = "English Voice"
        mock_voice2.id = "en_voice"

        mock_engine.getProperty.return_value = [mock_voice1, mock_voice2]

        agent = VoiceAgent()

        mock_engine.setProperty.assert_any_call("voice", "en_voice")


def test_voice_agent_announce_workout_start():
    with patch("agents.voice_agent.pyttsx3") as mock_pyttsx3:
        mock_engine = MagicMock()
        mock_pyttsx3.init.return_value = mock_engine
        mock_engine.getProperty.return_value = []

        agent = VoiceAgent()
        agent.announce_workout_start()

        assert agent.speech_queue.qsize() == 1
