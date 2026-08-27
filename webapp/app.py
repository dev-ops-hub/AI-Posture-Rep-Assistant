from __future__ import annotations

import os
import signal
import threading
import time

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request

from webapp.session_manager import SessionConfig, WorkoutSessionManager

load_dotenv()

app = Flask(__name__)
session_manager = WorkoutSessionManager()


def _shutdown_server(delay: float = 0.5) -> None:
    """Terminates the running Flask process shortly after the response is sent."""

    def _kill() -> None:
        time.sleep(delay)
        os.kill(os.getpid(), signal.SIGINT)

    threading.Thread(target=_kill, daemon=True).start()


def _default_config() -> dict[str, float | str | int]:
    return {
        "camera_index": int(os.getenv("CAMERA_INDEX", "0")),
        "weight_kg": float(os.getenv("USER_WEIGHT_KG", "70")),
        "goal": os.getenv("FITNESS_GOAL", "general fitness"),
        "met_value": float(os.getenv("WORKOUT_MET", "5.0")),
    }


@app.route("/")
def index():
    return render_template("index.html", defaults=_default_config())


@app.route("/api/config")
def api_config():
    return jsonify(_default_config())


@app.route("/api/start", methods=["POST"])
def api_start():
    payload = request.get_json(silent=True) or {}
    defaults = _default_config()
    config = SessionConfig(
        camera_index=int(payload.get("camera_index", defaults["camera_index"])),
        weight_kg=float(payload.get("weight_kg", defaults["weight_kg"])),
        goal=str(payload.get("goal", defaults["goal"])),
        met_value=float(payload.get("met_value", defaults["met_value"])),
    )
    result = session_manager.start(config)
    status_code = 200 if result.get("ok") else 400
    return jsonify(result), status_code


@app.route("/api/pause", methods=["POST"])
def api_pause():
    result = session_manager.toggle_pause()
    status_code = 200 if result.get("ok") else 400
    return jsonify(result), status_code


@app.route("/api/stop", methods=["POST"])
def api_stop():
    result = session_manager.stop()
    status_code = 200 if result.get("ok") else 400
    return jsonify(result), status_code


@app.route("/api/quit", methods=["POST"])
def api_quit():
    result = session_manager.quit()
    _shutdown_server()
    return jsonify(result), 200


@app.route("/api/status")
def api_status():
    return jsonify(session_manager.get_status())


def _mjpeg_generator():
    while True:
        frame = session_manager.get_frame()
        if frame:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.05)


@app.route("/video_feed")
def video_feed():
    return Response(_mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("WEB_PORT", "5000")), debug=False, threaded=True)
