const el = (id) => document.getElementById(id);

const badge = el("session-badge");
const btnStart = el("btn-start");
const btnPause = el("btn-pause");
const btnStop = el("btn-stop");
const btnQuit = el("btn-quit");
const errorBanner = el("error-banner");
const latestNoteBox = el("latest-note");
const latestNoteText = el("latest-note-text");
const videoFeed = el("video-feed");

const reportOverlay = el("report-overlay");
const reportSummary = el("report-summary");
const reportTips = el("report-tips");
const reportCoach = el("report-coach");
const btnReportClose = el("btn-report-close");

let pollTimer = null;
let currentState = "idle";

function setBadge(state) {
    badge.textContent = state.charAt(0).toUpperCase() + state.slice(1);
    badge.className = `badge ${state}`;
}

function setButtons(state) {
    currentState = state;
    btnStart.disabled = state === "running" || state === "paused";
    btnPause.disabled = !(state === "running" || state === "paused");
    btnPause.textContent = state === "paused" ? "▶ Resume" : "⏸ Pause";
    btnStop.disabled = !(state === "running" || state === "paused");
    btnQuit.disabled = state === "idle" || state === "closed";
}

function showError(message) {
    if (!message) {
        errorBanner.classList.add("hidden");
        errorBanner.textContent = "";
        return;
    }
    errorBanner.textContent = message;
    errorBanner.classList.remove("hidden");
}

async function postJSON(url, body) {
    const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body || {}),
    });
    return response.json();
}

function refreshVideoFeed() {
    videoFeed.src = `/video_feed?t=${Date.now()}`;
}

async function pollStatus() {
    try {
        const status = await (await fetch("/api/status")).json();
        setBadge(status.session_state);
        setButtons(status.session_state);
        showError(status.error);

        el("stat-reps").textContent = status.reps;
        el("stat-elapsed").textContent = status.elapsed_formatted;
        el("stat-faults").textContent = status.posture_fault_count;
        el("stat-state").textContent = status.state;
        el("stat-knee").textContent = `${status.knee_angle_deg}°`;
        el("stat-spine").textContent = `${status.spine_angle_deg}°`;

        if (status.latest_note) {
            latestNoteBox.classList.remove("hidden");
            latestNoteText.textContent = status.latest_note;
        } else {
            latestNoteBox.classList.add("hidden");
        }
    } catch (err) {
        // Server may be briefly unavailable; ignore transient errors.
    }
}

function startPolling() {
    if (pollTimer) return;
    pollTimer = setInterval(pollStatus, 800);
}

function stopPolling() {
    if (pollTimer) {
        clearInterval(pollTimer);
        pollTimer = null;
    }
}

function showReport(report) {
    if (!report) return;

    reportSummary.innerHTML = `
        <div><span class="label">Exercise</span><span class="value">${report.exercise}</span></div>
        <div><span class="label">Total Reps</span><span class="value">${report.total_reps}</span></div>
        <div><span class="label">Duration</span><span class="value">${report.duration_formatted}</span></div>
        <div><span class="label">Calories</span><span class="value">${report.calories} kcal</span></div>
        <div><span class="label">Posture Faults</span><span class="value">${report.total_posture_faults}</span></div>
    `;

    reportTips.innerHTML = "";
    (report.improvement_tips || []).forEach((tip) => {
        const li = document.createElement("li");
        li.textContent = tip;
        reportTips.appendChild(li);
    });

    reportCoach.textContent = report.coach_summary || "";
    reportOverlay.classList.remove("hidden");
}

btnStart.addEventListener("click", async () => {
    showError("");
    const config = {
        camera_index: parseInt(el("cfg-camera").value, 10),
        weight_kg: parseFloat(el("cfg-weight").value),
        goal: el("cfg-goal").value,
        met_value: parseFloat(el("cfg-met").value),
    };
    const result = await postJSON("/api/start", config);
    if (!result.ok) {
        showError(result.error || "Could not start session.");
        return;
    }
    reportOverlay.classList.add("hidden");
    refreshVideoFeed();
    setButtons("running");
    startPolling();
});

btnPause.addEventListener("click", async () => {
    const result = await postJSON("/api/pause");
    if (!result.ok) {
        showError(result.error || "Could not toggle pause.");
        return;
    }
    setButtons(result.state);
});

btnStop.addEventListener("click", async () => {
    const result = await postJSON("/api/stop");
    if (!result.ok) {
        showError(result.error || "Could not stop session.");
        return;
    }
    setButtons("stopped");
    showReport(result.report);
});

btnQuit.addEventListener("click", async () => {
    const result = await postJSON("/api/quit");
    setButtons("closed");
    stopPolling();
    if (result.report) {
        showReport(result.report);
    }
    showError("Server shutting down. You can close this browser tab.");
    videoFeed.src = "";
});

btnReportClose.addEventListener("click", () => {
    reportOverlay.classList.add("hidden");
});

// Initial state sync on page load.
pollStatus();
startPolling();
