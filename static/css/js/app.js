

"use strict";


let selectedGenre    = "classical";
let lastFilename     = null;
let generatedNotes   = [];
let currentSynth     = null;
let currentAudioCtx   = null;
let currentAudioNodes = [];
let currentPlayTimer = null;
let audioRecorder    = null;
let genProgressTimer = null;
let trainPollTimer   = null;
let pianoAnim        = null;


document.addEventListener("DOMContentLoaded", () => {
  checkModelStatus();
  setInterval(checkModelStatus, 15_000);

  document.getElementById("useAI").addEventListener("change", function () {
    document.getElementById("modeHint").textContent = this.checked
      ? "LSTM model mode (requires trained weights)"
      : "Fast rule-based mode";
  });
});


function activateStep(n) {
  document.querySelectorAll(".pipe-step").forEach((el, i) => {
    el.classList.toggle("active", i === n);
    if (i < n) el.classList.add("done");
    else el.classList.remove("done");
  });
  document.querySelectorAll(".step-panel").forEach((el, i) => {
    el.classList.toggle("active", i === n);
  });
  document.querySelectorAll(".nav-btn").forEach(b => b.classList.remove("active"));
  if (n === 0) document.getElementById("nav0")?.classList.add("active");
  if (n === 2) document.getElementById("navH")?.classList.add("active");
}


function pickGenre(btn) {
  document.querySelectorAll(".genre-btn").forEach(b => b.classList.remove("selected"));
  btn.classList.add("selected");
  selectedGenre = btn.dataset.genre;
}


async function doGenerate() {
  const n_notes     = parseInt(document.getElementById("notesSlider").value);
  const bpm         = parseInt(document.getElementById("bpmSlider").value);
  const temperature = parseFloat(document.getElementById("tempSlider").value) / 10;
  const use_ai      = document.getElementById("useAI").checked;

  
  setGsbMessage("♩", "Generating your music…", `Genre: ${selectedGenre} · ${n_notes} notes`);
  setProgress(0);
  stopPianoAnim();
  startPianoAnim();

  
  let fakeP = 5;
  genProgressTimer = setInterval(() => {
    fakeP = Math.min(fakeP + Math.random() * 4, 88);
    setProgress(fakeP);
  }, 300);

  try {
    const res = await fetch("/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ genre: selectedGenre, n_notes, bpm, temperature, use_ai }),
    });
    const data = await res.json();

    clearInterval(genProgressTimer);
    setProgress(100);

    if (!data.success) throw new Error(data.error || "Generation failed");

    lastFilename = data.filename;
    generatedNotes = data.notes || [];
    stopPianoAnim();
    setGsbMessage("✓", "Done!", `${data.filename} · ${data.mode}`);

    
    setTimeout(() => showResult(data), 800);

  } catch (err) {
    clearInterval(genProgressTimer);
    stopPianoAnim();
    setGsbMessage("✗", "Error", err.message);
    setProgress(0);
  }
}

function doAnother() {
  lastFilename = null;
  generatedNotes = [];
  updateAudioButtons();
  stopPianoAnim();
}


function showResult(data) {
  activateStep(2);

  document.getElementById("playerFilename").textContent = data.filename;

  const n     = parseInt(document.getElementById("notesSlider").value);
  const bpm   = parseInt(document.getElementById("bpmSlider").value);
  const secs  = Math.round(n / bpm * 30);
  const mins  = Math.floor(secs / 60);
  const s     = secs % 60;
  const temp  = (parseInt(document.getElementById("tempSlider").value) / 10).toFixed(1);

  document.getElementById("playerMeta").textContent =
    `${data.mode} · ${n} notes · ${bpm} BPM · temp ${temp} · ~${mins}m ${String(s).padStart(2,"0")}s`;

  const dlBtn = document.getElementById("downloadBtn");
  dlBtn.href         = data.download;
  dlBtn.download     = data.filename;

  updateAudioButtons();
  loadHistory();
}


async function loadHistory() {
  try {
    const res   = await fetch("/files");
    const data  = await res.json();
    const list  = document.getElementById("historyList");
    const wrap  = document.getElementById("historyWrap");
    if (!list) return;
    if (!data.files.length) { wrap.style.display = "none"; return; }
    wrap.style.display = "block";
    list.innerHTML = data.files.map(f => `
      <div class="history-item">
        <span class="hi-name">${f}</span>
        <a class="hi-dl" href="/download/${encodeURIComponent(f)}" download="${f}">↓ MIDI</a>
      </div>`).join("");
  } catch (_) { /* silent */ }
}


async function startTraining() {
  const genre  = document.getElementById("trainGenre").value;
  const epochs = parseInt(document.getElementById("epochsSlider").value);
  const btn    = document.getElementById("trainBtn");
  const wrap   = document.getElementById("trainProgressWrap");

  btn.disabled    = true;
  wrap.style.display = "block";
  logTrain("Starting training pipeline…");

  try {
    const res  = await fetch("/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ genre, epochs }),
    });
    const data = await res.json();
    if (!data.success) throw new Error(data.error);

    pollTraining();
  } catch (err) {
    btn.disabled = false;
    logTrain(`Error: ${err.message}`);
  }
}

function pollTraining() {
  clearInterval(trainPollTimer);
  trainPollTimer = setInterval(async () => {
    try {
      const res  = await fetch("/train/status");
      const data = await res.json();

      document.getElementById("trainMsg").textContent  = data.message;
      document.getElementById("trainPct").textContent  = data.progress + "%";
      document.getElementById("trainFill").style.width = data.progress + "%";

      if (data.message !== document.getElementById("trainLog").dataset.last) {
        logTrain(data.message);
        document.getElementById("trainLog").dataset.last = data.message;
      }

      if (!data.running) {
        clearInterval(trainPollTimer);
        document.getElementById("trainBtn").disabled = false;
        if (data.error) {
          logTrain("✗ " + data.error);
        } else {
          logTrain("✓ Model saved → models/lstm_weights.h5");
          checkModelStatus();
        }
      }
    } catch (_) { /* silent */ }
  }, 1500);
}

function logTrain(msg) {
  const el = document.getElementById("trainLog");
  if (!el) return;
  const t  = new Date().toLocaleTimeString("en-GB", { hour12: false });
  el.innerHTML += `<div>[${t}] ${msg}</div>`;
  el.scrollTop  = el.scrollHeight;
}


async function checkModelStatus() {
  try {
    const res  = await fetch("/status");
    const data = await res.json();
    const dot  = document.querySelector(".pill-dot");
    const txt  = document.getElementById("modelPillText");

    if (data.model_cached) {
      dot.className  = "pill-dot ready";
      txt.textContent = "Model ready";
    } else {
      dot.className  = "pill-dot";
      txt.textContent = "No model — rule-based";
    }
  } catch (_) { /* silent */ }
}


function startPianoAnim() {
  const canvas = document.getElementById("pianoRoll");
  if (!canvas) return;
  const ctx    = canvas.getContext("2d");
  canvas.width = 800; canvas.height = 90;

  const NOTES     = [];
  const SPEED     = 2;
  const COLORS    = ["#6366f1","#8b5cf6","#4f46e5","#7c3aed","#a78bfa","#818cf8"];
  let   frameReq  = null;

  function spawnNote() {
    NOTES.push({
      x: canvas.width,
      y: 10 + Math.random() * 60,
      w: 20 + Math.random() * 35,
      h: 8 + Math.random() * 6,
      col: COLORS[Math.floor(Math.random() * COLORS.length)],
      alpha: 0,
    });
  }

  let spawnTick = 0;
  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "rgba(10,10,15,0.6)";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    spawnTick++;
    if (spawnTick % 8 === 0 && NOTES.length < 60) spawnNote();

    for (let i = NOTES.length - 1; i >= 0; i--) {
      const n = NOTES[i];
      n.x    -= SPEED;
      n.alpha = Math.min(1, n.alpha + 0.1);
      ctx.globalAlpha = n.alpha;
      ctx.fillStyle   = n.col;
      ctx.beginPath();
      ctx.roundRect(n.x, n.y, n.w, n.h, 3);
      ctx.fill();
      if (n.x + n.w < 0) NOTES.splice(i, 1);
    }
    ctx.globalAlpha = 1;
    frameReq = requestAnimationFrame(draw);
  }

  draw();
  pianoAnim = { stop: () => { cancelAnimationFrame(frameReq); } };
}

function stopPianoAnim() {
  if (pianoAnim) { pianoAnim.stop(); pianoAnim = null; }
  const canvas = document.getElementById("pianoRoll");
  if (canvas) {
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }
}

function updateAudioButtons() {
  const playBtn = document.getElementById("playAudioBtn");
  const exportBtn = document.getElementById("exportAudioBtn");
  const enabled = generatedNotes.length > 0;
  if (playBtn) playBtn.disabled = !enabled;
  if (exportBtn) exportBtn.disabled = !enabled;
}

function setAudioStatus(msg) {
  const el = document.getElementById("audioStatus");
  if (el) el.textContent = msg;
}

async function playGeneratedAudio() {
  if (!generatedNotes.length) return;
  stopGeneratedAudio();

  const ctx = getAudioContext();
  if (!ctx) {
    setAudioStatus("Audio is not supported in this browser.");
    return;
  }
  if (ctx.state === "suspended") await ctx.resume();

  setAudioStatus("Playing audio...");

  const bpm = parseInt(document.getElementById("bpmSlider").value);
  const step = (60 / bpm) * 0.5;
  const events = buildHumanizedEvents(generatedNotes, step);
  const endTime = scheduleHumanizedPlayback(ctx, ctx.destination, events, ctx.currentTime + 0.15);

  currentPlayTimer = setTimeout(() => {
    stopGeneratedAudio();
    setAudioStatus("Playback complete.");
  }, Math.max(0, (endTime - ctx.currentTime) * 1000) + 300);
}

function stopGeneratedAudio() {
  if (currentSynth) {
    if (typeof currentSynth.releaseAll === "function") currentSynth.releaseAll();
    if (typeof currentSynth.dispose === "function") currentSynth.dispose();
    currentSynth = null;
  }
  currentAudioNodes.forEach(node => {
    try { node.stop(); } catch (_) { /* already stopped */ }
    try { node.disconnect(); } catch (_) { /* already disconnected */ }
  });
  currentAudioNodes = [];
  if (currentPlayTimer) {
    clearTimeout(currentPlayTimer);
    currentPlayTimer = null;
  }
  if (audioRecorder && audioRecorder.state !== "inactive") {
    try { audioRecorder.stop(); } catch (_) { /* recorder already stopping */ }
  }
}

async function exportGeneratedAudio() {
  if (!generatedNotes.length) return;
  if (!window.MediaRecorder) {
    setAudioStatus("Audio export is not supported in this browser.");
    return;
  }

  stopGeneratedAudio();

  const ctx = getAudioContext();
  if (!ctx) {
    setAudioStatus("Audio is not supported in this browser.");
    return;
  }
  if (ctx.state === "suspended") await ctx.resume();

  setAudioStatus("Recording audio...");

  const dest = ctx.createMediaStreamDestination();
  const recorder = new MediaRecorder(dest.stream);
  const chunks = [];

  recorder.ondataavailable = event => {
    if (event.data && event.data.size) chunks.push(event.data);
  };

  recorder.onstop = () => {
    const blob = new Blob(chunks, { type: chunks[0]?.type || "audio/webm" });
    const ext = blob.type.includes("wav") ? "wav" : blob.type.includes("ogg") ? "ogg" : "webm";
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = (lastFilename ? lastFilename.replace(/\.mid$/, "") : "music") + "_audio." + ext;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 5000);
    setAudioStatus("Audio exported.");
    audioRecorder = null;
  };

  recorder.start();
  audioRecorder = recorder;

  const bpm = parseInt(document.getElementById("bpmSlider").value);
  const step = (60 / bpm) * 0.5;
  const events = buildHumanizedEvents(generatedNotes, step);
  const endTime = scheduleHumanizedPlayback(ctx, dest, events, ctx.currentTime + 0.15);

  setTimeout(() => {
    if (recorder.state !== "inactive") recorder.stop();
    stopGeneratedAudio();
  }, Math.max(0, (endTime - ctx.currentTime) * 1000) + 500);
}

function getAudioContext() {
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) return null;
  if (!currentAudioCtx) currentAudioCtx = new AudioContextClass();
  return currentAudioCtx;
}

function buildHumanizedEvents(patterns, step) {
  return patterns.map((pattern, idx) => {
    const phraseAccent = idx % 8 === 0 ? 0.08 : 0;
    const timingDrift = (Math.random() - 0.5) * Math.min(0.035, step * 0.22);
    const duration = step * (0.78 + Math.random() * 0.22);
    const velocity = Math.min(0.62, 0.34 + phraseAccent + Math.random() * 0.14);

    return {
      notes: parsePatternToFrequencies(pattern),
      time: Math.max(0, idx * step + timingDrift),
      duration,
      velocity,
    };
  }).filter(event => event.notes.length);
}

function scheduleHumanizedPlayback(ctx, destination, events, startAt) {
  const compressor = ctx.createDynamicsCompressor();
  compressor.threshold.value = -24;
  compressor.knee.value = 24;
  compressor.ratio.value = 4;
  compressor.attack.value = 0.006;
  compressor.release.value = 0.18;
  compressor.connect(destination);

  currentAudioNodes.push(compressor);

  let endTime = startAt;
  events.forEach(event => {
    const at = startAt + event.time;
    endTime = Math.max(endTime, at + event.duration);
    event.notes.forEach((freq, voiceIdx) => {
      scheduleVoice(ctx, compressor, freq, at + voiceIdx * 0.006, event.duration, event.velocity);
    });
  });

  return endTime;
}

function scheduleVoice(ctx, destination, frequency, startAt, duration, velocity) {
  const osc = ctx.createOscillator();
  const gain = ctx.createGain();
  const filter = ctx.createBiquadFilter();

  osc.type = "triangle";
  osc.frequency.setValueAtTime(frequency, startAt);
  osc.detune.setValueAtTime((Math.random() - 0.5) * 8, startAt);

  filter.type = "lowpass";
  filter.frequency.setValueAtTime(1800 + Math.random() * 600, startAt);
  filter.Q.setValueAtTime(0.7, startAt);

  const attack = 0.012 + Math.random() * 0.018;
  const release = 0.12 + Math.random() * 0.08;
  const stopAt = startAt + duration + release;

  gain.gain.setValueAtTime(0.0001, startAt);
  gain.gain.exponentialRampToValueAtTime(Math.max(0.0002, velocity), startAt + attack);
  gain.gain.exponentialRampToValueAtTime(0.0001, stopAt);

  osc.connect(filter);
  filter.connect(gain);
  gain.connect(destination);

  osc.start(startAt);
  osc.stop(stopAt + 0.02);
  osc.onended = () => {
    [osc, filter, gain].forEach(node => {
      try { node.disconnect(); } catch (_) { /* already disconnected */ }
    });
    currentAudioNodes = currentAudioNodes.filter(node => node !== osc);
  };

  currentAudioNodes.push(osc, filter, gain);
}

function parsePatternToFrequencies(pattern) {
  if (pattern.includes(".")) {
    return pattern.split(".")
      .map(num => parseInt(num, 10))
      .filter(num => Number.isFinite(num))
      .map(num => midiToFrequency(60 + num));
  }
  const midi = noteNameToMidi(pattern);
  return Number.isFinite(midi) ? [midiToFrequency(midi)] : [];
}

function noteNameToMidi(name) {
  const match = String(name).trim().match(/^([A-Ga-g])([#-]?)(-?\d+)$/);
  if (!match) return NaN;

  const base = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 }[match[1].toUpperCase()];
  const accidental = match[2] === "#" ? 1 : match[2] === "-" ? -1 : 0;
  const octave = parseInt(match[3], 10);
  return (octave + 1) * 12 + base + accidental;
}

function midiToFrequency(midi) {
  return 440 * Math.pow(2, (midi - 69) / 12);
}

/* ─────────────────────────────────────────────
   HELPERS
───────────────────────────────────────────── */
function setGsbMessage(icon, msg, sub) {
  document.getElementById("gsbIcon").textContent    = icon;
  document.getElementById("gsbMessage").textContent = msg;
  document.getElementById("gsbSub").textContent     = sub;
}

function setProgress(pct) {
  document.getElementById("genProgress").style.width = pct + "%";
}
