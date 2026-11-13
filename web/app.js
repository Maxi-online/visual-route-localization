const BUILD = "VLOC MVP";
const PROJECT_FULL_NAME = "VLOC MVP (Visual Localization)";
const PREVIEW_SRC = "/api/v1/runtime/preview";
const DEFAULT_LAT = 48.402872;
const DEFAULT_LNG = 37.342720;

const map = L.map("map", { zoomControl: true }).setView([DEFAULT_LAT, DEFAULT_LNG], 12);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19,
  attribution: "© OpenStreetMap",
}).addTo(map);

const marker = L.circleMarker([DEFAULT_LAT, DEFAULT_LNG], {
  radius: 8,
  weight: 2,
  fillOpacity: 0.7,
}).addTo(map);

const routeLayers = [];
const videoFeed = document.getElementById("videoFeed");

const indexSourceType = document.getElementById("indexSourceType");
const indexSourceSelect = document.getElementById("indexSourceSelect");
const indexSourceInput = document.getElementById("indexSourceInput");
const indexVideoFile = document.getElementById("indexVideoFile");
const indexPointsPath = document.getElementById("indexPointsPath");
const pointsFile = document.getElementById("pointsFile");
const indexMaxKeyframes = document.getElementById("indexMaxKeyframes");
const startIndexBtn = document.getElementById("startIndexBtn");

const runtimeSourceType = document.getElementById("runtimeSourceType");
const sourceSelect = document.getElementById("sourceSelect");
const sourceInput = document.getElementById("sourceInput");
const runtimeVideoFile = document.getElementById("runtimeVideoFile");
const resizeWidth = document.getElementById("resizeWidth");
const topK = document.getElementById("topK");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");

const ids = (id) => document.getElementById(id);
const setVal = (id, value) => ids(id).textContent = String(value);
const setHidden = (id, hidden) => ids(id).classList.toggle("hidden", hidden);

let startInFlight = false;
let indexInFlight = false;
let lastAcceptedTsMs = -1;
let markerAnimFrame = null;
let trackPoints = [];
let renderTrackTsMs = null;
let lastRenderNow = null;
let mapInteractionActive = false;
let remoteFrameSocket = null;
let backendIngestSocket = null;
let cameraStream = null;
let cameraBridgeTimer = null;
let cameraBridgeVideo = null;
let cameraBridgeCanvas = null;
let activeRuntimeSessionId = "";

function setMarkerVisible(visible) {
  marker.setStyle({
    opacity: visible ? 1.0 : 0.0,
    fillOpacity: visible ? 0.7 : 0.0,
  });
}

function bindRange(id, outId) {
  const el = ids(id);
  const out = ids(outId);
  const sync = () => {
    out.textContent = el.value;
  };
  el.addEventListener("input", sync);
  sync();
}

bindRange("resizeWidth", "resizeWidthVal");
bindRange("topK", "topKVal");
function fillSourceSelect(selectEl, sources) {
  selectEl.innerHTML = "";
  for (const src of sources || []) {
    const opt = document.createElement("option");
    opt.value = src.value || "";
    opt.textContent = src.label || src.value || "";
    opt.dataset.type = src.type || "video";
    selectEl.appendChild(opt);
  }
}

function currentSourceValue(selectEl, inputEl) {
  return (inputEl.value || "").trim() || (selectEl.value || "").trim();
}

function syncInputFromSelect(selectEl, inputEl) {
  inputEl.value = selectEl.value || "";
}

function shortSourceLabel(value) {
  const raw = String(value || "").trim();
  if (!raw) return "";
  const parts = raw.split(/[\\/]/).filter(Boolean);
  return parts.length > 0 ? parts[parts.length - 1] : raw;
}

function applyIndexSourceTypeUi() {
  const type = indexSourceType.value;
  setHidden("indexVideoFileWrap", type !== "video");
  setHidden("indexSelectWrap", type !== "video");
  indexSourceInput.placeholder = type === "rtsp"
    ? "rtsp://... для эталонного маршрута"
    : "укажите путь к эталонному видео или загрузите файл";
  const note = ids("indexModeNote");
  if (type === "video" || type === "rtsp") {
    note.textContent = "Для индексации можно указать путь к доступным файлам на этой машине или загрузить их через UI.";
  } else if (type === "websocket") {
    note.textContent = "Индексация из WebSocket пока не включена в MVP. Используйте видеофайл или RTSP.";
  } else {
    note.textContent = "Прямой WebRTC для индексации пока не включен в MVP. Используйте видеофайл или RTSP.";
  }
}

function applyRuntimeSourceTypeUi() {
  const type = runtimeSourceType.value;
  setHidden("runtimeSelectWrap", type !== "video");
  setHidden("runtimeVideoFileWrap", type !== "video");
  sourceInput.placeholder =
    type === "rtsp" ? "rtsp://..." :
    type === "websocket" ? "ws://remote-source (или оставить пустым и слать кадры в ingest URL)" :
    type === "webrtc" ? "опционально: описание/browser bridge" :
    "укажите путь к runtime-видео или загрузите файл";
  const note = ids("runtimeSourceNote");
  if (type === "video" || type === "rtsp") {
    note.innerHTML = [
      "`Video`: можно выбрать готовый runtime-источник, указать путь к локальному файлу или загрузить видео через UI.",
      "`Path`: используйте путь к доступному файлу на этой машине, если не хотите загружать файл через интерфейс.",
    ].join("<br />");
  } else if (type === "websocket") {
    note.innerHTML = [
      "`WebSocket`: можно указать удаленный `ws://` источник кадров или оставить поле пустым и отправлять кадры прямо в backend ingest URL.",
      "`Video upload` в этом режиме не используется.",
    ].join("<br />");
  } else {
    note.innerHTML = [
      "`WebRTC`: в MVP работает как browser camera bridge и отдает кадры в backend через WebSocket.",
      "`Video upload` в этом режиме не используется.",
    ].join("<br />");
  }
}

async function loadRuntimeSources() {
  const res = await fetch("/api/v1/runtime/sources");
  const data = await res.json();
  fillSourceSelect(sourceSelect, data.sources || []);
  if ((data.sources || []).length > 0 && !sourceInput.value.trim()) {
    syncInputFromSelect(sourceSelect, sourceInput);
  }
}

async function loadIndexSources() {
  const res = await fetch("/api/v1/index/sources");
  const data = await res.json();
  fillSourceSelect(indexSourceSelect, data.sources || []);
  if ((data.sources || []).length > 0 && !indexSourceInput.value.trim()) {
    syncInputFromSelect(indexSourceSelect, indexSourceInput);
  }
}

async function loadRoutes() {
  const res = await fetch("/api/v1/index/routes");
  const data = await res.json();
  setVal("kfCount", data.keyframes || 0);
  setVal("routeCount", (data.videos || []).length);
  for (const layer of routeLayers) {
    map.removeLayer(layer);
  }
  routeLayers.length = 0;
}

function syncRenderTrackToLive() {
  if (trackPoints.length === 0) {
    return;
  }
  const latest = trackPoints[trackPoints.length - 1];
  renderTrackTsMs = latest.ts_ms;
  marker.setLatLng([latest.lat, latest.lng]);
  lastRenderNow = null;
}

function setMapInteractionActive(active) {
  mapInteractionActive = active;
  if (!active) {
    syncRenderTrackToLive();
  } else {
    lastRenderNow = null;
  }
}

function ensureMarkerLoop() {
  if (markerAnimFrame !== null) return;

  const step = (now) => {
    lastRenderNow = now;

    if (trackPoints.length > 0) {
      const latest = trackPoints[trackPoints.length - 1];
      renderTrackTsMs = latest.ts_ms;
      marker.setLatLng([latest.lat, latest.lng]);
      if (trackPoints.length > 1) {
        trackPoints = [latest];
      }
    }

    markerAnimFrame = requestAnimationFrame(step);
  };

  markerAnimFrame = requestAnimationFrame(step);
}

function resetTrackPlayback() {
  trackPoints = [];
  renderTrackTsMs = null;
  lastRenderNow = null;
  lastAcceptedTsMs = -1;
  mapInteractionActive = false;
  activeRuntimeSessionId = "";
  marker.setLatLng([DEFAULT_LAT, DEFAULT_LNG]);
  setMarkerVisible(false);
}

function updateMarker(msg) {
  if (typeof msg.ts_ms === "number" && msg.ts_ms <= lastAcceptedTsMs) {
    return;
  }
  if (typeof msg.ts_ms === "number") {
    lastAcceptedTsMs = msg.ts_ms;
  }

  trackPoints.push({
    ts_ms: Number(msg.ts_ms),
    lat: Number(msg.lat),
    lng: Number(msg.lng),
  });
  if (trackPoints.length > 24) {
    trackPoints = trackPoints.slice(-24);
  }
  const latest = trackPoints[trackPoints.length - 1];
  setMarkerVisible(true);
  marker.setLatLng([latest.lat, latest.lng]);
  ensureMarkerLoop();
  setVal("lat", msg.lat.toFixed(6));
  setVal("lng", msg.lng.toFixed(6));
  setVal("conf", `${msg.confidence.toFixed(1)}%`);
  setVal("ref", msg.ref_keyframe_id === null ? "—" : String(msg.ref_keyframe_id));
}

function updateBridgeStatus(message, tone = "small") {
  const el = ids("bridgeStatus");
  el.className = tone;
  el.textContent = message || "";
}

async function fetchStatus() {
  const res = await fetch("/api/v1/runtime/status");
  const data = await res.json();
  ids("rtRunning").textContent = data.running ? "RUNNING" : "STOPPED";
  ids("rtRunning").className = data.running ? "mono status-ok" : "mono status-bad";
  ids("rtSourceType").textContent = data.source_type || "—";
  ids("rtDecoded").textContent = String(data.decoded_frames || 0);
  ids("rtLocalized").textContent = String(data.localized_frames || 0);
  ids("rtError").textContent = data.last_error || "—";
  ids("buildVersion").textContent = PROJECT_FULL_NAME;
  if (typeof data.runtime_session_id === "string") {
    activeRuntimeSessionId = data.runtime_session_id;
  }

  if (data.running) {
    if (!videoFeed.getAttribute("src")) {
      videoFeed.src = `${PREVIEW_SRC}?v=${Date.now()}`;
    }
  } else if (!data.has_frame) {
    videoFeed.removeAttribute("src");
  }
  const last = data.last_result;
  if (
    last &&
    typeof last.lat === "number" &&
    typeof last.lng === "number" &&
    last.confidence > 0 &&
    (!activeRuntimeSessionId || last.runtime_session_id === activeRuntimeSessionId)
  ) {
    updateMarker(last);
  }
  return data;
}

async function fetchIndexStatus() {
  const res = await fetch("/api/v1/index/status");
  const data = await res.json();
  ids("indexStatus").textContent = data.detail || "—";
  ids("indexError").textContent = data.error || "";
  if (typeof data.keyframes === "number") {
    setVal("kfCount", data.keyframes);
  }
  return data;
}

function socketOpen(url, binaryType = "arraybuffer") {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(url);
    ws.binaryType = binaryType;
    ws.onopen = () => resolve(ws);
    ws.onerror = () => reject(new Error(`WebSocket open failed: ${url}`));
  });
}

async function cleanupRuntimeBridge() {
  if (remoteFrameSocket) {
    try { remoteFrameSocket.close(); } catch (e) {}
    remoteFrameSocket = null;
  }
  if (backendIngestSocket) {
    try { backendIngestSocket.close(); } catch (e) {}
    backendIngestSocket = null;
  }
  if (cameraBridgeTimer !== null) {
    try { cancelAnimationFrame(cameraBridgeTimer); } catch (e) {}
    clearInterval(cameraBridgeTimer);
    cameraBridgeTimer = null;
  }
  if (cameraStream) {
    for (const track of cameraStream.getTracks()) {
      track.stop();
    }
    cameraStream = null;
  }
  cameraBridgeVideo = null;
  cameraBridgeCanvas = null;
  updateBridgeStatus("");
}

function normalizeIncomingWsText(text) {
  try {
    const obj = JSON.parse(text);
    if (obj && typeof obj === "object") {
      if (typeof obj.image_base64 === "string") {
        return JSON.stringify({
          image_base64: obj.image_base64,
          source_ts_ms: obj.source_ts_ms ?? obj.ts_ms ?? null,
        });
      }
      if (typeof obj.data === "string") {
        return JSON.stringify({
          image_base64: obj.data,
          source_ts_ms: obj.source_ts_ms ?? obj.ts_ms ?? null,
        });
      }
    }
  } catch (e) {}

  if (typeof text === "string" && text.startsWith("data:image/")) {
    return JSON.stringify({ image_base64: text, source_ts_ms: null });
  }
  return text;
}

async function startRemoteWebSocketBridge(remoteUrl, ingestUrl) {
  await cleanupRuntimeBridge();
  backendIngestSocket = await socketOpen(ingestUrl);
  remoteFrameSocket = await socketOpen(remoteUrl);
  remoteFrameSocket.binaryType = "arraybuffer";

  remoteFrameSocket.onmessage = async (event) => {
    if (!backendIngestSocket || backendIngestSocket.readyState !== WebSocket.OPEN) {
      return;
    }
    if (event.data instanceof ArrayBuffer) {
      backendIngestSocket.send(event.data);
      return;
    }
    if (event.data instanceof Blob) {
      backendIngestSocket.send(await event.data.arrayBuffer());
      return;
    }
    if (typeof event.data === "string") {
      backendIngestSocket.send(normalizeIncomingWsText(event.data));
    }
  };

  remoteFrameSocket.onclose = () => {
    updateBridgeStatus("Удаленный WebSocket-источник отключился.", "small status-warn");
  };
  backendIngestSocket.onclose = () => {
    updateBridgeStatus("Backend ingest WebSocket закрыт.", "small status-warn");
  };
  updateBridgeStatus(`WS bridge активен: ${remoteUrl} -> ${ingestUrl}`, "small status-ok");
}

async function startCameraBridge(ingestUrl) {
  await cleanupRuntimeBridge();
  backendIngestSocket = await socketOpen(ingestUrl);
  cameraStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  cameraBridgeVideo = document.createElement("video");
  cameraBridgeVideo.muted = true;
  cameraBridgeVideo.playsInline = true;
  cameraBridgeVideo.srcObject = cameraStream;
  await cameraBridgeVideo.play();
  cameraBridgeCanvas = document.createElement("canvas");
  const startTs = performance.now();
  const pushFrame = () => {
    if (!backendIngestSocket || backendIngestSocket.readyState !== WebSocket.OPEN) {
      return;
    }
    const w = cameraBridgeVideo.videoWidth || 640;
    const h = cameraBridgeVideo.videoHeight || 360;
    cameraBridgeCanvas.width = w;
    cameraBridgeCanvas.height = h;
    const ctx = cameraBridgeCanvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(cameraBridgeVideo, 0, 0, w, h);
    const dataUrl = cameraBridgeCanvas.toDataURL("image/jpeg", 0.78);
    backendIngestSocket.send(JSON.stringify({
      image_base64: dataUrl,
      source_ts_ms: performance.now() - startTs,
    }));
    cameraBridgeTimer = requestAnimationFrame(pushFrame);
  };
  cameraBridgeTimer = requestAnimationFrame(pushFrame);

  backendIngestSocket.onclose = () => {
    updateBridgeStatus("Backend ingest WebSocket закрыт.", "small status-warn");
  };
  updateBridgeStatus(`WebRTC/camera bridge активен -> ${ingestUrl}`, "small status-ok");
}

async function startRuntimeBridge(type, sourceValue, ingestUrl) {
  if (!ingestUrl) {
    return;
  }
  if (type === "websocket") {
    if (sourceValue && /^wss?:\/\//i.test(sourceValue)) {
      await startRemoteWebSocketBridge(sourceValue, ingestUrl);
    } else {
      await cleanupRuntimeBridge();
      updateBridgeStatus(`Backend ingest URL готов: ${ingestUrl}`, "small status-ok");
    }
    return;
  }
  if (type === "webrtc") {
    await startCameraBridge(ingestUrl);
  }
}

async function startIndexBuild() {
  if (indexInFlight) return;
  indexInFlight = true;
  startIndexBtn.disabled = true;
  ids("indexError").textContent = "";

  try {
    const sourceType = indexSourceType.value;
    const points = pointsFile.files[0];
    const pointsPath = (indexPointsPath.value || "").trim();
    if (!points && !pointsPath) {
      throw new Error("Для индексации нужно указать путь к points.txt или загрузить файл.");
    }

    const form = new FormData();
    form.append("source_type", sourceType);
    form.append("source_value", currentSourceValue(indexSourceSelect, indexSourceInput));
    form.append("sample_fps", "0");
    form.append("max_keyframes", String(Number(indexMaxKeyframes.value || 0)));
    form.append("points_value", pointsPath);
    if (points) {
      form.append("points_file", points);
    }
    if (indexVideoFile.files[0]) {
      form.append("video_file", indexVideoFile.files[0]);
    }

    const res = await fetch("/api/v1/index/upload-and-build", {
      method: "POST",
      body: form,
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "Не удалось запустить индексацию.");
    }
    ids("indexStatus").textContent = "Индексация запущена.";
    await fetchIndexStatus();
  } finally {
    indexInFlight = false;
    startIndexBtn.disabled = false;
  }
}

async function startRuntime() {
  if (startInFlight) return;
  startInFlight = true;
  startBtn.disabled = true;
  resetTrackPlayback();
  const sourceType = runtimeSourceType.value;
  const source = currentSourceValue(sourceSelect, sourceInput);
  const uploadedVideo = runtimeVideoFile.files[0] || null;
  try {
    if (sourceType === "video" && uploadedVideo) {
      startStatus.textContent = `Подготовка runtime: загружаем файл ${uploadedVideo.name}. Это может занять некоторое время для больших видео.`;
    } else {
      startStatus.textContent = "Подготовка runtime...";
    }

    const payload = new FormData();
    payload.append("source_type", sourceType);
    payload.append("stream_url", source);
    payload.append("resize_width", String(Number(resizeWidth.value)));
    payload.append("top_k", String(Number(topK.value)));
    if (uploadedVideo) {
      payload.append("video_file", uploadedVideo);
    }
    const res = await fetch("/api/v1/runtime/start", {
      method: "POST",
      body: payload,
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "Не удалось запустить runtime.");
    }
    activeRuntimeSessionId = String(data.runtime_session_id || "");
    const shownSource = uploadedVideo?.name || shortSourceLabel(source) || shortSourceLabel(data.stream_url) || sourceType;
    startStatus.textContent = `${data.status === "already_running" ? "Уже запущено" : "Запущено"}: ${shownSource}`;
    if (sourceType === "websocket" || sourceType === "webrtc") {
      await startRuntimeBridge(sourceType, source, data.ingest_url || "");
    } else {
      await cleanupRuntimeBridge();
    }
    await fetchStatus();
  } finally {
    startInFlight = false;
    startBtn.disabled = false;
  }
}

async function stopRuntime() {
  await cleanupRuntimeBridge();
  await fetch("/api/v1/runtime/stop", { method: "POST" });
  startStatus.textContent = "Остановлено";
  resetTrackPlayback();
  await fetchStatus();
}

async function safePoll() {
  try {
    const [runtimeStatus, indexStatus] = await Promise.all([fetchStatus(), fetchIndexStatus()]);
    if (indexStatus.state === "completed") {
      await loadRoutes();
      await loadIndexSources();
      await loadRuntimeSources();
    }
    if (!runtimeStatus.running) {
      await cleanupRuntimeBridge();
    }
  } catch (e) {
    console.error(e);
  }
}

indexSourceSelect.addEventListener("change", () => syncInputFromSelect(indexSourceSelect, indexSourceInput));
sourceSelect.addEventListener("change", () => syncInputFromSelect(sourceSelect, sourceInput));
indexSourceType.addEventListener("change", applyIndexSourceTypeUi);
runtimeSourceType.addEventListener("change", applyRuntimeSourceTypeUi);
startIndexBtn.addEventListener("click", async () => {
  try {
    await startIndexBuild();
  } catch (e) {
    ids("indexError").textContent = e.message || String(e);
  }
});
startBtn.addEventListener("click", async () => {
  try {
    await startRuntime();
  } catch (e) {
    startStatus.textContent = e.message || String(e);
  }
});
stopBtn.addEventListener("click", stopRuntime);

map.on("zoomstart movestart", () => setMapInteractionActive(true));
map.on("zoomend moveend", () => setMapInteractionActive(false));

const es = new EventSource("/api/v1/localization/stream");
es.onmessage = (ev) => {
  try {
    const msg = JSON.parse(ev.data);
    if (!msg || typeof msg.lat !== "number" || typeof msg.lng !== "number") return;
    if (msg.confidence <= 0) return;
    if (activeRuntimeSessionId && msg.runtime_session_id !== activeRuntimeSessionId) return;
    updateMarker(msg);
  } catch (e) {
    console.error(e);
  }
};

async function init() {
  setMarkerVisible(false);
  applyIndexSourceTypeUi();
  applyRuntimeSourceTypeUi();
  await loadIndexSources();
  await loadRuntimeSources();
  await loadRoutes();
  await safePoll();
  setInterval(safePoll, 1500);
}

init();
