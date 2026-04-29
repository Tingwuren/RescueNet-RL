const stateByDoc = new WeakMap();

function createState() {
  return {
    traces: [],
    traceName: "",
    duration: 60,
    rttMs: 80,
    bufferPackets: 100,
    windowMs: 500,
    capacity: [],
    sendingRate: [],
    stats: null,
    running: false,
    finished: false,
    failed: false,
    latestArtifact: null,
    bootstrapped: false,
    capacitySeries: [],
    sendingRateSeries: [],
    playbackStartAt: 0,
    playbackTimer: null,
    streamCompleted: false,
    pendingStats: null,
  };
}

function byId(doc, id) {
  return doc.getElementById(id);
}

function textHolder(doc, id) {
  const node = byId(doc, id);
  if (!node) return null;
  return node.querySelector("span") || node;
}

function setText(doc, id, value) {
  const holder = textHolder(doc, id);
  if (holder) holder.textContent = value;
}

function formatNumber(value, digits = 2) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return "--";
  return parsed.toFixed(digits);
}

function formatDateTime(value) {
  if (!value) return "--";
  try {
    return new Date(Number(value) * 1000 || value).toLocaleString("zh-CN", { hour12: false });
  } catch {
    return "--";
  }
}

function algorithmLabel(key) {
  const mapping = {
    ppo: "PPO（基线）",
    dqn: "DQN（大动作空间）",
    a3c: "A3C（多目标）",
    mppo: "MPPO（多头策略）",
  };
  return mapping[String(key || "").toLowerCase()] || (key ? String(key).toUpperCase() : "PPO");
}

function disasterLabel(type) {
  const mapping = {
    flood: "洪涝孤岛通信恢复",
    earthquake: "地震灾后断链恢复",
    landslide: "泥石流滑坡通信阻断恢复",
    typhoon: "台风灾后残余网络",
  };
  return mapping[type] || "链路仿真";
}

function nowTimeLabel() {
  try {
    return new Date().toLocaleTimeString("zh-CN", { hour12: false });
  } catch {
    return "";
  }
}

function normalizePoints(items) {
  if (!Array.isArray(items)) return [];
  return items
    .map((item) => ({ time_s: Number(item?.time_s), value: Number(item?.value) }))
    .filter((item) => Number.isFinite(item.time_s) && Number.isFinite(item.value));
}

async function readErrorResponse(response) {
  const text = await response.text();
  if (!text) return `${response.status} ${response.statusText}`;
  try {
    const parsed = JSON.parse(text);
    return parsed?.detail || text;
  } catch {
    return text;
  }
}

function currentLastTime(state) {
  if (!state.capacity.length) return 0;
  return Number(state.capacity[state.capacity.length - 1].time_s) || 0;
}

function currentPlaybackElapsed(state) {
  if (!state.playbackStartAt) return 0;
  const elapsed = (Date.now() - state.playbackStartAt) / 1000;
  return Math.max(0, Math.min(state.duration || 0, elapsed));
}

function averageOf(items) {
  if (!items.length) return 0;
  return items.reduce((sum, item) => sum + (Number(item.value) || 0), 0) / items.length;
}

function setHeaderTitle(doc, state) {
  const artifact = state.latestArtifact;
  if (!artifact) {
    setText(doc, "u3874_text", "Mahimahi 链路仿真（真实联调）");
    return;
  }
  const scenarioPart = artifact.scenario_name || disasterLabel(artifact.disaster_type) || "链路仿真";
  setText(doc, "u3874_text", `${scenarioPart}  ${algorithmLabel(artifact.algorithm)} 链路仿真`);
}

function setStartButton(doc, mode) {
  const button = byId(doc, "u3888");
  const buttonDiv = byId(doc, "u3888_div");
  if (!button || !buttonDiv) return;

  const palette = {
    idle: { label: "启动", bg: "#03b4f5", shadow: "0 10px 22px rgba(3, 180, 245, 0.24)", cursor: "pointer", opacity: "1" },
    running: { label: "仿真中...", bg: "#1d4ed8", shadow: "0 12px 24px rgba(29, 78, 216, 0.28)", cursor: "not-allowed", opacity: "0.92" },
    success: { label: "重新仿真", bg: "#22c55e", shadow: "0 12px 24px rgba(34, 197, 94, 0.22)", cursor: "pointer", opacity: "1" },
    error: { label: "重试", bg: "#ef4444", shadow: "0 12px 24px rgba(239, 68, 68, 0.22)", cursor: "pointer", opacity: "1" },
    disabled: { label: "启动", bg: "#94a3b8", shadow: "none", cursor: "not-allowed", opacity: "0.6" },
  };
  const style = palette[mode] || palette.idle;
  setText(doc, "u3888_text", style.label);
  button.style.pointerEvents = mode === "running" || mode === "disabled" ? "none" : "auto";
  button.style.cursor = style.cursor;
  button.style.opacity = style.opacity;
  buttonDiv.style.background = style.bg;
  buttonDiv.style.borderColor = "transparent";
  buttonDiv.style.boxShadow = style.shadow;
}

function ensureSurface(doc) {
  const host = byId(doc, "u3889");
  if (!host) return null;
  const image = byId(doc, "u3889_img");
  if (image) image.style.display = "none";
  host.style.position = "absolute";
  host.style.pointerEvents = "auto";
  host.style.background = "transparent";

  let surface = byId(doc, "link-live-surface");
  if (!surface) {
    surface = doc.createElement("div");
    surface.id = "link-live-surface";
    surface.style.cssText = "position:absolute;inset:0;font-family:'Segoe UI','PingFang SC','Microsoft YaHei',sans-serif;color:#334155;pointer-events:auto;";
    surface.innerHTML = `
      <div id="link-live-chart-wrap" style="position:absolute;left:0;top:0;width:100%;height:596px;border-radius:12px;background:linear-gradient(180deg,#f4f7fb 0%,#eef3fa 100%);overflow:hidden;">
        <svg id="link-live-chart" style="display:block;width:100%;height:100%;"></svg>
        <div id="link-live-chart-empty" style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-size:15px;color:#94a3b8;">选择 Trace 并点击“启动”开始真实 Mahimahi 仿真</div>
      </div>
      <div id="link-live-progress" style="position:absolute;left:0;top:621px;width:100%;display:flex;align-items:center;gap:10px;">
        <div style="flex:1;height:4px;border-radius:999px;background:#d8dee9;overflow:hidden;"><div id="link-live-progress-fill" style="width:0%;height:100%;background:#22c55e;transition:width .12s linear;"></div></div>
        <div id="link-live-progress-text" style="min-width:82px;font-size:12px;color:#64748b;text-align:right;">0.0 / 60s</div>
      </div>
      <div id="link-live-summary" style="position:absolute;left:0;top:653px;display:flex;gap:42px;align-items:center;font-size:15px;color:#334155;">
        <div>平均容量 <strong id="link-metric-capacity" style="margin-left:6px;color:#475569;font-weight:700;">--</strong></div>
        <div>平均发送速率 <strong id="link-metric-sending" style="margin-left:6px;color:#475569;font-weight:700;">--</strong></div>
        <div>采样点 <strong id="link-metric-samples" style="margin-left:6px;color:#475569;font-weight:700;">0</strong></div>
      </div>
      <div id="link-live-terminal" style="position:absolute;left:0;top:688px;width:100%;height:278px;border-radius:12px;background:#0b0d10;overflow:hidden;box-shadow:0 18px 40px rgba(15,23,42,0.12);">
        <div style="height:34px;padding:0 14px;display:flex;align-items:center;gap:8px;background:#1f1f1f;">
          <span style="width:10px;height:10px;border-radius:50%;background:#ff5f57;"></span>
          <span style="width:10px;height:10px;border-radius:50%;background:#febc2e;"></span>
          <span style="width:10px;height:10px;border-radius:50%;background:#28c840;"></span>
          <span id="link-live-terminal-title" style="margin-left:8px;font-size:12px;color:#9ca3af;font-family:monospace;">mahimahi -- idle</span>
        </div>
        <div id="link-live-terminal-body" style="height:244px;overflow:auto;padding:12px 16px;text-align:left;font-family:'Cascadia Code','SFMono-Regular',Consolas,monospace;font-size:12px;line-height:1.75;color:#d7dde8;white-space:pre-wrap;"></div>
      </div>`;
    host.appendChild(surface);
  }

  return {
    chartWrap: byId(doc, "link-live-chart-wrap"),
    chart: byId(doc, "link-live-chart"),
    empty: byId(doc, "link-live-chart-empty"),
    progressFill: byId(doc, "link-live-progress-fill"),
    progressText: byId(doc, "link-live-progress-text"),
    metricCapacity: byId(doc, "link-metric-capacity"),
    metricSending: byId(doc, "link-metric-sending"),
    metricSamples: byId(doc, "link-metric-samples"),
    terminalTitle: byId(doc, "link-live-terminal-title"),
    terminalBody: byId(doc, "link-live-terminal-body"),
  };
}

function appendTerminalLine(doc, message, tone = "info") {
  const surface = ensureSurface(doc);
  if (!surface?.terminalBody) return;
  const line = doc.createElement("div");
  line.style.textAlign = "left";
  line.style.whiteSpace = "pre-wrap";
  line.style.wordBreak = "break-word";
  line.style.color = tone === "error" ? "#fca5a5" : tone === "success" ? "#86efac" : tone === "warning" ? "#fcd34d" : "#d7dde8";
  line.textContent = `[${nowTimeLabel()}] ${message}`;
  surface.terminalBody.appendChild(line);
  surface.terminalBody.scrollTop = surface.terminalBody.scrollHeight;
}

function resetTerminal(doc, traceName) {
  const surface = ensureSurface(doc);
  if (!surface) return;
  if (surface.terminalBody) surface.terminalBody.innerHTML = "";
  if (surface.terminalTitle) surface.terminalTitle.textContent = `mahimahi -- ${traceName || "idle"}`;
}

function updateProgress(doc, state) {
  const surface = ensureSurface(doc);
  if (!surface) return;
  const current = state.running ? currentPlaybackElapsed(state) : currentLastTime(state);
  const percent = Math.max(0, Math.min(100, state.duration > 0 ? (current / state.duration) * 100 : 0));
  if (surface.progressFill) {
    surface.progressFill.style.width = `${percent.toFixed(2)}%`;
    surface.progressFill.style.background = state.failed ? "#ef4444" : "#22c55e";
  }
  if (surface.progressText) {
    surface.progressText.textContent = `${formatNumber(current, 1)} / ${formatNumber(state.duration, 0)}s`;
  }
}

function updateSummary(doc, state) {
  const surface = ensureSurface(doc);
  if (!surface) return;
  const avgCapacity = Number.isFinite(Number(state.pendingStats?.avg_capacity_mbps))
    ? Number(state.pendingStats.avg_capacity_mbps)
    : averageOf(state.capacitySeries.length ? state.capacitySeries : state.capacity);
  const avgSending = Number.isFinite(Number(state.pendingStats?.avg_sending_rate_mbps))
    ? Number(state.pendingStats.avg_sending_rate_mbps)
    : averageOf(state.sendingRateSeries.length ? state.sendingRateSeries : state.sendingRate);
  if (surface.metricCapacity) surface.metricCapacity.textContent = `${formatNumber(avgCapacity, 1)} Mbps`;
  if (surface.metricSending) surface.metricSending.textContent = `${formatNumber(avgSending, 1)} Mbps`;
  if (surface.metricSamples) surface.metricSamples.textContent = String(state.capacitySeries.length || state.sendingRateSeries.length || state.capacity.length || state.sendingRate.length || 0);
}

function syncVisibleSeries(doc, state, elapsed) {
  state.capacity = state.capacitySeries.filter((item) => item.time_s <= elapsed + 1e-6);
  state.sendingRate = state.sendingRateSeries.filter((item) => item.time_s <= elapsed + 1e-6);
  renderChart(doc, state);
}

function stopPlaybackLoop(view) {
  if (!view) return;
  if (typeof view.cancelAnimationFrame === "function") {
    view.cancelAnimationFrame(view.__linkPlaybackTimer || 0);
    view.__linkPlaybackTimer = 0;
  }
}

function finalizePlayback(doc, state) {
  state.running = false;
  state.finished = true;
  state.stats = state.pendingStats;
  state.capacity = state.capacitySeries.slice();
  state.sendingRate = state.sendingRateSeries.slice();
  renderChart(doc, state);
  updateProgress(doc, state);
  updateSummary(doc, state);
  setStartButton(doc, state.failed ? "error" : "success");
}

function startPlaybackLoop(doc, state) {
  const view = doc.defaultView || window;
  stopPlaybackLoop(view);
  state.playbackStartAt = Date.now();

  const step = () => {
    const elapsed = currentPlaybackElapsed(state);
    syncVisibleSeries(doc, state, elapsed);
    updateProgress(doc, state);

    if (state.failed) {
      state.running = false;
      setStartButton(doc, "error");
      return;
    }

    if (elapsed >= state.duration && state.streamCompleted) {
      finalizePlayback(doc, state);
      return;
    }

    view.__linkPlaybackTimer = view.requestAnimationFrame(step);
  };

  view.__linkPlaybackTimer = view.requestAnimationFrame(step);
}

function renderChart(doc, state) {
  const surface = ensureSurface(doc);
  if (!surface?.chartWrap || !surface.chart) return;
  const width = Math.max(300, Math.round(surface.chartWrap.clientWidth || 1582));
  const height = Math.max(240, Math.round(surface.chartWrap.clientHeight || 596));
  const paddingTop = 32;
  const paddingRight = 36;
  const paddingBottom = 42;
  const paddingLeft = 78;
  const plotWidth = Math.max(1, width - paddingLeft - paddingRight);
  const plotHeight = Math.max(1, height - paddingTop - paddingBottom);
  const capacity = state.capacity.slice();
  const sending = state.sendingRate.slice();
  const pointCount = Math.max(capacity.length, sending.length);

  if (!pointCount) {
    surface.chart.setAttribute("viewBox", `0 0 ${width} ${height}`);
    surface.chart.innerHTML = "";
    if (surface.empty) surface.empty.style.display = "flex";
    updateProgress(doc, state);
    updateSummary(doc, state);
    return;
  }

  if (surface.empty) surface.empty.style.display = "none";
  const xMax = Math.max(state.duration || 1, currentLastTime(state) || 0.001);
  const yMax = Math.max(1, ...capacity.map((item) => Number(item.value) || 0), ...sending.map((item) => Number(item.value) || 0)) * 1.12;
  const tx = (time) => paddingLeft + (Number(time) / xMax) * plotWidth;
  const ty = (value) => paddingTop + (1 - Number(value) / yMax) * plotHeight;
  const toPath = (points) => points.map((item, index) => `${index ? "L" : "M"}${tx(item.time_s).toFixed(2)} ${ty(item.value).toFixed(2)}`).join(" ");
  const toArea = (points) => {
    if (!points.length) return "";
    const baseline = paddingTop + plotHeight;
    return `${toPath(points)} L${tx(points[points.length - 1].time_s).toFixed(2)} ${baseline.toFixed(2)} L${tx(points[0].time_s).toFixed(2)} ${baseline.toFixed(2)} Z`;
  };

  const elements = [
    '<defs><linearGradient id="link-send-fill" x1="0" x2="0" y1="0" y2="1"><stop offset="0%" stop-color="rgba(56,189,248,0.20)"/><stop offset="100%" stop-color="rgba(56,189,248,0.04)"/></linearGradient></defs>'
  ];

  for (let yIndex = 0; yIndex <= 5; yIndex += 1) {
    const yValue = (yMax / 5) * yIndex;
    const y = ty(yValue);
    elements.push(`<line x1="${paddingLeft}" y1="${y.toFixed(2)}" x2="${width - paddingRight}" y2="${y.toFixed(2)}" stroke="rgba(148,163,184,0.16)" stroke-width="1"/>`);
    elements.push(`<text x="${paddingLeft - 12}" y="${(y + 4).toFixed(2)}" fill="#64748b" font-size="12" text-anchor="end">${formatNumber(yValue, 1)}</text>`);
  }

  for (let xIndex = 0; xIndex <= 6; xIndex += 1) {
    const xValue = (xMax / 6) * xIndex;
    const x = tx(xValue);
    elements.push(`<text x="${x.toFixed(2)}" y="${height - 10}" fill="#64748b" font-size="12" text-anchor="middle">${formatNumber(xValue, 0)}s</text>`);
  }

  if (sending.length) elements.push(`<path d="${toArea(sending)}" fill="url(#link-send-fill)"/>`);
  if (capacity.length) elements.push(`<path d="${toPath(capacity)}" fill="none" stroke="#64748b" stroke-width="2" stroke-dasharray="8 6" stroke-linecap="round" stroke-linejoin="round"/>`);
  if (sending.length) {
    elements.push(`<path d="${toPath(sending)}" fill="none" stroke="#38bdf8" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>`);
    const lastPoint = sending[sending.length - 1];
    elements.push(`<circle cx="${tx(lastPoint.time_s).toFixed(2)}" cy="${ty(lastPoint.value).toFixed(2)}" r="4.5" fill="#38bdf8" stroke="#ffffff" stroke-width="2"/>`);
  }

  elements.push('<line x1="96" y1="20" x2="140" y2="20" stroke="#64748b" stroke-width="2" stroke-dasharray="8 6"/>');
  elements.push('<text x="150" y="24" fill="#64748b" font-size="14">链路容量</text>');
  elements.push('<line x1="288" y1="20" x2="332" y2="20" stroke="#38bdf8" stroke-width="3"/>');
  elements.push('<text x="342" y="24" fill="#38bdf8" font-size="14">发送速率</text>');

  surface.chart.setAttribute("viewBox", `0 0 ${width} ${height}`);
  surface.chart.innerHTML = elements.join("");
  updateProgress(doc, state);
  updateSummary(doc, state);
}

function ensureTraceControl(doc, state) {
  const host = byId(doc, "u3883");
  if (!host) return null;
  host.style.position = "relative";
  host.style.pointerEvents = "auto";
  const textNode = byId(doc, "u3883_text");
  if (textNode) textNode.style.visibility = "hidden";

  let select = byId(doc, "link-trace-select");
  if (!select) {
    select = doc.createElement("select");
    select.id = "link-trace-select";
    select.style.cssText = "position:absolute;inset:0;z-index:3;width:100%;height:100%;padding:0 42px 0 10px;border:0;background:transparent;appearance:none;-webkit-appearance:none;font-family:'思源黑体 CN','PingFang SC',sans-serif;font-size:16px;color:#475569;outline:none;cursor:pointer;";
    select.addEventListener("change", (event) => {
      state.traceName = event.target.value;
      const selected = state.traces.find((item) => item.name === state.traceName);
      if (selected) appendTerminalLine(doc, `已切换 Trace：${selected.label || selected.name}`, "info");
    });
    host.appendChild(select);
  }
  return select;
}

function ensureDurationControl(doc, state) {
  const host = byId(doc, "u3886");
  if (!host) return null;
  host.style.position = "relative";
  host.style.pointerEvents = "auto";
  const textNode = byId(doc, "u3886_text");
  if (textNode) textNode.style.visibility = "hidden";

  let input = byId(doc, "link-duration-input");
  if (!input) {
    input = doc.createElement("input");
    input.id = "link-duration-input";
    input.type = "number";
    input.min = "1";
    input.max = "300";
    input.step = "1";
    input.style.cssText = "position:absolute;inset:0;z-index:3;width:100%;height:100%;padding:0 42px 0 10px;border:0;background:transparent;font-family:'思源黑体 CN','PingFang SC',sans-serif;font-size:16px;color:#475569;outline:none;";
    input.addEventListener("input", (event) => {
      const nextValue = Number(event.target.value);
      state.duration = Number.isFinite(nextValue) && nextValue > 0 ? Math.min(300, Math.max(1, nextValue)) : 60;
      updateProgress(doc, state);
      renderChart(doc, state);
    });
    host.appendChild(input);
  }
  return input;
}

function populateControls(doc, state) {
  const traceSelect = ensureTraceControl(doc, state);
  const durationInput = ensureDurationControl(doc, state);
  if (!traceSelect || !durationInput) return;

  traceSelect.innerHTML = "";
  state.traces.forEach((trace) => {
    const option = doc.createElement("option");
    option.value = trace.name;
    option.textContent = trace.label || trace.name;
    traceSelect.appendChild(option);
  });

  if (state.traceName && state.traces.some((trace) => trace.name === state.traceName)) {
    traceSelect.value = state.traceName;
  } else if (state.traces[0]) {
    state.traceName = state.traces[0].name;
    traceSelect.value = state.traceName;
  } else {
    state.traceName = "";
  }

  durationInput.value = String(state.duration || 60);
  setStartButton(doc, state.traceName ? "idle" : "disabled");
}

function handleChunk(doc, state, payload) {
  const capacity = normalizePoints(payload?.capacity);
  const sending = normalizePoints(payload?.sending_rate);
  state.capacitySeries.push(...capacity);
  state.sendingRateSeries.push(...sending);
  if (!state.playbackStartAt) {
    startPlaybackLoop(doc, state);
  }
  syncVisibleSeries(doc, state, currentPlaybackElapsed(state));

  const lastCapacity = capacity.length ? capacity[capacity.length - 1] : null;
  const lastSending = sending.length ? sending[sending.length - 1] : null;
  if (lastCapacity || lastSending) {
    appendTerminalLine(doc, `[chunk] t=${formatNumber(lastCapacity ? lastCapacity.time_s : lastSending.time_s, 1)}s  capacity=${formatNumber(lastCapacity ? lastCapacity.value : 0, 2)} Mbps  send_rate=${formatNumber(lastSending ? lastSending.value : 0, 2)} Mbps`, "info");
  }
}

function handleStreamEvent(doc, state, event) {
  if (!event || typeof event !== "object") return;
  if (event.type === "status") return appendTerminalLine(doc, `状态：${event.payload?.state || "unknown"}`, "info");
  if (event.type === "log") return appendTerminalLine(doc, event.payload?.message || "收到日志事件", "info");
  if (event.type === "data_chunk") return handleChunk(doc, state, event.payload || {});
  if (event.type === "result") {
    state.pendingStats = event.payload?.stats || null;
    updateSummary(doc, state);
    return appendTerminalLine(doc, `[done] avg_capacity=${formatNumber(state.pendingStats?.avg_capacity_mbps, 2)} Mbps  avg_send_rate=${formatNumber(state.pendingStats?.avg_sending_rate_mbps, 2)} Mbps`, "success");
  }
  if (event.type === "error") {
    state.failed = true;
    return appendTerminalLine(doc, event.payload?.message || "仿真失败", "error");
  }
  if (event.type === "end") {
    if (event.payload?.state === "failed") state.failed = true;
    if (event.payload?.state === "completed") state.streamCompleted = true;
  }
}

async function consumeSseStream(doc, state, response) {
  const reader = response.body?.getReader?.();
  if (!reader) throw new Error("浏览器不支持流式响应读取。");
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  const flushChunk = (rawChunk) => {
    if (!rawChunk) return;
    const payloadLines = rawChunk.split("\n").filter((line) => line.indexOf("data:") === 0).map((line) => line.slice(5).trim());
    if (!payloadLines.length) return;
    try {
      handleStreamEvent(doc, state, JSON.parse(payloadLines.join("\n")));
    } catch (error) {
      appendTerminalLine(doc, `解析流式事件失败：${error?.message || error}`, "error");
    }
  };

  while (true) {
    const result = await reader.read();
    buffer += decoder.decode(result.value || new Uint8Array(), { stream: !result.done });
    const segments = buffer.split("\n\n");
    buffer = segments.pop() || "";
    segments.forEach(flushChunk);
    if (result.done) {
      if (buffer.trim()) flushChunk(buffer);
      break;
    }
  }
}

async function startSimulation(doc, state) {
  if (state.running || !state.traceName) return;
  state.running = true;
  state.finished = false;
  state.failed = false;
  state.stats = null;
  state.pendingStats = null;
  state.capacity = [];
  state.sendingRate = [];
  state.capacitySeries = [];
  state.sendingRateSeries = [];
  state.streamCompleted = false;
  state.playbackStartAt = 0;
  stopPlaybackLoop(doc.defaultView || window);
  resetTerminal(doc, state.traceName);
  renderChart(doc, state);
  updateProgress(doc, state);
  updateSummary(doc, state);
  setStartButton(doc, "running");

  const selected = state.traces.find((trace) => trace.name === state.traceName);
  appendTerminalLine(doc, "链路仿真页已切换为真实 Mahimahi 联调模式。", "success");
  appendTerminalLine(doc, `$ mm-link /app/data/traces/${state.traceName}.trace /app/data/traces/${state.traceName}.trace`, "info");
  appendTerminalLine(doc, `加载 trace: ${(selected && (selected.label || selected.name)) || state.traceName}，时长=${formatNumber(state.duration, 0)}s，RTT=${state.rttMs}ms`, "info");

  try {
    const response = await fetch(`/api/mahimahi/simulate/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        trace_name: state.traceName,
        duration_s: state.duration,
        rtt_ms: state.rttMs,
        buffer_packets: state.bufferPackets,
        window_ms: state.windowMs,
      }),
    });
    if (!response.ok) throw new Error(await readErrorResponse(response));
    await consumeSseStream(doc, state, response);
    if (!state.capacitySeries.length && !state.failed) {
      state.streamCompleted = true;
      finalizePlayback(doc, state);
    }
  } catch (error) {
    state.failed = true;
    appendTerminalLine(doc, `启动链路仿真失败：${error?.message || error}`, "error");
  } finally {
    if (state.failed) {
      state.running = false;
      setStartButton(doc, "error");
    }
  }
}

function bindStartButton(doc, state) {
  const button = byId(doc, "u3888");
  if (!button || button.dataset.liveStartBound) return;
  button.dataset.liveStartBound = "true";
  button.style.pointerEvents = "auto";
  button.addEventListener("click", () => {
    void startSimulation(doc, state);
  });
}

function bindContextButton(doc, state) {
  const button = byId(doc, "u3891");
  if (!button || button.dataset.liveContextBound) return;
  button.dataset.liveContextBound = "true";
  button.style.pointerEvents = "auto";
  button.style.cursor = "pointer";
  setText(doc, "u3891_text", "同步最新策略");
  button.addEventListener("click", () => {
    void loadLatestArtifact(doc, state, true);
  });
}

async function loadLatestArtifact(doc, state, verbose) {
  try {
    const response = await fetch(`/api/train/latest-artifact`);
    if (!response.ok) return;
    state.latestArtifact = await response.json();
    setHeaderTitle(doc, state);
    if (verbose) {
      appendTerminalLine(doc, `已同步最新策略：${state.latestArtifact.scenario_name || "未命名场景"} / ${algorithmLabel(state.latestArtifact.algorithm)}`, "success");
      appendTerminalLine(doc, `最近更新时间：${formatDateTime(state.latestArtifact.updated_at)}`, "info");
    }
  } catch (error) {
    if (verbose) appendTerminalLine(doc, `同步最新策略失败：${error?.message || error}`, "warning");
  }
}

async function loadTraces(doc, state) {
  try {
    const response = await fetch(`/api/mahimahi/traces`);
    if (!response.ok) throw new Error(await readErrorResponse(response));
    const payload = await response.json();
    state.traces = Array.isArray(payload.traces) ? payload.traces : [];
    populateControls(doc, state);
    appendTerminalLine(doc, `Mahimahi trace 列表已加载，共 ${state.traces.length} 条。`, "success");
    if (state.traces[0]) appendTerminalLine(doc, `默认 Trace：${state.traces[0].label || state.traces[0].name}`, "info");
  } catch (error) {
    setStartButton(doc, "disabled");
    appendTerminalLine(doc, `加载 trace 列表失败：${error?.message || error}`, "error");
  }
}

export function injectPrototypeLink(doc) {
  if (!doc) return;
  let state = stateByDoc.get(doc);
  if (!state) {
    state = createState();
    stateByDoc.set(doc, state);
  }

  if (!byId(doc, "u3889") || !byId(doc, "u3883") || !byId(doc, "u3886") || !byId(doc, "u3888")) {
    return;
  }

  ensureSurface(doc);
  ensureTraceControl(doc, state);
  ensureDurationControl(doc, state);
  bindStartButton(doc, state);
  bindContextButton(doc, state);
  setHeaderTitle(doc, state);
  renderChart(doc, state);
  updateProgress(doc, state);
  updateSummary(doc, state);

  if (!state.bootstrapped) {
    state.bootstrapped = true;
    resetTerminal(doc, "");
    appendTerminalLine(doc, "链路仿真页已切换为真实 Mahimahi 联调模式，正在同步 trace 列表。", "info");
    void loadTraces(doc, state);
    void loadLatestArtifact(doc, state, false);
  }
}
