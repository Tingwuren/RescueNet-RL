<template>
  <div class="link-simulation">
    <img class="link-simulation__bg" :src="assetUrl('images/模型训练/u537.png')" alt="" />
    <img class="link-simulation__panel-shadow" :src="assetUrl('images/模型训练/u538.png')" alt="" />

    <main class="link-panel" aria-label="链路仿真">
      <div class="link-panel__scroll">
        <div class="link-title">
          <img class="link-title__ribbon" :src="assetUrl('images/模型训练/u541.png')" alt="" />
          <h1>链路仿真</h1>
        </div>

        <transition name="status-fade">
          <div v-if="statusMessage" :class="['status-toast', `status-toast--${statusTone}`]">
            {{ statusMessage }}
          </div>
        </transition>

        <section class="link-workbench">
          <header class="module-heading module-heading--top">
            <div>
              <i></i>
              <h2>{{ pageHeading }}</h2>
              <p>{{ headingDescription }}</p>
            </div>
            <div class="heading-actions">
              <button type="button" class="ghost-button" :disabled="isLoadingSessions" @click="refreshAll">
                {{ isLoadingSessions ? "刷新中..." : "刷新" }}
              </button>
              <button type="button" class="ghost-button ghost-button--blue" :disabled="!activeSessionId" @click="openReplay">
                场景回放
              </button>
            </div>
          </header>

          <div v-if="errorMessage" class="module-error">{{ errorMessage }}</div>

          <section v-if="showLinkResults" class="status-grid" aria-label="链路摘要">
            <article v-for="item in summaryCards" :key="item.label" class="summary-card">
              <small>{{ item.label }}</small>
              <strong>{{ item.value }}</strong>
              <span>{{ item.description }}</span>
            </article>
          </section>

          <section class="control-card">
            <div :class="['control-row', { 'control-row--idle': !simulationStarted }]">
              <label class="control-field control-field--wide">
                <span>回放记录</span>
                <select v-model="selectedSessionForControl" :disabled="isLoadingSessions || !sessions.length" @change="selectSession(selectedSessionForControl)">
                  <option v-if="!sessions.length" value="">暂无回放记录</option>
                  <option v-for="session in sessions" :key="session.id" :value="session.id">
                    {{ session.title }}
                  </option>
                </select>
              </label>

              <label v-if="simulationStarted" class="control-field">
                <span>当前阶段</span>
                <input :value="currentPhaseLabel" type="text" readonly />
              </label>

              <label v-if="simulationStarted" class="control-field control-field--frame">
                <span>帧序号</span>
                <input
                  v-model.number="currentFrameIndex"
                  type="number"
                  min="0"
                  :max="maxFrameIndex"
                  :disabled="!showLinkResults"
                />
              </label>

              <div class="control-actions">
                <button type="button" class="primary-button" :disabled="!activeSessionId || isLoadingMetrics" @click="startSimulation">
                  {{ startButtonLabel }}
                </button>
              </div>
            </div>

            <div v-if="showLinkResults" class="timeline-row">
              <div class="timeline-meta">
                <strong>Frame {{ currentFrameIndex + 1 }} / {{ displayFrameCount }}</strong>
                <span>{{ currentTimeLabel }}</span>
              </div>
              <input
                v-model.number="currentFrameIndex"
                class="timeline-range"
                type="range"
                min="0"
                :max="maxFrameIndex"
                step="1"
                :disabled="!metricsSeries.length"
              />
              <div class="timeline-phases" aria-hidden="true">
                <span v-for="phase in phaseMarkers" :key="phase.key" :style="{ left: `${phase.left}%` }">
                  {{ phase.label }}
                </span>
              </div>
            </div>
          </section>

          <section v-if="showLinkResults" class="link-group-grid" aria-label="三类关键链路">
            <article v-for="group in linkGroupCards" :key="group.id" :class="['link-group-card', `link-group-card--${group.statusTone}`]">
              <header>
                <div>
                  <span class="group-dot" :style="{ background: group.color }"></span>
                  <h3>{{ group.title }}</h3>
                </div>
                <strong>{{ group.statusLabel }}</strong>
              </header>
              <p>{{ group.description }}</p>
              <div class="group-metrics">
                <span>
                  <small>当前吞吐</small>
                  <b>{{ formatMbps(group.throughput) }}</b>
                  <em :class="group.delta >= 0 ? 'is-up' : 'is-down'">{{ signedMbps(group.delta) }}</em>
                </span>
                <span>
                  <small>利用率</small>
                  <b>{{ formatPercent(group.utilization) }}</b>
                  <i><span :style="{ width: `${Math.round(clamp01(group.utilization) * 100)}%`, background: group.color }"></span></i>
                </span>
                <span>
                  <small>成功率</small>
                  <b>{{ formatPercent(group.successRate) }}</b>
                  <em>{{ group.successHint }}</em>
                </span>
                <span>
                  <small>服务规模</small>
                  <b>{{ formatNumber(group.coveredUsers || group.activeLinks) }}</b>
                  <em>{{ group.scaleLabel }}</em>
                </span>
              </div>
              <div class="group-event">
                <small>当前事件</small>
                <span>{{ group.eventText }}</span>
              </div>
            </article>
          </section>

          <section v-if="showLinkResults" class="trend-grid" aria-label="链路趋势图">
            <article v-for="chart in chartModels" :key="chart.key" class="trend-card">
              <header>
                <div>
                  <h3>{{ chart.label }}</h3>
                  <p>{{ chart.description }}</p>
                </div>
                <span>{{ chart.currentLabel }}</span>
              </header>

              <svg :viewBox="`0 0 ${chart.width} ${chart.height}`" preserveAspectRatio="none" role="img" :aria-label="chart.label">
                <line
                  v-for="tick in chart.yTicks"
                  :key="`y-${tick.value}`"
                  :x1="chart.padding.left"
                  :x2="chart.width - chart.padding.right"
                  :y1="tick.y"
                  :y2="tick.y"
                  class="chart-grid-line"
                />
                <line
                  v-for="tick in chart.xTicks"
                  :key="`x-${tick.value}`"
                  :x1="tick.x"
                  :x2="tick.x"
                  :y1="chart.padding.top"
                  :y2="chart.height - chart.padding.bottom"
                  class="chart-grid-line chart-grid-line--vertical"
                />
                <path
                  v-for="series in chart.series"
                  :key="series.id"
                  :d="series.area"
                  :fill="series.fill"
                  opacity="0.12"
                />
                <path
                  v-for="series in chart.series"
                  :key="`${series.id}-line`"
                  :d="series.path"
                  fill="none"
                  :stroke="series.color"
                  stroke-width="2.6"
                  stroke-linecap="round"
                  stroke-linejoin="round"
                />
                <line
                  v-if="chart.currentX"
                  :x1="chart.currentX"
                  :x2="chart.currentX"
                  :y1="chart.padding.top"
                  :y2="chart.height - chart.padding.bottom"
                  class="chart-cursor"
                />
                <circle
                  v-for="point in chart.currentPoints"
                  :key="`${chart.key}-${point.id}`"
                  :cx="point.x"
                  :cy="point.y"
                  r="4.5"
                  :fill="point.color"
                  stroke="#ffffff"
                  stroke-width="2"
                />
                <text
                  v-for="tick in chart.yTicks"
                  :key="`label-y-${tick.value}`"
                  :x="chart.padding.left - 8"
                  :y="tick.y + 4"
                  text-anchor="end"
                  class="chart-axis-text"
                >
                  {{ tick.label }}
                </text>
                <text
                  v-for="tick in chart.xTicks"
                  :key="`label-x-${tick.value}`"
                  :x="tick.x"
                  :y="chart.height - 8"
                  text-anchor="middle"
                  class="chart-axis-text"
                >
                  {{ tick.label }}
                </text>
              </svg>

              <div class="chart-legend">
                <span v-for="series in chart.series" :key="series.id">
                  <i :style="{ background: series.color }"></i>{{ series.label }}
                </span>
              </div>
            </article>
          </section>

          <section class="terminal-section">
            <StreamingTerminal
              title="实时终端输出"
              subtitle="持续输出后端日志、链路指标加载和帧回放进度。"
              :lines="linkTerminalLines"
              :status="terminalStatus"
              placeholder="选择回放记录后点击“开始仿真”，终端将在这里实时输出。"
              exportable
              clearable
              @export="downloadTerminalLog"
              @clear="clearTerminalLog"
            />
          </section>
        </section>
      </div>
    </main>
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import axios from "axios";

import StreamingTerminal from "./StreamingTerminal.vue";
import { rescueApiBase } from "../utils/runtimeEndpoints";
import {
  appendSyncedTerminalLine,
  clearTerminalOutput,
  exportTerminalOutput,
  terminalHistoryLines,
} from "../utils/terminalOutput";
import {
  getActiveReplaySessionId,
  setActiveReplaySessionId,
} from "../utils/replaySessions";
import {
  buildUserNodeCountMessage,
  userNodeCountLogKey,
} from "../utils/scenarioNodeMetrics";

const API_BASE = rescueApiBase;

const LINK_GROUPS = [
  {
    id: "user_access",
    title: "现场设备到用户/终端链路",
    shortLabel: "用户接入",
    description: "无人机基站、车载基站、便携基站和卫星终端到受灾用户的接入能力。",
    color: "#1890ff",
  },
  {
    id: "residual_network",
    title: "残余网络链路",
    shortLabel: "残余网络",
    description: "灾后仍可用宏基站、微基站、专网基站和应急残余站点的承载能力。",
    color: "#13c2c2",
  },
  {
    id: "backhaul",
    title: "灾区到外界回程链路",
    shortLabel: "回程链路",
    description: "灾区内部网络到外部指挥中心、核心网、卫星回程或公网出口的连接能力。",
    color: "#722ed1",
  },
];

const STATUS_LABELS = {
  normal: "正常",
  active: "正常",
  degraded: "受损",
  congested: "拥塞",
  recovering: "恢复中",
  recovery: "恢复中",
  deploying: "部署中",
  offline: "离线",
  finished: "已完成",
};

const PHASE_LABELS = {
  initial: "初始",
  damaged: "受损",
  deploying: "部署中",
  recovery: "恢复",
  finished: "完成",
};

const sessions = ref([]);
const selectedSessionForControl = ref("");
const activeSessionId = ref("");
const activeSessionDetail = ref(null);
const metricsSeries = ref([]);
const logs = ref([]);
const currentFrameIndex = ref(0);
const isLoadingSessions = ref(false);
const isLoadingMetrics = ref(false);
const isPlaying = ref(false);
const simulationStarted = ref(false);
const terminalLines = ref([]);
const terminalStatus = ref("idle");
const linkTerminalLines = computed(() => terminalHistoryLines.value.slice(-500));
const errorMessage = ref("");
const statusMessage = ref("");
const statusTone = ref("success");

let playTimer = null;
let logPollTimer = null;
let statusTimer = null;
let lastSyncedLogCount = 0;
let lastTerminalFrameIndex = -1;
let lastLinkUserNodeLogKey = "";

const assetUrl = (path) => `${import.meta.env.BASE_URL}prototype/${path}`;

const clamp01 = (value) => Math.max(0, Math.min(1, Number(value || 0)));
const finiteNumber = (value, fallback = 0) => {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
};

const formatNumber = (value) => finiteNumber(value, 0).toLocaleString("zh-CN");
const formatMbps = (value) => `${finiteNumber(value, 0).toFixed(2)} Mbps`;
const signedMbps = (value) => {
  const parsed = finiteNumber(value, 0);
  const prefix = parsed >= 0 ? "+" : "";
  return `${prefix}${parsed.toFixed(2)} Mbps`;
};
const formatPercent = (value) => `${(clamp01(value) * 100).toFixed(1)}%`;
const formatLatency = (value) => `${finiteNumber(value, 0).toFixed(1)} ms`;

const formatTime = (value) => {
  const seconds = finiteNumber(value, 0);
  const minute = Math.floor(seconds / 60);
  const second = Math.floor(seconds % 60);
  return `${String(minute).padStart(2, "0")}:${String(second).padStart(2, "0")}`;
};

const appendTerminalLine = (message, options = {}) => {
  if (!message) return;
  terminalLines.value = appendSyncedTerminalLine(
    terminalLines.value,
    message,
    { level: options.level || "INFO", source: options.source || "LINK", timestamp: options.timestamp },
    500
  );
};

const appendLinkUserNodeCount = (prefix, ...sources) => {
  const key = userNodeCountLogKey(`link:${activeSessionId.value || ""}:${prefix}`, ...sources);
  if (key === lastLinkUserNodeLogKey) return;
  lastLinkUserNodeLogKey = key;
  appendTerminalLine(buildUserNodeCountMessage(prefix, ...sources), { level: "SCENE" });
};

const downloadTerminalLog = () => {
  exportTerminalOutput(terminalHistoryLines.value, "rescuenet-link-terminal.log");
};

const clearTerminalLog = () => {
  terminalLines.value = [];
  clearTerminalOutput();
};

const formatDateTime = (value) => {
  const timestamp = finiteNumber(value, 0);
  if (!timestamp) return "--";
  return new Date(timestamp * 1000).toLocaleString("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
};

const showStatus = (message, tone = "success", duration = 2400) => {
  statusMessage.value = message;
  statusTone.value = tone;
  if (statusTimer) window.clearTimeout(statusTimer);
  if (duration > 0) {
    statusTimer = window.setTimeout(() => {
      statusMessage.value = "";
    }, duration);
  }
};

const hashQuery = () => {
  if (typeof window === "undefined") return {};
  const queryText = window.location.hash.includes("?") ? window.location.hash.split("?").slice(1).join("?") : "";
  const params = new URLSearchParams(queryText);
  return {
    replayId: params.get("replay_id") || "",
    frameIndex: params.get("frame_index") == null ? null : Number(params.get("frame_index")),
  };
};

const normalizeSession = (session) => {
  const summary = session?.summary || {};
  return {
    ...session,
    id: session?.replay_id || session?.id || "",
    title: session?.title || session?.replay_id || session?.id || "未命名回放",
    source: session?.source || "test",
    createdAt: finiteNumber(session?.created_at || session?.createdAt, 0),
    scenarioName: session?.scenario_name || session?.scenarioName || "",
    algorithm: session?.algorithm || "--",
    frameCount: finiteNumber(session?.frame_count || session?.frameCount, 0),
    nodeCountTotal: finiteNumber(session?.node_count_total || session?.nodeCountTotal, 0),
    summary: {
      totalReward: finiteNumber(summary.total_reward ?? summary.totalReward, 0),
      coverageRatio: finiteNumber(summary.coverage_ratio ?? summary.coverageRatio, 0),
      broadcastRatio: finiteNumber(summary.broadcast_ratio ?? summary.broadcastRatio, 0),
      stepsTaken: finiteNumber(summary.steps_taken ?? summary.stepsTaken, 0),
      totalUsers: finiteNumber(summary.total_users ?? summary.totalUsers, 0),
      initialStations: finiteNumber(summary.initial_stations ?? summary.initialStations, 0),
      finalStations: finiteNumber(summary.final_stations ?? summary.finalStations, 0),
      connectedUsers: finiteNumber(summary.connected_users ?? summary.connectedUsers, 0),
      broadcastUsers: finiteNumber(summary.broadcast_users ?? summary.broadcastUsers, 0),
    },
  };
};

const groupIdFromRaw = (group) => {
  const raw = String(group?.group_id || group?.link_type || group?.id || "").trim().toLowerCase();
  if (raw === "broadcast" || raw.includes("residual") || raw.includes("broadcast")) return "residual_network";
  if (raw.includes("backhaul") || raw.includes("回程") || raw.includes("回传")) return "backhaul";
  return raw || "user_access";
};

const phaseForFrame = (frame, index, total) => {
  if (frame?.phase) return String(frame.phase);
  if (index <= 0) return "initial";
  const ratio = total <= 1 ? 1 : index / Math.max(1, total - 1);
  if (ratio < 0.18) return "damaged";
  if (ratio < 0.62) return "deploying";
  if (ratio < 0.95) return "recovery";
  return "finished";
};

const rawGroupValue = (group, keys, fallback = 0) => {
  for (const key of keys) {
    if (group?.[key] != null && group?.[key] !== "") return finiteNumber(group[key], fallback);
  }
  return fallback;
};

const normalizeRawGroup = (group) => {
  const id = groupIdFromRaw(group);
  const packetLoss = rawGroupValue(group, ["packet_loss_ratio", "loss_ratio"], null);
  const successRate =
    group?.success_rate != null
      ? finiteNumber(group.success_rate, 0)
      : packetLoss == null
        ? null
        : 1 - finiteNumber(packetLoss, 0);
  return {
    id,
    label: group?.group_label || group?.label || "",
    throughput: rawGroupValue(group, ["throughput_mbps", "avg_throughput_mbps", "backhaul_throughput_mbps"], 0),
    load: rawGroupValue(group, ["load_mbps", "traffic_load_mbps", "demand_mbps"], 0),
    capacity: rawGroupValue(group, ["available_capacity_mbps", "capacity_mbps"], 0),
    utilization:
      group?.utilization != null
        ? clamp01(group.utilization)
        : group?.utilization_ratio != null
          ? clamp01(group.utilization_ratio)
          : null,
    successRate: successRate == null ? null : clamp01(successRate),
    coveredUsers: rawGroupValue(group, ["covered_users", "user_count", "connected_users"], 0),
    activeLinks: rawGroupValue(group, ["active_links", "link_count"], 0),
    deviceCount: rawGroupValue(group, ["device_count", "station_count"], 0),
    latency: rawGroupValue(group, ["latency_ms", "avg_latency_ms"], 0),
    packetLoss: packetLoss == null ? null : clamp01(packetLoss),
    damageFactor: group?.damage_factor == null ? null : clamp01(group.damage_factor),
    status: group?.status ? String(group.status) : "",
    events: Array.isArray(group?.events) ? group.events : [],
  };
};

const worstStatus = (groups) => {
  const order = ["offline", "congested", "degraded", "recovering", "deploying", "normal", "active"];
  const statuses = groups.map((group) => String(group.status || "")).filter(Boolean);
  return order.find((status) => statuses.includes(status)) || statuses[0] || "";
};

const aggregateGroups = (groups) => {
  const byId = new Map();
  for (const group of groups.map(normalizeRawGroup)) {
    if (!LINK_GROUPS.some((def) => def.id === group.id)) continue;
    const bucket = byId.get(group.id) || [];
    bucket.push(group);
    byId.set(group.id, bucket);
  }

  const result = {};
  for (const [id, items] of byId.entries()) {
    const average = (key) => {
      const values = items.map((item) => item[key]).filter((value) => value != null && Number.isFinite(Number(value)));
      return values.length ? values.reduce((sum, value) => sum + Number(value), 0) / values.length : null;
    };
    const sum = (key) => items.reduce((total, item) => total + finiteNumber(item[key], 0), 0);
    result[id] = {
      id,
      label: items.find((item) => item.label)?.label || "",
      throughput: sum("throughput"),
      load: sum("load"),
      capacity: sum("capacity"),
      utilization: average("utilization"),
      successRate: average("successRate"),
      coveredUsers: sum("coveredUsers") || sum("activeLinks"),
      activeLinks: sum("activeLinks"),
      deviceCount: sum("deviceCount"),
      latency: average("latency") || 0,
      packetLoss: average("packetLoss"),
      damageFactor: average("damageFactor"),
      status: worstStatus(items),
      events: items.flatMap((item) => item.events || []),
    };
  }
  return result;
};

const deriveStatus = (group, phase) => {
  if (group.status) return group.status;
  if (group.successRate <= 0.05 && group.throughput <= 0.01) return "offline";
  if (group.utilization >= 0.9) return "congested";
  if (group.successRate < 0.72 || (group.damageFactor != null && group.damageFactor > 0.32)) return "degraded";
  if (phase === "deploying") return "deploying";
  if (phase === "recovery") return "recovering";
  return "normal";
};

const normalizeMetricsPayload = (payload) => {
  const rawFrames = Array.isArray(payload?.series)
    ? payload.series
    : Array.isArray(payload?.frames)
      ? payload.frames
      : Array.isArray(payload)
        ? payload
        : [];

  const baseFrames = rawFrames
    .map((frame, index) => {
      const groupsById = aggregateGroups(Array.isArray(frame?.groups) ? frame.groups : []);
      return {
        replayId: frame?.replay_id || payload?.replay_id || activeSessionId.value,
        frameIndex: finiteNumber(frame?.frame_index ?? frame?.frameIndex ?? index, index),
        time: finiteNumber(frame?.time_s ?? frame?.time ?? index, index),
        phase: phaseForFrame(frame, index, rawFrames.length),
        summary: frame?.summary || {},
        acceptance: frame?.acceptance || {},
        events: Array.isArray(frame?.events) ? frame.events : [],
        groupsById,
      };
    })
    .sort((left, right) => left.frameIndex - right.frameIndex);

  const peakByGroup = {};
  for (const frame of baseFrames) {
    for (const def of LINK_GROUPS) {
      const group = frame.groupsById[def.id];
      peakByGroup[def.id] = Math.max(peakByGroup[def.id] || 0, finiteNumber(group?.throughput, 0));
    }
  }

  return baseFrames.map((frame) => {
    const groups = LINK_GROUPS.map((def) => {
      const raw = frame.groupsById[def.id] || {};
      const peak = Math.max(peakByGroup[def.id] || 0, raw.throughput || 0.01, 0.01);
      const capacity = raw.capacity > 0 ? raw.capacity : peak * 1.18;
      const utilization =
        raw.utilization == null
          ? clamp01(capacity > 0 ? raw.throughput / capacity : 0)
          : clamp01(raw.utilization);
      const packetLoss = raw.packetLoss == null ? 1 - (raw.successRate ?? 0) : raw.packetLoss;
      const successRate = raw.successRate == null ? clamp01(1 - packetLoss) : clamp01(raw.successRate);
      const group = {
        ...def,
        label: def.title,
        throughput: finiteNumber(raw.throughput, 0),
        load: finiteNumber(raw.load, 0),
        capacity,
        utilization,
        successRate,
        coveredUsers: finiteNumber(raw.coveredUsers, 0),
        activeLinks: finiteNumber(raw.activeLinks, 0),
        deviceCount: finiteNumber(raw.deviceCount, 0),
        latency: finiteNumber(raw.latency, 0),
        packetLoss: clamp01(packetLoss),
        damageFactor: raw.damageFactor == null ? null : clamp01(raw.damageFactor),
        status: raw.status || "",
        events: raw.events || [],
      };
      return {
        ...group,
        status: deriveStatus(group, frame.phase),
      };
    });

    const summaryThroughput = groups.reduce((sum, group) => sum + group.throughput, 0);
    const summarySuccess = groups.reduce((sum, group) => sum + group.successRate, 0) / Math.max(1, groups.length);
    const coveredUsers = Math.max(...groups.map((group) => group.coveredUsers || 0), 0);
    return {
      ...frame,
      groups,
      summary: {
        ...frame.summary,
        avg_throughput_mbps: finiteNumber(frame.summary?.avg_throughput_mbps, summaryThroughput / Math.max(1, groups.length)),
        total_throughput_mbps: finiteNumber(frame.summary?.total_throughput_mbps, summaryThroughput),
        avg_success_rate: finiteNumber(frame.summary?.avg_success_rate, summarySuccess),
        covered_users: finiteNumber(frame.summary?.covered_users, coveredUsers),
      },
      acceptance: frame.acceptance || {},
    };
  });
};

const fetchJson = async (url) => {
  const response = await axios.get(url);
  return response.data;
};

const stopLogPolling = () => {
  if (logPollTimer) {
    window.clearInterval(logPollTimer);
    logPollTimer = null;
  }
};

const resetSimulationState = () => {
  stopPlayback();
  stopLogPolling();
  simulationStarted.value = false;
  terminalStatus.value = "idle";
  metricsSeries.value = [];
  logs.value = [];
  terminalLines.value = [];
  currentFrameIndex.value = 0;
  lastSyncedLogCount = 0;
  lastTerminalFrameIndex = -1;
};

const refreshSessions = async () => {
  isLoadingSessions.value = true;
  errorMessage.value = "";
  appendTerminalLine("前端操作：刷新链路仿真回放会话列表。", { level: "ACTION" });
  try {
    const query = hashQuery();
    const payload = await fetchJson(`${API_BASE}/replay/sessions?limit=50`);
    sessions.value = (payload?.sessions || []).map(normalizeSession);
    appendTerminalLine(`后端响应：链路回放会话 ${sessions.value.length} 条。`, { level: "BACKEND", source: "BACKEND" });
    const preferredId = query.replayId || getActiveReplaySessionId();
    const selected = sessions.value.find((session) => session.id === preferredId) || sessions.value[0] || null;
    if (selected) {
      await selectSession(selected.id, {
        frameIndex: Number.isFinite(query.frameIndex) ? query.frameIndex : currentFrameIndex.value,
        silent: true,
      });
    } else {
      resetSimulationState();
      activeSessionId.value = "";
      selectedSessionForControl.value = "";
      activeSessionDetail.value = null;
      appendTerminalLine("后端响应：暂无可用回放会话。", { level: "BACKEND", source: "BACKEND" });
    }
  } catch (error) {
    errorMessage.value = `回放会话加载失败：${error?.response?.data?.detail || error?.message || error}`;
    appendTerminalLine(errorMessage.value, { level: "ERROR", source: "BACKEND" });
  } finally {
    isLoadingSessions.value = false;
  }
};

const loadSessionDetail = async (id) => {
  const detail = normalizeSession(await fetchJson(`${API_BASE}/replay/sessions/${encodeURIComponent(id)}`));
  activeSessionDetail.value = detail;
  const existingIndex = sessions.value.findIndex((session) => session.id === id);
  if (existingIndex >= 0) sessions.value.splice(existingIndex, 1, detail);
};

const loadMetrics = async (id) => {
  const payload = await fetchJson(`${API_BASE}/replay/sessions/${encodeURIComponent(id)}/link-metrics`);
  metricsSeries.value = normalizeMetricsPayload(payload);
};

const loadLogs = async (id, options = {}) => {
  try {
    const payload = await fetchJson(`${API_BASE}/replay/sessions/${encodeURIComponent(id)}/logs?limit=1000`);
    const nextLines = Array.isArray(payload?.lines) ? payload.lines : [];
    logs.value = nextLines;
    if (options.appendToTerminal) {
      if (options.reset || nextLines.length < lastSyncedLogCount) {
        lastSyncedLogCount = 0;
      }
      const freshLines = nextLines.slice(lastSyncedLogCount);
      freshLines.forEach((line) => appendTerminalLine(line, { level: "BACKEND", source: "BACKEND" }));
      lastSyncedLogCount = nextLines.length;
      if (options.reset && !freshLines.length) {
        appendTerminalLine("后端暂无历史日志，已进入链路帧实时回放。");
      }
    }
  } catch {
    if (options.appendToTerminal) {
      appendTerminalLine("后端日志读取失败，继续使用前端帧回放输出。");
    } else {
      logs.value = [];
    }
  }
};

const selectSession = async (id, options = {}) => {
  if (!id) return;
  resetSimulationState();
  activeSessionId.value = id;
  selectedSessionForControl.value = id;
  setActiveReplaySessionId(id);
  errorMessage.value = "";
  try {
    const nextIndex = Number.isFinite(options.frameIndex) ? Number(options.frameIndex) : 0;
    currentFrameIndex.value = Math.max(0, nextIndex);
    await loadSessionDetail(id);
    appendTerminalLine(`已选择回放记录 replay_id=${id}。点击“开始仿真”后加载链路指标并输出实时终端。`);
    appendLinkUserNodeCount(`链路仿真接入灾害场景：${activeSessionDetail.value?.title || id}`, activeSessionDetail.value);
    if (!options.silent) showStatus("回放记录已选择，等待开始仿真。");
  } catch (error) {
    errorMessage.value = `回放元数据加载失败：${error?.response?.data?.detail || error?.message || error}`;
  }
};

const startLogPolling = () => {
  stopLogPolling();
  if (!activeSessionId.value) return;
  logPollTimer = window.setInterval(() => {
    void loadLogs(activeSessionId.value, { appendToTerminal: true });
  }, 1200);
};

const frameByIndex = (index) => {
  if (!metricsSeries.value.length) return null;
  return metricsSeries.value.reduce((nearest, frame) => {
    if (!nearest) return frame;
    return Math.abs(frame.frameIndex - index) < Math.abs(nearest.frameIndex - index) ? frame : nearest;
  }, null);
};

const appendFrameTerminalLine = (force = false) => {
  const frame = frameByIndex(currentFrameIndex.value);
  if (!frame) return;
  if (!force && frame.frameIndex === lastTerminalFrameIndex) return;
  lastTerminalFrameIndex = frame.frameIndex;
  appendTerminalLine(
    `Frame ${frame.frameIndex + 1}/${displayFrameCount.value} T+${formatTime(frame.time)} ` +
      `总吞吐 ${formatMbps(frame.summary.total_throughput_mbps)}，平均成功率 ${formatPercent(frame.summary.avg_success_rate)}。`
  );
};

const startSimulation = async () => {
  if (!activeSessionId.value || isLoadingMetrics.value) return;
  stopPlayback();
  stopLogPolling();
  simulationStarted.value = true;
  isLoadingMetrics.value = true;
  terminalStatus.value = "running";
  terminalLines.value = [];
  logs.value = [];
  metricsSeries.value = [];
  lastSyncedLogCount = 0;
  lastTerminalFrameIndex = -1;
  errorMessage.value = "";
  appendTerminalLine(`启动链路仿真 replay_id=${activeSessionId.value}。`);
  appendTerminalLine("正在加载链路指标并连接实时日志输出。");

  try {
    await Promise.all([
      loadSessionDetail(activeSessionId.value),
      loadMetrics(activeSessionId.value),
      loadLogs(activeSessionId.value, { appendToTerminal: true, reset: true }),
    ]);
    appendLinkUserNodeCount(`链路仿真接入灾害场景：${activeSession.value?.title || activeSessionId.value}`, activeSession.value, metricsSeries.value[0]);
    currentFrameIndex.value = Math.max(0, Math.min(maxFrameIndex.value, currentFrameIndex.value));
    appendTerminalLine(`链路指标加载完成，共 ${formatNumber(metricsSeries.value.length)} 帧。`);
    appendFrameTerminalLine(true);
    startLogPolling();
    if (metricsSeries.value.length > 1) {
      startPlayback();
    } else {
      terminalStatus.value = "completed";
    }
    showStatus("链路仿真已启动。");
  } catch (error) {
    terminalStatus.value = "failed";
    errorMessage.value = `链路仿真启动失败：${error?.response?.data?.detail || error?.message || error}`;
    appendTerminalLine(errorMessage.value);
  } finally {
    isLoadingMetrics.value = false;
  }
};

const refreshAll = async () => {
  await refreshSessions();
  showStatus("链路仿真页面已刷新。");
};

const openReplay = () => {
  if (!activeSessionId.value || typeof window === "undefined") return;
  window.location.hash = `/replay?replay_id=${encodeURIComponent(activeSessionId.value)}&frame_index=${encodeURIComponent(currentFrameIndex.value)}`;
};

const activeSession = computed(() => {
  if (activeSessionDetail.value?.id === activeSessionId.value) return activeSessionDetail.value;
  return sessions.value.find((session) => session.id === activeSessionId.value) || null;
});

const maxFrameIndex = computed(() => {
  const metricMax = metricsSeries.value.reduce((max, frame) => Math.max(max, frame.frameIndex), 0);
  const sessionMax = Math.max(0, finiteNumber(activeSession.value?.frameCount, 0) - 1);
  return Math.max(metricMax, sessionMax, 0);
});

const displayFrameCount = computed(() => Math.max(1, maxFrameIndex.value + 1));

const currentMetricFrame = computed(() => {
  if (!metricsSeries.value.length) return null;
  return metricsSeries.value.reduce((nearest, frame) => {
    if (!nearest) return frame;
    return Math.abs(frame.frameIndex - currentFrameIndex.value) < Math.abs(nearest.frameIndex - currentFrameIndex.value)
      ? frame
      : nearest;
  }, null);
});

const firstMetricFrame = computed(() => metricsSeries.value[0] || null);
const finalMetricFrame = computed(() => metricsSeries.value[metricsSeries.value.length - 1] || null);

const pageHeading = computed(() => {
  if (activeSession.value?.title) return `${activeSession.value.title} 链路仿真`;
  return "策略测试结果链路仿真";
});

const headingDescription = computed(() => {
  if (!activeSession.value) return "选择一次策略测试或场景回放记录后，点击开始仿真再加载链路侧恢复证明。";
  return `replay_id=${activeSession.value.id}，点击开始仿真后输出实时终端并展示链路结果。`;
});

const showLinkResults = computed(() => simulationStarted.value && metricsSeries.value.length > 0);
const startButtonLabel = computed(() => {
  if (isLoadingMetrics.value) return "启动中...";
  return simulationStarted.value ? "重新仿真" : "开始仿真";
});
const currentPhaseLabel = computed(() => PHASE_LABELS[currentMetricFrame.value?.phase] || "--");
const currentTimeLabel = computed(() => currentMetricFrame.value ? `T+${formatTime(currentMetricFrame.value.time)}` : "--");

const summaryCards = computed(() => {
  const current = currentMetricFrame.value;
  const final = finalMetricFrame.value;
  const session = activeSession.value;
  const finalSuccess = final?.summary?.avg_success_rate ?? session?.summary?.coverageRatio ?? 0;
  const currentThroughput = current?.summary?.total_throughput_mbps ?? 0;
  return [
    {
      label: "当前 replay_id",
      value: activeSessionId.value || "--",
      description: session ? `${formatDateTime(session.createdAt)} 创建` : "等待策略测试结果",
    },
    {
      label: "节点规模",
      value: formatNumber(session?.nodeCountTotal || session?.summary?.totalUsers || 0),
      description: `帧数 ${formatNumber(session?.frameCount || metricsSeries.value.length || 0)}`,
    },
    {
      label: "三类链路吞吐",
      value: formatMbps(currentThroughput),
      description: `${LINK_GROUPS.length} 类关键链路同步回放`,
    },
    {
      label: "终态成功率",
      value: formatPercent(finalSuccess),
      description: `覆盖用户 ${formatNumber(final?.summary?.covered_users || session?.summary?.connectedUsers || 0)}`,
    },
  ];
});

const groupFromFrame = (frame, groupId) => (frame?.groups || []).find((group) => group.id === groupId) || null;

const linkGroupCards = computed(() => {
  const current = currentMetricFrame.value;
  const first = firstMetricFrame.value;
  const final = finalMetricFrame.value;
  return LINK_GROUPS.map((def) => {
    const currentGroup = groupFromFrame(current, def.id) || { ...def, throughput: 0, utilization: 0, successRate: 0, coveredUsers: 0, activeLinks: 0, latency: 0, status: "offline" };
    const firstGroup = groupFromFrame(first, def.id);
    const finalGroup = groupFromFrame(final, def.id);
    const status = currentGroup.status || "normal";
    const statusTone = status === "congested" || status === "offline" ? "danger" : status === "degraded" ? "warning" : status === "recovering" || status === "deploying" ? "info" : "success";
    return {
      ...def,
      ...currentGroup,
      statusTone,
      statusLabel: STATUS_LABELS[status] || status,
      delta: finiteNumber(currentGroup.throughput, 0) - finiteNumber(firstGroup?.throughput, 0),
      successHint: `终态 ${formatPercent(finalGroup?.successRate ?? currentGroup.successRate)}`,
      scaleLabel: currentGroup.deviceCount ? `${formatNumber(currentGroup.deviceCount)} 台设备` : `${formatNumber(currentGroup.activeLinks)} 条链路`,
      eventText: eventTextForGroup(def, currentGroup, current),
    };
  });
});

const eventTextForGroup = (def, group, frame) => {
  const directEvent = group.events?.[0]?.message || group.events?.[0];
  if (directEvent) return String(directEvent);
  const phase = frame?.phase || "initial";
  if (phase === "damaged") return `${def.shortLabel} 受灾害影响，成功率降至 ${formatPercent(group.successRate)}，时延 ${formatLatency(group.latency)}。`;
  if (phase === "deploying") return `策略部署正在生效，${def.shortLabel} 当前承载 ${formatMbps(group.throughput)}。`;
  if (phase === "recovery" || phase === "finished") return `${def.shortLabel} 已恢复至 ${formatPercent(group.successRate)} 成功率，服务规模 ${formatNumber(group.coveredUsers || group.activeLinks)}。`;
  return `${def.shortLabel} 初始状态已记录，作为灾害影响和恢复效果的对照基线。`;
};

const phaseMarkers = computed(() => {
  const phases = [
    { key: "initial", label: "初始", ratio: 0 },
    { key: "damaged", label: "受损", ratio: 0.18 },
    { key: "deploying", label: "部署", ratio: 0.62 },
    { key: "recovery", label: "恢复", ratio: 0.95 },
  ];
  return phases.map((phase) => ({ ...phase, left: Math.round(phase.ratio * 100) }));
});

const chartMetricDefs = [
  {
    key: "throughput",
    label: "三类链路吞吐趋势",
    description: "Mbps",
    getter: (group) => group.throughput,
    formatter: (value) => finiteNumber(value, 0).toFixed(2),
  },
  {
    key: "utilization",
    label: "链路利用率趋势",
    description: "容量占用比例",
    getter: (group) => group.utilization,
    formatter: (value) => `${(clamp01(value) * 100).toFixed(0)}%`,
    maxFloor: 1,
  },
  {
    key: "successRate",
    label: "通信成功率趋势",
    description: "接入或传输成功比例",
    getter: (group) => group.successRate,
    formatter: (value) => `${(clamp01(value) * 100).toFixed(0)}%`,
    maxFloor: 1,
  },
  {
    key: "coveredUsers",
    label: "服务规模趋势",
    description: "覆盖用户或活跃链路数",
    getter: (group) => group.coveredUsers || group.activeLinks,
    formatter: (value) => formatNumber(value),
  },
];

const chartModels = computed(() => {
  const frames = metricsSeries.value;
  const width = 740;
  const height = 230;
  const padding = { top: 24, right: 20, bottom: 34, left: 58 };
  if (!frames.length) {
    return chartMetricDefs.map((def) => ({
      key: def.key,
      label: def.label,
      description: def.description,
      width,
      height,
      padding,
      yTicks: [],
      xTicks: [],
      series: [],
      currentPoints: [],
      currentLabel: "--",
      currentX: 0,
    }));
  }

  const maxFrame = Math.max(1, maxFrameIndex.value);
  const xForFrame = (frame) => padding.left + (finiteNumber(frame.frameIndex, 0) / maxFrame) * (width - padding.left - padding.right);
  const currentX = padding.left + (currentFrameIndex.value / maxFrame) * (width - padding.left - padding.right);

  return chartMetricDefs.map((def) => {
    const allValues = [];
    for (const frame of frames) {
      for (const group of frame.groups) {
        allValues.push(finiteNumber(def.getter(group), 0));
      }
    }
    const maxValue = Math.max(def.maxFloor || 0, ...allValues, 0.01) * (def.maxFloor ? 1 : 1.12);
    const yForValue = (value) => padding.top + (1 - finiteNumber(value, 0) / maxValue) * (height - padding.top - padding.bottom);
    const baseline = height - padding.bottom;
    const yTicks = [0, 0.25, 0.5, 0.75, 1].map((ratio) => {
      const value = maxValue * ratio;
      return {
        value,
        y: yForValue(value),
        label: def.formatter(value),
      };
    });
    const xTicks = [0, 0.25, 0.5, 0.75, 1].map((ratio) => {
      const frameIndex = Math.round(maxFrame * ratio);
      return {
        value: frameIndex,
        x: padding.left + ratio * (width - padding.left - padding.right),
        label: `${frameIndex}`,
      };
    });

    const series = LINK_GROUPS.map((groupDef) => {
      const points = frames.map((frame) => {
        const group = groupFromFrame(frame, groupDef.id) || {};
        return {
          x: xForFrame(frame),
          y: yForValue(def.getter(group)),
          value: finiteNumber(def.getter(group), 0),
        };
      });
      const path = points.map((point, index) => `${index ? "L" : "M"}${point.x.toFixed(2)} ${point.y.toFixed(2)}`).join(" ");
      const area = points.length
        ? `${path} L${points[points.length - 1].x.toFixed(2)} ${baseline} L${points[0].x.toFixed(2)} ${baseline} Z`
        : "";
      return {
        id: groupDef.id,
        label: groupDef.shortLabel,
        color: groupDef.color,
        fill: groupDef.color,
        path,
        area,
        points,
      };
    });
    const currentPoints = series.map((line) => {
      const point = line.points.reduce((nearest, item) => {
        if (!nearest) return item;
        return Math.abs(item.x - currentX) < Math.abs(nearest.x - currentX) ? item : nearest;
      }, null);
      return { id: line.id, color: line.color, x: point?.x || currentX, y: point?.y || baseline };
    });
    const current = currentMetricFrame.value;
    return {
      key: def.key,
      label: def.label,
      description: def.description,
      width,
      height,
      padding,
      yTicks,
      xTicks,
      series,
      currentPoints,
      currentX,
      currentLabel: current ? `Frame ${current.frameIndex} · ${formatTime(current.time)}` : "--",
    };
  });
});

const startPlayback = () => {
  if (!showLinkResults.value || maxFrameIndex.value <= 0) return;
  stopPlayback();
  isPlaying.value = true;
  terminalStatus.value = "running";
  appendTerminalLine("链路帧回放开始。");
  playTimer = window.setInterval(() => {
    if (currentFrameIndex.value >= maxFrameIndex.value) {
      appendTerminalLine("链路仿真回放完成。");
      terminalStatus.value = "completed";
      stopPlayback();
      return;
    }
    currentFrameIndex.value += 1;
    appendFrameTerminalLine();
  }, 850);
};

const stopPlayback = () => {
  isPlaying.value = false;
  if (playTimer) {
    window.clearInterval(playTimer);
    playTimer = null;
  }
};

watch(currentFrameIndex, (value) => {
  const next = Math.max(0, Math.min(maxFrameIndex.value, finiteNumber(value, 0)));
  if (next !== value) {
    currentFrameIndex.value = next;
    return;
  }
  if (showLinkResults.value) appendFrameTerminalLine();
});

onMounted(() => {
  void refreshAll();
});

onBeforeUnmount(() => {
  stopPlayback();
  stopLogPolling();
  if (statusTimer) window.clearTimeout(statusTimer);
});
</script>

<style scoped>
.link-simulation {
  position: relative;
  width: 1920px;
  height: 1010px;
  min-height: 1010px;
  overflow: hidden;
  background: #eef5ff;
  color: #1f2d3d;
  font-family: "Microsoft YaHei", "PingFang SC", "Source Han Sans CN", sans-serif;
}

.link-simulation__bg,
.link-simulation__panel-shadow {
  position: absolute;
  display: block;
  border: 0;
  pointer-events: none;
  user-select: none;
  z-index: 0;
}

.link-simulation__bg {
  left: 0;
  top: 0;
  width: 1920px;
  height: 1010px;
}

.link-simulation__panel-shadow {
  left: 97px;
  top: 0;
  width: 1740px;
  height: 1027px;
  opacity: 0.5;
}

.link-panel {
  position: absolute;
  left: 140px;
  top: 44px;
  width: 1652px;
  height: 930px;
  min-height: 0;
  overflow-x: hidden;
  overflow-y: auto;
  scrollbar-color: rgba(57, 97, 246, 0.45) rgba(225, 236, 255, 0.72);
  scrollbar-width: thin;
  z-index: 2;
}

.link-panel::-webkit-scrollbar {
  width: 8px;
}

.link-panel::-webkit-scrollbar-track {
  background: rgba(225, 236, 255, 0.72);
  border-radius: 999px;
}

.link-panel::-webkit-scrollbar-thumb {
  background: rgba(57, 97, 246, 0.45);
  border-radius: 999px;
}

.link-panel__scroll {
  position: relative;
  width: 1640px;
  min-height: 100%;
  box-sizing: border-box;
  padding: 84px 0 34px;
}

.link-title {
  position: absolute;
  left: 0;
  top: 0;
  width: 157px;
  height: 68px;
}

.link-title__ribbon {
  position: absolute;
  left: -14px;
  top: 2px;
  width: 157px;
  height: 66px;
}

.link-title h1 {
  position: absolute;
  left: 3px;
  top: 3px;
  width: 125px;
  margin: 0;
  color: #1890ff;
  font-family: "Source Han Sans CN", "Microsoft YaHei", sans-serif;
  font-size: 20px;
  font-weight: 700;
  line-height: 41px;
  text-align: center;
  text-shadow: 0 0 20px rgba(0, 200, 244, 0.5);
}

.link-workbench {
  width: 1628px;
  margin-left: 4px;
  box-sizing: border-box;
  border: 1px solid rgba(233, 233, 233, 1);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.96);
  box-shadow: 3px 3px 20px rgba(233, 233, 233, 0.9);
  color: #334155;
  font-size: 14px;
  padding: 14px;
}

.module-heading,
.section-card-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
}

.module-heading--top {
  min-height: 44px;
  margin-bottom: 12px;
}

.module-heading > div:first-child,
.section-card-header > div:first-child {
  display: flex;
  align-items: center;
  gap: 10px;
  min-width: 0;
}

.section-card-header > div:first-child {
  display: block;
}

.module-heading i {
  width: 6px;
  height: 20px;
  flex: 0 0 auto;
  background: linear-gradient(180deg, #6fcadf 0%, #05b7df 100%);
}

.module-heading h2,
.section-card-header h3,
.link-group-card h3,
.trend-card h3 {
  margin: 0;
  color: #1f2d3d;
  font-size: 16px;
  font-weight: 600;
}

.module-heading p,
.section-card-header p,
.trend-card header p,
.link-group-card p {
  min-width: 0;
  margin: 4px 0 0;
  color: #64748b;
  font-size: 12px;
  line-height: 1.5;
}

.heading-actions,
.control-actions {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  flex-wrap: wrap;
  gap: 10px;
}

.primary-button,
.ghost-button {
  height: 38px;
  min-width: 70px;
  border-radius: 6px;
  border: 1px solid rgba(57, 97, 246, 0.16);
  padding: 0 14px;
  font-size: 14px;
  cursor: pointer;
}

.primary-button {
  border-color: transparent;
  background: #3961f6;
  color: #ffffff;
  box-shadow: 0 10px 22px rgba(57, 97, 246, 0.2);
}

.ghost-button {
  background: rgba(255, 255, 255, 0.72);
  color: #0079fe;
}

.ghost-button--blue {
  background: rgba(0, 121, 254, 0.1);
}

.primary-button:disabled,
.ghost-button:disabled,
select:disabled,
input:disabled {
  cursor: not-allowed;
  opacity: 0.55;
}

.module-error {
  margin-bottom: 12px;
  padding: 10px 12px;
  border: 1px solid rgba(248, 113, 113, 0.3);
  border-radius: 8px;
  background: rgba(254, 242, 242, 0.9);
  color: #b91c1c;
}

.status-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin-bottom: 12px;
}

.summary-card,
.control-card,
.link-group-card,
.trend-card,
.terminal-card {
  border: 1px solid rgba(233, 233, 233, 1);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.88);
  box-shadow: 0 0 5px rgba(246, 246, 254, 0.95) inset;
}

.summary-card {
  min-height: 86px;
  box-sizing: border-box;
  padding: 14px;
}

.summary-card small,
.control-field span,
.section-card-header > span,
.trend-card header > span {
  display: block;
  color: #7a8aa0;
  font-size: 12px;
}

.summary-card strong {
  display: block;
  margin-top: 8px;
  color: #083289;
  font-size: 22px;
  line-height: 1.15;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.summary-card span {
  display: block;
  margin-top: 8px;
  color: #64748b;
  font-size: 12px;
}

.control-card {
  padding: 14px;
  margin-bottom: 12px;
}

.control-row {
  display: grid;
  grid-template-columns: minmax(360px, 1.6fr) minmax(150px, 0.55fr) minmax(120px, 0.35fr) minmax(420px, 1fr);
  gap: 12px;
  align-items: end;
}

.control-row--idle {
  grid-template-columns: minmax(520px, 1fr) minmax(360px, auto);
}

.control-field {
  display: grid;
  gap: 6px;
  min-width: 0;
}

.control-field select,
.control-field input {
  width: 100%;
  height: 40px;
  box-sizing: border-box;
  border: 1px solid rgba(233, 233, 233, 1);
  border-radius: 8px;
  background: #ffffff;
  box-shadow: 0 0 5px rgba(246, 246, 254, 1) inset;
  color: #334155;
  font-size: 14px;
  padding: 0 12px;
}

.timeline-row {
  position: relative;
  margin-top: 14px;
  padding-top: 8px;
}

.timeline-meta {
  display: flex;
  justify-content: space-between;
  gap: 14px;
  color: #64748b;
  font-size: 12px;
}

.timeline-meta strong {
  color: #083289;
}

.timeline-range {
  width: 100%;
  margin-top: 10px;
  accent-color: #3961f6;
}

.timeline-phases {
  position: relative;
  height: 18px;
  margin-top: 3px;
  color: #94a3b8;
  font-size: 11px;
}

.timeline-phases span {
  position: absolute;
  top: 0;
  transform: translateX(-50%);
}

.link-group-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
  margin-bottom: 12px;
}

.link-group-card {
  padding: 14px;
  border-top-width: 3px;
}

.link-group-card--success {
  border-top-color: #22c55e;
}

.link-group-card--warning {
  border-top-color: #f59e0b;
}

.link-group-card--danger {
  border-top-color: #ef4444;
}

.link-group-card--info {
  border-top-color: #1890ff;
}

.link-group-card header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.link-group-card header > div {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
}

.link-group-card header strong {
  flex: 0 0 auto;
  padding: 4px 9px;
  border-radius: 999px;
  background: rgba(0, 121, 254, 0.1);
  color: #0079fe;
  font-size: 12px;
  font-weight: 600;
}

.group-dot {
  width: 10px;
  height: 10px;
  flex: 0 0 auto;
  border-radius: 999px;
}

.group-metrics {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
  margin-top: 12px;
}

.group-metrics span {
  min-width: 0;
  border-radius: 8px;
  background: rgba(238, 245, 255, 0.8);
  padding: 10px;
}

.group-metrics small,
.group-event small {
  display: block;
  color: #7a8aa0;
  font-size: 12px;
}

.group-metrics b {
  display: block;
  margin-top: 5px;
  color: #083289;
  font-size: 17px;
  line-height: 1.2;
}

.group-metrics em {
  display: block;
  margin-top: 5px;
  color: #64748b;
  font-size: 12px;
  font-style: normal;
}

.group-metrics em.is-up {
  color: #16a34a;
}

.group-metrics em.is-down {
  color: #dc2626;
}

.group-metrics i {
  display: block;
  height: 5px;
  margin-top: 8px;
  overflow: hidden;
  border-radius: 999px;
  background: rgba(203, 213, 225, 0.8);
}

.group-metrics i span {
  display: block;
  height: 100%;
  padding: 0;
  border-radius: inherit;
}

.group-event {
  margin-top: 12px;
  border-top: 1px solid rgba(226, 232, 240, 0.9);
  padding-top: 10px;
}

.group-event span {
  display: block;
  margin-top: 5px;
  color: #334155;
  line-height: 1.55;
}

.trend-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
  margin-bottom: 12px;
}

.trend-card {
  min-height: 320px;
  padding: 14px;
  box-sizing: border-box;
}

.trend-card header {
  display: flex;
  justify-content: space-between;
  gap: 14px;
  margin-bottom: 10px;
}

.trend-card svg {
  display: block;
  width: 100%;
  height: 230px;
  border-radius: 8px;
  background: linear-gradient(180deg, #f7fbff 0%, #eef5ff 100%);
}

.chart-grid-line {
  stroke: rgba(148, 163, 184, 0.25);
  stroke-width: 1;
}

.chart-grid-line--vertical {
  stroke: rgba(148, 163, 184, 0.14);
}

.chart-cursor {
  stroke: rgba(57, 97, 246, 0.45);
  stroke-width: 1.5;
  stroke-dasharray: 4 4;
}

.chart-axis-text {
  fill: #7a8aa0;
  font-family: "Cascadia Code", "SFMono-Regular", Consolas, monospace;
  font-size: 11px;
}

.chart-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin-top: 9px;
  color: #475569;
  font-size: 12px;
}

.chart-legend span {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}

.chart-legend i {
  width: 18px;
  height: 3px;
  border-radius: 999px;
}

.terminal-section {
  margin-bottom: 12px;
}

.terminal-card {
  padding: 14px;
}

.terminal-status {
  flex: 0 0 auto;
  min-width: 72px;
  border-radius: 999px;
  padding: 5px 10px;
  background: rgba(100, 116, 139, 0.1);
  color: #64748b;
  font-weight: 600;
  text-align: center;
}

.terminal-status--running {
  background: rgba(0, 121, 254, 0.12);
  color: #0079fe;
}

.terminal-status--completed {
  background: rgba(34, 197, 94, 0.12);
  color: #15803d;
}

.terminal-status--failed {
  background: rgba(239, 68, 68, 0.12);
  color: #b91c1c;
}

.terminal-output {
  margin-top: 12px;
  height: 360px;
  overflow: auto;
  box-sizing: border-box;
  border: 1px solid rgba(15, 23, 42, 0.78);
  border-radius: 8px;
  background:
    linear-gradient(180deg, rgba(2, 6, 23, 0.98), rgba(15, 23, 42, 0.94)),
    radial-gradient(circle at top left, rgba(56, 189, 248, 0.12), transparent 34%);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05);
  color: #dbeafe;
  font-family: "Cascadia Code", "SFMono-Regular", Consolas, monospace;
  font-size: 12px;
  line-height: 1.72;
  padding: 14px 16px;
}

.terminal-output p {
  margin: 0;
  white-space: pre-wrap;
  word-break: break-word;
}

.terminal-placeholder {
  color: #64748b;
}

.status-toast {
  position: absolute;
  right: 14px;
  top: 2px;
  z-index: 40;
  max-width: 460px;
  border-radius: 8px;
  padding: 10px 14px;
  box-shadow: 0 12px 28px rgba(15, 23, 42, 0.13);
  font-size: 13px;
}

.status-toast--success {
  background: rgba(220, 252, 231, 0.96);
  color: #166534;
}

.status-toast--warning {
  background: rgba(254, 249, 195, 0.96);
  color: #854d0e;
}

.status-toast--error {
  background: rgba(254, 226, 226, 0.96);
  color: #991b1b;
}

.status-fade-enter-active,
.status-fade-leave-active {
  transition: opacity 0.18s ease, transform 0.18s ease;
}

.status-fade-enter-from,
.status-fade-leave-to {
  opacity: 0;
  transform: translateY(-6px);
}

@media (max-width: 1200px) {
  .status-grid,
  .link-group-grid,
  .trend-grid,
  .control-row {
    grid-template-columns: 1fr;
  }

  .control-actions,
  .heading-actions {
    justify-content: flex-start;
  }
}
</style>
