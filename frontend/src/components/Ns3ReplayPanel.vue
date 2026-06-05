<template>
  <section
    class="scenario-replay"
    :class="[`scenario-replay--${phaseKey}`, `scenario-replay--native-${nativePhaseKey}`, { 'scenario-replay--playing': playing }]"
    aria-label="真实后端场景回放界面"
  >
    <div ref="mapShellRef" class="map-container" @pointerleave="hoverNode = null">
      <canvas
        ref="canvasRef"
        class="overlay-canvas"
        @click="handleCanvasClick"
      ></canvas>
    </div>

    <header class="top-bar glass-panel">
      <div class="logo">
        <div class="logo-icon">SAT</div>
        <span class="logo-text">应急通信仿真实验平台</span>
      </div>
      <div class="top-center">
        <div class="experiment-select">
          <select
            :value="activeSessionId || ''"
            :disabled="loadingSessions"
            aria-label="选择后端回放记录"
            @change="selectReplaySession($event.target.value)"
          >
            <option value="">-- 选择后端场景回放记录 --</option>
            <option v-for="session in sessions" :key="session.id" :value="session.id">
              {{ session.title }}
            </option>
          </select>
        </div>
      </div>
      <div class="top-right">
        <button type="button" class="btn btn-outline" :disabled="loadingSessions" @click="refreshReplaySessions">
          {{ loadingSessions ? "同步中" : "刷新" }}
        </button>
        <button type="button" class="btn btn-outline" :disabled="!activeSession" @click="resetPlayback">复位</button>
        <button type="button" class="btn btn-primary" :disabled="!activeSession" @click="togglePlayback">
          {{ playing ? "暂停" : "启动" }}
        </button>
      </div>
    </header>

    <div class="settings-bar glass-panel">
      <div class="settings-title">全局参数调试</div>
      <label class="setting-group">
        <span class="setting-label">总帧数</span>
        <input class="setting-input" type="text" :value="activeSession?.frameCount || 0" readonly />
      </label>
      <label class="setting-group">
        <span class="setting-label">当前帧</span>
        <input class="setting-input" type="text" :value="frameIndex + 1" readonly />
      </label>
      <div class="setting-divider"></div>
      <label class="setting-group">
        <span class="setting-label">用户</span>
        <input class="setting-input" type="text" :value="scenarioCounts.user" readonly />
      </label>
      <label class="setting-group">
        <span class="setting-label">宏基站</span>
        <input class="setting-input" type="text" :value="scenarioCounts.macro" readonly />
      </label>
      <label class="setting-group">
        <span class="setting-label">背负基站</span>
        <input class="setting-input" type="text" :value="scenarioCounts.manpack" readonly />
      </label>
      <label class="setting-group">
        <span class="setting-label">微基站</span>
        <input class="setting-input" type="text" :value="scenarioCounts.smallCell" readonly />
      </label>
      <label class="setting-group">
        <span class="setting-label">中继节点</span>
        <input class="setting-input" type="text" :value="scenarioCounts.relay" readonly />
      </label>
      <button type="button" class="btn btn-primary settings-action" :disabled="!activeSessionId" @click="downloadArtifact('nodes')">
        导出节点清单
      </button>
    </div>

    <div class="disaster-alert" :class="{ show: activeSession, recovery: nativePhaseKey === 'recovery' || nativePhaseKey === 'restored' }">
      {{ nativeAlertText }}
    </div>

    <aside class="left-panel glass-panel">
      <div class="panel-header">网络实时遥测</div>
      <div class="panel-content">
        <div class="stat-row">
          <div class="stat-card">
            <div class="stat-label">推演时间 (T+)</div>
            <div class="stat-value primary">{{ currentTimeDisplay }}<span class="stat-unit">s</span></div>
          </div>
          <div class="stat-card">
            <div class="stat-label">帧渲染率</div>
            <div class="stat-value">{{ playing ? "60" : "--" }}<span class="stat-unit">fps</span></div>
          </div>
          <div class="stat-card">
            <div class="stat-label">在线节点</div>
            <div class="stat-value success">{{ activeNodesCount }}</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">活跃链路</div>
            <div class="stat-value primary">{{ activeLinksCount }}</div>
          </div>
          <div class="stat-card full">
            <div class="stat-label">集群吞吐量</div>
            <div class="stat-value success">{{ throughputDisplay }}<span class="stat-unit"> Mbps</span></div>
          </div>
          <div class="stat-card full">
            <div class="stat-label">端到端延迟</div>
            <div class="stat-value warning">{{ latencyDisplay }}<span class="stat-unit"> ms</span></div>
          </div>
        </div>

        <div class="status-card" :class="nativeStatusClass">
          <div class="status-dot" :class="nativeStatusDotClass"></div>
          <span class="status-text">{{ nativeStatusText }}</span>
        </div>

        <div class="panel-header panel-header--nested">全网吞吐趋势</div>
        <canvas ref="chartCanvasRef" class="mini-chart"></canvas>
      </div>
    </aside>

    <aside class="right-panel glass-panel">
      <div class="panel-header">装备编队状态</div>
      <div class="panel-content">
        <div v-for="item in equipmentCards" :key="item.key" class="equipment-item">
          <div class="equipment-left">
            <div class="equipment-icon" :style="{ color: item.color, background: item.color }"></div>
            <div>
              <div class="equipment-name">{{ item.label }}</div>
              <div class="equipment-count">{{ item.online }}/{{ item.total }} ACTIVE</div>
            </div>
          </div>
          <div class="equipment-status" :class="item.online > 0 ? 'online' : 'offline'">
            {{ item.online > 0 ? "ACTIVE" : "OFFLINE" }}
          </div>
        </div>
      </div>
    </aside>

    <section class="bottom-log-panel glass-panel">
      <div class="log-panel-header">
        <div class="log-title">实时终端输出</div>
        <div class="log-search">
          <input
            v-model.trim="logQuery"
            type="text"
            placeholder="输入关键字检索实时终端输出..."
            @keyup.enter="queryLogs"
          />
          <button type="button" class="btn btn-outline btn-compact" @click="queryLogs">检索</button>
          <button type="button" class="btn btn-outline btn-compact" :disabled="!terminalHistoryLines.length" @click="downloadTerminalLog">
            导出终端输出
          </button>
          <button type="button" class="btn btn-outline btn-compact" :disabled="!displayTerminalEntries.length" @click="clearTerminalLog">
            清空
          </button>
          <button type="button" class="btn btn-outline btn-compact" :disabled="!activeSessionId" @click="downloadArtifact('log')">
            下载后端日志
          </button>
        </div>
      </div>
      <div ref="terminalRef" class="event-log" role="log" aria-live="polite">
        <p
          v-for="(entry, index) in displayTerminalEntries"
          :key="`${index}-${entry.text}`"
          :class="`log-entry--${entry.level.toLowerCase()}`"
        >
          {{ entry.text }}
        </p>
      </div>
    </section>

    <aside class="legend glass-panel" aria-label="物理节点图例">
      <div class="legend-title">物理节点标识</div>
      <div class="legend-item"><span class="legend-dot legend-dot--user"></span> 用户终端 (USER)</div>
      <div class="legend-item"><span class="legend-dot legend-dot--macro"></span> 宏基站 (MACRO)</div>
      <div class="legend-item"><span class="legend-dot legend-dot--manpack"></span> 背负基站 (MANPACK)</div>
      <div class="legend-item"><span class="legend-dot legend-dot--small-cell"></span> 微基站 (SMALL_CELL)</div>
      <div class="legend-item"><span class="legend-dot legend-dot--relay"></span> 中继节点 (RELAY)</div>
      <div class="legend-title legend-title--links">空间链路标识</div>
      <div class="legend-item"><span class="legend-line legend-line--mesh"></span> D2D / Mesh 链路</div>
      <div class="legend-item"><span class="legend-line legend-line--backhaul"></span> 骨干回传链路</div>
    </aside>

    <div class="bottom-bar">
      <div class="time-display">{{ timelineStartText }}</div>
      <input
        v-model.number="frameIndex"
        type="range"
        class="timeline-slider"
        min="0"
        :max="maxFrameIndex"
        step="1"
        :disabled="!activeSession || loadingFrame"
        @input="markReplayStarted"
      />
      <div class="time-display"><span class="current">{{ maxTimeDisplay }}</span>s</div>
      <div class="playback-controls">
        <button type="button" class="control-btn" :disabled="!activeSession" title="复位" @click="jumpToStart">⏮</button>
        <button type="button" class="control-btn" :class="{ active: playing }" :disabled="!activeSession" @click="togglePlayback">
          {{ playing ? "⏸" : "▶" }}
        </button>
        <button type="button" class="control-btn" :disabled="!activeSession" title="直达结束" @click="jumpToEnd">⏭</button>
      </div>
    </div>

    <div v-if="replayError && !loadingSessions" class="replay-error">
      {{ replayError }}
    </div>
  </section>
</template>

<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, reactive, ref, watch } from "vue";
import { rescueApiBase } from "../utils/runtimeEndpoints";
import { getActiveReplaySessionId, setActiveReplaySessionId } from "../utils/replaySessions";
import {
  appendSharedTerminalLine,
  appendSharedTerminalLines,
  clearTerminalOutput,
  exportTerminalOutput,
  terminalHistoryLines,
} from "../utils/terminalOutput";
import {
  buildUserNodeCountMessage,
  userNodeCountLogKey,
} from "../utils/scenarioNodeMetrics";

const API_BASE = rescueApiBase;
const FRAME_SAMPLE_RATIO = 2;
const PLAYING_FRAME_SAMPLE_RATIO = FRAME_SAMPLE_RATIO;
const FRAME_PREFETCH_AHEAD = 2;
const FRAME_CACHE_LIMIT = 24;
const IDLE_RENDER_INTERVAL = 220;
const PLAYING_RENDER_INTERVAL = 110;
const USER_MOTION_CACHE_MS = 480;
const USER_MOTION_PLAYING_AMPLITUDE = 1.15;
const USER_MOTION_IDLE_AMPLITUDE = 0.45;
const USER_MOTION_X_PERIOD_MS = 5200;
const USER_MOTION_Y_PERIOD_MS = 6800;
const CHART_RENDER_INTERVAL = 360;
const TARGET_REPLAY_DURATION_MS = 90000;
const MIN_FRAME_DELAY_MS = 850;
const MAX_FRAME_DELAY_MS = 2800;
const REPLAY_SECONDS_PER_FRAME = 6;
const TILE_SIZE = 256;
const MAX_DRAWABLE_LINKS = 320;
const MAX_WARNING_RINGS = 42;
const MAX_HEAT_BLOBS = 18;
const MAX_COVERAGE_RINGS = 42;
const MAP_WIDTH = 5000;
const MAP_HEIGHT = 5000;
const PRECISE_COORDINATE_SOURCE = "deterministic_grid_cross_cell_v3";
const LEGACY_COORDINATE_SOURCE = "deterministic_grid_polar_v2";

const canvasRef = ref(null);
const mapShellRef = ref(null);
const terminalRef = ref(null);
const chartCanvasRef = ref(null);

const sessions = ref([]);
const activeSessionId = ref(null);
const activeSessionDetail = ref(null);
const currentFrame = ref(null);
const linkMetrics = ref(null);
const lastStableTelemetry = ref({ throughput: 0, latency: 0 });
const allLogLines = ref([]);
const clientLogEntries = ref([]);
const frameIndex = ref(0);
const playbackRate = ref(1);
const loadingSessions = ref(false);
const loadingFrame = ref(false);
const replayError = ref("");
const replayTerminalCleared = ref(false);
const playing = ref(false);
const hasReplayStarted = ref(false);
const logQuery = ref("");
const logQueryMode = ref(false);
const showSessionPicker = ref(false);
const selectedNode = ref(null);
const hoverNode = ref(null);
const dragging = ref(false);
const pointerWorld = reactive({ x: 0, y: 0 });
const mapView = reactive({ scale: 1, offsetX: 0, offsetY: 0 });
const layerToggles = reactive({
  heatmap: false,
  links: true,
  coverage: true,
  users: true,
  stations: true,
});

const frameCache = new Map();
const frameInflight = new Map();
let playbackTimer = null;
let frameRequestToken = 0;
let animationId = 0;
let resizeObserver = null;
let pointerState = null;
let canvasSize = { width: 1, height: 1, dpr: 1 };
let scenarioMapImage = null;
let scenarioMapUrlLoaded = "";
let scenarioMapReady = false;
const tileImageCache = new Map();
let tileMapCache = { key: "", canvas: null };
let worldPointCache = new WeakMap();
let screenPointCache = new WeakMap();
let stationCategoryCache = new WeakMap();
let gridShapeCache = { frame: null, rows: 10, cols: 12 };
let renderDataCache = { frame: null, users: [], stations: [], drawableLinks: [] };
let throughputSeries = [];
let renderDirty = true;
let chartDirty = true;
let lastRenderAt = 0;
let lastChartAt = 0;
let lastReplayUserNodeLogKey = "";
let terminalScrollFrame = 0;
let hashValueCache = new Map();
const MAX_HASH_CACHE_SIZE = 60000;

const layerButtons = [
  { key: "heatmap", label: "灾损热区" },
  { key: "links", label: "链路" },
  { key: "coverage", label: "覆盖圈" },
  { key: "users", label: "终端" },
  { key: "stations", label: "设备" },
];

const assetUrl = (path) => `${import.meta.env.BASE_URL}prototype/${path}`;

const formatNumber = (value, digits = 0) =>
  Number(value || 0).toLocaleString("zh-CN", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });

const firstPositiveNumber = (...values) => {
  let firstFinite = null;
  for (const value of values) {
    if (value === null || value === undefined || value === "") continue;
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) continue;
    if (firstFinite === null) firstFinite = numeric;
    if (numeric > 0) return numeric;
  }
  return firstFinite ?? 0;
};

const MIN_USABLE_THROUGHPUT_MBPS = 0.05;
const metricSummary = (metrics) => (metrics && typeof metrics === "object" ? metrics.summary || {} : {});
const metricThroughputValue = (metrics) => {
  const summary = metricSummary(metrics);
  return firstPositiveNumber(summary.total_throughput_mbps, summary.avg_throughput_mbps);
};
const metricLatencyValue = (metrics) => firstPositiveNumber(metricSummary(metrics).latency_ms);
const hasUsefulThroughputMetric = (metrics) => metricThroughputValue(metrics) >= MIN_USABLE_THROUGHPUT_MBPS;
const hasUsefulLatencyMetric = (metrics) => metricLatencyValue(metrics) > 0;
const usableLinkMetrics = (metrics) => {
  if (!metrics || typeof metrics !== "object" || !metrics.summary) return null;
  return hasUsefulThroughputMetric(metrics) || hasUsefulLatencyMetric(metrics) ? metrics : null;
};

const formatPercent = (value) => `${(Math.max(0, Math.min(1, Number(value || 0))) * 100).toFixed(1)}%`;

const sourceLabel = (source) => {
  if (source === "training") return "训练回放";
  if (source === "manual") return "人工导入";
  return "测试回放";
};

const fetchJson = async (url) => {
  const response = await fetch(url);
  const rawText = await response.text();
  let payload = null;
  if (rawText) {
    try {
      payload = JSON.parse(rawText);
    } catch {
      payload = null;
    }
  }

  if (!response.ok) {
    const message = payload?.detail || payload?.message || rawText || `请求失败 (${response.status})`;
    throw new Error(message);
  }
  return payload ?? {};
};

const DEVICE_DISPLAY_TEXT_REPLACEMENTS = [
  ["5G_700MHz 应急小区", "5G 700MHz应急基站"],
  ["5G_700MHz应急小区", "5G 700MHz应急基站"],
  ["5G 700MHz 应急小区", "5G 700MHz应急基站"],
  ["5G 700MHz应急小区", "5G 700MHz应急基站"],
  ["5G应急小区", "5G应急基站"],
];

const displayDeviceText = (value) => {
  if (value === null || value === undefined) return "";
  return DEVICE_DISPLAY_TEXT_REPLACEMENTS.reduce(
    (text, [source, target]) => text.replaceAll(source, target),
    String(value)
  );
};

const hashQueryReplayId = () => {
  if (typeof window === "undefined") return null;
  const queryText = window.location.hash.includes("?") ? window.location.hash.split("?").slice(1).join("?") : "";
  return new URLSearchParams(queryText).get("replay_id");
};

const normalizeSession = (session) => {
  const summary = session?.summary || {};
  return {
    ...session,
    id: session?.replay_id || session?.id,
    title: session?.title || session?.replay_id || "未命名回放",
    source: session?.source || "test",
    createdAt: Number(session?.created_at || session?.createdAt || 0),
    scenarioName: session?.scenario_name || session?.scenarioName || "",
    algorithm: session?.algorithm || "--",
    frameCount: Number(session?.frame_count || session?.frameCount || 0),
    nodeCountTotal: Number(session?.node_count_total || session?.nodeCountTotal || 0),
    mapWidth: Number(session?.map_width || session?.mapWidth || MAP_WIDTH),
    mapHeight: Number(session?.map_height || session?.mapHeight || MAP_HEIGHT),
    summary: {
      coverageRatio: Number(summary.coverage_ratio ?? summary.coverageRatio ?? 0),
      broadcastRatio: Number(summary.broadcast_ratio ?? summary.broadcastRatio ?? 0),
      totalReward: Number(summary.total_reward ?? summary.totalReward ?? 0),
      stepsTaken: Number(summary.steps_taken ?? summary.stepsTaken ?? 0),
      totalUsers: Number(summary.total_users ?? summary.totalUsers ?? 0),
      finalStations: Number(summary.final_stations ?? summary.finalStations ?? 0),
      connectedUsers: Number(summary.connected_users ?? summary.connectedUsers ?? 0),
      broadcastUsers: Number(summary.broadcast_users ?? summary.broadcastUsers ?? 0),
    },
  };
};

const normalizeFrame = (frame) => {
  const metrics = frame?.metrics || {};
  return {
    ...frame,
    frameIndex: Number(frame?.frame_index ?? frame?.frameIndex ?? 0),
    time: Number(frame?.time ?? frame?.frame_index ?? frame?.frameIndex ?? 0),
    mapWidth: Number(frame?.map_width || frame?.mapWidth || MAP_WIDTH),
    mapHeight: Number(frame?.map_height || frame?.mapHeight || MAP_HEIGHT),
    nodes: Array.isArray(frame?.nodes) ? frame.nodes : [],
    links: Array.isArray(frame?.links) ? frame.links : [],
    tp: Number(frame?.tp ?? metrics.avg_user_throughput ?? 0),
    loss: Number(frame?.loss ?? metrics.loss_ratio ?? 0),
    coverageRatio: Number(frame?.coverageRatio ?? frame?.coverage_ratio ?? metrics.coverage_ratio ?? 0),
    broadcastRatio: Number(frame?.broadcastRatio ?? frame?.broadcast_ratio ?? metrics.broadcast_ratio ?? 0),
    remainingBudget: Number(frame?.remainingBudget ?? frame?.remaining_budget ?? metrics.remaining_budget ?? 0),
    reward: Number(frame?.reward || 0),
    connectedUsers: Number(frame?.connected_users ?? metrics.connected_users ?? 0),
    broadcastUsers: Number(frame?.broadcast_users ?? metrics.broadcast_users ?? 0),
    userCount: Number(frame?.user_count ?? metrics.user_count ?? 0),
    stationCount: Number(frame?.station_count ?? metrics.station_count ?? 0),
    nodesTotal: Number(frame?.nodes_total ?? frame?.node_count_total ?? metrics.node_count_total ?? 0),
    nodesDrawn: Number(frame?.nodes_drawn ?? frame?.nodes?.length ?? 0),
  };
};

const activeSession = computed(
  () => activeSessionDetail.value || sessions.value.find((session) => session.id === activeSessionId.value) || null
);

const scenarioTheme = computed(() => {
  const text = `${activeSession.value?.scenarioName || ""} ${activeSession.value?.scenario?.name || ""} ${activeSession.value?.scenario?.disaster_type || ""}`.toLowerCase();
  if (text.includes("typhoon") || text.includes("台风") || text.includes("风暴潮")) {
    return {
      key: "typhoon",
      tint: "rgba(0, 240, 255, 0.16)",
      water: "rgba(0, 92, 135, 0.42)",
      land: "rgba(18, 72, 68, 0.5)",
      road: "rgba(0, 255, 157, 0.16)",
    };
  }
  if (
    text.includes("rain") ||
    text.includes("flood") ||
    text.includes("rainstorm") ||
    text.includes("暴雨") ||
    text.includes("洪水") ||
    text.includes("内涝")
  ) {
    return {
      key: "rainstorm",
      tint: "rgba(0, 132, 255, 0.18)",
      water: "rgba(0, 88, 160, 0.34)",
      land: "rgba(18, 62, 76, 0.5)",
      road: "rgba(109, 204, 255, 0.18)",
    };
  }
  if (text.includes("earthquake") || text.includes("quake") || text.includes("地震")) {
    return {
      key: "earthquake",
      tint: "rgba(255, 0, 85, 0.15)",
      water: "rgba(70, 50, 70, 0.24)",
      land: "rgba(88, 48, 40, 0.52)",
      road: "rgba(255, 183, 0, 0.18)",
    };
  }
  return {
    key: "default",
    tint: "rgba(0, 240, 255, 0.14)",
    water: "rgba(0, 78, 120, 0.28)",
    land: "rgba(18, 62, 76, 0.42)",
    road: "rgba(0, 240, 255, 0.16)",
  };
});

const scenarioMapUrl = computed(() => {
  const scenario = activeSession.value?.scenario || {};
  const frame = currentFrame.value || {};
  return (
    scenario.map_image_url ||
    scenario.map_url ||
    frame.map_image_url ||
    frame.map_url ||
    "/ns3-native/map.png"
  );
});

const scenarioGeoBounds = computed(() => {
  const frame = currentFrame.value || {};
  const session = activeSession.value || {};
  return normalizeGeoBounds(
    frame.geo_bounds ||
      frame.geoBounds ||
      session.geo_bounds ||
      session.geoBounds ||
      session.scenario?.geo_bounds ||
      session.scenario?.geoBounds ||
      session.scenario?.region_grid?.geo_bounds
  );
});

const maxFrameIndex = computed(() => Math.max(0, Number(activeSession.value?.frameCount || 0) - 1));

const currentUsableLinkMetrics = computed(() =>
  usableLinkMetrics(linkMetrics.value) ||
  usableLinkMetrics(currentFrame.value?.link_metrics) ||
  usableLinkMetrics(currentFrame.value?.linkMetrics)
);

const linkSummary = computed(() => currentUsableLinkMetrics.value?.summary || {});

const clusterThroughputMbps = computed(() => {
  const currentValue = firstPositiveNumber(
    linkSummary.value.total_throughput_mbps,
    linkSummary.value.avg_throughput_mbps,
    currentFrame.value?.cluster_throughput_mbps,
    currentFrame.value?.metrics?.cluster_throughput_mbps,
    currentFrame.value?.metrics?.total_throughput_mbps,
    currentFrame.value?.tp,
    currentFrame.value?.metrics?.avg_user_throughput
  );
  if (currentValue >= MIN_USABLE_THROUGHPUT_MBPS || lastStableTelemetry.value.throughput <= 0) {
    return currentValue;
  }
  return lastStableTelemetry.value.throughput;
});

const endToEndLatencyMs = computed(() => {
  const currentValue = firstPositiveNumber(
    linkSummary.value.latency_ms,
    currentFrame.value?.latency_ms,
    currentFrame.value?.metrics?.latency_ms
  );
  if (currentValue > 0 || lastStableTelemetry.value.latency <= 0) {
    return currentValue;
  }
  return lastStableTelemetry.value.latency;
});

const replayProgress = computed(() => {
  const max = Math.max(1, maxFrameIndex.value);
  return Math.max(0, Math.min(1, frameIndex.value / max));
});

const displayReplaySeconds = computed(() => Number(frameIndex.value || 0) * REPLAY_SECONDS_PER_FRAME);

const nativePhaseKey = computed(() => {
  if (!activeSession.value) return "normal";
  if (phaseKey.value === "completed") return "restored";
  if (phaseKey.value === "recovery" || phaseKey.value === "deploying") return "recovery";
  return "disaster";
});

const currentTimeDisplay = computed(() => formatNumber(displayReplaySeconds.value, 1));

const maxTimeDisplay = computed(() => {
  const endTime = Number(maxFrameIndex.value || 0) * REPLAY_SECONDS_PER_FRAME;
  return formatNumber(endTime, 0);
});

const timelineStartText = computed(() => {
  const seconds = Number(displayReplaySeconds.value || 0);
  const minute = Math.floor(seconds / 60);
  const second = Math.floor(seconds % 60);
  return `${String(minute).padStart(2, "0")}:${String(second).padStart(2, "0")}`;
});

const activeNodesCount = computed(() => {
  const frame = currentFrame.value || {};
  return formatNumber(frame.connectedUsers || frame.broadcastUsers || (frame.nodes || []).filter((node) => node.connected !== false).length || 0);
});

const activeLinksCount = computed(() => formatNumber(linkSummary.value.active_links || currentFrame.value?.links?.length || 0));
const throughputDisplay = computed(() => formatNumber(clusterThroughputMbps.value, 1));
const latencyDisplay = computed(() => formatNumber(endToEndLatencyMs.value, 0));

const nativeStatusClass = computed(() => {
  if (nativePhaseKey.value === "disaster") return "disaster";
  if (nativePhaseKey.value === "recovery" || nativePhaseKey.value === "restored") return "recovery";
  return "normal";
});

const nativeStatusDotClass = computed(() => {
  if (nativePhaseKey.value === "disaster") return "red";
  if (nativePhaseKey.value === "recovery" || nativePhaseKey.value === "restored") return "cyan";
  return "green";
});

const nativeStatusText = computed(() => {
  if (!activeSession.value) return "通信枢纽运行正常";
  if (nativePhaseKey.value === "disaster") return "极端灾害爆发 | 宏网熔断";
  if (nativePhaseKey.value === "restored") return "通信网络已恢复 | 覆盖稳定";
  return "应急链路重构中 | 网络恢复推进";
});

const nativeAlertText = computed(() => {
  if (!activeSession.value) return "请选择后端场景回放记录";
  if (nativePhaseKey.value === "disaster") return "监测到极端灾害 | 宏基站受损 | 应急自组网协议已激活";
  if (nativePhaseKey.value === "restored") return "主干链路已恢复 | 宏站回传重建 | 通信网络趋于稳定";
  return "应急链路接管中 | 回传骨干重建 | 覆盖持续回升";
});

const phaseKey = computed(() => {
  if (!activeSession.value) return "idle";
  const coverage = Number(currentFrame.value?.coverageRatio || 0);
  if (frameIndex.value <= 0) return "damaged";
  if (frameIndex.value >= maxFrameIndex.value && maxFrameIndex.value > 0) return "completed";
  if (replayProgress.value >= 0.74 || coverage >= 0.78) return "recovery";
  if (replayProgress.value >= 0.24 || coverage >= 0.28) return "deploying";
  return "damaged";
});

const stageDetail = computed(() => {
  const progress = replayProgress.value;
  if (!activeSession.value) return { id: "idle", label: "等待接入", type: "SYSTEM" };
  if (frameIndex.value >= maxFrameIndex.value && maxFrameIndex.value > 0) {
    return { id: "completed", label: "收敛归档", type: "SYS_FINISH" };
  }
  if (progress < 0.08) return { id: "monitor", label: "灾害监测", type: "SENSOR" };
  if (progress < 0.22) return { id: "damage", label: "宏站失效", type: "ALERT" };
  if (progress < 0.38) return { id: "cluster", label: "需求聚类", type: "TOPOLOGY" };
  if (progress < 0.58) return { id: "dispatch", label: "设备调度", type: "CMD_EXEC" };
  if (progress < 0.76) return { id: "links", label: "链路重构", type: "ROUTE" };
  if (progress < 0.92) return { id: "coverage", label: "覆盖恢复", type: "COVERAGE" };
  return { id: "stabilize", label: "稳定收敛", type: "METRIC" };
});

const phaseStatusText = computed(() => {
  const map = {
    idle: "等待连接后端回放",
    damaged: "极端灾害影响 | 残余网络受损",
    deploying: "策略部署推进 | 应急链路接管",
    recovery: "覆盖持续恢复 | 回传链路重构",
    completed: "回放完成 | 网络恢复稳定",
  };
  return map[phaseKey.value] || map.idle;
});

const phaseAlert = computed(() => {
  if (!activeSession.value) return "请选择后端回放会话";
  const coverage = formatPercent(currentFrame.value?.coverageRatio || 0);
  const deployment = currentFrame.value?.latest_deployment || currentFrame.value?.latestDeployment || null;
  if (phaseKey.value === "damaged") {
    return `监测到灾害损毁，当前覆盖率 ${coverage}，等待策略部署接管。`;
  }
  if (phaseKey.value === "completed") {
    return `主干链路已恢复，最终覆盖率 ${coverage}，场景回放完成。`;
  }
  if (deployment?.device?.device_label || deployment?.device_label) {
    const label = displayDeviceText(deployment?.device?.device_label || deployment?.device_label);
    const grid = deployment?.grid;
    const gridText = grid ? `G_${grid.row}_${grid.col}` : deployment?.region_label || "目标网格";
    const coordinateText = deploymentCoordinateLabel(deployment, deployment?.time_step ?? deployment?.sequence);
    return `${label} 已部署到 ${coordinateText || gridText}，覆盖率提升至 ${coverage}。`;
  }
  return `应急链路接管中，当前覆盖率 ${coverage}。`;
});

const phaseDotClass = computed(() => {
  if (phaseKey.value === "completed" || phaseKey.value === "recovery") return "cyan";
  if (phaseKey.value === "deploying") return "yellow";
  if (phaseKey.value === "damaged") return "red";
  return "green";
});

const sessionSubtitle = computed(() => {
  if (loadingSessions.value) return "正在连接后端回放会话";
  if (!activeSession.value) return replayError.value || "连接后端回放会话后显示真实策略测试数据";
  return `${sourceLabel(activeSession.value.source)} / ${String(activeSession.value.algorithm).toUpperCase()} / ${formatNumber(activeSession.value.nodeCountTotal)} 节点 / ${activeSession.value.frameCount} 帧`;
});

const terminalSubtitle = computed(() => {
  if (!activeSession.value) return "未选择回放";
  if (!hasReplayStarted.value) return "等待播放";
  if (playing.value) return "逐帧输出中";
  if (frameIndex.value >= maxFrameIndex.value && maxFrameIndex.value > 0) return "回放完成";
  return "已暂停";
});

const phaseStages = computed(() => {
  const max = maxFrameIndex.value;
  return [
    { id: "stage-monitor", phase: "damaged", label: "灾害监测", frame: 0 },
    { id: "stage-damaged", phase: "damaged", label: "宏站失效", frame: Math.min(max, Math.max(1, Math.round(max * 0.12))) },
    { id: "stage-cluster", phase: "deploying", label: "需求聚类", frame: Math.min(max, Math.round(max * 0.3)) },
    { id: "stage-dispatch", phase: "deploying", label: "设备调度", frame: Math.min(max, Math.round(max * 0.48)) },
    { id: "stage-link", phase: "deploying", label: "链路重构", frame: Math.min(max, Math.round(max * 0.64)) },
    { id: "stage-recovery", phase: "recovery", label: "覆盖恢复", frame: Math.min(max, Math.round(max * 0.82)) },
    { id: "stage-completed", phase: "completed", label: "回放完成", frame: max },
  ];
});

const telemetryCards = computed(() => {
  const frame = currentFrame.value || {};
  return [
    { label: "推演时间", value: formatNumber(frame.time ?? frame.frameIndex ?? 0, 1), unit: "s" },
    { label: "帧渲染率", value: playing.value ? "30" : "--", unit: "fps" },
    { label: "在线节点", value: formatNumber(frame.connectedUsers || 0), unit: "个" },
    { label: "活跃链路", value: formatNumber(linkSummary.value.active_links || frame.links?.length || 0), unit: "条" },
    { label: "集群吞吐量", value: formatNumber(clusterThroughputMbps.value, 1), unit: "Mbps" },
    { label: "端到端延迟", value: formatNumber(endToEndLatencyMs.value, 0), unit: "ms" },
  ];
});

const stationCategory = (node) => {
  if (node && typeof node === "object") {
    const cached = stationCategoryCache.get(node);
    if (cached) return cached;
  }
  const text = `${node?.type || ""} ${node?.base_station || ""} ${node?.device_type || ""} ${node?.device_label || ""} ${node?.label || ""}`.toLowerCase();
  let category = "manpack";
  if (text.includes("relay") || text.includes("mesh") || text.includes("wifi") || text.includes("satellite") || text.includes("ka")) category = "relay";
  else if (text.includes("manpack") || text.includes("shortwave") || text.includes("hf") || text.includes("背负")) category = "manpack";
  else if (text.includes("small_cell") || text.includes("small") || text.includes("micro") || text.includes("微站") || text.includes("微型")) category = "smallCell";
  else if (text.includes("macro") || text.includes("700") || text.includes("5g") || text.includes("宏")) category = "macro";
  if (node && typeof node === "object") stationCategoryCache.set(node, category);
  return category;
};

const stationCounts = computed(() => {
  const stations = (currentFrame.value?.nodes || []).filter((node) => !isUserNode(node));
  const counts = {
    manpack: { online: 0, total: 0 },
    relay: { online: 0, total: 0 },
    smallCell: { online: 0, total: 0 },
    macro: { online: 0, total: 0 },
  };
  stations.forEach((node) => {
    const key = stationCategory(node);
    counts[key].total += 1;
    if (node.connected !== false && node.status !== "offline") counts[key].online += 1;
  });
  return counts;
});

const scenarioCounts = computed(() => ({
  user: formatNumber(currentFrame.value?.userCount || activeSession.value?.summary?.totalUsers || 0),
  macro: formatNumber(stationCounts.value.macro.total),
  manpack: formatNumber(stationCounts.value.manpack.total),
  smallCell: formatNumber(stationCounts.value.smallCell.total),
  relay: formatNumber(stationCounts.value.relay.total),
}));

const equipmentCards = computed(() => {
  const counts = stationCounts.value;
  return [
    { key: "macro", label: "宏基站", color: "#00F0FF", icon: assetUrl("images/场景回放/u3725.png"), ...counts.macro },
    { key: "manpack", label: "背负式基站", color: "#FFB700", icon: assetUrl("images/场景回放/u3711.png"), ...counts.manpack },
    { key: "smallCell", label: "微型基站", color: "#B026FF", icon: assetUrl("images/场景回放/u3721.png"), ...counts.smallCell },
    { key: "relay", label: "自组网中继节点", color: "#FF0055", icon: assetUrl("images/场景回放/u3716.png"), ...counts.relay },
  ];
});

const visibleTerminalLines = computed(() => {
  if (!hasReplayStarted.value || !activeSession.value) return [];
  const current = frameIndex.value;
  const intro = [];
  const eventLines = [];
  const frameLines = [];
  const summary = [];

  allLogLines.value.forEach((line) => {
    if (/^\[(SESSION|SCENARIO|NODES)\]/.test(line)) {
      intro.push(line);
      return;
    }
    const deployMatch = line.match(/^\[DEPLOY\]\s+t=(\d+)/);
    if (deployMatch) {
      if (Number(deployMatch[1]) <= current) eventLines.push(line);
      return;
    }
    const frameMatch = line.match(/^\[FRAME\s+(\d+)\]/);
    if (frameMatch) {
      if (Number(frameMatch[1]) <= current) frameLines.push(line);
      return;
    }
    if (/^\[SUMMARY\]/.test(line)) {
      if (current >= maxFrameIndex.value) summary.push(line);
      return;
    }
    if (eventLines.length < current + 6) eventLines.push(line);
  });

  const lines = [...intro, ...eventLines, ...frameLines.slice(-Math.max(10, current + 1)), ...summary];
  return lines.slice(-160);
});

const stageLogEntries = computed(() => {
  if (!hasReplayStarted.value || !activeSession.value) return [];
  const coverage = formatPercent(currentFrame.value?.coverageRatio || 0);
  const connected = formatNumber(currentFrame.value?.connectedUsers || currentFrame.value?.broadcastUsers || 0);
  const messages = {
    "stage-monitor": ["SENSOR", "接入灾害场景边界，初始化暗色地图瓦片与网格损毁模型。"],
    "stage-damaged": ["ALERT", "检测到宏站离线与用户终端失联，灾损热区开始扩散。"],
    "stage-cluster": ["TOPOLOGY", `汇聚离线用户需求簇，当前恢复覆盖率 ${coverage}。`],
    "stage-dispatch": ["CMD_EXEC", "策略输出部署序列，设备编队按精确部署坐标入场。"],
    "stage-link": ["ROUTE", "D2D/Mesh 与回传链路开始重构，链路数据包持续探测。"],
    "stage-recovery": ["COVERAGE", `覆盖圈扩张，已恢复用户 ${connected} 个，网络进入恢复态。`],
    "stage-completed": ["SYS_FINISH", `最终覆盖率 ${coverage}，场景回放完成并归档。`],
  };
  return phaseStages.value
    .filter((stage) => frameIndex.value >= stage.frame)
    .map((stage) => {
      const [type, message] = messages[stage.id] || ["SYSTEM", stage.label];
      const seconds = Number(stage.frame || 0) * REPLAY_SECONDS_PER_FRAME;
      return {
        level: type.includes("ALERT") ? "ERROR" : type.includes("CMD") || type.includes("ROUTE") ? "INFO" : "SYSTEM",
        text: `[T+${seconds.toFixed(2)}s][${type}] ${message}`,
      };
    });
});

const parseLogLine = (line, fallbackIndex = 0) => {
  const text = String(line || "");
  if (text.startsWith(">")) {
    return { level: "SYSTEM", text };
  }
  const deployMatch = text.match(/^\[DEPLOY\]\s+t=(\d+)\s+(.+)$/);
  if (deployMatch) {
    const coordinateText = deploymentCoordinateLabel(null, Number(deployMatch[1]));
    const suffix = coordinateText && !/\b(coord|target)=|XY\(/i.test(deployMatch[2]) ? ` target=${coordinateText}` : "";
    const seconds = Number(deployMatch[1]) * REPLAY_SECONDS_PER_FRAME;
    return {
      level: "SYSTEM",
      text: displayDeviceText(`[T+${seconds.toFixed(2)}s][CMD_EXEC] ${deployMatch[2]}${suffix}`),
    };
  }
  const frameMatch = text.match(/^\[FRAME\s+(\d+)\]\s+(.+)$/);
  if (frameMatch) {
    const seconds = Number(frameMatch[1]) * REPLAY_SECONDS_PER_FRAME;
    return {
      level: "INFO",
      text: displayDeviceText(`[T+${seconds.toFixed(2)}s][TOPOLOGY] ${frameMatch[2]}`),
    };
  }
  const typedMatch = text.match(/^\[([A-Z_]+)\]\s*(.*)$/);
  if (typedMatch) {
    const type = typedMatch[1];
    const level = type.includes("ERROR") || type.includes("ALERT") ? "ERROR" : type.includes("WARN") ? "WARN" : "SYSTEM";
    return {
      level,
      text: displayDeviceText(`[T+${Number(displayReplaySeconds.value ?? fallbackIndex).toFixed(2)}s][${type}] ${typedMatch[2] || text}`),
    };
  }
  return {
    level: "INFO",
    text: displayDeviceText(`[T+${Number(displayReplaySeconds.value ?? fallbackIndex).toFixed(2)}s][DATA_TX] ${text}`),
  };
};

const terminalEntryTime = (entry) => {
  const match = String(entry?.text || "").match(/\[T\+([\d.]+)s\]/);
  return match ? Number(match[1]) : -1;
};

const displayTerminalLevel = (level) => {
  const raw = String(level || "INFO").toUpperCase();
  if (raw.includes("ERROR") || raw.includes("FAIL")) return "ERROR";
  if (raw.includes("WARN")) return "WARN";
  if (raw.includes("SYSTEM") || raw.includes("STATUS")) return "SYSTEM";
  return "INFO";
};

const sharedTerminalEntries = computed(() =>
  terminalHistoryLines.value.slice(-140).map((line) => {
    const level = String(line).match(/\[([A-Z_]+)\]/)?.[1] || "INFO";
    return {
      level: displayTerminalLevel(level),
      text: displayDeviceText(line),
    };
  })
);

const localReplayTerminalEntries = computed(() => {
  if (replayTerminalCleared.value && !logQueryMode.value) return [];
  if (logQueryMode.value && logQuery.value) {
    const keyword = logQuery.value.toLowerCase();
    const filtered = allLogLines.value
      .filter((line) => String(line).toLowerCase().includes(keyword))
      .slice(-200)
      .reverse()
      .map(parseLogLine);
    if (!filtered.length) {
      return [{ level: "WARN", text: `> 检索完毕，数据库中不存在包含 "${logQuery.value}" 的日志。` }];
    }
    return [
      { level: "SYSTEM", text: `> 检索成功，共捕获历史数据 ${filtered.length} 条记录 (仅展示最新 200 条)：` },
      ...filtered,
    ];
  }

  if (!activeSession.value) {
    return [{ level: "SYSTEM", text: "> SYSTEM INTIALIZED..." }];
  }

  if (!hasReplayStarted.value) {
    return [
      { level: "SYSTEM", text: "> SYSTEM INTIALIZED..." },
      {
        level: "SYSTEM",
        text: `[T+0.00s][SYS_INIT] 数字孪生框架构建完毕，已加载 ${activeSession.value.title}。`,
      },
    ];
  }

  return [...visibleTerminalLines.value.map(parseLogLine), ...stageLogEntries.value, ...clientLogEntries.value]
    .sort((a, b) => terminalEntryTime(a) - terminalEntryTime(b))
    .slice(-180);
});

const displayTerminalEntries = computed(() => {
  const seen = new Set();
  return [...sharedTerminalEntries.value, ...localReplayTerminalEntries.value]
    .filter((entry) => {
      const text = String(entry?.text || "");
      if (!text || seen.has(text)) return false;
      seen.add(text);
      return true;
    })
    .slice(-180);
});

const selectedNodeStyle = computed(() => {
  if (!selectedNode.value) return {};
  const point = screenPointForNode(selectedNode.value);
  return {
    left: `${Math.min(Math.max(point.x + 18, 16), canvasSize.width - 260)}px`,
    top: `${Math.min(Math.max(point.y - 16, 90), canvasSize.height - 170)}px`,
  };
});

const isUserNode = (node) => node?.type === "USER" || Number(node?.type) === 0 || node?.node_role === "user";

const nodeTitle = (node) => {
  if (isUserNode(node)) return String(node.id || "用户终端");
  return displayDeviceText(node.device_label || node.label || node.base_station || node.device_type || "应急设备");
};

const nodeSubtitle = (node) => {
  if (isUserNode(node)) {
    if (node.connected) return "通信已恢复";
    if (node.broadcast_served) return "已纳入广播覆盖";
    return "通信中断";
  }
  const role = node.node_role === "residual_base_station" ? "残余网络节点" : "策略部署设备";
  return `${role} / ${node.mode || node.broadcast_mode || node.device_type || "--"}`;
};

const formatSessionMeta = (session) => {
  const created = Number(session.createdAt || 0)
    ? new Date(Number(session.createdAt) * 1000).toLocaleString("zh-CN", {
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
      })
    : "--";
  return `${created} · ${formatNumber(session.nodeCountTotal)} 节点 · ${session.frameCount} 帧`;
};

function markReplayStarted() {
  hasReplayStarted.value = true;
}

function markRenderDirty(options = {}) {
  renderDirty = true;
  if (options.chart) chartDirty = true;
  if (options.frame) {
    worldPointCache = new WeakMap();
    screenPointCache = new WeakMap();
    stationCategoryCache = new WeakMap();
    gridShapeCache = { frame: null, rows: 10, cols: 12 };
    renderDataCache = { frame: null, users: [], stations: [], drawableLinks: [] };
  }
  if (options.frame || options.map) {
    screenPointCache = new WeakMap();
    tileMapCache = { key: "", canvas: null };
  }
}

function rememberFrame(cacheKey, payload) {
  if (!frameCache.has(cacheKey) && frameCache.size >= FRAME_CACHE_LIMIT) {
    const firstKey = frameCache.keys().next().value;
    if (firstKey) frameCache.delete(firstKey);
  }
  frameCache.set(cacheKey, payload);
}

function frameSampleRatio(options = {}) {
  if (options.preferDetail || !playing.value) return FRAME_SAMPLE_RATIO;
  return PLAYING_FRAME_SAMPLE_RATIO;
}

function frameCacheKey(id, index, sampleRatio = FRAME_SAMPLE_RATIO) {
  return `${id}:${Math.max(0, Number(index || 0))}:r${sampleRatio}`;
}

function normalizeLinkMetrics(metrics) {
  return usableLinkMetrics(metrics);
}

function frameThroughputValue(frame, metrics = null) {
  const summary = metrics?.summary || frame?.link_metrics?.summary || frame?.linkMetrics?.summary || {};
  return firstPositiveNumber(
    summary.total_throughput_mbps,
    summary.avg_throughput_mbps,
    frame?.cluster_throughput_mbps,
    frame?.metrics?.cluster_throughput_mbps,
    frame?.metrics?.total_throughput_mbps,
    frame?.tp,
    frame?.metrics?.avg_user_throughput
  );
}

function frameLatencyValue(frame, metrics = null) {
  const summary = metrics?.summary || frame?.link_metrics?.summary || frame?.linkMetrics?.summary || {};
  return firstPositiveNumber(
    summary.latency_ms,
    frame?.latency_ms,
    frame?.metrics?.latency_ms
  );
}

function rememberStableTelemetry(frame, metrics = null) {
  const throughput = frameThroughputValue(frame, metrics);
  const latency = frameLatencyValue(frame, metrics);
  if (throughput >= MIN_USABLE_THROUGHPUT_MBPS || latency > 0) {
    lastStableTelemetry.value = {
      throughput: throughput >= MIN_USABLE_THROUGHPUT_MBPS ? throughput : lastStableTelemetry.value.throughput,
      latency: latency > 0 ? latency : lastStableTelemetry.value.latency,
    };
  }
}

function stableThroughputSeriesValue(index, frame, metrics = null) {
  const value = frameThroughputValue(frame, metrics);
  if (value >= MIN_USABLE_THROUGHPUT_MBPS) return value;
  for (let cursor = Number(index || 0) - 1; cursor >= 0; cursor -= 1) {
    const previous = Number(throughputSeries[cursor]);
    if (Number.isFinite(previous) && previous >= MIN_USABLE_THROUGHPUT_MBPS) return previous;
  }
  return lastStableTelemetry.value.throughput >= MIN_USABLE_THROUGHPUT_MBPS ? lastStableTelemetry.value.throughput : value;
}

function cachedFrameEntry(id, index, sampleRatio) {
  const detailEntry = frameCache.get(frameCacheKey(id, index, FRAME_SAMPLE_RATIO));
  if (detailEntry) return { cacheKey: frameCacheKey(id, index, FRAME_SAMPLE_RATIO), entry: detailEntry };
  const key = frameCacheKey(id, index, sampleRatio);
  return { cacheKey: key, entry: frameCache.get(key) || null };
}

async function fetchReplayFramePayload(id, numericIndex, sampleRatio) {
  const cacheKey = frameCacheKey(id, numericIndex, sampleRatio);
  const cached = frameCache.get(cacheKey);
  if (cached) return { cacheKey, entry: cached };
  const pending = frameInflight.get(cacheKey);
  if (pending) return pending;

  const request = fetchJson(
    `${API_BASE}/replay/sessions/${encodeURIComponent(id)}/frames/${numericIndex}?sample_ratio=${sampleRatio}&include_links=true`
  )
    .then((framePayload) => {
      const frame = normalizeFrame(framePayload);
      const entry = {
        frame,
        linkMetrics: normalizeLinkMetrics(framePayload?.link_metrics || framePayload?.linkMetrics),
        sampleRatio,
      };
      rememberFrame(cacheKey, entry);
      return { cacheKey, entry };
    })
    .finally(() => {
      frameInflight.delete(cacheKey);
    });
  frameInflight.set(cacheKey, request);
  return request;
}

function prefetchReplayFrames(id, numericIndex) {
  if (!playing.value || !id) return;
  for (let offset = 1; offset <= FRAME_PREFETCH_AHEAD; offset += 1) {
    const nextIndex = numericIndex + offset;
    if (nextIndex > maxFrameIndex.value) break;
    const sampleRatio = frameSampleRatio();
    if (cachedFrameEntry(id, nextIndex, sampleRatio).entry) continue;
    void fetchReplayFramePayload(id, nextIndex, sampleRatio).catch(() => {});
  }
}

const syncedReplayTerminalKeys = new Set();

function replayTerminalKey(text) {
  return `${activeSessionId.value || "replay"}:${String(text || "")}`;
}

function appendReplayTerminalLine(message, options = {}) {
  if (!message) return;
  replayTerminalCleared.value = false;
  appendSharedTerminalLine(displayDeviceText(message), {
    level: options.level || "INFO",
    source: options.source || "REPLAY",
    timestamp: options.timestamp,
  });
}

function appendReplayUserNodeCount(prefix, ...sources) {
  const key = userNodeCountLogKey(`ns3-replay:${activeSessionId.value || ""}:${prefix}`, ...sources);
  if (key === lastReplayUserNodeLogKey) return;
  lastReplayUserNodeLogKey = key;
  appendReplayTerminalLine(buildUserNodeCountMessage(prefix, ...sources), { level: "SCENE" });
}

function syncReplayTerminalEntries(entries = []) {
  const pendingEntries = [];
  entries.forEach((entry) => {
    const text = displayDeviceText(entry?.text);
    if (!text || text === "> SYSTEM INTIALIZED...") return;
    const key = replayTerminalKey(text);
    if (syncedReplayTerminalKeys.has(key)) return;
    syncedReplayTerminalKeys.add(key);
    pendingEntries.push({
      text,
      level: entry.level || "INFO",
      source: "REPLAY",
    });
  });
  if (pendingEntries.length) {
    replayTerminalCleared.value = false;
    appendSharedTerminalLines(pendingEntries, { source: "REPLAY" });
  }
  if (syncedReplayTerminalKeys.size > 700) {
    syncedReplayTerminalKeys.clear();
  }
}

function downloadTerminalLog() {
  exportTerminalOutput(terminalHistoryLines.value, "rescuenet-replay-terminal.log");
}

function clearTerminalLog() {
  clearTerminalOutput();
  allLogLines.value = [];
  clientLogEntries.value = [];
  logQuery.value = "";
  logQueryMode.value = false;
  syncedReplayTerminalKeys.clear();
  replayTerminalCleared.value = true;
}

function appendClientLog(level, type, message) {
  const simTime = Number(displayReplaySeconds.value || 0).toFixed(2);
  clientLogEntries.value.push({
    level,
    text: `[T+${simTime}s][${type}] ${message}`,
  });
  clientLogEntries.value = clientLogEntries.value.slice(-60);
}

function toggleLayer(key) {
  layerToggles[key] = !layerToggles[key];
  markRenderDirty();
  drawReplayMap();
}

function refreshCanvasSize() {
  const canvas = canvasRef.value;
  const shell = mapShellRef.value;
  if (!canvas || !shell) return;
  const rect = shell.getBoundingClientRect();
  // The prototype shell is CSS-transformed; layout size keeps the canvas from double-scaling.
  const layoutWidth = shell.clientWidth || shell.offsetWidth || rect.width;
  const layoutHeight = shell.clientHeight || shell.offsetHeight || rect.height;
  const dpr = 1;
  canvasSize = {
    width: Math.max(1, Math.round(layoutWidth)),
    height: Math.max(1, Math.round(layoutHeight)),
    dpr,
  };
  canvas.width = Math.round(canvasSize.width * dpr);
  canvas.height = Math.round(canvasSize.height * dpr);
  canvas.style.width = `${canvasSize.width}px`;
  canvas.style.height = `${canvasSize.height}px`;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  markRenderDirty({ chart: true });
  drawReplayMap();
}

function baseTransform() {
  const width = Number(currentFrame.value?.mapWidth || activeSession.value?.mapWidth || MAP_WIDTH);
  const height = Number(currentFrame.value?.mapHeight || activeSession.value?.mapHeight || MAP_HEIGHT);
  const paddingX = Math.min(24, canvasSize.width * 0.02);
  const reservedTop = Math.min(146, canvasSize.height * 0.18);
  const reservedBottom = Math.min(286, canvasSize.height * 0.32);
  const drawWidth = Math.max(1, canvasSize.width - paddingX * 2);
  const drawHeight = Math.max(1, canvasSize.height - reservedTop - reservedBottom);
  const scaleX = (drawWidth / width) * mapView.scale;
  const scaleY = (drawHeight / height) * mapView.scale;
  return {
    scale: Math.min(scaleX, scaleY),
    scaleX,
    scaleY,
    offsetX: paddingX + mapView.offsetX,
    offsetY: reservedTop + mapView.offsetY,
    drawWidth,
    drawHeight,
    width,
    height,
  };
}

function worldToScreen(x, y, transform = null) {
  const t = transform || baseTransform();
  return {
    x: t.offsetX + x * t.scaleX,
    y: t.offsetY + y * t.scaleY,
  };
}

function hashNumber(value) {
  const text = String(value ?? "");
  const cached = hashValueCache.get(text);
  if (cached !== undefined) return cached;
  let hash = 2166136261;
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  hash += hash << 13;
  hash ^= hash >>> 7;
  hash += hash << 3;
  hash ^= hash >>> 17;
  hash += hash << 5;
  const normalized = (hash >>> 0) / 4294967296;
  if (hashValueCache.size > MAX_HASH_CACHE_SIZE) hashValueCache = new Map();
  hashValueCache.set(text, normalized);
  return normalized;
}

function normalizeGeoBounds(bounds) {
  if (!bounds) return null;
  const latMin = Number(bounds.lat_min ?? bounds.latMin);
  const latMax = Number(bounds.lat_max ?? bounds.latMax);
  const lonMin = Number(bounds.lon_min ?? bounds.lonMin);
  const lonMax = Number(bounds.lon_max ?? bounds.lonMax);
  if (![latMin, latMax, lonMin, lonMax].every(Number.isFinite)) return null;
  if (Math.abs(latMax - latMin) < 0.0001 || Math.abs(lonMax - lonMin) < 0.0001) return null;
  return {
    latMin: Math.min(latMin, latMax),
    latMax: Math.max(latMin, latMax),
    lonMin: Math.min(lonMin, lonMax),
    lonMax: Math.max(lonMin, lonMax),
  };
}

function mercatorProject(lat, lon, zoom) {
  const size = TILE_SIZE * 2 ** zoom;
  const safeLat = Math.max(-85.05112878, Math.min(85.05112878, Number(lat)));
  const sin = Math.sin((safeLat * Math.PI) / 180);
  return {
    x: ((Number(lon) + 180) / 360) * size,
    y: (0.5 - Math.log((1 + sin) / (1 - sin)) / (4 * Math.PI)) * size,
  };
}

function mapViewport(width, height, bounds) {
  if (!bounds) return null;
  let bestZoom = 5;
  for (let zoom = 5; zoom <= 14; zoom += 1) {
    const northWest = mercatorProject(bounds.latMax, bounds.lonMin, zoom);
    const southEast = mercatorProject(bounds.latMin, bounds.lonMax, zoom);
    const spanX = Math.abs(southEast.x - northWest.x);
    const spanY = Math.abs(southEast.y - northWest.y);
    if (spanX <= width * 0.82 && spanY <= height * 0.82) {
      bestZoom = zoom;
    }
  }
  const center = mercatorProject((bounds.latMin + bounds.latMax) / 2, (bounds.lonMin + bounds.lonMax) / 2, bestZoom);
  return {
    zoom: bestZoom,
    left: center.x - width / 2,
    top: center.y - height / 2,
  };
}

function cartoTileUrl(zoom, x, y) {
  const subdomains = ["a", "b", "c", "d"];
  const subdomain = subdomains[Math.abs(x + y) % subdomains.length];
  return `https://${subdomain}.basemaps.cartocdn.com/dark_all/${zoom}/${x}/${y}.png`;
}

function loadTileImage(url) {
  let tile = tileImageCache.get(url);
  if (tile) return tile;

  if (tileImageCache.size > 360) {
    const firstKey = tileImageCache.keys().next().value;
    if (firstKey) tileImageCache.delete(firstKey);
  }

  tile = { ready: false, failed: false, image: null };
  tileImageCache.set(url, tile);

  if (typeof window === "undefined") return tile;
  const image = new Image();
  image.crossOrigin = "anonymous";
  image.onload = () => {
    tile.ready = true;
    tile.image = image;
    markRenderDirty({ map: true });
  };
  image.onerror = () => {
    tile.failed = true;
  };
  image.src = url;
  return tile;
}

function frameGridShape(frame = currentFrame.value) {
  if (gridShapeCache.frame === frame) {
    return gridShapeCache;
  }
  const nodes = frame?.nodes || [];
  let rows = 0;
  let cols = 0;
  nodes.forEach((node) => {
    if (!node?.grid) return;
    rows = Math.max(rows, Number(node.grid.row) + 1);
    cols = Math.max(cols, Number(node.grid.col) + 1);
  });
  gridShapeCache = {
    frame,
    rows: Math.max(1, rows || 10),
    cols: Math.max(1, cols || 12),
  };
  return gridShapeCache;
}

function findDeploymentNode(deployment = null, step = null) {
  const nodes = (currentFrame.value?.nodes || []).filter((node) => node?.node_role === "planned_deployment");
  if (!nodes.length) return null;

  const sequence = deployment?.sequence ?? step;
  const timeStep = deployment?.time_step ?? step;
  const siteIndex = deployment?.site_index;
  const grid = deployment?.grid || null;
  const device = deployment?.device || {};
  const label = device.device_label || deployment?.device_label || deployment?.label;

  if (sequence !== undefined && sequence !== null) {
    const byId = nodes.find((node) => String(node.id) === `deploy:${sequence}` || String(node.deployment_id) === String(sequence));
    if (byId) return byId;
  }

  const byTime = nodes.find((node) => {
    if (timeStep === undefined || timeStep === null) return false;
    if (Number(node.time_step) !== Number(timeStep)) return false;
    return !label || node.device_label === label || node.label === label;
  });
  if (byTime) return byTime;

  const bySite = nodes.find((node) => {
    if (siteIndex === undefined || siteIndex === null) return false;
    if (String(node.site_index) !== String(siteIndex)) return false;
    return !label || node.device_label === label || node.label === label;
  });
  if (bySite) return bySite;

  if (grid) {
    return nodes.find((node) => {
      const sameGrid = Number(node.grid?.row) === Number(grid.row) && Number(node.grid?.col) === Number(grid.col);
      return sameGrid && (!label || node.device_label === label || node.label === label);
    }) || null;
  }

  return null;
}

function deploymentCoordinateLabel(deployment = null, step = null) {
  const node = findDeploymentNode(deployment, step);
  if (!node) return "";
  const point = worldPointForNode(node);
  if (!Number.isFinite(point.x) || !Number.isFinite(point.y)) return "";
  return `XY(${Math.round(point.x)}, ${Math.round(point.y)})`;
}

function worldPointForNode(node) {
  const cached = worldPointCache.get(node);
  if (cached) return cached;

  const width = Number(currentFrame.value?.mapWidth || activeSession.value?.mapWidth || MAP_WIDTH);
  const height = Number(currentFrame.value?.mapHeight || activeSession.value?.mapHeight || MAP_HEIGHT);
  const base = {
    x: Number(node?.x || 0),
    y: Number(node?.y || 0),
  };
  const coordinateSource = String(node?.coordinate_source || node?.coordinateSource || "");
  const trustedCoordinate = coordinateSource === PRECISE_COORDINATE_SOURCE;

  if (isUserNode(node)) {
    const point = deblockedUserWorldPoint(node, base, width, height, coordinateSource);
    worldPointCache.set(node, point);
    return point;
  }

  if (!node?.grid || trustedCoordinate || coordinateSource === LEGACY_COORDINATE_SOURCE) {
    const point = clampWorldPoint(base, width, height);
    worldPointCache.set(node, point);
    return point;
  }
  const { rows, cols } = frameGridShape();
  const cellWidth = width / cols;
  const cellHeight = height / rows;
  const row = Number(node.grid.row);
  const col = Number(node.grid.col);
  const anchor = Number.isFinite(row) && Number.isFinite(col)
    ? {
        x: ((col + 0.5) / cols) * width,
        y: ((row + 0.5) / rows) * height,
      }
    : base;
  const seed = `${node.id}:${node.site_index ?? ""}:${node.time_step ?? ""}:${node.device_label ?? ""}`;
  const role = String(node.node_role || "");
  const spread = role === "planned_deployment" ? 0.68 : isUserNode(node) ? 0.9 : 0.5;
  const angle = hashNumber(`${seed}:angle`) * Math.PI * 2;
  const radius = Math.sqrt(hashNumber(`${seed}:radius`)) * spread;
  const point = {
    x: Math.max(0, Math.min(width, anchor.x + Math.cos(angle) * cellWidth * 0.5 * radius)),
    y: Math.max(0, Math.min(height, anchor.y + Math.sin(angle) * cellHeight * 0.5 * radius)),
  };
  worldPointCache.set(node, point);
  return point;
}

function clampWorldPoint(point, width, height) {
  return {
    x: Math.max(0, Math.min(width, Number(point?.x || 0))),
    y: Math.max(0, Math.min(height, Number(point?.y || 0))),
  };
}

function deblockedUserWorldPoint(node, base, width, height, coordinateSource) {
  if (coordinateSource === PRECISE_COORDINATE_SOURCE) {
    return clampWorldPoint(base, width, height);
  }

  const { rows, cols } = frameGridShape();
  const grid = node?.grid || {};
  const row = Number(grid.row);
  const col = Number(grid.col);
  const cellWidth = width / Math.max(1, Number.isFinite(col) ? cols : 14);
  const cellHeight = height / Math.max(1, Number.isFinite(row) ? rows : 12);
  const seed = `user-deblock:${node?.id ?? ""}:${row}:${col}:${coordinateSource}`;
  const spread = coordinateSource === LEGACY_COORDINATE_SOURCE ? 1.18 : 0.78;
  const angle = hashNumber(`${seed}:angle`) * Math.PI * 2;
  const radius = (0.14 + Math.sqrt(hashNumber(`${seed}:radius`)) * 0.86) * spread;
  const flow = (Number.isFinite(row) ? row : hashNumber(`${seed}:row`) * 10) * 1.371
    + (Number.isFinite(col) ? col : hashNumber(`${seed}:col`) * 12) * 0.917
    + hashNumber(`${seed}:flow`);
  const point = {
    x:
      base.x
      + Math.cos(angle) * cellWidth * 0.5 * radius
      + (hashNumber(`${seed}:free-x`) - 0.5) * cellWidth * 0.18 * spread
      + Math.sin(flow * Math.PI) * cellWidth * 0.07 * spread,
    y:
      base.y
      + Math.sin(angle) * cellHeight * 0.5 * radius
      + (hashNumber(`${seed}:free-y`) - 0.5) * cellHeight * 0.18 * spread
      + Math.cos(flow * Math.PI * 0.83) * cellHeight * 0.07 * spread,
  };
  return clampWorldPoint(point, width, height);
}

function screenPointForNode(node, timestamp = performance.now(), transform = null) {
  const t = transform || baseTransform();
  const animated = isUserNode(node);
  const bucket = animated ? Math.floor(Number(timestamp || 0) / USER_MOTION_CACHE_MS) : 0;
  const cacheKey = [
    bucket,
    Math.round(t.scaleX * 100000),
    Math.round(t.scaleY * 100000),
    Math.round(t.offsetX * 10),
    Math.round(t.offsetY * 10),
    playing.value ? 1 : 0,
  ].join(":");
  const cached = screenPointCache.get(node);
  if (cached?.key === cacheKey) return cached.point;

  const world = worldPointForNode(node);
  const base = worldToScreen(world.x, world.y, t);
  if (!animated) {
    screenPointCache.set(node, { key: cacheKey, point: base });
    return base;
  }

  const phaseA = hashNumber(`${node?.id}-a`) * Math.PI * 2;
  const phaseB = hashNumber(`${node?.id}-b`) * Math.PI * 2;
  const amp = playing.value ? USER_MOTION_PLAYING_AMPLITUDE : USER_MOTION_IDLE_AMPLITUDE;
  const point = {
    x: base.x + Math.cos(timestamp / USER_MOTION_X_PERIOD_MS + phaseA) * amp,
    y: base.y + Math.sin(timestamp / USER_MOTION_Y_PERIOD_MS + phaseB) * amp,
  };
  screenPointCache.set(node, { key: cacheKey, point });
  return point;
}

function screenToWorld(x, y) {
  const t = baseTransform();
  return {
    x: (x - t.offsetX) / t.scaleX,
    y: (y - t.offsetY) / t.scaleY,
  };
}

function clientPointToCanvasPoint(event) {
  const rect = canvasRef.value?.getBoundingClientRect();
  if (!rect) return null;
  const scaleX = rect.width > 0 ? canvasSize.width / rect.width : 1;
  const scaleY = rect.height > 0 ? canvasSize.height / rect.height : 1;
  return {
    x: (event.clientX - rect.left) * scaleX,
    y: (event.clientY - rect.top) * scaleY,
  };
}

function nodeColor(node) {
  if (isUserNode(node)) {
    return "#00FF9D";
  }
  const key = stationCategory(node);
  return {
    macro: "#00F0FF",
    manpack: "#FFB700",
    smallCell: "#B026FF",
    relay: "#FF0055",
  }[key] || "#00F0FF";
}

function colorWithAlpha(color, alpha) {
  const hex = String(color || "").replace("#", "");
  if (hex.length === 6) {
    const value = Number.parseInt(hex, 16);
    const red = (value >> 16) & 255;
    const green = (value >> 8) & 255;
    const blue = value & 255;
    return `rgba(${red}, ${green}, ${blue}, ${alpha})`;
  }
  if (String(color).startsWith("rgb(")) {
    return String(color).replace("rgb(", "rgba(").replace(")", `, ${alpha})`);
  }
  return color;
}

function isLatestDeploymentNode(node) {
  const frame = currentFrame.value || {};
  const latest = frame.latest_deployment || frame.latestDeployment || null;
  const latestId = frame.latestDeploymentId || frame.latest_deployment_id;
  if (latestId && (node.id === latestId || node.deployment_id === latestId)) return true;

  const sequence = latest?.sequence ?? latest?.time_step;
  if (sequence !== undefined && sequence !== null && String(node.id) === `deploy:${sequence}`) return true;

  const latestGrid = latest?.grid;
  if (!latestGrid || !node.grid) return false;
  const sameGrid = Number(node.grid.row) === Number(latestGrid.row) && Number(node.grid.col) === Number(latestGrid.col);
  const latestLabel = latest?.device?.device_label || latest?.device_label;
  return sameGrid && (!latestLabel || latestLabel === node.device_label || latestLabel === node.label);
}

function loadScenarioMap() {
  const url = scenarioMapUrl.value;
  if (!url || (url === scenarioMapUrlLoaded && scenarioMapReady)) return;
  scenarioMapUrlLoaded = url;
  scenarioMapReady = false;
  scenarioMapImage = null;

  if (typeof window === "undefined") return;
  const image = new Image();
  image.crossOrigin = "anonymous";
  image.onload = () => {
    if (scenarioMapUrlLoaded !== url) return;
    scenarioMapImage = image;
    scenarioMapReady = true;
    drawReplayMap();
  };
  image.onerror = () => {
    if (scenarioMapUrlLoaded !== url) return;
    scenarioMapImage = null;
    scenarioMapReady = false;
    drawReplayMap();
  };
  image.src = url;
}

function getRenderData(frame) {
  if (renderDataCache.frame === frame) {
    return renderDataCache;
  }

  const nodes = frame?.nodes || [];
  const users = [];
  const stations = [];
  const nodeMap = new Map();

  nodes.forEach((node) => {
    nodeMap.set(String(node.id), node);
    if (isUserNode(node)) users.push(node);
    else stations.push(node);
  });

  const drawableLinks = [];
  for (const link of frame?.links || []) {
    const src = nodeMap.get(String(link.src));
    const dst = nodeMap.get(String(link.dst));
    if (!src || !dst) continue;
    if (src.connected === false || dst.connected === false || src.status === "offline" || dst.status === "offline") continue;
    drawableLinks.push({ src, dst, protocol: Number(link.protocol || 0) });
    if (drawableLinks.length >= MAX_DRAWABLE_LINKS) break;
  }

  renderDataCache = { frame, users, stations, drawableLinks };
  return renderDataCache;
}

function drawReplayMap(timestamp = performance.now()) {
  renderDirty = false;
  const canvas = canvasRef.value;
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(canvasSize.dpr, 0, 0, canvasSize.dpr, 0, 0);
  ctx.clearRect(0, 0, canvasSize.width, canvasSize.height);

  drawBackground(ctx);

  const frame = currentFrame.value;
  if (!frame?.nodes?.length) {
    drawEmptyState(ctx);
    return;
  }

  const { users, stations, drawableLinks } = getRenderData(frame);
  const t = baseTransform();
  const pulse = (Math.sin(timestamp / 420) + 1) / 2;

  drawMapFrame(ctx, t, timestamp);
  if (layerToggles.heatmap || phaseKey.value === "damaged") drawHeatmap(ctx, users, t, timestamp);
  drawStageEffects(ctx, users, stations, t, pulse, timestamp);
  if (layerToggles.coverage) drawCoverage(ctx, stations, t, timestamp);
  if (layerToggles.links) drawLinks(ctx, drawableLinks, pulse, timestamp, t);
  if (layerToggles.users) drawUsers(ctx, users, timestamp, t);
  if (layerToggles.stations) drawStations(ctx, stations, pulse, timestamp, t);
}

function drawBackground(ctx) {
  const gradient = ctx.createLinearGradient(0, 0, 0, canvasSize.height);
  gradient.addColorStop(0, "#06112b");
  gradient.addColorStop(0.48, "#071c35");
  gradient.addColorStop(1, "#04101f");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, canvasSize.width, canvasSize.height);

  ctx.save();
  ctx.globalAlpha = 0.12;
  ctx.strokeStyle = "rgba(0, 200, 244, 0.1)";
  ctx.lineWidth = 1;
  const gap = 68;
  for (let x = -80; x < canvasSize.width + 80; x += gap) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x + 180, canvasSize.height);
    ctx.stroke();
  }
  for (let y = 0; y < canvasSize.height; y += gap) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(canvasSize.width, y);
    ctx.stroke();
  }
  ctx.restore();
}

function drawMapFrame(ctx, transform, timestamp) {
  const x = transform.offsetX;
  const y = transform.offsetY;
  const width = transform.drawWidth;
  const height = transform.drawHeight;

  ctx.save();
  ctx.beginPath();
  ctx.rect(x, y, width, height);
  ctx.clip();

  const hasTileMap = drawScenarioTileMap(ctx, x, y, width, height, timestamp);
  if (!hasTileMap && scenarioMapReady && scenarioMapImage) {
    ctx.save();
    ctx.filter = "invert(95%) hue-rotate(180deg) brightness(0.6) contrast(1.2) grayscale(0.2)";
    ctx.globalAlpha = 0.82;
    ctx.drawImage(scenarioMapImage, x, y, width, height);
    ctx.restore();
  } else if (!hasTileMap) {
    drawScenarioFallbackMap(ctx, x, y, width, height, timestamp);
  }

  drawScenarioGridLayer(ctx, x, y, width, height, timestamp);
  drawScenarioMapOverlay(ctx, x, y, width, height, timestamp);
  ctx.fillStyle = scenarioTheme.value.tint;
  ctx.fillRect(x, y, width, height);
  ctx.restore();
}

function drawScenarioTileMap(ctx, x, y, width, height, timestamp) {
  const bounds = scenarioGeoBounds.value;
  const viewport = mapViewport(width, height, bounds);
  if (!viewport) return false;

  const cacheKey = [
    Math.round(width),
    Math.round(height),
    viewport.zoom,
    Math.round(viewport.left),
    Math.round(viewport.top),
    bounds.latMin.toFixed(4),
    bounds.latMax.toFixed(4),
    bounds.lonMin.toFixed(4),
    bounds.lonMax.toFixed(4),
    scenarioTheme.value.key,
  ].join(":");
  if (tileMapCache.key === cacheKey && tileMapCache.canvas) {
    ctx.drawImage(tileMapCache.canvas, x, y, width, height);
    return true;
  }

  const offscreen = document.createElement("canvas");
  offscreen.width = Math.max(1, Math.round(width));
  offscreen.height = Math.max(1, Math.round(height));
  const mapCtx = offscreen.getContext("2d");

  const gradient = mapCtx.createLinearGradient(0, 0, width, height);
  gradient.addColorStop(0, "#071225");
  gradient.addColorStop(0.48, "#0a1a2d");
  gradient.addColorStop(1, "#020611");
  mapCtx.fillStyle = gradient;
  mapCtx.fillRect(0, 0, width, height);

  const maxTile = 2 ** viewport.zoom;
  const minTileX = Math.floor(viewport.left / TILE_SIZE) - 1;
  const maxTileX = Math.floor((viewport.left + width) / TILE_SIZE) + 1;
  const minTileY = Math.floor(viewport.top / TILE_SIZE) - 1;
  const maxTileY = Math.floor((viewport.top + height) / TILE_SIZE) + 1;

  let pendingTiles = false;
  mapCtx.globalAlpha = 0.96;
  mapCtx.filter = "brightness(0.72) contrast(1.18) saturate(0.82)";
  for (let tileX = minTileX; tileX <= maxTileX; tileX += 1) {
    const wrappedX = ((tileX % maxTile) + maxTile) % maxTile;
    for (let tileY = minTileY; tileY <= maxTileY; tileY += 1) {
      if (tileY < 0 || tileY >= maxTile) continue;
      const tile = loadTileImage(cartoTileUrl(viewport.zoom, wrappedX, tileY));
      if (!tile.ready || !tile.image) {
        pendingTiles = true;
        continue;
      }
      mapCtx.drawImage(
        tile.image,
        tileX * TILE_SIZE - viewport.left,
        tileY * TILE_SIZE - viewport.top,
        TILE_SIZE,
        TILE_SIZE
      );
    }
  }
  mapCtx.filter = "none";

  const theme = scenarioTheme.value;
  mapCtx.globalAlpha = 1;
  mapCtx.fillStyle = "rgba(2, 8, 23, 0.44)";
  mapCtx.fillRect(0, 0, width, height);
  mapCtx.fillStyle = theme.tint;
  mapCtx.fillRect(0, 0, width, height);

  if (!pendingTiles) {
    tileMapCache = { key: cacheKey, canvas: offscreen };
  }
  ctx.drawImage(offscreen, x, y, width, height);
  return true;
}

function drawScenarioGridLayer(ctx, x, y, width, height, timestamp) {
  const theme = scenarioTheme.value;
  const progress = replayProgress.value;
  const key = theme.key;

  ctx.save();
  ctx.globalCompositeOperation = "screen";

  const color =
    key === "earthquake"
      ? [255, 80, 80]
      : key === "rainstorm"
        ? [0, 132, 255]
        : [0, 240, 255];
  const decay = Math.max(0.12, 1 - progress * 0.72);
  const blobCount = key === "default" ? 8 : 14;
  for (let index = 0; index < blobCount; index += 1) {
    const seed = `${key}:hazard:${index}`;
    const cx = x + width * (0.08 + hashNumber(`${seed}:x`) * 0.84);
    const cy = y + height * (0.12 + hashNumber(`${seed}:y`) * 0.74);
    const radius = Math.max(width, height) * (0.07 + hashNumber(`${seed}:radius`) * 0.1);
    const alpha = (0.035 + hashNumber(`${seed}:a`) * 0.055) * decay;
    const haze = ctx.createRadialGradient(cx, cy, 0, cx, cy, radius);
    haze.addColorStop(0, `rgba(${color[0]}, ${color[1]}, ${color[2]}, ${alpha})`);
    haze.addColorStop(0.58, `rgba(${color[0]}, ${color[1]}, ${color[2]}, ${alpha * 0.28})`);
    haze.addColorStop(1, `rgba(${color[0]}, ${color[1]}, ${color[2]}, 0)`);
    ctx.fillStyle = haze;
    ctx.fillRect(cx - radius, cy - radius, radius * 2, radius * 2);
  }

  if (progress > 0.18 && progress < 0.88) {
    const scanX = x + width * ((progress * 1.38) % 1);
    const scanGradient = ctx.createLinearGradient(scanX - width * 0.09, y, scanX + width * 0.09, y);
    scanGradient.addColorStop(0, "rgba(0, 255, 157, 0)");
    scanGradient.addColorStop(0.5, "rgba(0, 255, 157, 0.16)");
    scanGradient.addColorStop(1, "rgba(0, 255, 157, 0)");
    ctx.fillStyle = scanGradient;
    ctx.fillRect(scanX - width * 0.09, y, width * 0.18, height);
  }
  ctx.restore();
}

function drawScenarioFallbackMap(ctx, x, y, width, height, timestamp) {
  const theme = scenarioTheme.value;
  const gradient = ctx.createLinearGradient(x, y, x + width, y + height);
  gradient.addColorStop(0, "#081325");
  gradient.addColorStop(0.42, theme.land);
  gradient.addColorStop(1, "#020711");
  ctx.fillStyle = gradient;
  ctx.fillRect(x, y, width, height);

  ctx.save();
  ctx.globalCompositeOperation = "screen";
  ctx.fillStyle = theme.water;
  ctx.beginPath();
  ctx.moveTo(x + width * 0.08, y + height * 0.78);
  ctx.bezierCurveTo(x + width * 0.22, y + height * 0.55, x + width * 0.42, y + height * 0.82, x + width * 0.58, y + height * 0.58);
  ctx.bezierCurveTo(x + width * 0.72, y + height * 0.38, x + width * 0.92, y + height * 0.54, x + width * 1.04, y + height * 0.3);
  ctx.lineTo(x + width * 1.04, y + height * 1.04);
  ctx.lineTo(x - width * 0.04, y + height * 1.04);
  ctx.closePath();
  ctx.fill();

  ctx.strokeStyle = theme.road;
  ctx.lineWidth = 1.2;
  ctx.globalAlpha = 0.72;
  for (let index = 0; index < 14; index += 1) {
    const startY = y + ((index + 1) / 15) * height;
    ctx.beginPath();
    ctx.moveTo(x + width * -0.1, startY);
    ctx.bezierCurveTo(
      x + width * 0.24,
      startY + Math.sin(timestamp / 3000 + index) * 24,
      x + width * 0.62,
      startY - 48 + Math.cos(index) * 22,
      x + width * 1.1,
      startY + Math.sin(index * 1.7) * 36
    );
    ctx.stroke();
  }
  ctx.restore();
}

function drawScenarioMapOverlay(ctx, x, y, width, height, timestamp) {
  const key = scenarioTheme.value.key;
  const t = timestamp / 1000;

  ctx.save();
  ctx.globalCompositeOperation = "screen";

  if (key === "rainstorm") {
    ctx.globalAlpha = 0.24;
    ctx.fillStyle = "rgba(0, 132, 255, 0.16)";
    for (let index = 0; index < 6; index += 1) {
      const px = x + width * (0.12 + index * 0.15);
      const py = y + height * (0.58 + Math.sin(index * 1.8) * 0.18);
      ctx.beginPath();
      ctx.ellipse(px, py, width * (0.075 + (index % 2) * 0.025), height * 0.035, -0.45, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.strokeStyle = "rgba(109, 204, 255, 0.08)";
    ctx.lineWidth = 0.8;
    const offset = (t * 80) % 42;
    for (let index = -6; index < 20; index += 2) {
      const sx = x + index * 128 + offset;
      ctx.beginPath();
      ctx.moveTo(sx, y + height * 0.04);
      ctx.lineTo(sx + width * 0.09, y + height * 0.96);
      ctx.stroke();
    }
  } else if (key === "earthquake") {
    ctx.globalAlpha = 0.66;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    for (let index = 0; index < 4; index += 1) {
      const startX = x + width * (0.16 + index * 0.18);
      const startY = y + height * (0.18 + (index % 2) * 0.22);
      ctx.beginPath();
      ctx.moveTo(startX, startY);
      for (let segment = 1; segment <= 7; segment += 1) {
        ctx.lineTo(
          startX + width * 0.045 * segment + Math.sin(segment + index) * 18,
          startY + height * 0.07 * segment + Math.cos(segment * 1.7 + index) * 20
        );
      }
      ctx.strokeStyle = index % 2 ? "rgba(255, 183, 0, 0.32)" : "rgba(255, 0, 85, 0.26)";
      ctx.lineWidth = index % 2 ? 1.2 : 2;
      ctx.stroke();
    }

    ctx.strokeStyle = "rgba(255, 183, 0, 0.14)";
    for (let index = 0; index < 5; index += 1) {
      ctx.beginPath();
      ctx.ellipse(
        x + width * 0.54,
        y + height * 0.46,
        width * (0.08 + index * 0.055 + Math.sin(t + index) * 0.004),
        height * (0.04 + index * 0.035),
        0.25,
        0,
        Math.PI * 2
      );
      ctx.stroke();
    }
  } else if (key === "typhoon") {
    const cx = x + width * 0.64;
    const cy = y + height * 0.34;
    ctx.globalAlpha = 0.52;
    ctx.strokeStyle = "rgba(0, 240, 255, 0.28)";
    ctx.lineWidth = 1.4;
    for (let band = 0; band < 8; band += 1) {
      const radius = width * (0.055 + band * 0.035);
      ctx.beginPath();
      ctx.arc(cx, cy, radius, t * 0.25 + band * 0.42, t * 0.25 + band * 0.42 + Math.PI * 1.12);
      ctx.stroke();
    }

    ctx.fillStyle = "rgba(0, 92, 135, 0.26)";
    ctx.beginPath();
    ctx.moveTo(x + width * 0.7, y + height);
    ctx.bezierCurveTo(x + width * 0.76, y + height * 0.74, x + width * 0.94, y + height * 0.64, x + width * 1.03, y + height * 0.46);
    ctx.lineTo(x + width * 1.03, y + height * 1.03);
    ctx.closePath();
    ctx.fill();
  } else {
    ctx.globalAlpha = 0.32;
    ctx.strokeStyle = "rgba(0, 240, 255, 0.18)";
    ctx.lineWidth = 1;
    for (let index = 0; index < 5; index += 1) {
      ctx.beginPath();
      const py = y + height * (0.18 + index * 0.15);
      ctx.moveTo(x + width * 0.08, py);
      ctx.bezierCurveTo(x + width * 0.28, py - 34, x + width * 0.58, py + 38, x + width * 0.92, py - 12);
      ctx.stroke();
    }
  }

  ctx.restore();
}

function drawStageEffects(ctx, users, stations, transform, pulse, timestamp) {
  const stage = stageDetail.value.id;
  const frameWidth = transform.drawWidth;
  const frameHeight = transform.drawHeight;
  const origin = {
    x: transform.offsetX + frameWidth * 0.08,
    y: transform.offsetY + frameHeight * 0.88,
  };

  ctx.save();
  ctx.globalCompositeOperation = "screen";

  if (stage === "monitor" || stage === "damage") {
    let rendered = 0;
    for (const node of users) {
      if (rendered > 54) break;
      if (node.connected || node.broadcast_served) continue;
      if (hashNumber(`${node.id}:damage-ring`) < 0.78) continue;
      const point = screenPointForNode(node, timestamp, transform);
      const radius = 8 + ((timestamp / 90 + rendered * 3) % 18);
      const alpha = Math.max(0, 0.42 - radius / 48);
      ctx.beginPath();
      ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(255, 0, 85, ${alpha})`;
      ctx.lineWidth = 1;
      ctx.stroke();
      rendered += 1;
    }
  }

  if (stage === "cluster") {
    const cells = new Map();
    users.forEach((node) => {
      if (node.connected || node.broadcast_served || !node.grid) return;
      const key = `${node.grid.row}:${node.grid.col}`;
      const item = cells.get(key) || { count: 0, x: 0, y: 0 };
      const point = screenPointForNode(node, timestamp, transform);
      item.count += 1;
      item.x += point.x;
      item.y += point.y;
      cells.set(key, item);
    });
    [...cells.values()]
      .sort((a, b) => b.count - a.count)
      .slice(0, 9)
      .forEach((cell, index) => {
        const cx = cell.x / cell.count;
        const cy = cell.y / cell.count;
        ctx.beginPath();
        ctx.arc(cx, cy, 20 + index * 2 + pulse * 10, 0, Math.PI * 2);
        ctx.strokeStyle = "rgba(255, 183, 0, 0.34)";
        ctx.lineWidth = 1.2;
        ctx.setLineDash([6, 8]);
        ctx.lineDashOffset = -timestamp / 90;
        ctx.stroke();
      });
    ctx.setLineDash([]);
  }

  if (stage === "dispatch" || stage === "links") {
    const deployed = stations
      .filter((node) => node.node_role === "planned_deployment" && node.connected !== false && node.status !== "offline")
      .slice(-8);
    deployed.forEach((node, index) => {
      const target = screenPointForNode(node, timestamp, transform);
      const laneOffset = (index - deployed.length / 2) * 18;
      const control = {
        x: origin.x + (target.x - origin.x) * 0.48,
        y: Math.min(origin.y, target.y) - 80 - laneOffset,
      };
      ctx.beginPath();
      ctx.moveTo(origin.x, origin.y);
      ctx.quadraticCurveTo(control.x, control.y, target.x, target.y);
      ctx.strokeStyle = index % 2 ? "rgba(0, 240, 255, 0.26)" : "rgba(255, 183, 0, 0.32)";
      ctx.lineWidth = 1.2;
      ctx.setLineDash([8, 10]);
      ctx.lineDashOffset = -timestamp / 70;
      ctx.stroke();

      const travel = (timestamp / 1200 + index * 0.17) % 1;
      const mx = (1 - travel) * (1 - travel) * origin.x + 2 * (1 - travel) * travel * control.x + travel * travel * target.x;
      const my = (1 - travel) * (1 - travel) * origin.y + 2 * (1 - travel) * travel * control.y + travel * travel * target.y;
      ctx.beginPath();
      ctx.arc(mx, my, 3.5, 0, Math.PI * 2);
      ctx.fillStyle = index % 2 ? "rgba(0, 240, 255, 0.82)" : "rgba(255, 183, 0, 0.86)";
      ctx.fill();
    });
    ctx.setLineDash([]);
  }

  if (stage === "coverage" || stage === "stabilize" || stage === "completed") {
    const activeStations = stations.filter((node) => node.connected !== false && node.status !== "offline").slice(-18);
    activeStations.forEach((node, index) => {
      if (hashNumber(`${node.id}:wave`) < 0.35) return;
      const point = screenPointForNode(node, timestamp, transform);
      const radius = 24 + ((timestamp / 70 + index * 9) % 74);
      ctx.beginPath();
      ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(0, 255, 157, ${Math.max(0, 0.26 - radius / 360)})`;
      ctx.lineWidth = 1;
      ctx.stroke();
    });
  }

  ctx.restore();
}

function drawHeatmap(ctx, users, transform, timestamp) {
  let rendered = 0;
  ctx.save();
  ctx.globalCompositeOperation = "screen";
  for (const node of users) {
    if (rendered >= MAX_HEAT_BLOBS) break;
    if (node.connected || node.broadcast_served) continue;
    const gate = hashNumber(`${node.id}:heat-blob`);
    if (gate < 0.82) continue;
    const point = screenPointForNode(node, timestamp, transform);
    const radius = 92 + gate * 110;
    const gradient = ctx.createRadialGradient(point.x, point.y, 0, point.x, point.y, radius);
    gradient.addColorStop(0, "rgba(255, 70, 96, 0.24)");
    gradient.addColorStop(0.48, "rgba(255, 127, 80, 0.07)");
    gradient.addColorStop(1, "rgba(255, 70, 96, 0)");
    ctx.fillStyle = gradient;
    ctx.fillRect(point.x - radius, point.y - radius, radius * 2, radius * 2);
    rendered += 1;
  }
  ctx.restore();
}

function drawCoverage(ctx, stations, transform, timestamp) {
  const coverageCfg = {
    macro: { fill: "rgba(0, 240, 255, 0.05)", stroke: "rgba(0, 240, 255, 0.2)", radius: 800 },
    manpack: { fill: "rgba(255, 183, 0, 0.05)", stroke: "rgba(255, 183, 0, 0.2)", radius: 500 },
    smallCell: { fill: "rgba(176, 38, 255, 0.05)", stroke: "rgba(176, 38, 255, 0.2)", radius: 300 },
    relay: { fill: "rgba(255, 0, 85, 0.05)", stroke: "rgba(255, 0, 85, 0.2)", radius: 200 },
  };

  ctx.save();
  ctx.globalCompositeOperation = "screen";
  const activeStations = stations
    .filter((station) => station.connected !== false && station.status !== "offline")
    .filter((station) => isLatestDeploymentNode(station) || stationCategory(station) === "macro" || hashNumber(`${station.id}:coverage`) > 0.54)
    .slice(-MAX_COVERAGE_RINGS);
  activeStations.forEach((station) => {
    const category = stationCategory(station);
    const cfg = coverageCfg[category] || coverageCfg.macro;
    const point = screenPointForNode(station, timestamp, transform);
    const radiusRaw = Number(station.coverage_radius || station.coverage_radius_km || 0);
    const worldRadius = Number.isFinite(radiusRaw) && radiusRaw > 0
      ? (radiusRaw > 50 ? radiusRaw : cfg.radius * Math.max(0.7, Math.min(1.65, radiusRaw)))
      : cfg.radius;
    const radius = Math.max(24, Math.min(190, worldRadius * transform.scale));
    ctx.beginPath();
    ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
    ctx.fillStyle = cfg.fill;
    ctx.strokeStyle = cfg.stroke;
    ctx.lineWidth = 1;
    ctx.fill();
    ctx.stroke();
  });
  ctx.restore();
}

function drawLinks(ctx, drawable, pulse, timestamp, transform) {
  if (!drawable.length) return;
  ctx.save();
  ctx.globalCompositeOperation = "screen";
  ctx.lineCap = "round";
  drawable.forEach((link, index) => {
    const src = screenPointForNode(link.src, timestamp, transform);
    const dst = screenPointForNode(link.dst, timestamp, transform);
    const protocolColor = link.protocol === 1 ? "rgba(0, 240, 255, 0.22)" : "rgba(0, 255, 157, 0.32)";
    ctx.strokeStyle = protocolColor;
    ctx.lineWidth = 1;
    ctx.globalAlpha = 0.76 + ((index % 5) / 22) + pulse * 0.08;
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(dst.x, dst.y);
    ctx.lineTo(src.x, src.y);
    ctx.stroke();
    if (playing.value && index % 7 === 0) {
      const t = (timestamp / 1100 + (index % 13) / 13) % 1;
      ctx.beginPath();
      ctx.arc(dst.x + (src.x - dst.x) * t, dst.y + (src.y - dst.y) * t, 2.4, 0, Math.PI * 2);
      ctx.fillStyle = link.protocol === 1 ? "rgba(0, 240, 255, 0.88)" : "rgba(0, 255, 157, 0.86)";
      ctx.fill();
    }
  });
  ctx.restore();
}

function drawUsers(ctx, users, timestamp, transform) {
  ctx.save();
  ctx.globalCompositeOperation = "screen";
  ctx.globalAlpha = 0.9;
  ctx.fillStyle = "#00FF9D";
  for (const node of users) {
    if (!(node.connected || node.broadcast_served)) continue;
    const point = screenPointForNode(node, timestamp, transform);
    ctx.fillRect(point.x - 1.1, point.y - 1.1, 2.2, 2.2);
  }

  ctx.globalAlpha = 0.64;
  ctx.fillStyle = "#FF4B6A";
  for (const node of users) {
    if (node.connected || node.broadcast_served) continue;
    const point = screenPointForNode(node, timestamp, transform);
    ctx.fillRect(point.x - 1.35, point.y - 1.35, 2.7, 2.7);
  }

  let rings = 0;
  for (const node of users) {
    if (rings >= MAX_WARNING_RINGS) break;
    if (node.connected || node.broadcast_served || hashNumber(`${node.id}:offline-mark`) <= 0.91) continue;
    const point = screenPointForNode(node, timestamp, transform);
    const warningPulse = (Math.sin(timestamp / 360 + hashNumber(node.id) * Math.PI * 2) + 1) / 2;
    ctx.globalAlpha = 0.26 + warningPulse * 0.18;
    ctx.strokeStyle = "rgba(255, 0, 85, 0.62)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(point.x, point.y, 7 + warningPulse * 4, 0, Math.PI * 2);
    ctx.stroke();
    rings += 1;
  }
  ctx.restore();
}

function drawStations(ctx, stations, pulse, timestamp, transform) {
  const typeConfig = {
    macro: { radius: 6, glow: 15 },
    manpack: { radius: 4, glow: 10 },
    smallCell: { radius: 4, glow: 10 },
    relay: { radius: 3, glow: 10 },
  };

  ctx.save();
  ctx.globalCompositeOperation = "screen";
  stations.forEach((node) => {
    const point = screenPointForNode(node, timestamp, transform);
    const color = nodeColor(node);
    const cfg = typeConfig[stationCategory(node)] || typeConfig.macro;
    const isLatest = isLatestDeploymentNode(node);
    const offline = node.connected === false || node.status === "offline";
    ctx.shadowColor = color;
    ctx.shadowBlur = offline ? 0 : isLatest ? cfg.glow + 16 : cfg.glow;
    ctx.beginPath();
    ctx.arc(point.x, point.y, cfg.radius + (!offline && isLatest ? pulse * 3 : 0), 0, Math.PI * 2);
    ctx.fillStyle = offline ? "rgba(255, 0, 85, 0.18)" : color;
    ctx.fill();

    ctx.shadowBlur = 0;
    ctx.strokeStyle = offline ? "rgba(255, 0, 85, 0.7)" : "#ffffff";
    ctx.lineWidth = offline ? 1.2 : 1.5;
    ctx.stroke();

    if (offline) {
      ctx.beginPath();
      ctx.moveTo(point.x - cfg.radius - 3, point.y - cfg.radius - 3);
      ctx.lineTo(point.x + cfg.radius + 3, point.y + cfg.radius + 3);
      ctx.moveTo(point.x + cfg.radius + 3, point.y - cfg.radius - 3);
      ctx.lineTo(point.x - cfg.radius - 3, point.y + cfg.radius + 3);
      ctx.strokeStyle = "rgba(255, 0, 85, 0.72)";
      ctx.stroke();
    }

    if (!offline && isLatest) {
      ctx.beginPath();
      ctx.arc(point.x, point.y, cfg.radius + 9 + pulse * 8, 0, Math.PI * 2);
      ctx.strokeStyle = colorWithAlpha(color, 0.42);
      ctx.lineWidth = 1.2;
      ctx.stroke();
    }
  });
  ctx.restore();
}

function drawEmptyState(ctx) {
  ctx.save();
  ctx.fillStyle = "rgba(183, 224, 254, 0.82)";
  ctx.font = "18px Microsoft YaHei, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(loadingFrame.value ? "正在读取后端回放帧..." : "请选择一条后端回放记录", canvasSize.width / 2, canvasSize.height / 2);
  ctx.restore();
}

function drawThroughputChart() {
  chartDirty = false;
  const canvas = chartCanvasRef.value;
  if (!canvas) return;
  const parentWidth = canvas.parentElement?.clientWidth || 260;
  const dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
  const width = Math.max(1, Math.round(parentWidth));
  const height = 70;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const values = throughputSeries
    .slice(0, frameIndex.value + 1)
    .filter((value) => Number.isFinite(value) && value >= MIN_USABLE_THROUGHPUT_MBPS);
  if (!values.length && currentFrame.value) {
    const currentValue = frameThroughputValue(currentFrame.value, linkMetrics.value);
    if (currentValue >= MIN_USABLE_THROUGHPUT_MBPS) values.push(currentValue);
  }
  if (!values.length) return;

  const maxValue = Math.max(...values, 1) * 1.2;
  ctx.beginPath();
  values.forEach((value, index) => {
    const x = values.length <= 1 ? 0 : (index / Math.max(1, maxFrameIndex.value)) * width;
    const y = height - (value / maxValue) * height;
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = "#00F0FF";
  ctx.lineWidth = 2;
  ctx.shadowBlur = 10;
  ctx.shadowColor = "#00F0FF";
  ctx.stroke();
  ctx.shadowBlur = 0;
  ctx.lineTo((Math.max(values.length - 1, 0) / Math.max(1, maxFrameIndex.value)) * width, height);
  ctx.lineTo(0, height);
  ctx.closePath();
  ctx.fillStyle = "rgba(0, 240, 255, 0.15)";
  ctx.fill();
}

function animationLoop(timestamp) {
  const shouldAnimateMap = playing.value || dragging.value;
  const renderInterval = shouldAnimateMap ? PLAYING_RENDER_INTERVAL : IDLE_RENDER_INTERVAL;
  if ((renderDirty || shouldAnimateMap) && timestamp - lastRenderAt >= renderInterval) {
    drawReplayMap(timestamp);
    lastRenderAt = timestamp;
  }

  const shouldAnimateChart = playing.value && Boolean(currentFrame.value);
  if ((chartDirty || shouldAnimateChart) && timestamp - lastChartAt >= CHART_RENDER_INTERVAL) {
    drawThroughputChart();
    lastChartAt = timestamp;
  }
  animationId = window.requestAnimationFrame(animationLoop);
}

async function refreshReplaySessions() {
  loadingSessions.value = true;
  replayError.value = "";
  appendReplayTerminalLine("前端操作：刷新场景回放会话列表。", { level: "ACTION" });
  try {
    const payload = await fetchJson(`${API_BASE}/replay/sessions?limit=30`);
    sessions.value = (payload?.sessions || []).map(normalizeSession);
    appendReplayTerminalLine(`后端响应：场景回放会话 ${sessions.value.length} 条。`, {
      level: "BACKEND",
      source: "BACKEND",
    });
    const preferredId = hashQueryReplayId() || getActiveReplaySessionId();
    const selected = sessions.value.find((session) => session.id === preferredId) || sessions.value[0] || null;
    if (selected) {
      await selectReplaySession(selected.id, { keepPicker: true });
    }
  } catch (error) {
    replayError.value = `后端回放读取失败：${error?.message || error}`;
    appendReplayTerminalLine(replayError.value, { level: "ERROR", source: "BACKEND" });
  } finally {
    loadingSessions.value = false;
  }
}

async function selectReplaySession(id, options = {}) {
  if (!id) return;
  stopPlayback({ refreshDetail: false });
  hasReplayStarted.value = false;
  activeSessionId.value = id;
  setActiveReplaySessionId(id);
  selectedNode.value = null;
  mapView.scale = 1;
  mapView.offsetX = 0;
  mapView.offsetY = 0;
  frameCache.clear();
  throughputSeries = [];
  lastStableTelemetry.value = { throughput: 0, latency: 0 };
  markRenderDirty({ frame: true, chart: true });
  clientLogEntries.value = [];
  frameIndex.value = 0;
  logQuery.value = "";
  logQueryMode.value = false;
  replayError.value = "";
  if (!options.keepPicker) showSessionPicker.value = false;
  appendReplayTerminalLine(`前端操作：选择场景回放 replay_id=${id}。`, { level: "ACTION" });

  try {
    const detail = normalizeSession(await fetchJson(`${API_BASE}/replay/sessions/${encodeURIComponent(id)}`));
    activeSessionDetail.value = detail;
    loadScenarioMap();
    appendReplayTerminalLine(`后端响应：回放元数据已加载，帧数=${detail.frameCount} 节点=${detail.nodeCountTotal}。`, {
      level: "BACKEND",
      source: "BACKEND",
    });
    appendReplayUserNodeCount(`场景回放接入灾害场景：${detail.title || id}`, detail);
  } catch {
    activeSessionDetail.value = sessions.value.find((session) => session.id === id) || null;
    loadScenarioMap();
    appendReplayTerminalLine("后端响应：回放元数据详情读取失败，已使用列表摘要继续。", {
      level: "WARN",
      source: "BACKEND",
    });
    appendReplayUserNodeCount(`场景回放接入灾害场景：${activeSessionDetail.value?.title || id}`, activeSessionDetail.value);
  }

  await loadReplayFrame(id, 0);
  window.setTimeout(() => {
    if (activeSessionId.value === id) void loadReplayLogs(id);
  }, 0);
}

async function loadReplayFrame(id, index, options = {}) {
  if (!id) return;
  const numericIndex = Math.max(0, Math.min(maxFrameIndex.value || 0, Number(index || 0)));
  const sampleRatio = frameSampleRatio(options);
  const cached = cachedFrameEntry(id, numericIndex, sampleRatio);
  const token = (frameRequestToken += 1);
  loadingFrame.value = true;

  try {
    if (cached.entry) {
      const cachedMetrics = normalizeLinkMetrics(cached.entry.linkMetrics);
      cached.entry.linkMetrics = cachedMetrics;
      currentFrame.value = cached.entry.frame;
      linkMetrics.value = cachedMetrics;
      rememberStableTelemetry(cached.entry.frame, cachedMetrics);
      appendReplayUserNodeCount(`场景回放帧数据已接入：${activeSession.value?.title || id}`, activeSession.value, cached.entry.frame);
      throughputSeries[numericIndex] = stableThroughputSeriesValue(numericIndex, cached.entry.frame, cachedMetrics);
      markRenderDirty({ frame: true, chart: true });
      loadScenarioMap();
      drawReplayMap();
      if (!cachedMetrics) scheduleFrameMetrics(id, numericIndex, token, cached.cacheKey);
      prefetchReplayFrames(id, numericIndex);
      return;
    }

    const { cacheKey, entry } = await fetchReplayFramePayload(id, numericIndex, sampleRatio);

    if (token !== frameRequestToken || activeSessionId.value !== id) return;
    currentFrame.value = entry.frame;
    linkMetrics.value = entry.linkMetrics;
    rememberStableTelemetry(entry.frame, entry.linkMetrics);
    appendReplayUserNodeCount(`场景回放帧数据已接入：${activeSession.value?.title || id}`, activeSession.value, entry.frame);
    throughputSeries[numericIndex] = stableThroughputSeriesValue(numericIndex, entry.frame, entry.linkMetrics);
    if (hasReplayStarted.value && numericIndex % 5 === 0) {
      appendReplayTerminalLine(
        `后端响应：回放帧 ${numericIndex}/${maxFrameIndex.value} 已加载，覆盖率=${formatPercent(entry.frame.coverageRatio || 0)}。`,
        { level: "BACKEND", source: "BACKEND" }
      );
    }
    markRenderDirty({ frame: true, chart: true });
    loadScenarioMap();
    drawReplayMap();
    if (!entry.linkMetrics) scheduleFrameMetrics(id, numericIndex, token, cacheKey);
    prefetchReplayFrames(id, numericIndex);
  } catch (error) {
    if (token === frameRequestToken) {
      replayError.value = `回放帧读取失败：${error?.message || error}`;
      appendReplayTerminalLine(replayError.value, { level: "ERROR", source: "BACKEND" });
    }
  } finally {
    if (token === frameRequestToken) loadingFrame.value = false;
  }
}

function scheduleFrameMetrics(id, numericIndex, token, cacheKey) {
  window.setTimeout(() => {
    if (activeSessionId.value !== id || frameIndex.value !== numericIndex) return;
    void loadFrameMetrics(id, numericIndex, token, cacheKey);
  }, playing.value ? 80 : 240);
}

async function loadFrameMetrics(id, numericIndex, token, cacheKey) {
  try {
    const metricsPayload = await fetchJson(
      `${API_BASE}/replay/sessions/${encodeURIComponent(id)}/link-metrics?frame_index=${numericIndex}`
    );
    const cached = frameCache.get(cacheKey);
    if (cached) {
      const normalizedMetrics = normalizeLinkMetrics(metricsPayload);
      cached.linkMetrics = normalizedMetrics;
      throughputSeries[numericIndex] = stableThroughputSeriesValue(numericIndex, cached.frame, normalizedMetrics);
      rememberStableTelemetry(cached.frame, normalizedMetrics);
    }
    if (token !== frameRequestToken || activeSessionId.value !== id || frameIndex.value !== numericIndex) return;
    const normalizedMetrics = normalizeLinkMetrics(metricsPayload);
    if (!normalizedMetrics) return;
    linkMetrics.value = normalizedMetrics;
    throughputSeries[numericIndex] = stableThroughputSeriesValue(numericIndex, currentFrame.value, normalizedMetrics);
    rememberStableTelemetry(currentFrame.value, normalizedMetrics);
    markRenderDirty({ chart: true });
    if (hasReplayStarted.value) {
      appendReplayTerminalLine(`后端响应：回放帧 ${numericIndex} 链路指标已同步。`, {
        level: "BACKEND",
        source: "BACKEND",
      });
    }
  } catch {
    const cached = frameCache.get(cacheKey);
    if (cached) cached.linkMetrics = null;
  }
}

function requestCurrentFrameMetrics() {
  if (!activeSessionId.value) return;
  const numericIndex = Math.max(0, Number(frameIndex.value || 0));
  let cacheKey = frameCacheKey(activeSessionId.value, numericIndex, FRAME_SAMPLE_RATIO);
  let cached = frameCache.get(cacheKey);
  if (!cached) {
    cacheKey = frameCacheKey(activeSessionId.value, numericIndex, PLAYING_FRAME_SAMPLE_RATIO);
    cached = frameCache.get(cacheKey);
  }
  if (!cached || normalizeLinkMetrics(cached.linkMetrics)) return;
  scheduleFrameMetrics(activeSessionId.value, numericIndex, frameRequestToken, cacheKey);
}

async function loadReplayLogs(id) {
  if (!id) return;
  try {
    const payload = await fetchJson(`${API_BASE}/replay/sessions/${encodeURIComponent(id)}/logs?limit=500`);
    allLogLines.value = Array.isArray(payload?.lines) ? payload.lines : [];
    appendReplayTerminalLine(`后端响应：回放日志已加载 ${allLogLines.value.length} 行。`, {
      level: "BACKEND",
      source: "BACKEND",
    });
  } catch {
    allLogLines.value = [];
    appendReplayTerminalLine("后端响应：回放日志读取失败。", { level: "ERROR", source: "BACKEND" });
  }
}

function playbackDelay() {
  const frameCount = Math.max(1, maxFrameIndex.value);
  const targetDelay = TARGET_REPLAY_DURATION_MS / frameCount;
  const rate = Math.max(0.25, Number(playbackRate.value || 1));
  return Math.max(MIN_FRAME_DELAY_MS, Math.min(MAX_FRAME_DELAY_MS, targetDelay / rate));
}

function schedulePlayback() {
  if (playbackTimer) window.clearTimeout(playbackTimer);
  if (!playing.value) return;
  playbackTimer = window.setTimeout(() => {
    if (!playing.value) return;
    if (frameIndex.value >= maxFrameIndex.value) {
      appendClientLog("SYSTEM", "SYS_FINISH", "进程自然结束，网络已收敛至恢复态。");
      stopPlayback();
      return;
    }
    frameIndex.value += 1;
    schedulePlayback();
  }, playbackDelay());
}

function togglePlayback() {
  if (playing.value) {
    appendClientLog("WARN", "SYS_STATE", "物理推演已人工阻断挂起");
    stopPlayback();
    return;
  }
  if (!activeSession.value || maxFrameIndex.value <= 0) return;
  if (frameIndex.value >= maxFrameIndex.value) frameIndex.value = 0;
  hasReplayStarted.value = true;
  playing.value = true;
  appendReplayTerminalLine(`前端操作：启动场景回放 replay_id=${activeSessionId.value}。`, { level: "ACTION" });
  appendClientLog("INFO", "SYS_STATE", "物理引擎推进...");
  schedulePlayback();
}

function stopPlayback(options = {}) {
  playing.value = false;
  if (playbackTimer) {
    window.clearTimeout(playbackTimer);
    playbackTimer = null;
  }
  if (options.refreshDetail !== false && activeSessionId.value && activeSession.value) {
    void loadReplayFrame(activeSessionId.value, frameIndex.value, { preferDetail: true });
  }
  requestCurrentFrameMetrics();
}

function stepFrame(delta) {
  if (!activeSession.value) return;
  hasReplayStarted.value = true;
  stopPlayback({ refreshDetail: false });
  frameIndex.value = Math.max(0, Math.min(maxFrameIndex.value, frameIndex.value + delta));
  appendReplayTerminalLine(`前端操作：单步切换到回放帧 ${frameIndex.value}。`, { level: "ACTION" });
}

function resetPlayback() {
  stopPlayback({ refreshDetail: false });
  hasReplayStarted.value = false;
  logQueryMode.value = false;
  logQuery.value = "";
  clientLogEntries.value = [];
  selectedNode.value = null;
  frameIndex.value = 0;
  appendReplayTerminalLine("前端操作：复位场景回放。", { level: "ACTION" });
}

function jumpToStart() {
  if (!activeSession.value) return;
  resetPlayback();
}

function jumpToEnd() {
  if (!activeSession.value) return;
  hasReplayStarted.value = true;
  stopPlayback({ refreshDetail: false });
  appendReplayTerminalLine("前端操作：跳转到场景回放末帧。", { level: "ACTION" });
  appendClientLog("SYSTEM", "SYS_FINISH", "进程自然结束，网络已收敛至恢复态。");
  frameIndex.value = maxFrameIndex.value;
}

function jumpToStage(stage) {
  if (!activeSession.value) return;
  hasReplayStarted.value = true;
  stopPlayback({ refreshDetail: false });
  frameIndex.value = stage.frame;
  appendReplayTerminalLine(`前端操作：跳转回放阶段 ${stage.label}，frame=${stage.frame}。`, { level: "ACTION" });
}

function queryLogs() {
  if (!logQuery.value) {
    logQueryMode.value = false;
    appendReplayTerminalLine("前端操作：清空场景回放日志检索。", { level: "ACTION" });
    return;
  }
  logQueryMode.value = true;
  appendReplayTerminalLine(`前端操作：检索场景回放日志 keyword=${logQuery.value}。`, { level: "ACTION" });
}

function zoomMap(factor) {
  mapView.scale = Math.max(0.72, Math.min(3.8, mapView.scale * factor));
  drawReplayMap();
}

function resetMapView() {
  mapView.scale = 1;
  mapView.offsetX = 0;
  mapView.offsetY = 0;
  selectedNode.value = null;
  drawReplayMap();
}

function handleWheel(event) {
  zoomMap(event.deltaY < 0 ? 1.12 : 0.9);
}

function handlePointerDown(event) {
  const point = clientPointToCanvasPoint(event) || { x: event.clientX, y: event.clientY };
  pointerState = {
    id: event.pointerId,
    x: point.x,
    y: point.y,
    clientX: event.clientX,
    clientY: event.clientY,
    startOffsetX: mapView.offsetX,
    startOffsetY: mapView.offsetY,
    moved: false,
  };
  dragging.value = true;
  event.currentTarget?.setPointerCapture?.(event.pointerId);
}

function handlePointerMove(event) {
  const point = clientPointToCanvasPoint(event);
  if (point) {
    const world = screenToWorld(point.x, point.y);
    pointerWorld.x = Math.max(0, Math.min(MAP_WIDTH, world.x));
    pointerWorld.y = Math.max(0, Math.min(MAP_HEIGHT, world.y));
  }

  if (!pointerState || pointerState.id !== event.pointerId) {
    return;
  }

  const currentPoint = point || { x: event.clientX, y: event.clientY };
  const dx = currentPoint.x - pointerState.x;
  const dy = currentPoint.y - pointerState.y;
  const clientDx = event.clientX - pointerState.clientX;
  const clientDy = event.clientY - pointerState.clientY;
  if (Math.abs(clientDx) + Math.abs(clientDy) > 3) pointerState.moved = true;
  mapView.offsetX = pointerState.startOffsetX + dx;
  mapView.offsetY = pointerState.startOffsetY + dy;
  markRenderDirty();
}

function handlePointerUp(event) {
  event.currentTarget?.releasePointerCapture?.(event.pointerId);
  dragging.value = false;
  window.setTimeout(() => {
    pointerState = null;
  }, 0);
}

function handleCanvasClick(event) {
  if (pointerState?.moved) return;
  selectedNode.value = findNearestNode(event);
}

function findNearestNode(event) {
  const point = clientPointToCanvasPoint(event);
  if (!point || !currentFrame.value?.nodes?.length) return null;
  const { x, y } = point;
  let best = null;
  let bestDistance = Infinity;
  currentFrame.value.nodes.forEach((node) => {
    const point = screenPointForNode(node);
    const dx = point.x - x;
    const dy = point.y - y;
    const distance = dx * dx + dy * dy;
    if (distance < bestDistance) {
      bestDistance = distance;
      best = node;
    }
  });
  return bestDistance <= 18 * 18 ? best : null;
}

function downloadArtifact(type) {
  if (!activeSessionId.value || typeof window === "undefined") return;
  appendReplayTerminalLine(`前端操作：下载场景回放${type}文件 replay_id=${activeSessionId.value}。`, { level: "ACTION" });
  window.open(
    `${API_BASE}/replay/sessions/${encodeURIComponent(activeSessionId.value)}/download?type=${encodeURIComponent(type)}`,
    "_blank"
  );
  appendReplayTerminalLine("后端响应：已打开场景回放文件下载地址。", { level: "BACKEND", source: "BACKEND" });
}

async function scrollTerminalToBottom() {
  await nextTick();
  const viewport = terminalRef.value;
  if (!viewport) return;
  viewport.scrollTo({
    top: viewport.scrollHeight,
    behavior: "auto",
  });
}

function scheduleTerminalScroll() {
  if (typeof window === "undefined") {
    void scrollTerminalToBottom();
    return;
  }
  if (terminalScrollFrame) return;
  terminalScrollFrame = window.requestAnimationFrame(() => {
    terminalScrollFrame = 0;
    void scrollTerminalToBottom();
  });
}

watch(frameIndex, (nextIndex) => {
  if (!activeSessionId.value) return;
  void loadReplayFrame(activeSessionId.value, nextIndex, { preferDetail: !playing.value });
});

watch(playbackRate, () => {
  if (playing.value) schedulePlayback();
});

watch(scenarioMapUrl, () => {
  loadScenarioMap();
}, { immediate: true });

watch(
  () => [displayTerminalEntries.value.length, frameIndex.value],
  () => {
    scheduleTerminalScroll();
  },
  { immediate: true, flush: "post" }
);

watch(
  () => {
    const entries = localReplayTerminalEntries.value;
    const last = entries[entries.length - 1]?.text || "";
    return `${entries.length}:${last}:${logQueryMode.value ? logQuery.value : ""}`;
  },
  () => {
    syncReplayTerminalEntries(localReplayTerminalEntries.value);
  },
  { flush: "post" }
);

onMounted(() => {
  refreshCanvasSize();
  resizeObserver = new ResizeObserver(refreshCanvasSize);
  if (mapShellRef.value) resizeObserver.observe(mapShellRef.value);
  window.addEventListener("resize", refreshCanvasSize);
  animationId = window.requestAnimationFrame(animationLoop);
  void refreshReplaySessions();
  void scrollTerminalToBottom();
});

onBeforeUnmount(() => {
  stopPlayback({ refreshDetail: false });
  if (animationId) window.cancelAnimationFrame(animationId);
  if (terminalScrollFrame) window.cancelAnimationFrame(terminalScrollFrame);
  if (resizeObserver) resizeObserver.disconnect();
  window.removeEventListener("resize", refreshCanvasSize);
});
</script>

<style scoped>
.scenario-replay {
  position: relative;
  width: 100%;
  height: 100%;
  min-height: 100%;
  overflow: hidden;
  background:
    linear-gradient(180deg, rgba(2, 8, 23, 0.2), rgba(2, 8, 23, 0.86)),
    url("/prototype/images/模型训练/u537.png") center / cover no-repeat,
    #050810;
  color: #e8f7ff;
  font-family: "Microsoft YaHei", "PingFang SC", "Source Han Sans CN", sans-serif;
}

.replay-map {
  position: absolute;
  inset: 0;
  overflow: hidden;
  background: #050810;
}

.replay-map__canvas {
  position: absolute;
  inset: 0;
  display: block;
  width: 100%;
  height: 100%;
  cursor: crosshair;
}

.replay-map__canvas--dragging {
  cursor: grabbing;
}

.replay-map__scan {
  position: absolute;
  inset: 0;
  pointer-events: none;
  background:
    linear-gradient(90deg, rgba(0, 200, 244, 0.08), transparent 22%, transparent 78%, rgba(0, 200, 244, 0.06)),
    radial-gradient(circle at 50% 42%, rgba(0, 240, 255, 0.1), transparent 34%),
    linear-gradient(180deg, rgba(5, 8, 16, 0.22), transparent 18%, rgba(5, 8, 16, 0.76));
  box-shadow: inset 0 0 120px rgba(0, 0, 0, 0.76);
}

.scenario-replay--damaged .replay-map__scan {
  background:
    radial-gradient(circle at 48% 42%, rgba(255, 0, 85, 0.2), transparent 32%),
    linear-gradient(180deg, rgba(5, 8, 16, 0.28), transparent 18%, rgba(5, 8, 16, 0.8));
}

.scenario-replay--recovery .replay-map__scan,
.scenario-replay--completed .replay-map__scan {
  background:
    radial-gradient(circle at 50% 42%, rgba(0, 255, 157, 0.14), transparent 34%),
    linear-gradient(180deg, rgba(5, 8, 16, 0.24), transparent 18%, rgba(5, 8, 16, 0.74));
}

.replay-map__grid-label,
.replay-map__coords {
  position: absolute;
  pointer-events: none;
  color: rgba(183, 224, 254, 0.7);
  font-size: 12px;
  letter-spacing: 0;
}

.replay-map__grid-label--top {
  left: 50%;
  top: 20px;
  transform: translateX(-50%);
}

.replay-map__coords {
  right: 24px;
  bottom: 252px;
  font-family: Consolas, Monaco, monospace;
}

.native-replay-title {
  position: absolute;
  left: 18px;
  top: 16px;
  z-index: 12;
  display: flex;
  align-items: center;
  gap: 18px;
  min-width: 620px;
  max-width: 880px;
  height: 64px;
  padding: 0 18px 0 22px;
  border: 1px solid rgba(0, 200, 244, 0.36);
  background:
    linear-gradient(90deg, rgba(4, 16, 40, 0.92), rgba(0, 72, 126, 0.58)),
    rgba(0, 20, 42, 0.82);
  box-shadow: 0 0 28px rgba(0, 200, 244, 0.2), inset 0 0 26px rgba(0, 121, 254, 0.14);
  clip-path: polygon(0 0, calc(100% - 22px) 0, 100% 22px, 100% 100%, 0 100%);
}

.native-replay-title strong {
  display: block;
  max-width: 660px;
  overflow: hidden;
  color: #bff8ff;
  font-size: 19px;
  font-weight: 700;
  text-overflow: ellipsis;
  text-shadow: 0 0 18px rgba(0, 200, 244, 0.52);
  white-space: nowrap;
}

.native-replay-title span {
  display: block;
  margin-top: 6px;
  color: #8fc6e8;
  font-size: 13px;
}

.native-replay-title button,
.native-timeline button,
.native-timeline select,
.native-terminal button,
.native-session-refresh,
.map-tools button,
.layer-switches button,
.phase-rail button {
  min-height: 34px;
  border: 1px solid rgba(0, 200, 244, 0.44);
  border-radius: 4px;
  background: linear-gradient(180deg, rgba(0, 121, 254, 0.38), rgba(0, 64, 118, 0.54));
  color: #dff9ff;
  font-size: 14px;
  cursor: pointer;
  box-shadow: inset 0 0 14px rgba(0, 240, 255, 0.16);
}

.native-replay-title button {
  width: 96px;
  flex: 0 0 auto;
}

.native-replay-title button:hover,
.native-timeline button:hover,
.native-terminal button:hover,
.native-session-refresh:hover,
.map-tools button:hover,
.layer-switches button:hover,
.phase-rail button:hover {
  border-color: rgba(102, 255, 0, 0.5);
}

.native-replay-sessions,
.native-telemetry,
.native-equipment,
.native-terminal,
.native-timeline,
.native-legend,
.phase-rail,
.map-tools,
.layer-switches,
.status-card,
.node-popover {
  border: 1px solid rgba(0, 200, 244, 0.3);
  background:
    linear-gradient(135deg, rgba(3, 13, 32, 0.92), rgba(6, 46, 82, 0.62)),
    rgba(3, 12, 28, 0.86);
  box-shadow: 0 0 28px rgba(0, 121, 254, 0.16), inset 0 0 24px rgba(0, 200, 244, 0.08);
  backdrop-filter: blur(6px);
}

.native-replay-sessions {
  position: absolute;
  left: 18px;
  top: 92px;
  z-index: 30;
  width: 430px;
  max-height: 600px;
  overflow: hidden;
}

.native-panel-tab {
  width: 46px;
  min-height: 109px;
  padding: 13px 11px;
  float: left;
  background-image: url("/prototype/images/场景回放/u3659.png");
  background-size: 46px 109px;
  color: #b7e0fe;
  font-size: 15px;
  line-height: 1.1;
  text-align: center;
  writing-mode: vertical-rl;
  letter-spacing: 2px;
}

.native-panel-tab--right {
  float: right;
}

.native-panel-body {
  max-height: 600px;
  padding: 12px;
  margin-left: 46px;
  overflow: auto;
}

.native-session-refresh {
  width: 100%;
  margin-bottom: 10px;
}

.native-session-row {
  display: block;
  width: 100%;
  margin-bottom: 10px;
  padding: 11px 12px;
  border: 1px solid rgba(183, 224, 254, 0.22);
  border-radius: 4px;
  background: rgba(8, 32, 64, 0.72);
  color: #dff9ff;
  text-align: left;
}

.native-session-row strong {
  display: block;
  overflow: hidden;
  font-size: 13px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.native-session-row span,
.native-session-row small {
  display: block;
  margin-top: 5px;
  color: #8fc6e8;
  font-size: 12px;
}

.native-session-row--active {
  border-color: rgba(102, 255, 0, 0.52);
  background: rgba(0, 121, 254, 0.22);
  box-shadow: inset 0 0 18px rgba(102, 255, 0, 0.12);
}

.native-empty {
  margin: 0;
  color: #8fc6e8;
  font-size: 13px;
  line-height: 1.6;
}

.phase-rail {
  position: absolute;
  left: 18px;
  top: 98px;
  z-index: 10;
  width: 330px;
  padding: 14px;
}

.phase-rail button {
  display: grid;
  grid-template-columns: 18px 1fr 46px;
  align-items: center;
  gap: 8px;
  width: 100%;
  min-height: 42px;
  margin-bottom: 8px;
  padding: 0 10px;
  text-align: left;
  background: rgba(8, 32, 64, 0.72);
}

.phase-rail button i {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: #4b6b83;
  box-shadow: 0 0 0 4px rgba(75, 107, 131, 0.14);
}

.phase-rail button span {
  color: #dff9ff;
  font-size: 13px;
}

.phase-rail button small {
  color: #8fc6e8;
  text-align: right;
}

.phase-rail button.passed i {
  background: #00c8f4;
  box-shadow: 0 0 0 4px rgba(0, 200, 244, 0.14), 0 0 14px rgba(0, 200, 244, 0.42);
}

.phase-rail button.active {
  border-color: rgba(102, 255, 0, 0.55);
}

.phase-rail button.active i {
  background: #66ff00;
  box-shadow: 0 0 0 4px rgba(102, 255, 0, 0.14), 0 0 16px rgba(102, 255, 0, 0.5);
}

.native-legend {
  position: absolute;
  right: 18px;
  top: 98px;
  z-index: 10;
  width: 230px;
  padding: 12px 14px;
  color: #fff;
}

.native-legend__head {
  height: 34px;
  margin: -12px -14px 12px;
  padding: 8px 14px;
  background: linear-gradient(90deg, rgba(0, 121, 254, 0.22), transparent);
  color: #bff8ff;
  font-weight: 700;
}

.legend-row {
  display: flex;
  align-items: center;
  gap: 11px;
  min-height: 32px;
  color: #e8f7ff;
  font-size: 14px;
}

.legend-dot,
.legend-node {
  display: inline-block;
  width: 13px;
  height: 13px;
  border-radius: 999px;
  box-shadow: 0 0 12px currentColor;
}

.legend-dot--user {
  background: #00ff9d;
  color: #00ff9d;
}

.legend-dot--macro {
  background: #00f0ff;
  color: #00f0ff;
}

.legend-dot--manpack {
  background: #ffb700;
  color: #ffb700;
}

.legend-dot--small-cell {
  background: #b026ff;
  color: #b026ff;
}

.legend-dot--relay {
  background: #ff0055;
  color: #ff0055;
}

.legend-line {
  width: 28px;
  height: 0;
  border-top: 2px solid #00ff9d;
  box-shadow: 0 0 8px rgba(0, 255, 157, 0.55);
}

.legend-line--backhaul {
  border-top-color: #00f0ff;
  box-shadow: 0 0 8px rgba(0, 240, 255, 0.55);
}

.native-telemetry {
  position: absolute;
  left: 18px;
  bottom: 314px;
  z-index: 10;
  width: 970px;
  min-height: 112px;
}

.native-metric-grid {
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: 8px;
  padding: 14px 14px 12px 62px;
}

.native-metric-grid article {
  position: relative;
  min-height: 82px;
  padding: 13px 12px 10px;
  border: 1px solid rgba(183, 224, 254, 0.18);
  background:
    linear-gradient(180deg, rgba(0, 121, 254, 0.18), rgba(2, 14, 34, 0.72)),
    rgba(5, 22, 45, 0.86);
  box-shadow: inset 0 0 18px rgba(0, 200, 244, 0.08);
}

.native-metric-grid small,
.native-equipment-grid small {
  display: block;
  color: #9fd4f4;
  font-size: 13px;
}

.native-metric-grid strong {
  display: inline-block;
  margin-top: 7px;
  color: #66ff00;
  font-family: Arial, "Microsoft YaHei", sans-serif;
  font-size: 25px;
  font-weight: 400;
}

.native-metric-grid span {
  margin-left: 5px;
  color: #b7e0fe;
  font-size: 13px;
}

.native-equipment {
  position: absolute;
  right: 18px;
  bottom: 314px;
  z-index: 10;
  width: 628px;
  min-height: 112px;
}

.native-equipment-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 10px;
  padding: 15px 62px 14px 14px;
}

.native-equipment-grid article {
  display: grid;
  grid-template-columns: 38px 1fr;
  gap: 10px;
  align-items: center;
  min-height: 80px;
  padding: 10px;
  border: 1px solid rgba(183, 224, 254, 0.18);
  background: rgba(5, 22, 45, 0.78);
}

.native-equipment-grid img {
  max-width: 38px;
  max-height: 44px;
  object-fit: contain;
  filter: drop-shadow(0 0 10px rgba(0, 200, 244, 0.32));
}

.native-equipment-grid strong {
  display: block;
  margin-top: 7px;
  color: #b7e0fe;
  font-family: Arial, "Microsoft YaHei", sans-serif;
  font-size: 20px;
  font-weight: 400;
}

.native-equipment-grid strong span {
  color: #66ff00;
  font-size: 24px;
}

.native-timeline {
  position: absolute;
  left: 50%;
  bottom: 246px;
  z-index: 12;
  display: flex;
  align-items: center;
  gap: 12px;
  width: 1180px;
  height: 56px;
  padding: 0 16px;
  transform: translateX(-50%);
}

.native-timeline button {
  width: 82px;
}

.native-timeline label {
  flex: 1;
  display: grid;
  grid-template-columns: 128px minmax(0, 1fr);
  gap: 12px;
  align-items: center;
  color: #9fd4f4;
  font-size: 13px;
}

.native-timeline input[type="range"] {
  width: 100%;
  accent-color: #00c8f4;
}

.native-timeline select {
  width: 76px;
  height: 34px;
  color: #dff9ff;
}

.native-terminal {
  position: absolute;
  left: 18px;
  right: 18px;
  bottom: 18px;
  z-index: 12;
  height: 210px;
  overflow: hidden;
}

.native-terminal__head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 50px;
  padding: 0 12px 0 16px;
  border-bottom: 1px solid rgba(0, 200, 244, 0.24);
  background: linear-gradient(90deg, rgba(0, 121, 254, 0.2), transparent);
}

.native-terminal__head strong {
  display: block;
  color: #bff8ff;
  font-size: 16px;
}

.native-terminal__head span {
  display: block;
  margin-top: 3px;
  color: #8fc6e8;
  font-size: 12px;
}

.native-terminal__head button {
  width: 92px;
}

.native-terminal__viewport {
  height: 160px;
  overflow: auto;
  padding: 12px 16px;
  background: rgba(51, 51, 51, 0.9);
  color: #dbeafe;
  font-family: Consolas, Monaco, monospace;
  font-size: 12px;
  line-height: 1.65;
}

.native-terminal__viewport p {
  margin: 0 0 2px;
  white-space: pre-wrap;
  word-break: break-word;
}

.status-card {
  position: absolute;
  right: 270px;
  top: 20px;
  z-index: 12;
  display: inline-flex;
  align-items: center;
  gap: 10px;
  height: 44px;
  padding: 0 16px;
  color: #dff9ff;
}

.status-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: #66ff00;
  box-shadow: 0 0 14px currentColor;
}

.status-dot.red {
  background: #ff4d6d;
  color: #ff4d6d;
}

.status-dot.yellow {
  background: #facc15;
  color: #facc15;
}

.status-dot.cyan,
.status-dot.green {
  background: #00f0ff;
  color: #00f0ff;
}

.scenario-replay--playing .status-dot {
  animation: replayPulse 1.2s ease-in-out infinite;
}

.disaster-alert {
  position: absolute;
  left: 50%;
  top: 92px;
  z-index: 12;
  max-width: 760px;
  padding: 12px 20px;
  transform: translateX(-50%);
  border: 1px solid rgba(255, 77, 109, 0.3);
  background: rgba(70, 14, 32, 0.78);
  color: #ffd6df;
  box-shadow: 0 0 26px rgba(255, 77, 109, 0.18);
  opacity: 0;
  transition: opacity 0.2s ease;
}

.disaster-alert.show {
  opacity: 1;
}

.disaster-alert.recovery {
  border-color: rgba(0, 240, 255, 0.38);
  background: rgba(0, 240, 255, 0.12);
  color: #bff8ff;
  box-shadow: 0 0 26px rgba(0, 240, 255, 0.22);
}

.map-tools {
  position: absolute;
  right: 18px;
  top: 304px;
  z-index: 12;
  display: grid;
  gap: 8px;
  width: 82px;
  padding: 10px;
}

.map-tools button {
  width: 100%;
}

.layer-switches {
  position: absolute;
  right: 18px;
  top: 442px;
  z-index: 12;
  display: grid;
  gap: 8px;
  width: 150px;
  padding: 10px;
}

.layer-switches button {
  width: 100%;
  text-align: left;
  background: rgba(8, 32, 64, 0.72);
}

.layer-switches button.active {
  border-color: rgba(102, 255, 0, 0.52);
  color: #eaffd7;
}

.node-popover {
  position: absolute;
  z-index: 40;
  width: 240px;
  padding: 12px;
  color: #e8f7ff;
  pointer-events: none;
}

.node-popover strong,
.node-popover span {
  display: block;
}

.node-popover strong {
  margin-bottom: 6px;
  color: #66ff00;
  font-size: 14px;
}

.node-popover span {
  color: #b7e0fe;
  font-size: 12px;
  line-height: 1.6;
}

.replay-error {
  position: absolute;
  left: 50%;
  top: 50%;
  z-index: 60;
  max-width: 680px;
  padding: 14px 18px;
  transform: translate(-50%, -50%);
  border: 1px solid rgba(248, 113, 113, 0.34);
  background: rgba(69, 10, 10, 0.86);
  color: #fecaca;
}

.native-terminal button:disabled,
.native-timeline button:disabled,
.native-timeline select:disabled,
.native-session-refresh:disabled,
.phase-rail button:disabled {
  cursor: not-allowed;
  opacity: 0.46;
}

.panel-fade-enter-active,
.panel-fade-leave-active {
  transition: opacity 0.18s ease, transform 0.18s ease;
}

.panel-fade-enter-from,
.panel-fade-leave-to {
  opacity: 0;
  transform: translateY(-6px);
}

@keyframes replayPulse {
  0%,
  100% {
    opacity: 1;
  }
  50% {
    opacity: 0.55;
  }
}

@media (max-width: 900px) {
  .native-replay-title {
    left: 10px;
    right: 10px;
    min-width: 0;
    max-width: none;
  }

  .phase-rail,
  .native-legend,
  .native-telemetry,
  .native-equipment,
  .map-tools,
  .layer-switches,
  .status-card,
  .disaster-alert {
    display: none;
  }

  .native-timeline {
    left: 10px;
    right: 10px;
    bottom: 238px;
    width: auto;
    transform: none;
  }

  .native-terminal {
    left: 10px;
    right: 10px;
  }
}

.scenario-replay {
  --primary: #00f0ff;
  --primary-dark: #0088cc;
  --success: #00ff9d;
  --warning: #ffb700;
  --danger: #ff0055;
  --bg-base: #050810;
  --bg-panel: rgba(13, 20, 36, 0.75);
  --text-main: #f8fafc;
  --text-muted: #94a3b8;
  --border-glow: rgba(0, 240, 255, 0.25);
  --border-panel: rgba(255, 255, 255, 0.08);
  --shadow-panel: 0 8px 32px rgba(0, 0, 0, 0.6);
  --glow-text: 0 0 10px rgba(0, 240, 255, 0.5);
  --c-user: #00ff9d;
  --c-macro: #00f0ff;
  --c-manpack: #ffb700;
  --c-smallcell: #b026ff;
  --c-relay: #ff0055;
  background: var(--bg-base);
  color: var(--text-main);
  font-family: Inter, "Microsoft YaHei", "PingFang SC", sans-serif;
}

.map-container {
  position: fixed;
  inset: 0;
  z-index: 1;
  overflow: hidden;
  background: var(--bg-base);
}

.map-container::after {
  content: "";
  position: absolute;
  inset: 0;
  z-index: 120;
  pointer-events: none;
  opacity: 0.28;
  transition: opacity 0.35s ease, background 0.35s ease;
  background: radial-gradient(circle at 50% 46%, rgba(0, 240, 255, 0.08) 0%, rgba(0, 240, 255, 0.03) 42%, transparent 74%);
}

.scenario-replay--native-normal .map-container::after {
  opacity: 0.18;
}

.scenario-replay--native-disaster .map-container::after {
  opacity: 0.52;
  background: radial-gradient(circle at 50% 46%, rgba(255, 0, 85, 0.22) 0%, rgba(255, 0, 85, 0.12) 38%, transparent 72%);
}

.scenario-replay--native-recovery .map-container::after,
.scenario-replay--native-restored .map-container::after {
  opacity: 0.34;
  background: radial-gradient(circle at 50% 46%, rgba(0, 255, 157, 0.14) 0%, rgba(0, 240, 255, 0.1) 38%, transparent 72%);
}

.overlay-canvas {
  position: absolute;
  inset: 0;
  z-index: 500;
  display: block;
  width: 100%;
  height: 100%;
  cursor: pointer;
  touch-action: none;
  user-select: none;
}

.glass-panel {
  background: var(--bg-panel);
  border: 1px solid var(--border-panel);
  box-shadow: var(--shadow-panel);
  backdrop-filter: blur(16px);
}

.top-bar {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  z-index: 1000;
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 64px;
  padding: 0 24px;
  border-bottom: 1px solid var(--border-glow);
  background: linear-gradient(180deg, rgba(5, 8, 16, 0.9), rgba(5, 8, 16, 0.5));
}

.logo,
.top-right,
.log-search,
.playback-controls,
.setting-group {
  display: flex;
  align-items: center;
}

.logo {
  gap: 14px;
}

.logo-icon {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 40px;
  height: 40px;
  border: 1px solid var(--primary);
  border-radius: 8px;
  background: rgba(0, 240, 255, 0.1);
  color: var(--primary);
  font-family: "JetBrains Mono", Consolas, monospace;
  font-size: 12px;
  font-weight: 700;
  box-shadow: 0 0 15px rgba(0, 240, 255, 0.2);
}

.logo-text {
  color: var(--text-main);
  font-size: 18px;
  font-weight: 700;
  letter-spacing: 1px;
  text-transform: uppercase;
  text-shadow: var(--glow-text);
}

.top-center {
  min-width: 420px;
}

.experiment-select {
  position: relative;
}

.experiment-select select {
  width: 100%;
  min-width: 420px;
  max-width: 680px;
  height: 36px;
  padding: 0 36px 0 16px;
  appearance: none;
  border: 1px solid var(--border-panel);
  border-radius: 6px;
  outline: none;
  background: rgba(0, 0, 0, 0.4);
  color: var(--text-main);
  font-family: "JetBrains Mono", Consolas, monospace;
  font-size: 13px;
  cursor: pointer;
  transition: all 0.3s;
}

.experiment-select::after {
  content: "▼";
  position: absolute;
  right: 12px;
  top: 50%;
  transform: translateY(-50%);
  color: var(--primary);
  font-size: 10px;
  pointer-events: none;
}

.experiment-select select:hover,
.experiment-select select:focus {
  border-color: var(--primary);
  box-shadow: 0 0 10px rgba(0, 240, 255, 0.2);
}

.top-right {
  gap: 12px;
}

.btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  min-height: 34px;
  padding: 8px 20px;
  border-radius: 6px;
  font: inherit;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s;
}

.btn-primary {
  border: 1px solid var(--primary);
  background: rgba(0, 240, 255, 0.15);
  color: var(--primary);
  box-shadow: inset 0 0 10px rgba(0, 240, 255, 0.1);
}

.btn-primary:hover {
  background: var(--primary);
  color: #000;
  box-shadow: 0 0 20px rgba(0, 240, 255, 0.4);
}

.btn-outline {
  border: 1px solid var(--border-panel);
  background: transparent;
  color: var(--text-muted);
}

.btn-outline:hover {
  border-color: var(--text-muted);
  color: var(--text-main);
}

.btn-compact {
  min-height: 30px;
  padding: 6px 12px;
  font-size: 12px;
}

.settings-bar {
  position: fixed;
  top: 64px;
  left: 0;
  right: 0;
  z-index: 990;
  display: flex;
  align-items: center;
  gap: 20px;
  height: 56px;
  padding: 0 24px;
  border-bottom: 1px solid var(--border-panel);
  background: rgba(13, 20, 36, 0.85);
  color: var(--text-main);
  font-size: 12px;
}

.settings-title {
  margin-right: 10px;
  color: var(--primary);
  font-weight: 700;
  text-shadow: var(--glow-text);
}

.setting-group {
  gap: 8px;
}

.setting-label {
  color: var(--text-muted);
  font-weight: 500;
}

.setting-input {
  width: 65px;
  padding: 6px 8px;
  border: 1px solid var(--border-panel);
  border-radius: 4px;
  outline: none;
  background: rgba(0, 0, 0, 0.5);
  color: var(--primary);
  font-family: "JetBrains Mono", Consolas, monospace;
  text-align: center;
}

.setting-divider {
  width: 1px;
  height: 24px;
  margin: 0 4px;
  background: var(--border-panel);
}

.settings-action {
  margin-left: auto;
  min-height: 30px;
  padding: 6px 16px;
}

.left-panel,
.right-panel {
  position: fixed;
  top: 136px;
  z-index: 900;
  overflow-y: auto;
  max-height: calc(100vh - 430px);
  border-radius: 12px;
}

.left-panel {
  left: 20px;
  width: 300px;
}

.right-panel {
  right: 20px;
  width: 280px;
}

.panel-header {
  padding: 14px 20px;
  border-bottom: 1px solid var(--border-panel);
  color: var(--text-muted);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 1px;
  text-transform: uppercase;
}

.panel-content {
  padding: 20px;
}

.panel-header--nested {
  margin: 20px -20px 0;
  padding-top: 20px;
}

.stat-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
  margin-bottom: 20px;
}

.stat-card {
  padding: 12px;
  border: 1px solid var(--border-panel);
  border-radius: 8px;
  background: rgba(0, 0, 0, 0.3);
}

.stat-card.full {
  grid-column: span 2;
}

.stat-label {
  margin-bottom: 6px;
  color: var(--text-muted);
  font-size: 11px;
}

.stat-value {
  color: var(--text-main);
  font-family: "JetBrains Mono", Consolas, monospace;
  font-size: 22px;
  font-weight: 700;
}

.stat-value.primary {
  color: var(--primary);
  text-shadow: 0 0 10px rgba(0, 240, 255, 0.4);
}

.stat-value.success {
  color: var(--success);
  text-shadow: 0 0 10px rgba(0, 255, 157, 0.4);
}

.stat-value.warning {
  color: var(--warning);
  text-shadow: 0 0 10px rgba(255, 183, 0, 0.4);
}

.stat-unit {
  margin-left: 4px;
  color: var(--text-muted);
  font-size: 12px;
  font-weight: 400;
}

.left-panel .status-card {
  position: static;
  display: flex;
  align-items: center;
  gap: 12px;
  height: auto;
  padding: 14px;
  border-radius: 8px;
  color: var(--text-main);
  transform: none;
  box-shadow: none;
}

.left-panel .status-card.normal {
  border-color: rgba(0, 255, 157, 0.2);
  background: rgba(0, 255, 157, 0.05);
}

.left-panel .status-card.disaster {
  border-color: rgba(255, 0, 85, 0.4);
  background: rgba(255, 0, 85, 0.1);
}

.left-panel .status-card.recovery {
  border-color: rgba(0, 240, 255, 0.3);
  background: rgba(0, 240, 255, 0.1);
}

.status-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  box-shadow: 0 0 8px currentColor;
}

.status-dot.green {
  background: var(--success);
  color: var(--success);
}

.status-dot.red {
  background: var(--danger);
  color: var(--danger);
  animation: blink 1s infinite;
}

.status-dot.cyan {
  background: var(--primary);
  color: var(--primary);
  animation: pulse 1.4s ease-in-out infinite;
}

.status-text {
  font-size: 13px;
  font-weight: 600;
  letter-spacing: 0.5px;
}

.mini-chart {
  width: 100%;
  height: 70px;
  margin-top: 10px;
}

.equipment-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px;
  margin-bottom: 8px;
  border: 1px solid transparent;
  border-radius: 8px;
  background: rgba(0, 0, 0, 0.2);
  transition: all 0.2s;
}

.equipment-item:hover {
  border-color: var(--border-panel);
  background: rgba(0, 0, 0, 0.4);
}

.equipment-left {
  display: flex;
  align-items: center;
  gap: 12px;
}

.equipment-icon {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  box-shadow: 0 0 8px currentColor;
}

.equipment-name {
  color: var(--text-main);
  font-size: 12px;
  font-weight: 600;
}

.equipment-count {
  margin-top: 2px;
  color: var(--text-muted);
  font-family: "JetBrains Mono", Consolas, monospace;
  font-size: 11px;
}

.equipment-status {
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
}

.equipment-status.online {
  background: rgba(0, 255, 157, 0.1);
  color: var(--success);
}

.equipment-status.offline {
  background: rgba(255, 0, 85, 0.1);
  color: var(--danger);
}

.bottom-log-panel {
  position: fixed;
  left: 20px;
  right: 20px;
  bottom: 86px;
  z-index: 900;
  display: flex;
  flex-direction: column;
  height: 200px;
  overflow: hidden;
  border-radius: 12px;
}

.log-panel-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 50px;
  padding: 12px 20px;
  border-bottom: 1px solid var(--border-panel);
  background: rgba(0, 0, 0, 0.2);
}

.log-title {
  color: var(--text-muted);
  font-size: 13px;
  font-weight: 700;
  letter-spacing: 1px;
  text-transform: uppercase;
}

.log-search {
  gap: 10px;
}

.log-search input {
  width: 280px;
  padding: 6px 16px;
  border: 1px solid var(--border-panel);
  border-radius: 20px;
  outline: none;
  background: rgba(0, 0, 0, 0.5);
  color: var(--text-main);
  font-size: 12px;
  transition: 0.3s;
}

.log-search input:focus {
  border-color: var(--primary);
  box-shadow: 0 0 10px rgba(0, 240, 255, 0.2);
}

.event-log {
  flex: 1;
  overflow-y: auto;
  padding: 10px 20px;
  color: var(--text-muted);
  font-family: "JetBrains Mono", Consolas, monospace;
  font-size: 12px;
  line-height: 1.6;
}

.event-log p {
  margin: 0;
  padding: 4px 0;
  border-bottom: 1px dashed rgba(255, 255, 255, 0.05);
  white-space: pre-wrap;
  word-break: break-word;
}

.event-log .log-entry--error {
  color: var(--danger);
  text-shadow: 0 0 5px rgba(255, 0, 85, 0.4);
}

.event-log .log-entry--warn {
  color: var(--warning);
}

.event-log .log-entry--system {
  color: var(--primary);
  text-shadow: 0 0 5px rgba(0, 240, 255, 0.3);
}

.legend {
  position: fixed;
  right: 20px;
  bottom: 300px;
  z-index: 900;
  width: auto;
  padding: 16px;
  border-radius: 10px;
}

.legend-title {
  margin-bottom: 12px;
  color: var(--text-muted);
  font-size: 10px;
  letter-spacing: 1px;
  text-transform: uppercase;
}

.legend-title--links {
  margin-top: 16px;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 10px;
  min-height: 0;
  margin-bottom: 8px;
  color: var(--text-main);
  font-size: 11px;
}

.legend .legend-dot {
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  box-shadow: 0 0 8px currentColor;
}

.legend-dot--user {
  background: var(--c-user);
  color: var(--c-user);
}

.legend-dot--macro {
  background: var(--c-macro);
  color: var(--c-macro);
}

.legend-dot--manpack {
  background: var(--c-manpack);
  color: var(--c-manpack);
}

.legend-dot--small-cell {
  background: var(--c-smallcell);
  color: var(--c-smallcell);
}

.legend-dot--relay {
  background: var(--c-relay);
  color: var(--c-relay);
}

.legend .legend-line {
  width: 20px;
  height: 2px;
  box-shadow: 0 0 5px currentColor;
}

.legend-line--mesh {
  background: var(--c-user);
  color: var(--c-user);
}

.legend-line--backhaul {
  background: var(--c-macro);
  color: var(--c-macro);
}

.bottom-bar {
  position: fixed;
  left: 0;
  right: 0;
  bottom: 0;
  z-index: 1000;
  display: flex;
  align-items: center;
  gap: 20px;
  height: 70px;
  padding: 0 30px;
  border-top: 1px solid var(--border-glow);
  background: linear-gradient(0deg, rgba(5, 8, 16, 0.95), rgba(5, 8, 16, 0.7));
}

.time-display {
  min-width: 60px;
  color: var(--text-muted);
  font-family: "JetBrains Mono", Consolas, monospace;
  font-size: 13px;
}

.time-display .current {
  color: var(--primary);
  font-size: 18px;
  font-weight: 700;
  text-shadow: var(--glow-text);
}

.timeline-slider {
  flex: 1;
  height: 4px;
  appearance: none;
  border-radius: 2px;
  outline: none;
  background: rgba(255, 255, 255, 0.1);
}

.timeline-slider::-webkit-slider-thumb {
  width: 16px;
  height: 16px;
  appearance: none;
  border-radius: 50%;
  background: var(--primary);
  cursor: pointer;
  box-shadow: 0 0 15px var(--primary);
}

.playback-controls {
  gap: 10px;
}

.control-btn {
  width: 36px;
  height: 36px;
  border: 1px solid var(--border-panel);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.05);
  color: var(--text-main);
  cursor: pointer;
  transition: 0.2s;
}

.control-btn:hover {
  border-color: var(--primary);
  background: rgba(0, 240, 255, 0.2);
  color: var(--primary);
}

.control-btn.active {
  background: var(--primary);
  color: #000;
  box-shadow: 0 0 15px var(--primary);
}

.disaster-alert {
  position: fixed;
  top: 140px;
  left: 50%;
  z-index: 1100;
  display: none;
  align-items: center;
  gap: 10px;
  max-width: none;
  padding: 12px 30px;
  transform: translateX(-50%);
  border: 1px solid var(--danger);
  border-radius: 30px;
  background: rgba(255, 0, 85, 0.15);
  color: var(--danger);
  font-size: 14px;
  font-weight: 700;
  letter-spacing: 1px;
  box-shadow: 0 0 30px rgba(255, 0, 85, 0.4);
  animation: alert-flash 1s infinite;
}

.disaster-alert.show {
  display: flex;
  opacity: 1;
}

.disaster-alert.recovery {
  border-color: rgba(0, 240, 255, 0.38);
  background: rgba(0, 240, 255, 0.12);
  color: #bff8ff;
  box-shadow: 0 0 26px rgba(0, 240, 255, 0.22);
  animation: none;
}

.btn:disabled,
.control-btn:disabled,
.timeline-slider:disabled,
.experiment-select select:disabled {
  cursor: not-allowed;
  opacity: 0.46;
}

@keyframes alert-flash {
  0%,
  100% {
    opacity: 1;
  }
  50% {
    opacity: 0.6;
  }
}

@keyframes blink {
  0%,
  100% {
    opacity: 1;
  }
  50% {
    opacity: 0.35;
  }
}

@media (max-width: 1100px) {
  .top-center,
  .experiment-select select {
    min-width: 280px;
  }

  .settings-bar {
    overflow-x: auto;
  }

  .left-panel,
  .right-panel,
  .legend {
    display: none;
  }

  .bottom-log-panel {
    left: 10px;
    right: 10px;
  }
}

.scenario-replay {
  position: relative;
  width: 100%;
  height: 100%;
  min-height: 0;
  overflow: hidden;
}

.scenario-replay .map-container,
.scenario-replay .top-bar,
.scenario-replay .settings-bar,
.scenario-replay .left-panel,
.scenario-replay .right-panel,
.scenario-replay .bottom-log-panel,
.scenario-replay .legend,
.scenario-replay .bottom-bar,
.scenario-replay .disaster-alert,
.scenario-replay .replay-error {
  position: absolute;
}

.scenario-replay .map-container {
  inset: 0;
}

.scenario-replay .top-bar {
  top: 0;
  left: 0;
  right: 0;
}

.scenario-replay .settings-bar {
  top: 64px;
  left: 0;
  right: 0;
}

.scenario-replay .left-panel,
.scenario-replay .right-panel {
  top: 136px;
  max-height: calc(100% - 430px);
}

.scenario-replay .left-panel {
  left: 20px;
}

.scenario-replay .right-panel {
  right: 20px;
}

.scenario-replay .bottom-log-panel {
  left: 20px;
  right: 20px;
  bottom: 86px;
}

.scenario-replay .legend {
  right: 20px;
  bottom: 300px;
}

.scenario-replay .bottom-bar {
  left: 0;
  right: 0;
  bottom: 0;
}

.scenario-replay .disaster-alert {
  top: 140px;
  left: 50%;
}

.scenario-replay .replay-error {
  left: 50%;
  top: 50%;
  transform: translate(-50%, -50%);
}
</style>
