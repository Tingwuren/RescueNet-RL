<template>
  <div class="replay-panel">
    <div class="replay-toolbar">
      <label>
        回放数据
        <select v-model="selectedReplayKey" @change="onReplayChange">
          <option v-for="item in replayOptions" :key="item.key" :value="item.key">
            {{ item.label }}
          </option>
        </select>
      </label>
      <button type="button" @click="loadReplaySources" :disabled="loadingExperiments">
        {{ loadingExperiments ? "刷新中..." : "刷新回放列表" }}
      </button>
      <a class="native-link" :href="nativeReplayUrl" target="_blank" rel="noreferrer">打开 ns-3 原生回放页</a>
    </div>

    <div v-if="errorMessage" class="error-banner">{{ errorMessage }}</div>
    <div v-if="localReplayWarning" class="warning-banner">{{ localReplayWarning }}</div>

    <div class="native-shell">
      <aside class="native-panel native-panel--left">
        <div class="panel-title">网络状态实况</div>

        <article class="stat-card">
          <div class="stat-label">当前时间</div>
          <div class="stat-value stat-value--accent">{{ formattedCurrentTime }}</div>
        </article>

        <article class="stat-card">
          <div class="stat-label">系统吞吐量</div>
          <div class="stat-value stat-value--success">{{ formattedCurrentTp }}</div>
        </article>

        <article class="stat-card">
          <div class="stat-label">丢包率</div>
          <div class="stat-value" :class="{ 'stat-value--danger': currentLossPercent > 5 }">
            {{ formattedCurrentLoss }}
          </div>
        </article>

        <article class="chart-box">
          <div class="chart-title">吞吐量趋势 (Mbps)</div>
          <svg :viewBox="`0 0 ${chartWidth} ${chartHeight}`" preserveAspectRatio="none" class="chart-svg">
            <defs>
              <linearGradient id="tpGradient" x1="0" x2="0" y1="0" y2="1">
                <stop offset="0%" stop-color="#34d399" stop-opacity="0.38" />
                <stop offset="100%" stop-color="#34d399" stop-opacity="0.04" />
              </linearGradient>
            </defs>
            <path v-if="tpAreaPath" :d="tpAreaPath" fill="url(#tpGradient)" />
            <path v-if="tpLinePath" :d="tpLinePath" fill="none" stroke="#34d399" stroke-width="2.5" />
            <circle v-for="point in tpChartPoints" :key="`tp-${point.x}`" :cx="point.x" :cy="point.y" r="2.6" fill="#86efac" />
          </svg>
        </article>

        <article class="chart-box">
          <div class="chart-title">丢包率趋势 (%)</div>
          <svg :viewBox="`0 0 ${chartWidth} ${chartHeight}`" preserveAspectRatio="none" class="chart-svg">
            <defs>
              <linearGradient id="lossGradient" x1="0" x2="0" y1="0" y2="1">
                <stop offset="0%" stop-color="#f87171" stop-opacity="0.34" />
                <stop offset="100%" stop-color="#f87171" stop-opacity="0.05" />
              </linearGradient>
            </defs>
            <path v-if="lossAreaPath" :d="lossAreaPath" fill="url(#lossGradient)" />
            <path v-if="lossLinePath" :d="lossLinePath" fill="none" stroke="#f87171" stroke-width="2.5" />
            <circle v-for="point in lossChartPoints" :key="`loss-${point.x}`" :cx="point.x" :cy="point.y" r="2.6" fill="#fecaca" />
          </svg>
        </article>

        <article class="status-card" :class="{ 'status-card--danger': currentFrame?.disaster === 1 }">
          <div class="stat-label">灾害状态</div>
          <div class="status-value">
            {{ currentFrame?.disaster === 1 ? "灾后阶段" : "网络正常" }}
          </div>
          <p class="status-hint">{{ sourceDescription }}</p>
        </article>
      </aside>

      <section class="map-area">
        <div class="alert-banner" :class="{ 'alert-banner--visible': currentFrame?.disaster === 1 }">
          警告：检测到自然灾害，应急网络接管中
        </div>

        <div class="map-stage-header">
          <div>
            <p class="map-title">应急通信联合救援数字孪生平台</p>
            <p class="map-subtitle">{{ frameSummary || "请选择回放源并开始播放。" }}</p>
          </div>
          <div class="map-badges">
            <span class="map-badge">{{ sourceBadge }}</span>
            <span class="map-badge">{{ formattedCoverage }}</span>
            <span class="map-badge">{{ formattedBroadcast }}</span>
          </div>
        </div>

        <div class="replay-stage">
          <canvas ref="canvasRef"></canvas>
          <div class="legend">
            <div class="legend-item"><span class="legend-dot legend-dot--user"></span> 用户终端</div>
            <div class="legend-item"><span class="legend-dot legend-dot--macro"></span> 宏基站</div>
            <div class="legend-item"><span class="legend-dot legend-dot--manpack"></span> 背负式基站</div>
            <div class="legend-item"><span class="legend-line legend-line--wifi"></span> 广播/近距链路</div>
            <div class="legend-item"><span class="legend-line legend-line--lte"></span> 接入/回传链路</div>
          </div>
        </div>
      </section>

      <aside class="native-panel native-panel--right">
        <div class="panel-title">回放与装备</div>

        <article class="equip-card">
          <div class="equip-name">
            当前回放
            <span class="badge">{{ sourceBadge }}</span>
          </div>
          <div class="equip-desc">{{ sourceDescription }}</div>
        </article>

        <article class="equip-card equip-card--summary">
          <div class="equip-name">当前帧概况</div>
          <div class="summary-grid">
            <div class="summary-item">
              <span>用户</span>
              <strong>{{ currentUserCount }}</strong>
            </div>
            <div class="summary-item">
              <span>基站</span>
              <strong>{{ currentStationCount }}</strong>
            </div>
            <div class="summary-item">
              <span>链路</span>
              <strong>{{ parsedLinks.length }}</strong>
            </div>
            <div class="summary-item">
              <span>预算</span>
              <strong>{{ formattedBudget }}</strong>
            </div>
          </div>
        </article>

        <article v-for="item in equipmentCards" :key="item.title" class="equip-card" :class="item.className">
          <div class="equip-name">
            {{ item.title }}
            <span v-if="item.badge" class="badge">{{ item.badge }}</span>
          </div>
          <div class="equip-desc">{{ item.description }}</div>
        </article>
      </aside>
    </div>

    <div class="timeline" v-if="maxFrameIndex >= 0">
      <button type="button" class="timeline-btn" @click="step(-1)" :disabled="frameIndex <= 0">上一帧</button>
      <button type="button" class="timeline-btn timeline-btn--primary" @click="togglePlayback" :disabled="maxFrameIndex <= 0">
        {{ isPlaying ? "暂停" : "播放" }}
      </button>
      <button type="button" class="timeline-btn" @click="step(1)" :disabled="frameIndex >= maxFrameIndex">下一帧</button>
      <span class="timeline-time">0 s</span>
      <input
        type="range"
        class="timeline-slider"
        min="0"
        :max="maxFrameIndex"
        step="1"
        v-model.number="frameIndex"
        @change="loadFrame(frameIndex)"
      />
      <span class="timeline-time">{{ formattedMaxTime }}</span>
    </div>
  </div>
</template>

<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import axios from "axios";
import { ns3ApiBase, ns3WebBase, rescueApiBase } from "../utils/runtimeEndpoints";
import {
  getActiveReplaySessionId,
  listReplaySessions,
  saveReplaySessionFromSimulation,
  setActiveReplaySessionId,
} from "../utils/replaySessions";

const nativeReplayUrl = ns3WebBase;

const experiments = ref([]);
const localSessions = ref([]);
const transientSessions = ref([]);
const latestTrainingArtifact = ref(null);
const selectedReplayKey = ref(null);
const frameIndex = ref(0);
const maxFrameIndex = ref(-1);
const currentFrame = ref(null);
const loadingExperiments = ref(false);
const errorMessage = ref("");
const isPlaying = ref(false);
const canvasRef = ref(null);
const activeSource = ref("none");

let timer = null;

const allSessions = computed(() => [...transientSessions.value, ...localSessions.value]);

const trainingArtifactSignature = (artifact) =>
  artifact
    ? [
        artifact.scenario_name,
        artifact.algorithm || "ppo",
        artifact.reward_mode || "default",
        Number(artifact.updated_at || 0).toFixed(0),
      ].join("|")
    : null;

const fetchLatestTrainingArtifact = async () => {
  try {
    const { data } = await axios.get(`${rescueApiBase}/train/latest-artifact`, { timeout: 10000 });
    latestTrainingArtifact.value = data || null;
    return data || null;
  } catch (error) {
    if (error?.response?.status === 404) {
      latestTrainingArtifact.value = null;
      return null;
    }
    console.warn("Failed to fetch latest training artifact", error);
    return latestTrainingArtifact.value;
  }
};

const materializeTrainingReplay = async (artifact = latestTrainingArtifact.value) => {
  const scenarioName = artifact?.scenario_name;
  const checkpointPath = artifact?.checkpoint_path;
  const algorithm = artifact?.algorithm || "ppo";
  if (!scenarioName || !checkpointPath) return null;

  const artifactSignature = trainingArtifactSignature(artifact);
  const existing = listReplaySessions().find(
    (session) => session.source === "training" && session.artifactSignature === artifactSignature
  );
  if (existing) return existing;
  const transient = transientSessions.value.find(
    (session) => session.source === "training" && session.artifactSignature === artifactSignature
  );
  if (transient) return transient;

  const { data: simulation } = await axios.post(
      `${rescueApiBase}/simulate`,
      {
        scenario_name: scenarioName,
        checkpoint_path: checkpointPath,
        env_type: artifact?.env_type || "multimodal",
        algorithm,
        reward_mode: artifact?.reward_mode || null,
        episodes: 1,
        stochastic_eval: true,
        eval_seed: 13,
      },
      { timeout: 120000 }
  );

  return saveReplaySessionFromSimulation({
    scenarioName,
    algorithm,
    result: {
      ...simulation,
      source: "training",
    },
    sessionMeta: {
      source: "training",
      titlePrefix: "训练回放",
      artifactSignature,
    },
    persist: false,
  });
};

const replayOptions = computed(() => {
  const local = allSessions.value.map((session) => ({
    key: `test:${session.id}`,
    label: `[${session.source === "training" ? "训练" : "测试"}] ${session.title} · ${new Date(session.createdAt || Date.now()).toLocaleTimeString("zh-CN", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    })} (${session.frames?.length || 0} 帧)`,
  }));
  const trainingPending =
    latestTrainingArtifact.value &&
    !allSessions.value.some(
      (session) =>
        session.source === "training" &&
        session.artifactSignature === trainingArtifactSignature(latestTrainingArtifact.value)
    )
      ? [
          {
            key: "train:latest",
            label: `[训练] 最新训练产物 / ${latestTrainingArtifact.value.scenario_name || "unknown"} / ${(latestTrainingArtifact.value.algorithm || "ppo").toUpperCase()} (点击后生成回放)`,
          },
        ]
      : [];
  const ns3 = experiments.value.map((exp) => ({
    key: `ns3:${exp.id}`,
    label: `[ns-3] #${exp.id} ${exp.name || "演练"} (${exp.frames} 帧)`,
  }));
  return [...local, ...trainingPending, ...ns3];
});

const activeLocalSession = computed(() => {
  if (!selectedReplayKey.value?.startsWith("test:")) return null;
  const id = selectedReplayKey.value.slice("test:".length);
  return allSessions.value.find((item) => item.id === id) || null;
});

const activeNs3Experiment = computed(() => {
  if (!selectedReplayKey.value?.startsWith("ns3:")) return null;
  const id = Number(selectedReplayKey.value.slice("ns3:".length));
  return experiments.value.find((item) => Number(item.id) === id) || null;
});

const activeTrainingArtifact = computed(() =>
  selectedReplayKey.value === "train:latest" ? latestTrainingArtifact.value : null
);

const chartWidth = 280;
const chartHeight = 110;
const chartSeries = ref({ times: [], tps: [], losses: [] });

const equipmentCards = [
  {
    title: "背负式应急基站",
    badge: "自研",
    className: "equip-card--manpack",
    description: "700MHz 低频广覆盖，单兵携行快速开站，适合在灾后盲区内迅速恢复基础接入能力。",
  },
  {
    title: "高并发小型基站",
    badge: "自研",
    className: "equip-card--smallcell",
    description: "适合高密度人群区域，兼顾 Wi-Fi/5G 接入，优先保障视频回传与现场协同通信。",
  },
  {
    title: "多跳自组网中继",
    badge: "自研",
    className: "equip-card--relay",
    description: "适用于复杂地形与阻断场景，通过多跳转发和临时回传消除链路盲区。",
  },
  {
    title: "宏基站（公网）",
    badge: "",
    className: "",
    description: "传统公网基础设施，在灾害中可能部分失效，是应急通信接管与补盲的对照对象。",
  },
];

const normalizeLossPercent = (value) => {
  const numeric = Number(value || 0);
  return Math.abs(numeric) <= 1 ? numeric * 100 : numeric;
};

const setLocalChartSeries = (frames) => {
  const safeFrames = Array.isArray(frames) ? frames : [];
  chartSeries.value = {
    times: safeFrames.map((frame, index) => Number(frame.time ?? index)),
    tps: safeFrames.map((frame) => Number(frame.tp ?? 0)),
    losses: safeFrames.map((frame) => normalizeLossPercent(frame.loss ?? 0)),
  };
};

const setNs3ChartSeries = async (exp) => {
  if (!exp) {
    chartSeries.value = { times: [], tps: [], losses: [] };
    return;
  }
  const { data } = await axios.get(`${ns3ApiBase}/exp/${exp.id}/charts`, { timeout: 10000 });
  chartSeries.value = {
    times: Array.isArray(data?.times) ? data.times.map((value) => Number(value ?? 0)) : [],
    tps: Array.isArray(data?.tps) ? data.tps.map((value) => Number(value ?? 0)) : [],
    losses: Array.isArray(data?.losses) ? data.losses.map((value) => normalizeLossPercent(value)) : [],
  };
};

const visibleTimes = computed(() => chartSeries.value.times.slice(0, frameIndex.value + 1));
const visibleTps = computed(() => chartSeries.value.tps.slice(0, frameIndex.value + 1));
const visibleLosses = computed(() => chartSeries.value.losses.slice(0, frameIndex.value + 1));

const buildChartPoints = (series, maxValueHint = null) => {
  if (!series.length) return [];
  const maxValue = Math.max(
    maxValueHint ?? 0,
    ...series.map((value) => Number(value || 0)),
    1
  );
  const lastIndex = Math.max(1, series.length - 1);
  return series.map((value, index) => ({
    x: (index / lastIndex) * chartWidth,
    y: chartHeight - (Math.max(0, Number(value || 0)) / maxValue) * chartHeight,
  }));
};

const linePathFromPoints = (points) =>
  points
    .map((point, index) => `${index === 0 ? "M" : "L"} ${point.x} ${point.y}`)
    .join(" ");

const areaPathFromPoints = (points) => {
  if (!points.length) return "";
  const first = points[0];
  const last = points[points.length - 1];
  return `${linePathFromPoints(points)} L ${last.x} ${chartHeight} L ${first.x} ${chartHeight} Z`;
};

const tpChartPoints = computed(() => buildChartPoints(visibleTps.value));
const lossChartPoints = computed(() => buildChartPoints(visibleLosses.value, 100));
const tpLinePath = computed(() => linePathFromPoints(tpChartPoints.value));
const tpAreaPath = computed(() => areaPathFromPoints(tpChartPoints.value));
const lossLinePath = computed(() => linePathFromPoints(lossChartPoints.value));
const lossAreaPath = computed(() => areaPathFromPoints(lossChartPoints.value));

const sourceBadge = computed(() => {
  if (activeTrainingArtifact.value || activeSource.value === "training") return "训练";
  if (activeLocalSession.value?.source === "training") return "训练";
  if (activeLocalSession.value) return "测试";
  if (activeNs3Experiment.value) return "ns-3";
  return "未选择";
});

const sourceDescription = computed(() => {
  if (activeLocalSession.value) {
    return activeLocalSession.value.title;
  }
  if (activeTrainingArtifact.value) {
    return `最新训练产物：${activeTrainingArtifact.value.scenario_name || "unknown"} / ${(activeTrainingArtifact.value.algorithm || "ppo").toUpperCase()}`;
  }
  if (activeNs3Experiment.value) {
    return `实验 #${activeNs3Experiment.value.id} ${activeNs3Experiment.value.name || "演练"}`;
  }
  return "尚未选择回放源。";
});

const formattedCurrentTime = computed(() => `${Number(currentFrame.value?.time ?? 0).toFixed(1)} s`);
const formattedCurrentTp = computed(() => `${Number(currentFrame.value?.tp ?? 0).toFixed(2)} Mbps`);
const currentLossPercent = computed(() => normalizeLossPercent(currentFrame.value?.loss ?? 0));
const formattedCurrentLoss = computed(() => `${currentLossPercent.value.toFixed(2)} %`);
const formattedMaxTime = computed(() => `${Number(chartSeries.value.times.at(-1) ?? maxFrameIndex.value ?? 0).toFixed(0)} s`);
const formattedCoverage = computed(
  () => `覆盖 ${(Math.max(0, Math.min(1, Number(currentFrame.value?.coverageRatio ?? 0))) * 100).toFixed(1)}%`
);
const formattedBroadcast = computed(
  () => `广播 ${(Math.max(0, Math.min(1, Number(currentFrame.value?.broadcastRatio ?? 0))) * 100).toFixed(1)}%`
);
const formattedBudget = computed(() =>
  currentFrame.value?.remainingBudget != null ? Number(currentFrame.value.remainingBudget).toFixed(1) : "--"
);
const currentUserCount = computed(
  () => currentFrame.value?.userCount ?? parsedNodes.value.filter((node) => node.type === 0).length
);
const currentStationCount = computed(
  () => currentFrame.value?.stationCount ?? parsedNodes.value.filter((node) => node.type !== 0).length
);

const localReplayWarning = computed(() => {
  if (!activeLocalSession.value) return "";
  if (Number(activeLocalSession.value.schemaVersion || 0) >= 2) return "";
  return "该测试回放来自旧版本地会话结构，建议重新执行一次测试，以获得完整的累计基站、用户和连线展示。";
});

const frameSummary = computed(() => {
  const frame = currentFrame.value;
  if (!frame) return "";
  const action = frame.actionDesc;
  const metrics = [
    `用户 ${frame.userCount ?? parsedNodes.value.filter((node) => node.type === 0).length}`,
    `基站 ${frame.stationCount ?? parsedNodes.value.filter((node) => node.type !== 0).length}`,
    `在线 ${frame.connectedUsers ?? parsedNodes.value.filter((node) => node.type === 0 && node.online).length}`,
    `广播 ${frame.broadcastUsers ?? parsedNodes.value.filter((node) => node.type === 0 && node.broadcastServed).length}`,
  ];
  if (action?.location) {
    const regionText = action.region_label ? ` ${action.region_label}` : "";
    return `${frame.label || ""} | 部署站点 #${action.site_index} @(${action.location[0]}, ${action.location[1]})${regionText} | 通信 ${action.comm_mode || "-"} | 广播 ${action.broadcast_mode || "-"} | ${metrics.join(" | ")}`;
  }
  return [frame.label || "", metrics.join(" | ")].filter(Boolean).join(" | ");
});

const parsedNodes = computed(() => {
  const nodes = currentFrame.value?.nodes || [];
  return nodes
    .map((n) => {
      if (Array.isArray(n)) {
        return {
          id: Number(n[0] ?? 0),
          type: Number(n[1] ?? 0),
          x: Number(n[2] ?? 0),
          y: Number(n[3] ?? 0),
          rxBytes: Number(n[4] ?? 0),
          online: Number(n[5] ?? 1) === 1,
          broadcastServed: false,
          kind: null,
          coverageRadius: 0,
        };
      }
        return {
          id: Number(n.id ?? 0),
          type: Number(n.type ?? 0),
          x: Number(n.x ?? 0),
          y: Number(n.y ?? 0),
          rxBytes: Number(n.rxBytes ?? 0),
          online: Boolean(n.online ?? true),
          broadcastServed: Boolean(n.broadcastServed ?? false),
          kind: n.kind || null,
          coverageRadius: Number(n.coverageRadius ?? 0),
        };
      })
    .filter((n) => Number.isFinite(n.x) && Number.isFinite(n.y));
});

const parsedLinks = computed(() => {
  const links = currentFrame.value?.links || [];
  return links
    .map((l) => {
      if (Array.isArray(l)) {
        return { src: Number(l[0] ?? 0), dst: Number(l[1] ?? 0), protocol: Number(l[2] ?? 0) };
      }
      return {
        src: Number(l.srcId ?? l.src ?? 0),
        dst: Number(l.dstId ?? l.dst ?? 0),
        protocol: Number(l.protocol ?? 0),
      };
    })
    .filter((l) => Number.isFinite(l.src) && Number.isFinite(l.dst));
});

const colorByType = (type, online) => {
  if (!online) return "#64748b";
  if (type === 1) return "#38bdf8";
  if (type === 2) return "#f59e0b";
  if (type === 3) return "#10b981";
  if (type === 4) return "#22c55e";
  return "#38bdf8";
};

const nodeFill = (node) => {
  if (node.type === 0) {
    if (node.online) return "#34d399";
    if (node.broadcastServed) return "#facc15";
    return "#64748b";
  }
  if (node.kind === "deployed") return node.type === 1 ? "#60a5fa" : "#fb923c";
  return colorByType(node.type, node.online);
};

const nodeStroke = (node) => {
  if (node.type === 0) {
    if (node.online) return "rgba(240, 253, 250, 0.9)";
    if (node.broadcastServed) return "rgba(254, 249, 195, 0.92)";
    return "rgba(203, 213, 225, 0.8)";
  }
  if (node.kind === "deployed") return "rgba(255, 255, 255, 0.92)";
  return "rgba(224, 242, 254, 0.78)";
};

const nodeRadius = (node) => {
  if (node.type === 0) return node.online ? 3.4 : 2.8;
  if (node.type === 1) return node.kind === "deployed" ? 7.5 : 7;
  return node.kind === "deployed" ? 6.6 : 6.2;
};

const haloRadius = (node) => {
  if (node.type === 0) return 0;
  if (node.type === 1) return node.kind === "deployed" ? 18 : 16;
  return node.kind === "deployed" ? 14 : 12;
};

const localViewport = computed(() => {
  const session = activeLocalSession.value;
  if (!session) return null;
  return {
    minX: 0,
    maxX: Math.max(1, Number(session.mapWidth || currentFrame.value?.mapWidth || 5000)),
    minY: 0,
    maxY: Math.max(1, Number(session.mapHeight || currentFrame.value?.mapHeight || 5000)),
  };
});

const previousFrame = computed(() => {
  const session = activeLocalSession.value;
  if (!session) return null;
  const frames = session.frames || [];
  if (frameIndex.value <= 0 || frameIndex.value >= frames.length) return null;
  return frames[frameIndex.value - 1] || null;
});

const stopPlayback = () => {
  isPlaying.value = false;
  if (timer) {
    clearInterval(timer);
    timer = null;
  }
};

const drawFrame = () => {
  const canvas = canvasRef.value;
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const width = canvas.clientWidth || 980;
  const height = canvas.clientHeight || 560;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  ctx.fillStyle = "#081022";
  ctx.fillRect(0, 0, width, height);

  const nodes = parsedNodes.value;
  if (!nodes.length) {
    ctx.fillStyle = "#94a3b8";
    ctx.font = "15px sans-serif";
    ctx.fillText("暂无帧数据，请先加载实验", 24, 36);
    return;
  }

  const viewport = localViewport.value || {
    minX: Math.min(...nodes.map((n) => n.x)),
    maxX: Math.max(...nodes.map((n) => n.x)),
    minY: Math.min(...nodes.map((n) => n.y)),
    maxY: Math.max(...nodes.map((n) => n.y)),
  };
  const minX = viewport.minX;
  const maxX = viewport.maxX;
  const minY = viewport.minY;
  const maxY = viewport.maxY;
  const spanX = Math.max(1, maxX - minX);
  const spanY = Math.max(1, maxY - minY);
  const pad = 26;

  const project = (node) => {
    const x = pad + ((node.x - minX) / spanX) * (width - pad * 2);
    const y = pad + ((node.y - minY) / spanY) * (height - pad * 2);
    return { x, y };
  };

  ctx.strokeStyle = "rgba(56, 189, 248, 0.14)";
  ctx.lineWidth = 1;
  for (let index = 1; index < 6; index += 1) {
    const ratio = index / 6;
    const x = pad + ratio * (width - pad * 2);
    const y = pad + ratio * (height - pad * 2);
    ctx.beginPath();
    ctx.moveTo(x, pad);
    ctx.lineTo(x, height - pad);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(pad, y);
    ctx.lineTo(width - pad, y);
    ctx.stroke();
  }

  const nodeMap = new Map(nodes.map((n) => [n.id, { ...n, ...project(n) }]));
  const previousNodeMap = new Map(
    ((previousFrame.value?.nodes || []) || [])
      .map((node) => ({
        id: Number(node.id ?? 0),
        x: Number(node.x ?? 0),
        y: Number(node.y ?? 0),
      }))
      .filter((node) => Number.isFinite(node.id) && Number.isFinite(node.x) && Number.isFinite(node.y))
      .map((node) => [node.id, project(node)])
  );

  for (const link of parsedLinks.value) {
    const src = nodeMap.get(link.src);
    const dst = nodeMap.get(link.dst);
    if (!src || !dst) continue;
    ctx.beginPath();
    ctx.moveTo(src.x, src.y);
    ctx.lineTo(dst.x, dst.y);
    ctx.strokeStyle = link.protocol === 0 ? "rgba(52, 211, 153, 0.5)" : "rgba(34, 211, 238, 0.48)";
    ctx.lineWidth = link.protocol === 0 ? 1.5 : 1.8;
    ctx.stroke();
  }

  for (const node of nodeMap.values()) {
    const previous = previousNodeMap.get(node.id);
    if (!previous) continue;
    const moved = Math.abs(previous.x - node.x) > 0.5 || Math.abs(previous.y - node.y) > 0.5;
    if (!moved) continue;
    ctx.beginPath();
    ctx.moveTo(previous.x, previous.y);
    ctx.lineTo(node.x, node.y);
    ctx.strokeStyle = "rgba(250, 204, 21, 0.32)";
    ctx.lineWidth = 1.2;
    ctx.stroke();
  }

  for (const node of nodeMap.values()) {
    const halo = haloRadius(node);
    if (halo > 0) {
      ctx.beginPath();
      ctx.arc(node.x, node.y, halo, 0, Math.PI * 2);
      ctx.fillStyle =
        node.kind === "deployed"
          ? "rgba(96, 165, 250, 0.08)"
          : node.type === 1
            ? "rgba(56, 189, 248, 0.06)"
            : "rgba(251, 146, 60, 0.08)";
      ctx.fill();
    }
    ctx.beginPath();
    ctx.arc(node.x, node.y, nodeRadius(node), 0, Math.PI * 2);
    ctx.fillStyle = nodeFill(node);
    ctx.fill();
    ctx.lineWidth = node.type === 0 ? 0.8 : 1.5;
    ctx.strokeStyle = nodeStroke(node);
    ctx.stroke();
  }

  const latestDeploymentId = Number(currentFrame.value?.latestDeploymentId ?? -1);
  if (latestDeploymentId >= 0) {
    const deployedNode = nodeMap.get(latestDeploymentId);
    if (deployedNode) {
      ctx.beginPath();
      ctx.arc(deployedNode.x, deployedNode.y, nodeRadius(deployedNode) + 8, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(250, 204, 21, 0.9)";
      ctx.lineWidth = 1.8;
      ctx.stroke();
    }
  }

  if (currentFrame.value?.disaster === 1) {
    ctx.fillStyle = "rgba(239,68,68,0.86)";
    ctx.font = "bold 14px sans-serif";
    ctx.fillText("灾后阶段", width - 90, 24);
  }

  const summaryLine = [
    `用户 ${currentFrame.value?.userCount ?? nodes.filter((node) => node.type === 0).length}`,
    `基站 ${currentFrame.value?.stationCount ?? nodes.filter((node) => node.type !== 0).length}`,
    `链路 ${parsedLinks.value.length}`,
  ].join("   ");
  ctx.fillStyle = "rgba(226, 232, 240, 0.9)";
  ctx.font = "13px sans-serif";
  ctx.fillText(summaryLine, 24, height - 14);
};

const loadReplaySources = async () => {
  loadingExperiments.value = true;
  localSessions.value = listReplaySessions();
  const previousSelection = selectedReplayKey.value;
  const preferredLocal = getActiveReplaySessionId();
  try {
    await fetchLatestTrainingArtifact();
    localSessions.value = listReplaySessions();
    const { data } = await axios.get(`${ns3ApiBase}/experiments`, { timeout: 10000 });
    experiments.value = Array.isArray(data) ? data : [];
  } catch (error) {
    experiments.value = [];
    if (!localSessions.value.length) {
      errorMessage.value = `无法连接 ns-3 回放后端: ${error?.message || "未知错误"}`;
    }
  } finally {
    if (previousSelection && replayOptions.value.some((item) => item.key === previousSelection)) {
      selectedReplayKey.value = previousSelection;
    } else if (preferredLocal && replayOptions.value.some((item) => item.key === `test:${preferredLocal}`)) {
      selectedReplayKey.value = `test:${preferredLocal}`;
    } else {
      selectedReplayKey.value = replayOptions.value[0]?.key || null;
    }
    if (!selectedReplayKey.value) {
      maxFrameIndex.value = -1;
      currentFrame.value = null;
    } else {
      await onReplayChange();
    }
    loadingExperiments.value = false;
  }
};

const loadFrame = async (index) => {
  if (!selectedReplayKey.value) return;
  if (activeLocalSession.value) {
    const frames = activeLocalSession.value.frames || [];
    const frame = frames[index];
    if (!frame) return;
    currentFrame.value = frame;
    await nextTick();
    drawFrame();
    return;
  }
  try {
    const exp = activeNs3Experiment.value;
    if (!exp) return;
    const { data } = await axios.get(`${ns3ApiBase}/exp/${exp.id}/frame/${index}`, { timeout: 10000 });
    if (data?.error) {
      throw new Error(data.error);
    }
    currentFrame.value = {
      time: Number(data.time ?? 0),
      tp: Number(data.tp ?? 0),
      loss: Number(data.loss ?? 0),
      disaster: Number(data.disaster ?? 0),
      nodes: data.nodes || [],
      links: data.links || [],
    };
    await nextTick();
    drawFrame();
  } catch (error) {
    stopPlayback();
    errorMessage.value = `加载帧失败: ${error?.message || "未知错误"}`;
  }
};

const onReplayChange = async () => {
  stopPlayback();
  const local = activeLocalSession.value;
  if (local) {
    activeSource.value = local.source || "test";
    setActiveReplaySessionId(local.id);
    maxFrameIndex.value = Math.max(0, Number(local.frames?.length || 1) - 1);
    frameIndex.value = 0;
    errorMessage.value = "";
    setLocalChartSeries(local.frames || []);
    await loadFrame(0);
    return;
  }
  if (activeTrainingArtifact.value) {
    activeSource.value = "training";
    errorMessage.value = "";
    try {
      const session = await materializeTrainingReplay(activeTrainingArtifact.value);
      if (session) {
        transientSessions.value = [
          session,
          ...transientSessions.value.filter((item) => item.artifactSignature !== session.artifactSignature),
        ].slice(0, 3);
      }
      localSessions.value = listReplaySessions();
      if (session?.id) {
        selectedReplayKey.value = `test:${session.id}`;
        await onReplayChange();
        return;
      }
      errorMessage.value = "训练产物已发现，但未能生成训练回放。";
    } catch (error) {
      errorMessage.value = `训练回放生成失败: ${error?.message || "未知错误"}`;
    }
    maxFrameIndex.value = -1;
    currentFrame.value = null;
    return;
  }
  const exp = activeNs3Experiment.value;
  if (!exp) {
    activeSource.value = "none";
    return;
  }
  activeSource.value = "ns3";
  errorMessage.value = "";
  frameIndex.value = 0;
  maxFrameIndex.value = Math.max(0, Number(exp.frames || 1) - 1);
  await setNs3ChartSeries(exp);
  await loadFrame(0);
};

const step = async (delta) => {
  if (maxFrameIndex.value < 0) return;
  const target = Math.min(maxFrameIndex.value, Math.max(0, frameIndex.value + delta));
  frameIndex.value = target;
  await loadFrame(target);
};

const togglePlayback = () => {
  if (isPlaying.value) {
    stopPlayback();
    return;
  }
  if (maxFrameIndex.value <= 0) return;
  isPlaying.value = true;
  timer = setInterval(async () => {
    if (frameIndex.value >= maxFrameIndex.value) {
      stopPlayback();
      return;
    }
    frameIndex.value += 1;
    await loadFrame(frameIndex.value);
  }, 450);
};

onMounted(async () => {
  await loadReplaySources();
  window.addEventListener("resize", drawFrame);
  window.addEventListener("storage", loadReplaySources);
});

watch(
  () => currentFrame.value,
  () => {
    drawFrame();
  }
);

onBeforeUnmount(() => {
  stopPlayback();
  window.removeEventListener("resize", drawFrame);
  window.removeEventListener("storage", loadReplaySources);
});
</script>

<style scoped>
.replay-panel {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.replay-toolbar {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  align-items: end;
}

.replay-toolbar label {
  display: flex;
  flex-direction: column;
  gap: 6px;
  min-width: 260px;
}

.replay-toolbar select {
  border-radius: 10px;
  border: 1px solid rgba(148, 163, 184, 0.35);
  background: rgba(15, 23, 42, 0.7);
  color: #e2e8f0;
  padding: 8px 10px;
}

.replay-toolbar button,
.replay-controls button {
  border: 0;
  border-radius: 10px;
  background: linear-gradient(120deg, #0ea5e9, #0284c7);
  color: #f8fafc;
  font-weight: 600;
  padding: 9px 14px;
  cursor: pointer;
}

.native-link {
  color: #38bdf8;
  text-decoration: none;
  font-size: 0.9rem;
  padding-bottom: 4px;
}

.error-banner {
  background: rgba(127, 29, 29, 0.35);
  border: 1px solid rgba(248, 113, 113, 0.45);
  color: #fecaca;
  border-radius: 10px;
  padding: 10px 12px;
}

.warning-banner {
  background: rgba(120, 53, 15, 0.28);
  border: 1px solid rgba(251, 191, 36, 0.35);
  color: #fde68a;
  border-radius: 10px;
  padding: 10px 12px;
}

.native-shell {
  display: grid;
  grid-template-columns: 300px minmax(0, 1fr) 280px;
  gap: 14px;
  min-height: min(72vh, 860px);
}

.native-panel {
  border-radius: 18px;
  border: 1px solid rgba(56, 189, 248, 0.14);
  background: linear-gradient(180deg, rgba(15, 23, 42, 0.88), rgba(9, 14, 25, 0.96));
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.panel-title {
  font-size: 0.82rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #94a3b8;
  padding-bottom: 8px;
  border-bottom: 1px solid rgba(148, 163, 184, 0.14);
}

.stat-card,
.status-card {
  background: rgba(26, 35, 56, 0.84);
  border-radius: 14px;
  padding: 14px;
  border-left: 3px solid #38bdf8;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
}

.status-card {
  border-left-color: #22c55e;
}

.status-card--danger {
  border-left-color: #ef4444;
}

.stat-label,
.chart-title {
  font-size: 0.78rem;
  color: #94a3b8;
  margin-bottom: 6px;
}

.stat-value {
  font-size: 1.9rem;
  font-weight: 700;
  color: #e2e8f0;
}

.stat-value--accent {
  color: #38bdf8;
}

.stat-value--success {
  color: #34d399;
}

.stat-value--danger {
  color: #f87171;
}

.status-value {
  font-size: 1rem;
  font-weight: 700;
  color: #e2e8f0;
}

.status-hint {
  margin: 8px 0 0;
  color: #94a3b8;
  font-size: 0.88rem;
  line-height: 1.5;
}

.chart-box {
  background: rgba(26, 35, 56, 0.84);
  border-radius: 14px;
  padding: 12px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.chart-svg {
  width: 100%;
  height: 120px;
  display: block;
}

.map-area {
  position: relative;
  border-radius: 20px;
  overflow: hidden;
  border: 1px solid rgba(56, 189, 248, 0.18);
  background: linear-gradient(180deg, rgba(7, 12, 24, 0.98), rgba(4, 8, 16, 1));
  display: flex;
  flex-direction: column;
  min-height: 0;
}

.map-stage-header {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: flex-start;
  padding: 16px 18px 10px;
}

.map-title {
  margin: 0;
  font-size: 0.95rem;
  font-weight: 700;
  color: #e2e8f0;
}

.map-subtitle {
  margin: 6px 0 0;
  color: #94a3b8;
  font-size: 0.86rem;
  line-height: 1.45;
}

.map-badges {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 8px;
}

.map-badge,
.badge {
  display: inline-flex;
  align-items: center;
  padding: 4px 10px;
  border-radius: 999px;
  background: rgba(56, 189, 248, 0.14);
  border: 1px solid rgba(56, 189, 248, 0.2);
  color: #dbeafe;
  font-size: 0.75rem;
  font-weight: 700;
}

.alert-banner {
  position: absolute;
  top: 16px;
  left: 50%;
  transform: translateX(-50%) translateY(-12px);
  opacity: 0;
  pointer-events: none;
  z-index: 2;
  background: linear-gradient(135deg, rgba(239, 68, 68, 0.96), rgba(220, 38, 38, 0.96));
  color: #fff;
  padding: 10px 22px;
  border-radius: 999px;
  font-weight: 700;
  box-shadow: 0 10px 30px rgba(239, 68, 68, 0.35);
  transition: opacity 0.3s ease, transform 0.3s ease;
}

.alert-banner--visible {
  opacity: 1;
  transform: translateX(-50%) translateY(0);
}

.replay-stage {
  position: relative;
  flex: 1;
  min-height: 420px;
  padding: 0 0 8px;
}

.replay-stage canvas {
  display: block;
  width: 100%;
  height: 100%;
  min-height: 420px;
}

.legend {
  position: absolute;
  left: 18px;
  bottom: 18px;
  display: flex;
  flex-direction: column;
  gap: 6px;
  background: rgba(19, 26, 43, 0.88);
  border: 1px solid rgba(148, 163, 184, 0.18);
  border-radius: 12px;
  padding: 12px;
  color: #dbeafe;
  font-size: 0.76rem;
  backdrop-filter: blur(10px);
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 8px;
}

.legend-dot,
.legend-line {
  display: inline-block;
}

.legend-dot {
  width: 10px;
  height: 10px;
  border-radius: 999px;
}

.legend-dot--user {
  background: #34d399;
}

.legend-dot--macro {
  background: #38bdf8;
  border-radius: 2px;
}

.legend-dot--manpack {
  width: 0;
  height: 0;
  border-left: 6px solid transparent;
  border-right: 6px solid transparent;
  border-bottom: 12px solid #f59e0b;
}

.legend-line {
  width: 18px;
  height: 2px;
}

.legend-line--wifi {
  background: #34d399;
}

.legend-line--lte {
  background: #22d3ee;
}

.equip-card {
  background: rgba(26, 35, 56, 0.84);
  border-radius: 14px;
  padding: 14px;
  border-left: 4px solid #38bdf8;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.equip-card--manpack {
  border-left-color: #f59e0b;
}

.equip-card--smallcell {
  border-left-color: #8b5cf6;
}

.equip-card--relay {
  border-left-color: #f97316;
}

.equip-name {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  font-size: 0.95rem;
  font-weight: 700;
  color: #e2e8f0;
}

.equip-desc {
  color: #94a3b8;
  line-height: 1.6;
  font-size: 0.84rem;
}

.summary-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
}

.summary-item {
  border-radius: 12px;
  background: rgba(8, 16, 34, 0.85);
  padding: 10px;
  border: 1px solid rgba(148, 163, 184, 0.12);
}

.summary-item span {
  display: block;
  color: #94a3b8;
  font-size: 0.72rem;
}

.summary-item strong {
  display: block;
  margin-top: 4px;
  color: #e2e8f0;
  font-size: 1rem;
}

.timeline {
  display: flex;
  align-items: center;
  gap: 12px;
  border-radius: 16px;
  background: rgba(15, 23, 42, 0.84);
  border: 1px solid rgba(56, 189, 248, 0.14);
  padding: 12px 16px;
}

.timeline-btn {
  border: 0;
  border-radius: 10px;
  background: rgba(30, 41, 59, 0.92);
  color: #e2e8f0;
  padding: 8px 12px;
  font-weight: 600;
  cursor: pointer;
}

.timeline-btn--primary {
  background: linear-gradient(120deg, #0ea5e9, #0284c7);
  color: #f8fafc;
}

.timeline-btn:disabled {
  opacity: 0.45;
  cursor: not-allowed;
}

.timeline-time {
  color: #cbd5e1;
  font-weight: 600;
  min-width: 56px;
  text-align: center;
}

.timeline-slider {
  flex: 1;
  accent-color: #38bdf8;
}

@media (max-width: 900px) {
  .native-shell {
    grid-template-columns: 1fr;
  }

  .replay-stage,
  .replay-stage canvas {
    min-height: 360px;
  }

  .map-stage-header,
  .timeline {
    flex-direction: column;
    align-items: stretch;
  }

  .map-badges {
    justify-content: flex-start;
  }

  .timeline-time {
    min-width: 0;
    text-align: left;
  }
}
</style>
