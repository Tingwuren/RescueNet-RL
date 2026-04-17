<template>
  <div class="monitor">
  <div class="monitor__status">
    <div>
      <p class="monitor__label">当前状态</p>
      <p class="monitor__value">{{ status }}</p>
    </div>
    <div>
      <p class="monitor__label">最近更新</p>
      <p class="monitor__value">{{ latestUpdate }}</p>
    </div>
  </div>
    <div class="monitor__charts">
    <div class="monitor__chart">
      <div class="chart__header">
        <div>
          <p class="monitor__label">实时覆盖率（episode）</p>
          <p class="monitor__value">{{ latestAccuracy }}</p>
        </div>
        <small class="chart__hint">来源：每个 episode 终态覆盖率</small>
      </div>
      <div class="chart__viewport">
        <svg :viewBox="`0 0 ${chartWidth} ${chartHeight}`" preserveAspectRatio="none">
          <defs>
            <linearGradient id="areaGradient" x1="0" x2="0" y1="0" y2="1">
              <stop offset="0%" stop-color="#38bdf8" stop-opacity="0.4" />
              <stop offset="100%" stop-color="#38bdf8" stop-opacity="0.05" />
            </linearGradient>
          </defs>
          <path
            v-if="coverageAreaPath"
            :d="coverageAreaPath"
            fill="url(#areaGradient)"
            stroke="none"
            opacity="0.8"
          />
          <path
            v-if="coverageLinePath"
            :d="coverageLinePath"
            fill="none"
            stroke="#38bdf8"
            stroke-width="2"
          />
        </svg>
        <span
          v-for="(point, idx) in coveragePoints"
          :key="`cov-${idx}`"
          class="chart-point chart-point--coverage"
          :style="pointStyle(point)"
        />
        <p v-if="!coveragePoints.length" class="monitor__placeholder">暂无覆盖率数据，等待训练事件...</p>
      </div>
    </div>
    <div class="monitor__chart">
      <div class="chart__header">
        <div>
          <p class="monitor__label">实时广播覆盖（episode）</p>
          <p class="monitor__value">{{ latestBroadcast }}</p>
        </div>
        <small class="chart__hint">来源：每个 episode 终态广播率</small>
      </div>
      <div class="chart__viewport">
        <svg :viewBox="`0 0 ${chartWidth} ${chartHeight}`" preserveAspectRatio="none">
          <defs>
            <linearGradient id="broadcastGradient" x1="0" x2="0" y1="0" y2="1">
              <stop offset="0%" stop-color="#a855f7" stop-opacity="0.4" />
              <stop offset="100%" stop-color="#a855f7" stop-opacity="0.05" />
            </linearGradient>
          </defs>
          <path
            v-if="broadcastAreaPath"
            :d="broadcastAreaPath"
            fill="url(#broadcastGradient)"
            stroke="none"
            opacity="0.8"
          />
          <path
            v-if="broadcastLinePath"
            :d="broadcastLinePath"
            fill="none"
            stroke="#a855f7"
            stroke-width="2"
          />
        </svg>
        <span
          v-for="(point, idx) in broadcastPoints"
          :key="`bc-${idx}`"
          class="chart-point chart-point--broadcast"
          :style="pointStyle(point)"
        />
        <p v-if="!broadcastPoints.length" class="monitor__placeholder">暂无广播率数据，等待训练事件...</p>
      </div>
    </div>
    </div>
  <div class="monitor__events">
    <div class="console__header">
      <p class="monitor__label">训练控制台</p>
      <span>{{ consoleLines.length }} lines</span>
    </div>
    <div class="monitor__event-list console">
      <pre v-for="(line, idx) in consoleLines" :key="idx" class="console__line">{{ line }}</pre>
      <p v-if="!consoleLines.length" class="monitor__placeholder console__placeholder">暂无控制台输出，请启动训练。</p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from "vue";

const props = defineProps({
  events: {
    type: Array,
    default: () => [],
  },
  status: {
    type: String,
    default: "Idle",
  },
});

const latestUpdate = computed(() => {
  if (!props.events.length) {
    return "--";
  }
  const timestamp = props.events[props.events.length - 1].timestamp;
  return new Date(timestamp * 1000).toLocaleTimeString();
});

const chartWidth = 320;
const chartHeight = 140;

const buildEpisodeSeries = (keyEpisode) => {
  const series = [];
  for (const event of props.events) {
    const payload = event.payload || {};
    if (event.type === "episode" && typeof payload[keyEpisode] === "number") {
      series.push({ value: payload[keyEpisode], label: `Episode ${payload.episode}` });
    }
  }
  return series;
};

const coverageSeries = computed(() => buildEpisodeSeries("coverage"));
const broadcastSeries = computed(() => buildEpisodeSeries("broadcast"));

const latestAccuracy = computed(() => {
  if (!coverageSeries.value.length) return "--";
  const latest = coverageSeries.value[coverageSeries.value.length - 1].value;
  return `${(Math.max(0, Math.min(1, latest)) * 100).toFixed(2)}%`;
});

const latestBroadcast = computed(() => {
  if (!broadcastSeries.value.length) return "--";
  const latest = broadcastSeries.value[broadcastSeries.value.length - 1].value;
  return `${(Math.max(0, Math.min(1, latest)) * 100).toFixed(2)}%`;
});

const normalizePoints = (series) => {
  if (!series.length) return [];
  const maxIdx = Math.max(1, series.length - 1);
  return series.map((pt, idx) => {
    const x = (idx / maxIdx) * chartWidth;
    const clamped = Math.max(0, Math.min(1, pt.value));
    const y = chartHeight - clamped * chartHeight;
    return { x, y };
  });
};

const coveragePoints = computed(() => normalizePoints(coverageSeries.value));
const broadcastPoints = computed(() => normalizePoints(broadcastSeries.value));

const linePathFrom = (points) => {
  if (!points.length) return "";
  return points.reduce((path, point, idx) => {
    const cmd = idx === 0 ? "M" : "L";
    return `${path} ${cmd} ${point.x} ${point.y}`;
  }, "").trim();
};

const areaPathFrom = (points) => {
  if (!points.length) return "";
  const first = points[0];
  const last = points[points.length - 1];
  const line = linePathFrom(points);
  return `${line} L ${last.x} ${chartHeight} L ${first.x} ${chartHeight} Z`;
};

const coverageLinePath = computed(() => linePathFrom(coveragePoints.value));
const coverageAreaPath = computed(() => areaPathFrom(coveragePoints.value));
const broadcastLinePath = computed(() => linePathFrom(broadcastPoints.value));
const broadcastAreaPath = computed(() => areaPathFrom(broadcastPoints.value));

const pointStyle = (point) => ({
  left: `${(point.x / chartWidth) * 100}%`,
  top: `${(point.y / chartHeight) * 100}%`,
});

const percent = (value) => `${(Math.max(0, Math.min(1, Number(value || 0))) * 100).toFixed(2)}%`;
const fixed = (value, digits = 3) => Number(value || 0).toFixed(digits);
const timeText = (timestamp) => {
  if (!timestamp) return "--:--:--";
  return new Date(timestamp * 1000).toLocaleTimeString("zh-CN", { hour12: false });
};

const formatConsoleLine = (event) => {
  const payload = event?.payload || {};
  const prefix = `[${timeText(event?.timestamp)}]`;
  if (event?.message) return `${prefix} ${event.message}`;
  if (event?.type === "experiment_config") {
    return `${prefix} loaded config lr=${payload.learningRate ?? "-"} gamma=${payload.discountFactor ?? "-"} batch=${payload.batchSize ?? "-"} rollout=${payload.rolloutSteps ?? "-"}`;
  }
  if (event?.type === "status") {
    const step = payload.step != null ? ` step=${payload.step}` : "";
    return `${prefix} status=${payload.state || "unknown"}${step}`;
  }
  if (event?.type === "episode") {
    const parts = [
      `episode=${payload.episode ?? "-"}`,
      `reward=${fixed(payload.reward)}`,
      `coverage=${percent(payload.coverage)}`,
      `broadcast=${percent(payload.broadcast)}`,
    ];
    if (payload.steps != null) parts.push(`steps=${payload.steps}`);
    if (payload.total_timesteps != null) parts.push(`timesteps=${payload.total_timesteps}`);
    return `${prefix} ${parts.join(" | ")}`;
  }
  if (event?.type === "train") {
    return `${prefix} train step=${payload.step ?? "-"} reward=${fixed(payload.reward)} loss=${fixed(payload.loss)}`;
  }
  if (event?.type === "error" || event?.type === "training_replay_error") {
    return `${prefix} error: ${payload.message || "unknown error"}`;
  }
  if (event?.type === "training_replay_ready") {
    return `${prefix} replay: ${event.message || "训练回放已生成"}`;
  }
  return `${prefix} ${event?.type || "event"} ${JSON.stringify(payload)}`;
};

const consoleLines = computed(() => props.events.map(formatConsoleLine).slice(-80));
</script>

<style scoped>
.monitor {
  border: 1px solid rgba(100, 116, 139, 0.22);
  border-radius: 16px;
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 16px;
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.94), rgba(248, 250, 252, 0.9));
  color: #0f172a;
  box-shadow: 0 14px 28px rgba(15, 23, 42, 0.06);
}

.monitor__status {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}

.monitor__label {
  margin: 0;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #64748b;
}

.monitor__value {
  margin: 2px 0 0;
  font-size: 18px;
  font-weight: 600;
  color: #0f172a;
}

.monitor__status > div {
  padding: 14px;
  border-radius: 14px;
  border: 1px solid rgba(14, 165, 233, 0.16);
  background: rgba(224, 242, 254, 0.74);
}

.monitor__events {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.console__header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.console__header span {
  font-size: 12px;
  color: #64748b;
}

.monitor__charts {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px;
}

.monitor__chart {
  border: 1px solid rgba(100, 116, 139, 0.2);
  border-radius: 12px;
  padding: 12px;
  background: rgba(255, 255, 255, 0.86);
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.chart__header {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
}

.chart__hint {
  color: #64748b;
}

.chart__viewport {
  position: relative;
  width: 100%;
  min-height: 160px;
}

.chart__viewport svg {
  width: 100%;
  height: 160px;
  display: block;
}

.chart-point {
  position: absolute;
  width: 10px;
  height: 10px;
  border-radius: 999px;
  transform: translate(-50%, -50%);
  border: 2px solid rgba(255, 255, 255, 0.95);
  stroke: rgba(255, 255, 255, 0.95);
  stroke-width: 1.4;
  box-shadow: 0 3px 8px rgba(15, 23, 42, 0.18);
  pointer-events: none;
}

.chart-point--coverage {
  background: #22d3ee;
}

.chart-point--broadcast {
  background: #d946ef;
}

.monitor__event-list {
  max-height: 280px;
  overflow-y: auto;
}

.console {
  min-height: 220px;
  padding: 14px;
  border-radius: 14px;
  border: 1px solid rgba(15, 23, 42, 0.18);
  background:
    linear-gradient(180deg, rgba(15, 23, 42, 0.96), rgba(2, 6, 23, 0.96));
  box-shadow: inset 0 1px 0 rgba(148, 163, 184, 0.14);
}

.console__line {
  margin: 0;
  padding: 3px 0;
  color: #dbeafe;
  font-size: 12px;
  line-height: 1.6;
  white-space: pre-wrap;
  word-break: break-word;
}

.console__line::before {
  content: ">";
  margin-right: 8px;
  color: #38bdf8;
}

.console__placeholder {
  color: #94a3b8;
}

.monitor__placeholder {
  margin: 0;
  color: #64748b;
}

@media (max-width: 720px) {
  .monitor__status {
    grid-template-columns: 1fr;
  }

  .monitor__charts {
    grid-template-columns: 1fr;
  }
}
</style>
