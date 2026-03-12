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
          <g v-for="(point, idx) in coveragePoints" :key="`cov-${idx}`">
            <circle :cx="point.x" :cy="point.y" r="2.5" fill="#22d3ee" />
          </g>
        </svg>
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
          <g v-for="(point, idx) in broadcastPoints" :key="`bc-${idx}`">
            <circle :cx="point.x" :cy="point.y" r="2.5" fill="#d946ef" />
          </g>
        </svg>
        <p v-if="!broadcastPoints.length" class="monitor__placeholder">暂无广播率数据，等待训练事件...</p>
      </div>
    </div>
  <div class="monitor__events">
    <p class="monitor__label">实时事件</p>
    <div class="monitor__event-list">
      <div v-for="(event, idx) in events" :key="idx" class="event">
        <p class="event__type">{{ event.type }}</p>
          <pre>{{ formatPayload(event.payload) }}</pre>
        </div>
        <p v-if="!events.length" class="monitor__placeholder">暂无事件，请启动训练。</p>
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

const formatPayload = (payload) => {
  if (!payload) {
    return "";
  }
  try {
    return JSON.stringify(payload, null, 2);
  } catch (err) {
    return String(payload);
  }
};
</script>

<style scoped>
.monitor {
  border: 1px solid rgba(148, 163, 184, 0.3);
  border-radius: 12px;
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 16px;
  background: rgba(15, 23, 42, 0.4);
}

.monitor__status {
  display: flex;
  justify-content: space-between;
  border-bottom: 1px solid rgba(148, 163, 184, 0.2);
  padding-bottom: 12px;
}

.monitor__label {
  margin: 0;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #94a3b8;
}

.monitor__value {
  margin: 2px 0 0;
  font-size: 18px;
  font-weight: 600;
}

.monitor__events {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.monitor__chart {
  border: 1px solid rgba(148, 163, 184, 0.2);
  border-radius: 12px;
  padding: 12px;
  background: rgba(15, 23, 42, 0.35);
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
  color: #94a3b8;
}

.chart__viewport {
  position: relative;
  width: 100%;
  min-height: 160px;
}

.chart__viewport svg {
  width: 100%;
  height: 160px;
}

.monitor__event-list {
  max-height: 200px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.monitor__placeholder {
  margin: 0;
  color: #64748b;
}

.event {
  border: 1px solid rgba(148, 163, 184, 0.2);
  border-radius: 8px;
  padding: 8px;
  background: rgba(30, 64, 175, 0.15);
}

.event__type {
  margin: 0 0 4px;
  font-weight: 600;
  color: #93c5fd;
}

pre {
  margin: 0;
  font-size: 12px;
  color: #e2e8f0;
  white-space: pre-wrap;
}
</style>
