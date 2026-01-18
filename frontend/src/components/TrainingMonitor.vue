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
          <p class="monitor__label">实时准确率/覆盖率</p>
          <p class="monitor__value">{{ latestAccuracy }}</p>
        </div>
        <small class="chart__hint">来源：episode 覆盖率 + evaluation 覆盖率</small>
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
            v-if="areaPath"
            :d="areaPath"
            fill="url(#areaGradient)"
            stroke="none"
            opacity="0.8"
          />
          <path
            v-if="linePath"
            :d="linePath"
            fill="none"
            stroke="#38bdf8"
            stroke-width="2"
          />
          <g v-for="(point, idx) in normalizedPoints" :key="idx">
            <circle :cx="point.x" :cy="point.y" r="2.5" fill="#22d3ee" />
          </g>
        </svg>
        <p v-if="!normalizedPoints.length" class="monitor__placeholder">暂无覆盖率数据，等待训练事件...</p>
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

const accuracySeries = computed(() => {
  const series = [];
  for (const event of props.events) {
    const payload = event.payload || {};
    if (event.type === "episode" && typeof payload.coverage === "number") {
      series.push({ value: payload.coverage, label: `Episode ${payload.episode}` });
    }
    if (event.type === "evaluation" && typeof payload.avg_coverage === "number") {
      series.push({ value: payload.avg_coverage, label: `Eval@${payload.step}` });
    }
  }
  return series;
});

const latestAccuracy = computed(() => {
  if (!accuracySeries.value.length) return "--";
  const latest = accuracySeries.value[accuracySeries.value.length - 1].value;
  return `${(Math.max(0, Math.min(1, latest)) * 100).toFixed(2)}%`;
});

const chartWidth = 320;
const chartHeight = 140;

const normalizedPoints = computed(() => {
  const points = accuracySeries.value;
  if (!points.length) return [];
  const maxIdx = Math.max(1, points.length - 1);
  return points.map((pt, idx) => {
    const x = (idx / maxIdx) * chartWidth;
    const clamped = Math.max(0, Math.min(1, pt.value));
    const y = chartHeight - clamped * chartHeight;
    return { x, y };
  });
});

const linePath = computed(() => {
  if (!normalizedPoints.value.length) return "";
  return normalizedPoints.value.reduce((path, point, idx) => {
    const cmd = idx === 0 ? "M" : "L";
    return `${path} ${cmd} ${point.x} ${point.y}`;
  }, "").trim();
});

const areaPath = computed(() => {
  if (!normalizedPoints.value.length) return "";
  const first = normalizedPoints.value[0];
  const last = normalizedPoints.value[normalizedPoints.value.length - 1];
  const line = normalizedPoints.value.reduce((path, point, idx) => {
    const cmd = idx === 0 ? "M" : "L";
    return `${path} ${cmd} ${point.x} ${point.y}`;
  }, "");
  return `${line} L ${last.x} ${chartHeight} L ${first.x} ${chartHeight} Z`;
});

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
