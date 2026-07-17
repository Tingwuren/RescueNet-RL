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
              <stop offset="0%" stop-color="#1890ff" stop-opacity="0.4" />
              <stop offset="100%" stop-color="#1890ff" stop-opacity="0.05" />
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
            stroke="#1890ff"
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
              <stop offset="0%" stop-color="#409eff" stop-opacity="0.4" />
              <stop offset="100%" stop-color="#409eff" stop-opacity="0.05" />
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
            stroke="#409eff"
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
    <StreamingTerminal
      v-if="showTerminal"
      title="实时终端输出"
      subtitle="实时输出训练配置、后端 SSE 状态、episode/update 指标和回放生成结果。"
      :lines="consoleLines"
      :status="status"
      placeholder="暂无终端输出，请启动训练。"
    />
  </div>
</template>

<script setup>
import { computed } from "vue";
import StreamingTerminal from "./StreamingTerminal.vue";
import { buildTerminalLine } from "../utils/terminalOutput";

const props = defineProps({
  events: {
    type: Array,
    default: () => [],
  },
  status: {
    type: String,
    default: "Idle",
  },
  showTerminal: {
    type: Boolean,
    default: true,
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

const firstNumeric = (payload, keys) => {
  for (const key of keys) {
    const value = Number(payload?.[key]);
    if (Number.isFinite(value)) return value;
  }
  return null;
};

const buildMetricSeries = (keys) => {
  const series = [];
  for (const event of props.events) {
    if (event.type === "baseline" || event.type === "train") continue;
    const payload = event.payload || {};
    const value = firstNumeric(payload, keys);
    if (value == null) continue;
    const label =
      payload.episode != null
        ? `Episode ${payload.episode}`
        : payload.update != null
          ? `Update ${payload.update}`
          : payload.step != null
            ? `Step ${payload.step}`
            : event.type || "event";
    series.push({ value, label });
  }
  return series;
};

const coverageSeries = computed(() =>
  buildMetricSeries(["coverage", "mean_coverage", "episode_coverage", "comm_coverage", "coverage_ratio"])
);
const broadcastSeries = computed(() =>
  buildMetricSeries(["broadcast", "mean_broadcast", "episode_broadcast", "broadcast_coverage", "broadcast_ratio"])
);

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
const formatConsoleLine = (event) => {
  const payload = event?.payload || {};
  const timestamp = event?.timestamp;
  if (event?.message) {
    const levelMap = {
      backend: "BACKEND",
      device_state_sync: "SYNC",
      error: "ERROR",
      info: "INFO",
      scene_import: "ACTION",
      training_replay_error: "ERROR",
      training_replay_ready: "REPLAY",
      ui_action: "ACTION",
      warn: "WARN",
    };
    return buildTerminalLine(event.message, {
      level: levelMap[event?.type] || "INFO",
      source: "TRAIN",
      timestamp,
    });
  }
  if (event?.type === "log") {
    return buildTerminalLine(payload.message || JSON.stringify(payload), { level: "BACKEND", source: "TRAIN", timestamp });
  }
  if (event?.type === "experiment_config") {
    return buildTerminalLine(
      `loaded config lr=${payload.learningRate ?? payload.learning_rate ?? "-"} gamma=${payload.discountFactor ?? payload.discount_factor ?? "-"} batch=${payload.batchSize ?? payload.batch_size ?? "-"} rollout=${payload.rolloutSteps ?? payload.rollout_steps ?? "-"}`,
      { level: "CONFIG", source: "TRAIN", timestamp }
    );
  }
  if (event?.type === "status") {
    const step = payload.step != null ? ` step=${payload.step}` : "";
    return buildTerminalLine(`status=${payload.state || "unknown"}${step}`, { level: "STATUS", source: "TRAIN", timestamp });
  }
  if (event?.type === "baseline") {
    const coverage = firstNumeric(payload, ["avg_coverage", "coverage"]);
    const broadcast = firstNumeric(payload, ["avg_broadcast", "broadcast"]);
    return buildTerminalLine(
      `initial_state coverage=${percent(coverage)} | broadcast=${percent(broadcast)}`,
      { level: "BASE", source: "TRAIN", timestamp }
    );
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
    if (payload.hierarchy) {
      const summary = payload.hierarchy.summary || {};
      const rewards = payload.hierarchical_rewards || {};
      parts.push(`hmarl_region=${summary.target_region_id ?? "-"}`);
      parts.push(`l2_links=${summary.l2_link_count ?? 0}`);
      parts.push(`l3_devices=${summary.l3_deployed_devices ?? 0}`);
      parts.push(`hmarl_reward=${fixed(rewards.l3_final)}`);
    }
    return buildTerminalLine(parts.join(" | "), { level: "METRIC", source: "TRAIN", timestamp });
  }
  if (event?.type === "update") {
    const parts = [
      `update=${payload.update ?? "-"}`,
      `step=${payload.step ?? "-"}`,
      `reward=${fixed(payload.mean_reward)}`,
      `coverage=${percent(payload.mean_coverage)}`,
      `broadcast=${percent(payload.mean_broadcast)}`,
      `loss_pi=${fixed(payload.loss_pi)}`,
      `loss_v=${fixed(payload.loss_v)}`,
    ];
    if (payload.aux_loss != null) parts.push(`aux=${fixed(payload.aux_loss)}`);
    return buildTerminalLine(parts.join(" | "), { level: "METRIC", source: "TRAIN", timestamp });
  }
  if (event?.type === "train") {
    const parts = [
      `recovery episode=${payload.episode ?? "-"}`,
      `episode_step=${payload.episode_step ?? "-"}`,
      `global_step=${payload.step ?? "-"}`,
      `reward=${fixed(payload.reward)}`,
    ];
    if (payload.loss != null) parts.push(`loss=${fixed(payload.loss)}`);
    const coverage = firstNumeric(payload, ["coverage", "mean_coverage", "episode_coverage", "comm_coverage", "coverage_ratio"]);
    const broadcast = firstNumeric(payload, ["broadcast", "mean_broadcast", "episode_broadcast", "broadcast_coverage", "broadcast_ratio"]);
    if (coverage != null) parts.push(`coverage=${percent(coverage)}`);
    if (broadcast != null) parts.push(`broadcast=${percent(broadcast)}`);
    return buildTerminalLine(parts.join(" | "), { level: "METRIC", source: "TRAIN", timestamp });
  }
  if (event?.type === "error" || event?.type === "training_replay_error") {
    return buildTerminalLine(payload.message || "unknown error", { level: "ERROR", source: "TRAIN", timestamp });
  }
  if (event?.type === "training_replay_ready") {
    return buildTerminalLine(event.message || "训练回放已生成", { level: "REPLAY", source: "TRAIN", timestamp });
  }
  return buildTerminalLine(`${event?.type || "event"} ${JSON.stringify(payload)}`, { level: "EVENT", source: "TRAIN", timestamp });
};

const consoleLines = computed(() => props.events.map(formatConsoleLine).slice(-80));
</script>

<style scoped>
.monitor {
  display: flex;
  flex-direction: column;
  gap: 16px;
  color: #0f172a;
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
  border: 1px solid rgba(57, 97, 246, 0.16);
  background: rgba(231, 238, 255, 0.5);
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
  background: #1890ff;
}

.chart-point--broadcast {
  background: #409eff;
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
  color: #1890ff;
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
