<template>
  <div class="scene-graph">
    <div class="scene-graph__header">
      <div>
        <h3>{{ title }}</h3>
        <p>{{ subtitle }}</p>
      </div>
      <div class="scene-graph__stats">
        <span v-for="item in summaryItems" :key="item.label">
          {{ item.label }} {{ item.value }}
        </span>
      </div>
    </div>

    <div v-if="hasNodes" class="scene-graph__viewport">
      <svg :viewBox="`0 0 ${viewportWidth} ${viewportHeight}`" preserveAspectRatio="xMidYMid meet" role="img">
        <defs>
          <pattern id="scene-grid" width="40" height="40" patternUnits="userSpaceOnUse">
            <path d="M 40 0 L 0 0 0 40" fill="none" stroke="rgba(148, 163, 184, 0.14)" stroke-width="1" />
          </pattern>
        </defs>
        <rect
          :x="padding"
          :y="padding"
          :width="innerWidth"
          :height="innerHeight"
          rx="18"
          fill="rgba(15, 23, 42, 0.9)"
          stroke="rgba(148, 163, 184, 0.18)"
        />
        <rect :x="padding" :y="padding" :width="innerWidth" :height="innerHeight" rx="18" fill="url(#scene-grid)" />

        <g v-for="node in scaledNodes" :key="node.id">
          <circle
            :cx="node.x"
            :cy="node.y"
            :r="node.radius"
            :fill="node.color"
            :stroke="node.stroke"
            :stroke-width="node.strokeWidth"
          />
        </g>
      </svg>
    </div>

    <div v-else class="scene-graph__empty">
      当前场景没有可绘制节点。
    </div>

    <div class="scene-graph__legend">
      <span v-for="entry in legendEntries" :key="entry.type">
        <i :style="{ background: entry.color }"></i>
        {{ entry.label }}
      </span>
    </div>
  </div>
</template>

<script setup>
import { computed } from "vue";

const props = defineProps({
  scene: {
    type: Object,
    default: null,
  },
  title: {
    type: String,
    default: "场景图",
  },
  subtitle: {
    type: String,
    default: "",
  },
});

const viewportWidth = 680;
const viewportHeight = 420;
const padding = 24;
const innerWidth = viewportWidth - padding * 2;
const innerHeight = viewportHeight - padding * 2;

const styleMap = {
  USER: {
    label: "用户",
    color: "#facc15",
    stroke: "rgba(254, 249, 195, 0.75)",
    radius: 5,
    strokeWidth: 1.5,
  },
  MACRO_ENB: {
    label: "宏基站",
    color: "#38bdf8",
    stroke: "rgba(224, 242, 254, 0.75)",
    radius: 8,
    strokeWidth: 1.8,
  },
  MANPACK_ENB: {
    label: "便携基站",
    color: "#fb923c",
    stroke: "rgba(255, 237, 213, 0.78)",
    radius: 7,
    strokeWidth: 1.8,
  },
};

const rawNodes = computed(() => props.scene?.nodes || []);
const hasNodes = computed(() => rawNodes.value.length > 0);
const mapWidth = computed(() => Math.max(1, Number(props.scene?.map_width || 5000)));
const mapHeight = computed(() => Math.max(1, Number(props.scene?.map_height || 5000)));

const scaledNodes = computed(() =>
  rawNodes.value.map((node) => {
    const style = styleMap[node.type] || {
      label: node.type || "未知节点",
      color: "#cbd5f5",
      stroke: "rgba(255, 255, 255, 0.7)",
      radius: 6,
      strokeWidth: 1.5,
    };
    return {
      ...node,
      ...style,
      x: padding + (Number(node.x || 0) / mapWidth.value) * innerWidth,
      y: padding + (Number(node.y || 0) / mapHeight.value) * innerHeight,
    };
  })
);

const nodeCounts = computed(() =>
  rawNodes.value.reduce((acc, node) => {
    const key = node.type || "UNKNOWN";
    acc[key] = (acc[key] || 0) + 1;
    return acc;
  }, {})
);

const summaryItems = computed(() => [
  { label: "节点", value: rawNodes.value.length },
  { label: "用户", value: nodeCounts.value.USER || 0 },
  { label: "基站", value: (nodeCounts.value.MACRO_ENB || 0) + (nodeCounts.value.MANPACK_ENB || 0) },
]);

const legendEntries = computed(() => [
  { type: "USER", label: styleMap.USER.label, color: styleMap.USER.color },
  { type: "MACRO_ENB", label: styleMap.MACRO_ENB.label, color: styleMap.MACRO_ENB.color },
  { type: "MANPACK_ENB", label: styleMap.MANPACK_ENB.label, color: styleMap.MANPACK_ENB.color },
]);
</script>

<style scoped>
.scene-graph {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.scene-graph__header {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: flex-start;
}

.scene-graph__header h3 {
  margin: 0;
  font-size: 18px;
}

.scene-graph__header p {
  margin: 6px 0 0;
  color: #94a3b8;
}

.scene-graph__stats {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 8px;
}

.scene-graph__stats span,
.scene-graph__legend span {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 8px 10px;
  border-radius: 999px;
  background: rgba(15, 23, 42, 0.55);
  border: 1px solid rgba(148, 163, 184, 0.18);
  color: #dbeafe;
  font-size: 12px;
}

.scene-graph__viewport {
  border-radius: 20px;
  overflow: hidden;
  border: 1px solid rgba(148, 163, 184, 0.16);
  background: linear-gradient(180deg, rgba(15, 23, 42, 0.9), rgba(2, 6, 23, 0.95));
}

.scene-graph__viewport svg {
  width: 100%;
  height: auto;
  display: block;
}

.scene-graph__empty {
  padding: 28px;
  border-radius: 18px;
  border: 1px dashed rgba(148, 163, 184, 0.25);
  color: #94a3b8;
  text-align: center;
  background: rgba(15, 23, 42, 0.3);
}

.scene-graph__legend {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.scene-graph__legend i {
  width: 10px;
  height: 10px;
  border-radius: 999px;
  display: inline-block;
}

@media (max-width: 720px) {
  .scene-graph__header {
    flex-direction: column;
  }

  .scene-graph__stats {
    justify-content: flex-start;
  }
}
</style>
