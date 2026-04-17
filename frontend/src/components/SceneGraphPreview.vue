<template>
  <div class="scene-graph">
    <div v-if="showHeader" class="scene-graph__header">
      <div>
        <h3 v-if="title">{{ title }}</h3>
        <p v-if="subtitle">{{ subtitle }}</p>
      </div>
      <div class="scene-graph__stats">
        <span v-for="item in summaryItems" :key="item.label">
          {{ item.label }} {{ item.value }}
        </span>
      </div>
    </div>

    <div v-if="hasNodes && hasGeoMap" ref="mapRef" class="scene-graph__map" aria-label="真实地图场景预览"></div>

    <div v-else-if="hasNodes" class="scene-graph__viewport">
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
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

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
  scenarioName: {
    type: String,
    default: "",
  },
  sceneKind: {
    type: String,
    default: "imported",
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
    color: "#ef4444",
    stroke: "rgba(254, 226, 226, 0.95)",
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
const showHeader = computed(() => Boolean(props.title || props.subtitle));
const mapWidth = computed(() => Math.max(1, Number(props.scene?.map_width || 5000)));
const mapHeight = computed(() => Math.max(1, Number(props.scene?.map_height || 5000)));
const mapRef = ref(null);
let mapInstance = null;
let nodeLayer = null;
let canvasRenderer = null;

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

const hashNumber = (value) => {
  const text = String(value ?? "");
  let hash = 0;
  for (let index = 0; index < text.length; index += 1) {
    hash = (hash * 31 + text.charCodeAt(index)) % 9973;
  }
  return hash / 9973;
};

const hasGeoMap = computed(() => {
  const bounds = props.scene?.geo_bounds;
  return Boolean(
    bounds &&
      Number.isFinite(Number(bounds.lat_min)) &&
      Number.isFinite(Number(bounds.lat_max)) &&
      Number.isFinite(Number(bounds.lon_min)) &&
      Number.isFinite(Number(bounds.lon_max)) &&
      rawNodes.value.some((node) => Number.isFinite(Number(node.lat)) && Number.isFinite(Number(node.lon)))
  );
});

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
  { type: "USER", label: "未恢复用户", color: styleMap.USER.color },
  { type: "CONNECTED_USER", label: "已连接用户", color: "#22c55e" },
  { type: "BROADCAST_USER", label: "广播覆盖", color: "#facc15" },
  { type: "MACRO_ENB", label: styleMap.MACRO_ENB.label, color: styleMap.MACRO_ENB.color },
  { type: "MANPACK_ENB", label: styleMap.MANPACK_ENB.label, color: styleMap.MANPACK_ENB.color },
]);

const mapBounds = computed(() => {
  const bounds = props.scene?.geo_bounds || {};
  return [
    [Number(bounds.lat_min), Number(bounds.lon_min)],
    [Number(bounds.lat_max), Number(bounds.lon_max)],
  ];
});

const visualBounds = computed(() => {
  const bounds = props.scene?.geo_bounds || {};
  const latMin = Number(bounds.lat_min);
  const latMax = Number(bounds.lat_max);
  const lonMin = Number(bounds.lon_min);
  const lonMax = Number(bounds.lon_max);
  if (![latMin, latMax, lonMin, lonMax].every(Number.isFinite)) {
    return mapBounds.value;
  }

  const latSpan = latMax - latMin;
  const lonSpan = lonMax - lonMin;
  const scenario = props.scenarioName || props.scene?.name || "";

  if (scenario.includes("typhoon")) {
    // Use a coastal city land viewport for typhoon previews; the original Zhuhai bounds include large sea areas.
    return [
      [22.49, 113.86],
      [22.74, 114.24],
    ];
  }

  return [
    [latMin + latSpan * 0.08, lonMin + lonSpan * 0.08],
    [latMax - latSpan * 0.08, lonMax - lonSpan * 0.08],
  ];
});

const nodeCellCounts = computed(() => {
  const counts = new Map();
  rawNodes.value.forEach((node) => {
    const key = `${Math.round(Number(node.x) || 0)}:${Math.round(Number(node.y) || 0)}:${node.type || "UNKNOWN"}`;
    counts.set(key, (counts.get(key) || 0) + 1);
  });
  return counts;
});

const nodeCellRanks = computed(() => {
  const seen = new Map();
  const ranks = new Map();
  rawNodes.value.forEach((node) => {
    const key = `${Math.round(Number(node.x) || 0)}:${Math.round(Number(node.y) || 0)}:${node.type || "UNKNOWN"}`;
    const rank = seen.get(key) || 0;
    seen.set(key, rank + 1);
    ranks.set(node.id, rank);
  });
  return ranks;
});

const nodeExtent = computed(() => {
  const xs = rawNodes.value.map((node) => Number(node.x)).filter(Number.isFinite);
  const ys = rawNodes.value.map((node) => Number(node.y)).filter(Number.isFinite);
  return {
    minX: xs.length ? Math.min(...xs) : 0,
    maxX: xs.length ? Math.max(...xs) : 1,
    minY: ys.length ? Math.min(...ys) : 0,
    maxY: ys.length ? Math.max(...ys) : 1,
  };
});

const projectedGeoNodes = computed(() => {
  const [[latMin, lonMin], [latMax, lonMax]] = visualBounds.value;
  const extent = nodeExtent.value;
  const spanX = Math.max(1, extent.maxX - extent.minX);
  const spanY = Math.max(1, extent.maxY - extent.minY);
  const latSpan = latMax - latMin;
  const lonSpan = lonMax - lonMin;

  return rawNodes.value
    .map((node) => {
      const x = Number(node.x);
      const y = Number(node.y);
      if (!Number.isFinite(x) || !Number.isFinite(y)) return null;

      const cellKey = `${Math.round(x)}:${Math.round(y)}:${node.type || "UNKNOWN"}`;
      const cellCount = nodeCellCounts.value.get(cellKey) || 1;
      const cellRank = nodeCellRanks.value.get(node.id) || 0;
      const ring = Math.floor(Math.sqrt(cellRank));
      const angle = cellRank * 2.399963 + hashNumber(`${node.id}-angle`) * 0.7;
      const spreadBase = node.type === "USER" ? 0.018 : 0.006;
      const spread = Math.min(0.048, spreadBase * Math.max(1, Math.sqrt(cellCount) * 0.45));
      const radial = cellCount > 1 ? ((ring + 1) / Math.max(2, Math.sqrt(cellCount))) * spread : 0;
      const jitterX = Math.cos(angle) * radial + (hashNumber(`${node.id}-x`) - 0.5) * 0.006;
      const jitterY = Math.sin(angle) * radial + (hashNumber(`${node.id}-y`) - 0.5) * 0.006;
      const nx = clamp((x - extent.minX) / spanX + jitterX, 0.035, 0.965);
      const ny = clamp((y - extent.minY) / spanY + jitterY, 0.035, 0.965);

      return {
        ...node,
        lat: latMax - ny * latSpan,
        lon: lonMin + nx * lonSpan,
      };
    })
    .filter(Boolean);
});

const visibleGeoNodes = computed(() => {
  const users = projectedGeoNodes.value.filter((node) => node.type === "USER");
  const others = projectedGeoNodes.value.filter((node) => node.type !== "USER");
  const maxUsers = 260;
  if (users.length <= maxUsers) {
    return projectedGeoNodes.value;
  }
  const step = Math.ceil(users.length / maxUsers);
  const sampledUsers = users.filter((node, index) => index % step === 0 || hashNumber(node.id) > 0.985);
  return [...sampledUsers.slice(0, maxUsers), ...others];
});

const restoredUsers = computed(() =>
  visibleGeoNodes.value.filter((node) => node.type === "USER" && (node.connected || node.broadcast_served))
);

const stationNodes = computed(() => visibleGeoNodes.value.filter((node) => node.type !== "USER"));

const restorationLinks = computed(() => {
  if (props.sceneKind !== "deployment") return [];
  const stations = stationNodes.value;
  if (!stations.length) return [];

  const sourceUsers = restoredUsers.value.length
    ? restoredUsers.value
    : visibleGeoNodes.value.filter((node) => node.type === "USER");

  const distanceOf = (user, station) => {
    const dLat = Number(user.lat) - Number(station.lat);
    const dLon = Number(user.lon) - Number(station.lon);
    return dLat * dLat + dLon * dLon;
  };

  const links = [];
  const usedUsers = new Set();
  const perStation = stations.length > 16 ? 7 : 10;

  for (const station of stations) {
    const nearestUsers = sourceUsers
      .map((user) => ({ user, station, distance: distanceOf(user, station) }))
      .sort((a, b) => a.distance - b.distance)
      .slice(0, perStation);

    for (const link of nearestUsers) {
      if (usedUsers.has(link.user.id)) continue;
      usedUsers.add(link.user.id);
      links.push(link);
    }
  }

  const targetLinks = Math.min(sourceUsers.length, 210);
  if (links.length < targetLinks) {
    const nearestFallback = sourceUsers
      .filter((user) => !usedUsers.has(user.id))
      .map((user) => {
        let bestStation = null;
        let bestDistance = Infinity;
        for (const station of stations) {
          const distance = distanceOf(user, station);
          if (distance < bestDistance) {
            bestDistance = distance;
            bestStation = station;
          }
        }
        return bestStation ? { user, station: bestStation, distance: bestDistance } : null;
      })
      .filter(Boolean)
      .sort((a, b) => a.distance - b.distance);

    for (const link of nearestFallback) {
      if (links.length >= targetLinks) break;
      usedUsers.add(link.user.id);
      links.push(link);
    }
  }

  return links;
});

const restoredUserIds = computed(() => new Set(restorationLinks.value.map((link) => link.user.id)));

const destroyMap = () => {
  if (nodeLayer) {
    nodeLayer.remove();
    nodeLayer = null;
  }
  if (mapInstance) {
    mapInstance.remove();
    mapInstance = null;
  }
  canvasRenderer = null;
};

const renderLeafletMap = async () => {
  await nextTick();
  if (!hasGeoMap.value || !mapRef.value) {
    destroyMap();
    return;
  }

  if (!mapInstance) {
    mapInstance = L.map(mapRef.value, {
      zoomControl: false,
      attributionControl: false,
      scrollWheelZoom: false,
      preferCanvas: true,
    });
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      crossOrigin: true,
    }).addTo(mapInstance);
    canvasRenderer = L.canvas({ padding: 0.5 });
  }

  mapInstance.invalidateSize();
  mapInstance.fitBounds(visualBounds.value, { padding: [18, 18], animate: false, maxZoom: 12 });

  if (nodeLayer) {
    nodeLayer.remove();
  }
  nodeLayer = L.layerGroup().addTo(mapInstance);

  for (const link of restorationLinks.value) {
    L.polyline(
      [
        [Number(link.station.lat), Number(link.station.lon)],
        [Number(link.user.lat), Number(link.user.lon)],
      ],
      {
        color: "rgba(255, 255, 255, 0.78)",
        weight: 3.2,
        opacity: 0.72,
        interactive: false,
      }
    ).addTo(nodeLayer);

    L.polyline(
      [
        [Number(link.station.lat), Number(link.station.lon)],
        [Number(link.user.lat), Number(link.user.lon)],
      ],
      {
        color: "rgba(22, 163, 74, 0.92)",
        weight: 2.2,
        opacity: 0.9,
        dashArray: "6 5",
        interactive: false,
      }
    ).addTo(nodeLayer);
  }

  const orderedNodes = [
    ...visibleGeoNodes.value.filter((node) => node.type === "USER"),
    ...visibleGeoNodes.value.filter((node) => node.type !== "USER"),
  ];

  for (const node of orderedNodes) {
    const lat = Number(node.lat);
    const lon = Number(node.lon);
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) continue;
    const style = styleMap[node.type] || {
      color: "#64748b",
      stroke: "rgba(15, 23, 42, 0.42)",
      radius: 5,
      strokeWidth: 1.3,
    };
    L.circleMarker([lat, lon], {
      renderer: canvasRenderer,
      radius: node.type === "USER" ? 3.6 : style.radius + 1.2,
      fillColor:
        node.type === "USER"
          ? restoredUserIds.value.has(node.id) || node.connected
            ? "#22c55e"
            : node.broadcast_served
              ? "#facc15"
              : style.color
          : style.color,
      fillOpacity: node.type === "USER" ? 0.86 : 0.96,
      color: style.stroke,
      weight: node.type === "USER" ? 1.1 : style.strokeWidth + 1,
      opacity: 0.95,
    }).addTo(nodeLayer);
  }
};

watch(() => [props.scene, props.scenarioName], renderLeafletMap, { deep: false });
onMounted(renderLeafletMap);
onBeforeUnmount(destroyMap);
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

.scene-graph__map {
  width: 100%;
  aspect-ratio: 16 / 9;
  height: auto;
  min-height: 380px;
  max-height: 560px;
  border-radius: 8px;
  overflow: hidden;
  border: 1px solid rgba(148, 163, 184, 0.24);
  background: #eef2f6;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.8);
}

.scene-graph__map :deep(.leaflet-tile-pane) {
  filter: saturate(0.92) contrast(0.96) brightness(1.04);
}

.scene-graph__map :deep(.leaflet-container) {
  font-family: inherit;
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

  .scene-graph__map {
    min-height: 300px;
  }
}
</style>
