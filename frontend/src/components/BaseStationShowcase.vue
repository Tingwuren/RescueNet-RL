<template>
  <section class="showcase">
    <div class="showcase__header">
      <div>
        <h3>基站装备与候选站点</h3>
        <p>展示背负式、小型、中继基站形态，并结合当前场景的候选站点分布做部署预览。</p>
      </div>
      <div class="showcase__summary">
        <span>候选站点 {{ candidateSites.length }}</span>
        <span>命名区域 {{ namedRegionCount }}</span>
        <span>设备类型 {{ stationProfiles.length }}</span>
      </div>
    </div>

    <div class="catalog-grid">
      <article v-for="item in catalogCards" :key="item.key" class="catalog-card">
        <img :src="item.image" :alt="item.title" />
        <div class="catalog-card__body">
          <span class="catalog-card__tag">{{ item.tag }}</span>
          <h4>{{ item.title }}</h4>
          <p>{{ item.description }}</p>
          <div class="catalog-card__chips">
            <span v-for="match in item.matches" :key="match">{{ match }}</span>
            <span v-if="!item.matches.length">当前场景未配置对应设备</span>
          </div>
        </div>
      </article>
    </div>

    <div class="station-grid">
      <article v-for="station in detailedStations" :key="station.name" class="station-card">
        <div class="station-card__media">
          <img :src="station.image" :alt="station.label" />
        </div>
        <div class="station-card__content">
          <div class="station-card__title">
            <strong>{{ station.label }}</strong>
            <span :class="['station-badge', `station-badge--${station.category}`]">{{ station.categoryLabel }}</span>
          </div>
          <p>支持模式：{{ station.displayModes.join(" / ") }}</p>
          <div class="station-card__stats">
            <span>吞吐 {{ station.max_throughput.toFixed(0) }} Mbps</span>
            <span>用户 {{ station.max_users }}</span>
            <span>设备成本 {{ station.device_cost.toFixed(2) }}</span>
            <span>带宽成本 {{ station.bandwidth_cost.toFixed(3) }}</span>
          </div>
        </div>
      </article>
    </div>

    <div class="site-board">
      <div class="site-board__main">
        <div class="site-map">
          <svg :viewBox="`0 0 ${mapWidth} ${mapHeight}`" preserveAspectRatio="xMidYMid meet">
            <rect :x="mapPadding" :y="mapPadding" :width="innerMapWidth" :height="innerMapHeight" rx="20" fill="#08111f" />

            <g v-for="cell in namedCells" :key="cell.key">
              <rect
                :x="mapPadding + cell.col * cellSize"
                :y="mapPadding + cell.row * cellSize"
                :width="cellSize"
                :height="cellSize"
                fill="rgba(250, 204, 21, 0.12)"
                stroke="rgba(250, 204, 21, 0.35)"
              />
            </g>

            <g stroke="rgba(148, 163, 184, 0.12)" stroke-width="1">
              <line
                v-for="row in gridRows + 1"
                :key="`row-${row}`"
                :x1="mapPadding"
                :y1="mapPadding + (row - 1) * cellSize"
                :x2="mapPadding + innerMapWidth"
                :y2="mapPadding + (row - 1) * cellSize"
              />
              <line
                v-for="col in gridCols + 1"
                :key="`col-${col}`"
                :x1="mapPadding + (col - 1) * cellSize"
                :y1="mapPadding"
                :x2="mapPadding + (col - 1) * cellSize"
                :y2="mapPadding + innerMapHeight"
              />
            </g>

            <g v-for="site in candidateSiteMarkers" :key="site.site_index">
              <circle :cx="site.cx" :cy="site.cy" :r="site.radius" :fill="site.color" :stroke="site.stroke" stroke-width="2" />
            </g>
          </svg>
        </div>

        <div class="site-legend">
          <span v-for="item in siteLegend" :key="item.key">
            <i :style="{ background: item.color }"></i>
            {{ item.label }} {{ item.count }}
          </span>
        </div>
      </div>

      <div class="site-board__aside">
        <div class="site-board__aside-header">
          <h4>代表性候选站点</h4>
          <p>优先展示命名区域和不同站点态势。</p>
        </div>
        <article v-for="site in featuredSites" :key="site.site_index" class="site-card">
          <div class="site-card__title">
            <strong>站点 #{{ site.site_index }}</strong>
            <span :class="['site-badge', `site-badge--${site.categoryKey}`]">{{ site.category }}</span>
          </div>
          <p>{{ site.region_label }}</p>
          <small>网格 {{ site.x }}, {{ site.y }}</small>
          <small>
            纬度 {{ site.lat_lon_bounds.lat_min.toFixed(3) }} - {{ site.lat_lon_bounds.lat_max.toFixed(3) }}，
            经度 {{ site.lat_lon_bounds.lon_min.toFixed(3) }} - {{ site.lat_lon_bounds.lon_max.toFixed(3) }}
          </small>
        </article>
      </div>
    </div>
  </section>
</template>

<script setup>
import { computed } from "vue";
import backpackStationImg from "../assets/base-stations/backpack-station.svg";
import compactStationImg from "../assets/base-stations/compact-station.svg";
import relayStationImg from "../assets/base-stations/relay-station.svg";

const props = defineProps({
  scenario: {
    type: Object,
    required: true,
  },
});

const stationProfiles = computed(() => props.scenario?.base_stations || []);
const candidateSites = computed(() => props.scenario?.candidate_site_preview || []);
const gridRows = computed(() => props.scenario?.region_grid?.rows || 1);
const gridCols = computed(() => props.scenario?.region_grid?.cols || 1);
const namedCells = computed(() =>
  Object.entries(props.scenario?.region_grid?.cell_labels || {}).map(([key, label]) => {
    const [row, col] = key.split(",").map((value) => Number(value));
    return { key, row, col, label };
  })
);
const namedRegionCount = computed(() => namedCells.value.length);

const categoryMap = {
  backpack: {
    key: "backpack",
    title: "背负式基站",
    tag: "Mobile Pack",
    image: backpackStationImg,
    description: "强调单兵携行和快速进场，适合断路、积水或狭窄街巷中的临时接入。",
    label: "背负式",
  },
  compact: {
    key: "compact",
    title: "小型基站",
    tag: "Compact Node",
    image: compactStationImg,
    description: "适合道路交汇、临时指挥点和居民集中区的快速补盲与容量恢复。",
    label: "小型",
  },
  relay: {
    key: "relay",
    title: "中继基站",
    tag: "Relay Node",
    image: relayStationImg,
    description: "用于远距离回传、跨障碍覆盖和核心通信链路续接，适合做跨区中继。",
    label: "中继",
  },
};

const sitePalette = {
  "重点保障": { key: "priority", label: "重点保障", color: "#facc15", stroke: "#fef3c7" },
  "核心覆盖": { key: "core", label: "核心覆盖", color: "#38bdf8", stroke: "#dbeafe" },
  "中继转发": { key: "relay", label: "中继转发", color: "#f97316", stroke: "#ffedd5" },
  "边缘补盲": { key: "edge", label: "边缘补盲", color: "#4ade80", stroke: "#dcfce7" },
  "机动接入": { key: "mobile", label: "机动接入", color: "#c084fc", stroke: "#f3e8ff" },
};

const resolveStationCategory = (station) => {
  const text = `${station.name || ""} ${station.label || ""} ${(station.supported_modes || []).join(" ")}`.toLowerCase();
  if (/中继|relay|satellite|ku|ka/.test(text)) return "relay";
  if (/便携|热点|短波|wifi|uav|mesh/.test(text)) return "backpack";
  return "compact";
};

const resolveDisplayModes = (station) => {
  if (station.name === "mmwave_micro" || station.label === "mmWave 微站") {
    return ["5g700hz"];
  }
  return station.supported_modes || [];
};

const detailedStations = computed(() =>
  stationProfiles.value.map((station) => {
    const category = resolveStationCategory(station);
    return {
      ...station,
      category,
      categoryLabel: categoryMap[category].label,
      image: categoryMap[category].image,
      displayModes: resolveDisplayModes(station),
    };
  })
);

const catalogCards = computed(() =>
  Object.values(categoryMap).map((card) => ({
    ...card,
    matches: detailedStations.value
      .filter((station) => station.category === card.key)
      .map((station) => station.label),
  }))
);

const mapPadding = 18;
const cellSize = 18;
const innerMapWidth = computed(() => gridCols.value * cellSize);
const innerMapHeight = computed(() => gridRows.value * cellSize);
const mapWidth = computed(() => innerMapWidth.value + mapPadding * 2);
const mapHeight = computed(() => innerMapHeight.value + mapPadding * 2);

const candidateSiteMarkers = computed(() =>
  candidateSites.value.map((site) => {
    const palette = sitePalette[site.category] || sitePalette["机动接入"];
    return {
      ...site,
      color: palette.color,
      stroke: palette.stroke,
      radius: site.category === "重点保障" ? 5.5 : 4.2,
      cx: mapPadding + (Number(site.y) + 0.5) * cellSize,
      cy: mapPadding + (Number(site.x) + 0.5) * cellSize,
    };
  })
);

const siteLegend = computed(() =>
  Object.entries(sitePalette).map(([label, palette]) => ({
    key: palette.key,
    label,
    color: palette.color,
    count: candidateSites.value.filter((site) => site.category === label).length,
  }))
);

const featuredSites = computed(() => {
  const selected = [];
  const seen = new Set();
  const ordered = [...candidateSites.value].sort((left, right) => left.site_index - right.site_index);

  for (const site of ordered) {
    if (site.region_label && !String(site.region_label).startsWith("cell-")) {
      selected.push(site);
      seen.add(site.site_index);
      if (selected.length >= 4) break;
    }
  }

  for (const key of Object.keys(sitePalette)) {
    const match = ordered.find((site) => site.category === key && !seen.has(site.site_index));
    if (match) {
      selected.push(match);
      seen.add(match.site_index);
    }
  }

  return selected.slice(0, 8).map((site) => ({
    ...site,
    categoryKey: (sitePalette[site.category] || sitePalette["机动接入"]).key,
  }));
});
</script>

<style scoped>
.showcase {
  display: flex;
  flex-direction: column;
  gap: 24px;
  border: 1px solid rgba(148, 163, 184, 0.22);
  border-radius: 18px;
  padding: 20px;
  background: rgba(15, 23, 42, 0.3);
}

.showcase__header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 16px;
}

.showcase__header h3,
.site-board__aside-header h4,
.catalog-card__body h4 {
  margin: 0;
}

.showcase__header p,
.site-board__aside-header p {
  margin: 6px 0 0;
  color: #94a3b8;
}

.showcase__summary {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 8px;
}

.showcase__summary span,
.site-legend span,
.catalog-card__chips span {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 8px 10px;
  border-radius: 999px;
  border: 1px solid rgba(148, 163, 184, 0.2);
  background: rgba(15, 23, 42, 0.5);
  color: #dbeafe;
  font-size: 12px;
}

.catalog-grid,
.station-grid {
  display: grid;
  gap: 16px;
}

.catalog-grid {
  grid-template-columns: repeat(3, minmax(0, 1fr));
}

.catalog-card {
  overflow: hidden;
  border-radius: 18px;
  border: 1px solid rgba(148, 163, 184, 0.2);
  background: linear-gradient(180deg, rgba(15, 23, 42, 0.72), rgba(2, 6, 23, 0.92));
}

.catalog-card img,
.station-card__media img {
  display: block;
  width: 100%;
  height: auto;
}

.catalog-card__body,
.station-card__content {
  padding: 14px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.catalog-card__tag {
  display: inline-flex;
  width: fit-content;
  padding: 6px 10px;
  border-radius: 999px;
  background: rgba(125, 211, 252, 0.12);
  color: #7dd3fc;
  font-size: 12px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.catalog-card__body p,
.station-card__content p {
  margin: 0;
  color: #bfd0e5;
  line-height: 1.6;
}

.catalog-card__chips {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.station-grid {
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
}

.station-card {
  display: grid;
  grid-template-columns: 120px minmax(0, 1fr);
  gap: 0;
  border-radius: 16px;
  overflow: hidden;
  border: 1px solid rgba(148, 163, 184, 0.18);
  background: rgba(15, 23, 42, 0.55);
}

.station-card__media {
  background: linear-gradient(180deg, rgba(15, 23, 42, 0.92), rgba(2, 6, 23, 0.96));
}

.station-card__title,
.site-card__title {
  display: flex;
  justify-content: space-between;
  gap: 8px;
  align-items: flex-start;
}

.station-card__stats {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.station-card__stats span {
  font-size: 12px;
  color: #dbeafe;
  padding: 6px 8px;
  border-radius: 10px;
  background: rgba(30, 41, 59, 0.72);
}

.station-badge,
.site-badge {
  padding: 6px 10px;
  border-radius: 999px;
  font-size: 12px;
}

.station-badge--backpack,
.site-badge--mobile {
  background: rgba(192, 132, 252, 0.16);
  color: #e9d5ff;
}

.station-badge--compact,
.site-badge--core {
  background: rgba(56, 189, 248, 0.16);
  color: #bae6fd;
}

.station-badge--relay,
.site-badge--relay {
  background: rgba(249, 115, 22, 0.16);
  color: #fed7aa;
}

.site-badge--priority {
  background: rgba(250, 204, 21, 0.16);
  color: #fde68a;
}

.site-badge--edge {
  background: rgba(74, 222, 128, 0.16);
  color: #bbf7d0;
}

.site-board {
  display: grid;
  grid-template-columns: minmax(0, 1.3fr) minmax(280px, 0.9fr);
  gap: 20px;
}

.site-board__main,
.site-board__aside {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.site-map {
  border-radius: 18px;
  overflow: hidden;
  border: 1px solid rgba(148, 163, 184, 0.16);
  background: linear-gradient(180deg, rgba(15, 23, 42, 0.88), rgba(2, 6, 23, 0.95));
}

.site-map svg {
  width: 100%;
  height: auto;
  display: block;
}

.site-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.site-legend i {
  width: 10px;
  height: 10px;
  border-radius: 999px;
  display: inline-block;
}

.site-card {
  padding: 14px;
  border-radius: 14px;
  border: 1px solid rgba(148, 163, 184, 0.18);
  background: rgba(15, 23, 42, 0.52);
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.site-card p,
.site-card small {
  margin: 0;
  color: #bfd0e5;
  line-height: 1.6;
}

@media (max-width: 1200px) {
  .catalog-grid,
  .site-board {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 820px) {
  .showcase__header,
  .station-card,
  .station-card__title,
  .site-card__title {
    grid-template-columns: 1fr;
    flex-direction: column;
  }

  .showcase__summary {
    justify-content: flex-start;
  }

  .station-card {
    display: flex;
    flex-direction: column;
  }
}
</style>
