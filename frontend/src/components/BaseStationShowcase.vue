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

    <div class="equipment-grid">
      <button
        v-for="card in equipmentCards"
        :key="card.key"
        type="button"
        class="equipment-card"
        :class="{ 'equipment-card--active': selectedCatalogKey === card.category }"
        @click="() => openEquipmentModal(card)"
      >
        <div class="equipment-card__media">
          <img :src="card.image" :alt="card.title" />
          <span :class="['station-badge', `station-badge--${card.category}`]">{{ card.badge }}</span>
        </div>
        <div class="equipment-card__content">
          <div class="equipment-card__title">
            <strong>{{ card.title }}</strong>
            <small>点击查看装备介绍</small>
          </div>
          <p>{{ card.description }}</p>
          <div class="equipment-card__tags" aria-label="标签">
            <span v-for="tag in card.previewTags" :key="`${card.key}-${tag}`">{{ tag }}</span>
          </div>
        </div>
      </button>
    </div>

    <div v-if="activeStation" class="station-modal" role="dialog" aria-modal="true" @click.self="closeStationModal">
      <article class="station-modal__card">
        <button type="button" class="station-modal__close" aria-label="关闭设备介绍" @click="closeStationModal">×</button>
        <div class="station-modal__media">
          <img :src="activeStation.image" :alt="activeStation.label" />
        </div>
        <div class="station-modal__content">
          <span :class="['station-badge', `station-badge--${activeStation.category}`]">{{ activeStation.categoryLabel }}</span>
          <h3>{{ activeStation.label }}</h3>
          <p>{{ activeStation.intro }}</p>
          <div v-if="!activeStation.isCatalog" class="station-modal__stats">
            <span><small>峰值吞吐</small><strong>{{ activeStation.max_throughput.toFixed(0) }} Mbps</strong></span>
            <span><small>接入用户</small><strong>{{ activeStation.max_users }}</strong></span>
            <span><small>设备成本</small><strong>{{ activeStation.device_cost.toFixed(2) }}</strong></span>
            <span><small>带宽成本</small><strong>{{ activeStation.bandwidth_cost.toFixed(3) }}</strong></span>
          </div>
          <div class="station-modal__sections">
            <section>
              <strong>支持模式</strong>
              <p>{{ activeStation.displayModes.join(" / ") || "未配置" }}</p>
            </section>
            <section>
              <strong>参数设计</strong>
              <p>{{ activeStation.parameterDesign }}</p>
            </section>
            <section>
              <strong>奖励设计</strong>
              <p>{{ activeStation.rewardDesign }}</p>
            </section>
            <section v-if="activeStation.useCase">
              <strong>典型用途</strong>
              <p>{{ activeStation.useCase }}</p>
            </section>
          </div>
        </div>
      </article>
    </div>

    <div class="site-board">
      <div class="site-board__main">
        <div class="site-map">
          <svg :viewBox="`0 0 ${mapWidth} ${mapHeight}`" preserveAspectRatio="xMidYMid meet">
            <rect :x="mapPadding" :y="mapPadding" :width="innerMapWidth" :height="innerMapHeight" rx="20" fill="#f8fafc" />

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

            <g stroke="rgba(100, 116, 139, 0.22)" stroke-width="1">
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
import { computed, ref } from "vue";
import backpackStationImg from "../assets/base-stations/photos/backpack-station-real.jpg";
import compactStationImg from "../assets/base-stations/photos/compact-station-real.jpg";
import relayStationImg from "../assets/base-stations/photos/relay-station-real.jpg";

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
    intro: "面向道路受阻、楼宇遮挡和临时救援点的便携式接入装备，重点体现快速部署和低成本补盲。",
    parameterDesign: "在仿真中通常给中等吞吐、中等用户容量和较低设备成本，使策略更愿意把它部署到边缘盲区或用户密集但预算有限的位置。",
    rewardDesign: "奖励侧强调覆盖率提升和设备成本惩罚的平衡：如果它能用较低成本连接更多离线用户，策略会获得更高净收益。",
    useCase: "适合救援队随行、街巷补盲、临时安置点接入和小范围热点恢复。",
    label: "背负式",
  },
  compact: {
    key: "compact",
    title: "小型基站",
    tag: "Compact Node",
    image: compactStationImg,
    description: "适合道路交汇、临时指挥点和居民集中区的快速补盲与容量恢复。",
    intro: "面向局部高并发区域的小型化通信节点，兼顾覆盖范围、吞吐能力和部署成本。",
    parameterDesign: "在仿真中通常给较高吞吐和较高接入用户上限，同时设备成本高于背负式节点，用来测试算法的容量分配能力。",
    rewardDesign: "奖励侧更关注吞吐收益和覆盖收益：当候选站点处于人群集中区域时，高容量带来的收益会抵消更高成本。",
    useCase: "适合临时指挥部、医院周边、居民集中区、交通枢纽和救援物资集散点。",
    label: "小型",
  },
  relay: {
    key: "relay",
    title: "中继基站",
    tag: "Relay Node",
    image: relayStationImg,
    description: "用于远距离回传、跨障碍覆盖和核心通信链路续接，适合做跨区中继。",
    intro: "面向跨区连通和回传链路恢复的中继装备，重点解决灾后断链区域之间的通信接续问题。",
    parameterDesign: "在仿真中通常给更强覆盖或回传能力，但设备成本和带宽成本更高，避免算法无脑铺设中继节点。",
    rewardDesign: "奖励侧强调广播覆盖、链路恢复和成本约束：只有当它明显改善跨区覆盖或关键区域连通时，策略才会倾向选择。",
    useCase: "适合山地阻隔、桥梁中断、跨河通信、远距离回传和核心保障区域之间的链路接续。",
    label: "中继",
  },
};

const selectedCatalogKey = ref("backpack");
const activeStation = ref(null);

const openEquipmentModal = (card) => {
  selectedCatalogKey.value = card.category;
  activeStation.value = card.modal;
};

const closeStationModal = () => {
  activeStation.value = null;
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
      intro: `${categoryMap[category].label}装备用于${categoryMap[category].useCase.replace(/。$/, "")}，当前场景中对应 ${station.label}。`,
      parameterDesign: `${station.label} 的仿真参数设置为峰值吞吐 ${Number(station.max_throughput || 0).toFixed(0)} Mbps、最大接入 ${station.max_users} 用户、设备成本 ${Number(station.device_cost || 0).toFixed(2)}、带宽成本 ${Number(station.bandwidth_cost || 0).toFixed(3)}，用于约束算法在覆盖收益和部署代价之间做权衡。`,
      rewardDesign: `${station.label} 会通过覆盖率、广播覆盖和吞吐提升获得正向奖励，同时设备成本与带宽成本进入惩罚项，避免策略只追求高性能设备堆叠。`,
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

const equipmentCards = computed(() => [
  ...catalogCards.value.map((card) => ({
    key: `category-${card.key}`,
    category: card.key,
    badge: card.label,
    title: card.title,
    image: card.image,
    description: card.description,
    previewTags: card.matches.length ? card.matches.slice(0, 2) : ["当前场景未配置"],
    modal: {
      ...card,
      label: `${card.title}介绍`,
      category: card.key,
      categoryLabel: card.label,
      displayModes: card.matches.length ? card.matches : ["当前场景未配置对应设备"],
      isCatalog: true,
    },
  })),
  ...detailedStations.value.map((station) => ({
    key: `station-${station.name}`,
    category: station.category,
    badge: station.categoryLabel,
    title: station.label,
    image: station.image,
    description: station.intro,
    previewTags: station.displayModes.slice(0, 2),
    modal: station,
  })),
]);

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
  background:
    radial-gradient(circle at 100% 0%, rgba(14, 165, 233, 0.1), transparent 34%),
    linear-gradient(180deg, rgba(255, 255, 255, 0.96), rgba(248, 250, 252, 0.94));
  color: #0f172a;
  box-shadow: 0 16px 32px rgba(15, 23, 42, 0.07);
}

.showcase__header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 16px;
}

.showcase__header h3,
.site-board__aside-header h4,
.equipment-card__content h4 {
  margin: 0;
}

.showcase__header p,
.site-board__aside-header p {
  margin: 6px 0 0;
  color: #64748b;
}

.showcase__summary {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 8px;
}

.showcase__summary span,
.site-legend span {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 8px 10px;
  border-radius: 999px;
  border: 1px solid rgba(14, 165, 233, 0.16);
  background: rgba(224, 242, 254, 0.72);
  color: #075985;
  font-size: 12px;
}

.equipment-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 18px;
  align-items: stretch;
}

.equipment-card {
  overflow: hidden;
  border-radius: 22px;
  border: 1px solid rgba(14, 165, 233, 0.18);
  background:
    linear-gradient(135deg, rgba(224, 242, 254, 0.66), rgba(255, 255, 255, 0.97));
  color: #0f172a;
  text-align: left;
  cursor: pointer;
  font: inherit;
  box-shadow: 0 16px 30px rgba(15, 23, 42, 0.07);
  transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
  height: 386px;
}

.equipment-card:hover,
.equipment-card--active {
  border-color: rgba(2, 132, 199, 0.44);
  box-shadow: 0 22px 38px rgba(14, 165, 233, 0.15);
  transform: translateY(-2px);
}

.equipment-card__media img {
  display: block;
  width: 100%;
  height: 176px;
  object-fit: cover;
}

.equipment-card__content {
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 10px;
  min-height: 210px;
  height: 210px;
}

.equipment-card__content p {
  margin: 0;
  color: #475569;
  line-height: 1.6;
  display: -webkit-box;
  overflow: hidden;
  -webkit-box-orient: vertical;
}

.equipment-card__content p {
  min-height: 86px;
  -webkit-line-clamp: 3;
  font-size: 13px;
}

.equipment-card__title {
  display: flex;
  justify-content: space-between;
  gap: 8px;
  align-items: baseline;
  min-height: 48px;
}

.equipment-card__title strong {
  font-size: 18px;
  color: #0f172a;
  letter-spacing: 0.01em;
  display: -webkit-box;
  overflow: hidden;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
}

.equipment-card__title small {
  color: #0284c7;
  font-size: 12px;
  flex: 0 0 auto;
}

.equipment-card__media {
  position: relative;
  min-height: 176px;
  background: linear-gradient(180deg, rgba(224, 242, 254, 0.72), rgba(248, 250, 252, 0.96));
}

.equipment-card__media::after {
  content: "";
  position: absolute;
  inset: 0;
  background: linear-gradient(180deg, transparent 45%, rgba(15, 23, 42, 0.36));
  pointer-events: none;
}

.site-card__title {
  display: flex;
  justify-content: space-between;
  gap: 8px;
  align-items: flex-start;
}

.equipment-card__tags {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  min-height: 34px;
  max-height: 34px;
  overflow: hidden;
  margin-top: auto;
}

.equipment-card__tags span {
  padding: 7px 9px;
  border-radius: 999px;
  background: rgba(241, 245, 249, 0.9);
  border: 1px solid rgba(148, 163, 184, 0.18);
  color: #475569;
  font-size: 12px;
}

.station-modal {
  position: fixed;
  inset: 0;
  z-index: 50;
  padding: 32px;
  background: rgba(15, 23, 42, 0.58);
  backdrop-filter: blur(10px);
  display: grid;
  place-items: center;
}

.station-modal__card {
  position: relative;
  width: min(920px, 100%);
  max-height: min(760px, calc(100vh - 64px));
  overflow: auto;
  display: grid;
  grid-template-columns: minmax(280px, 0.85fr) minmax(0, 1.15fr);
  gap: 22px;
  padding: 22px;
  border-radius: 26px;
  border: 1px solid rgba(255, 255, 255, 0.72);
  background:
    radial-gradient(circle at 0% 0%, rgba(14, 165, 233, 0.18), transparent 34%),
    linear-gradient(135deg, rgba(248, 250, 252, 0.98), rgba(255, 255, 255, 0.96));
  box-shadow: 0 30px 80px rgba(15, 23, 42, 0.32);
}

.station-modal__close {
  position: absolute;
  top: 14px;
  right: 14px;
  width: 36px;
  height: 36px;
  border-radius: 999px;
  border: 1px solid rgba(148, 163, 184, 0.24);
  background: rgba(255, 255, 255, 0.9);
  color: #0f172a;
  font-size: 22px;
  line-height: 1;
  cursor: pointer;
}

.station-modal__media {
  min-height: 360px;
  border-radius: 22px;
  overflow: hidden;
  border: 1px solid rgba(148, 163, 184, 0.2);
  background: rgba(224, 242, 254, 0.5);
}

.station-modal__media img {
  width: 100%;
  height: 100%;
  min-height: 360px;
  display: block;
  object-fit: cover;
}

.station-modal__content {
  display: flex;
  flex-direction: column;
  gap: 14px;
  padding: 10px 10px 10px 0;
}

.station-modal__content h3,
.station-modal__content p,
.station-modal__sections p {
  margin: 0;
}

.station-modal__content h3 {
  font-size: 28px;
  color: #0f172a;
}

.station-modal__content p {
  color: #475569;
  line-height: 1.75;
}

.station-modal__stats {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
}

.station-modal__stats span,
.station-modal__sections section {
  padding: 14px;
  border-radius: 16px;
  border: 1px solid rgba(14, 165, 233, 0.16);
  background: rgba(255, 255, 255, 0.78);
}

.station-modal__stats span {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.station-modal__stats small {
  color: #64748b;
}

.station-modal__stats strong,
.station-modal__sections strong {
  color: #075985;
}

.station-modal__sections {
  display: grid;
  gap: 10px;
}

.station-badge,
.site-badge {
  padding: 6px 10px;
  border-radius: 999px;
  font-size: 12px;
}

.equipment-card__media .station-badge {
  position: absolute;
  left: 12px;
  bottom: 12px;
  z-index: 1;
  border: 1px solid rgba(255, 255, 255, 0.72);
  box-shadow: 0 8px 18px rgba(15, 23, 42, 0.2);
}

.station-badge--backpack,
.site-badge--mobile {
  background: rgba(243, 232, 255, 0.9);
  color: #7e22ce;
}

.station-badge--compact,
.site-badge--core {
  background: rgba(224, 242, 254, 0.9);
  color: #0369a1;
}

.station-badge--relay,
.site-badge--relay {
  background: rgba(255, 237, 213, 0.95);
  color: #c2410c;
}

.site-badge--priority {
  background: rgba(254, 249, 195, 0.95);
  color: #a16207;
}

.site-badge--edge {
  background: rgba(220, 252, 231, 0.95);
  color: #15803d;
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
  border: 1px solid rgba(148, 163, 184, 0.22);
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.96), rgba(241, 245, 249, 0.92));
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.88);
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
  border: 1px solid rgba(148, 163, 184, 0.2);
  background: rgba(255, 255, 255, 0.9);
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.site-card p,
.site-card small {
  margin: 0;
  color: #475569;
  line-height: 1.6;
}

@media (max-width: 1200px) {
  .equipment-grid,
  .site-board,
  .station-modal__card {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .site-board,
  .station-modal__card {
    grid-template-columns: 1fr;
  }

  .station-modal__media,
  .station-modal__media img {
    min-height: 260px;
  }
}

@media (max-width: 820px) {
  .showcase__header,
  .equipment-card__title,
  .site-card__title {
    grid-template-columns: 1fr;
    flex-direction: column;
  }

  .showcase__summary {
    justify-content: flex-start;
  }

  .equipment-grid {
    grid-template-columns: 1fr;
  }

  .station-modal {
    padding: 16px;
  }

  .station-modal__stats {
    grid-template-columns: 1fr;
  }
}
</style>
