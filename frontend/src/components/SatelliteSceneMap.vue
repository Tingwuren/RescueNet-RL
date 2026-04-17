<template>
  <div class="satellite-board" :class="`satellite-board--${stage}`">
    <div ref="mapEl" class="satellite-board__map" aria-label="广州灾情标准地图"></div>

    <div v-if="stageContextLevel >= 1" :class="['intake-layer', { 'intake-layer--context': stage !== 'intake' }]" aria-hidden="true">
      <span v-if="stage === 'intake'" class="intake-scan"></span>

      <span
        v-for="node in outageNodes"
        :key="`${node.x}-${node.y}`"
        :class="[
          'intake-outage-node',
          `intake-outage-node--${node.level || 'normal'}`,
          { 'intake-outage-node--served': (stage === 'training' || stage === 'deploy') && node.served },
        ]"
        :style="{ '--x': node.x, '--y': node.y, '--delay': node.delay }"
      ></span>

      <span
        v-for="area in outageAreas"
        :key="area.label"
        class="intake-outage-area-label"
        :style="{ '--x': area.x, '--y': area.y, '--delay': area.delay }"
      >
        {{ area.label }}
      </span>

      <span
        v-for="station in residualStations"
        :key="station.label"
        :class="['intake-station', `intake-station--${station.status}`]"
        :style="{ '--x': station.x, '--y': station.y, '--delay': station.delay }"
      >
        <i></i>
        <small>{{ station.label }}</small>
      </span>

      <span
        v-for="priority in priorityZones"
        :key="priority.label"
        class="intake-priority"
        :style="{ '--x': priority.x, '--y': priority.y, '--tone': priority.tone, '--delay': priority.delay }"
      >
        <i>{{ priority.short }}</i>
        <small>{{ priority.label }}</small>
      </span>

      <div v-if="stage === 'intake'" class="intake-status">
        <strong>灾情接入完成</strong>
        <span v-for="item in intakeStatus" :key="item">{{ item }}</span>
      </div>
    </div>

    <div v-if="stageContextLevel >= 2" :class="['candidate-layer', { 'candidate-layer--context': stage !== 'sites' }]" aria-hidden="true">
      <svg class="candidate-links" viewBox="0 0 100 100" preserveAspectRatio="none">
        <path d="M58 42 C62 38, 68 36, 72 30" />
        <path d="M49 46 C43 48, 38 52, 34 58" />
        <path d="M54 50 C58 55, 64 58, 70 60" />
      </svg>

      <span
        v-for="site in candidateSites"
        :key="site.label"
        :class="['candidate-site', `candidate-site--${site.priority}`]"
        :style="{ '--x': site.x, '--y': site.y, '--delay': site.delay }"
      >
        <i>{{ site.short }}</i>
        <small>{{ site.label }}</small>
      </span>

      <div class="candidate-note">
        <strong>候选部署点已生成</strong>
        <span>围绕断联片区、残余基站与重点保障点筛选</span>
      </div>
    </div>

    <div v-if="stage === 'training'" class="training-layer" aria-hidden="true">
      <svg class="training-rank-links" viewBox="0 0 100 100" preserveAspectRatio="none">
        <path class="training-service-link training-service-link--strong" d="M60 40 L66 28" />
        <path class="training-service-link training-service-link--strong" d="M60 40 L61 33" />
        <path class="training-service-link training-service-link--strong" d="M60 40 L64 22" />
        <path class="training-service-link" d="M63 30 L72 27" />
        <path class="training-service-link" d="M63 30 L74 34" />
        <path class="training-service-link" d="M55 52 L47 46" />
        <path class="training-service-link" d="M55 52 L51 52" />
        <path class="training-service-link" d="M55 52 L57 43" />
        <path class="training-service-link training-service-link--warn" d="M43 55 L31 60" />
        <path class="training-service-link training-service-link--warn" d="M43 55 L24 58" />
        <path class="training-service-link" d="M47 67 L47 57" />
        <path class="training-service-link" d="M47 67 L51 75" />
        <path class="training-service-link" d="M66 58 L75 56" />
        <path class="training-service-link" d="M72 61 L80 60" />
      </svg>
      <span
        v-for="(pick, index) in policyPicks"
        :key="pick.label"
        class="policy-pick"
        :style="{ '--x': pick.x, '--y': pick.y, '--delay': pick.delay }"
      >
        <i>{{ pick.short }}</i>
        <small>{{ pick.label }}</small>
        <em>{{ index + 1 }}</em>
      </span>
      <div class="training-status">
        <strong>策略部署关系</strong>
        <span>连线表示当前策略服务的断联用户节点</span>
      </div>
    </div>

    <div v-if="stage === 'deploy' || stage === 'evaluate'" :class="['deploy-layer', { 'deploy-layer--context': stage === 'evaluate' }]" aria-hidden="true">
      <svg class="deploy-route" viewBox="0 0 100 100" preserveAspectRatio="none">
        <path d="M60 40 C58 45, 56 49, 55 52 C51 53, 47 54, 43 55 C50 60, 57 60, 66 58" />
      </svg>
      <svg class="deploy-service-links" viewBox="0 0 100 100" preserveAspectRatio="none">
        <path class="deploy-service-link deploy-service-link--a" d="M60 40 L66 28" />
        <path class="deploy-service-link deploy-service-link--a" d="M60 40 L61 33" />
        <path class="deploy-service-link deploy-service-link--a" d="M63 30 L64 22" />
        <path class="deploy-service-link deploy-service-link--a" d="M63 30 L72 27" />
        <path class="deploy-service-link deploy-service-link--b" d="M55 52 L49 46" />
        <path class="deploy-service-link deploy-service-link--b" d="M55 52 L51 52" />
        <path class="deploy-service-link deploy-service-link--b" d="M55 52 L57 43" />
        <path class="deploy-service-link deploy-service-link--c" d="M43 55 L31 60" />
        <path class="deploy-service-link deploy-service-link--c" d="M43 55 L24 58" />
        <path class="deploy-service-link deploy-service-link--c" d="M47 67 L47 57" />
        <path class="deploy-service-link deploy-service-link--d" d="M66 58 L75 56" />
        <path class="deploy-service-link deploy-service-link--d" d="M72 61 L80 60" />
        <path class="deploy-service-link deploy-service-link--d" d="M47 67 L51 75" />
      </svg>
      <span
        v-for="node in deploymentNodes"
        :key="node.label"
        class="deployment-node"
        :style="{ '--x': node.x, '--y': node.y, '--delay': node.delay, '--range': node.range }"
      >
        <i>{{ node.short }}</i>
        <small>{{ node.label }}</small>
      </span>
      <div v-if="stage === 'deploy'" class="deploy-status">
        <strong>组网回放中</strong>
        <span>按策略顺序部署应急节点并扩展覆盖</span>
      </div>
    </div>

    <div v-if="stage === 'evaluate'" class="evaluation-layer" aria-hidden="true">
      <svg class="evaluation-links" viewBox="0 0 100 100" preserveAspectRatio="none">
        <path class="evaluation-link evaluation-link--good" d="M60 40 C65 42, 68 48, 66 58" />
        <path class="evaluation-link evaluation-link--good" d="M55 52 C58 55, 62 57, 66 58" />
        <path class="evaluation-link evaluation-link--warn" d="M43 55 C38 58, 36 62, 36 67" />
        <path class="evaluation-link evaluation-link--good" d="M63 30 C66 39, 69 50, 72 61" />
        <path class="evaluation-link evaluation-link--good" d="M60 40 C54 48, 50 58, 47 67" />
        <path class="evaluation-link evaluation-link--warn" d="M47 67 C56 68, 67 66, 72 61" />
      </svg>
      <span
        v-for="metric in linkMetrics"
        :key="metric.label"
        class="link-metric"
        :style="{ '--x': metric.x, '--y': metric.y, '--delay': metric.delay }"
      >
        <strong>{{ metric.value }}</strong>
        <small>{{ metric.label }}</small>
      </span>
      <span class="broadcast-fan"></span>
      <div class="evaluation-status">
        <strong>链路评估完成</strong>
        <span>吞吐、时延与广播覆盖已回写</span>
      </div>
    </div>

    <div class="satellite-board__hud">
      <div class="satellite-board__title">
        <span>SCENARIO 01</span>
        <strong>Guangzhou Urban Flooding</strong>
      </div>
      <div class="satellite-board__status">
        <span><i class="swatch swatch--critical"></i>重灾</span>
        <span><i class="swatch swatch--warning"></i>中灾</span>
        <span><i class="line"></i>主要河道</span>
      </div>
    </div>

    <div class="satellite-board__rail">
      <span v-for="item in monitorStats" :key="item.label">
        <small>{{ item.label }}</small>
        <strong>{{ item.value }}</strong>
      </span>
    </div>

    <div class="satellite-board__corner">
      <span>标准底图</span>
      <span>区划叠加</span>
      <span>灾情覆盖</span>
    </div>

    <div class="satellite-board__legend">
      <span v-for="item in pointLegend" :key="item.label">
        <i class="legend-pin" :style="{ '--pin-color': item.color }"></i>
        <small>{{ item.label }}</small>
      </span>
    </div>

    <div class="satellite-board__meta">
      <div class="map-scale">
        <span></span>
        <small>2 km</small>
      </div>
      <div class="map-coords">
        <span>N23.1291 / E113.2644</span>
        <span>EPSG:4326</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from "vue";
import L from "leaflet";

const props = defineProps({
  stage: {
    type: String,
    default: "intake",
  },
});

const stageOrder = ["intake", "sites", "training", "deploy", "evaluate"];
const stageContextLevel = computed(() => Math.max(1, stageOrder.indexOf(props.stage) + 1));

const mapEl = ref(null);
let map = null;

const monitorStats = [
  { label: "受灾单元", value: "17" },
  { label: "阻断路段", value: "6" },
  { label: "恢复节点", value: "32" },
  { label: "广播覆盖", value: "81%" },
];

const pointLegend = [
  { label: "指挥区", color: "#38bdf8" },
  { label: "居民区", color: "#f97316" },
  { label: "安置点", color: "#eab308" },
  { label: "通道", color: "#22c55e" },
];

const outageAreas = [
  { label: "核心断联区", x: "64%", y: "25%", delay: "1.35s" },
  { label: "居民断联片区", x: "32%", y: "55%", delay: "1.55s" },
  { label: "重点保障用户", x: "50%", y: "43%", delay: "1.75s" },
];

const outageNodes = [
  { x: "57%", y: "26%", delay: "0.95s", level: "high" },
  { x: "60%", y: "24%", delay: "1s", level: "high" },
  { x: "63%", y: "26%", delay: "1.05s", level: "high" },
  { x: "66%", y: "28%", delay: "1.1s", level: "high" },
  { x: "69%", y: "31%", delay: "1.15s", level: "high" },
  { x: "65%", y: "34%", delay: "1.2s", level: "high" },
  { x: "61%", y: "33%", delay: "1.25s", level: "high", served: true },
  { x: "58%", y: "36%", delay: "1.3s", level: "normal" },
  { x: "71%", y: "37%", delay: "1.35s", level: "normal" },
  { x: "54%", y: "31%", delay: "1.4s", level: "normal" },
  { x: "52%", y: "44%", delay: "1.45s", level: "priority" },
  { x: "49%", y: "46%", delay: "1.5s", level: "priority", served: true },
  { x: "46%", y: "49%", delay: "1.55s", level: "priority" },
  { x: "51%", y: "52%", delay: "1.6s", level: "priority", served: true },
  { x: "55%", y: "50%", delay: "1.65s", level: "normal" },
  { x: "43%", y: "52%", delay: "1.7s", level: "normal" },
  { x: "39%", y: "56%", delay: "1.75s", level: "normal" },
  { x: "35%", y: "57%", delay: "1.8s", level: "normal" },
  { x: "31%", y: "60%", delay: "1.85s", level: "normal", served: true },
  { x: "28%", y: "63%", delay: "1.9s", level: "normal" },
  { x: "33%", y: "66%", delay: "1.95s", level: "normal" },
  { x: "37%", y: "68%", delay: "2s", level: "normal" },
  { x: "42%", y: "70%", delay: "2.05s", level: "normal" },
  { x: "46%", y: "72%", delay: "2.1s", level: "normal" },
  { x: "30%", y: "55%", delay: "2.15s", level: "low" },
  { x: "36%", y: "62%", delay: "2.2s", level: "low" },
  { x: "41%", y: "60%", delay: "2.25s", level: "low" },
  { x: "68%", y: "48%", delay: "2.3s", level: "low" },
  { x: "72%", y: "52%", delay: "2.35s", level: "low" },
  { x: "75%", y: "56%", delay: "2.4s", level: "low", served: true },
  { x: "70%", y: "60%", delay: "2.45s", level: "low" },
  { x: "78%", y: "50%", delay: "2.5s", level: "low" },
  { x: "64%", y: "22%", delay: "2.55s", level: "high" },
  { x: "72%", y: "27%", delay: "2.6s", level: "high" },
  { x: "74%", y: "34%", delay: "2.65s", level: "normal" },
  { x: "57%", y: "43%", delay: "2.7s", level: "priority", served: true },
  { x: "47%", y: "57%", delay: "2.75s", level: "priority", served: true },
  { x: "24%", y: "58%", delay: "2.8s", level: "normal" },
  { x: "27%", y: "69%", delay: "2.85s", level: "low" },
  { x: "51%", y: "75%", delay: "2.9s", level: "low" },
  { x: "80%", y: "60%", delay: "2.95s", level: "low" },
  { x: "76%", y: "66%", delay: "3s", level: "low" },
];

const residualStations = [
  { label: "残余宏站", status: "online", x: "58%", y: "42%", delay: "2.3s" },
  { label: "残余宏站", status: "online", x: "72%", y: "30%", delay: "2.55s" },
  { label: "失效宏站", status: "offline", x: "47%", y: "37%", delay: "2.75s" },
  { label: "失效宏站", status: "offline", x: "30%", y: "58%", delay: "2.95s" },
  { label: "弱覆盖站", status: "weak", x: "67%", y: "56%", delay: "3.15s" },
];

const priorityZones = [
  { label: "指挥点", short: "指", tone: "#38bdf8", x: "70%", y: "45%", delay: "3.35s" },
  { label: "安置点", short: "安", tone: "#eab308", x: "42%", y: "70%", delay: "3.55s" },
  { label: "医院", short: "医", tone: "#ef4444", x: "54%", y: "59%", delay: "3.75s" },
  { label: "交通通道", short: "通", tone: "#22c55e", x: "78%", y: "63%", delay: "3.95s" },
];

const intakeStatus = ["网格载入完成", "残余网络扫描完成", "重点区域识别完成"];

const candidateSites = [
  { label: "高优先站点", short: "优", priority: "high", x: "60%", y: "40%", delay: "0.75s" },
  { label: "高优先站点", short: "优", priority: "high", x: "70%", y: "39%", delay: "0.92s" },
  { label: "高优先站点", short: "优", priority: "high", x: "43%", y: "55%", delay: "1.08s" },
  { label: "中继候选", short: "中", priority: "relay", x: "55%", y: "52%", delay: "1.26s" },
  { label: "容量补点", short: "容", priority: "normal", x: "36%", y: "67%", delay: "1.42s" },
  { label: "容量补点", short: "容", priority: "normal", x: "66%", y: "58%", delay: "1.58s" },
  { label: "边缘补盲", short: "补", priority: "low", x: "77%", y: "54%", delay: "1.72s" },
  { label: "边缘补盲", short: "补", priority: "low", x: "31%", y: "52%", delay: "1.86s" },
  { label: "应急补点", short: "急", priority: "normal", x: "63%", y: "30%", delay: "2s" },
  { label: "边缘补盲", short: "补", priority: "low", x: "47%", y: "67%", delay: "2.14s" },
  { label: "中继候选", short: "中", priority: "relay", x: "72%", y: "61%", delay: "2.28s" },
];

const policyPicks = [
  { label: "背负基站", short: "背", x: "60%", y: "40%", delay: "0.25s" },
  { label: "中继节点", short: "中", x: "55%", y: "52%", delay: "0.62s" },
  { label: "小型基站", short: "小", x: "43%", y: "55%", delay: "0.98s" },
  { label: "容量节点", short: "容", x: "66%", y: "58%", delay: "1.34s" },
];

const deploymentNodes = [
  { label: "背负式基站", short: "背", x: "60%", y: "40%", range: "96px", delay: "0.2s" },
  { label: "中继节点", short: "中", x: "55%", y: "52%", range: "112px", delay: "0.72s" },
  { label: "小型基站", short: "小", x: "43%", y: "55%", range: "104px", delay: "1.18s" },
  { label: "容量节点", short: "容", x: "66%", y: "58%", range: "92px", delay: "1.58s" },
  { label: "应急节点", short: "急", x: "63%", y: "30%", range: "84px", delay: "1.95s" },
  { label: "边缘补盲", short: "补", x: "47%", y: "67%", range: "78px", delay: "2.25s" },
];

const linkMetrics = [
  { label: "吞吐", value: "42Mbps", x: "78%", y: "36%", delay: "0.5s" },
  { label: "时延", value: "43ms", x: "72%", y: "68%", delay: "0.8s" },
  { label: "广播", value: "88%", x: "28%", y: "76%", delay: "1.1s" },
];

const districtPolygons = [
  {
    name: "荔湾区",
    color: "#f97316",
    center: [23.126, 113.238],
    points: [
      [23.146, 113.216],
      [23.152, 113.238],
      [23.136, 113.257],
      [23.115, 113.252],
      [23.108, 113.228],
    ],
  },
  {
    name: "天河区",
    color: "#38bdf8",
    center: [23.132, 113.361],
    points: [
      [23.158, 113.322],
      [23.162, 113.382],
      [23.138, 113.408],
      [23.102, 113.394],
      [23.098, 113.338],
    ],
  },
  {
    name: "海珠区",
    color: "#22c55e",
    center: [23.092, 113.317],
    points: [
      [23.118, 113.274],
      [23.126, 113.336],
      [23.094, 113.362],
      [23.064, 113.344],
      [23.056, 113.286],
    ],
  },
  {
    name: "番禺通道",
    color: "#eab308",
    center: [23.008, 113.383],
    points: [
      [23.046, 113.334],
      [23.046, 113.430],
      [22.99, 113.456],
      [22.96, 113.38],
    ],
  },
];

const impactAreas = [
  {
    color: "#ef4444",
    points: [
      [23.145, 113.338],
      [23.154, 113.385],
      [23.126, 113.404],
      [23.102, 113.368],
    ],
  },
  {
    color: "#f59e0b",
    points: [
      [23.082, 113.224],
      [23.102, 113.266],
      [23.076, 113.292],
      [23.044, 113.248],
    ],
  },
  {
    color: "#60a5fa",
    points: [
      [23.118, 113.286],
      [23.13, 113.326],
      [23.108, 113.344],
      [23.084, 113.312],
    ],
  },
];

const riverLines = [
  [
    [23.155, 113.19],
    [23.146, 113.234],
    [23.128, 113.279],
    [23.113, 113.327],
    [23.103, 113.378],
    [23.09, 113.43],
  ],
  [
    [23.075, 113.255],
    [23.086, 113.296],
    [23.078, 113.34],
    [23.056, 113.39],
  ],
];

const recoveryCorridor = [
  [23.145, 113.24],
  [23.132, 113.286],
  [23.117, 113.332],
  [23.094, 113.382],
];

const hotspots = [
  { label: "指挥区", color: "#38bdf8", coord: [23.132, 113.35], labelCoord: [23.149, 113.372] },
  { label: "居民区", color: "#f97316", coord: [23.118, 113.24], labelCoord: [23.096, 113.218] },
  { label: "安置点", color: "#eab308", coord: [23.072, 113.302], labelCoord: [23.053, 113.324] },
  { label: "通道", color: "#22c55e", coord: [23.02, 113.392], labelCoord: [22.998, 113.414] },
];

const makeLabelIcon = (title, note, tone, emphasis = false) =>
  L.divIcon({
    className: "satellite-map__label-wrap",
    html: `<div class="satellite-map__label ${emphasis ? "satellite-map__label--major" : ""}" style="--tone:${tone}">
      <strong>${title}</strong>
      ${note ? `<small>${note}</small>` : ""}
    </div>`,
    iconSize: emphasis ? [90, 28] : [74, 24],
    iconAnchor: emphasis ? [45, 14] : [37, 12],
  });

const makePulseIcon = (tone) =>
  L.divIcon({
    className: "satellite-map__pulse-wrap",
    html: `<div class="satellite-map__pulse" style="--tone:${tone}"><span></span></div>`,
    iconSize: [28, 28],
    iconAnchor: [14, 14],
  });

onMounted(() => {
  map = L.map(mapEl.value, {
    zoomControl: false,
    attributionControl: false,
    scrollWheelZoom: false,
    doubleClickZoom: false,
    boxZoom: false,
    keyboard: false,
    dragging: true,
    zoomSnap: 0.25,
  }).setView([23.11, 113.31], 11.25);

  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    subdomains: ["a", "b", "c"],
  }).addTo(map);

  L.tileLayer(
    "https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
    {
      maxZoom: 18,
      opacity: 0.1,
    }
  ).addTo(map);

  impactAreas.forEach((area) => {
    L.polygon(area.points, {
      color: area.color,
      weight: 1,
      opacity: 0.32,
      fillColor: area.color,
      fillOpacity: 0.12,
      interactive: false,
    }).addTo(map);
  });

  districtPolygons.forEach((district) => {
    L.polygon(district.points, {
      color: district.color,
      weight: 1.35,
      fillColor: district.color,
      fillOpacity: 0.04,
      dashArray: "4 6",
      interactive: false,
    }).addTo(map);

    L.marker(district.center, {
      icon: makeLabelIcon(district.name, "", district.color, district.name !== "番禺通道"),
      interactive: false,
    }).addTo(map);
  });

  riverLines.forEach((line, index) => {
    L.polyline(line, {
      color: index === 0 ? "#38bdf8" : "#7dd3fc",
      weight: index === 0 ? 3.2 : 2,
      opacity: 0.72,
      interactive: false,
    }).addTo(map);
  });

  L.polyline(recoveryCorridor, {
    color: "#22c55e",
    weight: 2.2,
    opacity: 0.58,
    dashArray: "10 8",
      interactive: false,
  }).addTo(map);

  hotspots.forEach((spot) => {
    L.marker(spot.coord, {
      icon: makePulseIcon(spot.color),
      interactive: false,
    }).addTo(map);

    L.marker(spot.labelCoord, {
      icon: makeLabelIcon(spot.label, "", spot.color, false),
      interactive: false,
    }).addTo(map);
  });
});

onBeforeUnmount(() => {
  if (map) {
    map.remove();
    map = null;
  }
});
</script>

<style scoped>
.satellite-board {
  position: relative;
  min-height: 620px;
  border-radius: 8px;
  overflow: hidden;
  border: 1px solid rgba(148, 163, 184, 0.16);
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.48),
    0 20px 40px rgba(15, 23, 42, 0.1);
}

.satellite-board--training :deep(.satellite-map__pulse),
.satellite-board--deploy :deep(.satellite-map__pulse),
.satellite-board--evaluate :deep(.satellite-map__pulse) {
  animation-duration: 1.65s;
}

.satellite-board--intake :deep(.leaflet-overlay-pane path) {
  animation: intakeAreaIn 1.2s ease both;
  transform-box: fill-box;
  transform-origin: center;
}

.satellite-board--intake :deep(.leaflet-marker-pane .satellite-map__pulse-wrap),
.satellite-board--intake :deep(.leaflet-marker-pane .satellite-map__label-wrap) {
  opacity: 0.45;
}

.satellite-board--intake :deep(.leaflet-overlay-pane svg path[stroke="#22c55e"]) {
  opacity: 0.14;
}

.satellite-board--sites :deep(.satellite-map__label),
.satellite-board--training :deep(.satellite-map__label) {
  border-color: rgba(8, 145, 178, 0.34);
  box-shadow: 0 0 18px rgba(8, 145, 178, 0.14);
}

.satellite-board--deploy :deep(.leaflet-overlay-pane path),
.satellite-board--evaluate :deep(.leaflet-overlay-pane path) {
  filter: drop-shadow(0 0 8px rgba(34, 197, 94, 0.16));
}

.satellite-board__map {
  position: absolute;
  inset: 0;
}

.intake-layer {
  position: absolute;
  inset: 0;
  z-index: 505;
  pointer-events: none;
}

.intake-scan {
  position: absolute;
  inset: -18% 0;
  width: 18%;
  transform: skewX(-18deg) translateX(-140%);
  background: linear-gradient(90deg, transparent, rgba(56, 189, 248, 0.2), transparent);
  filter: blur(2px);
  animation: intakeScan 2.4s ease 1.8s both;
}

.intake-station,
.intake-priority,
.intake-outage-area-label {
  position: absolute;
  left: var(--x);
  top: var(--y);
  transform: translate(-50%, -50%);
  opacity: 0;
}

.intake-station small,
.intake-priority small {
  padding: 2px 6px;
  border-radius: 7px;
  background: rgba(255, 255, 255, 0.88);
  color: #0f172a;
  font-size: 11px;
  font-weight: 700;
  text-shadow: none;
  border: 1px solid rgba(148, 163, 184, 0.16);
  box-shadow: 0 5px 12px rgba(15, 23, 42, 0.08);
}

.intake-outage-node {
  position: absolute;
  left: var(--x);
  top: var(--y);
  width: 6px;
  height: 6px;
  border-radius: 999px;
  background: #f87171;
  transform: translate(-50%, -50%);
  opacity: 0;
  box-shadow:
    0 0 0 3px rgba(248, 113, 113, 0.12),
    0 0 10px rgba(239, 68, 68, 0.26);
  animation: intakeMarkerIn 0.42s ease var(--delay) both;
}

.intake-outage-node--high {
  width: 8px;
  height: 8px;
  background: #dc2626;
  box-shadow:
    0 0 0 5px rgba(220, 38, 38, 0.14),
    0 0 15px rgba(220, 38, 38, 0.36);
}

.intake-outage-node--priority {
  width: 8px;
  height: 8px;
  background: #f59e0b;
  box-shadow:
    0 0 0 5px rgba(245, 158, 11, 0.14),
    0 0 14px rgba(245, 158, 11, 0.34);
}

.intake-outage-node--low {
  width: 5px;
  height: 5px;
  background: #fb7185;
  opacity: 0.78;
  box-shadow:
    0 0 0 3px rgba(251, 113, 133, 0.1),
    0 0 8px rgba(251, 113, 133, 0.22);
}

.intake-outage-area-label {
  padding: 4px 8px;
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.88);
  color: #7f1d1d;
  font-size: 11px;
  font-weight: 800;
  border: 1px solid rgba(239, 68, 68, 0.16);
  box-shadow: 0 8px 18px rgba(15, 23, 42, 0.1);
  backdrop-filter: blur(8px);
  animation: intakeMarkerIn 0.5s ease var(--delay) both;
}

.intake-station {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  animation: intakeStationIn 0.44s ease var(--delay) both;
}

.intake-station i {
  position: relative;
  width: 18px;
  height: 18px;
  border-radius: 6px 6px 4px 4px;
  background: #22c55e;
  box-shadow:
    0 0 0 5px rgba(34, 197, 94, 0.14),
    0 0 14px rgba(34, 197, 94, 0.28);
}

.intake-station i::before {
  content: "";
  position: absolute;
  left: 7px;
  top: -8px;
  width: 4px;
  height: 10px;
  border-radius: 999px;
  background: currentColor;
  color: inherit;
}

.intake-station--online i {
  color: #22c55e;
  background: #22c55e;
}

.intake-station--offline i {
  color: #94a3b8;
  background: #94a3b8;
  box-shadow:
    0 0 0 5px rgba(100, 116, 139, 0.12),
    0 0 0 1px rgba(239, 68, 68, 0.35);
}

.intake-station--weak i {
  color: #f59e0b;
  background: #f59e0b;
  box-shadow:
    0 0 0 5px rgba(245, 158, 11, 0.13),
    0 0 14px rgba(245, 158, 11, 0.24);
}

.intake-priority {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  animation: intakePriorityIn 0.5s ease var(--delay) both;
}

.intake-priority i {
  display: grid;
  place-items: center;
  width: 22px;
  height: 22px;
  border-radius: 8px;
  background: var(--tone);
  color: #fff;
  font-size: 11px;
  font-style: normal;
  font-weight: 800;
  box-shadow:
    0 0 0 6px color-mix(in srgb, var(--tone) 16%, transparent),
    0 8px 18px rgba(15, 23, 42, 0.12);
}

.intake-status {
  position: absolute;
  right: 18px;
  bottom: 56px;
  display: grid;
  gap: 6px;
  min-width: 160px;
  padding: 10px 12px;
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.9);
  border: 1px solid rgba(71, 85, 105, 0.18);
  backdrop-filter: blur(10px);
  animation: intakeStatusIn 0.52s ease 4.15s both;
}

.intake-status strong {
  color: #0f172a;
  font-size: 0.88rem;
}

.intake-status span {
  color: #334155;
  font-size: 0.76rem;
  font-weight: 600;
}

.candidate-layer {
  position: absolute;
  inset: 0;
  z-index: 512;
  pointer-events: none;
}

.candidate-links {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  overflow: visible;
}

.candidate-links path {
  fill: none;
  stroke: rgba(8, 145, 178, 0.72);
  stroke-width: 0.36;
  stroke-linecap: round;
  stroke-dasharray: 4 3;
  vector-effect: non-scaling-stroke;
  filter: drop-shadow(0 0 7px rgba(8, 145, 178, 0.22));
  animation: candidateLinkDraw 1.1s ease 0.7s both;
}

.candidate-site {
  position: absolute;
  left: var(--x);
  top: var(--y);
  display: inline-flex;
  align-items: center;
  gap: 7px;
  transform: translate(-50%, -50%);
  opacity: 0;
  animation: candidateSiteIn 0.5s ease var(--delay) both;
}

.candidate-site i {
  display: grid;
  place-items: center;
  width: 28px;
  height: 28px;
  border-radius: 8px;
  background: #0891b2;
  color: #fff;
  font-size: 12px;
  font-style: normal;
  font-weight: 800;
  box-shadow:
    0 0 0 7px rgba(8, 145, 178, 0.14),
    0 10px 20px rgba(15, 23, 42, 0.14);
}

.candidate-site small {
  padding: 4px 8px;
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.9);
  color: #0f172a;
  font-size: 11px;
  font-weight: 800;
  border: 1px solid rgba(8, 145, 178, 0.16);
  box-shadow: 0 8px 18px rgba(15, 23, 42, 0.1);
  backdrop-filter: blur(8px);
}

.candidate-site--high i {
  background: linear-gradient(135deg, #0f172a, #0891b2);
  box-shadow:
    0 0 0 7px rgba(8, 145, 178, 0.18),
    0 0 22px rgba(8, 145, 178, 0.34);
}

.candidate-site--relay i {
  background: linear-gradient(135deg, #2563eb, #14b8a6);
}

.candidate-site--normal i {
  background: #22c55e;
  box-shadow:
    0 0 0 7px rgba(34, 197, 94, 0.14),
    0 10px 20px rgba(15, 23, 42, 0.12);
}

.candidate-site--low i {
  background: #94a3b8;
  box-shadow:
    0 0 0 6px rgba(100, 116, 139, 0.12),
    0 8px 16px rgba(15, 23, 42, 0.1);
}

.candidate-note {
  position: absolute;
  right: 18px;
  bottom: 56px;
  display: grid;
  gap: 5px;
  max-width: 240px;
  padding: 10px 12px;
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.9);
  border: 1px solid rgba(8, 145, 178, 0.16);
  box-shadow: 0 12px 24px rgba(15, 23, 42, 0.1);
  backdrop-filter: blur(10px);
  animation: intakeStatusIn 0.5s ease 1.65s both;
}

.candidate-note strong {
  color: #0f172a;
  font-size: 0.88rem;
}

.candidate-note span {
  color: #334155;
  font-size: 0.74rem;
  line-height: 1.35;
}

.candidate-layer--context {
  opacity: 0.28;
}

.candidate-layer--context .candidate-note {
  display: none;
}

.candidate-layer--context .candidate-site small {
  display: none;
}

.candidate-layer--context .candidate-site--low,
.candidate-layer--context .candidate-site--normal {
  opacity: 0.42;
}

.satellite-board--training .policy-pick small,
.satellite-board--deploy .deployment-node small {
  background: rgba(255, 255, 255, 0.78);
  font-size: 10px;
  padding: 3px 6px;
}

.satellite-board--training .intake-layer--context .intake-outage-area-label,
.satellite-board--training .intake-layer--context .intake-priority small,
.satellite-board--training .intake-layer--context .intake-station small {
  display: none;
}

.satellite-board--training .intake-layer--context .intake-outage-node {
  opacity: 0.32;
}

.satellite-board--training .intake-layer--context .intake-outage-node--high,
.satellite-board--training .intake-layer--context .intake-outage-node--priority {
  opacity: 0.48;
}

.satellite-board--training .intake-layer--context .intake-outage-node--served,
.satellite-board--deploy .intake-layer--context .intake-outage-node--served {
  opacity: 1;
  width: 10px;
  height: 10px;
  background: #f59e0b;
  box-shadow:
    0 0 0 6px rgba(245, 158, 11, 0.18),
    0 0 18px rgba(245, 158, 11, 0.42);
  animation: none;
}

.training-layer,
.deploy-layer,
.evaluation-layer {
  position: absolute;
  inset: 0;
  z-index: 520;
  pointer-events: none;
}

.training-rank-links,
.deploy-route,
.deploy-service-links,
.evaluation-links {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  overflow: visible;
}

.training-service-link {
  fill: none;
  stroke: rgba(8, 145, 178, 0.95);
  stroke-width: 3;
  stroke-linecap: round;
  stroke-dasharray: 10 7;
  vector-effect: non-scaling-stroke;
  filter:
    drop-shadow(0 0 5px rgba(8, 145, 178, 0.52))
    drop-shadow(0 0 12px rgba(8, 145, 178, 0.24));
  opacity: 0;
  animation: serviceLinkIn 0.45s ease 0.22s both;
}

.training-service-link--strong {
  stroke: rgba(34, 197, 94, 0.98);
  filter:
    drop-shadow(0 0 5px rgba(34, 197, 94, 0.55))
    drop-shadow(0 0 12px rgba(34, 197, 94, 0.24));
}

.training-service-link--warn {
  stroke: rgba(245, 158, 11, 0.98);
  filter:
    drop-shadow(0 0 5px rgba(245, 158, 11, 0.5))
    drop-shadow(0 0 12px rgba(245, 158, 11, 0.22));
}

.training-served-user {
  position: absolute;
  left: var(--x);
  top: var(--y);
  width: 10px;
  height: 10px;
  border-radius: 999px;
  background: #f59e0b;
  transform: translate(-50%, -50%);
  opacity: 0;
  box-shadow:
    0 0 0 5px rgba(245, 158, 11, 0.16),
    0 0 14px rgba(245, 158, 11, 0.34);
  animation: trainingUserIn 0.38s ease var(--delay) both;
}

.policy-pick {
  position: absolute;
  left: var(--x);
  top: var(--y);
  display: inline-flex;
  align-items: center;
  gap: 7px;
  transform: translate(-50%, -50%);
  opacity: 0;
  animation: policyPickIn 0.46s ease var(--delay) both;
}

.policy-pick i {
  display: grid;
  place-items: center;
  width: 34px;
  height: 34px;
  border-radius: 8px;
  background: linear-gradient(135deg, #0f172a, #0891b2);
  color: #f8fafc;
  font-size: 12px;
  font-style: normal;
  font-weight: 900;
  box-shadow:
    0 0 0 7px rgba(15, 23, 42, 0.12),
    0 10px 20px rgba(15, 23, 42, 0.2);
}

.policy-pick em {
  position: absolute;
  left: 22px;
  top: -9px;
  display: grid;
  place-items: center;
  width: 18px;
  height: 18px;
  border-radius: 999px;
  background: #f8fafc;
  color: #0f172a;
  font-size: 10px;
  font-style: normal;
  font-weight: 900;
  border: 1px solid rgba(8, 145, 178, 0.18);
  box-shadow: 0 6px 12px rgba(15, 23, 42, 0.12);
}

.policy-pick small,
.deployment-node small,
.link-metric small {
  padding: 4px 8px;
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.92);
  color: #0f172a;
  font-size: 11px;
  font-weight: 800;
  border: 1px solid rgba(15, 23, 42, 0.12);
  box-shadow: 0 8px 18px rgba(15, 23, 42, 0.1);
  backdrop-filter: blur(8px);
}

.training-status,
.deploy-status,
.evaluation-status {
  position: absolute;
  right: 18px;
  bottom: 56px;
  display: grid;
  gap: 5px;
  max-width: 250px;
  padding: 10px 12px;
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.9);
  border: 1px solid rgba(15, 23, 42, 0.12);
  box-shadow: 0 12px 24px rgba(15, 23, 42, 0.1);
  backdrop-filter: blur(10px);
  animation: intakeStatusIn 0.5s ease 1.42s both;
}

.training-status strong,
.deploy-status strong,
.evaluation-status strong {
  color: #0f172a;
  font-size: 0.88rem;
}

.training-status span,
.deploy-status span,
.evaluation-status span {
  color: #334155;
  font-size: 0.74rem;
  line-height: 1.35;
}

.deploy-layer--context {
  opacity: 0.62;
}

.deploy-layer--context .deploy-status {
  display: none;
}

.deploy-route path {
  fill: none;
  stroke: #22c55e;
  stroke-width: 0.55;
  stroke-linecap: round;
  stroke-linejoin: round;
  vector-effect: non-scaling-stroke;
  filter: drop-shadow(0 0 10px rgba(34, 197, 94, 0.28));
  stroke-dasharray: 140;
  stroke-dashoffset: 140;
  animation: deployRouteDraw 1.9s ease 0.35s both;
}

.deploy-service-link {
  fill: none;
  stroke: rgba(34, 197, 94, 0.92);
  stroke-width: 2.8;
  stroke-linecap: round;
  stroke-dasharray: 10 7;
  vector-effect: non-scaling-stroke;
  opacity: 0;
  filter:
    drop-shadow(0 0 5px rgba(34, 197, 94, 0.48))
    drop-shadow(0 0 12px rgba(34, 197, 94, 0.22));
  animation: deployServiceLinkIn 0.5s ease both;
}

.deploy-service-link--a {
  animation-delay: 0.48s, 0.9s;
}

.deploy-service-link--b {
  stroke: rgba(20, 184, 166, 0.92);
  animation-delay: 0.95s, 1.35s;
}

.deploy-service-link--c {
  stroke: rgba(245, 158, 11, 0.92);
  animation-delay: 1.38s, 1.78s;
}

.deploy-service-link--d {
  animation-delay: 1.75s, 2.15s;
}

.deployment-node {
  position: absolute;
  left: var(--x);
  top: var(--y);
  display: inline-flex;
  align-items: center;
  gap: 7px;
  transform: translate(-50%, -50%);
  opacity: 0;
  animation: deployNodeIn 0.45s ease var(--delay) both;
}

.deployment-node::before {
  content: "";
  position: absolute;
  left: 14px;
  top: 14px;
  width: var(--range);
  height: var(--range);
  border-radius: 999px;
  transform: translate(-50%, -50%) scale(0.4);
  background: radial-gradient(circle, rgba(34, 197, 94, 0.18), rgba(34, 197, 94, 0.08) 46%, transparent 70%);
  opacity: 0;
  animation: coverageBloom 1.1s ease calc(var(--delay) + 0.18s) both;
}

.deployment-node i {
  position: relative;
  z-index: 1;
  display: grid;
  place-items: center;
  width: 30px;
  height: 30px;
  border-radius: 8px;
  background: linear-gradient(135deg, #15803d, #22c55e);
  color: #fff;
  font-size: 12px;
  font-style: normal;
  font-weight: 900;
  box-shadow:
    0 0 0 7px rgba(34, 197, 94, 0.14),
    0 10px 20px rgba(15, 23, 42, 0.14);
}

.evaluation-link {
  fill: none;
  stroke-width: 2.2;
  stroke-linecap: round;
  vector-effect: non-scaling-stroke;
  stroke-dasharray: none;
  animation: none;
}

.evaluation-link--good {
  stroke: #22c55e;
  filter: drop-shadow(0 0 9px rgba(34, 197, 94, 0.3));
}

.evaluation-link--warn {
  stroke: #f59e0b;
  filter: drop-shadow(0 0 9px rgba(245, 158, 11, 0.28));
}

.link-metric {
  position: absolute;
  left: var(--x);
  top: var(--y);
  display: inline-flex;
  align-items: center;
  gap: 6px;
  transform: translate(-50%, -50%);
  opacity: 0;
  animation: intakeMarkerIn 0.44s ease var(--delay) both;
}

.link-metric strong {
  padding: 4px 7px;
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.9);
  color: #0f172a;
  font-size: 0.76rem;
  line-height: 1;
  border: 1px solid rgba(34, 197, 94, 0.2);
  box-shadow: 0 8px 16px rgba(15, 23, 42, 0.1);
  backdrop-filter: blur(8px);
}

.link-metric small {
  padding: 3px 6px;
  background: rgba(15, 23, 42, 0.78);
  color: #f8fafc;
  font-size: 10px;
  border-color: rgba(15, 23, 42, 0.12);
  box-shadow: none;
}

.broadcast-fan {
  position: absolute;
  left: 43%;
  top: 55%;
  width: 220px;
  height: 220px;
  transform: translate(-50%, -50%);
  border-radius: 999px;
  background:
    conic-gradient(from 280deg, rgba(56, 189, 248, 0.14), rgba(34, 197, 94, 0.1), transparent 112deg),
    radial-gradient(circle, transparent 0 28%, rgba(56, 189, 248, 0.08) 29%, transparent 62%);
  opacity: 0;
  animation: broadcastFanIn 0.7s ease 0.25s both;
}

.intake-layer--context {
  opacity: 1;
}

.intake-layer--context .intake-outage-node {
  opacity: 0.62;
  animation: contextNodeSettle 0.36s ease both;
}

.intake-layer--context .intake-outage-node--high {
  opacity: 0.72;
}

.intake-layer--context .intake-outage-node--priority {
  opacity: 0.78;
}

.intake-layer--context .intake-outage-area-label {
  opacity: 0.78;
  animation: contextLabelSettle 0.36s ease both;
}

.intake-layer--context .intake-priority,
.intake-layer--context .intake-station {
  opacity: 0.84;
  animation: contextLabelSettle 0.36s ease both;
}

:deep(.leaflet-container) {
  width: 100%;
  height: 100%;
  background: #dbeafe;
  font-family: inherit;
}

:deep(.leaflet-control-container) {
  display: none;
}

.satellite-board__map::after {
  content: "";
  position: absolute;
  inset: 0;
  pointer-events: none;
  background:
    linear-gradient(180deg, rgba(255, 255, 255, 0.08), transparent 20%),
    linear-gradient(0deg, rgba(255, 255, 255, 0.16), transparent 18%);
  z-index: 450;
}

.satellite-board__hud,
.satellite-board__rail,
.satellite-board__corner,
.satellite-board__legend,
.satellite-board__meta {
  position: absolute;
  z-index: 500;
}

.satellite-board__hud {
  top: 12px;
  left: 14px;
  right: 14px;
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: center;
  padding: 8px 12px;
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.44);
  backdrop-filter: blur(8px);
  border: 1px solid rgba(148, 163, 184, 0.1);
}

.satellite-board__title {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.satellite-board__title span {
  color: #64748b;
  font-size: 10px;
  letter-spacing: 0.16em;
}

.satellite-board__title strong {
  color: #0f172a;
  font-size: 0.95rem;
  font-weight: 600;
}

.satellite-board__status {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.satellite-board__status span,
.satellite-board__corner span {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 8px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.66);
  backdrop-filter: blur(8px);
  color: #334155;
  font-size: 10px;
  border: 1px solid rgba(148, 163, 184, 0.12);
}

.swatch,
.line {
  flex: 0 0 auto;
}

.swatch {
  width: 8px;
  height: 8px;
  border-radius: 999px;
}

.swatch--critical {
  background: #ef4444;
}

.swatch--warning {
  background: #f59e0b;
}

.swatch--watch {
  background: #60a5fa;
}

.line {
  width: 14px;
  height: 0;
  border-top: 2px solid #38bdf8;
}

.line--road {
  border-top-style: dashed;
}

.satellite-board__rail {
  left: 14px;
  right: 14px;
  bottom: 42px;
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 8px;
}

.satellite-board__rail span {
  padding: 8px 10px;
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.62);
  backdrop-filter: blur(8px);
  border: 1px solid rgba(148, 163, 184, 0.12);
}

.satellite-board__rail small {
  display: block;
  color: #64748b;
  font-size: 10px;
  margin-bottom: 4px;
}

.satellite-board__rail strong {
  color: #0f172a;
  font-size: 13px;
}

.satellite-board__corner {
  right: 14px;
  bottom: 106px;
  display: flex;
  flex-direction: column;
  align-items: end;
  gap: 6px;
}

.satellite-board__legend {
  left: 14px;
  bottom: 106px;
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  max-width: 56%;
}

.satellite-board__legend span {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  padding: 6px 9px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.64);
  backdrop-filter: blur(8px);
  border: 1px solid rgba(148, 163, 184, 0.12);
  color: #334155;
}

.satellite-board__legend small {
  font-size: 10px;
}

.legend-pin {
  width: 10px;
  height: 10px;
  border-radius: 999px;
  background: var(--pin-color);
  box-shadow:
    0 0 0 3px color-mix(in srgb, var(--pin-color) 18%, transparent),
    0 0 12px color-mix(in srgb, var(--pin-color) 42%, transparent);
}

.satellite-board__meta {
  left: 14px;
  right: 14px;
  bottom: 10px;
  display: flex;
  justify-content: space-between;
  align-items: end;
  gap: 12px;
}

.map-scale {
  display: flex;
  flex-direction: column;
  gap: 4px;
  color: white;
  text-shadow: 0 1px 2px rgba(15, 23, 42, 0.7);
  font-size: 10px;
}

.map-scale span {
  display: block;
  width: 84px;
  height: 8px;
  border-left: 2px solid rgba(255, 255, 255, 0.92);
  border-right: 2px solid rgba(255, 255, 255, 0.92);
  border-top: 2px solid rgba(255, 255, 255, 0.92);
}

.map-coords {
  display: flex;
  gap: 10px;
  padding: 6px 9px;
  border-radius: 999px;
  background: rgba(15, 23, 42, 0.3);
  backdrop-filter: blur(8px);
  color: rgba(255, 255, 255, 0.9);
  font-size: 10px;
  letter-spacing: 0.08em;
}

:deep(.satellite-map__pulse-wrap) {
  background: transparent;
  border: none;
}

:deep(.satellite-map__pulse) {
  position: relative;
  width: 20px;
  height: 20px;
  border-radius: 999px;
  background: color-mix(in srgb, var(--tone) 24%, white 76%);
  border: 1px solid color-mix(in srgb, var(--tone) 40%, white 60%);
  box-shadow:
    0 0 0 6px color-mix(in srgb, var(--tone) 16%, transparent),
    0 0 20px color-mix(in srgb, var(--tone) 34%, transparent);
}

:deep(.satellite-map__pulse span) {
  position: absolute;
  inset: 4px;
  border-radius: 999px;
  background: var(--tone);
  box-shadow: 0 0 10px color-mix(in srgb, var(--tone) 46%, transparent);
}

:deep(.satellite-map__label-wrap) {
  background: transparent;
  border: none;
}

:deep(.satellite-map__label) {
  padding: 5px 8px;
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.92);
  backdrop-filter: blur(8px);
  border: 1px solid color-mix(in srgb, var(--tone) 34%, rgba(148, 163, 184, 0.22));
  box-shadow: 0 8px 18px rgba(15, 23, 42, 0.14);
  color: #0f172a;
}

:deep(.satellite-map__label strong) {
  display: block;
  font-size: 11px;
  line-height: 1.15;
  font-weight: 800;
}

:deep(.satellite-map__label small) {
  display: block;
  font-size: 10px;
  color: #334155;
  font-weight: 600;
}

:deep(.satellite-map__label--major) {
  background: rgba(255, 255, 255, 0.82);
}

@keyframes mapPulse {
  0% {
    transform: scale(0.96);
  }
  70% {
    transform: scale(1.08);
  }
  100% {
    transform: scale(0.96);
  }
}

@keyframes intakeAreaIn {
  from {
    opacity: 0;
    transform: scale(0.88);
  }
  to {
    opacity: 1;
    transform: scale(1);
  }
}

@keyframes intakeAreaBreathe {
  0%,
  100% {
    filter: drop-shadow(0 0 0 rgba(239, 68, 68, 0));
  }
  50% {
    filter: drop-shadow(0 0 10px rgba(239, 68, 68, 0.16));
  }
}

@keyframes intakeScan {
  from {
    transform: skewX(-18deg) translateX(-140%);
  }
  to {
    transform: skewX(-18deg) translateX(720%);
  }
}

@keyframes intakeMarkerIn {
  from {
    opacity: 0;
    transform: translate(-50%, -50%) scale(0.72);
  }
  to {
    opacity: 1;
    transform: translate(-50%, -50%) scale(1);
  }
}

@keyframes intakeStationIn {
  from {
    opacity: 0;
    transform: translate(-50%, -50%) translateY(8px);
  }
  to {
    opacity: 1;
    transform: translate(-50%, -50%) translateY(0);
  }
}

@keyframes intakePriorityIn {
  from {
    opacity: 0;
    transform: translate(-50%, -50%) scale(0.82);
  }
  to {
    opacity: 1;
    transform: translate(-50%, -50%) scale(1);
  }
}

@keyframes intakeStatusIn {
  from {
    opacity: 0;
    transform: translateY(8px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@keyframes outageTwinkle {
  0%,
  100% {
    transform: translate(-50%, -50%) scale(1);
    opacity: 0.86;
  }
  50% {
    transform: translate(-50%, -50%) scale(1.28);
    opacity: 1;
  }
}

@keyframes candidateLinkDraw {
  from {
    stroke-dashoffset: 18;
    opacity: 0;
  }
  to {
    stroke-dashoffset: 0;
    opacity: 1;
  }
}

@keyframes candidateSiteIn {
  from {
    opacity: 0;
    transform: translate(-50%, -50%) scale(0.72);
  }
  to {
    opacity: 1;
    transform: translate(-50%, -50%) scale(1);
  }
}

@keyframes candidatePulse {
  0%,
  100% {
    transform: scale(1);
  }
  50% {
    transform: scale(1.08);
  }
}

@keyframes contextNodeSettle {
  from {
    transform: translate(-50%, -50%) scale(1.18);
  }
  to {
    transform: translate(-50%, -50%) scale(1);
  }
}

@keyframes contextLabelSettle {
  from {
    transform: translate(-50%, -50%) translateY(4px);
  }
  to {
    transform: translate(-50%, -50%) translateY(0);
  }
}

@keyframes policyPickIn {
  from {
    opacity: 0;
    transform: translate(-50%, -50%) scale(0.78);
  }
  to {
    opacity: 1;
    transform: translate(-50%, -50%) scale(1);
  }
}

@keyframes trainingUserIn {
  from {
    opacity: 0;
    transform: translate(-50%, -50%) scale(0.58);
  }
  to {
    opacity: 1;
    transform: translate(-50%, -50%) scale(1);
  }
}

@keyframes serviceLinkIn {
  from {
    opacity: 0;
    stroke-dashoffset: 32;
  }
  to {
    opacity: 1;
    stroke-dashoffset: 0;
  }
}

@keyframes trainingLinkFlow {
  to {
    stroke-dashoffset: -24;
  }
}

@keyframes servedUserPulse {
  0%,
  100% {
    transform: translate(-50%, -50%) scale(1);
  }
  50% {
    transform: translate(-50%, -50%) scale(1.22);
  }
}

@keyframes deployRouteDraw {
  to {
    stroke-dashoffset: 0;
  }
}

@keyframes deployNodeIn {
  from {
    opacity: 0;
    transform: translate(-50%, -50%) scale(0.72);
  }
  to {
    opacity: 1;
    transform: translate(-50%, -50%) scale(1);
  }
}

@keyframes deployServiceLinkIn {
  from {
    opacity: 0;
    stroke-dashoffset: 34;
  }
  to {
    opacity: 1;
    stroke-dashoffset: 0;
  }
}

@keyframes coverageBloom {
  from {
    opacity: 0;
    transform: translate(-50%, -50%) scale(0.3);
  }
  to {
    opacity: 1;
    transform: translate(-50%, -50%) scale(1);
  }
}

@keyframes evalFlow {
  to {
    stroke-dashoffset: -28;
  }
}

@keyframes broadcastFanIn {
  from {
    opacity: 0;
    transform: translate(-50%, -50%) scale(0.72);
  }
  to {
    opacity: 1;
    transform: translate(-50%, -50%) scale(1);
  }
}

@media (max-width: 1200px) {
  .satellite-board {
    min-height: 500px;
  }

  .satellite-board__rail {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 720px) {
  .satellite-board {
    min-height: 360px;
  }

  .satellite-board__status,
  .satellite-board__corner,
  .satellite-board__rail,
  .satellite-board__legend {
    display: none;
  }

  .map-coords {
    display: none;
  }
}
</style>
