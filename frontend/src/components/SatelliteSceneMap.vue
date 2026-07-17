<template>
  <div class="satellite-board" :class="`satellite-board--${stage}`">
    <div ref="mapEl" class="satellite-board__map" aria-label="河南南阳暴雨场景卫星道路地图"></div>

    <div v-if="stageContextLevel >= 1" :class="['intake-layer', { 'intake-layer--context': stage !== 'intake' }]" aria-hidden="true">
      <span v-if="stage === 'intake'" class="intake-scan"></span>

      <span
        v-for="node in outageNodes"
        :key="`${node.x}-${node.y}`"
        :class="[
          'intake-outage-node',
          stage === 'intake' || stage === 'sites' || stage === 'training' ? 'intake-outage-node--high' : `intake-outage-node--${node.level || 'normal'}`,
          { 'intake-outage-node--served': (stage === 'training' || stage === 'deploy' || stage === 'evaluate') && isNodeRecovered(node) },
        ]"
        :style="{
          '--x': node.x,
          '--y': node.y,
          '--delay': node.delay,
          '--temp-restore-delay': getTemporaryRestoreDelay(node),
          '--disconnect-delay': getTemporaryDisconnectDelay(node),
          '--restore-delay': getNodeRestoreDelay(node),
        }"
      ></span>

    </div>

    <div v-if="stage === 'sites'" class="candidate-layer" aria-hidden="true">
      <span v-if="stage === 'sites'" class="candidate-selection-scan"></span>
      <svg class="candidate-links" viewBox="0 0 100 100" preserveAspectRatio="none">
        <path d="M58 42 C62 38, 68 36, 72 30" />
        <path d="M49 46 C43 48, 38 52, 34 58" />
        <path d="M54 50 C58 55, 64 58, 70 60" />
      </svg>

      <span
        v-for="site in candidateSites"
        :key="site.label"
        :class="['candidate-site', `node-kind--${site.type}`, { 'candidate-site--selected': site.selected }]"
        :style="{ '--x': site.x, '--y': site.y, '--delay': site.delay }"
      >
        <i></i>
      </span>
    </div>

    <div v-if="stage === 'training'" class="training-layer" aria-hidden="true">
      <svg class="training-rank-links" viewBox="0 0 100 100" preserveAspectRatio="none">
        <path
          v-for="link in trainingLinks"
          :key="link.d"
          class="training-service-link"
          :class="`training-service-link--round-${link.round}`"
          :style="{ '--clear-delay': link.clearDelay || '99s' }"
          :d="link.d"
        />
      </svg>
      <span
        v-for="pick in trainingStations"
        :key="`${pick.round}-${pick.label}`"
        :class="['policy-pick', `node-kind--${pick.type}`]"
        :style="{ '--x': pick.x, '--y': pick.y, '--delay': pick.delay, '--clear-delay': pick.clearDelay || '99s' }"
      >
        <span class="training-deploy-pulse"></span>
        <i></i>
      </span>
    </div>

    <div v-if="stage === 'deploy' || stage === 'evaluate'" :class="['deploy-layer', { 'deploy-layer--context': stage === 'evaluate' }]" aria-hidden="true">
      <svg class="deploy-service-links" viewBox="0 0 100 100" preserveAspectRatio="none">
        <path
          v-for="link in deploymentLinks"
          :key="link.d"
          class="deploy-service-link"
          :d="link.d"
        />
      </svg>
      <span
        v-for="node in deploymentNodes"
        :key="node.label"
        :class="['deployment-node', `node-kind--${node.type}`]"
        :style="{ '--x': node.x, '--y': node.y, '--delay': node.delay, '--range': node.range }"
      >
        <span class="training-deploy-pulse"></span>
        <i></i>
      </span>
    </div>

    <div v-if="stage === 'evaluate'" class="evaluation-metrics-card" aria-label="链路评估性能指标">
      <div class="evaluation-metrics-card__header">
        <span>Link Evaluation</span>
        <strong>性能指标输出</strong>
      </div>
      <div class="evaluation-metrics-card__grid">
        <span v-for="metric in performanceMetrics" :key="metric.label">
          <small>{{ metric.label }}</small>
          <strong>{{ metric.value }}</strong>
        </span>
      </div>
    </div>

    <div class="satellite-board__hud">
      <div class="satellite-board__status">
        <span v-for="item in nodeLegend" :key="item.label">
          <i class="node-swatch" :style="{ '--node-color': item.color }"></i>
          {{ item.label }}
        </span>
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

const nodeLegend = [
  { label: "正常用户节点", color: "#38bdf8" },
  { label: "受灾用户节点", color: "#ef4444" },
  { label: "背负式基站", color: "#f59e0b" },
  { label: "微型基站", color: "#a78bfa" },
  { label: "自组网中继节点", color: "#14b8a6" },
];

const mapEl = ref(null);
let map = null;

const monitorStats = [
  { label: "受灾单元", value: "17" },
  { label: "阻断路段", value: "6" },
  { label: "恢复节点", value: "32" },
  { label: "广播覆盖", value: "81%" },
];

const outageNodes = [
  { x: "57%", y: "26%", delay: "0.95s", level: "high" },
  { x: "60%", y: "24%", delay: "1s", level: "high" },
  { x: "63%", y: "26%", delay: "1.05s", level: "high" },
  { x: "66%", y: "28%", delay: "1.1s", level: "high", served: true, restoreDelay: "1.65s" },
  { x: "69%", y: "31%", delay: "1.15s", level: "high" },
  { x: "65%", y: "34%", delay: "1.2s", level: "high" },
  { x: "61%", y: "33%", delay: "1.25s", level: "high", served: true, restoreDelay: "1.45s" },
  { x: "58%", y: "36%", delay: "1.3s", level: "normal" },
  { x: "71%", y: "37%", delay: "1.35s", level: "normal" },
  { x: "54%", y: "31%", delay: "1.4s", level: "normal" },
  { x: "52%", y: "44%", delay: "1.45s", level: "priority" },
  { x: "49%", y: "46%", delay: "1.5s", level: "priority", served: true, restoreDelay: "3.05s" },
  { x: "46%", y: "49%", delay: "1.55s", level: "priority" },
  { x: "51%", y: "52%", delay: "1.6s", level: "priority", served: true, restoreDelay: "3.25s" },
  { x: "55%", y: "50%", delay: "1.65s", level: "normal", served: true, restoreDelay: "3.45s" },
  { x: "43%", y: "52%", delay: "1.7s", level: "normal" },
  { x: "39%", y: "56%", delay: "1.75s", level: "normal" },
  { x: "35%", y: "57%", delay: "1.8s", level: "normal" },
  { x: "31%", y: "60%", delay: "1.85s", level: "normal", served: true, restoreDelay: "5.05s" },
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
  { x: "75%", y: "56%", delay: "2.4s", level: "low" },
  { x: "70%", y: "60%", delay: "2.45s", level: "low" },
  { x: "78%", y: "50%", delay: "2.5s", level: "low" },
  { x: "64%", y: "22%", delay: "2.55s", level: "high" },
  { x: "72%", y: "27%", delay: "2.6s", level: "high" },
  { x: "74%", y: "34%", delay: "2.65s", level: "normal" },
  { x: "57%", y: "43%", delay: "2.7s", level: "priority", served: true, restoreDelay: "1.85s" },
  { x: "47%", y: "57%", delay: "2.75s", level: "priority", served: true, restoreDelay: "5.25s" },
  { x: "24%", y: "58%", delay: "2.8s", level: "normal", served: true, restoreDelay: "5.45s" },
  { x: "27%", y: "69%", delay: "2.85s", level: "low" },
  { x: "51%", y: "75%", delay: "2.9s", level: "low" },
  { x: "80%", y: "60%", delay: "2.95s", level: "low" },
  { x: "76%", y: "66%", delay: "3s", level: "low" },
  { x: "18%", y: "28%", delay: "3.05s", level: "low" },
  { x: "22%", y: "42%", delay: "3.1s", level: "normal" },
  { x: "17%", y: "76%", delay: "3.15s", level: "low" },
  { x: "36%", y: "18%", delay: "3.2s", level: "normal" },
  { x: "84%", y: "24%", delay: "3.25s", level: "high" },
  { x: "88%", y: "44%", delay: "3.3s", level: "low" },
  { x: "86%", y: "74%", delay: "3.35s", level: "normal" },
  { x: "58%", y: "84%", delay: "3.4s", level: "priority" },
];

const nodePositionKey = (node) => `${node.x},${node.y}`;
const finalRecoveryDelayByNode = {
  "57%,26%": "5.62s",
  "60%,24%": "5.66s",
  "63%,26%": "5.7s",
  "66%,28%": "5.74s",
  "69%,31%": "5.78s",
  "65%,34%": "5.82s",
  "61%,33%": "5.86s",
  "58%,36%": "5.9s",
  "71%,37%": "5.94s",
  "54%,31%": "5.98s",
  "52%,44%": "6.02s",
  "49%,46%": "6.06s",
  "46%,49%": "6.1s",
  "51%,52%": "6.14s",
  "55%,50%": "6.18s",
  "43%,52%": "6.22s",
  "39%,56%": "6.26s",
  "35%,57%": "6.3s",
  "31%,60%": "6.34s",
  "28%,63%": "6.38s",
  "33%,66%": "6.42s",
  "37%,68%": "6.46s",
  "42%,70%": "6.5s",
  "46%,72%": "6.54s",
  "30%,55%": "6.58s",
  "36%,62%": "6.62s",
  "41%,60%": "6.66s",
  "68%,48%": "6.7s",
  "72%,52%": "6.74s",
  "75%,56%": "6.78s",
  "70%,60%": "6.82s",
  "78%,50%": "6.86s",
  "64%,22%": "6.9s",
  "72%,27%": "6.94s",
  "74%,34%": "6.98s",
  "57%,43%": "7.02s",
  "47%,57%": "7.06s",
  "24%,58%": "7.1s",
  "27%,69%": "7.14s",
  "51%,75%": "7.18s",
  "80%,60%": "7.22s",
  "76%,66%": "7.26s",
  "18%,28%": "7.3s",
  "22%,42%": "7.34s",
  "17%,76%": "7.38s",
  "36%,18%": "7.42s",
  "84%,24%": "7.46s",
  "88%,44%": "7.5s",
  "86%,74%": "7.54s",
  "58%,84%": "7.58s",
};

const temporaryRecoveryPlan = {
  "66%,28%": { restore: "1.45s", disconnect: "2.45s" },
  "61%,33%": { restore: "1.55s", disconnect: "2.45s" },
  "57%,43%": { restore: "1.65s", disconnect: "2.45s" },
  "49%,46%": { restore: "1.75s", disconnect: "2.45s" },
  "51%,52%": { restore: "1.85s", disconnect: "2.45s" },
  "55%,50%": { restore: "1.95s", disconnect: "2.45s" },
  "31%,60%": { restore: "2.05s", disconnect: "2.45s" },
  "24%,58%": { restore: "2.15s", disconnect: "2.45s" },
  "47%,57%": { restore: "2.25s", disconnect: "2.45s" },
  "57%,26%": { restore: "3.45s", disconnect: "4.65s" },
  "60%,24%": { restore: "3.52s", disconnect: "4.65s" },
  "63%,26%": { restore: "3.59s", disconnect: "4.65s" },
  "69%,31%": { restore: "3.66s", disconnect: "4.65s" },
  "65%,34%": { restore: "3.73s", disconnect: "4.65s" },
  "52%,44%": { restore: "3.8s", disconnect: "4.65s" },
  "46%,49%": { restore: "3.87s", disconnect: "4.65s" },
  "43%,52%": { restore: "3.94s", disconnect: "4.65s" },
  "58%,36%": { restore: "4.01s", disconnect: "4.65s" },
  "71%,37%": { restore: "4.08s", disconnect: "4.65s" },
  "39%,56%": { restore: "4.15s", disconnect: "4.65s" },
  "35%,57%": { restore: "4.22s", disconnect: "4.65s" },
  "33%,66%": { restore: "4.29s", disconnect: "4.65s" },
  "37%,68%": { restore: "4.36s", disconnect: "4.65s" },
  "42%,70%": { restore: "4.43s", disconnect: "4.65s" },
};

const isNodeRecovered = (node) => Boolean(finalRecoveryDelayByNode[nodePositionKey(node)]);
const getTemporaryRestoreDelay = (node) => temporaryRecoveryPlan[nodePositionKey(node)]?.restore || "99s";
const getTemporaryDisconnectDelay = (node) => temporaryRecoveryPlan[nodePositionKey(node)]?.disconnect || "99s";
const getNodeRestoreDelay = (node) => finalRecoveryDelayByNode[nodePositionKey(node)] || "99s";

const candidateSites = [
  { label: "背负式基站", type: "backpack", x: "60%", y: "40%", delay: "0.45s", selected: true },
  { label: "自组网中继节点", type: "relay", x: "55%", y: "52%", delay: "0.7s", selected: true },
  { label: "微型基站", type: "micro", x: "43%", y: "55%", delay: "0.95s", selected: true },
  { label: "背负式基站", type: "backpack", x: "66%", y: "58%", delay: "1.2s" },
  { label: "自组网中继节点", type: "relay", x: "72%", y: "61%", delay: "1.45s" },
];

const trainingStations = [
  { round: 1, label: "背负式基站", type: "backpack", x: "60%", y: "40%", delay: "0.45s", clearDelay: "2.45s" },
  { round: 1, label: "自组网中继节点", type: "relay", x: "55%", y: "52%", delay: "0.65s", clearDelay: "2.45s" },
  { round: 1, label: "微型基站", type: "micro", x: "43%", y: "55%", delay: "0.85s", clearDelay: "2.45s" },
  { round: 2, label: "背负式基站", type: "backpack", x: "63%", y: "34%", delay: "2.65s", clearDelay: "4.65s" },
  { round: 2, label: "自组网中继节点", type: "relay", x: "52%", y: "45%", delay: "2.85s", clearDelay: "4.65s" },
  { round: 2, label: "微型基站", type: "micro", x: "37%", y: "62%", delay: "3.05s", clearDelay: "4.65s" },
  { round: 3, label: "背负式基站", type: "backpack", x: "72%", y: "36%", delay: "4.85s" },
  { round: 3, label: "自组网中继节点", type: "relay", x: "64%", y: "56%", delay: "5.05s" },
  { round: 3, label: "微型基站", type: "micro", x: "32%", y: "66%", delay: "5.25s" },
];

const trainingLinks = [
  { round: 1, clearDelay: "2.45s", d: "M60 40 C62 35, 64 31, 66 28" },
  { round: 1, clearDelay: "2.45s", d: "M60 40 C60 37, 60 35, 61 33" },
  { round: 1, clearDelay: "2.45s", d: "M60 40 C59 41, 58 42, 57 43" },
  { round: 1, clearDelay: "2.45s", d: "M55 52 C53 50, 51 48, 49 46" },
  { round: 1, clearDelay: "2.45s", d: "M55 52 C54 52, 52 52, 51 52" },
  { round: 1, clearDelay: "2.45s", d: "M55 52 C55 51, 55 50, 55 50" },
  { round: 1, clearDelay: "2.45s", d: "M43 55 C39 57, 35 59, 31 60" },
  { round: 1, clearDelay: "2.45s", d: "M43 55 C37 55, 31 56, 24 58" },
  { round: 1, clearDelay: "2.45s", d: "M43 55 C44 56, 46 56, 47 57" },
  { round: 2, clearDelay: "4.65s", d: "M63 34 C60 31, 58 28, 57 26" },
  { round: 2, clearDelay: "4.65s", d: "M63 34 C62 29, 61 26, 60 24" },
  { round: 2, clearDelay: "4.65s", d: "M63 34 C64 31, 64 28, 63 26" },
  { round: 2, clearDelay: "4.65s", d: "M63 34 C65 32, 67 31, 69 31" },
  { round: 2, clearDelay: "4.65s", d: "M63 34 C64 34, 65 34, 65 34" },
  { round: 2, clearDelay: "4.65s", d: "M52 45 C52 45, 52 44, 52 44" },
  { round: 2, clearDelay: "4.65s", d: "M52 45 C50 46, 48 48, 46 49" },
  { round: 2, clearDelay: "4.65s", d: "M52 45 C50 48, 47 51, 43 52" },
  { round: 2, clearDelay: "4.65s", d: "M52 45 C54 41, 56 38, 58 36" },
  { round: 2, clearDelay: "4.65s", d: "M52 45 C58 42, 65 39, 71 37" },
  { round: 2, clearDelay: "4.65s", d: "M37 62 C37 60, 38 58, 39 56" },
  { round: 2, clearDelay: "4.65s", d: "M37 62 C36 60, 35 58, 35 57" },
  { round: 2, clearDelay: "4.65s", d: "M37 62 C36 64, 34 65, 33 66" },
  { round: 2, clearDelay: "4.65s", d: "M37 62 C37 64, 37 66, 37 68" },
  { round: 2, clearDelay: "4.65s", d: "M37 62 C39 65, 41 67, 42 70" },
  { round: 3, d: "M72 36 C69 29, 66 24, 64 22" },
  { round: 3, d: "M72 36 C72 33, 72 30, 72 27" },
  { round: 3, d: "M72 36 C73 35, 74 35, 74 34" },
  { round: 3, d: "M72 36 C76 31, 80 27, 84 24" },
  { round: 3, d: "M72 36 C78 38, 83 41, 88 44" },
  { round: 3, d: "M72 36 C75 41, 77 46, 78 50" },
  { round: 3, d: "M72 36 C77 47, 79 55, 80 60" },
  { round: 3, d: "M72 36 C79 46, 81 58, 76 66" },
  { round: 3, d: "M72 36 C80 48, 84 62, 86 74" },
  { round: 3, d: "M64 56 C65 53, 67 50, 68 48" },
  { round: 3, d: "M64 56 C67 54, 70 53, 72 52" },
  { round: 3, d: "M64 56 C66 58, 68 59, 70 60" },
  { round: 3, d: "M64 56 C68 55, 72 55, 75 56" },
  { round: 3, d: "M64 56 C63 66, 60 76, 58 84" },
  { round: 3, d: "M64 56 C59 62, 55 69, 51 75" },
  { round: 3, d: "M32 66 C31 65, 29 64, 28 63" },
  { round: 3, d: "M32 66 C31 62, 31 58, 30 55" },
  { round: 3, d: "M32 66 C33 64, 35 63, 36 62" },
  { round: 3, d: "M32 66 C35 63, 38 61, 41 60" },
  { round: 3, d: "M32 66 C30 67, 28 68, 27 69" },
  { round: 3, d: "M32 66 C27 70, 22 74, 17 76" },
  { round: 3, d: "M32 66 C24 55, 20 42, 18 28" },
  { round: 3, d: "M32 66 C27 59, 23 50, 22 42" },
  { round: 3, d: "M32 66 C31 48, 33 31, 36 18" },
  { round: 3, d: "M64 56 C58 52, 55 42, 54 31" },
  { round: 3, d: "M64 56 C56 61, 50 67, 46 72" },
];

const deploymentNodes = [
  { label: "背负式基站", type: "backpack", x: "72%", y: "36%", range: "104px", delay: "0.3s" },
  { label: "自组网中继节点", type: "relay", x: "64%", y: "56%", range: "118px", delay: "0.5s" },
  { label: "微型基站", type: "micro", x: "32%", y: "66%", range: "96px", delay: "0.7s" },
];

const deploymentLinks = trainingLinks
  .filter((link) => link.round === 3)
  .map((link) => ({ d: link.d }));

const performanceMetrics = [
  { label: "平均吞吐", value: "42 Mbps" },
  { label: "端到端时延", value: "43 ms" },
  { label: "广播覆盖", value: "88%" },
  { label: "恢复用户", value: "318 / 342" },
];

const districtPolygons = [
  {
    name: "卧龙区",
    color: "#f97316",
    center: [32.995, 112.515],
    points: [
      [33.04, 112.45],
      [33.03, 112.58],
      [32.99, 112.62],
      [32.94, 112.56],
      [32.95, 112.46],
    ],
  },
  {
    name: "宛城区",
    color: "#38bdf8",
    center: [32.99, 112.57],
    points: [
      [33.04, 112.54],
      [33.03, 112.67],
      [32.97, 112.7],
      [32.93, 112.6],
      [32.95, 112.52],
    ],
  },
  {
    name: "镇平县",
    color: "#22c55e",
    center: [33.04, 112.24],
    points: [
      [33.12, 112.14],
      [33.11, 112.35],
      [33.02, 112.4],
      [32.95, 112.28],
      [32.99, 112.12],
    ],
  },
  {
    name: "唐白河平原通道",
    color: "#eab308",
    center: [32.77, 112.83],
    points: [
      [32.9, 112.64],
      [32.86, 112.92],
      [32.67, 113.02],
      [32.58, 112.76],
    ],
  },
];

const impactAreas = [
  {
    color: "#ef4444",
    points: [
      [33.18, 112.18],
      [33.16, 112.43],
      [33.03, 112.42],
      [33.0, 112.21],
    ],
  },
  {
    color: "#f59e0b",
    points: [
      [33.02, 112.46],
      [33.04, 112.65],
      [32.92, 112.68],
      [32.9, 112.49],
    ],
  },
  {
    color: "#60a5fa",
    points: [
      [32.9, 112.6],
      [32.82, 112.83],
      [32.68, 112.92],
      [32.64, 112.7],
    ],
  },
];

const riverLines = [
  [
    [33.16, 112.3],
    [33.08, 112.42],
    [32.99, 112.54],
    [32.88, 112.68],
    [32.78, 112.82],
    [32.66, 112.98],
  ],
  [
    [32.85, 111.95],
    [32.84, 112.12],
    [32.8, 112.28],
    [32.76, 112.46],
  ],
];

const recoveryCorridor = [
  [33.12, 112.25],
  [33.05, 112.42],
  [32.98, 112.55],
  [32.84, 112.76],
  [32.72, 112.92],
];

onMounted(() => {
  map = L.map(mapEl.value, {
    zoomControl: false,
    attributionControl: false,
    scrollWheelZoom: false,
    touchZoom: false,
    doubleClickZoom: false,
    boxZoom: false,
    keyboard: false,
    dragging: false,
    tap: false,
    zoomSnap: 0.25,
  }).setView([32.99, 112.54], 10.25);

  L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
    maxZoom: 19,
    subdomains: ["a", "b", "c"],
    crossOrigin: true,
  }).addTo(map);

  L.tileLayer(
    "https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
    {
      maxZoom: 18,
      opacity: 0.18,
    }
  ).addTo(map);

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
  border: 1px solid rgba(148, 163, 184, 0.18);
  background: #0f172a;
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.08),
    0 24px 48px rgba(2, 6, 23, 0.24);
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
  background: rgba(15, 23, 42, 0.78);
  color: #e2e8f0;
  font-size: 11px;
  font-weight: 700;
  text-shadow: 0 1px 2px rgba(2, 6, 23, 0.64);
  border: 1px solid rgba(148, 163, 184, 0.24);
  box-shadow: 0 8px 18px rgba(2, 6, 23, 0.22);
}

.intake-outage-node {
  position: absolute;
  left: var(--x);
  top: var(--y);
  width: 6px;
  height: 6px;
  border-radius: 999px;
  background: #38bdf8;
  transform: translate(-50%, -50%);
  opacity: 0;
  box-shadow:
    0 0 0 3px rgba(56, 189, 248, 0.14),
    0 0 10px rgba(56, 189, 248, 0.28);
  animation: intakeMarkerIn 0.42s ease var(--delay) both;
}

.intake-outage-node--high {
  width: 8px;
  height: 8px;
  background: #ef4444;
  box-shadow:
    0 0 0 5px rgba(239, 68, 68, 0.16),
    0 0 15px rgba(239, 68, 68, 0.36);
}

.intake-outage-node--priority {
  width: 8px;
  height: 8px;
  background: #ef4444;
  box-shadow:
    0 0 0 5px rgba(239, 68, 68, 0.16),
    0 0 14px rgba(239, 68, 68, 0.34);
}

.intake-outage-node--low {
  width: 5px;
  height: 5px;
  background: #38bdf8;
  opacity: 0.78;
  box-shadow:
    0 0 0 3px rgba(56, 189, 248, 0.1),
    0 0 8px rgba(56, 189, 248, 0.22);
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
  color: #22c55e;
  background: #22c55e;
  opacity: 0.42;
  box-shadow:
    0 0 0 5px rgba(34, 197, 94, 0.1),
    0 0 0 1px rgba(34, 197, 94, 0.28);
}

.intake-station--weak i {
  color: #22c55e;
  background: #22c55e;
  opacity: 0.68;
  box-shadow:
    0 0 0 5px rgba(34, 197, 94, 0.13),
    0 0 14px rgba(34, 197, 94, 0.24);
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
  stroke: rgba(20, 184, 166, 0.72);
  stroke-width: 0.42;
  stroke-linecap: round;
  stroke-dasharray: 7 5;
  vector-effect: non-scaling-stroke;
  filter: drop-shadow(0 0 9px rgba(20, 184, 166, 0.28));
  animation: candidateLinkDraw 1.3s ease 1.55s both;
}

.candidate-selection-scan {
  position: absolute;
  left: 57%;
  top: 50%;
  width: 220px;
  height: 220px;
  border-radius: 999px;
  transform: translate(-50%, -50%) scale(0.26);
  border: 1px solid rgba(56, 189, 248, 0.58);
  background: radial-gradient(circle, rgba(56, 189, 248, 0.12), transparent 62%);
  opacity: 0;
  animation: candidateSelectionScan 1.45s ease 0.65s both;
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

.satellite-board--sites .candidate-site:not(.candidate-site--selected) {
  animation: candidateSiteIn 0.5s ease var(--delay) both, candidateRejected 0.44s ease 1.95s forwards;
}

.candidate-site--selected::before {
  content: "";
  position: absolute;
  left: 50%;
  top: 50%;
  width: 38px;
  height: 38px;
  border-radius: 999px;
  transform: translate(-50%, -50%) scale(0.4);
  border: 1px solid color-mix(in srgb, var(--node-tone) 58%, white 12%);
  background: radial-gradient(circle, color-mix(in srgb, var(--node-tone) 18%, transparent), transparent 64%);
  opacity: 0;
  animation: candidateSelectRing 0.78s ease 1.72s forwards, candidateSelectPulse 1.4s ease 2.5s infinite;
}

.candidate-site i {
  display: grid;
  place-items: center;
  width: 18px;
  height: 18px;
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

.node-kind--normal-user {
  --node-tone: #38bdf8;
}

.node-kind--affected-user {
  --node-tone: #ef4444;
}

.node-kind--backpack {
  --node-tone: #f59e0b;
}

.node-kind--micro {
  --node-tone: #a78bfa;
}

.node-kind--relay {
  --node-tone: #14b8a6;
}

.candidate-site.node-kind--backpack i,
.candidate-site.node-kind--micro i,
.candidate-site.node-kind--relay i,
.policy-pick.node-kind--backpack i,
.policy-pick.node-kind--micro i,
.policy-pick.node-kind--relay i,
.deployment-node.node-kind--backpack i,
.deployment-node.node-kind--micro i,
.deployment-node.node-kind--relay i {
  background: var(--node-tone);
  border-radius: 8px;
  box-shadow:
    0 0 0 7px color-mix(in srgb, var(--node-tone) 18%, transparent),
    0 0 22px color-mix(in srgb, var(--node-tone) 36%, transparent);
}

.candidate-site.node-kind--relay i,
.policy-pick.node-kind--relay i,
.deployment-node.node-kind--relay i {
  border-radius: 999px;
}

.candidate-site.node-kind--backpack i,
.candidate-site.node-kind--micro i,
.candidate-site.node-kind--relay i {
  box-shadow:
    0 0 0 5px color-mix(in srgb, var(--node-tone) 14%, transparent),
    0 0 14px color-mix(in srgb, var(--node-tone) 32%, transparent);
}

.candidate-site.node-kind--micro i,
.policy-pick.node-kind--micro i,
.deployment-node.node-kind--micro i {
  border-radius: 4px;
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

.satellite-board--training .intake-layer--context .intake-outage-node--served {
  opacity: 1;
  width: 10px;
  height: 10px;
  background: #ef4444;
  box-shadow:
    0 0 0 5px rgba(239, 68, 68, 0.16),
    0 0 15px rgba(239, 68, 68, 0.36);
  animation:
    restoreUserNode 0.42s ease var(--temp-restore-delay, 99s) forwards,
    disconnectUserNode 0.28s ease var(--disconnect-delay, 99s) forwards,
    restoreUserNode 0.52s ease var(--restore-delay, 6.2s) forwards;
}

.satellite-board--deploy .intake-layer--context .intake-outage-node--served,
.satellite-board--evaluate .intake-layer--context .intake-outage-node--served {
  opacity: 1;
  width: 10px;
  height: 10px;
  background: #38bdf8;
  box-shadow:
    0 0 0 6px rgba(56, 189, 248, 0.18),
    0 0 18px rgba(56, 189, 248, 0.42);
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
.deploy-service-links {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  overflow: visible;
}

.training-service-link {
  fill: none;
  stroke: rgba(125, 211, 252, 0.56);
  stroke-width: 1.35;
  stroke-linecap: round;
  stroke-dasharray: 8 8;
  vector-effect: non-scaling-stroke;
  filter: drop-shadow(0 0 6px rgba(125, 211, 252, 0.18));
  opacity: 0;
  animation:
    serviceLinkIn 0.52s ease var(--link-delay, 0.8s) both,
    trainingLinkClear 0.26s ease var(--clear-delay, 99s) forwards;
}

.training-service-link--round-1 {
  --link-delay: 1.15s;
}

.training-service-link--round-2 {
  --link-delay: 3.25s;
}

.training-service-link--round-3 {
  --link-delay: 5.45s;
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
  animation:
    policyPickIn 0.46s ease var(--delay) both,
    trainingStationClear 0.28s ease var(--clear-delay, 99s) forwards;
}

.training-deploy-pulse {
  position: absolute;
  left: 50%;
  top: 50%;
  width: 42px;
  height: 42px;
  border-radius: 999px;
  transform: translate(-50%, -50%) scale(0.35);
  border: 1px solid color-mix(in srgb, var(--node-tone) 58%, white 12%);
  background: radial-gradient(circle, color-mix(in srgb, var(--node-tone) 16%, transparent), transparent 64%);
  opacity: 0;
}

.training-deploy-pulse {
  animation: trainingDeployRing 0.82s ease 0s forwards;
}

.policy-pick i {
  display: grid;
  place-items: center;
  width: 18px;
  height: 18px;
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
.deployment-node small {
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

.deploy-service-link {
  fill: none;
  stroke: rgba(125, 211, 252, 0.58);
  stroke-width: 1.35;
  stroke-linecap: round;
  stroke-dasharray: 8 8;
  vector-effect: non-scaling-stroke;
  opacity: 0;
  filter: drop-shadow(0 0 6px rgba(125, 211, 252, 0.18));
  animation: deployServiceLinkIn 0.52s ease 1.05s both;
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
  left: 9px;
  top: 9px;
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
  width: 18px;
  height: 18px;
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

.evaluation-metrics-card {
  position: absolute;
  left: 16px;
  bottom: 16px;
  z-index: 640;
  width: min(360px, calc(100% - 32px));
  padding: 14px;
  border-radius: 8px;
  border: 1px solid rgba(148, 163, 184, 0.2);
  background: rgba(15, 23, 42, 0.72);
  backdrop-filter: blur(14px);
  box-shadow: 0 18px 36px rgba(2, 6, 23, 0.26);
  animation: evaluationMetricsIn 0.42s ease 0.18s both;
}

.evaluation-metrics-card__header {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: baseline;
  margin-bottom: 12px;
}

.evaluation-metrics-card__header span {
  color: #94a3b8;
  font-size: 10px;
  letter-spacing: 0.16em;
  text-transform: uppercase;
}

.evaluation-metrics-card__header strong {
  color: #f8fafc;
  font-size: 0.92rem;
}

.evaluation-metrics-card__grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
}

.evaluation-metrics-card__grid span {
  padding-top: 8px;
  border-top: 1px solid rgba(148, 163, 184, 0.22);
}

.evaluation-metrics-card__grid small {
  display: block;
  margin-bottom: 4px;
  color: #94a3b8;
  font-size: 10px;
}

.evaluation-metrics-card__grid strong {
  color: #e0f2fe;
  font-size: 1rem;
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
  background: #0f172a;
  font-family: inherit;
}

:deep(.leaflet-tile-pane) {
  filter: saturate(1.02) contrast(0.98) brightness(1.18);
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
    linear-gradient(180deg, rgba(15, 23, 42, 0.26), transparent 24%),
    radial-gradient(circle at 54% 34%, rgba(56, 189, 248, 0.12), transparent 34%),
    linear-gradient(0deg, rgba(15, 23, 42, 0.32), transparent 24%);
  z-index: 450;
}

.satellite-board__hud,
.satellite-board__rail,
.satellite-board__corner,
.satellite-board__legend,
.satellite-board__meta {
  position: absolute;
  z-index: 640;
}

.satellite-board__hud {
  top: 12px;
  left: 14px;
  max-width: min(560px, calc(100% - 28px));
  display: inline-flex;
  justify-content: flex-start;
  align-items: center;
  width: fit-content;
  padding: 8px;
  border-radius: 14px;
  background: rgba(15, 23, 42, 0.68);
  backdrop-filter: blur(12px);
  border: 1px solid rgba(148, 163, 184, 0.18);
}

.satellite-board__title {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.satellite-board__title span {
  color: #94a3b8;
  font-size: 10px;
  letter-spacing: 0.16em;
}

.satellite-board__title strong {
  color: #e2e8f0;
  font-size: 0.95rem;
  font-weight: 600;
}

.satellite-board__status {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-start;
  gap: 6px;
  width: fit-content;
  max-width: 100%;
}

.satellite-board__status span,
.satellite-board__corner span {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 8px;
  border-radius: 999px;
  background: rgba(15, 23, 42, 0.56);
  backdrop-filter: blur(8px);
  color: #cbd5e1;
  font-size: 10px;
  border: 1px solid rgba(148, 163, 184, 0.18);
}

.swatch,
.line,
.node-swatch {
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

.node-swatch {
  width: 10px;
  height: 10px;
  border-radius: 999px;
  background: var(--node-color);
  box-shadow:
    0 0 0 3px color-mix(in srgb, var(--node-color) 18%, transparent),
    0 0 12px color-mix(in srgb, var(--node-color) 44%, transparent);
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
  background: rgba(15, 23, 42, 0.58);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(148, 163, 184, 0.18);
}

.satellite-board__rail small {
  display: block;
  color: #94a3b8;
  font-size: 10px;
  margin-bottom: 4px;
}

.satellite-board__rail strong {
  color: #f8fafc;
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
  background: rgba(15, 23, 42, 0.56);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(148, 163, 184, 0.18);
  color: #cbd5e1;
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
  background: rgba(15, 23, 42, 0.58);
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
  background: rgba(15, 23, 42, 0.76);
  backdrop-filter: blur(8px);
  border: 1px solid color-mix(in srgb, var(--tone) 34%, rgba(148, 163, 184, 0.22));
  box-shadow: 0 10px 22px rgba(2, 6, 23, 0.26);
  color: #f8fafc;
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
  color: #cbd5e1;
  font-weight: 600;
}

:deep(.satellite-map__label--major) {
  background: rgba(15, 23, 42, 0.84);
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

@keyframes candidateSelectionScan {
  0% {
    opacity: 0;
    transform: translate(-50%, -50%) scale(0.26);
  }
  22% {
    opacity: 0.82;
  }
  100% {
    opacity: 0;
    transform: translate(-50%, -50%) scale(1.12);
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

@keyframes candidateRejected {
  to {
    opacity: 0.3;
    transform: translate(-50%, -50%) scale(0.78);
  }
}

@keyframes candidateSelectRing {
  from {
    opacity: 0;
    transform: translate(-50%, -50%) scale(0.4);
  }
  to {
    opacity: 1;
    transform: translate(-50%, -50%) scale(1);
  }
}

@keyframes candidateSelectPulse {
  0%,
  100% {
    opacity: 0.72;
    transform: translate(-50%, -50%) scale(0.92);
  }
  50% {
    opacity: 1;
    transform: translate(-50%, -50%) scale(1.1);
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

@keyframes trainingStationClear {
  to {
    opacity: 0;
    transform: translate(-50%, -50%) scale(0.68);
  }
}

@keyframes trainingDeployRing {
  0% {
    opacity: 0;
    transform: translate(-50%, -50%) scale(0.35);
  }
  45% {
    opacity: 1;
  }
  100% {
    opacity: 0;
    transform: translate(-50%, -50%) scale(1.16);
  }
}

@keyframes trainingLinkClear {
  to {
    opacity: 0;
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

@keyframes restoreUserNode {
  0% {
    background: #ef4444;
    box-shadow:
      0 0 0 5px rgba(239, 68, 68, 0.16),
      0 0 15px rgba(239, 68, 68, 0.36);
  }
  48% {
    transform: translate(-50%, -50%) scale(1.28);
  }
  100% {
    background: #38bdf8;
    box-shadow:
      0 0 0 6px rgba(56, 189, 248, 0.18),
      0 0 18px rgba(56, 189, 248, 0.42);
  }
}

@keyframes disconnectUserNode {
  to {
    background: #ef4444;
    box-shadow:
      0 0 0 5px rgba(239, 68, 68, 0.16),
      0 0 15px rgba(239, 68, 68, 0.36);
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

@keyframes evaluationMetricsIn {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
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
