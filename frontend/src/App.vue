<template>
  <div class="app-shell">
    <div class="app-shell__aurora app-shell__aurora--left"></div>
    <div class="app-shell__aurora app-shell__aurora--right"></div>

    <header class="topbar">
      <div class="topbar__head">
        <a class="brand" href="#/">
          <span class="brand__mark" aria-hidden="true">
            <i></i>
            <i></i>
            <i></i>
          </span>
          <div class="brand__copy">
            <span class="brand__eyebrow">指挥控制台</span>
            <strong>应急通信数字孪生<span>仿真平台</span></strong>
            <p>灾情接入、场景训练、真实回放与链路评估统一工作台。</p>
          </div>
        </a>
      </div>

      <nav class="topbar__nav" aria-label="主导航">
        <a
          v-for="item in navItems"
          :key="item.key"
          :href="item.href"
          :class="['nav-chip', { 'nav-chip--active': isNavItemActive(item) }]"
        >
          <strong class="nav-chip__label">{{ item.label }}</strong>
        </a>
      </nav>
    </header>

    <main class="app-main">
      <section v-if="currentRoute === 'home'" class="landing-view landing-view--map">
        <div class="map-command-screen" :class="`map-command-screen--${activeMissionStage.key}`">
          <section class="mission-console" aria-label="恢复任务控制台">
            <div class="map-command-screen__header">
              <div>
                <span class="eyebrow">Recovery Mission Console</span>
                <h1>面向灾害场景的多模融合应急通信仿真平台</h1>
                <p>播放一次从灾情接入、策略训练到组网回放的恢复任务。</p>
              </div>
              <div class="map-command-screen__live">
                <span></span>
                <strong>{{ missionPlaying ? "RUNNING" : "READY" }}</strong>
                <small>{{ activeMissionStage.location }}</small>
              </div>
            </div>

            <div class="map-command-screen__status">
              <span v-for="item in activeMissionStage.metrics" :key="item.label">
                <small>{{ item.label }}</small>
                <strong>{{ item.value }}</strong>
              </span>
            </div>

            <div class="map-command-screen__timeline" :style="{ '--mission-progress': missionProgress }">
              <button
                v-for="(step, index) in missionStages"
                :key="step.key"
                type="button"
                :class="[
                  'mission-step',
                  {
                    'mission-step--active': index === activeMissionIndex,
                    'mission-step--complete': index < activeMissionIndex,
                  },
                ]"
                @click="selectMissionStage(index)"
              >
                <i>{{ step.icon }}</i>
                <span class="mission-step__copy">
                  <strong>{{ step.label }}</strong>
                  <small>{{ step.result }}</small>
                </span>
                <em>{{ index < activeMissionIndex ? "已完成" : index === activeMissionIndex ? "进行中" : "待执行" }}</em>
              </button>
            </div>

            <div class="map-command-screen__mission">
              <div class="mission-actions">
                <button type="button" class="primary-cta" @click="playMission">
                  {{ missionPlaying ? "重新播放任务" : "播放恢复任务" }}
                </button>
                <a class="secondary-cta" href="#/algorithm">进入场景&环境导入</a>
              </div>
            </div>
          </section>

          <section class="map-command-screen__stage" aria-label="灾情地图与节点基站信息">
            <SatelliteSceneMap class="map-command-screen__map" :stage="activeMissionStage.key" />

            <div class="mission-path" aria-hidden="true">
              <span class="mission-path__line"></span>
              <span class="mission-path__node mission-path__node--a"></span>
              <span class="mission-path__node mission-path__node--b"></span>
              <span class="mission-path__node mission-path__node--c"></span>
            </div>
          </section>
        </div>
      </section>

      <section v-else-if="currentRoute === 'algorithm'" class="module-view">
        <section class="algorithm-stage panel-shell">
          <ScenarioTrainingPanel />
        </section>
      </section>

      <section v-else-if="currentRoute === 'scene'" class="module-view">
        <Ns3ReplayPanel v-if="sceneTab === 'replay'" />

        <section v-else class="scene-stage panel-shell">
          <div class="scene-stage__intro">
            <div>
              <span class="scene-stage__tag">{{ activeSceneTab.stageTag }}</span>
              <h2>{{ activeSceneTab.heading }}</h2>
              <p>{{ activeSceneTab.intro }}</p>
            </div>
            <div class="scene-stage__chips">
              <span v-for="item in sceneChips" :key="item">{{ item }}</span>
            </div>
          </div>

          <MahimahiSimulator />
        </section>
      </section>

      <section v-else-if="currentRoute === 'tester'" class="module-view">
        <section class="scene-stage panel-shell">
          <CustomEnvironmentTester />
        </section>
      </section>

      <section v-else-if="currentRoute === 'device'" class="module-view">
        <div class="module-hero panel-shell panel-shell--module panel-shell--device">
          <div>
            <span class="eyebrow">Device Operations</span>
            <h1>虚拟设备模拟工作台</h1>
            <p>设备能力与适配场景集中展示。</p>
          </div>
          <div class="module-hero__badges">
            <span>设备图谱</span>
            <span>参数与能力解释</span>
            <span>装备介绍</span>
          </div>
        </div>

        <div class="device-grid">
          <article
            v-for="device in deviceCards"
            :key="device.key"
            class="device-card panel-shell"
            @click="activeDevice = device"
          >
            <div class="device-card__media" :style="{ '--device-tone': device.tone }">
              <span>{{ device.short }}</span>
            </div>
            <div class="device-card__body">
              <div class="device-card__title">
                <strong>{{ device.title }}</strong>
                <small>{{ device.tag }}</small>
              </div>
              <p>{{ device.description }}</p>
              <div class="device-card__stats">
                <span v-for="stat in device.stats" :key="stat">{{ stat }}</span>
              </div>
            </div>
          </article>
        </div>

        <section class="device-showcase panel-shell">
          <div class="device-showcase__header">
            <div>
              <span class="eyebrow">Device Catalog</span>
              <h2>装备能力展示</h2>
              <p>直接展示设备能力、参数说明和典型用途。</p>
            </div>
            <div class="device-showcase__meta">
              <span>场景 {{ selectedScenario ? formatScenarioName(selectedScenario.name) : "加载中" }}</span>
              <span>设备 {{ selectedScenario?.base_stations?.length || 0 }}</span>
              <span>目录 {{ deviceCards.length }}</span>
            </div>
          </div>

          <BaseStationShowcase v-if="selectedScenario" :scenario="selectedScenario" />
          <div v-else class="device-showcase__loading">正在加载默认场景与设备目录…</div>
        </section>
      </section>
    </main>

    <div v-if="activeDevice" class="device-modal" @click.self="activeDevice = null">
      <div class="device-modal__panel panel-shell">
        <button type="button" class="device-modal__close" @click="activeDevice = null">关闭</button>
        <span class="eyebrow">Device Brief</span>
        <h2>{{ activeDevice.title }}</h2>
        <p class="device-modal__desc">{{ activeDevice.longDescription }}</p>

        <div class="device-modal__section">
          <h3>典型用途</h3>
          <p>{{ activeDevice.useCase }}</p>
        </div>

        <div class="device-modal__section">
          <h3>能力说明</h3>
          <div class="device-modal__chips">
            <span v-for="stat in activeDevice.stats" :key="stat">{{ stat }}</span>
          </div>
        </div>

        <div class="device-modal__section">
          <h3>与策略设计关系</h3>
          <p>{{ activeDevice.strategyNote }}</p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from "vue";
import axios from "axios";
import ScenarioTrainingPanel from "./components/ScenarioTrainingPanel.vue";
import Ns3ReplayPanel from "./components/Ns3ReplayPanel.vue";
import MahimahiSimulator from "./components/MahimahiSimulator.vue";
import CustomEnvironmentTester from "./components/CustomEnvironmentTester.vue";
import BaseStationShowcase from "./components/BaseStationShowcase.vue";
import SatelliteSceneMap from "./components/SatelliteSceneMap.vue";
import { rescueApiBase } from "./utils/runtimeEndpoints";
import { formatScenarioName } from "./utils/scenarioLabels";

const navItems = [
  { key: "home", label: "首页", href: "#/" },
  { key: "algorithm", label: "场景&环境导入", href: "#/algorithm" },
  { key: "tester", label: "策略测试", href: "#/tester" },
  { key: "replay", label: "真实回放", href: "#/replay" },
  { key: "mahimahi", label: "链路仿真", href: "#/mahimahi" },
  { key: "device", label: "设备模拟", href: "#/device" },
];

const missionModules = [
  {
    key: "algorithm",
    kicker: "01 / Scene Intake",
    title: "场景&环境导入",
    desc: "录入灾区、设备与 RL 组网算法，再配置高级设置。",
    href: "#/algorithm",
  },
  {
    key: "tester",
    kicker: "02 / Evaluation",
    title: "策略测试",
    desc: "自动匹配权重并验证恢复效果。",
    href: "#/tester",
  },
  {
    key: "scene",
    kicker: "03 / Replay",
    title: "场景模拟",
    desc: "回放 ns-3 与链路仿真过程。",
    href: "#/scene",
  },
];

const missionStages = [
  {
    key: "intake",
    code: "01",
    icon: "灾",
    label: "灾情接入",
    result: "断链区域入库",
    kicker: "Incident Intake",
    title: "导入洪涝灾情与断链区域",
    summary: "读取受灾网格、残余网络和重点保障区域。",
    location: "广州中心城区 / 洪涝场景",
    metrics: [
      { label: "灾情等级", value: "严重" },
      { label: "断联用户", value: "342" },
      { label: "残余基站", value: "7" },
      { label: "广播可达率", value: "24%" },
    ],
  },
  {
    key: "sites",
    code: "02",
    icon: "点",
    label: "候选站点",
    result: "部署点位生成",
    kicker: "Candidate Sites",
    title: "生成应急基站候选部署点",
    summary: "结合人口、道路和设备能力生成部署集合。",
    location: "候选位点 28 / 重点区域 6",
    metrics: [
      { label: "候选站点", value: "28" },
      { label: "设备类型", value: "3" },
      { label: "重点区域", value: "6" },
      { label: "预算上限", value: "12" },
    ],
  },
  {
    key: "training",
    code: "03",
    icon: "训",
    label: "策略训练",
    result: "推荐序列收敛",
    kicker: "RL Policy",
    title: "强化学习策略搜索部署顺序",
    summary: "PPO 迭代选择站点与设备组合。",
    location: "PPO 训练 / 演示权重",
    metrics: [
      { label: "当前策略", value: "PPO" },
      { label: "评估覆盖率", value: "68%" },
      { label: "平均奖励", value: "+42.6" },
      { label: "有效动作", value: "91%" },
    ],
  },
  {
    key: "deploy",
    code: "04",
    icon: "网",
    label: "组网回放",
    result: "节点链路恢复",
    kicker: "Deployment Replay",
    title: "按策略顺序回放应急组网过程",
    summary: "回放基站与中继节点的部署顺序。",
    location: "ns-3 回放 / 快速演示",
    metrics: [
      { label: "恢复进度", value: "68%" },
      { label: "部署节点", value: "12" },
      { label: "恢复用户", value: "233" },
      { label: "活动链路", value: "19" },
    ],
  },
  {
    key: "evaluate",
    code: "05",
    icon: "测",
    label: "链路评估",
    result: "性能指标输出",
    kicker: "Link Evaluation",
    title: "评估吞吐、时延与广播覆盖",
    summary: "输出吞吐、时延和广播覆盖结果。",
    location: "Mahimahi / Link Trace",
    metrics: [
      { label: "覆盖恢复率", value: "81%" },
      { label: "平均时延", value: "43ms" },
      { label: "广播可达率", value: "88%" },
      { label: "任务状态", value: "完成" },
    ],
  },
];

const sceneTabs = [
  {
    key: "replay",
    label: "真实回放",
    desc: "ns-3 组网过程逐帧可视化",
    stageTag: "Replay Stage",
    heading: "真实场景组网回放",
    intro: "回放页以场景过程为主，缺少结果时可直接生成一轮 ns-3 演练。",
    summaryTitle: "场景回放",
    summaryText: "主画布优先，聚焦组网过程。",
    chips: ["主后端统一接入", "支持预置实验", "逐帧场景重建"],
  },
  {
    key: "mahimahi",
    label: "链路仿真",
    desc: "trace 驱动的容量与发送速率分析",
    stageTag: "Link Analysis",
    heading: "Mahimahi 网络链路分析",
    intro: "用 trace 容量、吞吐和发送速率补足场景回放里的链路细节。",
    summaryTitle: "链路仿真",
    summaryText: "补充容量、吞吐、发送速率视角。",
    chips: ["Trace 回放", "发送速率曲线", "容量窗口分析"],
  },
];

const deviceCards = [
  {
    key: "backpack",
    title: "背负式应急基站",
    short: "BP",
    tag: "Mobile Pack",
    tone: "#3b82f6",
    description: "适合灾后快速进场与单兵携行部署，用于盲区接入恢复和应急广播补盲。",
    longDescription: "背负式应急基站强调快速部署和机动接入，在道路受阻、地形破碎或通信链路断裂场景中具有更高适应性。",
    useCase: "适用于狭窄街巷、积水区域、局部断电区域和救援队伍伴随式部署。",
    strategyNote: "在策略设计中通常承担快速恢复覆盖和提升广播可达率的职责，适合作为低成本补盲节点。",
    stats: ["低时延接入", "单兵携行", "快速开站"],
  },
  {
    key: "compact",
    title: "高并发小型基站",
    short: "SC",
    tag: "Compact Cell",
    tone: "#14b8a6",
    description: "适合人群聚集区、临时指挥点和道路交汇点，兼顾覆盖密度和容量恢复。",
    longDescription: "高并发小型基站面向局部高负载区域，在现场指挥、救援协同和视频回传等任务中具备更好的容量支撑能力。",
    useCase: "适用于临时指挥中心、安置点、道路汇聚口和高密用户热点区域。",
    strategyNote: "在奖励设计中通常承担提升吞吐、保障重点区域容量和降低热点拥塞的职责。",
    stats: ["高容量", "热点覆盖", "多接入模式"],
  },
  {
    key: "relay",
    title: "多跳自组网中继",
    short: "RL",
    tag: "Relay Mesh",
    tone: "#f59e0b",
    description: "适用于复杂地形和远距离回传，强调跨障碍、跨区域的链路续接能力。",
    longDescription: "多跳中继设备用于在公网受损或地形阻断条件下续接链路，通过多跳回传维持跨区通信与任务协同。",
    useCase: "适用于山地、隧道、断桥、坍塌区周边和广域补链场景。",
    strategyNote: "在资源调度中通常承担回传稳定性和关键链路打通任务，适合做中继转发或核心续接节点。",
    stats: ["远距回传", "多跳组网", "跨障碍覆盖"],
  },
];

const sceneTab = ref("replay");
const activeMissionIndex = ref(0);
const missionPlaying = ref(false);
let missionTimer = null;
const missionStageDurations = {
  intake: 1800,
  sites: 2600,
  training: 8600,
  deploy: 3600,
  evaluate: 2200,
};
const activeDevice = ref(null);
const currentRoute = ref("home");
const scenarios = ref([]);

const clearMissionTimer = () => {
  if (missionTimer) {
    clearInterval(missionTimer);
    missionTimer = null;
  }
};

const selectMissionStage = (index) => {
  clearMissionTimer();
  missionPlaying.value = false;
  activeMissionIndex.value = index;
};

const playMission = () => {
  clearMissionTimer();
  missionPlaying.value = true;
  activeMissionIndex.value = 0;

  const advanceMission = () => {
    if (activeMissionIndex.value >= missionStages.length - 1) {
      clearMissionTimer();
      missionPlaying.value = false;
      return;
    }
    activeMissionIndex.value += 1;
    const stageKey = missionStages[activeMissionIndex.value]?.key;
    missionTimer = setTimeout(advanceMission, missionStageDurations[stageKey] || 1800);
  };

  missionTimer = setTimeout(advanceMission, missionStageDurations.intake);
};

const normalizeRoute = (hash) => {
  const route = hash.replace(/^#\/?/, "").replace(/\/+$/, "").trim();
  if (!route) {
    return { route: "home", sceneTab: "replay" };
  }
  if (route === "training") {
    return { route: "algorithm", sceneTab: sceneTab.value };
  }
  if (route === "testing") {
    return { route: "tester", sceneTab: sceneTab.value };
  }
  if (route === "scene") {
    return { route: "scene", sceneTab: "replay" };
  }
  if (route === "replay") {
    return { route: "scene", sceneTab: "replay" };
  }
  if (route === "mahimahi") {
    return { route: "scene", sceneTab: "mahimahi" };
  }
  if (navItems.some((item) => item.key === route)) {
    return { route, sceneTab: sceneTab.value };
  }
  return { route: "home", sceneTab: "replay" };
};

const isNavItemActive = (item) => {
  if (item.key === "replay" || item.key === "mahimahi") {
    return currentRoute.value === "scene" && sceneTab.value === item.key;
  }
  return currentRoute.value === item.key;
};

const syncRoute = () => {
  const result = normalizeRoute(window.location.hash);
  currentRoute.value = result.route;
  sceneTab.value = result.sceneTab;
};

const fetchScenarios = async () => {
  try {
    const { data } = await axios.get(`${rescueApiBase}/scenarios`, { timeout: 10000 });
    scenarios.value = data?.scenarios || [];
  } catch {
    scenarios.value = [];
  }
};

const selectedScenario = computed(() => scenarios.value[0] || null);

const activeMissionStage = computed(() => missionStages[activeMissionIndex.value] || missionStages[0]);

const missionProgress = computed(() => {
  if (missionStages.length <= 1) {
    return "0%";
  }
  return `${(activeMissionIndex.value / (missionStages.length - 1)) * 100}%`;
});

const activeSceneTab = computed(() => sceneTabs.find((tab) => tab.key === sceneTab.value) || sceneTabs[0]);

const sceneChips = computed(() => activeSceneTab.value.chips);

onMounted(() => {
  syncRoute();
  fetchScenarios();
  window.addEventListener("hashchange", syncRoute);
});

onBeforeUnmount(() => {
  clearMissionTimer();
  window.removeEventListener("hashchange", syncRoute);
});
</script>

<style scoped>
.app-shell {
  --sidebar-width: 282px;
  position: relative;
  width: 100%;
  max-width: none;
  margin: 0;
  min-height: 100vh;
  isolation: isolate;
}

.app-shell__aurora {
  display: none;
}

.app-shell__aurora--left {
  top: -12rem;
  left: -14rem;
  background: rgba(20, 184, 166, 0.35);
}

.app-shell__aurora--right {
  top: 6rem;
  right: -14rem;
  background: rgba(59, 130, 246, 0.3);
}

.topbar {
  position: fixed;
  top: 0;
  left: 0;
  bottom: 0;
  width: calc(var(--sidebar-width) + 1px);
  display: flex;
  flex-direction: column;
  align-items: stretch;
  gap: 18px;
  min-height: 0;
  padding: 22px 18px 18px;
  border: none;
  border-right: 1px solid rgba(71, 85, 105, 0.16);
  background: linear-gradient(180deg, #ffffff, #f3f6fa 68%, #eef3f8);
  backdrop-filter: blur(22px);
  box-shadow: none;
  overflow-y: auto;
  overscroll-behavior: contain;
  z-index: 40;
}

.topbar::before {
  display: none;
}

.topbar__head {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.brand {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 14px;
  color: #081226;
  text-decoration: none;
  max-width: 100%;
  text-align: center;
}

.brand__mark {
  position: relative;
  display: grid;
  place-items: center;
  width: 70px;
  height: 70px;
  border-radius: 22px;
  background:
    radial-gradient(circle at 30% 28%, rgba(255, 255, 255, 0.95), rgba(226, 232, 240, 0.9)),
    linear-gradient(180deg, rgba(248, 250, 252, 0.98), rgba(226, 232, 240, 0.9));
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.84),
    0 14px 30px rgba(15, 23, 42, 0.08);
  overflow: hidden;
}

.brand__mark::after {
  content: "";
  position: absolute;
  inset: 10px;
  border-radius: 16px;
  border: 1px solid rgba(14, 165, 233, 0.12);
}

.brand__mark i {
  position: absolute;
  display: block;
  width: 10px;
  border-radius: 999px;
  background: linear-gradient(180deg, #0ea5e9, #0f172a);
  animation: sidebarSignalPulse 2.6s ease-in-out infinite;
}

.brand__mark i:nth-child(1) {
  left: 22px;
  height: 22px;
  bottom: 22px;
}

.brand__mark i:nth-child(2) {
  left: calc(50% - 5px);
  height: 32px;
  bottom: 18px;
}

.brand__mark i:nth-child(3) {
  right: 22px;
  height: 42px;
  bottom: 14px;
}

.brand__copy {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.brand__eyebrow {
  font-size: 11px;
  letter-spacing: 0.24em;
  text-transform: uppercase;
  color: #475569;
}

.brand strong {
  font-size: clamp(1.12rem, 1.8vw, 1.5rem);
  line-height: 1.18;
  letter-spacing: 0.04em;
}

.brand strong span {
  display: block;
}

.brand p {
  margin: 0;
  color: #64748b;
  font-size: 12px;
  line-height: 1.65;
}

.eyebrow {
  font-size: 11px;
  letter-spacing: 0.24em;
  text-transform: uppercase;
  color: #334155;
}

.topbar__nav {
  display: flex;
  flex-direction: column;
  justify-content: flex-start;
  gap: 8px;
  width: 100%;
  flex: 1 1 auto;
}

.nav-chip {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  min-width: 0;
  width: 100%;
  padding: 14px 16px;
  min-height: 58px;
  border: none;
  border-left: 4px solid transparent;
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.82), rgba(244, 247, 250, 0.72));
  color: #0f172a;
  text-decoration: none;
  text-align: center;
  overflow: hidden;
  box-shadow: inset 0 0 0 1px rgba(148, 163, 184, 0.16);
  transition:
    transform 0.2s ease,
    border-color 0.2s ease,
    background 0.2s ease,
    color 0.2s ease,
    box-shadow 0.2s ease;
}

.nav-chip::after {
  content: "";
  position: absolute;
  inset: 0;
  background: linear-gradient(120deg, transparent, rgba(255, 255, 255, 0.56), transparent);
  transform: translateX(-130%);
  transition: transform 0.36s ease;
  pointer-events: none;
}

.nav-chip__label {
  display: block;
  width: 100%;
  font-size: 15px;
  line-height: 1.15;
  letter-spacing: 0.04em;
  color: #0f172a;
}

.nav-chip:hover {
  transform: translateX(2px);
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.96), rgba(233, 239, 245, 0.84));
  border-left-color: rgba(30, 41, 59, 0.32);
  box-shadow:
    inset 0 0 0 1px rgba(100, 116, 139, 0.2),
    0 10px 18px rgba(15, 23, 42, 0.05);
}

.nav-chip:hover::after {
  transform: translateX(125%);
}

.nav-chip--active {
  background:
    linear-gradient(135deg, rgba(224, 242, 254, 0.94), rgba(241, 245, 249, 0.96)),
    radial-gradient(circle at top right, rgba(14, 165, 233, 0.12), transparent 44%);
  border-left-color: #0ea5e9;
  box-shadow:
    inset 0 0 0 1px rgba(14, 165, 233, 0.12),
    0 16px 28px rgba(15, 23, 42, 0.06);
}

.nav-chip--active .nav-chip__label {
  color: #075985;
}

@keyframes sidebarSignalPulse {
  0%,
  100% {
    transform: translateY(0);
    opacity: 0.9;
  }
  50% {
    transform: translateY(-1px);
    opacity: 1;
  }
}

.app-main {
  display: flex;
  flex-direction: column;
  gap: 28px;
  min-height: 100vh;
  margin-left: var(--sidebar-width);
}

.landing-view,
.module-view {
  display: flex;
  flex-direction: column;
  gap: 24px;
}

.landing-view--map {
  gap: 0;
  min-height: 100vh;
}

.map-command-screen {
  position: relative;
  display: grid;
  grid-template-columns: minmax(0, 1fr) clamp(360px, 28vw, 430px);
  height: 100vh;
  min-height: 0;
  overflow: hidden;
  border: none;
  background: #f8fafc;
  box-shadow: none;
  animation: commandScreenIn 0.72s cubic-bezier(0.2, 0.8, 0.2, 1) both;
  isolation: isolate;
}

.module-view > .panel-shell {
  border-radius: 0;
  border: none;
  box-shadow: none;
}

.mission-console {
  position: relative;
  z-index: 2;
  grid-column: 2;
  grid-row: 1;
  display: grid;
  grid-template-rows: auto auto minmax(260px, 1fr) auto;
  gap: clamp(12px, 1.5vh, 18px);
  min-height: 0;
  padding: clamp(18px, 1.85vw, 24px);
  background:
    linear-gradient(180deg, rgba(255, 255, 255, 0.96), rgba(241, 245, 249, 0.92)),
    radial-gradient(circle at 0% 0%, rgba(20, 184, 166, 0.08), transparent 42%);
  border-left: 1px solid rgba(15, 23, 42, 0.1);
}

.map-command-screen__stage {
  position: relative;
  grid-column: 1;
  grid-row: 1;
  min-width: 0;
  min-height: 0;
  background: #0f172a;
  overflow: hidden;
  isolation: isolate;
}

.map-command-screen__map {
  position: relative;
  z-index: 0;
  width: 100%;
  height: 100%;
  min-height: 0;
  border: none;
  border-radius: 0;
  box-shadow: none;
  isolation: isolate;
}

.map-command-screen__map :deep(.satellite-board) {
  min-height: 100%;
  height: 100%;
  border: none;
  border-radius: 0;
  box-shadow: none;
}

.map-command-screen__map :deep(.satellite-board__hud) {
  top: 18px;
  left: 18px;
  right: 18px;
}

.map-command-screen__shade {
  display: none;
}

.map-command-screen__header,
.map-command-screen__status,
.map-command-screen__mission,
.map-command-screen__timeline {
  position: relative;
  z-index: 3;
  transform: translateZ(0);
}

.mission-path {
  display: none;
}

.mission-path__line {
  position: absolute;
  left: 57%;
  top: 42%;
  width: 28%;
  height: 3px;
  transform: rotate(23deg);
  transform-origin: left center;
  border-radius: 999px;
  background: linear-gradient(90deg, rgba(8, 145, 178, 0), rgba(8, 145, 178, 0.72), rgba(34, 197, 94, 0.88));
  box-shadow: 0 0 24px rgba(34, 197, 94, 0.32);
  clip-path: inset(0 100% 0 0);
  animation: missionLineDraw 1.35s ease forwards;
}

.mission-path__node {
  position: absolute;
  width: 14px;
  height: 14px;
  border-radius: 999px;
  background: #0891b2;
  box-shadow:
    0 0 0 8px rgba(8, 145, 178, 0.16),
    0 0 20px rgba(8, 145, 178, 0.34);
  opacity: 0;
  animation: missionNodeIn 0.6s ease forwards;
}

.mission-path__node--a {
  left: 56%;
  top: 41%;
}

.mission-path__node--b {
  left: 68%;
  top: 48%;
  animation-delay: 0.42s;
}

.mission-path__node--c {
  left: 83%;
  top: 55%;
  background: #22c55e;
  box-shadow:
    0 0 0 8px rgba(34, 197, 94, 0.16),
    0 0 24px rgba(34, 197, 94, 0.36);
  animation-delay: 0.84s;
}

.map-command-screen--intake .mission-path,
.map-command-screen--sites .mission-path {
  opacity: 0.28;
}

.map-command-screen--training .mission-path,
.map-command-screen--deploy .mission-path,
.map-command-screen--evaluate .mission-path {
  opacity: 1;
}

.map-command-screen__header {
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  align-items: stretch;
  gap: 10px;
  padding: 0;
}

.map-command-screen__header h1 {
  max-width: 100%;
  margin: 10px 0 0;
  color: #07111f;
  font-size: clamp(1.32rem, 1.72vw, 1.86rem);
  line-height: 1;
  letter-spacing: -0.03em;
}

.map-command-screen__header p {
  max-width: 35rem;
  margin: 8px 0 0;
  color: #475569;
  font-size: 0.88rem;
}

.map-command-screen__live {
  display: inline-grid;
  grid-template-columns: auto auto;
  justify-content: center;
  justify-items: center;
  align-items: center;
  gap: 5px 8px;
  min-width: 190px;
  width: fit-content;
  align-self: center;
  padding: 9px 12px;
  border-radius: 8px;
  border: 1px solid rgba(15, 23, 42, 0.1);
  background: rgba(255, 255, 255, 0.72);
  backdrop-filter: blur(16px);
  box-shadow: 0 18px 34px rgba(15, 23, 42, 0.08);
  text-align: center;
}

.map-command-screen__live span {
  width: 9px;
  height: 9px;
  border-radius: 999px;
  background: #22c55e;
  box-shadow: 0 0 0 7px rgba(34, 197, 94, 0.14);
}

.map-command-screen__live strong {
  color: #0f172a;
  letter-spacing: 0.16em;
  font-size: 0.78rem;
}

.map-command-screen__live small {
  grid-column: 1 / -1;
  color: #64748b;
}

.map-command-screen__status {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  max-width: none;
  gap: 10px 14px;
  margin: 0;
}

.map-command-screen__status span {
  min-width: 120px;
  padding: 7px 0 0;
  border-top: 1px solid rgba(15, 23, 42, 0.2);
}

.map-command-screen__status small {
  display: block;
  margin-bottom: 4px;
  color: #64748b;
  font-size: 0.72rem;
  letter-spacing: 0.12em;
}

.map-command-screen__status strong {
  color: #0f172a;
  font-size: clamp(1rem, 1.45vw, 1.28rem);
  letter-spacing: -0.035em;
}

.map-command-screen__mission {
  width: 100%;
  align-self: end;
  margin: 0;
  padding: 0;
  border: none;
  border-radius: 0;
  background: transparent;
  backdrop-filter: none;
  box-shadow: none;
  animation: missionPanelIn 0.54s ease both;
}

.map-command-screen__mission h2 {
  margin: 6px 0 6px;
  color: #0f172a;
  font-size: clamp(1rem, 1.35vw, 1.2rem);
  line-height: 1.18;
  letter-spacing: -0.02em;
}

.map-command-screen__mission p {
  margin: 0;
  color: #475569;
  line-height: 1.42;
  font-size: 0.82rem;
  max-width: 28rem;
  min-height: 1.15rem;
}

.mission-actions {
  display: flex;
  flex-wrap: nowrap;
  gap: 8px;
  margin: 0;
}

.map-command-screen__mission .primary-cta,
.map-command-screen__mission .secondary-cta {
  flex: 1 1 0;
  min-height: 40px;
  justify-content: center;
  padding: 0 12px;
  font-size: 0.86rem;
}

.mission-links {
  display: grid;
  gap: 1px;
  overflow: hidden;
  border: 1px solid rgba(15, 23, 42, 0.08);
  border-radius: 8px;
  background: rgba(15, 23, 42, 0.06);
}

.mission-link {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 4px 12px;
  padding: 13px 14px;
  color: #0f172a;
  text-decoration: none;
  background: rgba(255, 255, 255, 0.74);
  transition: transform 0.22s ease, border-color 0.22s ease, background 0.22s ease;
}

.mission-link:hover {
  transform: translateX(4px);
  background: rgba(240, 249, 255, 0.92);
}

.mission-link small {
  color: #0891b2;
  font-size: 0.68rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
}

.mission-link strong {
  grid-row: 2;
  font-size: 1.03rem;
}

.mission-link span {
  grid-row: 2;
  color: #64748b;
  font-size: 0.82rem;
  text-align: right;
}

.map-command-screen__timeline {
  position: relative;
  z-index: 4;
  display: grid;
  grid-template-rows: repeat(5, minmax(54px, 1fr));
  gap: 1px;
  width: 100%;
  max-width: none;
  min-height: 0;
  margin: 0;
  overflow: visible;
  padding: 2px 0;
  border: none;
  background: transparent;
  backdrop-filter: none;
  box-shadow: none;
}

.map-command-screen__timeline::before,
.map-command-screen__timeline::after {
  content: "";
  position: absolute;
  left: 16px;
  top: 28px;
  width: 3px;
  height: calc(100% - 56px);
  border-radius: 999px;
  pointer-events: none;
}

.map-command-screen__timeline::before {
  background: rgba(148, 163, 184, 0.28);
}

.map-command-screen__timeline::after {
  height: calc((100% - 56px) * var(--mission-progress));
  background: linear-gradient(180deg, #0891b2, #14b8a6, #22c55e);
  box-shadow: 0 0 18px rgba(20, 184, 166, 0.24);
  transition: height 0.6s cubic-bezier(0.2, 0.8, 0.2, 1);
}

.mission-step {
  position: relative;
  z-index: 1;
  display: flex;
  flex-direction: row;
  align-items: center;
  gap: 10px;
  min-width: 0;
  min-height: 0;
  padding: 6px 0;
  border: none;
  background: transparent;
  color: #334155;
  text-align: left;
  white-space: normal;
  transition: color 0.24s ease, transform 0.24s ease;
}

.mission-step i {
  display: grid;
  place-items: center;
  width: 29px;
  height: 29px;
  border-radius: 8px;
  border: 1px solid rgba(148, 163, 184, 0.38);
  background:
    linear-gradient(180deg, rgba(255, 255, 255, 0.88), rgba(241, 245, 249, 0.72));
  color: #475569;
  font-size: 0.9rem;
  font-weight: 800;
  font-style: normal;
  box-shadow:
    0 0 0 5px rgba(255, 255, 255, 0.82),
    0 6px 14px rgba(15, 23, 42, 0.08);
  transition: transform 0.24s ease, background 0.24s ease, color 0.24s ease, border-color 0.24s ease, box-shadow 0.24s ease;
}

.mission-step__copy {
  display: grid;
  gap: 2px;
  min-height: auto;
  min-width: 0;
}

.mission-step strong,
.mission-step__copy strong {
  color: #0f172a;
  font-size: 0.82rem;
  line-height: 1.15;
}

.mission-step small,
.mission-step__copy small {
  color: #64748b;
  font-size: 0.63rem;
  line-height: 1.2;
}

.mission-step em {
  margin-left: auto;
  padding: 3px 6px;
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.58);
  color: #64748b;
  font-size: 0.6rem;
  font-style: normal;
  line-height: 1;
  border: 1px solid rgba(148, 163, 184, 0.16);
}

.mission-step--complete i {
  border-color: rgba(34, 197, 94, 0.5);
  background: linear-gradient(135deg, #14b8a6, #22c55e);
  color: #f8fafc;
  box-shadow:
    0 0 0 6px rgba(220, 252, 231, 0.84),
    0 8px 20px rgba(34, 197, 94, 0.18);
}

.mission-step--complete em {
  border-color: rgba(34, 197, 94, 0.22);
  background: rgba(220, 252, 231, 0.72);
  color: #15803d;
}

.mission-step--active {
  transform: translateX(4px);
}

.mission-step--active i {
  width: 34px;
  height: 34px;
  border-color: #0891b2;
  background: linear-gradient(135deg, #0f172a, #0891b2);
  color: #f8fafc;
  font-size: 1rem;
  box-shadow:
    0 0 0 6px rgba(8, 145, 178, 0.14),
    0 0 0 10px rgba(8, 145, 178, 0.08),
    0 10px 22px rgba(15, 23, 42, 0.2);
}

.mission-step--active strong,
.mission-step--active .mission-step__copy strong {
  color: #0f172a;
}

.mission-step--active em {
  border-color: rgba(8, 145, 178, 0.24);
  background: rgba(224, 242, 254, 0.78);
  color: #0891b2;
  font-weight: 700;
}

@keyframes commandScreenIn {
  from {
    opacity: 0;
    transform: translateY(18px) scale(0.985);
  }
  to {
    opacity: 1;
    transform: translateY(0) scale(1);
  }
}

@keyframes missionPanelIn {
  from {
    opacity: 0;
    transform: translateY(8px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@keyframes missionLineDraw {
  to {
    clip-path: inset(0 0 0 0);
  }
}

@keyframes missionNodeIn {
  from {
    opacity: 0;
    transform: scale(0.72);
  }
  to {
    opacity: 1;
    transform: scale(1);
  }
}

@keyframes metricRefresh {
  from {
    opacity: 0.72;
    transform: translateY(4px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@keyframes missionStagePulse {
  0%,
  100% {
    transform: scale(1);
  }
  50% {
    transform: scale(1.06);
  }
}

.panel-shell {
  border-radius: 24px;
  border: 1px solid rgba(71, 85, 105, 0.12);
  background:
    linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(247, 249, 252, 0.94)),
    radial-gradient(circle at top right, rgba(148, 163, 184, 0.07), transparent 42%);
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.72),
    0 14px 36px rgba(15, 23, 42, 0.08);
  color: #0f172a;
}

.panel-shell--hero {
  display: grid;
  grid-template-columns: minmax(280px, 0.78fr) minmax(0, 1.22fr);
  gap: 20px;
  padding: 30px;
  min-height: 680px;
  overflow: hidden;
  background:
    radial-gradient(circle at top left, rgba(56, 189, 248, 0.22), transparent 28%),
    radial-gradient(circle at 88% 16%, rgba(249, 115, 22, 0.14), transparent 24%),
    linear-gradient(135deg, rgba(252, 254, 255, 0.98), rgba(236, 246, 255, 0.94));
  color: #0f172a;
}

.landing-hero__copy {
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: 16px;
  max-width: 560px;
}

.landing-hero__copy h1,
.module-hero h1 {
  margin: 0;
  font-size: clamp(2.2rem, 3.1vw, 3.25rem);
  line-height: 0.98;
  letter-spacing: -0.035em;
}

.landing-hero__copy h1 {
  width: fit-content;
  max-width: 100%;
  white-space: nowrap;
  color: #0f172a;
}

.panel-shell--hero .eyebrow {
  color: #0f766e;
}

.hero-kicker {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.hero-kicker__meta {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.hero-kicker__meta span {
  padding: 6px 10px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.62);
  border: 1px solid rgba(56, 189, 248, 0.12);
  color: #64748b;
  font-size: 11px;
  letter-spacing: 0.14em;
}

.landing-hero__summary,
.module-hero p {
  margin: 0;
  max-width: 44rem;
  line-height: 1.55;
  color: #475569;
  font-size: 0.94rem;
}

.landing-hero__summary {
  color: #475569;
  max-width: 32rem;
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.landing-hero__summary span {
  padding: 7px 11px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.7);
  border: 1px solid rgba(148, 163, 184, 0.12);
  color: #334155;
  font-size: 12px;
}

.hero-status-bar {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.hero-status-pill {
  min-width: 108px;
  padding: 10px 12px;
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.68);
  border: 1px solid rgba(56, 189, 248, 0.14);
  backdrop-filter: blur(16px);
  box-shadow: 0 10px 24px rgba(148, 163, 184, 0.12);
}

.hero-status-pill small {
  display: block;
  color: #64748b;
  font-size: 11px;
  margin-bottom: 6px;
  letter-spacing: 0.08em;
}

.hero-status-pill strong {
  color: #0f172a;
  font-size: 1rem;
  font-weight: 700;
}

.landing-hero__actions {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 4px;
}

.primary-cta,
.secondary-cta,
.module-card__link {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-height: 48px;
  padding: 0 20px;
  border-radius: 8px;
  text-decoration: none;
  border: 0;
  transition: transform 0.2s ease, box-shadow 0.2s ease, background 0.2s ease;
}

.primary-cta {
  background: linear-gradient(135deg, #0891b2, #2563eb);
  color: #f8fafc;
  box-shadow: 0 14px 30px rgba(37, 99, 235, 0.24);
}

.secondary-cta,
.module-card__link {
  background: rgba(255, 255, 255, 0.82);
  border: 1px solid rgba(56, 189, 248, 0.12);
  color: #0f172a;
}

.primary-cta:hover,
.secondary-cta:hover,
.module-card__link:hover {
  transform: translateY(-1px);
}

.module-card__points span,
.module-card__points span,
.device-card__stats span,
.device-showcase__meta span,
.device-modal__chips span,
.scene-stage__chips span {
  padding: 8px 11px;
  border-radius: 999px;
  background: rgba(248, 250, 252, 0.92);
  border: 1px solid rgba(71, 85, 105, 0.12);
  color: #334155;
  font-size: 12px;
}

.hero-command-row {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
}

.hero-command-card {
  padding: 12px 14px;
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.66);
  border: 1px solid rgba(56, 189, 248, 0.12);
  box-shadow: 0 12px 24px rgba(148, 163, 184, 0.1);
}

.hero-command-card small {
  display: block;
  margin-bottom: 8px;
  color: #0891b2;
  font-size: 10px;
  letter-spacing: 0.16em;
}

.hero-command-card strong {
  color: #0f172a;
  font-size: 0.98rem;
}

.scene-cockpit {
  position: relative;
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: 14px;
  min-width: 0;
}

.scene-cockpit__stage {
  position: relative;
  min-height: 560px;
  border-radius: 28px;
  overflow: hidden;
  border: 1px solid rgba(56, 189, 248, 0.18);
  background:
    radial-gradient(circle at 50% 48%, rgba(59, 130, 246, 0.14), transparent 28%),
    linear-gradient(180deg, rgba(248, 252, 255, 0.96), rgba(226, 240, 251, 0.94));
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.72),
    0 28px 48px rgba(96, 165, 250, 0.18);
}

.scene-cockpit__hud {
  position: absolute;
  top: 18px;
  left: 18px;
  right: 18px;
  z-index: 2;
  display: flex;
  justify-content: space-between;
  gap: 12px;
  font-size: 11px;
  letter-spacing: 0.16em;
  color: rgba(14, 116, 144, 0.72);
}

.scene-cockpit__legend,
.scene-cockpit__monitor,
.scene-cockpit__map-meta {
  position: absolute;
  z-index: 4;
}

.scene-cockpit__legend {
  left: 12px;
  bottom: 56px;
}

.scene-cockpit__monitor {
  right: 12px;
  top: 44px;
}

.legend-card,
.monitor-card {
  width: 144px;
  padding: 10px 11px;
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.74);
  border: 1px solid rgba(56, 189, 248, 0.14);
  box-shadow: 0 14px 26px rgba(148, 163, 184, 0.14);
  backdrop-filter: blur(14px);
}

.legend-card strong,
.monitor-card strong {
  display: block;
  color: #0f172a;
}

.legend-list,
.monitor-list {
  display: grid;
  gap: 6px;
  margin-top: 8px;
}

.legend-list span,
.monitor-list span {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  color: #475569;
  font-size: 11px;
}

.legend-dot,
.legend-line {
  display: inline-block;
  flex: 0 0 auto;
}

.legend-dot {
  width: 10px;
  height: 10px;
  border-radius: 999px;
  margin-right: 6px;
}

.legend-dot--critical {
  background: #f87171;
  box-shadow: 0 0 0 4px rgba(248, 113, 113, 0.18);
}

.legend-dot--warning {
  background: #fbbf24;
  box-shadow: 0 0 0 4px rgba(251, 191, 36, 0.18);
}

.legend-dot--watch {
  background: #60a5fa;
  box-shadow: 0 0 0 4px rgba(96, 165, 250, 0.16);
}

.legend-line {
  width: 18px;
  height: 0;
  margin-right: 6px;
  border-top: 2px solid rgba(14, 165, 233, 0.56);
}

.legend-line--road {
  border-top-style: dashed;
  border-top-color: rgba(100, 116, 139, 0.6);
}

.monitor-card small {
  display: block;
  color: #0891b2;
  font-size: 10px;
  letter-spacing: 0.16em;
  margin-bottom: 6px;
}

.monitor-list span {
  align-items: baseline;
  padding: 7px 0;
  border-top: 1px solid rgba(148, 163, 184, 0.12);
}

.monitor-list span:first-child {
  border-top: none;
  padding-top: 0;
}

.monitor-list span small {
  margin: 0;
  color: #64748b;
  letter-spacing: 0;
  font-size: 11px;
}

.monitor-list span strong {
  font-size: 13px;
  color: #0f172a;
}

.scene-cockpit__grid,
.scene-cockpit__terrain,
.scene-cockpit__river,
.scene-cockpit__impact-zone,
.scene-cockpit__roads,
.scene-cockpit__zones,
.scene-cockpit__sweep,
.scene-cockpit__corridor {
  position: absolute;
  inset: 0;
}

.scene-cockpit__grid {
  background-image:
    linear-gradient(rgba(56, 189, 248, 0.1) 1px, transparent 1px),
    linear-gradient(90deg, rgba(56, 189, 248, 0.1) 1px, transparent 1px);
  background-size: 42px 42px;
  mask-image: linear-gradient(180deg, rgba(0, 0, 0, 0.8), transparent 96%);
}

.scene-cockpit__terrain {
  background:
    radial-gradient(circle at 12% 18%, rgba(148, 163, 184, 0.16), transparent 20%),
    radial-gradient(circle at 82% 72%, rgba(148, 163, 184, 0.14), transparent 18%),
    radial-gradient(circle at 20% 30%, rgba(16, 185, 129, 0.1), transparent 18%),
    radial-gradient(circle at 74% 24%, rgba(59, 130, 246, 0.09), transparent 16%),
    radial-gradient(circle at 36% 72%, rgba(245, 158, 11, 0.08), transparent 14%),
    linear-gradient(145deg, rgba(255, 255, 255, 0.24), rgba(203, 213, 225, 0.12));
  opacity: 0.92;
  filter: saturate(1.08) contrast(1.02);
}

.scene-cockpit__river {
  inset: 8% 12% 10% 10%;
  background:
    linear-gradient(
      118deg,
      transparent 0%,
      transparent 28%,
      rgba(56, 189, 248, 0.2) 36%,
      rgba(125, 211, 252, 0.38) 42%,
      rgba(56, 189, 248, 0.22) 48%,
      transparent 56%,
      transparent 100%
    );
  filter: blur(8px);
  opacity: 0.8;
}

.scene-cockpit__overlay {
  position: absolute;
  inset: 0;
  z-index: 1;
  width: 100%;
  height: 100%;
  pointer-events: none;
}

.scene-cockpit__overlay path {
  fill: none;
  vector-effect: non-scaling-stroke;
}

.scene-cockpit__overlay--hydro {
  opacity: 0.9;
}

.scene-cockpit__overlay--hydro path:first-child {
  stroke: rgba(14, 165, 233, 0.44);
  stroke-width: 3.2;
  stroke-linecap: round;
  stroke-linejoin: round;
  filter: drop-shadow(0 0 10px rgba(56, 189, 248, 0.18));
}

.scene-cockpit__overlay--hydro path:last-child {
  stroke: rgba(56, 189, 248, 0.28);
  stroke-width: 2;
  stroke-linecap: round;
  stroke-linejoin: round;
  stroke-dasharray: 6 4;
}

.scene-cockpit__overlay--districts path {
  fill: rgba(255, 255, 255, 0.16);
  stroke: rgba(71, 85, 105, 0.22);
  stroke-width: 1.35;
  stroke-linejoin: round;
}

.scene-cockpit__overlay--districts path:nth-child(2n) {
  fill: rgba(191, 219, 254, 0.16);
}

.scene-cockpit__overlay--districts path:nth-child(3n) {
  fill: rgba(187, 247, 208, 0.14);
}

.scene-cockpit__impact-zone {
  z-index: 1;
}

.impact-zone {
  position: absolute;
  display: block;
  border-radius: 999px;
  filter: blur(10px);
  mix-blend-mode: multiply;
}

.impact-zone--critical {
  left: 56%;
  top: 20%;
  width: 28%;
  height: 22%;
  background: radial-gradient(circle, rgba(248, 113, 113, 0.34) 0%, rgba(248, 113, 113, 0.16) 48%, transparent 76%);
}

.impact-zone--warning {
  left: 18%;
  top: 54%;
  width: 30%;
  height: 24%;
  background: radial-gradient(circle, rgba(251, 191, 36, 0.28) 0%, rgba(251, 191, 36, 0.14) 44%, transparent 74%);
}

.impact-zone--watch {
  left: 42%;
  top: 40%;
  width: 24%;
  height: 20%;
  background: radial-gradient(circle, rgba(59, 130, 246, 0.2) 0%, rgba(59, 130, 246, 0.1) 42%, transparent 72%);
}

.road {
  position: absolute;
  display: block;
  border-radius: 999px;
  background: linear-gradient(90deg, rgba(148, 163, 184, 0.1), rgba(255, 255, 255, 0.6), rgba(148, 163, 184, 0.1));
  box-shadow: 0 0 0 1px rgba(148, 163, 184, 0.08);
}

.road::after {
  content: "";
  position: absolute;
  inset: 42% 4%;
  background-image: linear-gradient(90deg, rgba(14, 165, 233, 0.22) 0 12px, transparent 12px 20px);
  border-radius: 999px;
}

.road--a {
  left: 12%;
  top: 28%;
  width: 68%;
  height: 10px;
  transform: rotate(-12deg);
}

.road--b {
  left: 26%;
  top: 60%;
  width: 50%;
  height: 10px;
  transform: rotate(18deg);
}

.road--c {
  left: 46%;
  top: 16%;
  width: 8px;
  height: 62%;
  transform: rotate(4deg);
}

.scene-zone {
  position: absolute;
  left: var(--zone-x);
  top: var(--zone-y);
  width: var(--zone-w);
  height: var(--zone-h);
  transform: rotate(var(--zone-rotate));
  border-radius: 22px;
  background:
    linear-gradient(180deg, rgba(255, 255, 255, 0.34), rgba(191, 219, 254, 0.12)),
    repeating-linear-gradient(90deg, rgba(148, 163, 184, 0.08) 0 10px, transparent 10px 20px);
  border: 1px solid rgba(148, 163, 184, 0.14);
  box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.16);
}

.scene-cockpit__sweep {
  background:
    radial-gradient(circle at 42% 54%, rgba(56, 189, 248, 0.22), transparent 16%),
    conic-gradient(from 220deg at 42% 54%, transparent 0deg, rgba(56, 189, 248, 0.18) 56deg, transparent 96deg);
  filter: blur(2px);
  opacity: 0.85;
}

.scene-cockpit__corridor {
  background:
    linear-gradient(125deg, transparent 34%, rgba(34, 197, 94, 0.16) 47%, transparent 60%),
    linear-gradient(180deg, transparent 40%, rgba(249, 115, 22, 0.12) 68%, transparent 82%);
}

.scene-cockpit__core {
  position: absolute;
  left: 50%;
  top: 50%;
  transform: translate(-50%, -50%);
  min-width: 180px;
  padding: 20px 24px;
  border-radius: 24px;
  text-align: center;
  border: 1px solid rgba(56, 189, 248, 0.18);
  background: rgba(255, 255, 255, 0.7);
  backdrop-filter: blur(18px);
  box-shadow: 0 18px 36px rgba(148, 163, 184, 0.18);
}

.scene-cockpit__core span {
  display: block;
  margin-bottom: 8px;
  color: #0891b2;
  font-size: 11px;
  letter-spacing: 0.16em;
}

.scene-cockpit__core strong {
  color: #0f172a;
  font-size: 1.5rem;
  letter-spacing: 0.02em;
}

.scene-hotspot {
  position: absolute;
  left: var(--spot-x);
  top: var(--spot-y);
  transform: translate(-50%, -50%);
  z-index: 3;
}

.scene-hotspot__pulse {
  width: 16px;
  height: 16px;
  border-radius: 999px;
  background: var(--spot-tone);
  box-shadow:
    0 0 0 8px color-mix(in srgb, var(--spot-tone) 18%, transparent),
    0 0 22px color-mix(in srgb, var(--spot-tone) 54%, transparent);
}

.scene-hotspot__label {
  margin-top: 12px;
  padding: 9px 11px;
  border-radius: 14px;
  min-width: 116px;
  background: rgba(255, 255, 255, 0.78);
  border: 1px solid color-mix(in srgb, var(--spot-tone) 36%, rgba(148, 163, 184, 0.2));
  backdrop-filter: blur(14px);
  box-shadow: 0 14px 24px rgba(148, 163, 184, 0.16);
}

.scene-hotspot__label strong {
  display: block;
  color: #0f172a;
  font-size: 12px;
  margin-bottom: 4px;
}

.scene-hotspot__label small {
  color: #64748b;
  font-size: 10px;
}

.scene-label {
  position: absolute;
  left: var(--label-x);
  top: var(--label-y);
  transform: translate(-50%, -50%);
  z-index: 3;
  padding: 7px 10px;
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.74);
  border: 1px solid rgba(148, 163, 184, 0.12);
  color: #475569;
  font-size: 11px;
  letter-spacing: 0.04em;
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 80px;
}

.scene-label strong {
  color: #0f172a;
  font-size: 12px;
}

.scene-label small {
  color: #64748b;
  font-size: 10px;
}

.scene-label--major {
  background: rgba(255, 255, 255, 0.82);
  box-shadow: 0 10px 20px rgba(148, 163, 184, 0.14);
}

.scene-cockpit__map-meta {
  left: 12px;
  right: 12px;
  bottom: 12px;
  display: flex;
  justify-content: space-between;
  align-items: end;
  pointer-events: none;
}

.map-scale {
  display: flex;
  flex-direction: column;
  gap: 6px;
  color: #475569;
  font-size: 11px;
}

.map-scale span {
  display: block;
  width: 86px;
  height: 10px;
  border-left: 2px solid rgba(15, 23, 42, 0.54);
  border-right: 2px solid rgba(15, 23, 42, 0.54);
  border-top: 2px solid rgba(15, 23, 42, 0.54);
}

.map-coords {
  display: flex;
  gap: 12px;
  padding: 7px 10px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.66);
  border: 1px solid rgba(148, 163, 184, 0.12);
  color: #64748b;
  font-size: 10px;
  letter-spacing: 0.08em;
}

.scene-cockpit__footer {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
}

.command-card {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 12px 14px;
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.72);
  border: 1px solid rgba(56, 189, 248, 0.12);
  box-shadow: 0 12px 24px rgba(148, 163, 184, 0.12);
  animation: cardReveal 0.7s ease both;
}

.command-card:nth-child(2) {
  animation-delay: 0.12s;
}

.command-card:nth-child(3) {
  animation-delay: 0.24s;
}

.command-card small {
  color: #0891b2;
  letter-spacing: 0.14em;
  font-size: 10px;
}

.command-card strong {
  color: #0f172a;
}

.command-card span {
  color: #475569;
  font-size: 13px;
  line-height: 1.5;
}

@keyframes cockpitSweep {
  0% {
    transform: rotate(0deg) scale(1);
    opacity: 0.45;
  }
  50% {
    transform: rotate(180deg) scale(1.04);
    opacity: 0.88;
  }
  100% {
    transform: rotate(360deg) scale(1);
    opacity: 0.45;
  }
}

@keyframes hotspotPulse {
  0% {
    transform: scale(0.96);
    box-shadow:
      0 0 0 0 color-mix(in srgb, var(--spot-tone) 30%, transparent),
      0 0 18px color-mix(in srgb, var(--spot-tone) 44%, transparent);
  }
  70% {
    transform: scale(1.08);
    box-shadow:
      0 0 0 14px color-mix(in srgb, var(--spot-tone) 0%, transparent),
      0 0 26px color-mix(in srgb, var(--spot-tone) 56%, transparent);
  }
  100% {
    transform: scale(0.96);
    box-shadow:
      0 0 0 0 color-mix(in srgb, var(--spot-tone) 0%, transparent),
      0 0 18px color-mix(in srgb, var(--spot-tone) 44%, transparent);
  }
}

@keyframes cardFloat {
  0%,
  100% {
    transform: translateY(0);
  }
  50% {
    transform: translateY(-4px);
  }
}

@keyframes corridorGlow {
  0%,
  100% {
    opacity: 0.55;
  }
  50% {
    opacity: 0.9;
  }
}

@keyframes cardReveal {
  0% {
    opacity: 0;
    transform: translateY(18px);
  }
  100% {
    opacity: 1;
    transform: translateY(0);
  }
}

@keyframes impactPulse {
  0%,
  100% {
    opacity: 0.7;
    transform: scale(0.98);
  }
  50% {
    opacity: 1;
    transform: scale(1.04);
  }
}

@media (prefers-reduced-motion: reduce) {
  .scene-cockpit__sweep,
  .scene-cockpit__corridor,
  .scene-hotspot__pulse,
  .scene-hotspot__label,
  .command-card,
  .impact-zone {
    animation: none;
  }
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 20px;
}

.stat-card {
  padding: 24px;
  min-height: 156px;
}

.stat-card__label {
  display: inline-block;
  margin-bottom: 12px;
  color: #475569;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  font-size: 12px;
}

.stat-card strong {
  display: block;
  margin-bottom: 10px;
  font-size: 1.45rem;
}

.stat-card p,
.module-card p,
.info-card p,
.toolbar-note span,
.device-card p,
.device-modal__desc,
.device-modal__section p,
.scene-stage__intro p {
  margin: 0;
  color: #475569;
  line-height: 1.75;
}

.module-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 20px;
}

.module-card {
  position: relative;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  gap: 16px;
  padding: 32px;
  min-height: 300px;
}

.module-card::before {
  content: "";
  position: absolute;
  inset: 0;
  background:
    radial-gradient(circle at 18% 16%, rgba(15, 23, 42, 0.08), transparent 24%),
    linear-gradient(135deg, rgba(255, 255, 255, 0.26), transparent 56%);
  pointer-events: none;
}

.module-card > * {
  position: relative;
}

.module-card__head {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.module-card__head strong,
.info-card h3,
.scene-stage__intro h2,
.device-showcase__header h2,
.device-modal h2 {
  font-size: 1.35rem;
  margin: 0;
}

.module-card__tag,
.scene-stage__tag {
  display: inline-flex;
  width: fit-content;
  padding: 6px 10px;
  border-radius: 999px;
  background: rgba(15, 23, 42, 0.05);
  color: #334155;
  font-size: 12px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.module-card__head strong {
  font-size: clamp(1.9rem, 2.6vw, 3rem);
  line-height: 1;
  letter-spacing: -0.06em;
}

.module-hero {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 20px;
  padding: 28px 30px;
}

.panel-shell--module {
  background:
    linear-gradient(180deg, rgba(250, 251, 252, 0.98), rgba(245, 247, 250, 0.94)),
    radial-gradient(circle at right top, rgba(148, 163, 184, 0.08), transparent 38%);
}

.panel-shell--scene {
  background:
    linear-gradient(180deg, rgba(249, 251, 253, 0.98), rgba(242, 246, 250, 0.94)),
    radial-gradient(circle at right top, rgba(100, 116, 139, 0.1), transparent 42%);
}

.panel-shell--device {
  background:
    linear-gradient(180deg, rgba(251, 250, 248, 0.98), rgba(247, 244, 239, 0.94)),
    radial-gradient(circle at right top, rgba(120, 113, 108, 0.08), transparent 42%);
}

.module-hero__badges {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 10px;
}

.module-hero__badges span,
.toolbar-note strong {
  padding: 10px 12px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.88);
  border: 1px solid rgba(71, 85, 105, 0.12);
}

.panel-shell--toolbar {
  padding: 16px 18px;
  display: flex;
  justify-content: space-between;
  gap: 18px;
  align-items: center;
}

.segmented-control {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.segment-btn {
  display: inline-flex;
  flex-direction: column;
  gap: 3px;
  min-width: 180px;
  padding: 12px 14px;
  border-radius: 14px;
  border: 1px solid rgba(71, 85, 105, 0.12);
  background: rgba(255, 255, 255, 0.8);
  color: #0f172a;
}

.segment-btn small {
  color: #64748b;
}

.segment-btn--active {
  background: linear-gradient(135deg, rgba(226, 232, 240, 0.76), rgba(241, 245, 249, 0.92));
  border-color: rgba(30, 41, 59, 0.18);
}

.toolbar-note {
  display: flex;
  flex-direction: column;
  gap: 6px;
  max-width: 26rem;
}

.algorithm-stage {
  padding: 26px;
  display: flex;
  flex-direction: column;
  gap: 20px;
  min-height: 100vh;
}

.algorithm-stage__header {
  display: flex;
  justify-content: space-between;
  gap: 18px;
  align-items: flex-start;
}

.algorithm-stage__header h2 {
  margin: 12px 0 8px;
  font-size: clamp(1.8rem, 3vw, 3rem);
  line-height: 1;
  letter-spacing: -0.055em;
}

.algorithm-stage__header p {
  max-width: 58rem;
  margin: 0;
  color: #475569;
  line-height: 1.75;
}

.algorithm-stage__chips {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 8px;
}

.algorithm-stage__chips span {
  padding: 8px 11px;
  border-radius: 999px;
  background: rgba(15, 23, 42, 0.05);
  border: 1px solid rgba(148, 163, 184, 0.14);
  color: #334155;
  font-size: 12px;
}

.info-card {
  padding: 22px;
  display: flex;
  flex-direction: column;
  gap: 12px;
  min-height: 150px;
}

.info-card__eyebrow {
  color: #475569;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  font-size: 11px;
}

.device-modal__close {
  padding: 10px 14px;
  border-radius: 999px;
  border: 1px solid rgba(71, 85, 105, 0.12);
  background: rgba(255, 255, 255, 0.9);
  color: #0f172a;
}

.scene-stage {
  padding: 24px;
  display: flex;
  flex-direction: column;
  gap: 18px;
  min-height: 100vh;
}

.info-card__list,
.info-card__steps {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.info-card__list span,
.info-card__steps span {
  padding: 8px 11px;
  border-radius: 999px;
  background: rgba(248, 250, 252, 0.92);
  border: 1px solid rgba(71, 85, 105, 0.12);
  color: #334155;
  font-size: 12px;
}

.scene-stage__intro,
.device-showcase__header {
  display: flex;
  justify-content: space-between;
  gap: 18px;
  align-items: flex-start;
}

.device-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 18px;
}

.device-card {
  display: grid;
  grid-template-columns: 88px minmax(0, 1fr);
  gap: 18px;
  padding: 22px;
  cursor: pointer;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.device-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 24px 48px rgba(15, 23, 42, 0.12);
}

.device-card__media {
  width: 88px;
  height: 88px;
  border-radius: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: color-mix(in srgb, var(--device-tone) 14%, white 86%);
  border: 1px solid color-mix(in srgb, var(--device-tone) 32%, white 68%);
  color: #0f172a;
  font-weight: 800;
  font-size: 1.4rem;
}

.device-card__body {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.device-card__title {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.device-card__title small {
  color: #64748b;
}

.device-showcase {
  padding: 24px;
  display: flex;
  flex-direction: column;
  gap: 18px;
}

.device-showcase__loading {
  padding: 28px;
  border-radius: 22px;
  border: 1px dashed rgba(148, 163, 184, 0.24);
  color: #64748b;
  text-align: center;
}

.device-modal {
  position: fixed;
  inset: 0;
  background: rgba(15, 23, 42, 0.42);
  backdrop-filter: blur(8px);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 24px;
  z-index: 50;
}

.device-modal__panel {
  width: min(760px, 100%);
  padding: 26px;
  display: flex;
  flex-direction: column;
  gap: 18px;
  position: relative;
}

.device-modal__close {
  position: absolute;
  top: 20px;
  right: 20px;
}

.device-modal__section {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.device-modal__section h3 {
  margin: 0;
  font-size: 1.05rem;
}

@media (max-width: 1200px) {
  .panel-shell--hero,
  .device-grid,
  .module-grid,
  .stats-grid {
    grid-template-columns: 1fr;
  }

  .scene-cockpit__footer {
    grid-template-columns: 1fr;
  }

  .hero-command-row {
    grid-template-columns: 1fr;
  }

  .panel-shell--hero {
    min-height: auto;
    padding: 24px;
  }

  .map-command-screen {
    grid-template-columns: 1fr;
    height: auto;
    min-height: 0;
  }

  .mission-console {
    grid-column: auto;
    grid-row: auto;
    border-left: none;
    border-bottom: 1px solid rgba(15, 23, 42, 0.1);
  }

  .map-command-screen__stage {
    grid-column: auto;
    grid-row: auto;
  }

  .map-command-screen__stage,
  .map-command-screen__map {
    min-height: 460px;
  }

  .map-command-screen__header {
    flex-direction: column;
  }

  .map-command-screen__header h1 {
    white-space: normal;
  }

  .map-command-screen__status {
    max-width: 560px;
  }

  .landing-hero__copy {
    max-width: none;
  }

  .scene-cockpit__stage {
    min-height: 440px;
  }

  .scene-cockpit__monitor {
    top: 40px;
    right: 10px;
  }

  .scene-cockpit__legend {
    left: 10px;
    bottom: 52px;
  }

  .module-hero,
  .panel-shell--toolbar,
  .scene-stage__intro,
  .device-showcase__header {
    flex-direction: column;
    align-items: stretch;
  }
}

@media (min-width: 981px) and (max-width: 1200px) {
  .map-command-screen {
    grid-template-columns: minmax(0, 1fr) clamp(330px, 31vw, 380px);
    height: 100vh;
    min-height: 620px;
  }

  .mission-console {
    grid-column: 2;
    grid-row: 1;
    border-left: 1px solid rgba(15, 23, 42, 0.1);
    border-bottom: none;
  }

  .map-command-screen__stage {
    grid-column: 1;
    grid-row: 1;
  }

  .map-command-screen__stage,
  .map-command-screen__map {
    min-height: 0;
  }
}

@media (max-width: 900px) {
  .app-shell {
    --sidebar-width: 196px;
  }

  .topbar {
    padding: 16px 12px 14px;
  }

  .brand__mark {
    width: 58px;
    height: 58px;
    border-radius: 18px;
  }

  .brand__mark::after {
    inset: 8px;
    border-radius: 14px;
  }

  .brand__mark i {
    width: 8px;
  }

  .brand__mark i:nth-child(1) {
    left: 18px;
    height: 18px;
    bottom: 19px;
  }

  .brand__mark i:nth-child(2) {
    left: calc(50% - 4px);
    height: 26px;
    bottom: 16px;
  }

  .brand__mark i:nth-child(3) {
    right: 18px;
    height: 34px;
    bottom: 12px;
  }

  .brand p {
    display: none;
  }

  .brand strong {
    font-size: 1rem;
    letter-spacing: 0.04em;
  }

  .nav-chip {
    min-height: 54px;
    padding: 10px;
  }

  .device-card {
    grid-template-columns: 1fr;
  }

  .device-card__media {
    width: 72px;
    height: 72px;
  }
}

@media (max-width: 720px) {
  .app-shell {
    --sidebar-width: 144px;
  }

  .topbar {
    padding: 12px 8px 10px;
  }

  .brand {
    gap: 10px;
  }

  .brand__mark {
    width: 48px;
    height: 48px;
    border-radius: 14px;
  }

  .brand__mark::after {
    inset: 6px;
    border-radius: 10px;
  }

  .brand__mark i {
    width: 7px;
  }

  .brand__mark i:nth-child(1) {
    left: 14px;
    height: 14px;
    bottom: 16px;
  }

  .brand__mark i:nth-child(2) {
    left: calc(50% - 3.5px);
    height: 20px;
    bottom: 13px;
  }

  .brand__mark i:nth-child(3) {
    right: 14px;
    height: 26px;
    bottom: 10px;
  }

  .brand strong {
    font-size: 0.82rem;
    line-height: 1.25;
  }

  .brand__eyebrow,
  .brand p {
    display: none;
  }

  .nav-chip {
    min-height: 48px;
    padding: 10px 6px;
  }

  .nav-chip__label {
    font-size: 13px;
    line-height: 1.2;
  }

  .map-command-screen {
    min-height: 760px;
  }

  .map-command-screen__shade {
    background:
      linear-gradient(180deg, rgba(248, 250, 252, 0.94) 0%, rgba(248, 250, 252, 0.72) 44%, rgba(248, 250, 252, 0.2) 100%),
      linear-gradient(0deg, rgba(15, 23, 42, 0.18), transparent 42%);
  }

  .map-command-screen__header {
    padding: 22px;
  }

  .map-command-screen__status {
    margin: 20px 22px 0;
    grid-template-columns: 1fr 1fr;
  }

  .map-command-screen__status span {
    padding: 12px;
  }

  .map-command-screen__mission {
    width: auto;
    margin: 26px 22px 0;
    padding: 18px;
  }

  .mission-path {
    display: none;
  }

  .mission-link {
    grid-template-columns: 1fr;
  }

  .mission-link span {
    grid-row: auto;
    text-align: left;
  }

  .map-command-screen__timeline {
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 14px 6px;
    width: auto;
    max-width: none;
    margin: 18px 22px 0;
    padding: 18px 12px 14px;
    border-radius: 8px;
  }

  .map-command-screen__timeline::before,
  .map-command-screen__timeline::after {
    display: none;
  }

  .mission-step {
    min-height: 62px;
    flex-direction: row;
    justify-content: flex-start;
    align-items: center;
    gap: 10px;
    text-align: left;
    padding: 8px;
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.56);
  }

  .mission-step i {
    flex: 0 0 auto;
    width: 36px;
    height: 36px;
    box-shadow: none;
  }

  .mission-step__copy {
    min-height: auto;
  }

  .mission-step em {
    margin-left: auto;
  }

  .mission-step--active {
    transform: none;
    background: rgba(240, 249, 255, 0.86);
  }

  .mission-step--active i {
    width: 40px;
    height: 40px;
  }

  .panel-shell--hero,
  .module-hero,
  .algorithm-stage,
  .scene-stage,
  .device-showcase,
  .device-modal__panel,
  .module-card,
  .stat-card,
  .info-card {
    padding: 20px;
  }

  .scene-cockpit__stage {
    min-height: 340px;
  }

  .legend-card,
  .monitor-card {
    width: 128px;
    padding: 9px 10px;
  }

  .scene-cockpit__legend,
  .scene-cockpit__monitor,
  .scene-cockpit__map-meta {
    transform: scale(0.94);
    transform-origin: bottom left;
  }

  .landing-hero__copy h1,
  .module-hero h1 {
    line-height: 1;
  }

  .landing-hero__copy h1 {
    white-space: normal;
  }
}
</style>
