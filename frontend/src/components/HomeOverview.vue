<template>
  <section class="overview">
    <div v-if="errorMessage" class="overview__error">{{ errorMessage }}</div>

    <section class="overview__hero">
      <div class="overview__copy">
        <span class="overview__eyebrow">核心态势</span>
        <h2>从灾情接入、策略训练到组网回放的恢复过程演示</h2>
        <p>
          当前首页接入真实场景元数据、训练产物状态、Mahimahi 链路能力和 ns-3 实验状态，
          作为全局调度入口。
        </p>

        <div class="overview__actions">
          <a href="#/train">进入模型训练</a>
          <a href="#/tester" class="overview__actions-secondary">前往策略测试</a>
        </div>

        <div class="overview__headline-metrics">
          <article v-for="item in headlineMetrics" :key="item.label">
            <small>{{ item.label }}</small>
            <strong>{{ item.value }}</strong>
          </article>
        </div>
      </div>

      <div class="overview__stage">
        <div class="overview__toolbar">
          <label>
            <span>场景选择</span>
            <select v-model="selectedScenarioName">
              <option v-for="scenario in scenarios" :key="scenario.name" :value="scenario.name">
                {{ formatScenarioName(scenario.name) }}
              </option>
            </select>
          </label>
          <button type="button" @click="refreshAll">刷新数据</button>
        </div>

        <SceneGraphPreview
          v-if="scenePreview?.scene"
          :scene="scenePreview.scene"
          :scenario-name="selectedScenarioName"
          title="灾情场景预览"
          subtitle="基于 /api/simulate/scene 的实时预览"
        />
        <div v-else class="overview__placeholder">正在同步场景图…</div>
      </div>
    </section>

    <section class="overview__panel">
      <div class="overview__panel-header">
        <div>
          <h3>核心指标</h3>
          <p>切换不同指标组查看灾情、训练和链路态势。</p>
        </div>
        <div class="overview__tabs">
          <button
            v-for="group in metricGroups"
            :key="group.key"
            type="button"
            :class="{ 'is-active': activeMetricGroup === group.key }"
            @click="activeMetricGroup = group.key"
          >
            {{ group.label }}
          </button>
        </div>
      </div>

      <div class="overview__metric-grid">
        <article v-for="item in activeMetrics" :key="item.label" class="overview__metric-card">
          <small>{{ item.label }}</small>
          <strong>{{ item.value }}</strong>
          <p>{{ item.hint }}</p>
        </article>
      </div>
    </section>

    <section class="overview__summary-grid">
      <article class="overview__summary-card">
        <span>最新训练产物</span>
        <strong>{{ latestArtifactLabel }}</strong>
        <p>{{ latestArtifactNote }}</p>
      </article>
      <article class="overview__summary-card">
        <span>场景回放</span>
        <strong>{{ replayCountText }}</strong>
        <p>由训练或策略测试真实结果生成，可直接进入场景回放页逐帧查看。</p>
      </article>
      <article class="overview__summary-card">
        <span>链路仿真</span>
        <strong>{{ linkStatusText }}</strong>
        <p>{{ linkStatusNote }}</p>
      </article>
    </section>
  </section>
</template>

<script setup>
import { computed, onMounted, ref, watch } from "vue";
import axios from "axios";

import SceneGraphPreview from "./SceneGraphPreview.vue";
import { listReplaySessions } from "../utils/replaySessions";
import { rescueApiBase } from "../utils/runtimeEndpoints";
import { formatDisasterType, formatScenarioName } from "../utils/scenarioLabels";

const API_BASE = rescueApiBase;

const scenarios = ref([]);
const selectedScenarioName = ref("");
const scenePreview = ref(null);
const latestArtifact = ref(null);
const artifactCount = ref(0);
const ns3Status = ref(null);
const mahimahiStatus = ref(null);
const mahimahiTraceCount = ref(0);
const replayCount = ref(0);
const activeMetricGroup = ref("core");
const errorMessage = ref("");

const metricGroups = [
  { key: "core", label: "灾情指标" },
  { key: "training", label: "训练指标" },
  { key: "network", label: "链路指标" },
];

const currentScenario = computed(() =>
  scenarios.value.find((scenario) => scenario.name === selectedScenarioName.value) || null
);

const initialState = computed(() => scenePreview.value?.initial_state || null);

const percentageText = (value) => `${(Math.max(0, Math.min(1, Number(value || 0))) * 100).toFixed(1)}%`;
const timeText = (value) => {
  if (!value) return "暂无";
  return new Date(Number(value) * 1000).toLocaleString("zh-CN", { hour12: false });
};

const headlineMetrics = computed(() => [
  {
    label: "灾害类型",
    value: formatDisasterType(currentScenario.value?.disaster_type),
  },
  {
    label: "断联用户",
    value: Number(initialState.value?.total_users || currentScenario.value?.num_users || 0).toLocaleString("zh-CN"),
  },
  {
    label: "残余基站",
    value: String((initialState.value?.residual_base_stations || []).length || 0),
  },
  {
    label: "广播可达率",
    value: percentageText(initialState.value?.broadcast_ratio || 0),
  },
]);

const metricMap = computed(() => ({
  core: [
    {
      label: "场景名称",
      value: formatScenarioName(currentScenario.value?.name),
      hint: "当前灾害场景",
    },
    {
      label: "候选站点",
      value: Number(currentScenario.value?.candidate_sites || 0).toLocaleString("zh-CN"),
      hint: "可供部署的站点数量",
    },
    {
      label: "用户规模",
      value: Number(currentScenario.value?.num_users || 0).toLocaleString("zh-CN"),
      hint: "场景总用户数",
    },
    {
      label: "覆盖率",
      value: percentageText(initialState.value?.coverage_ratio || 0),
      hint: "导入场景初始覆盖率",
    },
  ],
  training: [
    {
      label: "最新算法",
      value: latestArtifact.value?.algorithm?.toUpperCase() || "--",
      hint: "来自 /api/train/latest-artifact",
    },
    {
      label: "奖励模式",
      value: latestArtifact.value?.reward_mode || currentScenario.value?.default_reward_profile || "--",
      hint: "最近一次训练配置",
    },
    {
      label: "训练产物数",
      value: artifactCount.value.toLocaleString("zh-CN"),
      hint: "来自 artifacts 目录",
    },
    {
      label: "最近更新时间",
      value: latestArtifact.value?.updated_at ? timeText(latestArtifact.value.updated_at) : "暂无",
      hint: "最近产物落盘时间",
    },
  ],
  network: [
    {
      label: "Mahimahi 可用",
      value: mahimahiStatus.value?.mahimahi_available ? "是" : "否",
      hint: "当前链路模拟后端状态",
    },
    {
      label: "Trace 数量",
      value: mahimahiTraceCount.value.toLocaleString("zh-CN"),
      hint: "可用链路轨迹文件",
    },
    {
      label: "ns-3 实验数",
      value: Number(ns3Status.value?.experiment_count || 0).toLocaleString("zh-CN"),
      hint: "当前数据库中可回放实验",
    },
    {
      label: "仿真状态",
      value: ns3Status.value?.running ? "运行中" : "空闲",
      hint: "ns-3 任务进程状态",
    },
  ],
}));

const activeMetrics = computed(() => metricMap.value[activeMetricGroup.value] || metricMap.value.core);

const latestArtifactLabel = computed(() => {
  if (!latestArtifact.value) return "暂无训练产物";
  const scenario = formatScenarioName(latestArtifact.value.scenario_name);
  return `${latestArtifact.value.algorithm?.toUpperCase()} / ${scenario}`;
});

const latestArtifactNote = computed(() => {
  if (!latestArtifact.value) return "训练页仍可直接启动真实训练任务并刷新首页状态。";
  return `环境 ${latestArtifact.value.env_type || "--"}，奖励模式 ${latestArtifact.value.reward_mode || "--"}。`;
});

const replayCountText = computed(() => `${replayCount.value} 条可回放记录`);

const linkStatusText = computed(() =>
  ns3Status.value?.running ? "ns-3 运行中" : mahimahiStatus.value?.mahimahi_available ? "Mahimahi 已接入" : "链路服务待检查"
);

const linkStatusNote = computed(() => {
  const experiments = Number(ns3Status.value?.experiment_count || 0);
  return `ns-3 实验 ${experiments} 条，Trace ${mahimahiTraceCount.value} 条。`;
});

const fetchScenePreview = async () => {
  if (!selectedScenarioName.value) return;
  try {
    const { data } = await axios.post(`${API_BASE}/simulate/scene`, {
      scenario_name: selectedScenarioName.value,
      env_type: "multimodal",
      custom_base_stations: null,
    });
    scenePreview.value = data;
  } catch (error) {
    console.error("Failed to fetch scene preview", error);
    errorMessage.value = "场景预览加载失败，请检查后端服务状态。";
  }
};

const refreshAll = async () => {
  errorMessage.value = "";
  try {
    const [
      scenariosResp,
      latestArtifactResp,
      artifactsResp,
      ns3Resp,
      mahimahiResp,
      tracesResp,
    ] = await Promise.all([
      axios.get(`${API_BASE}/scenarios`),
      axios.get(`${API_BASE}/train/latest-artifact`).catch(() => ({ data: null })),
      axios.get(`${API_BASE}/train/artifacts`).catch(() => ({ data: { artifacts: [] } })),
      axios.get(`${API_BASE}/ns3/status`).catch(() => ({ data: null })),
      axios.get(`${API_BASE}/mahimahi/status`).catch(() => ({ data: null })),
      axios.get(`${API_BASE}/mahimahi/traces`).catch(() => ({ data: { traces: [] } })),
    ]);

    scenarios.value = Array.isArray(scenariosResp.data?.scenarios) ? scenariosResp.data.scenarios : [];
    if (!selectedScenarioName.value && scenarios.value.length) {
      selectedScenarioName.value = scenarios.value[0].name;
    }

    latestArtifact.value = latestArtifactResp.data;
    artifactCount.value = Array.isArray(artifactsResp.data?.artifacts) ? artifactsResp.data.artifacts.length : 0;
    ns3Status.value = ns3Resp.data;
    mahimahiStatus.value = mahimahiResp.data;
    mahimahiTraceCount.value = Array.isArray(tracesResp.data?.traces) ? tracesResp.data.traces.length : 0;
    replayCount.value = listReplaySessions().length;

    await fetchScenePreview();
  } catch (error) {
    console.error("Failed to refresh home overview", error);
    errorMessage.value = "首页数据加载失败，请确认 /api 可访问。";
  }
};

watch(selectedScenarioName, () => {
  void fetchScenePreview();
});

onMounted(() => {
  void refreshAll();
});
</script>

<style scoped>
.overview {
  display: flex;
  flex-direction: column;
  gap: 22px;
}

.overview__error {
  padding: 14px 16px;
  border-radius: 16px;
  border: 1px solid rgba(239, 68, 68, 0.18);
  background: rgba(254, 242, 242, 0.92);
  color: #b91c1c;
}

.overview__hero {
  display: grid;
  grid-template-columns: minmax(320px, 1.05fr) minmax(0, 1fr);
  gap: 22px;
  align-items: stretch;
}

.overview__copy,
.overview__stage,
.overview__panel,
.overview__summary-card {
  border: 1px solid rgba(148, 163, 184, 0.18);
  border-radius: 24px;
  background: rgba(255, 255, 255, 0.84);
  box-shadow: 0 18px 40px rgba(59, 130, 246, 0.08);
}

.overview__copy {
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  min-height: 520px;
  padding: 28px;
  background:
    radial-gradient(circle at 12% 18%, rgba(56, 189, 248, 0.16), transparent 28%),
    linear-gradient(135deg, rgba(15, 23, 42, 0.88), rgba(24, 58, 121, 0.82)),
    linear-gradient(180deg, rgba(255, 255, 255, 0.84), rgba(255, 255, 255, 0.84));
  color: #eff6ff;
}

.overview__eyebrow {
  display: inline-flex;
  width: fit-content;
  padding: 7px 12px;
  border-radius: 999px;
  border: 1px solid rgba(125, 211, 252, 0.22);
  color: #bae6fd;
  font-size: 12px;
  letter-spacing: 0.12em;
}

.overview__copy h2 {
  margin: 16px 0 10px;
  max-width: 12ch;
  font-size: 42px;
  line-height: 1.12;
}

.overview__copy p {
  max-width: 34rem;
  color: rgba(219, 234, 254, 0.86);
  line-height: 1.8;
}

.overview__actions {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin-top: 18px;
}

.overview__actions a {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  height: 46px;
  padding: 0 18px;
  border-radius: 12px;
  background: linear-gradient(135deg, #38bdf8, #2563eb);
  color: #ffffff;
  text-decoration: none;
  font-weight: 700;
}

.overview__actions-secondary {
  background: rgba(255, 255, 255, 0.1) !important;
  border: 1px solid rgba(191, 219, 254, 0.24);
}

.overview__headline-metrics {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
  margin-top: 26px;
}

.overview__headline-metrics article {
  padding: 14px 16px;
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.08);
  border: 1px solid rgba(191, 219, 254, 0.12);
}

.overview__headline-metrics small,
.overview__metric-card small,
.overview__summary-card span {
  display: block;
  font-size: 12px;
  letter-spacing: 0.08em;
}

.overview__headline-metrics small {
  color: rgba(191, 219, 254, 0.8);
}

.overview__headline-metrics strong,
.overview__metric-card strong,
.overview__summary-card strong {
  display: block;
  margin-top: 6px;
  font-size: 24px;
  font-weight: 700;
}

.overview__stage {
  padding: 18px;
}

.overview__toolbar {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: center;
  margin-bottom: 16px;
}

.overview__toolbar label {
  display: flex;
  flex-direction: column;
  gap: 6px;
  color: #496486;
  font-size: 13px;
}

.overview__toolbar select,
.overview__toolbar button {
  height: 40px;
  border-radius: 12px;
}

.overview__toolbar select {
  min-width: 220px;
  padding: 0 12px;
  border: 1px solid rgba(148, 163, 184, 0.22);
  background: #ffffff;
}

.overview__toolbar button {
  padding: 0 16px;
  border: 0;
  background: rgba(37, 99, 235, 0.1);
  color: #2563eb;
  font-weight: 700;
}

.overview__placeholder {
  display: grid;
  min-height: 460px;
  place-items: center;
  border-radius: 20px;
  background: linear-gradient(180deg, #f8fbff, #edf5ff);
  color: #6b87ae;
}

.overview__panel {
  padding: 22px;
}

.overview__panel-header {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: flex-start;
  margin-bottom: 16px;
}

.overview__panel-header h3 {
  margin: 0;
  font-size: 22px;
  color: #17315d;
}

.overview__panel-header p {
  margin: 6px 0 0;
  color: #6881a7;
}

.overview__tabs {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.overview__tabs button {
  height: 38px;
  padding: 0 16px;
  border-radius: 999px;
  border: 1px solid rgba(148, 163, 184, 0.18);
  background: #ffffff;
  color: #5b7599;
  font-weight: 600;
}

.overview__tabs .is-active {
  border-color: rgba(37, 99, 235, 0.3);
  color: #2563eb;
  background: rgba(37, 99, 235, 0.08);
}

.overview__metric-grid,
.overview__summary-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 14px;
}

.overview__metric-card {
  padding: 18px;
  border-radius: 20px;
  background: linear-gradient(180deg, #ffffff, #f7fbff);
  border: 1px solid rgba(148, 163, 184, 0.16);
}

.overview__metric-card small {
  color: #7189ab;
}

.overview__metric-card strong {
  color: #153464;
}

.overview__metric-card p,
.overview__summary-card p {
  margin: 8px 0 0;
  color: #6881a7;
  line-height: 1.7;
  font-size: 13px;
}

.overview__summary-card {
  padding: 18px;
}

.overview__summary-card span {
  color: #6881a7;
}

.overview__summary-card strong {
  color: #17315d;
}

@media (max-width: 1200px) {
  .overview__hero,
  .overview__metric-grid,
  .overview__summary-grid {
    grid-template-columns: 1fr;
  }

  .overview__copy {
    min-height: auto;
  }
}

@media (max-width: 720px) {
  .overview__copy,
  .overview__stage,
  .overview__panel,
  .overview__summary-card {
    border-radius: 18px;
  }

  .overview__copy {
    padding: 20px;
  }

  .overview__copy h2 {
    font-size: 30px;
  }

  .overview__toolbar,
  .overview__panel-header {
    flex-direction: column;
  }

  .overview__toolbar select {
    min-width: 0;
    width: 100%;
  }

  .overview__headline-metrics {
    grid-template-columns: 1fr;
  }
}
</style>
