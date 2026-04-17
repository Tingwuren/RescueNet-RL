<template>
  <div class="training-panel">
    <div class="panel-header">
      <div class="scenario-select">
        <label>训练场景</label>
        <div class="scenario-options">
          <button
            v-for="scenario in scenarios"
            :key="scenario.name"
            :class="['scenario-chip', { 'scenario-chip--active': scenario.name === selectedScenario }]"
            @click="() => selectScenario(scenario.name)"
          >
            <strong>{{ formatScenarioName(scenario.name) }}</strong>
          </button>
        </div>
      </div>
    </div>

    <div v-if="loadError || actionError" class="error-banner">
      {{ loadError || actionError }}
    </div>

    <div class="panel-body">
      <div class="scenario-details" v-if="currentScenario">
        <article v-for="metric in scenarioMetrics" :key="metric.label" class="scenario-metric">
          <span>{{ metric.label }}</span>
          <strong>{{ metric.value }}</strong>
          <small>{{ metric.hint }}</small>
        </article>
      </div>
      <div class="reward-panel" v-if="rewardProfiles.length">
        <div class="reward-panel__header">
          <h3>奖励函数配置</h3>
          <p>
            当前选择：
            <strong>{{ activeRewardProfile?.label || "默认" }}</strong>
            <span v-if="activeRewardProfile?.description"> · {{ activeRewardProfile.description }}</span>
          </p>
        </div>
        <div class="reward-grid">
          <button
            v-for="profile in rewardProfiles"
            :key="profile.key"
            type="button"
            class="reward-card"
            :class="{ 'reward-card--active': profile.key === selectedRewardMode }"
            @click="() => selectRewardMode(profile.key)"
          >
            <div class="reward-card__title">
              <strong>{{ profile.label }}</strong>
              <small>{{ profile.description }}</small>
            </div>
            <div class="reward-card__weights">
              <span>覆盖 {{ formatWeight(profile.coverage_weight) }}</span>
              <span>带宽 {{ formatWeight(profile.bandwidth_weight) }}</span>
              <span>吞吐 {{ formatWeight(profile.throughput_weight) }}</span>
              <span>设备成本 {{ formatWeight(profile.device_cost_weight) }}</span>
              <span>带宽成本 {{ formatWeight(profile.bandwidth_cost_weight) }}</span>
            </div>
          </button>
        </div>
      </div>
      <form class="training-form" @submit.prevent="startTraining">
        <div class="training-form__header">
          <div>
            <h3>实验参数配置</h3>
            <p>默认参数可直接运行，也可以按实验需求手动修改。</p>
          </div>
          <span>Preset: disaster-rl-default</span>
        </div>

        <section class="config-section">
          <div class="config-section__title">
            <strong>基础训练参数</strong>
            <small>直接参与后端训练请求</small>
          </div>
          <div class="config-grid">
            <label class="config-field">
              <span>总训练步数</span>
              <input type="number" min="2000" step="1000" v-model.number="totalTimesteps" />
            </label>
            <label class="config-field">
              <span>环境类型</span>
              <select v-model="envType">
                <option value="multimodal">多模融合环境</option>
                <option value="baseline">基线环境</option>
              </select>
            </label>
            <label class="config-field">
              <span>评估方式</span>
              <select v-model="stochasticEval">
                <option :value="true">随机策略评估</option>
                <option :value="false">确定性策略评估</option>
              </select>
            </label>
          </div>
        </section>

        <section class="config-section">
          <div class="config-section__title">
            <strong>算法选择</strong>
            <small>选择训练策略，未接入项保持禁用</small>
          </div>
          <div class="algo-options">
            <button
              type="button"
              v-for="algo in algorithms"
              :key="algo.value"
              :class="['algo-chip', { 'algo-chip--active': algo.value === selectedAlgorithm }]"
              :disabled="algo.disabled"
              :title="algo.disabled ? '预留按钮，暂未接入训练后端' : ''"
              @click="() => !algo.disabled && selectAlgorithm(algo.value)"
            >
              <strong>{{ algo.label }}</strong>
              <small>{{ algo.desc }}</small>
            </button>
          </div>
        </section>

        <section class="config-section">
          <div class="config-section__title">
            <strong>高级算法参数</strong>
            <small>用于实验记录与后续算法扩展</small>
          </div>
          <div class="config-grid">
            <label class="config-field">
              <span>学习率</span>
              <input type="number" min="0.00001" max="0.01" step="0.00001" v-model.number="learningRate" />
            </label>
            <label class="config-field">
              <span>折扣因子 γ</span>
              <input type="number" min="0.8" max="0.999" step="0.001" v-model.number="discountFactor" />
            </label>
            <label class="config-field">
              <span>Batch Size</span>
              <input type="number" min="32" max="2048" step="32" v-model.number="batchSize" />
            </label>
            <label class="config-field">
              <span>Rollout 步长</span>
              <input type="number" min="64" max="4096" step="64" v-model.number="rolloutSteps" />
            </label>
            <label class="config-field">
              <span>熵系数</span>
              <input type="number" min="0" max="0.2" step="0.001" v-model.number="entropyCoef" />
            </label>
            <label class="config-field">
              <span>Clip Range</span>
              <input type="number" min="0.05" max="0.5" step="0.01" v-model.number="clipRange" />
            </label>
          </div>
        </section>

        <section class="config-section">
          <div class="config-section__title">
            <strong>输出与演示参数</strong>
            <small>控制日志、回放和可视化输出</small>
          </div>
          <div class="config-grid">
            <label class="config-field">
              <span>日志刷新窗口</span>
              <input type="number" min="10" max="200" step="5" v-model.number="logWindow" />
            </label>
            <label class="config-field">
              <span>评估间隔</span>
              <input type="number" min="1000" max="50000" step="1000" v-model.number="evalInterval" />
            </label>
            <label class="config-field">
              <span>训练后回放</span>
              <select v-model="autoReplay">
                <option :value="true">自动生成回放</option>
                <option :value="false">仅保留训练日志</option>
              </select>
            </label>
          </div>
        </section>

        <button type="submit" :disabled="!selectedScenario || isStarting || isLoadingScenarios">
          {{ isStarting ? "启动中..." : "开始训练" }}
        </button>
        <p v-if="!selectedScenario && !isLoadingScenarios" class="form-hint">
          训练场景未加载成功，请先确认前端可访问训练后端。
        </p>
      </form>
    </div>

    <TrainingMonitor :events="eventLog" :status="runStatus" />
  </div>
</template>

<script setup>
import { onMounted, ref, computed, watch } from "vue";
import axios from "axios";
import TrainingMonitor from "./TrainingMonitor.vue";
import { buildRegionMetrics, formatDistance } from "../utils/regionMetrics";
import { rescueApiBase } from "../utils/runtimeEndpoints";
import { saveReplaySessionFromSimulation } from "../utils/replaySessions";
import { formatScenarioName } from "../utils/scenarioLabels";

const API_BASE = rescueApiBase;

const scenarios = ref([]);
const algorithms = [
  { value: "ppo", label: "PPO", desc: "基线" },
  { value: "dqn", label: "DQN", desc: "大动作空间" },
  { value: "a3c", label: "A3C", desc: "多目标" },
  { value: "mppo", label: "MPPO", desc: "多头策略" },
  { value: "custom", label: "自创算法", desc: "预留中", disabled: true },
];
const selectedScenario = ref(null);
const selectedRewardMode = ref(null);
const selectedAlgorithm = ref("ppo");
const totalTimesteps = ref(12000);
const envType = ref("multimodal");
const stochasticEval = ref(true);
const learningRate = ref(0.0003);
const discountFactor = ref(0.99);
const batchSize = ref(256);
const rolloutSteps = ref(1024);
const entropyCoef = ref(0.01);
const clipRange = ref(0.2);
const logWindow = ref(50);
const evalInterval = ref(5000);
const autoReplay = ref(true);
const isStarting = ref(false);
const isLoadingScenarios = ref(false);
const eventLog = ref([]);
const runStatus = ref("Idle");
const loadError = ref("");
const actionError = ref("");
const activeRunMeta = ref(null);
const replayRunIdInFlight = ref(null);
let eventSource = null;

const currentScenario = computed(() => scenarios.value.find((item) => item.name === selectedScenario.value));
const rewardProfiles = computed(() => currentScenario.value?.reward_profiles || []);
const regionMetrics = computed(() => buildRegionMetrics(currentScenario.value?.region_grid));
const scenarioMetrics = computed(() => {
  if (!currentScenario.value) return [];
  const metrics = [
    {
      label: "用户规模",
      value: Number(currentScenario.value.num_users || 0).toLocaleString("zh-CN"),
      hint: "受灾终端数量",
    },
    {
      label: "候选站点",
      value: Number(currentScenario.value.candidate_sites || 0).toLocaleString("zh-CN"),
      hint: "可部署位置",
    },
    {
      label: "最大步长",
      value: Number(currentScenario.value.max_steps || 0).toLocaleString("zh-CN"),
      hint: "单次训练上限",
    },
  ];
  if (regionMetrics.value) {
    metrics.push(
      {
        label: "区域跨度",
        value: `${formatDistance(regionMetrics.value.widthKm)} × ${formatDistance(regionMetrics.value.heightKm)}`,
        hint: "灾区覆盖范围",
      },
      {
        label: "单网格",
        value: `${formatDistance(regionMetrics.value.cellWidthKm)} × ${formatDistance(regionMetrics.value.cellHeightKm)}`,
        hint: "空间离散粒度",
      }
    );
  }
  return metrics;
});
const activeRewardProfile = computed(() =>
  rewardProfiles.value.find((profile) => profile.key === selectedRewardMode.value)
);

watch(selectedAlgorithm, (algorithm) => {
  if (algorithm === "dqn" && totalTimesteps.value < 40000) {
    totalTimesteps.value = 40000;
    return;
  }
  if (algorithm !== "dqn" && totalTimesteps.value === 40000) {
    totalTimesteps.value = 12000;
  }
});

const fetchScenarios = async () => {
  isLoadingScenarios.value = true;
  loadError.value = "";
  try {
    const { data } = await axios.get(`${API_BASE}/scenarios`);
    scenarios.value = data.scenarios || [];
    if (!selectedScenario.value && scenarios.value.length) {
      selectedScenario.value = scenarios.value[0].name;
    }
    initializeRewardMode(selectedScenario.value);
  } catch (error) {
    console.error("Failed to load scenarios", error);
    loadError.value = `无法连接训练后端: ${error?.message || "未知错误"}`;
  } finally {
    isLoadingScenarios.value = false;
  }
};

const selectScenario = (scenarioName) => {
  selectedScenario.value = scenarioName;
  initializeRewardMode(scenarioName);
};

const initializeRewardMode = (scenarioName) => {
  if (!scenarioName) return;
  const scenario = scenarios.value.find((item) => item.name === scenarioName);
  if (!scenario) return;
  const defaultKey =
    scenario.default_reward_profile ||
    (Array.isArray(scenario.reward_profiles) && scenario.reward_profiles.length
      ? scenario.reward_profiles[0].key
      : null);
  selectedRewardMode.value = defaultKey;
};

const selectRewardMode = (modeKey) => {
  selectedRewardMode.value = modeKey;
};

const formatWeight = (value) => Number(value ?? 0).toFixed(2);

const selectAlgorithm = (value) => {
  selectedAlgorithm.value = value;
};

const resolveTrainingCheckpoint = async (runMeta) => {
  const matchesRun = (artifact) =>
    artifact?.checkpoint_path &&
    artifact?.scenario_name === runMeta.scenarioName &&
    artifact?.algorithm === runMeta.algorithm;

  try {
    const { data } = await axios.get(`${API_BASE}/train/latest-artifact`, { timeout: 10000 });
    if (matchesRun(data)) {
      return data.checkpoint_path;
    }
  } catch (error) {
    console.warn("Failed to load latest training artifact", error);
  }

  const { data } = await axios.get(`${API_BASE}/train/artifacts`, { timeout: 10000 });
  const match = (Array.isArray(data?.artifacts) ? data.artifacts : []).find(matchesRun);
  if (!match?.checkpoint_path) {
    throw new Error(`未找到 ${runMeta.scenarioName} / ${runMeta.algorithm.toUpperCase()} 的训练权重。`);
  }
  return match.checkpoint_path;
};

const generateReplayFromTraining = async (runMeta) => {
  if (!autoReplay.value) return;
  if (!runMeta?.runId) return;
  if (replayRunIdInFlight.value === runMeta.runId) return;
  replayRunIdInFlight.value = runMeta.runId;
  try {
    const checkpointPath = await resolveTrainingCheckpoint(runMeta);
    const { data } = await axios.post(`${API_BASE}/simulate`, {
      scenario_name: runMeta.scenarioName,
      env_type: "multimodal",
      algorithm: runMeta.algorithm,
      checkpoint_path: checkpointPath,
      reward_mode: runMeta.rewardMode,
      stochastic_eval: true,
      eval_seed: 13,
      episodes: 1,
    });
    const savedReplay = saveReplaySessionFromSimulation({
      scenarioName: runMeta.scenarioName,
      algorithm: runMeta.algorithm,
      result: {
        ...data,
        source: "training",
      },
    });
    eventLog.value = [
      ...eventLog.value.slice(-30),
      {
        type: "training_replay_ready",
        timestamp: Date.now() / 1000,
        payload: {
          scenario: runMeta.scenarioName,
          algorithm: runMeta.algorithm,
        },
        message: savedReplay?.persisted
          ? "训练完成后已自动生成一条回放，可在回放页刷新列表后查看。"
          : "训练完成后已生成回放，但浏览器本地存储空间不足，未能持久保存。",
      },
    ];
  } catch (error) {
    console.error("Failed to generate replay from training", error);
    eventLog.value = [
      ...eventLog.value.slice(-30),
      {
        type: "training_replay_error",
        timestamp: Date.now() / 1000,
        payload: {
          message: error?.message || "自动生成训练回放失败",
        },
      },
    ];
  } finally {
    replayRunIdInFlight.value = null;
  }
};

const startTraining = async () => {
  if (!selectedScenario.value) return;
  isStarting.value = true;
  actionError.value = "";
  eventLog.value = [];
  runStatus.value = "starting";
  activeRunMeta.value = null;
  replayRunIdInFlight.value = null;
  closeEventSource();
  try {
    const { data } = await axios.post(`${API_BASE}/train`, {
      scenario_name: selectedScenario.value,
      env_type: envType.value,
      algorithm: selectedAlgorithm.value,
      total_timesteps: totalTimesteps.value,
      stochastic_eval: stochasticEval.value,
      reward_mode: selectedRewardMode.value,
    });
    activeRunMeta.value = {
      runId: data.run_id,
      scenarioName: selectedScenario.value,
      algorithm: selectedAlgorithm.value,
      rewardMode: selectedRewardMode.value,
      config: {
        learningRate: learningRate.value,
        discountFactor: discountFactor.value,
        batchSize: batchSize.value,
        rolloutSteps: rolloutSteps.value,
        entropyCoef: entropyCoef.value,
        clipRange: clipRange.value,
        logWindow: logWindow.value,
        evalInterval: evalInterval.value,
        autoReplay: autoReplay.value,
      },
    };
    eventLog.value = [
      {
        type: "experiment_config",
        timestamp: Date.now() / 1000,
        payload: activeRunMeta.value.config,
        message: "已加载前端实验参数配置，后端开始执行训练任务。",
      },
    ];
    subscribeToEvents(data.run_id);
  } catch (error) {
    console.error("Failed to start training", error);
    runStatus.value = "error";
    actionError.value = `启动训练失败: ${error?.message || "未知错误"}`;
  } finally {
    isStarting.value = false;
  }
};

const subscribeToEvents = (runId) => {
  runStatus.value = "running";
  const streamUrl = `${API_BASE}/train/${runId}/stream`;
  eventSource = new EventSource(streamUrl);
  eventSource.onmessage = (event) => {
    if (!event.data) return;
    try {
      const payload = JSON.parse(event.data);
      if (payload.type === "end") {
        runStatus.value = payload.status;
        closeEventSource();
        return;
      }
      eventLog.value = [...eventLog.value.slice(-30), payload];
      if (payload.type === "status" && payload.payload?.state) {
        runStatus.value = payload.payload.state;
        if (payload.payload.state === "completed" && activeRunMeta.value?.runId === runId) {
          void generateReplayFromTraining(activeRunMeta.value);
        }
      }
    } catch (err) {
      console.warn("Failed to parse event", err);
    }
  };
  eventSource.onerror = () => {
    closeEventSource();
    runStatus.value = "disconnected";
  };
};

const closeEventSource = () => {
  if (eventSource) {
    eventSource.close();
    eventSource = null;
  }
};

watch(currentScenario, (scenario) => {
  if (!scenario) {
    selectedRewardMode.value = null;
    return;
  }
  const availableKeys = (scenario.reward_profiles || []).map((profile) => profile.key);
  if (availableKeys.length === 0) {
    selectedRewardMode.value = null;
    return;
  }
  if (!availableKeys.includes(selectedRewardMode.value)) {
    const fallback = scenario.default_reward_profile || availableKeys[0];
    selectedRewardMode.value = fallback;
  }
});

onMounted(fetchScenarios);
</script>

<style scoped>
.training-panel {
  display: flex;
  flex-direction: column;
  gap: 24px;
}

.error-banner {
  border: 1px solid rgba(248, 113, 113, 0.55);
  border-radius: 12px;
  padding: 12px 14px;
  background: rgba(127, 29, 29, 0.25);
  color: #fecaca;
}

.panel-header {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.scenario-select label {
  font-size: 12px;
  color: #475569;
}

.scenario-options {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 8px;
}

.scenario-chip {
  background: rgba(248, 250, 252, 0.94);
  border: 1px solid rgba(100, 116, 139, 0.28);
  border-radius: 10px;
  padding: 10px 16px;
  color: #0f172a;
  min-width: 140px;
  text-align: left;
  box-shadow: 0 8px 18px rgba(15, 23, 42, 0.06);
}

.scenario-chip--active {
  border-color: #0284c7;
  background: rgba(224, 242, 254, 0.95);
  color: #075985;
  box-shadow: 0 0 0 2px rgba(14, 165, 233, 0.18);
}

.panel-body {
  display: grid;
  grid-template-columns: repeat(12, minmax(0, 1fr));
  gap: 24px;
  align-items: start;
}

.scenario-details {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 12px;
}

.scenario-metric {
  min-height: 104px;
  border: 1px solid rgba(14, 165, 233, 0.18);
  border-radius: 16px;
  padding: 14px;
  background:
    radial-gradient(circle at 100% 0%, rgba(56, 189, 248, 0.12), transparent 42%),
    rgba(255, 255, 255, 0.92);
  box-shadow: 0 12px 24px rgba(15, 23, 42, 0.06);
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  color: #0f172a;
}

.scenario-metric span {
  font-size: 12px;
  letter-spacing: 0.08em;
  color: #64748b;
}

.scenario-metric strong {
  font-size: 20px;
  line-height: 1.2;
  color: #075985;
}

.scenario-metric small {
  color: #64748b;
}

.reward-panel {
  grid-column: span 6;
  border: 1px solid rgba(100, 116, 139, 0.22);
  border-radius: 16px;
  padding: 16px;
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.94), rgba(248, 250, 252, 0.9));
  display: flex;
  flex-direction: column;
  gap: 12px;
  color: #1e293b;
  min-height: 100%;
  box-shadow: 0 14px 28px rgba(15, 23, 42, 0.06);
}

.reward-panel__header h3 {
  margin: 0;
  font-size: 16px;
}

.reward-panel__header p {
  margin: 4px 0 0;
  font-size: 12px;
  color: #475569;
}

.reward-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 14px;
}

.reward-card {
  min-height: 168px;
  border-radius: 14px;
  border: 1px solid rgba(100, 116, 139, 0.24);
  padding: 16px;
  text-align: left;
  background: rgba(255, 255, 255, 0.96);
  color: #0f172a;
  transition: border-color 0.2s ease, background 0.2s ease;
  cursor: pointer;
  box-shadow: 0 8px 18px rgba(15, 23, 42, 0.05);
}

.reward-card--active {
  border-color: #0284c7;
  background: rgba(224, 242, 254, 0.96);
  box-shadow: 0 0 0 2px rgba(14, 165, 233, 0.16);
}

.reward-card__title {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin-bottom: 14px;
}

.reward-card__title strong {
  font-size: 16px;
}

.reward-card__title small {
  font-size: 12px;
  color: #64748b;
  line-height: 1.5;
}

.reward-card__weights {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(128px, 1fr));
  gap: 8px;
  font-size: 12px;
  color: #475569;
}

.reward-card__weights span {
  padding: 7px 9px;
  border-radius: 999px;
  background: rgba(241, 245, 249, 0.9);
  border: 1px solid rgba(148, 163, 184, 0.14);
}

.algo-options {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 8px;
}

.algo-chip {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 4px;
  padding: 10px 12px;
  border-radius: 10px;
  border: 1px solid rgba(100, 116, 139, 0.24);
  background: rgba(255, 255, 255, 0.94);
  color: #0f172a;
  cursor: pointer;
  transition: all 0.2s ease;
  box-shadow: 0 8px 18px rgba(15, 23, 42, 0.05);
}

.algo-chip:disabled {
  cursor: not-allowed;
  opacity: 0.55;
}

.algo-chip--active {
  border-color: #0284c7;
  box-shadow: 0 0 0 2px rgba(14, 165, 233, 0.16);
  background: rgba(224, 242, 254, 0.96);
  color: #075985;
}

.algo-chip strong {
  font-size: 14px;
}

.algo-chip small {
  color: #64748b;
}

.training-form {
  grid-column: span 6;
  display: flex;
  flex-direction: column;
  gap: 14px;
  border: 1px solid rgba(100, 116, 139, 0.22);
  border-radius: 16px;
  padding: 16px;
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.94), rgba(248, 250, 252, 0.9));
  box-shadow: 0 14px 28px rgba(15, 23, 42, 0.06);
  min-height: 100%;
}

.training-form__header {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: flex-start;
  padding-bottom: 10px;
  border-bottom: 1px solid rgba(148, 163, 184, 0.18);
}

.training-form__header h3 {
  margin: 0;
  font-size: 17px;
  color: #0f172a;
}

.training-form__header p {
  margin: 4px 0 0;
  color: #64748b;
  font-size: 12px;
}

.training-form__header > span {
  flex: 0 0 auto;
  padding: 7px 10px;
  border-radius: 999px;
  background: rgba(224, 242, 254, 0.8);
  color: #075985;
  font-size: 11px;
  border: 1px solid rgba(14, 165, 233, 0.18);
}

.config-section {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 12px;
  border-radius: 14px;
  background: rgba(241, 245, 249, 0.62);
  border: 1px solid rgba(148, 163, 184, 0.14);
}

.config-section__title {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: baseline;
}

.config-section__title strong {
  color: #0f172a;
  font-size: 14px;
}

.config-section__title small {
  color: #64748b;
  font-size: 11px;
}

.config-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 10px;
}

.config-field {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.config-field span,
.training-form label > span {
  color: #475569;
  font-size: 12px;
  font-weight: 600;
}

.form-hint {
  margin: 0;
  color: #fca5a5;
  font-size: 0.92rem;
}

input[type="number"],
select {
  width: 100%;
  padding: 10px 12px;
  border-radius: 8px;
  border: 1px solid rgba(100, 116, 139, 0.28);
  background: rgba(255, 255, 255, 0.95);
  color: #0f172a;
}

button[type="submit"] {
  padding: 12px;
  border: none;
  border-radius: 999px;
  background: linear-gradient(90deg, #2563eb, #0ea5e9);
  color: #fff;
  font-weight: 600;
  transition: opacity 0.2s ease;
}

button:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

@media (max-width: 1100px) {
  .reward-panel,
  .training-form {
    grid-column: 1 / -1;
  }

  .scenario-details {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 720px) {
  .scenario-details {
    grid-template-columns: 1fr;
  }
}
</style>
