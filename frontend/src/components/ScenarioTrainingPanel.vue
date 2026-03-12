<template>
  <div class="training-panel">
    <div class="panel-header">
      <div>
        <h2>灾害场景训练</h2>
        <p>选择场景，触发训练（支持 PPO / DQN / A3C / MPPO），并通过事件流实时查看指标。</p>
      </div>
      <div class="scenario-select">
        <label>训练场景</label>
        <div class="scenario-options">
          <button
            v-for="scenario in scenarios"
            :key="scenario.name"
            :class="['scenario-chip', { 'scenario-chip--active': scenario.name === selectedScenario }]"
            @click="() => selectScenario(scenario.name)"
          >
            <strong>{{ scenario.name }}</strong>
            <small>{{ scenario.disaster_type }}</small>
          </button>
        </div>
      </div>
    </div>

    <div class="panel-body">
      <div class="scenario-details" v-if="currentScenario">
        <p>用户数：{{ currentScenario.num_users }}</p>
        <p>候选站点：{{ currentScenario.candidate_sites }}</p>
        <p>最大步长：{{ currentScenario.max_steps }}</p>
      </div>
      <div class="reward-panel" v-if="rewardProfiles.length">
        <div class="reward-panel__header">
          <h3>奖励函数配置</h3>
          <p>
            当前选择：
            <strong>{{ activeRewardProfile?.label || "默认" }}</strong>
            <span v-if="activeRewardProfile?.description">（{{ activeRewardProfile.description }}）</span>
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
        <label>
          总训练步数
          <input type="number" min="2000" step="1000" v-model.number="totalTimesteps" />
        </label>
        <label>
          训练算法
          <div class="algo-options">
            <button
              type="button"
              v-for="algo in algorithms"
              :key="algo.value"
              :class="['algo-chip', { 'algo-chip--active': algo.value === selectedAlgorithm }]"
              @click="() => selectAlgorithm(algo.value)"
            >
              <strong>{{ algo.label }}</strong>
              <small>{{ algo.desc }}</small>
            </button>
          </div>
        </label>
        <button type="submit" :disabled="!selectedScenario || isStarting">
          {{ isStarting ? "启动中..." : "开始训练" }}
        </button>
      </form>
    </div>

    <TrainingMonitor :events="eventLog" :status="runStatus" />
  </div>
</template>

<script setup>
import { onMounted, ref, computed, watch } from "vue";
import axios from "axios";
import TrainingMonitor from "./TrainingMonitor.vue";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000/api";

const scenarios = ref([]);
const algorithms = [
  { value: "ppo", label: "PPO", desc: "基线" },
  { value: "dqn", label: "DQN", desc: "大动作空间" },
  { value: "a3c", label: "A3C", desc: "多目标" },
  { value: "mppo", label: "MPPO", desc: "多头策略" },
];
const selectedScenario = ref(null);
const selectedRewardMode = ref(null);
const selectedAlgorithm = ref("ppo");
const totalTimesteps = ref(12000);
const isStarting = ref(false);
const eventLog = ref([]);
const runStatus = ref("Idle");
let eventSource = null;

const currentScenario = computed(() => scenarios.value.find((item) => item.name === selectedScenario.value));
const rewardProfiles = computed(() => currentScenario.value?.reward_profiles || []);
const activeRewardProfile = computed(() =>
  rewardProfiles.value.find((profile) => profile.key === selectedRewardMode.value)
);

const fetchScenarios = async () => {
  try {
    const { data } = await axios.get(`${API_BASE}/scenarios`);
    scenarios.value = data.scenarios || [];
    if (!selectedScenario.value && scenarios.value.length) {
      selectedScenario.value = scenarios.value[0].name;
    }
    initializeRewardMode(selectedScenario.value);
  } catch (error) {
    console.error("Failed to load scenarios", error);
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

const startTraining = async () => {
  if (!selectedScenario.value) return;
  isStarting.value = true;
  eventLog.value = [];
  runStatus.value = "starting";
  closeEventSource();
  try {
    const { data } = await axios.post(`${API_BASE}/train`, {
      scenario_name: selectedScenario.value,
      env_type: "multimodal",
      algorithm: selectedAlgorithm.value,
      total_timesteps: totalTimesteps.value,
      stochastic_eval: true,
      reward_mode: selectedRewardMode.value,
    });
    subscribeToEvents(data.run_id);
  } catch (error) {
    console.error("Failed to start training", error);
    runStatus.value = "error";
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

.panel-header {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.scenario-select label {
  font-size: 12px;
  color: #94a3b8;
}

.scenario-options {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 8px;
}

.scenario-chip {
  background: rgba(30, 41, 59, 0.7);
  border: 1px solid rgba(148, 163, 184, 0.4);
  border-radius: 10px;
  padding: 10px 16px;
  color: inherit;
  min-width: 140px;
  text-align: left;
}

.scenario-chip--active {
  border-color: #38bdf8;
  background: rgba(56, 189, 248, 0.1);
}

.panel-body {
  display: flex;
  gap: 24px;
  flex-wrap: wrap;
}

.scenario-details {
  flex: 1 1 200px;
  border: 1px solid rgba(148, 163, 184, 0.3);
  border-radius: 12px;
  padding: 16px;
  background: rgba(15, 23, 42, 0.4);
}

.reward-panel {
  flex: 1 1 360px;
  border: 1px solid rgba(148, 163, 184, 0.3);
  border-radius: 12px;
  padding: 16px;
  background: rgba(15, 23, 42, 0.4);
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.reward-panel__header h3 {
  margin: 0;
  font-size: 16px;
}

.reward-panel__header p {
  margin: 4px 0 0;
  font-size: 12px;
  color: #cbd5f5;
}

.reward-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
}

.reward-card {
  flex: 1 1 160px;
  border-radius: 10px;
  border: 1px solid rgba(148, 163, 184, 0.3);
  padding: 10px 12px;
  text-align: left;
  background: rgba(15, 23, 42, 0.6);
  color: inherit;
  transition: border-color 0.2s ease, background 0.2s ease;
  cursor: pointer;
}

.reward-card--active {
  border-color: #38bdf8;
  background: rgba(56, 189, 248, 0.15);
}

.reward-card__title {
  display: flex;
  flex-direction: column;
  gap: 2px;
  margin-bottom: 8px;
}

.reward-card__title strong {
  font-size: 14px;
}

.reward-card__title small {
  font-size: 11px;
  color: #94a3b8;
}

.reward-card__weights {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
  gap: 4px;
  font-size: 11px;
  color: #cbd5f5;
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
  border: 1px solid rgba(148, 163, 184, 0.3);
  background: rgba(148, 163, 184, 0.08);
  color: #e2e8f0;
  cursor: pointer;
  transition: all 0.2s ease;
}

.algo-chip--active {
  border-color: #38bdf8;
  box-shadow: 0 0 0 1px rgba(56, 189, 248, 0.25);
  background: linear-gradient(120deg, rgba(56, 189, 248, 0.1), rgba(14, 165, 233, 0.08));
}

.algo-chip strong {
  font-size: 14px;
}

.algo-chip small {
  color: #94a3b8;
}

.training-form {
  flex: 1 1 240px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

input[type="number"] {
  width: 100%;
  padding: 10px 12px;
  border-radius: 8px;
  border: 1px solid rgba(148, 163, 184, 0.6);
  background: rgba(15, 23, 42, 0.2);
  color: inherit;
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
</style>
