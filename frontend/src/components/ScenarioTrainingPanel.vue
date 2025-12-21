<template>
  <div class="training-panel">
    <div class="panel-header">
      <div>
        <h2>灾害场景训练</h2>
        <p>选择场景，触发 PPO 训练，并通过事件流实时查看指标。</p>
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
      <form class="training-form" @submit.prevent="startTraining">
        <label>
          总训练步数
          <input type="number" min="2000" step="1000" v-model.number="totalTimesteps" />
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
import { onMounted, ref, computed } from "vue";
import axios from "axios";
import TrainingMonitor from "./TrainingMonitor.vue";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000/api";

const scenarios = ref([]);
const selectedScenario = ref(null);
const totalTimesteps = ref(12000);
const isStarting = ref(false);
const eventLog = ref([]);
const runStatus = ref("Idle");
let eventSource = null;

const currentScenario = computed(() => scenarios.value.find((item) => item.name === selectedScenario.value));

const fetchScenarios = async () => {
  try {
    const { data } = await axios.get(`${API_BASE}/scenarios`);
    scenarios.value = data.scenarios || [];
    if (!selectedScenario.value && scenarios.value.length) {
      selectedScenario.value = scenarios.value[0].name;
    }
  } catch (error) {
    console.error("Failed to load scenarios", error);
  }
};

const selectScenario = (scenarioName) => {
  selectedScenario.value = scenarioName;
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
      total_timesteps: totalTimesteps.value,
      stochastic_eval: true,
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
