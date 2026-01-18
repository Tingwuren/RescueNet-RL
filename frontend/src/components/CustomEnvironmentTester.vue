<template>
  <div class="tester">
    <h2>自定义环境测试</h2>
    <p class="subtitle">
      选择残余基站并设置位置，模拟部分基础设施仍然可用的场景；若不添加则表示完全受灾。
    </p>
    <div v-if="errorMessage" class="error-banner">
      {{ errorMessage }}
    </div>
    <form class="tester__form" @submit.prevent="runSimulation">
      <label>
        场景选择
        <select v-model="scenarioName">
          <option v-for="scenario in scenarios" :key="scenario.name" :value="scenario.name">
            {{ scenario.name }} ({{ scenario.disaster_type }})
          </option>
        </select>
      </label>
      <label>
        算法选择
        <div class="algo-options">
          <button
            type="button"
            v-for="algo in algorithms"
            :key="algo.value"
            :class="['algo-chip', { 'algo-chip--active': algo.value === selectedAlgorithm }]"
            @click="selectedAlgorithm = algo.value"
          >
            <strong>{{ algo.label }}</strong>
            <small>{{ algo.desc }}</small>
          </button>
        </div>
      </label>
      <label>
        Checkpoint 路径
        <input type="text" v-model="checkpointPath" placeholder="artifacts/ppo_policy.pt" />
      </label>
      <div class="devices">
        <div class="devices__header">
          <div>
            <p>基站列表</p>
            <small>未添加即表示无残余基站。</small>
          </div>
          <button type="button" @click="addBaseStation" :disabled="!baseStationOptions.length">
            添加基站
          </button>
        </div>
        <div class="device-row" v-for="(station, index) in baseStations" :key="index">
          <label>
            基站类型
            <select v-model="station.base_station" @change="() => syncStationMode(station)">
              <option v-for="option in baseStationOptions" :key="option.name" :value="option.name">
                {{ option.label }}
              </option>
            </select>
          </label>
          <label>
            X
            <input type="number" min="0" :max="gridLimit" v-model.number="station.x" />
          </label>
          <label>
            Y
            <input type="number" min="0" :max="gridLimit" v-model.number="station.y" />
          </label>
          <div class="station-meta">
            <p>支持模式：{{ formatModes(station.base_station) }}</p>
            <p>激活模式：{{ station.mode || resolveDefaultMode(station.base_station) || "自动" }}</p>
          </div>
          <button type="button" class="remove-btn" @click="removeBaseStation(index)">移除</button>
        </div>
        <p v-if="!baseStations.length" class="hint">尚未添加基站，将按完全受灾进行测试。</p>
      </div>
      <button type="submit" class="run-btn" :disabled="isRunning">
        {{ isRunning ? "测试中..." : "开始测试" }}
      </button>
    </form>

    <div v-if="simulationResult" class="tester__result">
      <h3>测试结果</h3>
      <p>平均奖励：{{ simulationResult.avg_reward.toFixed(2) }}</p>
      <p>平均覆盖率：{{ (simulationResult.avg_final_coverage * 100).toFixed(2) }}%</p>
      <div v-for="report in simulationResult.reports" :key="report.episode" class="report">
        <h4>
          Episode {{ report.episode }} - {{ report.scenario?.name }} ({{ report.scenario?.disaster_type }})
        </h4>
        <div class="report__stats">
          <p>总奖励：{{ report.total_reward.toFixed(2) }}</p>
          <p>终态覆盖：{{ (report.final_state.coverage_ratio * 100).toFixed(2) }}%</p>
          <p>终态广播：{{ (report.final_state.broadcast_ratio * 100).toFixed(2) }}%</p>
          <p>剩余预算：{{ report.final_state.remaining_budget.toFixed(1) }}</p>
        </div>
        <details>
          <summary>查看设备恢复情况</summary>
          <table>
            <thead>
              <tr>
                <th>ID</th>
                <th>位置</th>
                <th>需求</th>
                <th>连接状态</th>
                <th>广播</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="device in report.final_state.user_details" :key="device.id">
                <td>{{ device.id }}</td>
                <td>{{ device.position?.[0] }}, {{ device.position?.[1] }}</td>
                <td>{{ device.demand?.toFixed(1) }} Mbps</td>
                <td>{{ device.connected ? "在线" : "离线" }}</td>
                <td>{{ device.broadcast_served ? "已覆盖" : "未覆盖" }}</td>
              </tr>
            </tbody>
          </table>
        </details>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, ref, watch } from "vue";
import axios from "axios";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000/api";

const scenarios = ref([]);
const scenarioName = ref("typhoon_residual");
const algorithms = [
  { value: "ppo", label: "PPO", desc: "基线" },
  { value: "dqa", label: "DQA", desc: "大动作空间" },
  { value: "n3c", label: "N3C", desc: "多目标" },
  { value: "mppo", label: "MPPO", desc: "多头策略" },
];
const selectedAlgorithm = ref("ppo");
const checkpointPath = ref("artifacts/ppo_policy.pt");
const baseStations = ref([]);
const simulationResult = ref(null);
const isRunning = ref(false);
const errorMessage = ref("");

const currentScenario = computed(() => scenarios.value.find((scenario) => scenario.name === scenarioName.value));
const baseStationOptions = computed(() => currentScenario.value?.base_stations || []);
const gridLimit = computed(() => Math.max(0, (currentScenario.value?.grid_size || 10) - 1));

const fetchScenarios = async () => {
  try {
    const { data } = await axios.get(`${API_BASE}/scenarios`);
    scenarios.value = data.scenarios || [];
    if (scenarios.value.length && !scenarioName.value) {
      scenarioName.value = scenarios.value[0].name;
    }
  } catch (error) {
    console.error("Failed to load scenarios", error);
  }
};

const resolveDefaultMode = (baseKey) => {
  const option = baseStationOptions.value.find((item) => item.name === baseKey);
  if (!option) return null;
  const modes = option.supported_modes || [];
  return modes.length ? modes[0] : null;
};

const formatModes = (baseKey) => {
  const option = baseStationOptions.value.find((item) => item.name === baseKey);
  if (!option || !option.supported_modes) return "未知";
  return option.supported_modes.join(" / ");
};

const syncStationMode = (station) => {
  const option = baseStationOptions.value.find((item) => item.name === station.base_station);
  if (!option) {
    station.mode = null;
    return;
  }
  if (!option.supported_modes?.includes(station.mode)) {
    station.mode = resolveDefaultMode(station.base_station);
  }
};

const addBaseStation = () => {
  if (!baseStationOptions.value.length) return;
  const baseKey = baseStationOptions.value[0].name;
  baseStations.value.push({
    base_station: baseKey,
    mode: resolveDefaultMode(baseKey),
    x: 0,
    y: 0,
  });
};

const removeBaseStation = (index) => {
  baseStations.value.splice(index, 1);
};

const runSimulation = async () => {
  isRunning.value = true;
  simulationResult.value = null;
  errorMessage.value = "";
  try {
    const { data } = await axios.post(`${API_BASE}/simulate`, {
      scenario_name: scenarioName.value,
      algorithm: selectedAlgorithm.value,
      checkpoint_path: checkpointPath.value,
      env_type: "multimodal",
      stochastic_eval: true,
      episodes: 1,
      custom_devices: [],
      custom_base_stations: baseStations.value,
    });
    simulationResult.value = data;
  } catch (error) {
    console.error("Simulation failed", error);
    const apiMsg = error?.response?.data?.detail || error?.message || "模拟请求失败";
    errorMessage.value = typeof apiMsg === "string" ? apiMsg : JSON.stringify(apiMsg);
  } finally {
    isRunning.value = false;
  }
};

watch(scenarioName, () => {
  baseStations.value = [];
});

watch(
  selectedAlgorithm,
  (algo) => {
    // auto-suggest checkpoint name per algorithm
    checkpointPath.value = `artifacts/${algo}_policy.pt`;
  },
  { immediate: true }
);

onMounted(() => {
  fetchScenarios();
});
</script>

<style scoped>
.tester {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.subtitle {
  margin: 0;
  color: #94a3b8;
}

.tester__form {
  border: 1px solid rgba(148, 163, 184, 0.3);
  border-radius: 12px;
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 16px;
  background: rgba(15, 23, 42, 0.4);
}

select,
input[type="number"],
input[type="text"] {
  width: 100%;
  padding: 10px 12px;
  border-radius: 8px;
  border: 1px solid rgba(148, 163, 184, 0.6);
  background: rgba(15, 23, 42, 0.2);
  color: inherit;
}

.checkbox,
.checkbox-inline {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 14px;
}

.devices {
  border: 1px solid rgba(148, 163, 184, 0.3);
  border-radius: 12px;
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.devices__header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.devices__header button {
  padding: 6px 12px;
  border-radius: 999px;
  border: 1px solid rgba(148, 163, 184, 0.6);
  background: transparent;
  color: inherit;
}

.device-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 12px;
  border: 1px solid rgba(148, 163, 184, 0.2);
  border-radius: 8px;
  padding: 12px;
  align-items: end;
}

.device-row label {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.remove-btn {
  border: none;
  background: rgba(239, 68, 68, 0.2);
  color: #fecaca;
  border-radius: 999px;
  padding: 6px 12px;
}

.station-meta {
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 12px;
  color: #cbd5f5;
  padding: 4px 0;
}

.hint {
  margin: 0;
  color: #94a3b8;
}

.run-btn {
  padding: 12px;
  border: none;
  border-radius: 999px;
  background: linear-gradient(90deg, #22d3ee, #3b82f6);
  color: #fff;
  font-weight: 600;
}

.algo-options {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
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

.tester__result {
  border: 1px solid rgba(148, 163, 184, 0.4);
  border-radius: 12px;
  padding: 16px;
  background: rgba(15, 23, 42, 0.4);
}

.error-banner {
  border: 1px solid rgba(239, 68, 68, 0.4);
  background: rgba(239, 68, 68, 0.15);
  color: #fecaca;
  border-radius: 10px;
  padding: 10px 12px;
}

.report {
  margin-top: 16px;
  border-top: 1px solid rgba(148, 163, 184, 0.3);
  padding-top: 12px;
}

.report__stats {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
}

table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 8px;
}

th,
td {
  border: 1px solid rgba(148, 163, 184, 0.3);
  padding: 6px;
  text-align: center;
}
</style>
