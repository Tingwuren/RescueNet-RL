<template>
  <div class="tester">
    <h2>自定义环境测试</h2>
    <p class="subtitle">
      输入受灾设备（坐标、带宽需求、初始状态），让训练好的模型输出组网策略，并查看恢复结果。
    </p>
    <form class="tester__form" @submit.prevent="runSimulation">
      <label>
        场景选择
        <select v-model="scenarioName">
          <option v-for="scenario in scenarios" :key="scenario.name" :value="scenario.name">
            {{ scenario.name }} ({{ scenario.disaster_type }})
          </option>
        </select>
      </label>
      <div class="devices">
        <div class="devices__header">
          <p>设备列表</p>
          <button type="button" @click="addDevice">添加设备</button>
        </div>
        <div class="device-row" v-for="(device, index) in devices" :key="index">
          <label>
            X
            <input type="number" min="0" max="20" v-model.number="device.x" />
          </label>
          <label>
            Y
            <input type="number" min="0" max="20" v-model.number="device.y" />
          </label>
          <label>
            需求(Mbps)
            <input type="number" min="1" max="50" step="1" v-model.number="device.demand" />
          </label>
          <label class="checkbox-inline">
            <input type="checkbox" v-model="device.connected" />
            已连接
          </label>
          <label class="checkbox-inline">
            <input type="checkbox" v-model="device.broadcast_served" />
            享受广播
          </label>
          <button type="button" class="remove-btn" @click="removeDevice(index)">移除</button>
        </div>
        <p v-if="!devices.length" class="hint">尚未添加设备。</p>
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
import { onMounted, ref } from "vue";
import axios from "axios";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000/api";

const scenarios = ref([]);
const scenarioName = ref("typhoon_residual");
const devices = ref([]);
const simulationResult = ref(null);
const isRunning = ref(false);

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

const addDevice = () => {
  devices.value.push({
    x: 0,
    y: 0,
    demand: 10,
    connected: false,
    broadcast_served: false,
  });
};

const removeDevice = (index) => {
  devices.value.splice(index, 1);
};

const runSimulation = async () => {
  isRunning.value = true;
  simulationResult.value = null;
  try {
    const { data } = await axios.post(`${API_BASE}/simulate`, {
      scenario_name: scenarioName.value,
      env_type: "multimodal",
      stochastic_eval: true,
      episodes: 1,
      custom_devices: devices.value,
    });
    simulationResult.value = data;
  } catch (error) {
    console.error("Simulation failed", error);
  } finally {
    isRunning.value = false;
  }
};

onMounted(() => {
  fetchScenarios();
  addDevice();
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
input[type="number"] {
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
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  border: 1px solid rgba(148, 163, 184, 0.2);
  border-radius: 8px;
  padding: 8px;
}

.device-row label {
  flex: 1 1 80px;
}

.remove-btn {
  border: none;
  background: rgba(239, 68, 68, 0.2);
  color: #fecaca;
  border-radius: 999px;
  padding: 6px 12px;
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

.tester__result {
  border: 1px solid rgba(148, 163, 184, 0.4);
  border-radius: 12px;
  padding: 16px;
  background: rgba(15, 23, 42, 0.4);
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
