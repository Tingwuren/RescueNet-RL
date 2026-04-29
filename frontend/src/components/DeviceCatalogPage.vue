<template>
  <section class="device-page">
    <div v-if="errorMessage" class="device-page__error">{{ errorMessage }}</div>

    <section class="device-page__hero">
      <div class="device-page__hero-copy">
        <span class="device-page__eyebrow">Device Catalog</span>
        <h2>基于真实场景设备库的装备管理与能力展示</h2>
        <p>
          设备页直接接入 `/api/scenarios` 返回的基站资料，按当前场景展示设备类型、能力参数、
          支持模式与用途说明。
        </p>
      </div>

      <div class="device-page__hero-actions">
        <label>
          <span>场景选择</span>
          <select v-model="selectedScenarioName">
            <option v-for="scenario in scenarios" :key="scenario.name" :value="scenario.name">
              {{ formatScenarioName(scenario.name) }}
            </option>
          </select>
        </label>
        <button type="button" @click="fetchScenarios">刷新设备库</button>
      </div>
    </section>

    <section class="device-page__stats">
      <article v-for="item in stats" :key="item.label">
        <small>{{ item.label }}</small>
        <strong>{{ item.value }}</strong>
        <p>{{ item.hint }}</p>
      </article>
    </section>

    <BaseStationShowcase v-if="currentScenario" :scenario="currentScenario" />

    <section v-if="currentScenario" class="device-page__table-card">
      <div class="device-page__table-header">
        <div>
          <h3>设备参数清单</h3>
          <p>参数直接来自当前场景定义的基站配置。</p>
        </div>
        <span>{{ currentScenario.base_stations?.length || 0 }} 类设备</span>
      </div>

      <div class="device-page__table-wrap">
        <table class="device-page__table">
          <thead>
            <tr>
              <th>设备名称</th>
              <th>峰值吞吐</th>
              <th>接入用户</th>
              <th>设备成本</th>
              <th>带宽成本</th>
              <th>支持模式</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="station in currentScenario.base_stations || []" :key="station.name">
              <td>{{ station.label || station.name }}</td>
              <td>{{ Number(station.max_throughput || 0).toFixed(0) }} Mbps</td>
              <td>{{ station.max_users }}</td>
              <td>{{ Number(station.device_cost || 0).toFixed(2) }}</td>
              <td>{{ Number(station.bandwidth_cost || 0).toFixed(3) }}</td>
              <td>{{ (station.supported_modes || []).join(" / ") || "未配置" }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>
  </section>
</template>

<script setup>
import { computed, onMounted, ref } from "vue";
import axios from "axios";

import BaseStationShowcase from "./BaseStationShowcase.vue";
import { rescueApiBase } from "../utils/runtimeEndpoints";
import { formatDisasterType, formatScenarioName } from "../utils/scenarioLabels";

const API_BASE = rescueApiBase;

const scenarios = ref([]);
const selectedScenarioName = ref("");
const errorMessage = ref("");

const currentScenario = computed(() =>
  scenarios.value.find((scenario) => scenario.name === selectedScenarioName.value) || null
);

const stats = computed(() => {
  const stations = currentScenario.value?.base_stations || [];
  const modeCount = stations.reduce((total, station) => total + (station.supported_modes || []).length, 0);
  return [
    {
      label: "灾害类型",
      value: formatDisasterType(currentScenario.value?.disaster_type),
      hint: "当前场景所属灾害分类",
    },
    {
      label: "设备类型",
      value: String(stations.length),
      hint: "当前场景可用基站类型数量",
    },
    {
      label: "支持模式",
      value: String(modeCount),
      hint: "所有设备支持的通信模式总数",
    },
    {
      label: "候选站点",
      value: Number(currentScenario.value?.candidate_sites || 0).toLocaleString("zh-CN"),
      hint: "设备部署候选位置数量",
    },
  ];
});

const fetchScenarios = async () => {
  errorMessage.value = "";
  try {
    const { data } = await axios.get(`${API_BASE}/scenarios`);
    scenarios.value = Array.isArray(data?.scenarios) ? data.scenarios : [];
    if (!scenarios.value.length) {
      selectedScenarioName.value = "";
      return;
    }
    if (!scenarios.value.some((scenario) => scenario.name === selectedScenarioName.value)) {
      selectedScenarioName.value = scenarios.value[0].name;
    }
  } catch (error) {
    console.error("Failed to load scenarios for device catalog", error);
    errorMessage.value = "设备库加载失败，请检查后端服务。";
  }
};

onMounted(() => {
  void fetchScenarios();
});
</script>

<style scoped>
.device-page {
  display: flex;
  flex-direction: column;
  gap: 18px;
}

.device-page__error,
.device-page__hero,
.device-page__stats article,
.device-page__table-card {
  border: 1px solid rgba(148, 163, 184, 0.18);
  border-radius: 24px;
  background: rgba(255, 255, 255, 0.9);
  box-shadow: 0 18px 40px rgba(59, 130, 246, 0.08);
}

.device-page__error {
  padding: 14px 16px;
  color: #b91c1c;
  background: rgba(254, 242, 242, 0.92);
}

.device-page__hero {
  display: flex;
  justify-content: space-between;
  gap: 18px;
  align-items: center;
  padding: 22px;
  background:
    radial-gradient(circle at 85% 10%, rgba(59, 130, 246, 0.12), transparent 26%),
    linear-gradient(180deg, rgba(255, 255, 255, 0.96), rgba(245, 249, 255, 0.94));
}

.device-page__eyebrow,
.device-page__stats small,
.device-page__table-header span {
  color: #728aac;
  font-size: 12px;
  letter-spacing: 0.08em;
}

.device-page__eyebrow {
  display: inline-flex;
  padding: 7px 12px;
  border-radius: 999px;
  background: rgba(37, 99, 235, 0.08);
  color: #2563eb;
}

.device-page__hero-copy h2,
.device-page__table-header h3 {
  margin: 10px 0 0;
  color: #17315d;
}

.device-page__hero-copy p,
.device-page__stats p,
.device-page__table-header p {
  margin: 8px 0 0;
  color: #6881a7;
  line-height: 1.7;
}

.device-page__hero-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  justify-content: flex-end;
}

.device-page__hero-actions label {
  display: flex;
  flex-direction: column;
  gap: 6px;
  color: #5d7699;
  font-size: 13px;
}

.device-page__hero-actions select,
.device-page__hero-actions button {
  height: 40px;
  border-radius: 12px;
}

.device-page__hero-actions select {
  min-width: 220px;
  padding: 0 12px;
  border: 1px solid rgba(148, 163, 184, 0.22);
  background: #ffffff;
}

.device-page__hero-actions button {
  align-self: flex-end;
  padding: 0 16px;
  border: 0;
  background: rgba(37, 99, 235, 0.1);
  color: #2563eb;
  font-weight: 700;
}

.device-page__stats {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 14px;
}

.device-page__stats article {
  padding: 18px;
}

.device-page__stats strong {
  display: block;
  margin-top: 8px;
  color: #17315d;
  font-size: 24px;
}

.device-page__table-card {
  padding: 18px;
}

.device-page__table-header {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: flex-start;
  margin-bottom: 12px;
}

.device-page__table-wrap {
  overflow: auto;
}

.device-page__table {
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
}

.device-page__table th,
.device-page__table td {
  padding: 12px 10px;
  border-bottom: 1px solid rgba(226, 232, 240, 0.84);
  text-align: left;
  white-space: nowrap;
}

.device-page__table thead th {
  color: #5d7699;
  font-weight: 600;
}

.device-page__table tbody td {
  color: #17315d;
}

@media (max-width: 1200px) {
  .device-page__stats {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 840px) {
  .device-page__hero,
  .device-page__table-header {
    flex-direction: column;
    align-items: flex-start;
  }

  .device-page__hero-actions {
    width: 100%;
    justify-content: flex-start;
  }

  .device-page__hero-actions select {
    min-width: 0;
    width: 100%;
  }
}
</style>
