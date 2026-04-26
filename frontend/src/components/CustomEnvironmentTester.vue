<template>
  <div class="tester">
    <div class="tester__intro">
      <h2>自定义环境测试</h2>
      <p class="subtitle">导入场景图后直接启动测试，终端持续输出过程日志。</p>
    </div>

    <div v-if="errorMessage" class="error-banner">
      {{ errorMessage }}
    </div>

    <form class="tester__form" @submit.prevent="runSimulation">
      <section class="tester__section">
        <label>
          场景选择
          <select v-model="scenarioName">
            <option v-for="scenario in scenarios" :key="scenario.name" :value="scenario.name">
              {{ formatScenarioName(scenario.name) }}
            </option>
          </select>
        </label>

      </section>

      <section v-if="activeSceneGraph" class="tester__section">
        <div class="section-header">
          <div v-if="sceneTabOptions.length > 1" class="scene-tabs">
            <button
              v-for="option in sceneTabOptions"
              :key="option.key"
              type="button"
              :class="['scene-tab', { 'scene-tab--active': sceneGraphTab === option.key }]"
              @click="sceneGraphTab = option.key"
            >
              {{ option.label }}
            </button>
          </div>
        </div>

        <SceneGraphPreview
          :scene="activeSceneGraph"
          :scenario-name="scenarioName"
          :scene-kind="sceneGraphTab"
        />

        <div v-if="regionGrid" class="region-hint">
          <p>
            区域：{{ regionGrid.name }}
            <small>离散网格 {{ regionGrid.rows }} × {{ regionGrid.cols }}</small>
          </p>
          <p v-if="regionMetrics">
            实际跨度：约 {{ formatDistance(regionMetrics.widthKm) }} × {{ formatDistance(regionMetrics.heightKm) }}
            <small>单网格约 {{ formatDistance(regionMetrics.cellWidthKm) }} × {{ formatDistance(regionMetrics.cellHeightKm) }}</small>
          </p>
          <p class="bounds">
            经纬边界：纬度 {{ regionGrid.geo_bounds?.lat_min }}–{{ regionGrid.geo_bounds?.lat_max }}，
            经度 {{ regionGrid.geo_bounds?.lon_min }}–{{ regionGrid.geo_bounds?.lon_max }}
          </p>
        </div>

      </section>

      <section class="tester__section">
        <div class="section-header">
          <div>
            <h3>测试配置</h3>
          </div>
        </div>

        <label>
          算法选择
          <div class="algo-options">
            <button
              type="button"
              v-for="algo in algorithms"
              :key="algo.value"
              :class="['algo-chip', { 'algo-chip--active': algo.value === selectedAlgorithm }]"
              :disabled="algo.disabled"
              :title="algo.disabled ? '预留按钮，暂未接入后端实现' : ''"
              @click="() => !algo.disabled && (selectedAlgorithm = algo.value)"
            >
              <strong>{{ algo.label }}</strong>
              <small>{{ algo.desc }}</small>
            </button>
          </div>
        </label>

        <div class="devices">
          <div class="devices__header">
            <div>
              <p>残余基站</p>
              <small>留空则沿用场景默认配置。</small>
            </div>
            <button type="button" @click="addBaseStation" :disabled="!baseStationOptions.length || isRunning">
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
              网格行
              <input type="number" min="0" :max="gridRowLimit" v-model.number="station.x" />
            </label>
            <label>
              网格列
              <input type="number" min="0" :max="gridColLimit" v-model.number="station.y" />
            </label>
            <div class="station-meta">
              <p>支持模式：{{ formatModes(station.base_station) }}</p>
              <p>激活模式：{{ station.mode || resolveDefaultMode(station.base_station) || "自动" }}</p>
            </div>
            <button type="button" class="remove-btn" @click="removeBaseStation(index)">移除</button>
          </div>

          <p v-if="!baseStations.length" class="hint">尚未自定义残余基站，将沿用当前场景的默认残余网络。</p>
        </div>

        <div class="run-actions">
          <button type="submit" class="run-btn" :disabled="isRunning || isImporting || !hasMatchingCheckpoint">
            {{ isRunning ? "测试中..." : isImporting ? "同步场景中..." : "开始测试" }}
          </button>
          <p class="run-hint" v-if="!hasMatchingCheckpoint">
            当前场景和算法没有匹配的训练权重，请先在场景&环境导入中训练该组合。
          </p>
          <p class="run-hint" v-if="!isSceneReady">
            场景图会自动同步，完成后即可开始测试。
          </p>
        </div>
      </section>
    </form>

    <div class="tester__section tester__section--terminal">
      <StreamingTerminal
        title="实时输出终端"
        subtitle="状态、动作和恢复指标会持续写入这里。"
        :lines="terminalLines"
        :status="terminalStatus"
      />
    </div>

    <div v-if="simulationResult" class="tester__result">
      <h3>测试结果</h3>
      <p>平均奖励：{{ simulationResult.avg_reward.toFixed(2) }}</p>
      <p>平均覆盖率：{{ (simulationResult.avg_final_coverage * 100).toFixed(2) }}%</p>
      <div class="export-actions export-actions--replay">
        <a class="replay-link" href="#/replay">前往回放页选择本次测试结果</a>
      </div>
      <div v-if="sceneExport" class="export-panel">
        <h4>场景导出</h4>
        <p>受灾场景文件：{{ sceneExport.disaster_scene_path }}</p>
        <p>部署后场景文件：{{ sceneExport.deployment_scene_path }}</p>
        <div class="export-actions">
          <button type="button" @click="downloadExport(sceneExport.disaster_scene, exportFilename('disaster'))">
            下载受灾场景 JSON
          </button>
          <button type="button" @click="downloadExport(sceneExport.deployment_scene, exportFilename('deployment'))">
            下载部署后场景 JSON
          </button>
        </div>
      </div>
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
                <th>位置 / 区域</th>
                <th>需求</th>
                <th>连接状态</th>
                <th>广播</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="device in report.final_state.user_details" :key="device.id">
                <td>{{ device.id }}</td>
                <td>
                  <div>{{ device.position?.[0] }}, {{ device.position?.[1] }}</div>
                  <small v-if="device.region_label">{{ device.region_label }}</small>
                  <small v-else class="muted">网格单元</small>
                  <small v-if="device.lat_lon_bounds">
                    (Lat {{ device.lat_lon_bounds.lat_min.toFixed(3) }}~{{ device.lat_lon_bounds.lat_max.toFixed(3) }},
                    Lon {{ device.lat_lon_bounds.lon_min.toFixed(3) }}~{{ device.lat_lon_bounds.lon_max.toFixed(3) }})
                  </small>
                </td>
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
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import axios from "axios";
import SceneGraphPreview from "./SceneGraphPreview.vue";
import StreamingTerminal from "./StreamingTerminal.vue";
import { buildRegionMetrics, formatDistance } from "../utils/regionMetrics";
import { rescueApiBase } from "../utils/runtimeEndpoints";
import { saveReplaySessionFromSimulation } from "../utils/replaySessions";
import { formatScenarioName } from "../utils/scenarioLabels";

const API_BASE = rescueApiBase;

const scenarios = ref([]);
const scenarioName = ref("typhoon_residual");
const algorithms = [
  { value: "ppo", label: "PPO", desc: "基线" },
  { value: "dqn", label: "DQN", desc: "大动作空间" },
  { value: "a3c", label: "A3C", desc: "多目标" },
  { value: "mppo", label: "MPPO", desc: "多头策略" },
  { value: "custom", label: "自创算法", desc: "预留中", disabled: true },
];
const selectedAlgorithm = ref("ppo");
const checkpointPath = ref("artifacts/ppo_policy.pt");
const trainingArtifacts = ref([]);
const evaluationSeed = ref(13);
const baseStations = ref([]);
const importedScene = ref(null);
const importedSceneSignature = ref("");
const simulationResult = ref(null);
const terminalLines = ref([]);
const terminalStatus = ref("idle");
const sceneGraphTab = ref("imported");
const isImporting = ref(false);
const isRunning = ref(false);
const errorMessage = ref("");
let autoImportTimer = null;

const currentScenario = computed(() => scenarios.value.find((scenario) => scenario.name === scenarioName.value));
const baseStationOptions = computed(() => currentScenario.value?.base_stations || []);
const regionGrid = computed(() => currentScenario.value?.region_grid || null);
const regionMetrics = computed(() => buildRegionMetrics(regionGrid.value));
const sceneExport = computed(() => simulationResult.value?.scene_export || null);
const gridRows = computed(() => regionGrid.value?.rows || currentScenario.value?.grid_size || 10);
const gridCols = computed(() => regionGrid.value?.cols || currentScenario.value?.grid_size || 10);
const gridRowLimit = computed(() => Math.max(0, gridRows.value - 1));
const gridColLimit = computed(() => Math.max(0, gridCols.value - 1));
const matchingCheckpoint = computed(() =>
  trainingArtifacts.value.find(
    (artifact) =>
      artifact.scenario_name === scenarioName.value &&
      artifact.algorithm === selectedAlgorithm.value &&
      artifact.checkpoint_path
  )
);
const hasMatchingCheckpoint = computed(() => Boolean(matchingCheckpoint.value));

const currentSceneSignature = computed(() =>
  JSON.stringify({
    scenario: scenarioName.value,
    baseStations: baseStations.value.map((station) => ({
      x: Number(station.x ?? 0),
      y: Number(station.y ?? 0),
      base_station: station.base_station || "",
      mode: station.mode || null,
    })),
  })
);

const isSceneReady = computed(
  () => Boolean(importedScene.value) && importedSceneSignature.value === currentSceneSignature.value
);

const sceneStatusTone = computed(() => {
  if (isImporting.value) return "loading";
  if (isSceneReady.value) return "ready";
  if (importedScene.value) return "stale";
  return "idle";
});

const sceneTabOptions = computed(() => {
  const tabs = [];
  if (importedScene.value?.scene) {
    tabs.push({ key: "imported", label: "导入场景" });
  }
  if (sceneExport.value?.deployment_scene) {
    tabs.push({ key: "deployment", label: "部署后场景" });
  }
  return tabs;
});

const activeSceneGraph = computed(() => {
  if (sceneGraphTab.value === "deployment" && sceneExport.value?.deployment_scene) {
    return sceneExport.value.deployment_scene;
  }
  return importedScene.value?.scene || null;
});

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

const fetchTrainingArtifacts = async () => {
  try {
    const { data } = await axios.get(`${API_BASE}/train/artifacts`, { timeout: 10000 });
    trainingArtifacts.value = Array.isArray(data?.artifacts) ? data.artifacts : [];
    syncCheckpointPath();
  } catch (error) {
    console.warn("Failed to load training artifacts", error);
    trainingArtifacts.value = [];
    syncCheckpointPath();
  }
};

const syncCheckpointPath = () => {
  checkpointPath.value = matchingCheckpoint.value?.checkpoint_path || "";
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

const exportFilename = (suffix) => {
  const scenario = scenarioName.value || "scenario";
  return `${scenario}_${suffix}_scene.json`;
};

const downloadExport = (payload, filename) => {
  if (!payload) return;
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
};

const formatTerminalTime = (timestamp) =>
  new Date(timestamp).toLocaleTimeString("zh-CN", {
    hour12: false,
  });

const appendTerminalLine = (message, timestamp = Date.now()) => {
  if (!message) return;
  terminalLines.value = [...terminalLines.value.slice(-399), `[${formatTerminalTime(timestamp)}] ${message}`];
};

const importSceneGraph = async () => {
  if (!scenarioName.value || isRunning.value) return false;
  const importSignature = currentSceneSignature.value;
  isImporting.value = true;
  errorMessage.value = "";
  simulationResult.value = null;
  sceneGraphTab.value = "imported";
  terminalStatus.value = "importing";

  try {
    const { data } = await axios.post(`${API_BASE}/simulate/scene`, {
      scenario_name: scenarioName.value,
      env_type: "multimodal",
      custom_base_stations: buildCustomBaseStationsPayload(),
    });
    if (importSignature !== currentSceneSignature.value) {
      return false;
    }
    importedScene.value = data;
    importedSceneSignature.value = importSignature;
    appendTerminalLine(
      `已自动同步场景 ${scenarioName.value}，用户 ${data.initial_state?.total_users ?? 0}，残余基站 ${
        (data.initial_state?.residual_base_stations || []).length
      } 个。`
    );
    terminalStatus.value = "idle";
    return true;
  } catch (error) {
    console.error("Failed to import scene graph", error);
    const apiMsg = error?.response?.data?.detail || error?.message || "场景图导入失败";
    errorMessage.value = typeof apiMsg === "string" ? apiMsg : JSON.stringify(apiMsg);
    appendTerminalLine(`场景图导入失败：${errorMessage.value}`);
    terminalStatus.value = "failed";
    return false;
  } finally {
    isImporting.value = false;
  }
};

const scheduleSceneImport = (delay = 500) => {
  if (!scenarioName.value || isRunning.value) return;
  if (autoImportTimer) {
    clearTimeout(autoImportTimer);
  }
  autoImportTimer = window.setTimeout(() => {
    autoImportTimer = null;
    void importSceneGraph();
  }, delay);
};

const buildImportedDevices = () =>
  (importedScene.value?.initial_state?.user_details || [])
    .filter((device) => Array.isArray(device.position) && device.position.length >= 2)
    .map((device) => ({
      x: Number(device.position[0]),
      y: Number(device.position[1]),
      demand: Number(device.demand ?? 10),
      connected: Boolean(device.connected),
      broadcast_served: Boolean(device.broadcast_served),
    }));

const buildCustomBaseStationsPayload = () => (baseStations.value.length ? baseStations.value : null);

const readErrorResponse = async (response) => {
  const rawText = await response.text();
  if (!rawText) {
    return `请求失败 (${response.status})`;
  }
  try {
    const parsed = JSON.parse(rawText);
    return parsed?.detail || parsed?.message || rawText;
  } catch {
    return rawText;
  }
};

const handleSimulationEvent = (event) => {
  const payload = event?.payload || {};

  if (event.type === "status") {
    const state = payload.state || "idle";
    if (state === "running" || state === "initializing") {
      terminalStatus.value = "running";
    } else if (state === "completed") {
      terminalStatus.value = "completed";
    } else if (state === "failed") {
      terminalStatus.value = "failed";
    }
    return;
  }

  if (event.type === "log") {
    appendTerminalLine(payload.message, (event.timestamp || 0) * 1000 || Date.now());
    return;
  }

  if (event.type === "result") {
    simulationResult.value = payload;
    const savedReplay = saveReplaySessionFromSimulation({
      scenarioName: scenarioName.value,
      algorithm: selectedAlgorithm.value,
      result: payload,
    });
    appendTerminalLine(
      savedReplay?.persisted
        ? "测试结果已保存，可在回放页直接选择本次测试进行回放。"
        : "测试结果已生成回放，但浏览器本地存储空间不足，未能持久保存。"
    );
    sceneGraphTab.value = payload?.scene_export?.deployment_scene ? "deployment" : "imported";
    return;
  }

  if (event.type === "error") {
    errorMessage.value = payload.message || "模拟请求失败";
    appendTerminalLine(`测试失败：${errorMessage.value}`, (event.timestamp || 0) * 1000 || Date.now());
    terminalStatus.value = "failed";
    return;
  }

  if (event.type === "end") {
    terminalStatus.value = payload.state === "completed" ? "completed" : "failed";
  }
};

const processSseChunk = (rawChunk) => {
  const dataLines = rawChunk
    .split("\n")
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).trim());

  if (!dataLines.length) return;

  try {
    const parsed = JSON.parse(dataLines.join("\n"));
    handleSimulationEvent(parsed);
  } catch (error) {
    console.warn("Failed to parse simulation stream event", error, rawChunk);
  }
};

const consumeSimulationStream = async (response) => {
  if (!response.body) {
    throw new Error("当前浏览器不支持流式响应。");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
    let boundaryIndex = buffer.indexOf("\n\n");

    while (boundaryIndex !== -1) {
      const chunk = buffer.slice(0, boundaryIndex).trim();
      buffer = buffer.slice(boundaryIndex + 2);
      if (chunk) {
        processSseChunk(chunk);
      }
      boundaryIndex = buffer.indexOf("\n\n");
    }
  }

  const tail = buffer.trim();
  if (tail) {
    processSseChunk(tail);
  }
};

const runSimulation = async () => {
  syncCheckpointPath();
  if (!checkpointPath.value) {
    errorMessage.value = "当前场景和算法没有匹配的训练权重，请先在场景&环境导入中训练该组合。";
    appendTerminalLine(errorMessage.value);
    return;
  }
  if (!isSceneReady.value) {
    appendTerminalLine("场景图不是最新，正在自动同步后启动测试。");
    const imported = await importSceneGraph();
    if (!imported || !isSceneReady.value) {
      errorMessage.value = "场景图自动同步失败，请检查后端连接后重试。";
      return;
    }
  }

  isRunning.value = true;
  simulationResult.value = null;
  errorMessage.value = "";
  terminalLines.value = [];
  terminalStatus.value = "running";
  sceneGraphTab.value = "imported";
  appendTerminalLine(`使用导入场景快照启动测试：${scenarioName.value}。`);

  try {
    const response = await fetch(`${API_BASE}/simulate/stream`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        scenario_name: scenarioName.value,
        algorithm: selectedAlgorithm.value,
        checkpoint_path: checkpointPath.value,
        env_type: "multimodal",
        stochastic_eval: true,
        eval_seed: evaluationSeed.value,
        episodes: 1,
        custom_devices: buildImportedDevices(),
        custom_base_stations: buildCustomBaseStationsPayload(),
      }),
    });

    if (!response.ok) {
      throw new Error(await readErrorResponse(response));
    }

    await consumeSimulationStream(response);

    if (!simulationResult.value && !errorMessage.value) {
      errorMessage.value = "测试已结束，但未收到结果数据。";
      appendTerminalLine(errorMessage.value);
      terminalStatus.value = "failed";
    }
  } catch (error) {
    console.error("Simulation failed", error);
    errorMessage.value = error?.message || "模拟请求失败";
    appendTerminalLine(`测试失败：${errorMessage.value}`);
    terminalStatus.value = "failed";
  } finally {
    isRunning.value = false;
  }
};

watch(scenarioName, () => {
  syncCheckpointPath();
  baseStations.value = [];
  importedScene.value = null;
  importedSceneSignature.value = "";
  simulationResult.value = null;
  terminalLines.value = [];
  terminalStatus.value = "idle";
  errorMessage.value = "";
  sceneGraphTab.value = "imported";
  scheduleSceneImport(0);
});

watch(
  baseStations,
  () => {
    errorMessage.value = "";
    simulationResult.value = null;
    importedSceneSignature.value = "";
    if (sceneGraphTab.value === "deployment") {
      sceneGraphTab.value = "imported";
    }
    scheduleSceneImport(500);
  },
  { deep: true }
);

watch(
  selectedAlgorithm,
  (algo) => {
    syncCheckpointPath();
  },
  { immediate: true }
);

onMounted(async () => {
  await fetchScenarios();
  await fetchTrainingArtifacts();
  scheduleSceneImport(0);
});

onBeforeUnmount(() => {
  if (autoImportTimer) {
    clearTimeout(autoImportTimer);
  }
});
</script>

<style scoped>
.tester {
  display: flex;
  flex-direction: column;
  gap: 18px;
  color: #0f172a;
}

.tester__intro h2 {
  margin: 0;
}

.subtitle {
  margin: 6px 0 0;
  color: #64748b;
}

.tester__form,
.tester__result,
.tester__section {
  border: 1px solid rgba(148, 163, 184, 0.3);
  border-radius: 16px;
  padding: 18px;
  background:
    radial-gradient(circle at 100% 0%, rgba(14, 165, 233, 0.09), transparent 32%),
    rgba(255, 255, 255, 0.94);
  color: #0f172a;
  box-shadow: 0 14px 28px rgba(15, 23, 42, 0.06);
}

.tester__form {
  display: flex;
  flex-direction: column;
  gap: 18px;
}

.tester__section {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.tester__section--terminal {
  padding: 20px;
}

.section-header {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: flex-start;
}

.section-header h3 {
  margin: 0;
  font-size: 18px;
}

.section-header p {
  margin: 6px 0 0;
  color: #64748b;
}

.import-btn,
.scene-tab,
.devices__header button,
.export-actions button {
  padding: 8px 14px;
  border-radius: 999px;
  border: 1px solid rgba(14, 165, 233, 0.22);
  background: rgba(224, 242, 254, 0.72);
  color: #075985;
  font-weight: 600;
}

.import-btn {
  min-width: 120px;
}

.scene-tabs {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.scene-tab--active {
  border-color: rgba(2, 132, 199, 0.42);
  background: rgba(14, 165, 233, 0.16);
  color: #075985;
}

.region-hint {
  border-left: 4px solid #0ea5e9;
  background: rgba(224, 242, 254, 0.72);
  padding: 8px 12px;
  border-radius: 10px;
  color: #0f172a;
}

.region-hint .bounds {
  margin: 4px 0 0;
  color: #64748b;
  font-size: 13px;
}

.muted {
  color: #94a3b8;
}

label {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

select,
input[type="number"],
input[type="text"] {
  width: 100%;
  padding: 10px 12px;
  border-radius: 10px;
  border: 1px solid rgba(100, 116, 139, 0.28);
  background: rgba(255, 255, 255, 0.96);
  color: #0f172a;
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
  border: 1px solid rgba(100, 116, 139, 0.22);
  background: rgba(255, 255, 255, 0.94);
  color: #0f172a;
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

.devices {
  border: 1px solid rgba(148, 163, 184, 0.2);
  border-radius: 14px;
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 12px;
  background: rgba(248, 250, 252, 0.72);
}

.devices__header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
}

.devices__header p,
.report h4,
.export-panel h4 {
  margin: 0;
}

.device-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 12px;
  border: 1px solid rgba(148, 163, 184, 0.22);
  border-radius: 14px;
  padding: 12px;
  align-items: end;
  background: rgba(255, 255, 255, 0.92);
}

.remove-btn {
  border: none;
  background: rgba(254, 226, 226, 0.95);
  color: #b91c1c;
  border-radius: 999px;
  padding: 8px 12px;
}

.station-meta {
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 12px;
  color: #64748b;
  padding: 4px 0;
}

.hint {
  margin: 0;
  color: #64748b;
}

.run-actions {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.run-btn {
  padding: 12px;
  border: none;
  border-radius: 999px;
  background: linear-gradient(90deg, #22d3ee, #3b82f6);
  color: #fff;
  font-weight: 600;
}

.run-hint {
  margin: 0;
  color: #b45309;
  font-size: 13px;
}

.run-btn:disabled,
.import-btn:disabled,
.devices__header button:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.tester__result {
  background: rgba(255, 255, 255, 0.94);
}

.error-banner {
  border: 1px solid rgba(239, 68, 68, 0.4);
  background: rgba(254, 226, 226, 0.92);
  color: #991b1b;
  border-radius: 10px;
  padding: 10px 12px;
}

.export-panel {
  margin-top: 16px;
  border: 1px solid rgba(14, 165, 233, 0.2);
  border-radius: 10px;
  padding: 12px;
  background: rgba(224, 242, 254, 0.72);
}

.export-panel p {
  margin: 4px 0;
  word-break: break-all;
}

.export-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 10px;
}

.export-actions--replay {
  margin-top: 12px;
}

.replay-link {
  display: inline-flex;
  align-items: center;
  padding: 8px 14px;
  border-radius: 999px;
  border: 1px solid rgba(14, 165, 233, 0.28);
  color: #075985;
  text-decoration: none;
  background: rgba(224, 242, 254, 0.88);
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

th {
  background: rgba(241, 245, 249, 0.9);
  color: #334155;
}

td {
  background: rgba(255, 255, 255, 0.78);
}

@media (max-width: 720px) {
  .section-header,
  .devices__header {
    flex-direction: column;
    align-items: stretch;
  }

  .scene-tabs {
    width: 100%;
  }
}
</style>
