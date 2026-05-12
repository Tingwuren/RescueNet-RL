<template>
  <div class="training-panel">
    <div class="panel-header">
      <div class="panel-header__intro">
        <span class="panel-header__eyebrow">Scene & Environment Intake</span>
        <div>
          <h2>场景&环境导入</h2>
          <p>录入受灾区域、应急设备与 RL 组网算法，并在高级设置里配置训练与仿真参数。</p>
        </div>
      </div>

      <div class="scenario-select">
        <label>基础仿真场景</label>
        <div class="scenario-options">
          <button
            v-for="scenario in scenarios"
            :key="scenario.name"
            :class="['scenario-chip', { 'scenario-chip--active': scenario.name === selectedScenario }]"
            @click="selectScenario(scenario.name)"
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
      <div v-if="currentScenario" class="scenario-details">
        <article v-for="metric in scenarioMetrics" :key="metric.label" class="scenario-metric">
          <span>{{ metric.label }}</span>
          <strong>{{ metric.value }}</strong>
          <small>{{ metric.hint }}</small>
        </article>
      </div>

      <section class="intake-workbench">
        <div class="intake-workbench__header">
          <div>
            <h3>录入与导入工作台</h3>
            <p>支持文件导入和手动补录，当前填写内容会作为本次训练任务的前端导入清单。</p>
          </div>

          <div class="intake-workbench__status">
            <span>
              <small>受灾区域</small>
              <strong>{{ disasterImportStatus }}</strong>
            </span>
            <span>
              <small>应急设备</small>
              <strong>{{ equipmentImportStatus }}</strong>
            </span>
            <span>
              <small>组网算法</small>
              <strong>{{ algorithmImportStatus }}</strong>
            </span>
          </div>
        </div>

        <div class="intake-grid">
          <article class="import-card">
            <div class="import-card__header">
              <div class="import-card__title">
                <strong>导入受灾区域信息</strong>
                <small>补录灾害类型、重点保障区域和受影响规模。</small>
              </div>

              <label class="import-trigger">
                <input type="file" accept=".json,.geojson,.csv" @change="handleImportFile($event, 'disaster')" />
                <span>{{ disasterRegionFileName ? "重新导入" : "导入文件" }}</span>
              </label>
            </div>

            <div class="config-grid">
              <label class="config-field">
                <span>灾害类型</span>
                <select v-model="disasterType">
                  <option value="flood">洪水</option>
                  <option value="earthquake">地震</option>
                  <option value="landslide">滑坡</option>
                  <option value="typhoon">台风</option>
                </select>
              </label>

              <label class="config-field">
                <span>受灾等级</span>
                <select v-model="disasterSeverity">
                  <option value="moderate">中等</option>
                  <option value="severe">严重</option>
                  <option value="critical">特别严重</option>
                </select>
              </label>

              <label class="config-field">
                <span>受影响网格数</span>
                <input type="number" min="1" step="1" v-model.number="affectedGridCount" />
              </label>

              <label class="config-field">
                <span>受影响人数</span>
                <input type="number" min="1" step="1" v-model.number="impactedPopulation" />
              </label>

              <label class="config-field config-field--wide">
                <span>重点保障区域</span>
                <input
                  type="text"
                  v-model="priorityZone"
                  placeholder="例如：医院 / 指挥中心 / 临时安置点"
                />
              </label>

              <label class="config-field config-field--wide">
                <span>受灾区域说明</span>
                <textarea
                  rows="4"
                  v-model="disasterNotes"
                  placeholder="填写断链区域、道路阻断、GeoJSON 来源或现场补录说明"
                ></textarea>
              </label>
            </div>

            <p class="import-card__footnote">
              当前文件：{{ disasterRegionFileName || "未导入文件，使用表单录入" }}
            </p>
          </article>

          <article class="import-card">
            <div class="import-card__header">
              <div class="import-card__title">
                <strong>导入应急设备信息</strong>
                <small>录入现场保障队伍、预算和优先调度设备。</small>
              </div>

              <label class="import-trigger">
                <input type="file" accept=".json,.csv,.xlsx" @change="handleImportFile($event, 'equipment')" />
                <span>{{ equipmentFileName ? "重新导入" : "导入设备清单" }}</span>
              </label>
            </div>

            <div class="config-grid">
              <label class="config-field">
                <span>保障队伍</span>
                <input type="text" v-model="dispatchUnit" placeholder="例如：前线应急通信保障队" />
              </label>

              <label class="config-field">
                <span>现场小组数</span>
                <input type="number" min="1" step="1" v-model.number="supportTeamCount" />
              </label>

              <label class="config-field">
                <span>部署预算上限</span>
                <input type="number" min="1" step="1" v-model.number="deploymentBudget" />
              </label>

              <label class="config-field">
                <span>优先设备组合</span>
                <input type="text" v-model="priorityEquipment" placeholder="例如：背负式基站 + 多跳中继" />
              </label>

              <label class="config-field config-field--wide">
                <span>设备录入说明</span>
                <textarea
                  rows="4"
                  v-model="equipmentNotes"
                  placeholder="填写设备来源、剩余电量、运载条件或补给说明"
                ></textarea>
              </label>
            </div>

            <div v-if="equipmentLibrary.length" class="import-library">
              <span v-for="station in equipmentLibrary" :key="station.name">
                {{ station.label || station.name }}
              </span>
            </div>

            <p class="import-card__footnote">
              当前文件：{{ equipmentFileName || "未导入文件，沿用当前场景设备库" }}
            </p>
          </article>

          <article class="import-card">
            <div class="import-card__header">
              <div class="import-card__title">
                <strong>导入组网算法</strong>
                <small>选择 RL 算法，或导入外部策略包作为实验记录。</small>
              </div>

              <label class="import-trigger">
                <input type="file" accept=".json,.yaml,.zip,.pt,.onnx" @change="handleImportFile($event, 'algorithm')" />
                <span>{{ algorithmFileName ? "重新导入" : "导入策略包" }}</span>
              </label>
            </div>

            <div class="algo-options">
              <button
                v-for="algo in algorithms"
                :key="algo.value"
                type="button"
                :class="['algo-chip', { 'algo-chip--active': algo.value === selectedAlgorithm }]"
                :disabled="algo.disabled"
                :title="algo.disabled ? '预留按钮，暂未接入训练后端' : ''"
                @click="() => !algo.disabled && selectAlgorithm(algo.value)"
              >
                <strong>{{ algo.label }}</strong>
                <small>{{ algo.desc }}</small>
              </button>
            </div>

            <div class="config-grid">
              <label class="config-field">
                <span>策略来源</span>
                <select v-model="policySource">
                  <option value="platform">平台内置 RL 算法</option>
                  <option value="external">外部导入策略包</option>
                </select>
              </label>

              <label class="config-field">
                <span>组网目标</span>
                <input type="text" v-model="policyGoal" placeholder="例如：覆盖恢复优先" />
              </label>

              <label class="config-field config-field--wide">
                <span>组网算法说明</span>
                <textarea
                  rows="4"
                  v-model="algorithmNotes"
                  placeholder="填写算法版本、权重来源、实验目的或策略差异说明"
                ></textarea>
              </label>
            </div>

            <p class="import-card__footnote">
              当前算法：{{ selectedAlgorithmLabel }} · {{ algorithmFileName || "使用平台内置策略" }}
            </p>
          </article>
        </div>
      </section>

      <div class="settings-stack">
        <div v-if="rewardProfiles.length" class="reward-panel">
          <div class="reward-panel__header">
            <h3>恢复目标配置</h3>
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
              @click="selectRewardMode(profile.key)"
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
              <h3>高级设置</h3>
              <p>算法参数和仿真场景参数分区展示，便于后续复现实验。</p>
            </div>
            <span>Preset: scene-env-import</span>
          </div>

          <section class="config-section">
            <div class="config-section__title">
              <strong>算法参数</strong>
              <small>直接影响训练收敛、采样和探索强度</small>
            </div>

            <div class="config-grid">
              <label class="config-field">
                <span>总训练步数</span>
                <input type="number" min="2000" step="1000" v-model.number="totalTimesteps" />
              </label>

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
              <strong>仿真场景参数</strong>
              <small>环境、评估方式和结果输出控制</small>
            </div>

            <div class="config-grid">
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

              <label class="config-field">
                <span>仿真时窗（小时）</span>
                <input type="number" min="1" max="72" step="1" v-model.number="simulationWindowHours" />
              </label>

              <label class="config-field">
                <span>目标覆盖率 (%)</span>
                <input type="number" min="10" max="100" step="1" v-model.number="coverageTarget" />
              </label>

              <label class="config-field">
                <span>业务负载等级</span>
                <select v-model="trafficLoadProfile">
                  <option value="low">低负载</option>
                  <option value="medium">中负载</option>
                  <option value="high">高负载</option>
                </select>
              </label>

              <label class="config-field">
                <span>恢复目标</span>
                <select v-model="priorityObjective">
                  <option value="coverage_first">覆盖优先</option>
                  <option value="balanced">覆盖与时延平衡</option>
                  <option value="capacity_first">容量优先</option>
                </select>
              </label>

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
            {{ isStarting ? "启动中..." : "确认导入并启动训练" }}
          </button>

          <p v-if="!selectedScenario && !isLoadingScenarios" class="form-hint">
            基础仿真场景未加载成功，请先确认前端可访问训练后端。
          </p>
        </form>
      </div>
    </div>

    <TrainingMonitor :events="eventLog" :status="runStatus" />
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
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

const disasterRegionFileName = ref("");
const disasterType = ref("flood");
const disasterSeverity = ref("severe");
const affectedGridCount = ref(24);
const impactedPopulation = ref(320);
const priorityZone = ref("医院 / 指挥中心 / 临时安置点");
const disasterNotes = ref("");

const equipmentFileName = ref("");
const dispatchUnit = ref("前线应急通信保障队");
const supportTeamCount = ref(6);
const deploymentBudget = ref(12);
const priorityEquipment = ref("");
const equipmentNotes = ref("");

const algorithmFileName = ref("");
const policySource = ref("platform");
const policyGoal = ref("覆盖恢复优先");
const algorithmNotes = ref("");

const totalTimesteps = ref(12000);
const envType = ref("multimodal");
const stochasticEval = ref(true);
const learningRate = ref(0.0003);
const discountFactor = ref(0.99);
const batchSize = ref(256);
const rolloutSteps = ref(1024);
const entropyCoef = ref(0.01);
const clipRange = ref(0.2);
const simulationWindowHours = ref(6);
const coverageTarget = ref(85);
const trafficLoadProfile = ref("high");
const priorityObjective = ref("coverage_first");
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
const equipmentLibrary = computed(() => currentScenario.value?.base_stations || []);
const selectedAlgorithmMeta = computed(
  () => algorithms.find((item) => item.value === selectedAlgorithm.value) || algorithms[0]
);
const selectedAlgorithmLabel = computed(() => selectedAlgorithmMeta.value?.label || "--");

const disasterImportStatus = computed(() =>
  disasterRegionFileName.value || `${affectedGridCount.value} 个网格已录入`
);
const equipmentImportStatus = computed(() =>
  equipmentFileName.value || `${equipmentLibrary.value.length || 0} 类设备待命`
);
const algorithmImportStatus = computed(() =>
  algorithmFileName.value || `${selectedAlgorithmLabel.value} / ${policySource.value === "platform" ? "内置" : "外部"}`
);

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
    {
      label: "录入灾区",
      value: `${affectedGridCount.value} 个网格`,
      hint: `${Number(impactedPopulation.value || 0).toLocaleString("zh-CN")} 人受影响`,
    },
    {
      label: "设备调度",
      value: `${supportTeamCount.value} 个小组`,
      hint: `预算上限 ${deploymentBudget.value}`,
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

const handleImportFile = (event, target) => {
  const file = event?.target?.files?.[0];
  const fileName = file?.name || "";

  if (target === "disaster") {
    disasterRegionFileName.value = fileName;
  }
  if (target === "equipment") {
    equipmentFileName.value = fileName;
  }
  if (target === "algorithm") {
    algorithmFileName.value = fileName;
    if (fileName) {
      policySource.value = "external";
    }
  }
};

const buildImportManifest = () => ({
  disaster: {
    fileName: disasterRegionFileName.value || null,
    disasterType: disasterType.value,
    disasterSeverity: disasterSeverity.value,
    affectedGridCount: affectedGridCount.value,
    impactedPopulation: impactedPopulation.value,
    priorityZone: priorityZone.value,
    notes: disasterNotes.value,
  },
  equipment: {
    fileName: equipmentFileName.value || null,
    dispatchUnit: dispatchUnit.value,
    supportTeamCount: supportTeamCount.value,
    deploymentBudget: deploymentBudget.value,
    priorityEquipment: priorityEquipment.value,
    notes: equipmentNotes.value,
  },
  algorithm: {
    fileName: algorithmFileName.value || null,
    source: policySource.value,
    selectedAlgorithm: selectedAlgorithm.value,
    policyGoal: policyGoal.value,
    notes: algorithmNotes.value,
  },
  advancedSettings: {
    envType: envType.value,
    stochasticEval: stochasticEval.value,
    totalTimesteps: totalTimesteps.value,
    learningRate: learningRate.value,
    discountFactor: discountFactor.value,
    batchSize: batchSize.value,
    rolloutSteps: rolloutSteps.value,
    entropyCoef: entropyCoef.value,
    clipRange: clipRange.value,
    simulationWindowHours: simulationWindowHours.value,
    coverageTarget: coverageTarget.value,
    trafficLoadProfile: trafficLoadProfile.value,
    priorityObjective: priorityObjective.value,
    logWindow: logWindow.value,
    evalInterval: evalInterval.value,
    autoReplay: autoReplay.value,
  },
});

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

    const importManifest = buildImportManifest();
    activeRunMeta.value = {
      runId: data.run_id,
      scenarioName: selectedScenario.value,
      algorithm: selectedAlgorithm.value,
      rewardMode: selectedRewardMode.value,
      config: importManifest.advancedSettings,
    };

    eventLog.value = [
      {
        type: "scene_import",
        timestamp: Date.now() / 1000,
        payload: importManifest,
        message: `已确认 ${formatScenarioName(selectedScenario.value)} 的场景导入清单，准备启动 ${selectedAlgorithmLabel.value} 训练。`,
      },
      {
        type: "experiment_config",
        timestamp: Date.now() / 1000,
        payload: importManifest.advancedSettings,
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
  } else if (!availableKeys.includes(selectedRewardMode.value)) {
    const fallback = scenario.default_reward_profile || availableKeys[0];
    selectedRewardMode.value = fallback;
  }

  const preferredEquipment = (scenario.base_stations || [])
    .slice(0, 2)
    .map((station) => station.label || station.name)
    .filter(Boolean)
    .join(" + ");
  priorityEquipment.value = preferredEquipment || priorityEquipment.value || "背负式基站 + 多跳中继";
  impactedPopulation.value = Number(scenario.num_users || impactedPopulation.value);
});

onMounted(fetchScenarios);
onBeforeUnmount(closeEventSource);
</script>

<style scoped>
.training-panel {
  display: flex;
  flex-direction: column;
  gap: 24px;
}

.panel-header {
  display: grid;
  grid-template-columns: minmax(0, 1.1fr) minmax(0, 1fr);
  gap: 16px;
}

.panel-header__intro,
.scenario-select,
.intake-workbench,
.training-form,
.reward-panel {
  border: 1px solid rgba(100, 116, 139, 0.22);
  border-radius: 18px;
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.96), rgba(248, 250, 252, 0.92));
  box-shadow: 0 14px 28px rgba(15, 23, 42, 0.06);
}

.panel-header__intro {
  padding: 20px;
  background:
    radial-gradient(circle at top right, rgba(14, 165, 233, 0.18), transparent 36%),
    linear-gradient(180deg, rgba(255, 255, 255, 0.97), rgba(248, 250, 252, 0.94));
}

.panel-header__eyebrow {
  display: inline-flex;
  margin-bottom: 10px;
  font-size: 11px;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: #0369a1;
}

.panel-header__intro h2 {
  margin: 0;
  font-size: 28px;
  color: #0f172a;
}

.panel-header__intro p {
  margin: 8px 0 0;
  color: #475569;
  line-height: 1.6;
}

.scenario-select {
  padding: 18px;
}

.scenario-select label {
  font-size: 12px;
  color: #475569;
}

.scenario-options {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 10px;
}

.scenario-chip {
  min-width: 148px;
  padding: 10px 16px;
  border-radius: 12px;
  border: 1px solid rgba(100, 116, 139, 0.28);
  background: rgba(248, 250, 252, 0.94);
  color: #0f172a;
  text-align: left;
  box-shadow: 0 8px 18px rgba(15, 23, 42, 0.06);
}

.scenario-chip--active {
  border-color: #0284c7;
  background: rgba(224, 242, 254, 0.95);
  color: #075985;
  box-shadow: 0 0 0 2px rgba(14, 165, 233, 0.18);
}

.error-banner {
  border: 1px solid rgba(248, 113, 113, 0.55);
  border-radius: 14px;
  padding: 12px 14px;
  background: rgba(127, 29, 29, 0.25);
  color: #fecaca;
}

.panel-body {
  display: grid;
  grid-template-columns: minmax(0, 1.08fr) minmax(0, 0.92fr);
  gap: 24px;
  align-items: start;
}

.scenario-details {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 12px;
}

.scenario-metric {
  min-height: 110px;
  padding: 14px;
  border-radius: 16px;
  border: 1px solid rgba(14, 165, 233, 0.18);
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

.intake-workbench {
  grid-column: 1;
  padding: 18px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.intake-workbench__header {
  display: flex;
  justify-content: space-between;
  gap: 18px;
  align-items: flex-start;
}

.intake-workbench__header h3 {
  margin: 0;
  font-size: 18px;
  color: #0f172a;
}

.intake-workbench__header p {
  margin: 6px 0 0;
  font-size: 12px;
  color: #64748b;
  line-height: 1.5;
}

.intake-workbench__status {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
  min-width: min(100%, 360px);
}

.intake-workbench__status span {
  padding: 10px 12px;
  border-radius: 14px;
  background: rgba(241, 245, 249, 0.75);
  border: 1px solid rgba(148, 163, 184, 0.14);
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.intake-workbench__status small {
  font-size: 11px;
  color: #64748b;
}

.intake-workbench__status strong {
  font-size: 13px;
  color: #0f172a;
}

.intake-grid {
  display: grid;
  gap: 14px;
}

.import-card {
  border-radius: 16px;
  border: 1px solid rgba(148, 163, 184, 0.16);
  background: rgba(241, 245, 249, 0.58);
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.import-card__header {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: flex-start;
}

.import-card__title {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.import-card__title strong {
  font-size: 15px;
  color: #0f172a;
}

.import-card__title small {
  color: #64748b;
  line-height: 1.5;
}

.import-trigger {
  position: relative;
  overflow: hidden;
  flex: 0 0 auto;
  cursor: pointer;
}

.import-trigger input {
  position: absolute;
  inset: 0;
  opacity: 0;
  cursor: pointer;
}

.import-trigger span {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 108px;
  padding: 10px 14px;
  border-radius: 999px;
  border: 1px solid rgba(14, 165, 233, 0.24);
  background: rgba(224, 242, 254, 0.85);
  color: #075985;
  font-size: 12px;
  font-weight: 600;
}

.import-library {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.import-library span {
  padding: 7px 10px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.92);
  border: 1px solid rgba(148, 163, 184, 0.16);
  color: #334155;
  font-size: 12px;
}

.import-card__footnote {
  margin: 0;
  font-size: 12px;
  color: #64748b;
}

.reward-panel {
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 12px;
  color: #1e293b;
}

.settings-stack {
  grid-column: 2;
  display: flex;
  flex-direction: column;
  gap: 16px;
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
  padding: 18px;
  display: flex;
  flex-direction: column;
  gap: 14px;
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

.config-field--wide {
  grid-column: 1 / -1;
}

.config-field span {
  color: #475569;
  font-size: 12px;
  font-weight: 600;
}

input[type="text"],
input[type="number"],
select,
textarea {
  width: 100%;
  padding: 10px 12px;
  border-radius: 10px;
  border: 1px solid rgba(100, 116, 139, 0.28);
  background: rgba(255, 255, 255, 0.96);
  color: #0f172a;
  font: inherit;
}

textarea {
  resize: vertical;
  min-height: 92px;
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

.form-hint {
  margin: 0;
  color: #fca5a5;
  font-size: 0.92rem;
}

@media (max-width: 1220px) {
  .panel-header,
  .panel-body {
    grid-template-columns: 1fr;
  }

  .intake-workbench,
  .settings-stack,
  .training-form {
    grid-column: 1 / -1;
  }
}

@media (max-width: 900px) {
  .intake-workbench__header {
    flex-direction: column;
  }

  .intake-workbench__status {
    width: 100%;
    min-width: 0;
  }
}

@media (max-width: 720px) {
  .scenario-details,
  .intake-workbench__status {
    grid-template-columns: 1fr;
  }

  .import-card__header,
  .training-form__header,
  .config-section__title {
    flex-direction: column;
    align-items: flex-start;
  }
}
</style>
