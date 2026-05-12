<template>
  <div class="training-page">
    <!-- Header -->
    <div class="training-page__header">
      <div class="training-page__title-group">
        <div class="training-page__title-icon"></div>
        <h1 class="training-page__title">模型训练</h1>
      </div>
    </div>

    <!-- Scrollable content area -->
    <div class="training-page__viewport" ref="viewportRef">
      <div class="training-page__content">
        <!-- Error banner -->
        <div v-if="loadError || actionError" class="alert alert--error">
          {{ loadError || actionError }}
        </div>

        <!-- ==================== 场景录入 ==================== -->
        <section class="section">
          <div class="section__header">
            <div class="section__header-bg"></div>
            <strong class="section__label">场景录入</strong>
            <span class="section__accent"></span>
          </div>

          <div class="form-grid form-grid--2col">
            <label class="field">
              <span class="field__label"><span class="field__required">*</span> 灾害类型</span>
              <select v-model="disasterType" class="field__input">
                <option value="flood">洪水</option>
                <option value="earthquake">地震</option>
                <option value="landslide">滑坡</option>
                <option value="typhoon">台风</option>
              </select>
            </label>

            <label class="field">
              <span class="field__label"><span class="field__required">*</span> 灾害等级</span>
              <select v-model="disasterSeverity" class="field__input">
                <option value="moderate">中等</option>
                <option value="severe">严重</option>
                <option value="critical">特别严重</option>
              </select>
            </label>

            <label class="field">
              <span class="field__label">受影响网格数</span>
              <input v-model.number="affectedGridCount" type="number" min="1" class="field__input" />
            </label>

            <label class="field">
              <span class="field__label">受影响人数</span>
              <input v-model.number="impactedPopulation" type="number" min="1" class="field__input" />
            </label>

            <label class="field">
              <span class="field__label">保障队伍</span>
              <input v-model="dispatchUnit" type="text" class="field__input" placeholder="前线应急通信保障队" />
            </label>

            <label class="field">
              <span class="field__label">优先设备组合</span>
              <input v-model="priorityEquipment" type="text" class="field__input" placeholder="背负式基站 + 多跳中继" />
            </label>

            <label class="field field--wide">
              <span class="field__label"><span class="field__required">*</span> 受灾区域说明</span>
              <textarea
                v-model="disasterNotes"
                rows="3"
                class="field__input field__input--textarea"
                placeholder="请输入受灾区域说明"
              ></textarea>
            </label>
          </div>
        </section>

        <!-- ==================== 算法选择 ==================== -->
        <section class="section">
          <div class="section__header">
            <div class="section__header-bg"></div>
            <strong class="section__label">导入组网算法</strong>
            <button type="button" class="section__import-btn" @click="handleImportPolicy">导入策略包</button>
            <span class="section__accent"></span>
          </div>

          <div class="algo-grid">
            <button
              v-for="algo in algorithmCards"
              :key="algo.value"
              type="button"
              :class="['algo-card', { 'algo-card--active': selectedAlgorithm === algo.value }]"
              @click="selectAlgorithm(algo.value)"
            >
              <span class="algo-card__name">{{ algo.label }}</span>
              <span class="algo-card__desc">{{ algo.desc }}</span>
            </button>
          </div>

          <div class="form-grid form-grid--2col" style="margin-top: 18px">
            <label class="field">
              <span class="field__label">策略来源</span>
              <select v-model="policySource" class="field__input">
                <option value="platform">平台内置 RL 算法</option>
                <option value="external">外部导入策略包</option>
              </select>
            </label>

            <div></div>

            <label class="field field--wide">
              <span class="field__label">组网算法说明</span>
              <textarea
                v-model="algorithmNotes"
                rows="3"
                class="field__input field__input--textarea"
                placeholder="填写组网算法说明"
              ></textarea>
            </label>
          </div>

          <!-- 训练记录 Tab -->
          <div class="history-tab-row">
            <button type="button" class="history-tab" @click="toggleHistoryPanel">
              训练记录
            </button>
          </div>
        </section>

        <!-- ==================== 训练记录面板 ==================== -->
        <section v-if="showHistoryPanel" class="section section--history">
          <div class="history-panel">
            <div class="history-panel__header">
              <strong class="history-panel__title">模型训练历史记录</strong>
              <button type="button" class="history-panel__close" @click="showHistoryPanel = false">&#10005;</button>
            </div>

            <!-- Filter row -->
            <div class="history-panel__filters">
              <select v-model="historyFilterAlgorithm" class="field__input" style="width:260px">
                <option value="">请选择算法</option>
                <option value="ppo">PPO（基线）</option>
                <option value="dqn">DQN（大动作空间）</option>
                <option value="a3c">A3C（多目标）</option>
                <option value="mppo">MPPO（多头策略）</option>
              </select>
              <select v-model="historyFilterScenario" class="field__input" style="width:260px">
                <option value="">请选择场景类型</option>
                <option value="flood">洪水</option>
                <option value="earthquake">地震</option>
                <option value="landslide">滑坡</option>
                <option value="typhoon">台风</option>
              </select>
              <button type="button" class="history-panel__query-btn" @click="fetchTrainingHistory">查询</button>
            </div>

            <!-- Table -->
            <div class="history-panel__table-wrap">
              <table class="history-table">
                <thead>
                  <tr>
                    <th>序号</th>
                    <th>场景名称</th>
                    <th>训练算法</th>
                    <th>场景类型</th>
                    <th>执行状态</th>
                    <th>操作人</th>
                    <th>时间</th>
                    <th>操作</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-if="!filteredHistory.length">
                    <td colspan="8" class="history-table__empty">暂无训练记录</td>
                  </tr>
                  <tr v-for="(record, idx) in paginatedHistory" :key="record.id || idx">
                    <td>{{ (historyPage - 1) * historyPageSize + idx + 1 }}</td>
                    <td>{{ formatScenarioName(record.scenario_name) }}</td>
                    <td>{{ (record.algorithm || '').toUpperCase() }}</td>
                    <td>{{ formatDisasterType(record.disaster_type) }}</td>
                    <td>
                      <span :class="['status-badge', statusBadgeClass(record.status)]">
                        {{ statusLabel(record.status) }}
                      </span>
                    </td>
                    <td>{{ record.operator || '系统' }}</td>
                    <td>{{ formatTime(record.created_at || record.updated_at) }}</td>
                    <td>
                      <button type="button" class="history-action-btn" @click="viewHistoryDetail(record)">查看</button>
                      <button type="button" class="history-action-btn" @click="deleteHistoryRecord(record)">删除</button>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>

            <!-- Pagination -->
            <div class="history-panel__pagination" v-if="filteredHistory.length">
              <span class="history-panel__total">共 {{ filteredHistory.length }} 条</span>
              <div class="history-panel__pages">
                <button
                  v-for="page in totalHistoryPages"
                  :key="page"
                  type="button"
                  :class="['history-page-btn', { 'history-page-btn--active': page === historyPage }]"
                  @click="historyPage = page"
                >{{ page }}</button>
              </div>
              <span class="history-panel__page-size">{{ historyPageSize }}条/页</span>
            </div>
          </div>
        </section>

        <!-- ==================== 参数设置 ==================== -->
        <section class="section">
          <div class="section__header">
            <div class="section__header-bg"></div>
            <strong class="section__label">参数设置</strong>
            <span class="section__accent"></span>
          </div>

          <div class="tabs">
            <button
              v-for="tab in paramTabs"
              :key="tab.key"
              type="button"
              :class="['tab', { 'tab--active': activeParamTab === tab.key }]"
              @click="activeParamTab = tab.key"
            >
              {{ tab.label }}
            </button>
          </div>

          <!-- 算法参数 -->
          <div v-show="activeParamTab === 'algorithm'" class="form-grid form-grid--3col">
            <label class="field">
              <span class="field__label"><span class="field__required">*</span> 总训练步数</span>
              <input v-model.number="totalTimesteps" type="number" min="1000" step="1000" class="field__input" />
            </label>

            <label class="field">
              <span class="field__label"><span class="field__required">*</span> 学习率</span>
              <input v-model.number="learningRate" type="number" min="0.00001" max="0.01" step="0.00001" class="field__input" />
            </label>

            <label class="field">
              <span class="field__label"><span class="field__required">*</span> 折扣因子 γ</span>
              <input v-model.number="discountFactor" type="number" min="0.8" max="0.999" step="0.001" class="field__input" />
            </label>

            <label class="field">
              <span class="field__label"><span class="field__required">*</span> Batch Size</span>
              <input v-model.number="batchSize" type="number" min="32" max="2048" step="32" class="field__input" />
            </label>

            <label class="field">
              <span class="field__label"><span class="field__required">*</span> Rollout 步长</span>
              <input v-model.number="rolloutSteps" type="number" min="64" max="4096" step="64" class="field__input" />
            </label>

            <label class="field">
              <span class="field__label">熵系数</span>
              <input v-model.number="entropyCoef" type="number" min="0" max="0.2" step="0.001" class="field__input" />
            </label>

            <label class="field">
              <span class="field__label">Clip Range</span>
              <input v-model.number="clipRange" type="number" min="0.05" max="0.5" step="0.01" class="field__input" />
            </label>
          </div>

          <!-- 仿真场景参数 -->
          <div v-show="activeParamTab === 'simulation'" class="form-grid form-grid--3col">
            <label class="field">
              <span class="field__label">环境类型</span>
              <select v-model="envType" class="field__input">
                <option value="multimodal">多模融合环境</option>
                <option value="baseline">基线环境</option>
              </select>
            </label>

            <label class="field">
              <span class="field__label">评估方式</span>
              <select v-model="stochasticEval" class="field__input">
                <option :value="true">随机策略评估</option>
                <option :value="false">确定性策略评估</option>
              </select>
            </label>

            <label class="field">
              <span class="field__label">仿真时窗（小时）</span>
              <input v-model.number="simulationWindowHours" type="number" min="1" max="72" class="field__input" />
            </label>

            <label class="field">
              <span class="field__label">目标覆盖率 (%)</span>
              <input v-model.number="coverageTarget" type="number" min="10" max="100" class="field__input" />
            </label>

            <label class="field">
              <span class="field__label">业务负载等级</span>
              <select v-model="trafficLoadProfile" class="field__input">
                <option value="low">低负载</option>
                <option value="medium">中负载</option>
                <option value="high">高负载</option>
              </select>
            </label>

            <label class="field">
              <span class="field__label">恢复目标</span>
              <select v-model="priorityObjective" class="field__input">
                <option value="coverage_first">覆盖优先</option>
                <option value="balanced">覆盖与时延平衡</option>
                <option value="capacity_first">容量优先</option>
              </select>
            </label>

            <label class="field">
              <span class="field__label">日志刷新窗口</span>
              <input v-model.number="logWindow" type="number" min="10" max="200" step="5" class="field__input" />
            </label>

            <label class="field">
              <span class="field__label">评估间隔</span>
              <input v-model.number="evalInterval" type="number" min="1000" max="50000" step="1000" class="field__input" />
            </label>

            <label class="field">
              <span class="field__label">训练后回放</span>
              <select v-model="autoReplay" class="field__input">
                <option :value="true">自动生成回放</option>
                <option :value="false">仅保留训练日志</option>
              </select>
            </label>
          </div>
        </section>

        <!-- ==================== Action Bar ==================== -->
        <div class="action-bar">
          <button
            type="button"
            class="btn-start"
            :class="{ 'btn-start--stop': runStatus === 'running' }"
            :disabled="!selectedScenarioName || isStarting"
            @click="handleMainAction"
          >
            {{ actionButtonLabel }}
          </button>
          <span class="action-bar__hint" v-if="!selectedScenarioName">
            请等待场景加载完成后再启动训练
          </span>
        </div>

        <!-- ==================== Training Monitor ==================== -->
        <section v-if="showMonitor" class="section section--monitor">
          <div class="section__header">
            <div class="section__header-bg"></div>
            <strong class="section__label">训练结果</strong>
            <span class="section__accent"></span>
          </div>

          <TrainingMonitor :events="eventLog" :status="runStatus" />
        </section>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import axios from "axios";
import TrainingMonitor from "./TrainingMonitor.vue";
import { rescueApiBase } from "../utils/runtimeEndpoints";
import { saveReplaySessionFromSimulation } from "../utils/replaySessions";
import { formatScenarioName } from "../utils/scenarioLabels";

const API_BASE = rescueApiBase;

const viewportRef = ref(null);

const scenarios = ref([]);
const loadError = ref("");
const actionError = ref("");
const isStarting = ref(false);
const runStatus = ref("Idle");
const eventLog = ref([]);
const selectedRewardMode = ref(null);
const replayRunIdInFlight = ref(null);
const activeRunMeta = ref(null);
const activeParamTab = ref("algorithm");

// History panel state
const showHistoryPanel = ref(false);
const trainingHistory = ref([]);
const historyFilterAlgorithm = ref("");
const historyFilterScenario = ref("");
const historyPage = ref(1);
const historyPageSize = ref(10);

let eventSource = null;

// Form state
const disasterType = ref("typhoon");
const disasterSeverity = ref("severe");
const affectedGridCount = ref(24);
const impactedPopulation = ref(320);
const dispatchUnit = ref("前线应急通信保障队");
const priorityEquipment = ref("背负式基站 + 多跳中继");
const disasterNotes = ref("");
const selectedAlgorithm = ref("ppo");
const policySource = ref("platform");
const algorithmNotes = ref("");
const totalTimesteps = ref(12000);
const learningRate = ref(0.0003);
const discountFactor = ref(0.99);
const batchSize = ref(256);
const rolloutSteps = ref(1024);
const entropyCoef = ref(0.01);
const clipRange = ref(0.2);
const envType = ref("multimodal");
const stochasticEval = ref(true);
const simulationWindowHours = ref(6);
const coverageTarget = ref(85);
const trafficLoadProfile = ref("high");
const priorityObjective = ref("coverage_first");
const logWindow = ref(50);
const evalInterval = ref(5000);
const autoReplay = ref(true);

const algorithmCards = [
  { value: "ppo", label: "PPO", desc: "基线" },
  { value: "dqn", label: "DQN", desc: "大动作空间" },
  { value: "a3c", label: "A3C", desc: "多目标" },
  { value: "mppo", label: "MPPO", desc: "多头策略" },
];

const paramTabs = [
  { key: "algorithm", label: "算法参数" },
  { key: "simulation", label: "仿真场景参数" },
];

const selectedScenario = computed(() => {
  if (!scenarios.value.length) return null;
  const mapped = scenarios.value.find((item) => item.disaster_type === disasterType.value);
  return mapped || scenarios.value[0];
});

const selectedScenarioName = computed(() => selectedScenario.value?.name || null);
const selectedScenarioLabel = computed(() =>
  selectedScenarioName.value ? formatScenarioName(selectedScenarioName.value) : "未加载"
);
const showMonitor = computed(() => runStatus.value !== "Idle" || eventLog.value.length > 0);

const actionButtonLabel = computed(() => {
  if (isStarting.value) return "启动中...";
  if (runStatus.value === "running") return "停止训练";
  return "启动训练";
});

// History computed
const filteredHistory = computed(() => {
  let items = trainingHistory.value;
  if (historyFilterAlgorithm.value) {
    items = items.filter((r) => r.algorithm === historyFilterAlgorithm.value);
  }
  if (historyFilterScenario.value) {
    items = items.filter((r) => r.disaster_type === historyFilterScenario.value);
  }
  return items;
});

const totalHistoryPages = computed(() => Math.max(1, Math.ceil(filteredHistory.value.length / historyPageSize.value)));

const paginatedHistory = computed(() => {
  const start = (historyPage.value - 1) * historyPageSize.value;
  return filteredHistory.value.slice(start, start + historyPageSize.value);
});

const formatDisasterType = (type) => {
  const map = { flood: "洪水", earthquake: "地震", landslide: "滑坡", typhoon: "台风" };
  return map[type] || type || "--";
};

const formatTime = (ts) => {
  if (!ts) return "--";
  return new Date(Number(ts) * 1000).toLocaleString("zh-CN", { hour12: false });
};

const statusLabel = (status) => {
  const map = { running: "运行中", completed: "已完成", failed: "失败", stopped: "已停止" };
  return map[status] || status || "已完成";
};

const statusBadgeClass = (status) => {
  if (status === "running") return "status-badge--running";
  if (status === "failed") return "status-badge--failed";
  return "status-badge--completed";
};

const handleMainAction = () => {
  if (runStatus.value === "running") {
    stopTraining();
  } else {
    startTraining();
  }
};

const handleImportPolicy = () => {
  policySource.value = "external";
};

const toggleHistoryPanel = () => {
  showHistoryPanel.value = !showHistoryPanel.value;
  if (showHistoryPanel.value) {
    historyPage.value = 1;
    fetchTrainingHistory();
  }
};

// --- API calls ---

const fetchScenarios = async () => {
  loadError.value = "";
  try {
    const { data } = await axios.get(`${API_BASE}/scenarios`);
    scenarios.value = Array.isArray(data?.scenarios) ? data.scenarios : [];
  } catch (error) {
    console.error("Failed to load scenarios", error);
    loadError.value = `无法连接训练后端: ${error?.message || "未知错误"}`;
  }
};

const fetchTrainingHistory = async () => {
  try {
    const { data } = await axios.get(`${API_BASE}/train/artifacts`, { timeout: 10000 });
    const artifacts = Array.isArray(data?.artifacts) ? data.artifacts : [];
    trainingHistory.value = artifacts.map((a) => ({
      id: a.checkpoint_path || a.scenario_name + (a.algorithm || ""),
      scenario_name: a.scenario_name,
      algorithm: a.algorithm,
      disaster_type: a.disaster_type || "",
      status: a.status || "completed",
      operator: a.operator || "系统",
      created_at: a.created_at || a.updated_at,
      updated_at: a.updated_at,
      checkpoint_path: a.checkpoint_path,
      reward_mode: a.reward_mode,
    }));
  } catch (error) {
    console.warn("Failed to load training history", error);
  }
};

const viewHistoryDetail = (record) => {
  if (record.checkpoint_path) {
    eventLog.value = [
      ...eventLog.value,
      {
        type: "info",
        timestamp: Date.now() / 1000,
        message: `查看训练记录: ${formatScenarioName(record.scenario_name)} / ${(record.algorithm || "").toUpperCase()} 路径: ${record.checkpoint_path}`,
      },
    ];
  }
};

const deleteHistoryRecord = async (record) => {
  try {
    trainingHistory.value = trainingHistory.value.filter((r) => r.id !== record.id);
  } catch (error) {
    console.warn("Failed to delete history record", error);
  }
};

const selectAlgorithm = (value) => {
  selectedAlgorithm.value = value;
  if (value === "dqn" && totalTimesteps.value < 40000) {
    totalTimesteps.value = 40000;
  }
  if (value !== "dqn" && totalTimesteps.value === 40000) {
    totalTimesteps.value = 12000;
  }
};

const closeEventSource = () => {
  if (eventSource) {
    eventSource.close();
    eventSource = null;
  }
};

const resolveTrainingCheckpoint = async (runMeta) => {
  const matchesRun = (artifact) =>
    artifact?.checkpoint_path &&
    artifact?.scenario_name === runMeta.scenarioName &&
    artifact?.algorithm === runMeta.algorithm;

  try {
    const { data } = await axios.get(`${API_BASE}/train/latest-artifact`, { timeout: 10000 });
    if (matchesRun(data)) return data.checkpoint_path;
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
  if (!runMeta?.runId || replayRunIdInFlight.value === runMeta.runId) return;
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

    saveReplaySessionFromSimulation({
      scenarioName: runMeta.scenarioName,
      algorithm: runMeta.algorithm,
      result: { ...data, source: "training" },
    });
  } catch (error) {
    console.error("Failed to generate replay from training", error);
  } finally {
    replayRunIdInFlight.value = null;
  }
};

const subscribeToEvents = (runId) => {
  runStatus.value = "running";
  eventSource = new EventSource(`${API_BASE}/train/${runId}/stream`);

  eventSource.onmessage = (event) => {
    if (!event.data) return;
    try {
      const payload = JSON.parse(event.data);
      if (payload.type === "end") {
        runStatus.value = payload.status;
        closeEventSource();
        return;
      }

      eventLog.value = [...eventLog.value, payload].slice(-80);

      if (payload.type === "status" && payload.payload?.state) {
        runStatus.value = payload.payload.state;
        if (payload.payload.state === "completed" && activeRunMeta.value?.runId === runId) {
          void generateReplayFromTraining(activeRunMeta.value);
        }
      }
    } catch (error) {
      console.warn("Failed to parse training event", error);
    }
  };

  eventSource.onerror = () => {
    closeEventSource();
    runStatus.value = "disconnected";
  };
};

const startTraining = async () => {
  if (!selectedScenarioName.value) return;

  isStarting.value = true;
  actionError.value = "";
  eventLog.value = [
    {
      type: "scene_import",
      timestamp: Date.now() / 1000,
      message: `已确认 ${selectedScenarioLabel.value} 场景，准备启动 ${selectedAlgorithm.value.toUpperCase()} 训练。`,
    },
  ];
  runStatus.value = "starting";
  closeEventSource();

  try {
    const rewardMode =
      selectedRewardMode.value ||
      selectedScenario.value?.default_reward_profile ||
      selectedScenario.value?.reward_profiles?.[0]?.key ||
      null;

    const { data } = await axios.post(`${API_BASE}/train`, {
      scenario_name: selectedScenarioName.value,
      env_type: envType.value,
      algorithm: selectedAlgorithm.value,
      total_timesteps: totalTimesteps.value,
      stochastic_eval: stochasticEval.value,
      reward_mode: rewardMode,
    });

    activeRunMeta.value = {
      runId: data.run_id,
      scenarioName: selectedScenarioName.value,
      algorithm: selectedAlgorithm.value,
      rewardMode,
    };

    subscribeToEvents(data.run_id);
  } catch (error) {
    console.error("Failed to start training", error);
    runStatus.value = "error";
    actionError.value = `启动训练失败: ${error?.message || "未知错误"}`;
  } finally {
    isStarting.value = false;
  }
};

const stopTraining = () => {
  closeEventSource();
  runStatus.value = "stopped";
  eventLog.value = [
    ...eventLog.value,
    {
      type: "status",
      timestamp: Date.now() / 1000,
      message: "用户手动停止了训练。",
    },
  ];
};

watch(selectedScenario, (scenario) => {
  if (!scenario) return;
  selectedRewardMode.value =
    scenario.default_reward_profile || scenario.reward_profiles?.[0]?.key || null;
  impactedPopulation.value = Number(scenario.num_users || impactedPopulation.value);
  priorityEquipment.value =
    scenario.base_stations?.slice(0, 2).map((item) => item.label || item.name).filter(Boolean).join(" + ") ||
    priorityEquipment.value;
}, { immediate: true });

onMounted(fetchScenarios);
onBeforeUnmount(closeEventSource);
</script>

<style scoped>
/* ===== Page shell ===== */
.training-page {
  position: relative;
  width: 1920px;
  height: 1010px;
  overflow: hidden;
  font-family: "Source Han Sans CN", "Noto Sans SC", "PingFang SC", "Microsoft YaHei", system-ui, -apple-system, sans-serif;
  background:
    linear-gradient(180deg, #d6e8fc 0%, #eaf4fd 30%, #f0f5fa 60%, #f0f2f5 100%);
}

/* ===== Header ===== */
.training-page__header {
  position: absolute;
  left: 133px;
  top: 14px;
  z-index: 4;
}

.training-page__title-group {
  display: flex;
  align-items: center;
  gap: 10px;
}

.training-page__title-icon {
  width: 128px;
  height: 42px;
  border-radius: 8px;
  background: linear-gradient(135deg, #00e3ff, #1890ff, #0050b3);
  box-shadow: 0 0 20px rgba(0, 200, 244, 0.4);
}

.training-page__title {
  margin: 0;
  font-size: 20px;
  font-weight: 700;
  color: #1890ff;
  text-shadow: 0 0 20px rgba(0, 200, 244, 0.5);
}

/* ===== Viewport ===== */
.training-page__viewport {
  position: absolute;
  left: 147px;
  top: 68px;
  width: 1631px;
  height: 878px;
  overflow-y: auto;
  overflow-x: hidden;
  scrollbar-width: none;
  -ms-overflow-style: none;
}

.training-page__viewport::-webkit-scrollbar {
  display: none;
}

.training-page__content {
  display: flex;
  flex-direction: column;
  gap: 3px;
  padding-bottom: 40px;
}

/* ===== Alerts ===== */
.alert {
  padding: 12px 16px;
  border-radius: 8px;
  font-size: 14px;
  line-height: 1.5;
  margin-bottom: 6px;
}

.alert--error {
  background: rgba(248, 216, 215, 0.92);
  border: 1px solid rgba(220, 114, 116, 0.3);
  color: #b42318;
}

/* ===== Sections ===== */
.section {
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.76);
  border: 1px solid rgba(233, 233, 233, 0.5);
  padding: 18px 20px 20px;
  box-shadow: 0 0 5px rgba(246, 246, 254, 0.8) inset;
}

.section--monitor {
  padding: 0;
  border: 0;
  background: transparent;
  box-shadow: none;
}

.section--history {
  padding: 0;
  background: rgba(255, 255, 255, 0.92);
}

.section__header {
  position: relative;
  display: flex;
  align-items: center;
  margin-bottom: 18px;
  margin-left: -20px;
  margin-right: -20px;
  padding-left: 20px;
  padding-right: 20px;
}

.section__header-bg {
  position: absolute;
  inset: -6px -20px -2px -20px;
  background: linear-gradient(180deg,
    rgba(15, 23, 42, 0.08) 0%,
    rgba(15, 23, 42, 0.04) 50%,
    rgba(15, 23, 42, 0) 100%);
  border-bottom: 2px solid rgba(5, 183, 223, 0.35);
  border-radius: 6px 6px 0 0;
}

.section__label {
  position: relative;
  font-size: 16px;
  font-weight: 700;
  color: #333333;
  padding-left: 14px;
  white-space: nowrap;
}

.section__label::before {
  content: "";
  position: absolute;
  left: 0;
  top: 2px;
  bottom: 2px;
  width: 6px;
  border-radius: 2px;
  background: linear-gradient(180deg, rgba(111, 202, 223, 1), rgba(5, 183, 223, 1));
}

.section__import-btn {
  position: relative;
  margin-left: auto;
  margin-right: 12px;
  padding: 8px 18px;
  border: 1px solid #b7e0fe;
  border-radius: 10px;
  background: #3961f6;
  color: #ffffff;
  font-size: 14px;
  font-family: inherit;
  cursor: pointer;
  transition: background 0.2s;
  white-space: nowrap;
}

.section__import-btn:hover {
  background: #409eff;
}

.section__accent {
  position: relative;
  flex: 1;
  margin-left: 14px;
  height: 2px;
  background: linear-gradient(90deg, rgba(5, 183, 223, 0.7), transparent);
}

/* ===== Form Grid ===== */
.form-grid {
  display: grid;
  gap: 14px 16px;
}

.form-grid--2col {
  grid-template-columns: 1fr 1fr;
}

.form-grid--3col {
  grid-template-columns: repeat(3, 1fr);
}

/* ===== Form Fields ===== */
.field {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.field--wide {
  grid-column: 1 / -1;
}

.field__label {
  font-size: 14px;
  color: #333333;
  font-weight: 400;
}

.field__required {
  color: #ff0000;
}

.field__input {
  height: 45px;
  padding: 0 11px;
  border: 1px solid #e9e9e9;
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.76);
  color: #333333;
  font-size: 16px;
  font-family: inherit;
  outline: none;
  box-shadow: 0 0 5px rgba(246, 246, 254, 0.8) inset;
  transition: border-color 0.2s, box-shadow 0.2s;
}

.field__input::placeholder {
  color: #9ea6bb;
}

.field__input:focus {
  border-color: #1890ff;
  box-shadow: 0 0 5px rgba(246, 246, 254, 0.8) inset, 0 0 0 1px rgba(24, 144, 255, 0.12);
}

.field__input--textarea {
  height: auto;
  min-height: 83px;
  padding: 10px 11px;
  resize: vertical;
}

.field select.field__input {
  appearance: none;
  cursor: pointer;
  background-color: #ffffff;
  background-image:
    linear-gradient(45deg, transparent 50%, #9ea6bb 50%),
    linear-gradient(135deg, #9ea6bb 50%, transparent 50%);
  background-position:
    calc(100% - 18px) calc(50% - 3px),
    calc(100% - 12px) calc(50% - 3px);
  background-size: 6px 6px, 6px 6px;
  background-repeat: no-repeat;
  padding-right: 36px;
}

.field select.field__input option {
  background: #ffffff;
  color: #333333;
}

.field select.field__input:hover {
  border-color: #1890ff;
}

/* ===== Algorithm Cards ===== */
.algo-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 14px;
}

.algo-card {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 6px;
  min-height: 86px;
  padding: 14px 10px;
  border: 1px solid rgba(183, 224, 254, 0.5);
  border-radius: 10px;
  background: rgba(255, 255, 255, 0.7);
  color: #333333;
  cursor: pointer;
  transition: all 0.2s ease;
  font-family: inherit;
  box-shadow: 0 0 5px rgba(246, 246, 254, 0.5);
}

.algo-card:hover {
  border-color: #1890ff;
  background: rgba(231, 238, 255, 0.5);
  color: #333333;
}

.algo-card--active {
  border-color: #3961f6;
  background: rgba(231, 238, 255, 0.5);
  color: #3961f6;
  box-shadow: 0 0 6px rgba(57, 97, 246, 0.15);
}

.algo-card__name {
  font-size: 18px;
  font-weight: 700;
}

.algo-card__desc {
  font-size: 13px;
  opacity: 0.7;
}

/* ===== History Tab ===== */
.history-tab-row {
  display: flex;
  justify-content: center;
  margin-top: 18px;
  padding-top: 14px;
  border-top: 1px solid rgba(233, 233, 233, 0.5);
}

.history-tab {
  padding: 6px 20px;
  border: 1px solid #f2f2f2;
  border-radius: 8px;
  background: transparent;
  color: #0079fe;
  font-size: 14px;
  font-family: inherit;
  cursor: pointer;
  transition: all 0.2s;
}

.history-tab:hover {
  background: rgba(0, 102, 255, 0.067);
}

/* ===== History Panel ===== */
.history-panel {
  padding: 20px;
}

.history-panel__header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.history-panel__title {
  font-size: 18px;
  font-weight: 500;
  color: #333333;
}

.history-panel__close {
  width: 30px;
  height: 30px;
  border: 0;
  border-radius: 6px;
  background: transparent;
  color: #999999;
  font-size: 14px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
}

.history-panel__close:hover {
  background: rgba(0, 0, 0, 0.05);
  color: #333333;
}

.history-panel__filters {
  display: flex;
  gap: 10px;
  align-items: center;
  margin-bottom: 16px;
}

.history-panel__query-btn {
  height: 40px;
  padding: 0 18px;
  border: 0;
  border-radius: 6px;
  background: #3961f6;
  color: #ffffff;
  font-size: 14px;
  font-family: inherit;
  cursor: pointer;
}

.history-panel__query-btn:hover {
  opacity: 0.85;
}

.history-panel__table-wrap {
  overflow-x: auto;
}

.history-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
}

.history-table thead th {
  padding: 12px 10px;
  border-bottom: 1px solid #e4e4e4;
  background: #f7f8fa;
  color: #333333;
  font-weight: 700;
  font-size: 16px;
  text-align: left;
  white-space: nowrap;
}

.history-table tbody td {
  padding: 12px 10px;
  border-bottom: 1px solid rgba(228, 228, 228, 0.5);
  color: #333333;
  white-space: nowrap;
}

.history-table__empty {
  text-align: center !important;
  color: #999999 !important;
  padding: 40px 10px !important;
}

.status-badge {
  display: inline-block;
  padding: 3px 12px;
  border-radius: 4px;
  font-size: 13px;
  font-weight: 500;
}

.status-badge--completed {
  background: rgba(220, 243, 227, 0.8);
  color: #339900;
}

.status-badge--running {
  background: rgba(231, 238, 255, 0.6);
  color: #3961f6;
}

.status-badge--failed {
  background: rgba(248, 216, 215, 0.6);
  color: #dc7274;
}

.history-action-btn {
  padding: 4px 12px;
  border: 0;
  border-radius: 3px;
  background: transparent;
  color: #3961f6;
  font-size: 14px;
  font-family: inherit;
  cursor: pointer;
  transition: background 0.2s;
}

.history-action-btn:hover {
  background: rgba(57, 97, 246, 0.08);
}

.history-action-btn + .history-action-btn {
  margin-left: 4px;
}

/* ===== History Pagination ===== */
.history-panel__pagination {
  display: flex;
  justify-content: flex-end;
  align-items: center;
  gap: 10px;
  margin-top: 16px;
  padding-top: 12px;
}

.history-panel__total {
  font-size: 14px;
  color: #999999;
}

.history-panel__pages {
  display: flex;
  gap: 4px;
}

.history-page-btn {
  width: 35px;
  height: 35px;
  border: 1px solid #e4e4e4;
  border-radius: 3px;
  background: #ffffff;
  color: #999999;
  font-size: 14px;
  font-family: inherit;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
}

.history-page-btn:hover {
  border-color: #0079fe;
}

.history-page-btn--active {
  border-color: #0079fe;
  background: #0079fe;
  color: #ffffff;
}

.history-panel__page-size {
  font-size: 14px;
  color: #999999;
}

/* ===== Tabs ===== */
.tabs {
  display: flex;
  gap: 4px;
  margin-bottom: 18px;
}

.tab {
  padding: 9px 20px;
  border: 1px solid #e9e9e9;
  border-radius: 8px 8px 0 0;
  background: rgba(255, 255, 255, 0.5);
  color: #666666;
  font-size: 14px;
  font-family: inherit;
  cursor: pointer;
  transition: all 0.2s;
  border-bottom: 0;
}

.tab:hover {
  color: #333333;
  border-color: #1890ff;
  background: rgba(255, 255, 255, 0.75);
}

.tab--active {
  background: rgba(231, 238, 255, 0.5);
  border-color: #3961f6;
  color: #3961f6;
  font-weight: 600;
}

/* ===== Action Bar ===== */
.action-bar {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: 16px;
  margin-top: 4px;
}

.btn-start {
  padding: 11px 28px;
  border: 1px solid #b7e0fe;
  border-radius: 10px;
  background: #3961f6;
  color: #ffffff;
  font-size: 16px;
  font-weight: 400;
  font-family: inherit;
  cursor: pointer;
  transition: all 0.2s ease;
  box-shadow: 0 0 5px rgba(246, 246, 254, 0.5);
}

.btn-start:hover:not(:disabled) {
  background: #409eff;
  border-color: #b7e0fe;
}

.btn-start:disabled {
  cursor: not-allowed;
  opacity: 0.5;
}

.btn-start--stop {
  background: #dc7274;
  border-color: #f8d8d7;
}

.btn-start--stop:hover:not(:disabled) {
  background: #e8898b;
  border-color: #f8d8d7;
}

.action-bar__hint {
  font-size: 14px;
  color: #9ea6bb;
}
</style>
