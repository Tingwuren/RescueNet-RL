<template>
  <div class="training-page">
    <img class="page-bg" :src="assetUrl('images/模型训练/u537.png')" alt="" />
    <img class="page-panel-shadow" :src="assetUrl('images/模型训练/u538.png')" alt="" />

    <main class="training-shell" aria-label="模型训练">
      <div class="training-shell__scroll" :style="{ height: `${pageHeight}px` }">
        <div class="page-title">
          <img class="page-title__ribbon" :src="assetUrl('images/模型训练/u541.png')" alt="" />
          <h1>模型训练</h1>
        </div>

        <button type="button" class="record-button" @click="toggleHistoryPanel">训练记录</button>

        <div v-if="loadError || actionError" class="module-error">
          {{ loadError || actionError }}
        </div>

        <section class="module-panel scenario-panel" aria-label="场景录入">
          <header class="module-heading">
            <div>
              <i></i>
              <h2>场景录入</h2>
              <p>{{ selectedScenarioLabel }}</p>
            </div>
            <div class="module-actions">
              <button type="button" class="ghost-button" @click="loadSceneSnapshot">导入场景</button>
              <button type="button" class="ghost-button ghost-button--blue" @click="saveSceneSnapshot">保存场景</button>
            </div>
          </header>

          <div class="dataset-controls">
            <div class="dataset-choice dataset-choice--scenario">
              <span class="dataset-choice__label">灾害场景</span>
              <div class="dataset-option-grid dataset-option-grid--scenario" aria-label="灾害场景选择">
                <button
                  v-for="option in disasterScenarioOptions"
                  :key="option.key"
                  type="button"
                  :class="['dataset-option-card', { 'dataset-option-card--active': disasterType === option.key }]"
                  @click="selectDisasterType(option.key)"
                >
                  <span class="dataset-option-card__name">{{ option.label }}</span>
                  <span class="dataset-option-card__desc">{{ option.description }}</span>
                </button>
              </div>
            </div>

            <div class="dataset-choice dataset-choice--severity">
              <span class="dataset-choice__label">受灾等级</span>
              <div class="dataset-option-grid dataset-option-grid--severity" aria-label="受灾等级选择">
                <button
                  v-for="option in disasterSeverityOptions"
                  :key="option.key"
                  type="button"
                  :class="['dataset-option-card', { 'dataset-option-card--active': disasterSeverity === option.key }]"
                  @click="selectSeverity(option.key)"
                >
                  <span class="dataset-option-card__name">{{ option.label }}</span>
                  <span class="dataset-option-card__desc">{{ option.description }}</span>
                </button>
              </div>
            </div>
          </div>

          <div v-if="selectedSeverityInsight.length" class="severity-insight-card" aria-label="当前灾损指标">
            <div class="severity-insight-card__title">
              <strong>{{ severityLabel(selectedScenario) }}灾损指标</strong>
              <span>{{ selectedSeveritySummary }}</span>
            </div>
            <div class="severity-insight-grid">
              <span v-for="item in selectedSeverityInsight" :key="item.label">
                <small>{{ item.label }}</small>
                <strong>{{ item.value }}</strong>
                <em>{{ item.hint }}</em>
              </span>
            </div>
          </div>

          <div class="summary-grid" aria-label="当前场景统计">
            <article v-for="item in scenarioStats" :key="item.label">
              <small>{{ item.label }}</small>
              <strong>{{ item.value }}</strong>
              <span>{{ item.hint }}</span>
            </article>
          </div>

          <div class="scenario-description-card" aria-label="受灾区域说明">
            <h3>受灾区域说明</h3>
            <p>{{ scenarioDisasterDescription }}</p>
          </div>

          <div class="device-access-card" aria-label="应急设备接入">
            <div class="sub-panel-heading">
              <div>
                <h3>应急设备接入</h3>
                <p>{{ deviceAccessSummary }}</p>
              </div>
              <div class="module-actions">
                <button
                  type="button"
                  class="ghost-button danger-link"
                  :disabled="isSyncingDevices || !selectedScenarioName || !accessDevices.length"
                  @click="clearTrainingResidualNetwork"
                >
                  清空残余网络
                </button>
                <button type="button" class="ghost-button" :disabled="isSyncingDevices || !selectedScenarioName" @click="addAccessDevice">
                  {{ isSyncingDevices ? "同步中..." : "添加设备" }}
                </button>
              </div>
            </div>

            <div class="device-table">
              <div class="device-table__head">
                <span>序号</span>
                <span>设备名称</span>
                <span>支持模式</span>
                <span>数量</span>
                <span>X 网格</span>
                <span>Y 网格</span>
                <span>状态</span>
                <span>操作</span>
              </div>
              <div v-if="!accessDevices.length" class="device-table__empty">
                暂无设备接入，当前训练将按无残余网络执行；点击“添加设备”可重新接入设备。
              </div>
              <div v-for="(device, index) in accessDevices" :key="device.id" class="device-table__row">
                <span>{{ index + 1 }}</span>
                <span class="device-name-cell" :title="accessDeviceDisplayName(device)">
                  {{ accessDeviceDisplayName(device) }}
                </span>
                <span>{{ deviceModesForValue(device.device, device.mode) }}</span>
                <input v-model.number="device.count" type="number" min="1" :disabled="isSyncingDevices" @change="syncAccessDevice(index, { persist: true })" />
                <input v-model.number="device.x" type="number" min="0" :max="gridBounds.maxX" :disabled="isSyncingDevices" @change="syncAccessDevice(index, { persist: true })" />
                <input v-model.number="device.y" type="number" min="0" :max="gridBounds.maxY" :disabled="isSyncingDevices" @change="syncAccessDevice(index, { persist: true })" />
                <span class="device-state">{{ device.statusLabel || "已导入" }}</span>
                <button type="button" class="ghost-button danger-link device-remove-button" :disabled="isSyncingDevices" @click="removeAccessDevice(index)">
                  移除
                </button>
              </div>
            </div>
          </div>
        </section>

        <section class="module-panel algorithm-panel" aria-label="导入组网算法">
          <header class="module-heading">
            <div>
              <i></i>
              <h2>导入组网算法</h2>
              <p>当前选择：{{ selectedAlgorithmLabel }}</p>
            </div>
          </header>

          <div class="algorithm-card-grid">
            <button
              v-for="algo in algorithmCards"
              :key="algo.value"
              type="button"
              :class="['algorithm-card', { 'algorithm-card--active': selectedAlgorithm === algo.value }]"
              @click="selectAlgorithm(algo.value)"
            >
              <span class="algorithm-card__name">{{ algo.label }}</span>
              <span class="algorithm-card__desc">{{ algo.desc }}</span>
            </button>
          </div>
        </section>

        <section class="module-panel reward-panel" aria-label="奖励配置设置">
          <header class="module-heading">
            <div>
              <i></i>
              <h2>奖励配置设置</h2>
              <p>当前选择：{{ rewardModeLabel(selectedRewardMode) }}</p>
            </div>
          </header>

          <div class="reward-card-grid">
            <button
              v-for="option in rewardModeCards"
              :key="option.value"
              type="button"
              :class="['reward-card', { 'reward-card--active': selectedRewardMode === option.value }]"
              @click="selectRewardMode(option)"
            >
              <span class="reward-card__name">{{ option.label }}</span>
              <span class="reward-card__desc">{{ option.desc }}</span>
            </button>
          </div>
        </section>

        <section class="module-panel parameter-panel" aria-label="参数设置">
          <header class="module-heading">
            <div>
              <i></i>
              <h2>参数设置</h2>
              <p>算法参数和仿真场景参数</p>
            </div>
            <button type="button" class="ghost-button ghost-button--blue" @click="applyTrainingParameters">
              应用参数
            </button>
          </header>

          <div class="tabs">
            <button
              v-for="tab in paramTabs"
              :key="tab.key"
              type="button"
              :class="['tab', { 'tab--active': activeParamTab === tab.key }]"
              @click="selectParamTab(tab.key)"
            >
              {{ tab.label }}
            </button>
          </div>

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

        <div class="action-bar">
          <span class="run-state">当前状态：{{ statusLabel(runStatus) }}</span>
          <button
            type="button"
            class="primary-button train-button"
            :class="{ 'train-button--stop': runStatus === 'running' }"
            :disabled="!selectedScenarioName || isStarting"
            @click="handleMainAction"
          >
            {{ actionButtonLabel }}
          </button>
          <span class="action-bar__hint" v-if="!selectedScenarioName">
            请等待场景加载完成后再启动训练
          </span>
        </div>

        <section class="module-panel result-panel" aria-label="训练结果">
          <header class="module-heading">
            <div>
              <i></i>
              <h2>训练结果</h2>
              <p>{{ eventLog.length ? `${eventLog.length} 条训练事件` : "等待训练事件" }}</p>
            </div>
          </header>
          <div class="result-stack">
            <StreamingTerminal
              title="实时终端输出"
              subtitle="同步显示设备管理、训练参数、后端训练流和跨页面操作输出。"
              :lines="trainingTerminalLines"
              :status="trainingTerminalStatus"
              placeholder="等待训练操作或其他页面终端输出..."
              exportable
              clearable
              @export="downloadTerminalLog"
              @clear="clearTerminalLog"
            />
            <TrainingMonitor :events="eventLog" :status="runStatus" :show-terminal="false" />
          </div>
        </section>
      </div>
    </main>

    <transition name="fade">
      <div v-if="showHistoryPanel" class="prototype-modal" @click.self="showHistoryPanel = false">
        <section class="prototype-dialog prototype-dialog--history" aria-label="模型训练历史记录">
          <header>
            <h2>模型训练历史记录</h2>
            <button type="button" @click="showHistoryPanel = false">关闭</button>
          </header>

          <div class="history-panel__filters">
            <select v-model="historyFilterAlgorithm" class="field__input">
              <option value="">请选择算法</option>
              <option value="ppo">PPO（基线）</option>
              <option value="dqn">DQN（大动作空间）</option>
              <option value="a3c">A3C（多目标）</option>
              <option value="mppo">MPPO（多头策略）</option>
              <option value="hmarl">HMARL（层次协同）</option>
            </select>
            <select v-model="historyFilterScenario" class="field__input">
              <option value="">请选择场景类型</option>
              <option value="暴雨">暴雨</option>
              <option value="台风">台风</option>
              <option value="地震">地震</option>
            </select>
            <button type="button" class="primary-button" @click="fetchTrainingHistory">查询</button>
          </div>

          <table>
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
                <td colspan="8" class="empty-row">暂无训练记录</td>
              </tr>
              <tr v-for="(record, idx) in paginatedHistory" :key="record.id || idx">
                <td>{{ (historyPage - 1) * historyPageSize + idx + 1 }}</td>
                <td>{{ trainingScenarioName(record) }}</td>
                <td>{{ (record.algorithm || "").toUpperCase() }}</td>
                <td>{{ trainingScenarioTypeName(record) }}</td>
                <td>
                  <span :class="['status-badge', statusBadgeClass(record.status)]">
                    {{ statusLabel(record.status) }}
                  </span>
                </td>
                <td>{{ record.operator || "系统" }}</td>
                <td>{{ formatTime(record.created_at || record.updated_at) }}</td>
                <td>
                  <button type="button" class="history-action-btn" @click="viewHistoryDetail(record)">查看</button>
                  <button type="button" class="history-action-btn" @click="deleteHistoryRecord(record)">删除</button>
                </td>
              </tr>
            </tbody>
          </table>

          <div class="history-panel__pagination" v-if="filteredHistory.length">
            <span class="history-panel__total">共 {{ filteredHistory.length }} 条</span>
            <div class="history-panel__pages">
              <button
                v-for="page in totalHistoryPages"
                :key="page"
                type="button"
                :class="['history-page-btn', { 'history-page-btn--active': page === historyPage }]"
                @click="historyPage = page"
              >
                {{ page }}
              </button>
            </div>
            <span class="history-panel__page-size">{{ historyPageSize }}条/页</span>
          </div>
        </section>
      </div>
    </transition>

    <transition name="fade">
      <div v-if="showHistoryDetailModal" class="prototype-modal prototype-modal--detail" @click.self="closeHistoryDetailModal">
        <section
          class="prototype-dialog prototype-dialog--history-detail"
          role="dialog"
          aria-modal="true"
          aria-label="训练结果详情"
        >
          <header>
            <div>
              <h2>{{ historyDetailTitle }}</h2>
              <p>
                {{ (historyDetailRecord?.algorithm || "").toUpperCase() || "--" }}
                <span>·</span>
                {{ statusLabel(historyDetailRecord?.status || historyDetail?.status) }}
                <span>·</span>
                {{ formatTime(historyDetail?.updated_at || historyDetailRecord?.updated_at || historyDetailRecord?.created_at) }}
              </p>
            </div>
            <button type="button" @click="closeHistoryDetailModal">关闭</button>
          </header>

          <div v-if="isLoadingHistoryDetail" class="history-detail__loading">
            正在加载训练结果...
          </div>

          <div v-else-if="historyDetailError" class="history-detail__error">
            {{ historyDetailError }}
          </div>

          <div v-else class="history-detail">
            <div class="history-detail__cards">
              <article v-for="card in historyDetailSummaryCards" :key="card.label">
                <small>{{ card.label }}</small>
                <strong>{{ card.value }}</strong>
              </article>
            </div>

            <section v-if="historyDetailCurvePoints.length" class="history-detail__chart">
              <div class="history-detail__chart-header">
                <h3>训练收敛曲线</h3>
                <div class="history-detail__legend" aria-label="曲线图例">
                  <span><i class="history-detail__legend-coverage"></i> 覆盖率 {{ formatPercent(historyDetailFinalCoverage) }}</span>
                  <span><i class="history-detail__legend-broadcast"></i> 广播率 {{ formatPercent(historyDetailFinalBroadcast) }}</span>
                </div>
              </div>
              <svg viewBox="0 0 640 180" role="img" aria-label="训练过程中覆盖率和广播率从低到高收敛曲线">
                <line x1="28" y1="24" x2="28" y2="152" class="history-detail__axis" />
                <line x1="28" y1="152" x2="616" y2="152" class="history-detail__axis" />
                <line x1="28" y1="88" x2="616" y2="88" class="history-detail__grid-line" />
                <line x1="28" y1="24" x2="616" y2="24" class="history-detail__grid-line" />
                <text x="6" y="29" class="history-detail__axis-label">100%</text>
                <text x="12" y="93" class="history-detail__axis-label">50%</text>
                <text x="15" y="157" class="history-detail__axis-label">0%</text>
                <polyline :points="historyDetailCoveragePolyline" class="history-detail__line history-detail__line--coverage" />
                <polyline :points="historyDetailBroadcastPolyline" class="history-detail__line history-detail__line--broadcast" />
              </svg>
            </section>

            <section v-if="historyDetailTestCards.length" class="history-detail__test">
              <h3>测试结果</h3>
              <div>
                <article v-for="card in historyDetailTestCards" :key="card.label">
                  <small>{{ card.label }}</small>
                  <strong>{{ card.value }}</strong>
                </article>
              </div>
            </section>

            <div class="history-detail__grid">
              <section class="history-detail__block">
                <h3>训练配置</h3>
                <dl>
                  <div>
                    <dt>场景</dt>
                    <dd>{{ trainingScenarioName(historyDetail || historyDetailRecord) }}</dd>
                  </div>
                  <div>
                    <dt>奖励模式</dt>
                    <dd>{{ historyDetail?.reward_mode || "--" }}</dd>
                  </div>
                  <div>
                    <dt>环境</dt>
                    <dd>{{ historyDetail?.env_type || historyDetailRecord?.env_type || "--" }}</dd>
                  </div>
                  <div>
                    <dt>评估协议</dt>
                    <dd>{{ historyDetail?.evaluation_protocol || historyDetailRecord?.evaluation_protocol || "--" }}</dd>
                  </div>
                  <div>
                    <dt>学习率</dt>
                    <dd>{{ historyDetailAlgorithmConfig.learning_rate ?? "--" }}</dd>
                  </div>
                  <div>
                    <dt>折扣因子</dt>
                    <dd>{{ historyDetailAlgorithmConfig.gamma ?? "--" }}</dd>
                  </div>
                  <div>
                    <dt>Rollout 步长</dt>
                    <dd>{{ historyDetailTrainConfig.rollout_steps ?? "--" }}</dd>
                  </div>
                  <div>
                    <dt>评估间隔</dt>
                    <dd>{{ historyDetailTrainConfig.eval_interval_steps ?? historyDetailTrainConfig.eval_interval ?? "--" }}</dd>
                  </div>
                </dl>
              </section>

              <section class="history-detail__block">
                <h3>产物信息</h3>
                <dl>
                  <div>
                    <dt>Checkpoint</dt>
                    <dd>{{ historyDetail?.checkpoint_path || historyDetailRecord?.checkpoint_path || "--" }}</dd>
                  </div>
                  <div>
                    <dt>运行目录</dt>
                    <dd>{{ historyDetail?.run_dir || historyDetailRecord?.run_dir || "--" }}</dd>
                  </div>
                  <div>
                    <dt>操作人</dt>
                    <dd>{{ historyDetail?.operator || historyDetailRecord?.operator || "系统" }}</dd>
                  </div>
                  <div>
                    <dt>更新时间</dt>
                    <dd>{{ formatTime(historyDetail?.updated_at || historyDetailRecord?.updated_at) }}</dd>
                  </div>
                </dl>
              </section>
            </div>

            <section class="history-detail__block history-detail__block--wide">
              <h3>评估结果</h3>
              <table>
                <thead>
                  <tr>
                    <th>序号</th>
                    <th>Step</th>
                    <th>平均奖励</th>
                    <th>平均覆盖率</th>
                    <th>平均广播覆盖</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-if="!historyDetailEvalRows.length">
                    <td colspan="5" class="empty-row">暂无评估记录</td>
                  </tr>
                  <tr v-for="(item, index) in historyDetailEvalRows" :key="`${item.step || index}-${index}`">
                    <td>{{ index + 1 }}</td>
                    <td>{{ formatInteger(item.step) }}</td>
                    <td>{{ formatMetric(item.avg_reward, 3) }}</td>
                    <td>{{ formatPercent(item.avg_coverage) }}</td>
                    <td>{{ formatPercent(item.avg_broadcast) }}</td>
                  </tr>
                </tbody>
              </table>
            </section>
          </div>
        </section>
      </div>
    </transition>
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import axios from "axios";
import StreamingTerminal from "./StreamingTerminal.vue";
import TrainingMonitor from "./TrainingMonitor.vue";
import { rescueApiBase } from "../utils/runtimeEndpoints";
import { saveReplaySessionFromSimulation, setActiveReplaySessionId } from "../utils/replaySessions";
import {
  appendSharedTerminalLine,
  clearTerminalOutput,
  exportTerminalOutput,
  terminalHistoryLines,
} from "../utils/terminalOutput";
import {
  compareDisasterScenarioOrder,
  formatDisasterType,
  formatPlainDisasterName,
  formatScenarioName,
  preferredDisasterSeverityKey,
} from "../utils/scenarioLabels";
import {
  buildUserNodeCountMessage,
  userNodeCountLogKey,
} from "../utils/scenarioNodeMetrics";

const API_BASE = rescueApiBase;
const SCENE_STORAGE_KEY = "prototype-training-scenes";
const SCENE_ACCESS_TIMEOUT_MS = 45000;
const DEVICE_SYNC_TIMEOUT_MS = SCENE_ACCESS_TIMEOUT_MS;
const DEFAULT_DEVICE_TEMPLATES = [
  { id: "cellular-macro", name: "蜂窝宏站", label: "蜂窝宏站", deviceType: "宏基站", modes: ["蜂窝通信"] },
  { id: "wifi-hotspot", name: "WiFi6 热点", label: "WiFi6 热点", deviceType: "背负式基站", modes: ["WiFi 通信"] },
  { id: "satellite-relay", name: "卫星中继", label: "卫星中继", deviceType: "中继设备", modes: ["卫星通信"] },
  { id: "shortwave-station", name: "短波台", label: "短波台", deviceType: "临时设备/车载设备", modes: ["短波通信"] },
];
const DEVICE_DISPLAY_TEXT_REPLACEMENTS = [
  ["5G 700MHz 应急小区", "5G 700MHz应急基站"],
  ["5G 700MHz应急小区", "5G 700MHz应急基站"],
  ["5G应急小区", "5G应急基站"],
];
const DISASTER_SEVERITY_LABELS = {
  level_1: "一般",
  level_2: "中等",
  level_3: "严重",
  level_4: "特别严重",
  level_1_general: "一般",
  level_2_moderate: "中等",
  level_3_severe: "严重",
  level_4_extreme: "特别严重",
};
const FALLBACK_SEVERITY_RATES = {
  level_1: { damageRate: 0.08, offlineRate: 0.02 },
  level_2: { damageRate: 0.22, offlineRate: 0.08 },
  level_3: { damageRate: 0.45, offlineRate: 0.22 },
  level_4: { damageRate: 0.68, offlineRate: 0.38 },
};
const EXTREME_DISASTER_USER_COUNTS = {
  extreme_rainstorm: 3500,
  super_typhoon: 3200,
  destructive_earthquake: 3900,
};
const FALLBACK_DISASTER_CATALOG = [
  {
    scenario: "extreme_rainstorm",
    display_name: "超强暴雨",
    disaster_type: "rainstorm",
    num_users: EXTREME_DISASTER_USER_COUNTS.extreme_rainstorm,
    grid_size: { rows: 10, cols: 12 },
  },
  {
    scenario: "super_typhoon",
    display_name: "特大台风",
    disaster_type: "typhoon",
    num_users: EXTREME_DISASTER_USER_COUNTS.super_typhoon,
    grid_size: { rows: 20, cols: 12 },
  },
  {
    scenario: "destructive_earthquake",
    display_name: "强破坏地震",
    disaster_type: "earthquake",
    num_users: EXTREME_DISASTER_USER_COUNTS.destructive_earthquake,
    grid_size: { rows: 5, cols: 12 },
  },
];
const FALLBACK_REWARD_MODES = [
  { value: "bandwidth_priority", label: "带宽优先", desc: "吞吐保障" },
  { value: "cost_priority", label: "设备开销最小优先", desc: "设备开销" },
  { value: "coverage_balance", label: "考虑覆盖", desc: "覆盖均衡" },
  { value: "coverage_priority", label: "覆盖优先", desc: "覆盖恢复" },
];
const DISASTER_STATION_TYPE_LABELS = {
  backpack_micro_cell: "背负式 5G 700MHz 微站",
  low_band_macro_cell: "低频宏基站",
  temporary_macro_cell: "临时宏基站",
  fixed_satellite_gateway: "固定 Ka 应急卫星网关",
  vehicle_satellite_terminal: "车载 Ka 卫星终端",
  command_vehicle_radio: "指挥车短波电台",
  field_shortwave_station: "野战短波台",
  portable_hotspot: "便携式 WiFi 6 热点",
  shelter_mesh_node: "避难所 WiFi 6 Mesh 节点",
  vehicle_wifi_node: "车载 WiFi 6 节点",
};

const assetUrl = (path) => `${import.meta.env.BASE_URL}prototype/${path}`;

const scenarios = ref([]);
const loadError = ref("");
const actionError = ref("");
const isStarting = ref(false);
const isSyncingDevices = ref(false);
const runStatus = ref("Idle");
const eventLog = ref([]);
const selectedRewardMode = ref(null);
const replayRunIdInFlight = ref(null);
const activeRunMeta = ref(null);
const activeParamTab = ref("algorithm");
const accessDevices = ref([]);
const selectedScenarioName = ref("");

// History panel state
const showHistoryPanel = ref(false);
const trainingHistory = ref([]);
const historyFilterAlgorithm = ref("");
const historyFilterScenario = ref("");
const historyPage = ref(1);
const historyPageSize = ref(10);
const showHistoryDetailModal = ref(false);
const historyDetailRecord = ref(null);
const historyDetail = ref(null);
const isLoadingHistoryDetail = ref(false);
const historyDetailError = ref("");

let eventSource = null;
let preserveSnapshotAccessDevices = false;
let lastScenarioUserNodeLogKey = "";

const trainingTerminalLines = computed(() => terminalHistoryLines.value.slice(-500));
const trainingTerminalStatus = computed(() => {
  const normalized = String(runStatus.value || "idle").toLowerCase();
  if (normalized === "idle") return "idle";
  if (normalized === "starting") return "starting";
  if (normalized === "running") return "running";
  if (normalized === "completed") return "completed";
  if (normalized === "stopped") return "stopped";
  if (normalized === "disconnected") return "disconnected";
  if (normalized === "failed" || normalized === "error") return "failed";
  return "idle";
});

const trainingTerminalMeta = (type = "info") => {
  const normalized = String(type || "info").toLowerCase();
  if (normalized === "ui_action") return { level: "ACTION", source: "TRAIN" };
  if (normalized === "backend") return { level: "BACKEND", source: "BACKEND" };
  if (normalized === "device_state_sync") return { level: "SYNC", source: "TRAIN" };
  if (normalized === "warn" || normalized === "warning") return { level: "WARN", source: "TRAIN" };
  if (normalized === "error" || normalized === "training_replay_error") return { level: "ERROR", source: "TRAIN" };
  if (normalized === "training_replay_ready") return { level: "REPLAY", source: "BACKEND" };
  return { level: "INFO", source: "TRAIN" };
};

const appendTrainingTerminalLine = (message, options = {}) => {
  if (!message) return;
  appendSharedTerminalLine(message, {
    level: options.level || "INFO",
    source: options.source || "TRAIN",
    timestamp: options.timestamp,
  });
};

const appendTrainingEvent = (message, type = "info", payload = {}) => {
  if (!message) return;
  appendTrainingTerminalLine(message, trainingTerminalMeta(type));
  eventLog.value = [
    ...eventLog.value.slice(-79),
    {
      type,
      timestamp: Date.now() / 1000,
      payload,
      message,
    },
  ];
};

const selectedScenarioDisplayLabel = () =>
  selectedScenario.value ? displayScenarioWithSeverity(selectedScenario.value) : selectedScenarioName.value || "未选择场景";

const appendScenarioUserNodeCount = (scenario) => {
  if (!scenario) return;
  const key = userNodeCountLogKey(`train:${scenario.name}`, scenario);
  if (key === lastScenarioUserNodeLogKey) return;
  lastScenarioUserNodeLogKey = key;
  appendTrainingTerminalLine(
    buildUserNodeCountMessage(`模型训练接入灾害场景：${displayScenarioWithSeverity(scenario)}`, scenario),
    { level: "SCENE", source: "TRAIN" }
  );
};

const downloadTerminalLog = () => {
  exportTerminalOutput(terminalHistoryLines.value, "rescuenet-training-terminal.log");
};

const clearTerminalLog = () => {
  clearTerminalOutput();
};

// Form state
const disasterType = ref("");
const disasterSeverity = ref("");
const selectedAlgorithm = ref("ppo");
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
  { value: "ppo", label: "基于 PPO 的覆盖恢复策略优化方案", desc: "覆盖优先 / 稳定基线" },
  { value: "dqn", label: "基于 DQN 的离散站点部署决策方案", desc: "离散动作 / 快速推演" },
  { value: "a3c", label: "基于 A3C 的多目标协同训练方案", desc: "异步更新 / 多目标" },
  { value: "mppo", label: "基于 MPPO 的多头策略组网方案", desc: "多头策略 / 资源协同" },
  { value: "hmarl", label: "层次化多智能体通信资源配置与组网方案", desc: "自研方案 / 分层协同" },
];

const paramTabs = [
  { key: "algorithm", label: "算法参数" },
  { key: "simulation", label: "仿真场景参数" },
];

const scenarioSourceKey = (scenario) => scenario?.source_scenario || String(scenario?.name || "").split("__")[0] || "";
const scenarioSeverityKey = (scenario) => scenario?.severity_level || String(scenario?.name || "").split("__")[1] || "";
const scenarioDisasterLabel = (scenario) => {
  const display = scenario?.display_name || "";
  const displayName = display.includes("/") ? display.split("/")[0].trim() : display;
  return formatPlainDisasterName(
    scenario?.disaster_type,
    scenarioSourceKey(scenario),
    displayName,
    scenario?.name
  ) || formatDisasterType(scenario?.disaster_type) || formatScenarioName(scenarioSourceKey(scenario));
};
const normalizeSeverityLabel = (value) => {
  const key = String(value || "").trim();
  if (!key) return "";
  if (DISASTER_SEVERITY_LABELS[key]) return DISASTER_SEVERITY_LABELS[key];
  const match = key.match(/^level_(\d+)/i);
  if (match) return `等级 ${match[1]}`;
  return key;
};
const severityLabel = (scenario) => {
  const key = scenarioSeverityKey(scenario);
  return DISASTER_SEVERITY_LABELS[key] || normalizeSeverityLabel(scenario?.severity_label) || formatScenarioName(key);
};
const displayScenarioWithSeverity = (scenario) => (scenario ? `${scenarioDisasterLabel(scenario)} / ${severityLabel(scenario)}` : "未加载");
const scenarioOrderValues = (scenario) => [
  scenarioDisasterLabel(scenario),
  scenario?.disaster_type,
  scenarioSourceKey(scenario),
  scenario?.display_name,
  scenario?.name,
];
const compareScenarioRecords = (left, right) =>
  compareDisasterScenarioOrder(scenarioOrderValues(left), scenarioOrderValues(right)) ||
  scenarioSeverityKey(left).localeCompare(scenarioSeverityKey(right), "zh-CN");

const normalizeOptionalRate = (value) => {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return null;
  return Math.max(0, Math.min(1, Math.abs(numeric) > 1 ? numeric / 100 : numeric));
};

const severityRank = (scenarioOrKey) => {
  const key = typeof scenarioOrKey === "string" ? scenarioOrKey : scenarioSeverityKey(scenarioOrKey);
  const matched = String(key || "").match(/level_(\d+)/i);
  return matched ? Number(matched[1]) : 1;
};

const scenarioDamageStats = (scenario) => {
  const profiles = Object.values(scenario?.mode_profiles || {});
  const aggregate = profiles.reduce(
    (acc, profile) => {
      acc.damaged += Number(profile?.damaged_station_count || 0);
      acc.offline += Number(profile?.offline_station_count || 0);
      acc.physical += Number(profile?.physical_station_count || 0);
      return acc;
    },
    { damaged: 0, offline: 0, physical: 0 }
  );
  return {
    damaged: aggregate.damaged,
    offline: aggregate.offline,
    physical: aggregate.physical,
  };
};

const rateFromCounts = (part, total) => {
  const numerator = Number(part);
  const denominator = Number(total);
  if (!Number.isFinite(numerator) || !Number.isFinite(denominator) || denominator <= 0) return null;
  return Math.max(0, Math.min(1, numerator / denominator));
};

const severityRatesFromScenario = (scenario) => {
  const stats = scenarioDamageStats(scenario);
  const key = scenarioSeverityKey(scenario);
  const fallback = FALLBACK_SEVERITY_RATES[key] || FALLBACK_SEVERITY_RATES[`level_${severityRank(scenario)}`] || {};
  const damageRate =
    normalizeOptionalRate(scenario?.damage_rate ?? scenario?.severity_meta?.damage_rate ?? scenario?.meta?.damage_rate) ??
    rateFromCounts(stats.damaged, stats.physical) ??
    fallback.damageRate ??
    null;
  const offlineRate =
    normalizeOptionalRate(scenario?.offline_rate ?? scenario?.severity_meta?.offline_rate ?? scenario?.meta?.offline_rate) ??
    rateFromCounts(stats.offline, stats.physical) ??
    fallback.offlineRate ??
    null;
  return { damageRate, offlineRate };
};

const severityRateDescription = (rates) => {
  const parts = [];
  if (rates.damageRate != null) parts.push(`损毁 ${formatPercent(rates.damageRate)}`);
  if (rates.offlineRate != null) parts.push(`离线 ${formatPercent(rates.offlineRate)}`);
  return parts.join(" / ") || "等待同步参数";
};

const severityOptionFromScenario = (scenario) => {
  const rates = severityRatesFromScenario(scenario);
  return {
    key: scenarioSeverityKey(scenario),
    label: severityLabel(scenario),
    description: severityRateDescription(rates),
    scenarioName: scenario.name,
    damageRate: rates.damageRate,
    offlineRate: rates.offlineRate,
    damageRateText: rates.damageRate == null ? "--" : formatPercent(rates.damageRate),
    offlineRateText: rates.offlineRate == null ? "--" : formatPercent(rates.offlineRate),
  };
};

const scenarioRecordsFromDisasterCatalog = (catalog) =>
  (Array.isArray(catalog) && catalog.length ? catalog : FALLBACK_DISASTER_CATALOG).flatMap((item) => {
    const source = item.scenario || item.name || item.disaster_scenario || "";
    const levels = item.severity_levels || ["level_1", "level_2", "level_3", "level_4"];
    const entries = Array.isArray(levels) ? levels.map((key) => [key, {}]) : Object.entries(levels || {});
    const grid = item.region_grid || item.grid_size || {};
    const rows = Number(grid.rows || item.rows || item.grid_size || 12);
    const cols = Number(grid.cols || item.cols || item.grid_size || rows);
    const numUsers = Number(item.num_users || item.unique_user_count || EXTREME_DISASTER_USER_COUNTS[source] || 0);
    return entries.map(([severityKey, meta]) => ({
      name: `${source}__${severityKey}`,
      display_name: `${item.display_name || item.label || source} / ${meta?.label || DISASTER_SEVERITY_LABELS[severityKey] || severityKey}`,
      source_scenario: source,
      severity_level: severityKey,
      severity_label: meta?.label || DISASTER_SEVERITY_LABELS[severityKey] || severityKey,
      severity_description: meta?.description,
      damage_rate: meta?.damage_rate,
      offline_rate: meta?.offline_rate,
      disaster_type: item.disaster_type || item.type || source,
      grid_size: Math.max(rows, cols),
      region_grid: { ...(typeof grid === "object" ? grid : {}), rows, cols },
      num_users: numUsers,
      candidate_sites: Number(item.candidate_sites || rows * cols),
      max_steps: Number(item.max_steps || 72),
      has_residual_network: Boolean(item.has_residual_network),
      reward_profiles: [],
      base_stations: [],
      base_station_deployments: [],
      residual_base_stations: [],
    }));
  });

const selectedScenario = computed(() => {
  if (!scenarios.value.length) return null;
  return scenarios.value.find((item) => item.name === selectedScenarioName.value) || scenarios.value[0];
});
const selectedAlgorithmLabel = computed(
  () => algorithmCards.find((item) => item.value === selectedAlgorithm.value)?.label || selectedAlgorithm.value.toUpperCase()
);
const evaluationProtocol = computed(() =>
  selectedScenario.value?.disaster_type === "earthquake" ? "earthquake_stress" : "standard"
);
const selectedScenarioLabel = computed(() =>
  selectedScenario.value ? displayScenarioWithSeverity(selectedScenario.value) : "未加载"
);
const visibleDeviceRowsForLayout = computed(() => Math.min(accessDevices.value.length, 6));
const pageHeight = computed(() => 1930 + Math.max(0, visibleDeviceRowsForLayout.value - 2) * 38);

const gridBounds = computed(() => {
  const rows = Math.max(1, Number(selectedScenario.value?.region_grid?.rows || selectedScenario.value?.grid_size || 24));
  const cols = Math.max(1, Number(selectedScenario.value?.region_grid?.cols || selectedScenario.value?.grid_size || 24));
  return {
    rows,
    cols,
    maxX: Math.max(0, rows - 1),
    maxY: Math.max(0, cols - 1),
  };
});

const disasterScenarioOptions = computed(() => {
  const byScenario = new Map();
  scenarios.value.forEach((scenario) => {
    const key = scenarioSourceKey(scenario);
    if (!key || byScenario.has(key)) return;
    const count = scenarios.value.filter((item) => scenarioSourceKey(item) === key).length;
    byScenario.set(key, {
      key,
      label: scenarioDisasterLabel(scenario),
      description: `${count} 个等级 / ${scenarioDisasterLabel(scenario)}`,
    });
  });
  return [...byScenario.values()].sort((left, right) =>
    compareDisasterScenarioOrder([left.label, left.key], [right.label, right.key])
  );
});

const severityScenariosForSelectedType = computed(() =>
  scenarios.value
    .filter((scenario) => scenarioSourceKey(scenario) === disasterType.value)
    .slice()
    .sort((left, right) => severityRank(left) - severityRank(right))
);

const disasterSeverityOptions = computed(() =>
  severityScenariosForSelectedType.value.map((scenario) => severityOptionFromScenario(scenario))
);

const selectedSeverityOption = computed(() =>
  disasterSeverityOptions.value.find((item) => item.key === disasterSeverity.value) || disasterSeverityOptions.value[0] || null
);

const selectedSeveritySummary = computed(() => selectedSeverityOption.value?.description || "");

const selectedSeverityInsight = computed(() => {
  const scenario = selectedScenario.value;
  const option = selectedSeverityOption.value;
  if (!scenario || !option) return [];
  return [
    {
      label: "损毁率",
      value: option.damageRateText,
      hint: "基站或回传受损比例",
    },
    {
      label: "离线率",
      value: option.offlineRateText,
      hint: "完全不可用站点比例",
    },
  ];
});

const accessDeviceCount = computed(() =>
  accessDevices.value.reduce((sum, row) => sum + Math.max(1, Number(row.count || 1)), 0)
);

const scenarioDisasterDescription = computed(() => {
  const scenario = selectedScenario.value;
  if (!scenario) return "当前暂无可用灾害场景，请等待后端场景数据加载完成。";
  const rows = Number(scenario.region_grid?.rows || scenario.grid_size || gridBounds.value.rows || 0);
  const cols = Number(scenario.region_grid?.cols || scenario.grid_size || gridBounds.value.cols || 0);
  const users = Number(scenario.num_users || 0).toLocaleString("zh-CN");
  const candidateSites = Number(scenario.candidate_sites || 0).toLocaleString("zh-CN");
  const deviceCount = Number(accessDeviceCount.value || 0);
  const devices = deviceCount.toLocaleString("zh-CN");
  const maxSteps = Number(scenario.max_steps || 0).toLocaleString("zh-CN");
  const deviceText = deviceCount
    ? `系统已接入 ${devices} 台来自真实场景的应急通信设备，用于训练模型在受损网络条件下完成覆盖恢复、广播保障和资源调度决策`
    : "系统当前按无残余网络训练，不预置任何残余基站或应急通信设备";
  return `当前选择${scenarioDisasterLabel(scenario)}灾害的${severityLabel(scenario)}等级场景，受灾区域被离散为 ${rows} x ${cols} 个通信恢复网格，包含约 ${users} 个受灾终端和 ${candidateSites} 个可部署候选站点；${deviceText}，单轮环境最大推演步长为 ${maxSteps}。`;
});

const scenarioStats = computed(() => {
  const scenario = selectedScenario.value;
  return [
    {
      label: "用户规模",
      value: Number(scenario?.num_users || 0).toLocaleString("zh-CN"),
      hint: "受灾终端数量",
    },
    {
      label: "候选站点",
      value: Number(scenario?.candidate_sites || 0).toLocaleString("zh-CN"),
      hint: "可部署位置",
    },
    {
      label: "训练步长",
      value: Number(totalTimesteps.value || 0).toLocaleString("zh-CN"),
      hint: `最大步长 ${Number(scenario?.max_steps || 0).toLocaleString("zh-CN")}`,
    },
    {
      label: "接入设备",
      value: String(accessDeviceCount.value),
      hint: deviceAccessSummary.value,
    },
  ];
});

const deviceProfiles = computed(() => {
  const stations = Array.isArray(selectedScenario.value?.base_stations) ? selectedScenario.value.base_stations : [];
  return stations.length ? stations : DEFAULT_DEVICE_TEMPLATES;
});

const deviceProfileValue = (profile) =>
  String(profile?.id || profile?.name || profile?.label || profile?.base_station || profile?.station_type || "");

const deviceProfileLabel = (profile) =>
  displayDeviceText(profile?.label || profile?.name || profile?.device_name || profile?.base_station || profile?.station_type || "应急设备");

const displayText = (...values) =>
  displayDeviceText(values
    .map((value) => String(value || "").trim())
    .find(Boolean) || "");

const displayDeviceText = (value) => {
  if (value === null || value === undefined) return "";
  return DEVICE_DISPLAY_TEXT_REPLACEMENTS.reduce(
    (text, [source, target]) => text.replaceAll(source, target),
    String(value)
  );
};

const disasterStationTypeLabel = (stationType, fallback) =>
  DISASTER_STATION_TYPE_LABELS[String(stationType || "")] || fallback || stationType || "场景基站";

const deviceProfileModes = (profile) => {
  const raw =
    profile?.modes ||
    profile?.supported_modes ||
    profile?.communication_modes ||
    profile?.comm_modes ||
    profile?.mode ||
    profile?.communication_type;
  if (Array.isArray(raw)) return raw.filter(Boolean).map(String);
  if (raw) return [String(raw)];
  const text = `${profile?.name || ""} ${profile?.label || ""}`.toLowerCase();
  if (text.includes("wifi")) return ["WiFi 通信"];
  if (text.includes("satellite") || text.includes("卫星")) return ["卫星通信"];
  if (text.includes("shortwave") || text.includes("短波")) return ["短波通信"];
  return ["蜂窝通信"];
};

const deviceModesLabel = (profile) => deviceProfileModes(profile).join(" / ");
const accessDeviceDisplayName = (device) =>
  displayText(device?.deviceName, device?.device_name) ||
  disasterStationTypeLabel(
    device?.stationType || device?.station_type,
    displayText(device?.stationLabel, device?.station_label, device?.label, device?.device)
  ) ||
  "应急设备";

const deviceProfileByValue = (value) =>
  deviceProfiles.value.find((profile) => deviceProfileValue(profile) === value) || deviceProfiles.value[0] || null;

const scenarioBaseStationDeployments = () =>
  Array.isArray(selectedScenario.value?.base_station_deployments)
    ? selectedScenario.value.base_station_deployments
    : Array.isArray(selectedScenario.value?.residual_base_stations)
      ? selectedScenario.value.residual_base_stations
      : [];

const defaultStationMode = (profile) => {
  if (!profile) return null;
  return deviceProfileModes(profile)[0] || null;
};

const stationStatusLabel = (status) => {
  const labels = {
    active: "在线",
    degraded: "降级",
    offline: "离线",
    planned: "待部署",
    residual: "残余可用",
    deployed: "已部署",
  };
  return labels[String(status || "").toLowerCase()] || "已导入";
};

const deviceModesForValue = (value, mode = null) => {
  if (mode) return String(mode);
  const profile = deviceProfileByValue(value);
  return profile ? deviceModesLabel(profile) : "--";
};

const clampGridCoord = (value, maxValue) => {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 0;
  return Math.max(0, Math.min(Math.max(0, Number(maxValue) || 0), Math.round(numeric)));
};

const accessDeviceFromDeployment = (deployment, index) => {
  const baseStationName = deployment.base_station || deployment.baseStationName || "";
  const profile = deviceProfiles.value.find((item) => deviceProfileValue(item) === baseStationName) || deviceProfileByValue(baseStationName);
  const mode = deployment.mode || defaultStationMode(profile);
  const x = clampGridCoord(deployment.x, gridBounds.value.maxX);
  const y = clampGridCoord(deployment.y, gridBounds.value.maxY);
  const deploymentUid = deployment.device_uid || deployment.id || deployment.deployment_id || null;
  const sourceStationName = displayText(deployment.station_label, deployment.label);
  const rawDeviceName = displayText(deployment.device_name);
  const deviceName = rawDeviceName && rawDeviceName !== sourceStationName ? rawDeviceName : "";
  const stationLabel = disasterStationTypeLabel(
    deployment.station_type,
    displayText(deployment.label, deviceProfileLabel(profile), baseStationName) || "场景基站"
  );

  return {
    id: deploymentUid || `scenario-deployment:${selectedScenarioName.value}:${index}:${baseStationName}:${mode || "mode"}:${x}:${y}`,
    deploymentId: deployment.deployment_id || deploymentUid,
    device: baseStationName || deviceProfileValue(profile),
    mode,
    count: Math.max(1, Number(deployment.quantity || 1)),
    x,
    y,
    status: deployment.status || "active",
    statusLabel: deployment.statusLabel || stationStatusLabel(deployment.status),
    stationType: deployment.station_type || baseStationName,
    deviceName,
    stationLabel,
    maxUsers: Number(deployment.max_users ?? deployment.cell_user_count ?? profile?.max_users ?? 0),
    maxThroughput: Number(deployment.max_throughput ?? deployment.downlink_bandwidth_mbps ?? profile?.max_throughput ?? 0),
    coverageRadiusKm: Number(deployment.coverage_radius_km ?? deployment.source_coverage_radius_km ?? profile?.coverage_radius_km ?? 0),
  };
};

const accessDeviceRowsFromDeployments = (deployments) =>
  (Array.isArray(deployments) ? deployments : [])
    .map(accessDeviceFromDeployment)
    .filter((row) => row.device);

const initializeAccessDevicesFromScenario = () => {
  accessDevices.value = accessDeviceRowsFromDeployments(scenarioBaseStationDeployments());
};

const deviceAccessSummary = computed(() => {
  const total = accessDevices.value.reduce((sum, row) => sum + Number(row.count || 0), 0);
  return total ? `${total} 台设备已接入训练配置` : "无残余网络：未接入任何设备";
});

const fallbackRewardModeLabel = (key) => FALLBACK_REWARD_MODES.find((item) => item.value === key)?.label || key || "默认";

const rewardModeCards = computed(() => {
  const profiles = Array.isArray(selectedScenario.value?.reward_profiles) ? selectedScenario.value.reward_profiles : [];
  if (!profiles.length) return FALLBACK_REWARD_MODES;
  return profiles.map((profile) => ({
    value: profile.key || profile.value,
    label: profile.label || fallbackRewardModeLabel(profile.key || profile.value),
    desc:
      profile.description ||
      `覆盖 ${formatWeight(profile.coverage_weight)} / 带宽 ${formatWeight(profile.bandwidth_weight)}`,
  }));
});

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
    items = items.filter((r) => trainingScenarioName(r) === historyFilterScenario.value);
  }
  return items;
});

const totalHistoryPages = computed(() => Math.max(1, Math.ceil(filteredHistory.value.length / historyPageSize.value)));

const paginatedHistory = computed(() => {
  const start = (historyPage.value - 1) * historyPageSize.value;
  return filteredHistory.value.slice(start, start + historyPageSize.value);
});

const formatWeight = (value) => Number(value ?? 0).toFixed(2);

const formatMetric = (value, digits = 2) => {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "--";
  return numeric.toFixed(digits);
};

const formatPercent = (value) => {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "--";
  const percent = Math.abs(numeric) <= 1 ? numeric * 100 : numeric;
  return `${percent.toFixed(2)}%`;
};

const normalizeHistoryRate = (value) => {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 0;
  const rate = Math.abs(numeric) <= 1 ? numeric : numeric / 100;
  return Math.max(0, Math.min(1, rate));
};

const formatInteger = (value) => {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "--";
  return Math.round(numeric).toLocaleString("zh-CN");
};

const trainingScenarioName = (record) =>
  formatPlainDisasterName(record?.disaster_type, record?.scenario_name) || formatScenarioName(record?.scenario_name);

const trainingScenarioTypeName = (record) =>
  formatPlainDisasterName(record?.disaster_type, record?.scenario_name) || formatDisasterType(record?.disaster_type);

const historyDetailTitle = computed(() => `${trainingScenarioName(historyDetail.value || historyDetailRecord.value)}训练结果`);

const historyDetailTrainConfig = computed(() => historyDetail.value?.config?.train || {});
const historyDetailAlgorithmConfig = computed(() => historyDetail.value?.config?.algorithm || {});

const historyDetailSummaryCards = computed(() => {
  const detail = historyDetail.value || {};
  return [
    { label: "训练轮次", value: formatInteger(detail.episode_count) },
    { label: "总步数", value: formatInteger(detail.total_timesteps) },
    { label: "最佳奖励", value: formatMetric(detail.best_reward, 3) },
    { label: "最终奖励", value: formatMetric(detail.last_reward, 3) },
    { label: "最佳覆盖率", value: formatPercent(detail.best_coverage) },
    { label: "最终广播覆盖", value: formatPercent(detail.last_broadcast) },
  ];
});

const historyDetailCurveRows = computed(() => {
  const detail = historyDetail.value || {};
  const curveRows = Array.isArray(detail.curve_history) ? detail.curve_history : [];
  const evalRows = Array.isArray(detail.eval_history) ? detail.eval_history : [];
  const rows = curveRows.length ? curveRows : evalRows;
  return rows
    .map((item) => ({
      step: Number(item?.step ?? item?.global_step),
      avg_reward: item?.avg_reward ?? item?.reward,
      avg_coverage: item?.avg_coverage ?? item?.coverage,
      avg_broadcast: item?.avg_broadcast ?? item?.broadcast,
    }))
    .filter((item) => Number.isFinite(item.step));
});

const historyDetailEvalRows = computed(() => {
  return historyDetailCurveRows.value.slice(-10);
});

const historyDetailCurvePoints = computed(() => {
  return historyDetailCurveRows.value.map((item) => ({
    step: item.step,
    coverage: normalizeHistoryRate(item.avg_coverage),
    broadcast: normalizeHistoryRate(item.avg_broadcast),
  }));
});

const buildRatePolyline = (points, key) => {
  const width = 640;
  const height = 180;
  const paddingX = 28;
  const paddingTop = 24;
  const paddingBottom = 28;
  const plotWidth = width - paddingX - 24;
  const plotHeight = height - paddingTop - paddingBottom;
  if (!points.length) return "";
  return points
    .map((point, index) => {
      const x = points.length === 1 ? paddingX : paddingX + (index / (points.length - 1)) * plotWidth;
      const y = paddingTop + (1 - normalizeHistoryRate(point[key])) * plotHeight;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
};

const historyDetailCoveragePolyline = computed(() => buildRatePolyline(historyDetailCurvePoints.value, "coverage"));
const historyDetailBroadcastPolyline = computed(() => buildRatePolyline(historyDetailCurvePoints.value, "broadcast"));
const historyDetailFinalCoverage = computed(() =>
  historyDetailCurvePoints.value.at(-1)?.coverage ?? historyDetail.value?.last_coverage
);
const historyDetailFinalBroadcast = computed(() =>
  historyDetailCurvePoints.value.at(-1)?.broadcast ?? historyDetail.value?.last_broadcast
);

const historyDetailTestCards = computed(() => {
  const test = historyDetail.value?.test_results;
  if (!test) return [];
  return [
    { label: "测试覆盖率", value: formatPercent(test.coverage_rate) },
    { label: "测试广播率", value: formatPercent(test.broadcast_rate) },
    { label: "平均奖励", value: formatMetric(test.avg_reward, 3) },
    { label: "测试轮数", value: formatInteger(test.episodes) },
  ];
});

const rewardModeLabel = (key) => {
  const matched = rewardModeCards.value.find((item) => item.value === key) || FALLBACK_REWARD_MODES.find((item) => item.value === key);
  return matched?.label || key || "默认";
};

const formatTime = (ts) => {
  if (!ts) return "--";
  const numeric = Number(ts);
  const date = Number.isFinite(numeric)
    ? new Date(numeric < 1e12 ? numeric * 1000 : numeric)
    : new Date(ts);
  if (Number.isNaN(date.getTime())) return "--";
  return date.toLocaleString("zh-CN", { hour12: false });
};

const statusLabel = (status) => {
  const map = {
    Idle: "待启动",
    starting: "启动中",
    running: "运行中",
    completed: "已完成",
    failed: "失败",
    stopped: "已停止",
    disconnected: "连接中断",
    error: "失败",
  };
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

const toggleHistoryPanel = () => {
  showHistoryPanel.value = !showHistoryPanel.value;
  if (showHistoryPanel.value) {
    historyPage.value = 1;
    fetchTrainingHistory();
  }
};

const syncScenarioSelectors = () => {
  const scenario = selectedScenario.value;
  if (!scenario) return;
  disasterType.value = scenarioSourceKey(scenario);
  disasterSeverity.value = scenarioSeverityKey(scenario);
};

const preferredScenarioForSource = (sourceKey, fallbackSeverityKey = "") => {
  const candidates = scenarios.value.filter((scenario) => scenarioSourceKey(scenario) === sourceKey);
  if (!candidates.length) return null;
  if (fallbackSeverityKey) {
    const fallback = candidates.find((scenario) => scenarioSeverityKey(scenario) === fallbackSeverityKey);
    if (fallback) return fallback;
  }
  const severityKey = preferredDisasterSeverityKey(
    candidates.map((scenario) => ({ key: scenarioSeverityKey(scenario), label: severityLabel(scenario) }))
  );
  return candidates.find((scenario) => scenarioSeverityKey(scenario) === severityKey) || candidates[0];
};

const selectScenarioByParts = (sourceKey, severityKey = "") => {
  const next = preferredScenarioForSource(sourceKey, severityKey);
  if (!next) return;
  if (next?.name && next.name !== selectedScenarioName.value) {
    selectedScenarioName.value = next.name;
  } else {
    syncScenarioSelectors();
  }
};

const selectDisasterType = (sourceKey) => {
  const option = disasterScenarioOptions.value.find((item) => item.key === sourceKey);
  appendTrainingEvent(`切换灾害场景：${option?.label || formatScenarioName(sourceKey)}。`, "ui_action", {
    source_scenario: sourceKey,
  });
  selectScenarioByParts(sourceKey);
};

const selectSeverity = (severityKey) => {
  const option = disasterSeverityOptions.value.find((item) => item.key === severityKey);
  appendTrainingEvent(`切换受灾等级：${option?.label || severityKey}。`, "ui_action", {
    source_scenario: disasterType.value,
    severity_level: severityKey,
  });
  selectScenarioByParts(disasterType.value, severityKey);
};

const cloneAccessDevices = () => accessDevices.value.map((row) => ({ ...row }));

const updateSelectedScenarioDeployments = (baseStations) => {
  const index = scenarios.value.findIndex((scenario) => scenario.name === selectedScenarioName.value);
  if (index < 0) return;
  scenarios.value.splice(index, 1, {
    ...scenarios.value[index],
    base_station_deployments: baseStations,
    residual_base_stations: baseStations,
  });
};

const persistAccessDevices = async (previousRows = null, options = {}) => {
  if (!selectedScenarioName.value) return false;
  isSyncingDevices.value = true;
  actionError.value = "";
  const operation = options.operation || "replace_base_stations";
  try {
    const baseStations = buildTrainingBaseStations();
    const endpoint = options.operation
      ? `${API_BASE}/scenarios/${encodeURIComponent(selectedScenarioName.value)}/device-state`
      : `${API_BASE}/scenarios/${encodeURIComponent(selectedScenarioName.value)}/base-stations`;
    const payload = options.operation
      ? { base_stations: baseStations, operation: options.operation }
      : { base_stations: baseStations };
    appendTrainingEvent(
      `开始同步场景设备：scenario=${selectedScenarioName.value} operation=${operation} devices=${baseStations.length} timeout=${DEVICE_SYNC_TIMEOUT_MS}ms。`,
      "device_state_sync",
      { operation, device_count: baseStations.length, scenario_name: selectedScenarioName.value }
    );
    const { data } = await axios.put(endpoint, payload, { timeout: DEVICE_SYNC_TIMEOUT_MS });
    const persistedStations = Array.isArray(data?.base_stations)
      ? data.base_stations
      : Array.isArray(data?.devices)
        ? data.devices
        : baseStations;
    updateSelectedScenarioDeployments(persistedStations);
    accessDevices.value = accessDeviceRowsFromDeployments(persistedStations);
    appendTrainingEvent(
      persistedStations.length
        ? `场景设备已同步到后端数据库：${persistedStations.length} 台。`
        : "场景设备已清空，当前训练配置为无残余网络。",
      "device_state_sync",
      { operation, device_count: persistedStations.length, scenario_name: selectedScenarioName.value }
    );
    return true;
  } catch (error) {
    if (previousRows) accessDevices.value = previousRows;
    const message = error?.response?.data?.detail || error?.message || String(error);
    actionError.value = `场景设备同步失败: ${message}`;
    appendTrainingEvent(actionError.value, "error", {
      operation,
      scenario_name: selectedScenarioName.value,
    });
    return false;
  } finally {
    isSyncingDevices.value = false;
  }
};

const addAccessDevice = async () => {
  const profile = deviceProfiles.value[accessDevices.value.length % Math.max(1, deviceProfiles.value.length)];
  if (!profile) return;
  const previousRows = cloneAccessDevices();
  const mode = defaultStationMode(profile);
  appendTrainingEvent(`添加训练接入设备：${deviceProfileLabel(profile)}，模式=${mode || "--"}。`, "ui_action", {
    scenario_name: selectedScenarioName.value,
    device: deviceProfileValue(profile),
  });
  accessDevices.value = [
    {
      id: `training-access:${selectedScenarioName.value}:${Date.now()}`,
      device: deviceProfileValue(profile),
      mode,
      count: 1,
      x: Math.min(gridBounds.value.maxX, 2 + accessDevices.value.length * 2),
      y: Math.min(gridBounds.value.maxY, 2 + accessDevices.value.length * 2),
      status: "active",
      statusLabel: "已接入",
      stationType: deviceProfileValue(profile),
      deviceName: "",
      stationLabel: deviceProfileLabel(profile),
      maxUsers: Number(profile.max_users || 0),
      maxThroughput: Number(profile.max_throughput || 0),
      coverageRadiusKm: Number(profile.coverage_radius_km || 0),
    },
    ...accessDevices.value,
  ];
  await persistAccessDevices(previousRows);
};

const removeAccessDevice = async (index) => {
  if (index < 0 || index >= accessDevices.value.length) return;
  const previousRows = cloneAccessDevices();
  const removed = accessDevices.value[index];
  appendTrainingEvent(`移除训练接入设备：${accessDeviceDisplayName(removed) || `第 ${index + 1} 行`}。`, "ui_action", {
    scenario_name: selectedScenarioName.value,
    device: removed?.device,
  });
  accessDevices.value = accessDevices.value.filter((_, rowIndex) => rowIndex !== index);
  await persistAccessDevices(previousRows, accessDevices.value.length ? {} : { operation: "clear_residual_network" });
};

const clearTrainingResidualNetwork = async () => {
  if (!selectedScenarioName.value || isSyncingDevices.value) return;
  if (!accessDevices.value.length) {
    appendTrainingEvent("当前训练配置已经是无残余网络。", "info");
    return;
  }
  if (!window.confirm("确认删除当前训练场景的所有残余基站并切换为无残余网络？可在设备管理中恢复原始场景基站。")) return;
  const previousRows = cloneAccessDevices();
  appendTrainingEvent(`清空训练残余网络：${selectedScenarioDisplayLabel()}。`, "ui_action", {
    scenario_name: selectedScenarioName.value,
    device_count: accessDevices.value.length,
  });
  accessDevices.value = [];
  await persistAccessDevices(previousRows, { operation: "clear_residual_network" });
};

const syncAccessDevice = async (index, options = {}) => {
  const row = accessDevices.value[index];
  if (!row) return;
  const profile = deviceProfileByValue(row.device);
  if (options.refreshProfile) {
    row.mode = defaultStationMode(profile);
    row.stationType = deviceProfileValue(profile);
    row.deviceName = "";
    row.stationLabel = deviceProfileLabel(profile);
    row.maxUsers = Number(profile?.max_users || 0);
    row.maxThroughput = Number(profile?.max_throughput || 0);
    row.coverageRadiusKm = Number(profile?.coverage_radius_km || 0);
  } else {
    row.mode = row.mode || defaultStationMode(profile);
    row.stationType = row.stationType || deviceProfileValue(profile);
    row.deviceName = row.deviceName || "";
    row.stationLabel = row.stationLabel || deviceProfileLabel(profile);
    row.maxUsers = Number(row.maxUsers || profile?.max_users || 0);
    row.maxThroughput = Number(row.maxThroughput || profile?.max_throughput || 0);
    row.coverageRadiusKm = Number(row.coverageRadiusKm || profile?.coverage_radius_km || 0);
  }
  row.count = Math.max(1, Number(row.count || 1));
  row.x = clampGridCoord(row.x, gridBounds.value.maxX);
  row.y = clampGridCoord(row.y, gridBounds.value.maxY);
  row.status = row.status || "active";
  row.statusLabel = row.statusLabel || "已接入";
  if (options.persist) {
    appendTrainingEvent(
      `更新训练接入设备：${accessDeviceDisplayName(row)} count=${row.count} grid=(${row.x}, ${row.y})。`,
      "ui_action",
      { scenario_name: selectedScenarioName.value, device: row.device, count: row.count, x: row.x, y: row.y }
    );
    await persistAccessDevices();
  }
};

const buildTrainingBaseStations = (rows = accessDevices.value) =>
  rows.flatMap((row) =>
    Array.from({ length: Math.max(1, Number(row.count || 1)) }, (_, index) => {
      const profile = deviceProfileByValue(row.device);
      const x = (clampGridCoord(row.x, gridBounds.value.maxX) + index) % Math.max(1, gridBounds.value.rows);
      const y = (clampGridCoord(row.y, gridBounds.value.maxY) + index) % Math.max(1, gridBounds.value.cols);
      const deviceUid = index === 0 ? row.id : `${row.id}:copy:${index + 1}`;
      return row.device
        ? {
            device_uid: deviceUid || null,
            deployment_id: row.deploymentId || null,
            base_station: row.device,
            mode: row.mode || defaultStationMode(profile),
            x,
            y,
            status: row.status || "active",
            device_name: displayText(row.deviceName) || null,
            station_type: row.stationType || row.device || null,
            station_label: row.stationLabel || deviceProfileLabel(profile) || null,
            cell_user_count: Number(row.maxUsers || profile?.max_users || 0),
            coverage_radius_km: Number(row.coverageRadiusKm || profile?.coverage_radius_km || 0),
            max_throughput: Number(row.maxThroughput || profile?.max_throughput || 0),
            downlink_bandwidth_mbps: Number(row.maxThroughput || profile?.max_throughput || 0),
            max_users: Number(row.maxUsers || profile?.max_users || 0),
          }
        : null;
    }).filter(Boolean)
  );

const sceneSnapshot = () => ({
  disasterType: disasterType.value,
  disasterSeverity: disasterSeverity.value,
  selectedScenarioName: selectedScenarioName.value,
  selectedAlgorithm: selectedAlgorithm.value,
  selectedRewardMode: selectedRewardMode.value,
  accessDevices: accessDevices.value,
});

const saveSceneSnapshot = () => {
  const items = JSON.parse(localStorage.getItem(SCENE_STORAGE_KEY) || "[]");
  const next = [{ ...sceneSnapshot(), savedAt: Date.now() }, ...items].slice(0, 10);
  localStorage.setItem(SCENE_STORAGE_KEY, JSON.stringify(next));
  appendTrainingEvent("当前训练场景已保存到浏览器本地记录。", "ui_action");
};

const loadSceneSnapshot = () => {
  const [snapshot] = JSON.parse(localStorage.getItem(SCENE_STORAGE_KEY) || "[]");
  if (!snapshot) {
    appendTrainingEvent("暂无可导入的本地训练场景记录。", "warn");
    return;
  }
  const hasSnapshotAccessDevices = Array.isArray(snapshot.accessDevices);
  const previousScenarioName = selectedScenarioName.value;
  if (snapshot.selectedScenarioName && scenarios.value.some((scenario) => scenario.name === snapshot.selectedScenarioName)) {
    selectedScenarioName.value = snapshot.selectedScenarioName;
  } else if (snapshot.disasterType) {
    selectScenarioByParts(snapshot.disasterType, snapshot.disasterSeverity);
  }
  preserveSnapshotAccessDevices = hasSnapshotAccessDevices && selectedScenarioName.value !== previousScenarioName;
  selectedAlgorithm.value = snapshot.selectedAlgorithm || selectedAlgorithm.value;
  selectedRewardMode.value = snapshot.selectedRewardMode || selectedRewardMode.value;
  accessDevices.value = hasSnapshotAccessDevices ? snapshot.accessDevices : [];
  appendTrainingEvent(`已导入本地训练场景快照：${snapshot.selectedScenarioName || snapshot.disasterType || "未命名场景"}。`, "ui_action");
};

// --- API calls ---

const fetchScenarios = async () => {
  loadError.value = "";
  appendTrainingTerminalLine("前端操作：加载训练场景列表。", { level: "ACTION" });
  try {
    const { data } = await axios.get(`${API_BASE}/scenarios`, { timeout: SCENE_ACCESS_TIMEOUT_MS });
    scenarios.value = (Array.isArray(data?.scenarios) ? data.scenarios : []).slice().sort(compareScenarioRecords);
    if (!scenarios.value.length) {
      const catalogResponse = await axios.get(`${API_BASE}/disaster-scenarios`, { timeout: SCENE_ACCESS_TIMEOUT_MS });
      scenarios.value = scenarioRecordsFromDisasterCatalog(catalogResponse.data?.scenarios).sort(compareScenarioRecords);
    }
    if (scenarios.value.length && !scenarios.value.some((item) => item.name === selectedScenarioName.value)) {
      selectedScenarioName.value = preferredScenarioForSource(scenarioSourceKey(scenarios.value[0]))?.name || scenarios.value[0].name;
    }
    syncScenarioSelectors();
    appendTrainingEvent(`训练场景列表已加载：${scenarios.value.length} 个场景。`, "backend");
  } catch (error) {
    console.error("Failed to load scenarios", error);
    scenarios.value = scenarioRecordsFromDisasterCatalog([]).sort(compareScenarioRecords);
    selectedScenarioName.value = preferredScenarioForSource(scenarioSourceKey(scenarios.value[0]))?.name || scenarios.value[0]?.name || "";
    syncScenarioSelectors();
    loadError.value = `后端场景接口暂不可用，已启用本地灾害场景兜底: ${error?.message || "未知错误"}`;
    appendTrainingEvent(loadError.value, "warn");
  }
};

const fetchTrainingHistory = async () => {
  appendTrainingTerminalLine("前端操作：查询训练历史记录。", { level: "ACTION" });
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
      env_type: a.env_type,
      reward_mode: a.reward_mode,
      evaluation_protocol: a.evaluation_protocol,
      run_dir: a.run_dir,
    }));
    appendTrainingTerminalLine(`后端响应：训练历史记录 ${trainingHistory.value.length} 条。`, {
      level: "BACKEND",
      source: "BACKEND",
    });
  } catch (error) {
    console.warn("Failed to load training history", error);
    appendTrainingTerminalLine(`后端响应：训练历史记录读取失败：${error?.response?.data?.detail || error?.message || error}`, {
      level: "ERROR",
      source: "BACKEND",
    });
  }
};

const closeHistoryDetailModal = () => {
  showHistoryDetailModal.value = false;
  historyDetailError.value = "";
};

const viewHistoryDetail = async (record) => {
  historyDetailRecord.value = record;
  historyDetail.value = null;
  historyDetailError.value = "";
  showHistoryDetailModal.value = true;
  appendTrainingTerminalLine(`前端操作：查看训练历史详情 run_dir=${record?.run_dir || "--"}。`, { level: "ACTION" });

  if (!record?.run_dir) {
    historyDetailError.value = "当前训练记录缺少运行目录，无法加载训练结果详情。";
    appendTrainingTerminalLine("训练历史详情读取被拦截：记录缺少运行目录。", { level: "WARN" });
    return;
  }

  isLoadingHistoryDetail.value = true;
  try {
    const { data } = await axios.get(`${API_BASE}/train/artifacts/detail`, {
      params: { run_dir: record.run_dir },
      timeout: 10000,
    });
    historyDetail.value = data || null;
    appendTrainingTerminalLine("后端响应：训练历史详情已加载。", { level: "BACKEND", source: "BACKEND" });
  } catch (error) {
    console.warn("Failed to load training history detail", error);
    const message = error?.response?.data?.detail || error?.message || "未知错误";
    historyDetailError.value = `训练结果加载失败：${message}`;
    appendTrainingTerminalLine(`后端响应：训练历史详情加载失败：${message}`, { level: "ERROR", source: "BACKEND" });
  } finally {
    isLoadingHistoryDetail.value = false;
  }
};

const deleteHistoryRecord = async (record) => {
  try {
    trainingHistory.value = trainingHistory.value.filter((r) => r.id !== record.id);
  } catch (error) {
    console.warn("Failed to delete history record", error);
  }
};

const clampNumber = (value, min, max, fallback) => {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return fallback;
  return Math.max(min, Math.min(max, numeric));
};

const clampInteger = (value, min, max, fallback) => Math.trunc(clampNumber(value, min, max, fallback));

const selectParamTab = (key) => {
  activeParamTab.value = key;
  const label = paramTabs.find((tab) => tab.key === key)?.label || key;
  appendTrainingEvent(`切换参数页签：${label}。`, "ui_action", { tab: key });
};

const selectRewardMode = (option) => {
  if (!option?.value) return;
  selectedRewardMode.value = option.value;
  appendTrainingEvent(`选择奖励配置：${option.label || option.value}。`, "ui_action", {
    reward_mode: option.value,
  });
};

const applyTrainingParameters = () => {
  totalTimesteps.value = clampInteger(totalTimesteps.value, 1000, 5000000, 12000);
  learningRate.value = Number(clampNumber(learningRate.value, 0.00001, 0.01, 0.0003).toFixed(5));
  discountFactor.value = Number(clampNumber(discountFactor.value, 0.8, 0.999, 0.99).toFixed(3));
  batchSize.value = clampInteger(batchSize.value, 32, 2048, 256);
  rolloutSteps.value = clampInteger(rolloutSteps.value, 64, 4096, 1024);
  entropyCoef.value = Number(clampNumber(entropyCoef.value, 0, 0.2, 0.01).toFixed(3));
  clipRange.value = Number(clampNumber(clipRange.value, 0.05, 0.5, 0.2).toFixed(2));
  simulationWindowHours.value = clampInteger(simulationWindowHours.value, 1, 72, 6);
  coverageTarget.value = clampInteger(coverageTarget.value, 10, 100, 85);
  logWindow.value = clampInteger(logWindow.value, 10, 200, 50);
  evalInterval.value = clampInteger(evalInterval.value, 1000, 50000, 5000);

  const snapshot = {
    algorithm: selectedAlgorithm.value,
    reward_mode: selectedRewardMode.value,
    total_timesteps: totalTimesteps.value,
    learning_rate: learningRate.value,
    discount_factor: discountFactor.value,
    batch_size: batchSize.value,
    rollout_steps: rolloutSteps.value,
    entropy_coef: entropyCoef.value,
    clip_range: clipRange.value,
    env_type: envType.value,
    stochastic_eval: stochasticEval.value,
    simulation_window_hours: simulationWindowHours.value,
    coverage_target: coverageTarget.value,
    traffic_load_profile: trafficLoadProfile.value,
    priority_objective: priorityObjective.value,
    log_window: logWindow.value,
    eval_interval: evalInterval.value,
    auto_replay: autoReplay.value,
  };

  appendTrainingEvent(
    `应用训练参数：步数=${totalTimesteps.value} 学习率=${learningRate.value} batch=${batchSize.value} rollout=${rolloutSteps.value} 奖励=${rewardModeLabel(selectedRewardMode.value)}。`,
    "ui_action",
    snapshot
  );
  appendTrainingTerminalLine("配置响应：参数已写入训练页状态，下一次启动训练会随 /api/train 请求提交后端。", {
    level: "OK",
    source: "TRAIN",
  });
};

const selectAlgorithm = (value) => {
  selectedAlgorithm.value = value;
  if (value === "dqn" && totalTimesteps.value < 40000) {
    totalTimesteps.value = 40000;
  }
  if (value !== "dqn" && totalTimesteps.value === 40000) {
    totalTimesteps.value = 12000;
  }
  appendTrainingEvent(`切换训练算法：${selectedAlgorithmLabel.value}。`, "ui_action", { algorithm: value });
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
    artifact?.algorithm === runMeta.algorithm &&
    (!runMeta.evaluationProtocol || !artifact?.evaluation_protocol || artifact.evaluation_protocol === runMeta.evaluationProtocol);

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
  if (!autoReplay.value) return;
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
      evaluation_protocol: runMeta.evaluationProtocol,
      stochastic_eval: true,
      eval_seed: 13,
      episodes: 1,
      replay_source: "training",
    });

    let replayMessage = "训练完成后已自动生成后端回放，可在回放页刷新列表后查看。";
    if (data?.replay_session_id) {
      setActiveReplaySessionId(data.replay_session_id);
      replayMessage = `训练完成后已自动生成后端回放：${data.replay_session_id}。`;
    } else {
      saveReplaySessionFromSimulation({
        scenarioName: runMeta.scenarioName,
        algorithm: runMeta.algorithm,
        result: { ...data, source: "training" },
      });
      replayMessage = "训练完成后已自动生成一条回放，可在回放页刷新列表后查看。";
    }
    eventLog.value = [
      ...eventLog.value.slice(-79),
      {
        type: "training_replay_ready",
        timestamp: Date.now() / 1000,
        message: replayMessage,
      },
    ];
    appendTrainingTerminalLine(`后端响应：${replayMessage}`, { level: "REPLAY", source: "BACKEND" });
  } catch (error) {
    console.error("Failed to generate replay from training", error);
    const replayErrorMessage = error?.message || "自动生成训练回放失败";
    eventLog.value = [
      ...eventLog.value.slice(-79),
      {
        type: "training_replay_error",
        timestamp: Date.now() / 1000,
        message: replayErrorMessage,
      },
    ];
    appendTrainingTerminalLine(`后端响应：${replayErrorMessage}`, { level: "ERROR", source: "BACKEND" });
  } finally {
    replayRunIdInFlight.value = null;
  }
};

const finiteTerminalNumber = (value) => {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
};

const normalizedTerminalRatio = (value) => {
  const numeric = finiteTerminalNumber(value);
  if (numeric == null) return null;
  return Math.abs(numeric) > 1 ? numeric / 100 : numeric;
};

const formatSignedTerminalMetric = (value, digits = 2) => {
  const numeric = finiteTerminalNumber(value);
  if (numeric == null) return "--";
  return `${numeric >= 0 ? "+" : ""}${numeric.toFixed(digits)}`;
};

const formatTerminalThroughput = (value) => {
  const numeric = finiteTerminalNumber(value);
  if (numeric == null) return "";
  return `${numeric.toFixed(2)}Mbps`;
};

const trainingStepText = (payload = {}) => {
  const step = finiteTerminalNumber(payload.global_step ?? payload.step);
  const total = finiteTerminalNumber(payload.total_timesteps ?? totalTimesteps.value);
  if (step == null && total == null) return "";
  const currentText = step == null ? "--" : formatInteger(step);
  const totalText = total == null || total <= 0 ? "" : `/${formatInteger(total)}`;
  const progressText = step != null && total != null && total > 0 ? ` (${formatPercent(step / total)})` : "";
  return `step=${currentText}${totalText}${progressText}`;
};

const trainingUsersText = (payload = {}) => {
  const totalUsers = finiteTerminalNumber(payload.total_users ?? selectedScenario.value?.num_users);
  const coverage = normalizedTerminalRatio(payload.coverage);
  const connectedUsers = finiteTerminalNumber(payload.connected_users) ?? (
    totalUsers != null && coverage != null ? Math.round(totalUsers * coverage) : null
  );
  if (totalUsers == null || totalUsers <= 0 || connectedUsers == null) return "";
  return `${formatInteger(connectedUsers)}/${formatInteger(totalUsers)}用户`;
};

const appendTerminalMetric = (parts, label, value, formatter = formatMetric) => {
  const numeric = finiteTerminalNumber(value);
  if (numeric == null) return;
  parts.push(`${label}=${formatter(numeric)}`);
};

const hierarchyTerminalText = (hierarchy = {}) => {
  if (!hierarchy || typeof hierarchy !== "object" || !Object.keys(hierarchy).length) return "";
  const parts = [];
  if (hierarchy.target_region_id != null) parts.push(`目标区=${hierarchy.target_region_id}`);
  if (hierarchy.target_users != null) parts.push(`目标用户=${formatInteger(hierarchy.target_users)}`);
  if (hierarchy.l2_link_count != null) parts.push(`链路=${formatInteger(hierarchy.l2_link_count)}`);
  if (hierarchy.l3_deployed_devices != null) parts.push(`部署=${formatInteger(hierarchy.l3_deployed_devices)}`);
  if (hierarchy.hierarchical_reward != null) parts.push(`层级奖励=${formatSignedTerminalMetric(hierarchy.hierarchical_reward)}`);
  return parts.length ? `HMARL[${parts.join(" ")}]` : "";
};

const episodeTerminalMessage = (payload = {}) => {
  const parts = [`Episode #${formatInteger(payload.episode)}`];
  const stepText = trainingStepText(payload);
  if (stepText) parts.push(stepText);
  if (payload.steps != null) parts.push(`len=${formatInteger(payload.steps)}`);
  parts.push(`reward=${formatSignedTerminalMetric(payload.reward)}`);
  if (payload.coverage != null) {
    const usersText = trainingUsersText(payload);
    parts.push(`coverage=${formatPercent(payload.coverage)}${usersText ? ` (${usersText})` : ""}`);
  }
  if (payload.broadcast != null) parts.push(`broadcast=${formatPercent(payload.broadcast)}`);
  const avgThroughput = formatTerminalThroughput(payload.avg_user_throughput);
  if (avgThroughput) parts.push(`avg_tp=${avgThroughput}`);
  const recentThroughput = formatTerminalThroughput(payload.recent_throughput);
  if (recentThroughput) parts.push(`recent_tp=${recentThroughput}`);
  appendTerminalMetric(parts, "budget", payload.remaining_budget, (value) => value.toFixed(0));
  appendTerminalMetric(parts, "device_cost", payload.device_cost);
  appendTerminalMetric(parts, "bandwidth_cost", payload.bandwidth_cost);
  const hierarchyText = hierarchyTerminalText(payload.hierarchy);
  if (hierarchyText) parts.push(hierarchyText);
  if (payload.reason) parts.push(`reason=${payload.reason}`);
  return parts.join(" | ");
};

const updateTerminalMessage = (payload = {}) => {
  const parts = [`Update #${formatInteger(payload.update ?? payload.step)}`];
  const stepText = trainingStepText(payload);
  if (stepText) parts.push(stepText);
  appendTerminalMetric(parts, "mean_reward", payload.mean_reward, formatSignedTerminalMetric);
  if (payload.mean_coverage != null) parts.push(`mean_coverage=${formatPercent(payload.mean_coverage)}`);
  if (payload.mean_broadcast != null) parts.push(`mean_broadcast=${formatPercent(payload.mean_broadcast)}`);
  appendTerminalMetric(parts, "loss_pi", payload.loss_pi ?? payload.policy_loss);
  appendTerminalMetric(parts, "loss_v", payload.loss_v ?? payload.value_loss);
  appendTerminalMetric(parts, "aux_loss", payload.aux_loss);
  appendTerminalMetric(parts, "q_loss", payload.q_loss);
  appendTerminalMetric(parts, "entropy", payload.entropy);
  appendTerminalMetric(parts, "epsilon", payload.epsilon);
  return parts.join(" | ");
};

const evaluationTerminalMessage = (payload = {}) => {
  const parts = ["Eval"];
  const stepText = trainingStepText(payload);
  if (stepText) parts.push(stepText);
  appendTerminalMetric(parts, "avg_reward", payload.avg_reward, formatSignedTerminalMetric);
  if (payload.avg_coverage != null) parts.push(`avg_coverage=${formatPercent(payload.avg_coverage)}`);
  if (payload.avg_broadcast != null) parts.push(`avg_broadcast=${formatPercent(payload.avg_broadcast)}`);
  return parts.join(" | ");
};

const completedTerminalMessage = (payload = {}) => {
  const parts = ["训练完成"];
  const stepText = trainingStepText(payload);
  if (stepText) parts.push(stepText);
  if (payload.episodes != null) parts.push(`episodes=${formatInteger(payload.episodes)}`);
  return parts.join(" | ");
};

const trainingStreamMessage = (payload) => {
  const eventPayload = payload?.payload || {};
  if (payload?.message) return payload.message;
  if (eventPayload?.message) return eventPayload.message;
  if (payload?.type === "episode") return episodeTerminalMessage(eventPayload);
  if (payload?.type === "update") return updateTerminalMessage(eventPayload);
  if (payload?.type === "evaluation") return evaluationTerminalMessage(eventPayload);
  if (payload?.type === "completed") return completedTerminalMessage(eventPayload);
  if (payload?.type === "status" && eventPayload?.state) {
    const stepText = trainingStepText(eventPayload);
    return `训练状态：${eventPayload.state}${stepText ? ` | ${stepText}` : ""}`;
  }
  if (payload?.type) return `训练事件：${payload.type}`;
  return "训练事件";
};

const trainingStreamLevel = (payload) => {
  const type = String(payload?.type || "EVENT").toLowerCase();
  if (type === "episode") return "EPISODE";
  if (type === "update") return "UPDATE";
  if (type === "evaluation") return "EVAL";
  if (type === "completed") return "OK";
  if (type === "status") return "STATUS";
  if (type === "log") return "LOG";
  if (type === "error") return "ERROR";
  return type.toUpperCase();
};

const subscribeToEvents = (runId) => {
  runStatus.value = "running";
  eventSource = new EventSource(`${API_BASE}/train/${runId}/stream`);

  eventSource.onopen = () => {
    appendTrainingTerminalLine(`后端响应：训练 SSE 已连接 run_id=${runId}。`, { level: "OK", source: "BACKEND" });
  };

  eventSource.onmessage = (event) => {
    if (!event.data) return;
    try {
      const payload = JSON.parse(event.data);
      if (payload.type === "end") {
        runStatus.value = payload.status;
        appendTrainingTerminalLine(`后端响应：训练流结束 status=${payload.status || "unknown"}。`, {
          level: payload.status === "completed" ? "OK" : "STATUS",
          source: "BACKEND",
          timestamp: payload.timestamp,
        });
        closeEventSource();
        return;
      }

      eventLog.value = [...eventLog.value, payload].slice(-80);
      appendTrainingTerminalLine(trainingStreamMessage(payload), {
        level: trainingStreamLevel(payload),
        source: "BACKEND",
        timestamp: payload.timestamp,
      });

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
    appendTrainingEvent("训练 SSE 连接中断，请检查后端训练任务状态。", "error");
  };
};

const startTraining = async () => {
  if (!selectedScenarioName.value) return;

  const trainingBaseStations = buildTrainingBaseStations();
  isStarting.value = true;
  actionError.value = "";
  eventLog.value = [
      {
        type: "scene_import",
        timestamp: Date.now() / 1000,
        payload: {
          scenario_name: selectedScenarioName.value,
          disaster_type: selectedScenario.value?.disaster_type,
          source_scenario: disasterType.value,
          severity_level: disasterSeverity.value,
          affected_grid_count: selectedScenario.value?.candidate_sites,
          impacted_population: selectedScenario.value?.num_users,
          devices: trainingBaseStations,
        },
        message: `已确认 ${selectedScenarioLabel.value} 场景，准备启动 ${selectedAlgorithm.value.toUpperCase()} 训练。`,
      },
      {
        type: "experiment_config",
        timestamp: Date.now() / 1000,
        payload: {
          reward_mode: selectedRewardMode.value,
          total_timesteps: totalTimesteps.value,
          learning_rate: learningRate.value,
          batch_size: batchSize.value,
          rollout_steps: rolloutSteps.value,
        },
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

    appendTrainingTerminalLine(
      `前端操作：提交训练任务 scenario=${selectedScenarioName.value} algorithm=${selectedAlgorithm.value} reward=${rewardMode || "default"} devices=${trainingBaseStations.length}。`,
      { level: "ACTION" }
    );
    appendTrainingTerminalLine(
      `训练请求参数：steps=${totalTimesteps.value} lr=${learningRate.value} gamma=${discountFactor.value} batch=${batchSize.value} rollout=${rolloutSteps.value} eval_interval=${evalInterval.value}。`,
      { level: "CONFIG" }
    );

    const { data } = await axios.post(`${API_BASE}/train`, {
      scenario_name: selectedScenarioName.value,
      env_type: envType.value,
      algorithm: selectedAlgorithm.value,
      total_timesteps: totalTimesteps.value,
      stochastic_eval: stochasticEval.value,
      reward_mode: rewardMode,
      evaluation_protocol: evaluationProtocol.value,
      learning_rate: learningRate.value,
      discount_factor: discountFactor.value,
      batch_size: batchSize.value,
      rollout_steps: rolloutSteps.value,
      entropy_coef: entropyCoef.value,
      clip_range: clipRange.value,
      eval_interval: evalInterval.value,
      custom_base_stations: trainingBaseStations,
    });
    appendTrainingTerminalLine(`后端响应：训练任务已创建 run_id=${data.run_id || "--"}。`, {
      level: "OK",
      source: "BACKEND",
    });

    activeRunMeta.value = {
      runId: data.run_id,
      scenarioName: selectedScenarioName.value,
      algorithm: selectedAlgorithm.value,
      rewardMode,
      evaluationProtocol: evaluationProtocol.value,
    };

    subscribeToEvents(data.run_id);
  } catch (error) {
    console.error("Failed to start training", error);
    runStatus.value = "error";
    actionError.value = `启动训练失败: ${error?.message || "未知错误"}`;
    appendTrainingEvent(actionError.value, "error");
  } finally {
    isStarting.value = false;
  }
};

const stopTraining = () => {
  closeEventSource();
  runStatus.value = "stopped";
  appendTrainingTerminalLine("前端操作：用户手动停止训练。", { level: "ACTION" });
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
  disasterType.value = scenarioSourceKey(scenario);
  disasterSeverity.value = scenarioSeverityKey(scenario);
  selectedRewardMode.value =
    scenario.default_reward_profile || scenario.reward_profiles?.[0]?.key || null;
  if (preserveSnapshotAccessDevices) {
    preserveSnapshotAccessDevices = false;
  } else {
    initializeAccessDevicesFromScenario();
  }
  appendScenarioUserNodeCount(scenario);
}, { immediate: true });

watch([historyFilterAlgorithm, historyFilterScenario], () => {
  historyPage.value = 1;
});

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

/* ===== Native prototype shell ===== */
.training-page {
  min-height: 1010px;
  background: #eef5ff;
  color: #1f2d3d;
  font-family: "Microsoft YaHei", "PingFang SC", "Source Han Sans CN", sans-serif;
}

.page-bg,
.page-panel-shadow {
  position: absolute;
  display: block;
  border: 0;
  pointer-events: none;
  user-select: none;
  z-index: 0;
}

.page-bg {
  left: 0;
  top: 0;
  width: 1920px;
  height: 1010px;
}

.page-panel-shadow {
  left: 97px;
  top: 0;
  width: 1740px;
  height: 1027px;
  opacity: 0.5;
}

.training-shell {
  position: absolute;
  left: 140px;
  top: 44px;
  z-index: 2;
  width: 1652px;
  height: 930px;
  min-height: 0;
  overflow-x: hidden;
  overflow-y: auto;
  scrollbar-color: rgba(57, 97, 246, 0.45) rgba(225, 236, 255, 0.72);
  scrollbar-width: thin;
}

.training-shell::-webkit-scrollbar {
  width: 8px;
}

.training-shell::-webkit-scrollbar-track {
  background: rgba(225, 236, 255, 0.72);
  border-radius: 999px;
}

.training-shell::-webkit-scrollbar-thumb {
  background: rgba(57, 97, 246, 0.45);
  border-radius: 999px;
}

.training-shell__scroll {
  position: relative;
  box-sizing: border-box;
  width: 1640px;
  min-height: 100%;
  padding-top: 82px;
}

.page-title {
  position: absolute;
  left: 0;
  top: 0;
  width: 157px;
  height: 68px;
}

.page-title__ribbon {
  position: absolute;
  left: -14px;
  top: 2px;
  width: 157px;
  height: 66px;
}

.page-title h1 {
  position: absolute;
  left: 3px;
  top: 3px;
  width: 125px;
  margin: 0;
  color: #1890ff;
  font-family: "Source Han Sans CN", "Microsoft YaHei", sans-serif;
  font-size: 20px;
  font-weight: 700;
  line-height: 41px;
  text-align: center;
  text-shadow: 0 0 20px rgba(0, 200, 244, 0.5);
}

.record-button {
  position: absolute;
  left: 153px;
  top: 14px;
  width: 99px;
  height: 33px;
  border: 1px solid #f2f2f2;
  border-radius: 8px;
  background: rgba(255, 255, 255, 0);
  color: #0079fe;
  font-size: 16px;
}

.record-button:hover {
  background: rgba(0, 102, 255, 0.07);
}

.module-panel,
.summary-grid article,
.scenario-description-card,
.severity-insight-card,
.device-access-card {
  box-sizing: border-box;
  border: 1px solid rgba(233, 233, 233, 1);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.96);
  box-shadow: 3px 3px 20px rgba(233, 233, 233, 0.9);
}

.module-panel {
  width: 1628px;
  padding: 14px;
  margin: 0 0 14px 4px;
}

.module-heading,
.sub-panel-heading {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  min-height: 36px;
  margin-bottom: 12px;
}

.module-heading > div:first-child {
  display: flex;
  align-items: center;
  gap: 10px;
  min-width: 0;
}

.module-heading i {
  width: 6px;
  height: 20px;
  flex: 0 0 auto;
  background: linear-gradient(180deg, #6fcadf 0%, #05b7df 100%);
}

.module-heading h2,
.sub-panel-heading h3 {
  margin: 0;
  color: #1f2d3d;
  font-size: 16px;
  font-weight: 400;
}

.module-heading p,
.sub-panel-heading p {
  min-width: 0;
  max-width: 780px;
  margin: 0 0 0 8px;
  overflow: hidden;
  color: #64748b;
  font-size: 12px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.sub-panel-heading {
  align-items: center;
  margin-bottom: 10px;
}

.sub-panel-heading p {
  margin: 4px 0 0;
}

.module-actions {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.primary-button,
.ghost-button {
  height: 34px;
  padding: 0 14px;
  border-radius: 6px;
  font-size: 14px;
  font-family: inherit;
  cursor: pointer;
}

.primary-button {
  border: 1px solid #b7e0fe;
  background: #3961f6;
  color: #fff;
}

.ghost-button {
  border: 1px solid #d7e3f4;
  background: #fff;
  color: #17315d;
}

.ghost-button--blue {
  border-color: #b7e0fe;
  background: #ebf5ff;
  color: #2563eb;
}

.primary-button:disabled,
.ghost-button:disabled {
  cursor: not-allowed;
  opacity: 0.55;
}

.danger-link {
  border-color: #fecaca;
  color: #b91c1c;
}

.module-error {
  width: 1628px;
  margin: 0 0 12px 4px;
  box-sizing: border-box;
  border: 1px solid #fecaca;
  border-radius: 6px;
  background: #fef2f2;
  color: #991b1b;
  padding: 8px 10px;
}

.dataset-controls {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  gap: 12px;
  align-items: stretch;
  margin-bottom: 12px;
}

.dataset-choice {
  min-width: 0;
  border: 1px solid rgba(183, 224, 254, 0.55);
  border-radius: 10px;
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.92), rgba(248, 251, 255, 0.88));
  padding: 10px;
  box-shadow: 0 0 5px rgba(246, 246, 254, 0.55);
}

.dataset-choice__label {
  display: block;
  margin-bottom: 8px;
  color: #334155;
  font-size: 13px;
  font-weight: 700;
  line-height: 18px;
}

.dataset-option-grid {
  display: grid;
  gap: 8px;
  max-height: 72px;
  overflow-y: auto;
  padding-right: 2px;
  scrollbar-width: thin;
}

.dataset-option-grid--scenario,
.dataset-option-grid--severity {
  grid-template-columns: repeat(4, minmax(0, 1fr));
}

.dataset-option-card {
  min-height: 60px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: 4px;
  padding: 9px 10px;
  border: 1px solid rgba(183, 224, 254, 0.5);
  border-radius: 10px;
  background: rgba(255, 255, 255, 0.78);
  color: #333333;
  cursor: pointer;
  font-family: inherit;
  text-align: left;
  transition: all 0.2s ease;
}

.dataset-option-card:hover {
  border-color: #1890ff;
  background: rgba(231, 238, 255, 0.62);
}

.dataset-option-card--active {
  border-color: #3961f6;
  background: rgba(231, 238, 255, 0.72);
  color: #3961f6;
  box-shadow: 0 0 6px rgba(57, 97, 246, 0.16);
}

.dataset-option-card__name,
.dataset-option-card__desc {
  display: block;
  max-width: 100%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.dataset-option-card__name {
  font-size: 14px;
  font-weight: 700;
  line-height: 20px;
}

.dataset-option-card__desc {
  color: #64748b;
  font-size: 12px;
  line-height: 16px;
}

.severity-insight-grid small {
  display: block;
  color: #64748b;
  font-size: 11px;
  font-weight: 500;
  line-height: 14px;
}

.severity-insight-grid strong {
  display: block;
  overflow: hidden;
  color: #172554;
  font-size: 13px;
  font-variant-numeric: tabular-nums;
  font-weight: 700;
  line-height: 17px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.severity-insight-card {
  margin: -2px 0 12px;
  padding: 12px 14px;
}

.severity-insight-card__title {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 10px;
}

.severity-insight-card__title strong {
  color: #1f2d3d;
  font-size: 14px;
  font-weight: 700;
  line-height: 20px;
}

.severity-insight-card__title span {
  min-width: 0;
  overflow: hidden;
  color: #475569;
  font-size: 12px;
  line-height: 18px;
  text-align: right;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.severity-insight-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
}

.severity-insight-grid > span {
  min-width: 0;
  padding: 8px 10px;
  border: 1px solid rgba(183, 224, 254, 0.5);
  border-radius: 6px;
  background: rgba(248, 251, 255, 0.82);
}

.severity-insight-grid strong {
  margin-top: 2px;
  font-size: 16px;
  line-height: 22px;
}

.severity-insight-grid em {
  display: block;
  overflow: hidden;
  color: #64748b;
  font-size: 11px;
  font-style: normal;
  line-height: 15px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.summary-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin-bottom: 14px;
}

.summary-grid article {
  min-height: 82px;
  padding: 13px 14px;
}

.summary-grid small,
.summary-grid span {
  color: #64748b;
  font-size: 12px;
}

.summary-grid strong {
  display: block;
  margin: 5px 0;
  color: #1f2d3d;
  font-size: 24px;
}

.scenario-description-card {
  margin-bottom: 14px;
  padding: 13px 14px;
}

.scenario-description-card h3 {
  margin: 0 0 8px;
  color: #1f2d3d;
  font-size: 15px;
  font-weight: 700;
  line-height: 22px;
}

.scenario-description-card p {
  margin: 0;
  color: #334155;
  font-size: 14px;
  line-height: 24px;
}

.form-grid--compact {
  margin-top: 14px;
}

.device-access-card {
  margin-top: 14px;
  padding: 12px;
}

.device-table {
  max-height: 430px;
  overflow: auto;
}

.device-table__head,
.device-table__row {
  display: grid;
  grid-template-columns: 54px minmax(220px, 1fr) minmax(160px, 0.9fr) 80px 82px 82px 86px 74px;
  align-items: center;
  gap: 10px;
}

.device-table__head {
  height: 34px;
  border: 1px solid #edf2f7;
  border-left: 0;
  border-right: 0;
  background: #f8fafc;
  color: #64748b;
  text-align: center;
}

.device-table__row {
  min-height: 48px;
  border-top: 1px solid #edf2f7;
  color: #334155;
  font-size: 13px;
}

.device-name-cell {
  min-width: 0;
  overflow: hidden;
  color: #17315d;
  font-weight: 600;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.device-table__row input,
.device-table__row select {
  width: 100%;
  height: 34px;
  border: 1px solid #d7e3f4;
  border-radius: 6px;
  background: #fff;
  color: #17315d;
  padding: 0 10px;
}

.device-remove-button {
  justify-self: center;
  height: 30px;
  padding: 0 12px;
  font-size: 12px;
}

.device-table__empty {
  min-height: 44px;
  display: flex;
  align-items: center;
  border-top: 1px solid #edf2f7;
  color: #64748b;
  font-size: 13px;
}

.device-state {
  display: inline-flex;
  justify-content: center;
  height: 28px;
  line-height: 28px;
  border-radius: 999px;
  background: rgba(34, 197, 94, 0.12);
  color: #15803d;
  white-space: nowrap;
}

.algorithm-card-grid,
.reward-card-grid {
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 14px;
}

.reward-card-grid {
  grid-template-columns: repeat(4, minmax(0, 1fr));
}

.algorithm-card,
.reward-card {
  min-height: 104px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 6px;
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

.algorithm-card:hover,
.reward-card:hover {
  border-color: #1890ff;
  background: rgba(231, 238, 255, 0.5);
}

.algorithm-card--active,
.reward-card--active {
  border-color: #3961f6;
  background: rgba(231, 238, 255, 0.5);
  color: #3961f6;
  box-shadow: 0 0 6px rgba(57, 97, 246, 0.15);
}

.algorithm-card__name,
.reward-card__name {
  max-width: 100%;
  font-size: 16px;
  font-weight: 700;
  line-height: 22px;
  text-align: center;
}

.algorithm-card__desc,
.reward-card__desc {
  max-width: 100%;
  overflow: hidden;
  font-size: 12px;
  line-height: 18px;
  opacity: 0.7;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.action-bar {
  width: 1628px;
  box-sizing: border-box;
  margin: 0 0 14px 4px;
  padding: 4px 0;
}

.run-state {
  color: #1890ff;
  font-weight: 700;
}

.train-button {
  min-width: 128px;
  height: 40px;
}

.train-button--stop {
  border-color: #fecaca;
  background: #dc7274;
}

.result-panel {
  padding: 14px;
}

.result-stack {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.prototype-modal {
  position: fixed;
  inset: 0;
  z-index: 5000;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(2, 6, 23, 0.5);
}

.prototype-modal--detail {
  z-index: 5010;
  background: rgba(2, 6, 23, 0.58);
}

.prototype-dialog {
  width: 1112px;
  max-height: 798px;
  overflow: auto;
  border: 1px solid #e4e4e4;
  border-radius: 10px;
  background: #fff;
  color: #333;
}

.prototype-dialog--history {
  width: 1180px;
}

.prototype-dialog--history-detail {
  width: 1120px;
  max-height: 820px;
}

.prototype-dialog header {
  height: 61px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  border-bottom: 1px solid #e4e4e4;
  padding: 0 12px;
}

.prototype-dialog header > div {
  min-width: 0;
}

.prototype-dialog h2 {
  margin: 0;
  font-family: "Source Han Sans CN", "Microsoft YaHei", sans-serif;
  font-size: 18px;
  font-weight: 500;
  line-height: 28px;
}

.prototype-dialog header p {
  margin: 4px 0 0;
  color: #64748b;
  font-size: 13px;
  line-height: 20px;
}

.prototype-dialog header p span {
  margin: 0 6px;
  color: #94a3b8;
}

.prototype-dialog header button {
  width: 58px;
  height: 34px;
  border: 1px solid #d7e3f4;
  border-radius: 6px;
  background: #fff;
  color: #64748b;
}

.prototype-dialog table {
  width: calc(100% - 40px);
  margin: 20px;
  border-collapse: collapse;
  font-size: 16px;
}

.prototype-dialog th,
.prototype-dialog td {
  height: 45px;
  border-bottom: 1px solid #eef2f7;
  padding: 0 14px;
  color: #666;
  text-align: left;
}

.prototype-dialog th {
  background: rgba(247, 248, 250, 0.5);
  font-weight: 700;
}

.prototype-dialog tbody tr:hover {
  background: rgba(231, 238, 255, 0.5);
}

.history-panel__filters {
  display: flex;
  gap: 10px;
  align-items: center;
  margin: 16px 20px 0;
}

.history-panel__filters .field__input {
  width: 260px;
}

.history-detail {
  padding: 20px;
}

.history-detail__loading,
.history-detail__error {
  min-height: 220px;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 24px;
  color: #64748b;
  font-size: 16px;
}

.history-detail__error {
  color: #b42318;
}

.history-detail__cards {
  display: grid;
  grid-template-columns: repeat(6, minmax(0, 1fr));
  gap: 10px;
  margin-bottom: 16px;
}

.history-detail__cards article,
.history-detail__block {
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  background: #f8fafc;
}

.history-detail__cards article {
  min-height: 82px;
  box-sizing: border-box;
  padding: 12px;
}

.history-detail__cards small {
  display: block;
  color: #64748b;
  font-size: 12px;
  line-height: 18px;
}

.history-detail__cards strong {
  display: block;
  margin-top: 6px;
  color: #0f172a;
  font-size: 20px;
  line-height: 28px;
}

.history-detail__chart,
.history-detail__test {
  margin-bottom: 16px;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  background: #fff;
  padding: 14px;
}

.history-detail__chart-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 10px;
}

.history-detail__chart h3,
.history-detail__test h3 {
  margin: 0;
  color: #0f172a;
  font-size: 16px;
  line-height: 24px;
}

.history-detail__legend {
  display: flex;
  gap: 16px;
  color: #475569;
  font-size: 13px;
  line-height: 20px;
}

.history-detail__legend span {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}

.history-detail__legend i {
  width: 18px;
  height: 3px;
  border-radius: 999px;
}

.history-detail__legend-coverage {
  background: #2563eb;
}

.history-detail__legend-broadcast {
  background: #059669;
}

.history-detail__chart svg {
  display: block;
  width: 100%;
  height: 220px;
  overflow: visible;
}

.history-detail__axis {
  stroke: #94a3b8;
  stroke-width: 1.2;
}

.history-detail__grid-line {
  stroke: #e2e8f0;
  stroke-width: 1;
}

.history-detail__axis-label {
  fill: #64748b;
  font-size: 10px;
}

.history-detail__line {
  fill: none;
  stroke-linecap: round;
  stroke-linejoin: round;
  stroke-width: 3;
}

.history-detail__line--coverage {
  stroke: #2563eb;
}

.history-detail__line--broadcast {
  stroke: #059669;
}

.history-detail__test > div {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
  margin-top: 10px;
}

.history-detail__test article {
  min-height: 70px;
  box-sizing: border-box;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  background: #f8fafc;
  padding: 10px 12px;
}

.history-detail__test small {
  display: block;
  color: #64748b;
  font-size: 12px;
  line-height: 18px;
}

.history-detail__test strong {
  display: block;
  margin-top: 5px;
  color: #0f172a;
  font-size: 18px;
  line-height: 26px;
}

.history-detail__grid {
  display: grid;
  grid-template-columns: minmax(0, 1.1fr) minmax(0, 0.9fr);
  gap: 14px;
  margin-bottom: 16px;
}

.history-detail__block {
  padding: 14px;
  overflow: hidden;
}

.history-detail__block--wide {
  background: #fff;
}

.history-detail__block h3 {
  margin: 0 0 10px;
  color: #0f172a;
  font-size: 16px;
  line-height: 24px;
}

.history-detail__block dl {
  display: grid;
  gap: 8px;
  margin: 0;
}

.history-detail__block dl div {
  display: grid;
  grid-template-columns: 98px minmax(0, 1fr);
  gap: 10px;
  min-height: 24px;
}

.history-detail__block dt {
  color: #64748b;
  font-size: 13px;
}

.history-detail__block dd {
  min-width: 0;
  margin: 0;
  color: #1f2937;
  font-size: 13px;
  line-height: 20px;
  overflow-wrap: anywhere;
}

.history-detail__block table {
  width: 100%;
  margin: 0;
  border-collapse: collapse;
  font-size: 14px;
}

.history-detail__block th,
.history-detail__block td {
  height: 40px;
  padding: 0 12px;
  border-bottom: 1px solid #eef2f7;
}

.empty-row {
  height: 96px !important;
  color: #999 !important;
  text-align: center !important;
}

.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.18s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>
