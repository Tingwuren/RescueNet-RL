<template>
  <div class="strategy-tester">
    <img class="strategy-tester__bg" :src="assetUrl('images/模型训练/u537.png')" alt="" />
    <img class="strategy-tester__panel-shadow" :src="assetUrl('images/模型训练/u538.png')" alt="" />

    <main class="strategy-panel" aria-label="策略测试">
      <div class="strategy-panel__scroll" :style="{ height: `${strategyPanelHeight}px` }">
      <div class="strategy-title">
        <img class="strategy-title__ribbon" :src="assetUrl('images/模型训练/u541.png')" alt="" />
        <h1>策略测试</h1>
      </div>

      <button type="button" class="record-button" @click="historyModalOpen = true">测试记录</button>

      <section class="map-shell" :style="{ top: `${mapPanelTop}px` }" aria-label="策略测试地图">
        <button
          type="button"
          :class="['scene-tab', 'scene-tab--imported', { 'scene-tab--active': activeSceneTab === 'imported' }]"
          @click="activeSceneTab = 'imported'"
        >
          导入的场景
        </button>
        <button
          type="button"
          :class="['scene-tab', 'scene-tab--deployment', { 'scene-tab--active': activeSceneTab === 'deployment' }]"
          @click="selectDeploymentTab"
        >
          部署后场景
        </button>

        <div class="info-pill info-pill--region" :title="regionText">{{ regionText }}</div>
        <div class="info-pill info-pill--span" :title="spanText">{{ spanText }}</div>

        <img class="satellite-map" :src="assetUrl('images/首页/u127.jpg')" alt="" />
        <div v-if="mapTiles.length" class="tile-map" aria-hidden="true">
          <img
            v-for="tile in mapTiles"
            :key="tile.key"
            :src="tile.url"
            :style="{ left: `${tile.left}px`, top: `${tile.top}px` }"
            alt=""
            draggable="false"
          />
          <span class="tile-map__label">{{ mapLabel }}</span>
        </div>
        <div class="node-layer" aria-label="地图节点">
          <span
            v-for="marker in mapMarkers"
            :key="marker.id"
            :class="['node-marker', `node-marker--${marker.tone}`, { 'node-marker--station': marker.kind === 'station' }]"
            :style="markerStyle(marker)"
            :tabindex="marker.kind === 'station' ? 0 : undefined"
            :role="marker.kind === 'station' ? 'img' : undefined"
            :aria-label="marker.kind === 'station' ? marker.title : undefined"
            @mouseenter="showStationTooltip(marker)"
            @focus="showStationTooltip(marker)"
            @mouseleave="hideStationTooltip"
            @blur="hideStationTooltip"
          ></span>
        </div>

        <div v-if="mapEmptyVisible" class="map-empty" role="status" aria-live="polite">
          <strong>{{ mapEmptyTitle }}</strong>
          <span>{{ mapEmptyDescription }}</span>
        </div>

        <transition name="station-tooltip-fade">
          <div
            v-if="stationTooltip.visible && stationTooltipMarker"
            class="station-tooltip"
            :style="stationTooltipStyle"
            role="tooltip"
          >
            <div class="station-tooltip__title">
              <span class="station-tooltip__dot" :style="{ background: stationTooltipMarker.color }"></span>
              <strong>{{ stationTooltipMarker.stationTypeLabel || "基站" }}</strong>
            </div>
            <dl class="station-tooltip__list">
              <div v-for="row in stationTooltipRows" :key="row.label">
                <dt>{{ row.label }}</dt>
                <dd :class="row.status ? ['station-tooltip__status', `station-tooltip__status--${stationTooltipMarker.status || 'unknown'}`] : null">
                  {{ row.value }}
                </dd>
              </div>
            </dl>
          </div>
        </transition>

        <div v-if="mapLegendItems.length" class="map-legend" aria-label="地图图例">
          <div v-for="item in mapLegendItems" :key="item.key" class="map-legend__item">
            <span :class="['map-legend__mark', `map-legend__mark--${item.shape}`]" :style="{ background: item.color }"></span>
            <span>{{ item.label }}</span>
          </div>
        </div>

        <div class="metric-card metric-card--nodes">
          <span>节点</span>
          <strong>{{ summary.nodes }}</strong>
        </div>
        <div class="metric-card metric-card--users">
          <span>用户</span>
          <strong>{{ summary.users }}</strong>
        </div>
        <div class="metric-card metric-card--stations">
          <span>基站</span>
          <strong>{{ summary.stations }}</strong>
        </div>
      </section>

      <transition name="status-fade">
        <div v-if="statusMessage" :class="['status-toast', `status-toast--${statusTone}`]">
          {{ statusMessage }}
        </div>
      </transition>

      <section class="module-panel module-panel--scenario" :style="{ height: `${scenarioPanelHeight}px` }">
        <header class="module-heading">
          <div>
            <i></i>
            <h2>灾害数据接入</h2>
          </div>
          <button type="button" class="ghost-button" :disabled="disasterLoading" @click="loadDisasterCatalogAndImports">
            刷新
          </button>
        </header>

        <div v-if="disasterError" class="module-error">{{ disasterError }}</div>

        <div class="dataset-controls">
          <div class="dataset-choice dataset-choice--scenario">
            <span class="dataset-choice__label">灾害场景</span>
            <div class="dataset-option-grid dataset-option-grid--scenario" aria-label="灾害场景选择">
              <button
                v-for="option in disasterScenarioOptions"
                :key="option.key"
                type="button"
                :class="['dataset-option-card', { 'dataset-option-card--active': selectedDisasterScenario === option.key }]"
                :disabled="disasterImporting"
                @click="selectDisasterScenarioCard(option.key)"
              >
                <span class="dataset-option-card__name">{{ option.label }}</span>
                <span class="dataset-option-card__desc">{{ disasterScenarioCardDescription(option) }}</span>
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
                :class="['dataset-option-card', { 'dataset-option-card--active': selectedDisasterSeverity === option.key }]"
                :disabled="disasterImporting"
                @click="selectDisasterSeverityCard(option.key)"
              >
                <span class="dataset-option-card__name">{{ option.label }}</span>
                <span class="dataset-option-card__desc">{{ disasterSeverityCardDescription(option) }}</span>
              </button>
            </div>
          </div>

          <div class="dataset-import-card">
            <label>
              会话采样
              <input v-model.number="disasterSessionSampleLimit" :disabled="disasterImporting" type="number" min="1" max="500" />
            </label>
            <button
              type="button"
              class="primary-button"
              :disabled="!selectedDisasterScenario || !selectedDisasterSeverity || disasterImporting"
              @click="createDisasterImport"
            >
              {{ disasterImporting ? "导入中..." : "导入数据" }}
            </button>
            <transition name="import-progress-fade">
              <div
                v-if="disasterImporting || disasterImportProgress > 0"
                :class="['import-progress', `import-progress--${disasterImportProgressTone}`]"
                role="status"
                aria-live="polite"
              >
                <div class="import-progress__meta">
                  <span>{{ disasterImportStage }}</span>
                  <strong>{{ Math.round(disasterImportProgress) }}%</strong>
                </div>
                <div class="import-progress__track">
                  <span :style="{ width: `${disasterImportProgress}%` }"></span>
                </div>
              </div>
            </transition>
          </div>
        </div>

        <div class="dataset-main">
          <div class="dataset-map-card">
            <div class="dataset-summary">
              <div>
                <h3>{{ disasterPreviewTitle }}</h3>
                <p>{{ disasterPreviewDescription }}</p>
                <p>有效边界：{{ disasterBoundsText(activeDisasterBounds) }}</p>
              </div>
              <div class="mini-metrics">
                <span><small>覆盖面积</small><strong>{{ formatMetric(Number(disasterCoverageArea), 1) }} km²</strong></span>
                <span><small>网格</small><strong>{{ disasterGrid.rows }} × {{ disasterGrid.cols }}</strong></span>
                <span><small>损毁率</small><strong>{{ formatPercent(disasterDamageRate) }}</strong></span>
                <span><small>离线率</small><strong>{{ formatPercent(disasterOfflineRate) }}</strong></span>
              </div>
            </div>
            <div class="dataset-visual-toolbar">
              <div class="heat-legend" aria-label="用户密度图例">
                <span>用户密度</span>
                <small>低</small>
                <i></i>
                <small>高</small>
              </div>
              <div class="station-status-legend" aria-label="基站状态图例">
                <span v-for="item in stationStatusLegend" :key="item.key">
                  <i :class="`station-status-dot station-status-dot--${item.key}`"></i>{{ item.label }}
                </span>
              </div>
            </div>
            <div class="dataset-grid" :style="{ '--grid-rows': disasterGrid.rows, '--grid-cols': disasterGrid.cols }">
              <span
                v-for="cell in disasterHeatmap"
                :key="`${cell.grid_row}-${cell.grid_col}`"
                class="dataset-heat"
                :style="heatCellStyle(cell)"
                :title="heatCellTitle(cell)"
              ></span>
              <span
                v-for="station in disasterDeployments"
                :key="station.deployment_id || `${station.station_type}-${station.grid_position?.row}-${station.grid_position?.col}`"
                :class="['dataset-station', `dataset-station--${station.status || 'unknown'}`]"
                :style="deploymentMarkerStyle(station)"
                :title="disasterDeploymentTitle(station)"
                :aria-label="disasterDeploymentTitle(station)"
              >
                <i></i>
              </span>
              <span v-if="!disasterHeatmap.length && !disasterDeployments.length" class="dataset-empty">
                导入或选择场景后显示网格热力图和当前站点状态
              </span>
            </div>
          </div>

          <aside class="dataset-side">
            <div class="compact-card">
              <div class="compact-card__title">
                <strong>导入概览</strong>
                <span>{{ selectedDisasterImport?.status || "未导入" }}</span>
              </div>
              <div class="mini-metrics mini-metrics--two">
                <span><small>基站</small><strong>{{ selectedImportStationCount }}</strong></span>
                <span><small>用户</small><strong>{{ selectedDisasterImport?.unique_user_count || "--" }}</strong></span>
                <span><small>活跃 / 降级</small><strong>{{ disasterStationCounts.active || 0 }}/{{ disasterStationCounts.degraded || 0 }}</strong></span>
                <span><small>离线</small><strong>{{ disasterStationCounts.offline || 0 }}</strong></span>
              </div>
            </div>
            <div class="compact-card import-list">
              <div class="compact-card__title">
                <strong>已导入场景</strong>
                <span>{{ disasterImports.length }}</span>
              </div>
              <div
                v-for="record in disasterImports"
                :key="record.import_id"
                :class="['import-row', { active: record.import_id === selectedDisasterImportId }]"
              >
                <div>
                  <strong>{{ record.disaster_scenario_label || record.disaster_scenario }} / {{ record.disaster_severity_label || record.disaster_severity }}</strong>
                  <small>{{ importRecordStatsText(record) }}</small>
                </div>
                <div class="row-actions">
                  <button type="button" @click="selectDisasterImport(record.import_id, false)">详情</button>
                  <button type="button" @click="selectDisasterImport(record.import_id, true)">用于仿真</button>
                  <button type="button" class="danger" @click="deleteDisasterImport(record.import_id)">移除</button>
                </div>
              </div>
              <p v-if="!disasterImports.length" class="empty-note">当前服务会话暂无导入记录。</p>
            </div>
          </aside>
        </div>
      </section>

      <section
        class="module-panel module-panel--device"
        :style="{ top: `${devicePanelTop}px`, height: `${devicePanelHeight}px` }"
      >
        <header class="module-heading">
          <div>
            <i></i>
            <h2>设备接入</h2>
            <p>{{ deviceSummaryLabel }}</p>
          </div>
          <div class="module-actions">
            <button
              type="button"
              class="ghost-button ghost-button--danger"
              :disabled="isRunning || isLoading || isClearingResidualNetwork || !scenarioName || !activeAppliedDeviceRows.length"
              @click="clearTestingResidualNetwork"
            >
              {{ isClearingResidualNetwork ? "清空中..." : "清空残余网络" }}
            </button>
            <button type="button" class="primary-button" :disabled="!hasImportedScene || isLoading || isClearingResidualNetwork" @click="addDeviceSlot">+ 添加设备</button>
          </div>
        </header>

        <div class="device-table">
          <div class="device-table__head">
            <span>序号</span><span>接入设备</span><span>通信方式</span><span>关键参数</span><span>数量</span><span>x（行）</span><span>y（列）</span><span>状态</span><span>操作</span>
          </div>
          <div v-for="(row, index) in appliedDeviceRows" :key="row.deviceId" class="device-table__row">
            <span>{{ index + 1 }}</span>
            <span class="device-name-cell" :title="deviceOptionLabel(row)">
              {{ deviceOptionLabel(row) }}
            </span>
            <span>{{ communicationCategoryLabel(row.communicationType) }}</span>
            <span>{{ formatDeviceParams(row) }}</span>
            <input v-model.number="row.quantity" type="number" min="1" @change="syncScenarioBaseStations" />
            <input v-model.number="row.x" type="number" min="0" :max="gridBounds.maxX" @change="syncScenarioBaseStations" />
            <input v-model.number="row.y" type="number" min="0" :max="gridBounds.maxY" @change="syncScenarioBaseStations" />
            <span class="device-state">{{ row.enabled ? row.status || "已接入" : "未启用" }}</span>
            <button type="button" class="danger-link" @click="removeDeviceSlot(row.deviceId)">移除</button>
          </div>
          <p v-if="!hasImportedScene" class="empty-note">
            场景尚未导入，场景就绪后会显示已有基站并允许配置设备接入。
          </p>
          <p v-else-if="!appliedDeviceRows.length" class="empty-note">
            暂无设备接入，当前测试将按无残余网络执行；点击“添加设备”可重新接入 1 台设备。
          </p>
        </div>
      </section>

      <section class="module-panel module-panel--algorithm" :style="{ top: `${algorithmPanelTop}px` }">
        <header class="module-heading">
          <div>
            <i></i>
            <h2>策略算法</h2>
            <p>{{ algorithmPanelStatus }}</p>
          </div>
        </header>
        <div class="algorithm-controls">
          <div class="algorithm-card-grid" aria-label="算法选择">
            <button
              v-for="option in algorithmOptions"
              :key="option.value"
              type="button"
              :class="[
                'algorithm-card',
                {
                  'algorithm-card--active': selectedAlgorithm === option.value,
                  'algorithm-card--disabled': !option.available,
                },
              ]"
              :disabled="isRunning || isLoading || !option.available"
              @click="selectAlgorithmForTest(option)"
            >
              <span class="algorithm-card__name">{{ option.label }}</span>
              <span class="algorithm-card__desc">{{ option.desc }}{{ option.available ? "" : " / 未训练" }}</span>
            </button>
          </div>
          <button type="button" class="primary-button algorithm-start-button" :disabled="startDisabled" @click="runSimulation">
            {{ startButtonText }}
          </button>
        </div>
      </section>

      <section class="module-panel module-panel--result" :style="{ top: `${resultPanelTop}px` }">
        <header class="module-heading">
          <div>
            <i></i>
            <h2>实时终端输出</h2>
          </div>
        </header>

        <div class="result-layout">
          <StreamingTerminal
            title="实时终端输出"
            subtitle="实时输出场景同步、设备配置、模型推理和后端流式测试结果。"
            :lines="strategyTerminalLines"
            :status="terminalStatus"
            placeholder="等待策略测试输出..."
            exportable
            clearable
            @export="downloadTerminalLog"
            @clear="clearTerminalLog"
          />

          <div class="result-box">
            <div class="result-metrics">
              <span><small>平均奖励</small><strong>{{ formatMetric(simulationResult?.avg_reward, 2) }}</strong></span>
              <span><small>平均覆盖率</small><strong>{{ formatPercent(simulationResult?.avg_final_coverage) }}</strong></span>
              <span><small>广播覆盖</small><strong>{{ formatPercent(finalState.broadcast_ratio) }}</strong></span>
              <span><small>剩余预算</small><strong>{{ formatMetric(finalState.remaining_budget, 1) }}</strong></span>
            </div>

            <div v-if="stationRecoverySummary" class="recovery-card">
              <div class="recovery-card__title">
                <strong>原始站点恢复过程</strong>
                <span>保留 {{ formatInteger(stationRecoverySummary.preserved_original_stations) }} 个原始站点</span>
              </div>
              <div class="recovery-card__metrics">
                <span><small>恢复前</small><strong>{{ stationRecoveryStatusText(stationRecoverySummary.before) }}</strong></span>
                <span><small>部署后</small><strong>{{ stationRecoveryStatusText(stationRecoverySummary.after) }}</strong></span>
                <span><small>恢复在线</small><strong>{{ formatInteger(stationRecoverySummary.restored_to_active) }}</strong></span>
                <span><small>部署新增</small><strong>{{ formatInteger(stationRecoverySummary.new_deployments) }}</strong></span>
              </div>
              <div v-if="stationRecoveryEvents.length" class="recovery-card__events">
                <span v-for="event in stationRecoveryEvents" :key="event.station_key">
                  {{ recoveryEventText(event) }}
                </span>
              </div>
            </div>

            <div class="export-card">
              <strong>场景导出</strong>
              <p>受灾场景文件：{{ sceneExport?.disaster_scene_path || "--" }}</p>
              <p>部署后场景文件：{{ sceneExport?.deployment_scene_path || "--" }}</p>
              <p>部署方案文件：{{ sceneExport?.deployment_plan_path || "--" }}</p>
              <div class="module-actions">
                <button type="button" class="ghost-button ghost-button--blue" :disabled="!sceneExport?.disaster_scene" @click="downloadSceneExport('disaster_scene')">
                  下载受灾场景
                </button>
                <button type="button" class="ghost-button ghost-button--blue" :disabled="!sceneExport?.deployment_plan" @click="downloadSceneExport('deployment_plan')">
                  下载部署方案文件
                </button>
                <a class="replay-button" href="#/replay">进行场景回放</a>
              </div>
            </div>

            <div class="result-table">
              <div class="result-table__head">
                <span>ID</span><span>位置 / 区域</span><span>需求</span><span>连接状态</span><span>广播</span>
              </div>
              <div v-for="device in resultDeviceRows" :key="device.id" class="result-table__row">
                <span>{{ device.id }}</span>
                <span>{{ formatDeviceLocation(device) }}</span>
                <span>{{ formatMetric(device.demand, 1) }} Mbps</span>
                <span>{{ device.connected ? "在线" : "离线" }}</span>
                <span>{{ device.broadcast_served ? "已覆盖" : "未覆盖" }}</span>
              </div>
              <p v-if="!resultDeviceRows.length" class="empty-note">暂无测试结果明细。</p>
            </div>
          </div>
        </div>
      </section>
      </div>
    </main>

    <div v-if="historyModalOpen" class="prototype-modal" @click.self="historyModalOpen = false">
      <section class="prototype-dialog prototype-dialog--history">
        <header>
          <h2>策略测试记录</h2>
          <button type="button" @click="historyModalOpen = false"></button>
        </header>
        <table>
          <thead>
            <tr>
              <th>序号</th>
              <th>场景</th>
              <th>算法</th>
              <th>测试时间</th>
              <th>平均奖励</th>
              <th>覆盖率</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(item, index) in historyRows" :key="item.id">
              <td>{{ index + 1 }}</td>
              <td>{{ item.scenarioLabel || item.scenarioName }}</td>
              <td>{{ item.algorithmLabel || item.algorithm }}</td>
              <td>{{ formatDateTime(item.createdAt) }}</td>
              <td>{{ formatMetric(item.avgReward, 2) }}</td>
              <td>{{ formatPercent(item.avgFinalCoverage) }}</td>
            </tr>
            <tr v-if="!historyRows.length">
              <td colspan="6" class="empty-row">暂无真实测试记录，请先执行一次策略测试。</td>
            </tr>
          </tbody>
        </table>
      </section>
    </div>
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from "vue";
import axios from "axios";

import StreamingTerminal from "./StreamingTerminal.vue";
import { buildRegionMetrics, formatDistance } from "../utils/regionMetrics";
import { rescueApiBase } from "../utils/runtimeEndpoints";
import { saveReplaySessionFromSimulation, setActiveReplaySessionId } from "../utils/replaySessions";
import {
  appendSyncedTerminalLine,
  clearTerminalOutput,
  exportTerminalOutput,
  terminalHistoryLines,
} from "../utils/terminalOutput";
import {
  compareDisasterScenarioOrder,
  formatPlainDisasterName,
  formatScenarioName,
  preferredDisasterSeverityKey,
} from "../utils/scenarioLabels";
import {
  buildUserNodeCountMessage,
  userNodeCountLogKey,
} from "../utils/scenarioNodeMetrics";

const API_BASE = rescueApiBase;
const TEST_HISTORY_KEY = "prototype-tester-history";
const DEVICE_LIBRARY_KEY = "prototype-tester-device-library-v1";
const DEVICE_BINDINGS_KEY = "prototype-tester-device-bindings-v1";
const SCENE_ACCESS_TIMEOUT_MS = 45000;

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

const COMMUNICATION_TYPE_OPTIONS = [
  { value: "cellular", label: "蜂窝通信" },
  { value: "wifi", label: "WiFi 通信" },
  { value: "satellite", label: "卫星通信" },
  { value: "shortwave", label: "短波通信" },
  { value: "mesh", label: "Mesh/UAV" },
  { value: "custom", label: "专用通信" },
];

const DISASTER_COMMUNICATION_PROFILES = {
  cellular_5g_700mhz: {
    category: "cellular",
    label: "5G 700MHz 蜂窝",
    mode: "5G_700MHz",
  },
  satellite_ka: {
    category: "satellite",
    label: "Ka 卫星",
    mode: "Satellite_Ka",
  },
  wifi6_mesh: {
    category: "wifi",
    label: "WiFi6 Mesh",
    mode: "WiFi6",
  },
  shortwave_hf: {
    category: "shortwave",
    label: "短波 HF",
    mode: "Shortwave_HF",
  },
};

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
const DEVICE_DISPLAY_TEXT_REPLACEMENTS = [
  ["5G 700MHz 应急小区", "5G 700MHz应急基站"],
  ["5G 700MHz应急小区", "5G 700MHz应急基站"],
  ["5G应急小区", "5G应急基站"],
];

const STATION_STATUS_LABELS = {
  active: "活跃",
  degraded: "降级",
  offline: "离线",
  residual: "保留",
  deployed: "新部署",
  planned: "计划部署",
  unknown: "未知",
};

const STATION_STATUS_LEGEND = [
  { key: "active", label: STATION_STATUS_LABELS.active },
  { key: "degraded", label: STATION_STATUS_LABELS.degraded },
  { key: "offline", label: STATION_STATUS_LABELS.offline },
  { key: "unknown", label: STATION_STATUS_LABELS.unknown },
];

const DEFAULT_DEVICE_TEMPLATES = [
  {
    id: "tester-cellular-macro",
    name: "测试蜂窝宏站",
    deviceType: "宏基站",
    communicationType: "cellular",
    quantity: 1,
    maxThroughput: 240,
    maxUsers: 180,
    enabled: true,
    status: "已导入",
  },
  {
    id: "tester-wifi-hotspot",
    name: "测试 WiFi6 热点",
    deviceType: "背负式基站",
    communicationType: "wifi",
    quantity: 1,
    maxThroughput: 160,
    maxUsers: 96,
    enabled: true,
    status: "已导入",
  },
  {
    id: "tester-satellite-relay",
    name: "测试卫星中继",
    deviceType: "中继设备",
    communicationType: "satellite",
    quantity: 1,
    maxThroughput: 150,
    maxUsers: 120,
    enabled: true,
    status: "已导入",
  },
  {
    id: "tester-shortwave-station",
    name: "测试短波台",
    deviceType: "临时设备/车载设备",
    communicationType: "shortwave",
    quantity: 1,
    maxThroughput: 24,
    maxUsers: 220,
    enabled: true,
    status: "已导入",
  },
];

const algorithms = [
  { value: "ppo", label: "基于 PPO 的覆盖恢复策略优化方案", desc: "覆盖优先 / 稳定基线" },
  { value: "dqn", label: "基于 DQN 的离散站点部署决策方案", desc: "离散动作 / 快速推演" },
  { value: "a3c", label: "基于 A3C 的多目标协同训练方案", desc: "异步更新 / 多目标" },
  { value: "mppo", label: "基于 MPPO 的多头策略组网方案", desc: "多头策略 / 资源协同" },
  { value: "hmarl", label: "层次化多智能体通信资源配置与组网方案", desc: "自研方案 / 分层协同" },
];

const fallbackScenarios = [
  { name: "flood_no_residual", disaster_type: "flood", grid_size: 22 },
  { name: "typhoon_residual", disaster_type: "typhoon", grid_size: 22 },
  { name: "earthquake_residual", disaster_type: "earthquake", grid_size: 22 },
];

const MAP_LAYER = {
  left: 1,
  top: 53,
  width: 1618,
  height: 745,
};
const BASEMAP_SHIFT_RATIOS = {
  rainstorm: 0.16,
  typhoon: 0.42,
};
const STATION_TOOLTIP_WIDTH = 248;
const STATION_TOOLTIP_HEIGHT = 164;
const STATION_TOOLTIP_GAP = 16;

const MARKER_AREA = {
  left: 91,
  top: 113,
  width: 1438,
  height: 610,
};

const SCENARIO_PANEL_TOP = 84;
const SCENARIO_PANEL_HEIGHT = 920;
const MAP_PANEL_GAP = 12;
const MAP_PANEL_TOP = SCENARIO_PANEL_TOP + SCENARIO_PANEL_HEIGHT + MAP_PANEL_GAP;
const MAP_PANEL_HEIGHT = 798;
const DEVICE_PANEL_TOP = MAP_PANEL_TOP + MAP_PANEL_HEIGHT + 30;
const ALGORITHM_PANEL_HEIGHT = 178;
const RESULT_PANEL_GAP = 22;

const STATION_COLORS = [
  "#2563eb",
  "#16a34a",
  "#f59e0b",
  "#7c3aed",
  "#0891b2",
  "#dc2626",
  "#475569",
];

const STATION_TYPE_COLORS = {
  backpack_micro_cell: "#f59e0b",
  low_band_macro_cell: "#2563eb",
  temporary_macro_cell: "#06b6d4",
  fixed_satellite_gateway: "#7c3aed",
  vehicle_satellite_terminal: "#0891b2",
  command_vehicle_radio: "#dc2626",
  field_shortwave_station: "#16a34a",
  portable_hotspot: "#f97316",
  shelter_mesh_node: "#14b8a6",
  vehicle_wifi_node: "#84cc16",
};

const USER_MARKER_COLORS = {
  offline: "#ef4444",
  online: "#38bdf8",
};
const USER_SCATTER_SPREAD = 1.18;
const USER_MARKER_SIZE_MIN = 3.4;
const USER_MARKER_SIZE_MAX = 4.7;

const STATION_STATUS_COLORS = {
  active: "#22c55e",
  degraded: "#f59e0b",
  offline: "#64748b",
  residual: "#2563eb",
  deployed: "#06b6d4",
  planned: "#60a5fa",
  unknown: "#94a3b8",
};

const scenarios = ref([]);
const trainingArtifacts = ref([]);
const scenarioName = ref("flood_no_residual");
const selectedAlgorithm = ref("ppo");
const checkpointPath = ref("");
const importedScene = ref(null);
const simulationResult = ref(null);
const activeSceneTab = ref("imported");
const isLoading = ref(false);
const isRunning = ref(false);
const isClearingResidualNetwork = ref(false);
const statusMessage = ref("");
const statusTone = ref("info");
const terminalLines = ref([]);
const terminalStatus = ref("idle");
const strategyTerminalLines = computed(() => terminalHistoryLines.value.slice(-500));
const historyModalOpen = ref(false);
const historyRows = ref([]);
const deviceRows = ref([]);
const disasterScenarios = ref([]);
const disasterScenarioDetails = ref({});
const disasterSeverityOverview = ref(null);
const disasterImports = ref([]);
const disasterImportDetails = ref({});
const selectedDisasterScenario = ref("");
const selectedDisasterSeverity = ref("level_4");
const selectedDisasterImportId = ref("");
const activeDisasterImportId = ref("");
const disasterSessionSampleLimit = ref(100);
const disasterLoading = ref(false);
const disasterImporting = ref(false);
const disasterImportProgress = ref(0);
const disasterImportStage = ref("准备导入");
const disasterImportProgressTone = ref("idle");
const disasterError = ref("");
const stationTooltip = ref({
  visible: false,
  marker: null,
});
let statusTimer = null;
let disasterImportProgressTimer = null;
let disasterImportProgressHideTimer = null;
let lastStrategyUserNodeLogKey = "";

const EXTREME_DISASTER_USER_COUNTS = {
  extreme_rainstorm: 3500,
  super_typhoon: 3200,
  destructive_earthquake: 3900,
};

const FALLBACK_DISASTER_SCENARIOS = [
  {
    scenario: "extreme_rainstorm",
    display_name: "超强暴雨",
    disaster_type: "rainstorm",
    num_users: EXTREME_DISASTER_USER_COUNTS.extreme_rainstorm,
    unique_user_count: EXTREME_DISASTER_USER_COUNTS.extreme_rainstorm,
    severity_levels: ["level_1", "level_2", "level_3", "level_4"],
  },
  {
    scenario: "super_typhoon",
    display_name: "特大台风",
    disaster_type: "typhoon",
    num_users: EXTREME_DISASTER_USER_COUNTS.super_typhoon,
    unique_user_count: EXTREME_DISASTER_USER_COUNTS.super_typhoon,
    severity_levels: ["level_1", "level_2", "level_3", "level_4"],
  },
  {
    scenario: "destructive_earthquake",
    display_name: "强破坏地震",
    disaster_type: "earthquake",
    num_users: EXTREME_DISASTER_USER_COUNTS.destructive_earthquake,
    unique_user_count: EXTREME_DISASTER_USER_COUNTS.destructive_earthquake,
    severity_levels: ["level_1", "level_2", "level_3", "level_4"],
  },
];

const assetUrl = (path) => `${import.meta.env.BASE_URL}prototype/${path}`;

const currentScenario = computed(
  () => scenarios.value.find((scenario) => scenario.name === scenarioName.value) || scenarios.value[0] || fallbackScenarios[0]
);

const disasterLabel = (type) => {
  const labels = {
    flood: "暴雨",
    rainstorm: "暴雨",
    earthquake: "地震",
    landslide: "泥石流",
    typhoon: "台风",
  };
  return labels[type] || formatScenarioName(type);
};

const scenarioLabel = (scenario) => {
  if (!scenario) return formatScenarioName(scenarioName.value);
  return formatPlainDisasterName(
    scenario.disaster_type,
    scenario.source_scenario,
    scenario.display_name,
    scenario.name
  ) || disasterLabel(scenario.disaster_type) || formatScenarioName(scenario.name);
};

const algorithmLabel = (value) => algorithms.find((item) => item.value === value)?.label || value?.toUpperCase() || "--";

const comboText = computed(() => `${scenarioLabel(currentScenario.value)} + ${algorithmLabel(selectedAlgorithm.value)}`);

const regionGrid = computed(() => currentScenario.value?.region_grid || null);
const regionMetrics = computed(() => buildRegionMetrics(regionGrid.value));

const regionText = computed(() => {
  const grid = regionGrid.value;
  const name = grid?.name || scenarioLabel(currentScenario.value);
  const rows = grid?.rows || currentScenario.value?.grid_size || 22;
  const cols = grid?.cols || currentScenario.value?.grid_size || 22;
  return `区域：${name}（离散网格 ${rows} × ${cols}）`;
});

const spanText = computed(() => {
  if (!regionMetrics.value) {
    return "实际跨度：约 67.1 km × 66.7 km 单网格约 3.05 km × 3.03 km";
  }
  return `实际跨度：约 ${formatDistance(regionMetrics.value.widthKm)} × ${formatDistance(
    regionMetrics.value.heightKm
  )} 单网格约 ${formatDistance(regionMetrics.value.cellWidthKm)} × ${formatDistance(regionMetrics.value.cellHeightKm)}`;
});

const matchingCheckpoint = computed(() =>
  trainingArtifacts.value.find(
    (artifact) =>
      artifact.scenario_name === scenarioName.value &&
      artifact.algorithm === selectedAlgorithm.value &&
      artifact.checkpoint_path
  )
);

const algorithmOptions = computed(() =>
  algorithms.map((algorithm) => {
    const artifact = trainingArtifacts.value.find(
      (item) => item.scenario_name === scenarioName.value && item.algorithm === algorithm.value && item.checkpoint_path
    );
    return {
      ...algorithm,
      available: Boolean(artifact?.checkpoint_path),
      checkpointPath: artifact?.checkpoint_path || "",
    };
  })
);

const activeScene = computed(() => {
  if (activeSceneTab.value === "deployment" && simulationResult.value?.scene_export?.deployment_scene) {
    return simulationResult.value.scene_export.deployment_scene;
  }
  return importedScene.value?.scene || null;
});

const hasImportedScene = computed(() => Boolean(importedScene.value?.scene));

const rawNodes = computed(() => activeScene.value?.nodes || []);

const sceneExport = computed(() => simulationResult.value?.scene_export || null);
const finalReport = computed(() => (Array.isArray(simulationResult.value?.reports) ? simulationResult.value.reports[0] : null));
const finalState = computed(() => finalReport.value?.final_state || {});
const stationRecoverySummary = computed(
  () => finalReport.value?.station_recovery_summary || sceneExport.value?.deployment_scene?.station_recovery_summary || null
);
const stationRecoveryEvents = computed(() =>
  (Array.isArray(stationRecoverySummary.value?.events) ? stationRecoverySummary.value.events : []).slice(0, 6)
);
const resultDeviceRows = computed(() => (Array.isArray(finalState.value.user_details) ? finalState.value.user_details.slice(0, 10) : []));
const summary = computed(() => {
  if (!activeDisasterImportId.value || !hasImportedScene.value) {
    return {
      nodes: 0,
      users: 0,
      stations: 0,
    };
  }
  const initialState = importedScene.value?.initial_state || {};
  const importedUsers = Number(activeDisasterImport.value?.unique_user_count);
  const users =
    (Number.isFinite(importedUsers) && importedUsers > 0 ? importedUsers : 0) ||
    Number(initialState.total_users) ||
    rawNodes.value.filter((node) => node.type === "USER").length ||
    0;
  const stationNodes = rawNodes.value.filter(isStationNode).length;
  const sceneStations =
    stationNodes ||
    initialState.residual_base_stations?.length ||
    activeAppliedDeviceRows.value.length ||
    0;
  const importedStations = Number(activeDisasterImport.value?.station_counts?.total);
  const stations =
    sceneStations ||
    (Number.isFinite(importedStations) && importedStations > 0 ? importedStations : 0) ||
    activeDisasterDeployments.value.length ||
    0;
  return {
    nodes: users + stations,
    users,
    stations,
  };
});

const disasterScenarioOptions = computed(() =>
  disasterScenarios.value
    .map((scenario) => ({
      key: scenario.scenario || scenario.name || scenario.disaster_scenario || "",
      label: formatPlainDisasterName(
        scenario.disaster_type,
        scenario.type,
        scenario.scenario,
        scenario.name,
        scenario.disaster_scenario,
        scenario.display_name,
        scenario.label,
        scenario.disaster_scenario_label
      ) || scenario.display_name || scenario.label || scenario.disaster_scenario_label || scenario.scenario || scenario.name || "未选择",
      raw: scenario,
    }))
    .filter((item) => item.key)
    .sort((left, right) =>
      compareDisasterScenarioOrder(
        [left.label, left.key, left.raw?.disaster_type, left.raw?.type, left.raw?.display_name],
        [right.label, right.key, right.raw?.disaster_type, right.raw?.type, right.raw?.display_name]
      )
    )
);

const currentDisasterScenarioOption = computed(
  () => disasterScenarioOptions.value.find((item) => item.key === selectedDisasterScenario.value) || null
);

const currentDisasterScenarioDetail = computed(() => disasterScenarioDetails.value[selectedDisasterScenario.value] || null);

const disasterSeverityOptions = computed(() => {
  const detail = currentDisasterScenarioDetail.value;
  const levels = detail?.severity_levels ?? currentDisasterScenarioOption.value?.raw?.severity_levels ?? [];
  if (Array.isArray(levels)) {
    return levels.map((key) => ({ key, label: DISASTER_SEVERITY_LABELS[key] || key, meta: {} }));
  }
  return Object.entries(levels || {}).map(([key, meta]) => ({
    key,
    label: meta?.label || DISASTER_SEVERITY_LABELS[key] || key,
    meta: meta || {},
  }));
});

const stationStatusLegend = STATION_STATUS_LEGEND;

const stationStatusLabel = (status) => STATION_STATUS_LABELS[String(status || "unknown")] || String(status || "未知");

const disasterScenarioCardDescription = (option) => {
  const raw = disasterScenarioDetails.value[option?.key] || option?.raw || {};
  const characteristics = Array.isArray(raw.characteristics) ? raw.characteristics : [];
  if (characteristics.length) return characteristics[0];
  const coverage = Number(raw.coverage_area_km2 || raw.coverage_area);
  if (Number.isFinite(coverage) && coverage > 0) return `覆盖面积 ${formatMetric(coverage, 1)} km²`;
  return raw.disaster_type || raw.type || "灾害数据场景";
};

const disasterSeverityCardDescription = (option) => {
  const damage = Number(option?.meta?.damage_rate);
  const offline = Number(option?.meta?.offline_rate);
  const parts = [];
  if (Number.isFinite(damage)) parts.push(`损毁 ${formatPercent(damage)}`);
  if (Number.isFinite(offline)) parts.push(`离线 ${formatPercent(offline)}`);
  return parts.join(" / ") || "等待同步参数";
};

const preferredDisasterSeverityOptionKey = () =>
  preferredDisasterSeverityKey(disasterSeverityOptions.value.map((item) => ({ key: item.key, label: item.label })));

const selectDisasterScenarioCard = async (key) => {
  if (disasterImporting.value || !key || key === selectedDisasterScenario.value) return;
  selectedDisasterScenario.value = key;
  await handleDisasterScenarioChange();
};

const selectDisasterSeverityCard = async (key) => {
  if (disasterImporting.value || !key || key === selectedDisasterSeverity.value) return;
  selectedDisasterSeverity.value = key;
  await handleDisasterSeverityChange();
};

const selectedDisasterImport = computed(() => {
  if (!selectedDisasterImportId.value) return null;
  return (
    disasterImportDetails.value[selectedDisasterImportId.value] ||
    disasterImports.value.find((item) => item.import_id === selectedDisasterImportId.value) ||
    null
  );
});

const selectedDisasterImportDetail = computed(() =>
  selectedDisasterImportId.value ? disasterImportDetails.value[selectedDisasterImportId.value] || null : null
);

const activeDisasterImport = computed(() => {
  if (!activeDisasterImportId.value) return null;
  return (
    disasterImportDetails.value[activeDisasterImportId.value] ||
    disasterImports.value.find((item) => item.import_id === activeDisasterImportId.value) ||
    null
  );
});

const scenarioNameFromImportRecord = (record) =>
  record?.disaster_scenario && record?.disaster_severity ? `${record.disaster_scenario}__${record.disaster_severity}` : "";

const countDeployments = (deployments) =>
  (Array.isArray(deployments) ? deployments : []).reduce(
    (sum, deployment) => sum + Math.max(1, Number(deployment?.quantity || 1)),
    0
  );

const scenarioForName = (name) => scenarios.value.find((item) => item.name === name) || null;

const scenarioDeploymentsForName = (name) => {
  const scenario = scenarioForName(name);
  return Array.isArray(scenario?.base_station_deployments)
    ? scenario.base_station_deployments
    : Array.isArray(scenario?.residual_base_stations)
      ? scenario.residual_base_stations
      : [];
};

const scenarioDeviceCountForName = (name) => countDeployments(scenarioDeploymentsForName(name));

const deploymentStatusCounts = (deployments, fallback = {}) => {
  const counts = { total: 0, active: 0, degraded: 0, offline: 0, planned: 0, ...fallback };
  if (!Array.isArray(deployments) || !deployments.length) return counts;
  const nextCounts = { total: 0, active: 0, degraded: 0, offline: 0, planned: 0 };
  deployments.forEach((deployment) => {
    const quantity = Math.max(1, Number(deployment?.quantity || 1));
    const status = String(deployment?.status || "active");
    nextCounts.total += quantity;
    nextCounts[status] = (nextCounts[status] || 0) + quantity;
  });
  return nextCounts;
};

const currentStationCountForImportRecord = (record) => {
  const importScenarioName = scenarioNameFromImportRecord(record);
  const scenarioCount = scenarioDeviceCountForName(importScenarioName);
  if (importScenarioName && importScenarioName === scenarioName.value) {
    return activeSimulationStationCount.value || scenarioCount || Number(record?.station_counts?.total || 0);
  }
  return scenarioCount || Number(record?.station_counts?.total || 0);
};

const stationCountsForImportRecord = (record) => {
  const importScenarioName = scenarioNameFromImportRecord(record);
  const deployments = scenarioDeploymentsForName(importScenarioName);
  return deploymentStatusCounts(deployments, record?.station_counts || {});
};

const importRecordStatsText = (record) => {
  const stationCount = currentStationCountForImportRecord(record);
  const userCount = Number(record?.unique_user_count || 0);
  return `${stationCount}站点 · ${userCount}用户`;
};

const stationRecoveryStatusText = (counts = {}) =>
  `在线 ${formatInteger(counts.active || 0)} / 降级 ${formatInteger(counts.degraded || 0)} / 离线 ${formatInteger(counts.offline || 0)}`;

const recoveryEventText = (event = {}) => {
  const grid = event.grid || {};
  const from = STATION_STATUS_LABELS[event.from_status] || event.from_status || "--";
  const to = STATION_STATUS_LABELS[event.to_status] || event.to_status || "--";
  const step = event.recovery_step == null ? "--" : event.recovery_step;
  return `${event.label || "原始基站"} (${grid.row ?? "--"}, ${grid.col ?? "--"}) ${from} -> ${to} / step=${step}`;
};

const recoverySummaryFromResult = (result) =>
  (Array.isArray(result?.reports) ? result.reports[0]?.station_recovery_summary : null) ||
  result?.scene_export?.deployment_scene?.station_recovery_summary ||
  null;

const appendStationRecoveryTerminal = (summary) => {
  if (!summary?.after?.total) return;
  appendTerminalEvent(
    `原始站点保留并恢复：${formatInteger(summary.preserved_original_stations)} 个；恢复前 ${stationRecoveryStatusText(summary.before)}；部署后 ${stationRecoveryStatusText(summary.after)}；在线率 ${formatPercent(summary.online_ratio_after)}。`,
    { level: "RECOVERY", source: "BACKEND" }
  );
  const restored = Number(summary.restored_to_active || 0);
  const partial = Number(summary.partially_recovered || 0);
  if (restored || partial) {
    appendTerminalEvent(`恢复过程：${formatInteger(restored)} 个原始基站恢复在线，${formatInteger(partial)} 个离线基站恢复为降级可用。`, {
      level: "RECOVERY",
      source: "BACKEND",
    });
  }
};

const selectedImportStationCount = computed(() => currentStationCountForImportRecord(selectedDisasterImport.value) || "--");

const activeSceneStationCount = computed(() =>
  rawNodes.value.filter(isStationNode).length || importedScene.value?.initial_state?.residual_base_stations?.length || 0
);

const activeSimulationStationCount = computed(
  () => activeSceneStationCount.value || countDeployments(activeAppliedDeviceRows.value) || scenarioDeviceCountForName(scenarioName.value)
);

const hasActiveDisasterImport = computed(() => Boolean(activeDisasterImportId.value));

const mapDataReady = computed(() => Boolean(hasImportedScene.value && hasActiveDisasterImport.value));

const mapEmptyVisible = computed(() => isLoading.value || !mapDataReady.value);

const mapEmptyTitle = computed(() => (isLoading.value ? "正在同步导入数据" : "暂无导入数据"));

const mapEmptyDescription = computed(() =>
  isLoading.value ? "同步完成后显示受灾节点和基站状态。" : "导入灾害数据并用于仿真后显示地图节点。"
);

const activeDisasterImportDetail = computed(() =>
  activeDisasterImportId.value ? disasterImportDetails.value[activeDisasterImportId.value] || null : null
);

const currentDisasterSeverityMeta = computed(
  () => disasterSeverityOptions.value.find((item) => item.key === selectedDisasterSeverity.value)?.meta || {}
);

const disasterPreviewTitle = computed(() => {
  const scenario = currentDisasterScenarioDetail.value || currentDisasterScenarioOption.value?.raw || {};
  const scenarioText = formatPlainDisasterName(
    scenario.disaster_type,
    scenario.type,
    scenario.scenario,
    scenario.name,
    scenario.disaster_scenario,
    scenario.display_name,
    scenario.label,
    selectedDisasterScenario.value
  ) || scenario.display_name || scenario.label || scenario.scenario || scenario.name || selectedDisasterScenario.value || "灾害场景";
  const severityText =
    disasterSeverityOverview.value?.severity_label ||
    currentDisasterSeverityMeta.value?.label ||
    DISASTER_SEVERITY_LABELS[selectedDisasterSeverity.value] ||
    selectedDisasterSeverity.value ||
    "受灾等级";
  return `${scenarioText} / ${severityText}`;
});

const disasterPreviewDescription = computed(() => {
  const scenario = currentDisasterScenarioDetail.value || currentDisasterScenarioOption.value?.raw || {};
  const characteristics = Array.isArray(scenario.characteristics) ? scenario.characteristics : [];
  return characteristics.slice(0, 2).join(" ") || "暂无场景特征说明。";
});

const activeDisasterBounds = computed(
  () =>
    selectedDisasterImport.value?.effective_geo_bounds ||
    disasterSeverityOverview.value?.effective_geo_bounds ||
    currentDisasterScenarioDetail.value?.effective_geo_bounds ||
    currentDisasterScenarioOption.value?.raw?.effective_geo_bounds ||
    null
);

const simulationDisasterBounds = computed(
  () => activeDisasterImport.value?.effective_geo_bounds || activeDisasterImportDetail.value?.effective_geo_bounds || null
);

const disasterGrid = computed(() => {
  const grid = selectedDisasterImport.value?.grid_size || currentDisasterScenarioDetail.value?.grid_size || { rows: 24, cols: 24 };
  if (typeof grid === "number") return { rows: grid, cols: grid };
  return {
    rows: Number(grid?.rows) || 24,
    cols: Number(grid?.cols) || 24,
  };
});

const activeDisasterMetrics = computed(() =>
  buildRegionMetrics({
    geo_bounds: activeDisasterBounds.value,
    rows: disasterGrid.value.rows,
    cols: disasterGrid.value.cols,
  })
);

const disasterCoverageArea = computed(
  () => currentDisasterScenarioDetail.value?.coverage_area_km2 || currentDisasterScenarioOption.value?.raw?.coverage_area_km2 || null
);

const disasterDamageRate = computed(
  () => Number(disasterSeverityOverview.value?.damage_rate ?? currentDisasterSeverityMeta.value?.damage_rate)
);

const disasterOfflineRate = computed(
  () => Number(disasterSeverityOverview.value?.offline_rate ?? currentDisasterSeverityMeta.value?.offline_rate)
);

const disasterHeatmap = computed(() =>
  Array.isArray(selectedDisasterImportDetail.value?.user_heatmap) ? selectedDisasterImportDetail.value.user_heatmap : []
);

const scenarioDeploymentAsDisasterStation = (deployment, index, scenario) => {
  const baseStationName = deployment?.base_station || deployment?.baseStationName || "";
  const station = (Array.isArray(scenario?.base_stations) ? scenario.base_stations : []).find((item) => item.name === baseStationName);
  const position = deployment?.grid_position || {};
  const x = Math.round(Number(deployment?.x ?? position.row ?? 0));
  const y = Math.round(Number(deployment?.y ?? position.col ?? 0));
  const mode = deployment?.mode || deployment?.comm_type || (Array.isArray(station?.supported_modes) ? station.supported_modes[0] : null);
  const deviceUid = deployment?.device_uid || deployment?.deployment_id || `${scenario?.name || scenarioName.value}:device:${index}`;
  const stationType = deployment?.station_type || baseStationName || deployment?.device_type || "station";
  const stationLabel =
    deployment?.station_label ||
    deployment?.label ||
    deployment?.device_name ||
    station?.label ||
    disasterStationTypeLabel(stationType);
  return {
    ...deployment,
    device_uid: deviceUid,
    deployment_id: deployment?.deployment_id || deviceUid,
    base_station: baseStationName,
    station_type: stationType,
    station_label: stationLabel,
    comm_type: deployment?.comm_type || mode,
    comm_label: deployment?.comm_label || deployment?.mode_label || communicationModeLabel(mode),
    mode,
    status: deployment?.status || "active",
    grid_position: { row: x, col: y },
    cell_user_count: Number(deployment?.cell_user_count ?? deployment?.max_users ?? deployment?.connected_users ?? station?.max_users ?? 0),
    coverage_radius_km: Number(deployment?.coverage_radius_km ?? deployment?.source_coverage_radius_km ?? deployment?.coverageRadiusKm ?? 0),
    downlink_bandwidth_mbps_avg: Number(deployment?.downlink_bandwidth_mbps ?? deployment?.max_throughput ?? station?.max_throughput ?? 0),
  };
};

const currentDisasterDeploymentsForRecord = (record, detail) => {
  const importScenarioName = scenarioNameFromImportRecord(record);
  const scenario = scenarioForName(importScenarioName);
  const deployments = scenarioDeploymentsForName(importScenarioName);
  if (deployments.length) {
    return deployments.map((deployment, index) => scenarioDeploymentAsDisasterStation(deployment, index, scenario));
  }
  return Array.isArray(detail?.deployments) ? detail.deployments : [];
};

const disasterDeployments = computed(() =>
  currentDisasterDeploymentsForRecord(selectedDisasterImport.value, selectedDisasterImportDetail.value)
);

const disasterStationCounts = computed(() => stationCountsForImportRecord(selectedDisasterImport.value));

const activeDisasterDeployments = computed(() =>
  currentDisasterDeploymentsForRecord(activeDisasterImport.value, activeDisasterImportDetail.value)
);

const activeDisasterHeatmap = computed(() =>
  Array.isArray(activeDisasterImportDetail.value?.user_heatmap) ? activeDisasterImportDetail.value.user_heatmap : []
);

const normalizeGridSize = (grid, fallback = { rows: 24, cols: 24 }) => {
  if (typeof grid === "number") return { rows: grid, cols: grid };
  return {
    rows: Number(grid?.rows) || Number(fallback?.rows) || 24,
    cols: Number(grid?.cols) || Number(fallback?.cols) || 24,
  };
};

const activeDisasterGrid = computed(() =>
  normalizeGridSize(activeDisasterImport.value?.grid_size || activeDisasterImportDetail.value?.grid_size, disasterGrid.value)
);

const activeDatasetImportIds = computed(() => (activeDisasterImportId.value ? [activeDisasterImportId.value] : []));

const gridBounds = computed(() => {
  const rows = regionGrid.value?.rows || currentScenario.value?.grid_size || 22;
  const cols = regionGrid.value?.cols || currentScenario.value?.grid_size || 22;
  return {
    maxX: Math.max(0, Number(rows) - 1),
    maxY: Math.max(0, Number(cols) - 1),
  };
});

const activeAppliedDeviceRows = computed(() =>
  deviceRows.value.filter((row) => row.applied && row.enabled !== false && Number(row.quantity) > 0)
);

const appliedDeviceRows = computed(() => {
  if (!hasImportedScene.value) return [];
  return activeAppliedDeviceRows.value;
});

const activeDeviceSummaryRows = computed(() => {
  if (!hasImportedScene.value) return [];
  return deviceRows.value.filter((row) => row.applied && row.enabled !== false && Number(row.quantity) > 0);
});

const displayDeviceText = (value) => {
  if (value === null || value === undefined) return "";
  return DEVICE_DISPLAY_TEXT_REPLACEMENTS.reduce(
    (text, [source, target]) => text.replaceAll(source, target),
    String(value)
  );
};

const deviceOptionLabel = (row) => displayDeviceText(row?.name || row?.stationLabel || row?.deviceType || "设备");

const scenarioPanelHeight = computed(() => SCENARIO_PANEL_HEIGHT);
const mapPanelTop = computed(() => MAP_PANEL_TOP);
const devicePanelTop = computed(() => DEVICE_PANEL_TOP);
const devicePanelHeight = computed(() => Math.max(420, 210 + Math.min(appliedDeviceRows.value.length, 6) * 52));
const algorithmPanelTop = computed(() => devicePanelTop.value + devicePanelHeight.value + RESULT_PANEL_GAP);
const resultPanelTop = computed(() => algorithmPanelTop.value + ALGORITHM_PANEL_HEIGHT + RESULT_PANEL_GAP);
const strategyPanelHeight = computed(() => Math.max(2200, resultPanelTop.value + 520));

const startDisabled = computed(() => isRunning.value || isLoading.value || !hasActiveDisasterImport.value || !matchingCheckpoint.value);

const algorithmPanelStatus = computed(() => {
  if (!hasActiveDisasterImport.value) return "请先导入灾害数据并用于仿真。";
  if (!algorithmOptions.value.some((option) => option.available)) return "当前场景暂无已训练模型。";
  if (!matchingCheckpoint.value) return "请选择已训练的算法模型。";
  return `已匹配模型：${checkpointPath.value || matchingCheckpoint.value.checkpoint_path}`;
});

const deviceSummaryLabel = computed(() => {
  if (!activeDeviceSummaryRows.value.length) return "设备接入：无残余网络";
  const grouped = new Map();
  activeDeviceSummaryRows.value.forEach((row) => {
    const name = row.name || communicationCategoryLabel(row.communicationType);
    grouped.set(name, (grouped.get(name) || 0) + Math.max(1, Number(row.quantity || 1)));
  });
  const entries = [...grouped.entries()];
  const text = entries.slice(0, 4).map(([name, count]) => `${name} ${count} 台`).join(" / ");
  const suffix = entries.length > 4 ? ` / 另 ${entries.length - 4} 类` : "";
  return `设备接入：${text}${suffix}`;
});

const isStationNode = (node) => {
  if (!node) return false;
  const type = String(node.type || node.visual_type || "").toUpperCase();
  if (type === "USER") return false;
  return Boolean(type || node.node_role === "residual_base_station" || node.base_station || node.device_uid || node.deployment_id);
};

const stationLabelByName = (name) => {
  const station = scenarioBaseStations().find((item) => item.name === name);
  return displayDeviceText(station?.label || name || "基站");
};

const stationColorForName = (name) => {
  const stations = scenarioBaseStations();
  const index = stations.findIndex((station) => station.name === name);
  if (index >= 0) return STATION_COLORS[index % STATION_COLORS.length];
  const text = String(name || "");
  const hash = [...text].reduce((value, char) => (value * 31 + char.charCodeAt(0)) % 9973, 0);
  return STATION_COLORS[hash % STATION_COLORS.length];
};

const stationTypeColor = (key, label) => STATION_TYPE_COLORS[String(key || "")] || stationColorForName(key || label);

const stationTypeInfo = ({ key, label, baseStationName, fallback = "基站" } = {}) => {
  const nextKey = String(key || baseStationName || label || fallback);
  const nextLabel =
    label ||
    (baseStationName ? stationLabelByName(baseStationName) : "") ||
    DISASTER_STATION_TYPE_LABELS[nextKey] ||
    nextKey ||
    fallback;
  return {
    key: nextKey,
    label: nextLabel,
    color: stationTypeColor(nextKey, nextLabel),
  };
};

const stableHash = (value) => {
  const text = String(value ?? "");
  let hash = 2166136261;
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  hash += hash << 13;
  hash ^= hash >>> 7;
  hash += hash << 3;
  hash ^= hash >>> 17;
  hash += hash << 5;
  return hash >>> 0;
};

const seededUnit = (seed, salt = 0) => stableHash(`${seed}:${salt}`) / 4294967296;

const clampNumber = (value, min, max) => Math.max(min, Math.min(max, value));

const softBoundNumber = (value, min, max, seed, salt, band) => {
  const number = Number(value);
  if (!Number.isFinite(number)) return min;
  if (max <= min) return min;
  if (number >= min && number <= max) return number;
  const safeBand = Math.min(Math.max(0, band), (max - min) / 2);
  const edgeOffset = safeBand * (0.25 + seededUnit(seed, salt) * 0.75);
  return number < min ? min + edgeOffset : max - edgeOffset;
};

const softBoundGridCoord = (value, max, seed, salt) => {
  const safeMax = Math.max(1, Number(max) || 1);
  const edgeInset = Math.min(0.08, safeMax / 4);
  const band = Math.min(0.78, Math.max(0.18, safeMax * 0.035));
  return softBoundNumber(value, edgeInset, safeMax - edgeInset, seed, salt, band);
};

const scatteredOffset = (seed, spread) => {
  const angle = seededUnit(seed, 1) * Math.PI * 2;
  const radius = Math.sqrt(seededUnit(seed, 2)) * spread;
  const driftAngle = seededUnit(seed, 3) * Math.PI * 2;
  const drift = (seededUnit(seed, 4) - 0.5) * 0.18;
  return {
    row: Math.sin(angle) * radius + Math.sin(driftAngle) * drift,
    col: Math.cos(angle) * radius + Math.cos(driftAngle) * drift,
  };
};

const userMarkerSize = (seed) => {
  const size = USER_MARKER_SIZE_MIN + seededUnit(seed, 5) * (USER_MARKER_SIZE_MAX - USER_MARKER_SIZE_MIN);
  return `${size.toFixed(2)}px`;
};

const quantileNumber = (values, ratio) => {
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!sorted.length) return null;
  const index = (sorted.length - 1) * ratio;
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  if (lower === upper) return sorted[lower];
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (index - lower);
};

const markerEnvelope = (markers) => {
  if (!Array.isArray(markers) || markers.length < 20) return null;
  const left = quantileNumber(markers.map((marker) => Number(marker.x)), 0.02);
  const right = quantileNumber(markers.map((marker) => Number(marker.x)), 0.98);
  const top = quantileNumber(markers.map((marker) => Number(marker.y)), 0.02);
  const bottom = quantileNumber(markers.map((marker) => Number(marker.y)), 0.98);
  if (![left, right, top, bottom].every(Number.isFinite)) return null;
  const padX = Math.max(MAP_LAYER.width * 0.04, (right - left) * 0.1);
  const padY = Math.max(MAP_LAYER.height * 0.04, (bottom - top) * 0.1);
  return {
    left: clampNumber(left - padX, MAP_LAYER.left, MAP_LAYER.left + MAP_LAYER.width),
    right: clampNumber(right + padX, MAP_LAYER.left, MAP_LAYER.left + MAP_LAYER.width),
    top: clampNumber(top - padY, MAP_LAYER.top, MAP_LAYER.top + MAP_LAYER.height),
    bottom: clampNumber(bottom + padY, MAP_LAYER.top, MAP_LAYER.top + MAP_LAYER.height),
  };
};

const markerInsideEnvelope = (marker, envelope) => {
  if (!envelope) return true;
  const x = Number(marker?.x);
  const y = Number(marker?.y);
  return Number.isFinite(x) && Number.isFinite(y) && x >= envelope.left && x <= envelope.right && y >= envelope.top && y <= envelope.bottom;
};

const constrainStationsToUsers = (stations, users) => {
  const envelope = markerEnvelope(users);
  return envelope ? stations.filter((station) => markerInsideEnvelope(station, envelope)) : stations;
};

const scenarioModeProfile = (mode) => {
  const modes = Array.isArray(currentScenario.value?.communication_modes) ? currentScenario.value.communication_modes : [];
  return modes.find((item) => item.key === mode || item.name === mode || item.mode === mode) || null;
};

const coverageKmToGridRadius = (coverageKm, metrics) => {
  const radiusKm = Number(coverageKm);
  if (!Number.isFinite(radiusKm) || radiusKm <= 0 || !metrics) return null;
  const avgCellKm = Math.max(0.01, (Number(metrics.cellWidthKm || 0) + Number(metrics.cellHeightKm || 0)) / 2);
  return radiusKm / avgCellKm;
};

const fallbackCoverageRadiusGrid = ({ stationTypeKey, stationTypeLabel, baseStationName, mode } = {}) => {
  const text = `${stationTypeKey || ""} ${stationTypeLabel || ""} ${baseStationName || ""} ${mode || ""}`.toLowerCase();
  if (/shortwave|短波|hf/.test(text)) return 5.2;
  if (/macro|宏|low_band|600|700/.test(text)) return 4.2;
  if (/satellite|卫星/.test(text)) return 4.8;
  if (/mesh|uav|无人机/.test(text)) return 3.2;
  if (/wifi|hotspot|热点/.test(text)) return 2.2;
  if (/micro|微站|mmwave|背负/.test(text)) return 2.4;
  return 3.0;
};

const stationCoverageRadiusGrid = ({ coverageRadiusGrid, coverageRadiusKm, mode, stationTypeKey, stationTypeLabel, baseStationName } = {}) => {
  const gridRadius = Number(coverageRadiusGrid);
  if (Number.isFinite(gridRadius) && gridRadius > 0) return gridRadius;

  const kmRadius = coverageKmToGridRadius(coverageRadiusKm, activeDisasterMetrics.value || regionMetrics.value);
  if (kmRadius) return kmRadius;

  const modeProfile = scenarioModeProfile(mode);
  const modeRadius = Number(modeProfile?.coverage_radius);
  if (Number.isFinite(modeRadius) && modeRadius > 0) return modeRadius;

  return fallbackCoverageRadiusGrid({ stationTypeKey, stationTypeLabel, baseStationName, mode });
};

const stationMaxUsersForBase = (baseStationName) => {
  const station = scenarioBaseStations().find((item) => item.name === baseStationName);
  const maxUsers = Number(station?.max_users);
  return Number.isFinite(maxUsers) && maxUsers > 0 ? maxUsers : null;
};

const deploymentKey = (station, index = 0) => {
  const position = station?.grid_position || {};
  return (
    station?.deployment_id ||
    `${station?.comm_type || "comm"}:${station?.station_type || "station"}:${position.row ?? 0}:${position.col ?? 0}:${index}`
  );
};

const disasterCommProfile = (commType) =>
  DISASTER_COMMUNICATION_PROFILES[String(commType || "")] || {
    category: "custom",
    label: commType || "专用通信",
    mode: null,
  };

const disasterStationTypeLabel = (stationType, fallback) =>
  DISASTER_STATION_TYPE_LABELS[String(stationType || "")] || fallback || stationType || "场景基站";

const disasterCommunicationLabel = (commType, fallback) => {
  const profile = disasterCommProfile(commType);
  if (!fallback || fallback === commType) return profile.label;
  return fallback;
};

const communicationModeLabel = (mode, fallback) => {
  const profile = scenarioModeProfile(mode);
  return profile?.label || profile?.name || profile?.key || fallback || mode || "--";
};

const disasterStationGridText = (station) => {
  const position = station?.grid_position || {};
  return `(${Number(position.row || 0)}, ${Number(position.col || 0)})`;
};

const stationGridTextFromPoint = (point) => {
  const maxRow = Math.max(0, Math.ceil(Number(point?.rows || 1)) - 1);
  const maxCol = Math.max(0, Math.ceil(Number(point?.cols || 1)) - 1);
  const row = clampNumber(Math.floor(Number(point?.gridRow || 0)), 0, maxRow);
  const col = clampNumber(Math.floor(Number(point?.gridCol || 0)), 0, maxCol);
  return `(${row}, ${col})`;
};

const disasterDeploymentTitle = (station) => {
  const status = station?.status || "unknown";
  return `${disasterStationTypeLabel(station?.station_type, station?.station_label)} / ${disasterCommunicationLabel(
    station?.comm_type,
    station?.comm_label
  )} / ${STATION_STATUS_LABELS[status] || status} / 网格 ${disasterStationGridText(station)}`;
};

const baseStationForDisasterDeployment = (station) => {
  if (station?.base_station) {
    const matchedByName = scenarioBaseStations().find((item) => item.name === station.base_station);
    if (matchedByName) return matchedByName;
  }
  const profile = disasterCommProfile(station?.comm_type);
  const stations = scenarioBaseStations();
  if (profile.mode) {
    const matched = stations.find((item) => Array.isArray(item.supported_modes) && item.supported_modes.includes(profile.mode));
    if (matched) return matched;
  }
  const category = profile.category;
  return stations.find((item) => stationCommunicationCategory(item) === category) || stations[0] || null;
};

const fallbackSceneNodePoint = (node) => {
  const width = Math.max(1, Number(activeScene.value?.map_width || 22));
  const height = Math.max(1, Number(activeScene.value?.map_height || 22));
  const x = Number(node?.x);
  const y = Number(node?.y);
  if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
  return {
    x: MARKER_AREA.left + (x / width) * MARKER_AREA.width,
    y: MARKER_AREA.top + (y / height) * MARKER_AREA.height,
  };
};

const isLayerPointUsable = (point) =>
  Boolean(
    point &&
      Number.isFinite(point.x) &&
      Number.isFinite(point.y) &&
      point.x >= MAP_LAYER.left - MAP_LAYER.width &&
      point.x <= MAP_LAYER.left + MAP_LAYER.width * 2 &&
      point.y >= MAP_LAYER.top - MAP_LAYER.height &&
      point.y <= MAP_LAYER.top + MAP_LAYER.height * 2
  );

const sceneNodePoint = (node) => {
  const lat = Number(node?.lat);
  const lon = Number(node?.lon);
  if (Number.isFinite(lat) && Number.isFinite(lon) && mapBounds.value) {
    const viewport = mapViewport(MAP_LAYER.width, MAP_LAYER.height, mapBounds.value);
    if (viewport) {
      const point = mercatorProject(lat, lon, viewport.zoom);
      const projected = {
        x: MAP_LAYER.left + point.x - viewport.left,
        y: MAP_LAYER.top + point.y - viewport.top,
      };
      if (isLayerPointUsable(projected)) return projected;
    }
  }
  return fallbackSceneNodePoint(node);
};

const activeSceneGridShape = () => ({
  rows: Math.max(1, Number(regionGrid.value?.rows || currentScenario.value?.grid_size || 22)),
  cols: Math.max(1, Number(regionGrid.value?.cols || currentScenario.value?.grid_size || 22)),
});

const sceneNodeGridPoint = (node) => {
  const { rows, cols } = activeSceneGridShape();
  const width = Math.max(1, Number(activeScene.value?.map_width || 5000));
  const height = Math.max(1, Number(activeScene.value?.map_height || 5000));
  return {
    gridRow: clampNumber((Number(node?.y || 0) / height) * rows, 0, rows),
    gridCol: clampNumber((Number(node?.x || 0) / width) * cols, 0, cols),
    rows,
    cols,
  };
};

const jitteredSceneNodePoint = (node, seed, spread = 0.62) => {
  const point = sceneNodePoint(node);
  if (!point) return null;
  const grid = sceneNodeGridPoint(node);
  const offset = scatteredOffset(seed, spread);
  const cellWidth = MARKER_AREA.width / Math.max(1, grid.cols);
  const cellHeight = MARKER_AREA.height / Math.max(1, grid.rows);
  const rawX = point.x + offset.col * cellWidth;
  const rawY = point.y + offset.row * cellHeight;
  return {
    x: softBoundNumber(rawX, MAP_LAYER.left + 2, MAP_LAYER.left + MAP_LAYER.width - 2, seed, 6, cellWidth * 0.9),
    y: softBoundNumber(rawY, MAP_LAYER.top + 2, MAP_LAYER.top + MAP_LAYER.height - 2, seed, 7, cellHeight * 0.9),
    gridRow: softBoundGridCoord(grid.gridRow + offset.row, grid.rows, seed, 8),
    gridCol: softBoundGridCoord(grid.gridCol + offset.col, grid.cols, seed, 9),
  };
};

const gridPointToMarker = (row, col, rows, cols, seed = "", spread = 0) => {
  const safeRows = Math.max(1, Number(rows) || 1);
  const safeCols = Math.max(1, Number(cols) || 1);
  const offset = seed ? scatteredOffset(seed, spread) : { row: 0, col: 0 };
  const gridRow = softBoundGridCoord(Number(row) + 0.5 + offset.row, safeRows, seed || `${row}:${col}:row`, 10);
  const gridCol = softBoundGridCoord(Number(col) + 0.5 + offset.col, safeCols, seed || `${row}:${col}:col`, 11);
  return {
    x: MARKER_AREA.left + (gridCol / safeCols) * MARKER_AREA.width,
    y: MARKER_AREA.top + (gridRow / safeRows) * MARKER_AREA.height,
    gridRow,
    gridCol,
  };
};

const annotateUsersByCoverage = (users, stations, sourceTotalUsers = users.length) => {
  if (!users.length) return [];
  if (!stations.length) {
    return users.map((user) => ({
      ...user,
      tone: "user-offline",
      color: USER_MARKER_COLORS.offline,
      opacity: 0.7,
      title: "断联用户",
    }));
  }

  const assigned = new Set();
  const scale = users.length / Math.max(users.length, Number(sourceTotalUsers) || users.length);

  stations
    .filter((station) => station.coversUsers !== false)
    .forEach((station) => {
      const radius = Number(station.coverageRadiusGrid);
      if (!Number.isFinite(radius) || radius <= 0) return;

      const baseCapacity = Number(station.maxUsers);
      const capacity = Number.isFinite(baseCapacity) && baseCapacity > 0
        ? Math.max(1, Math.round(baseCapacity * scale))
        : users.length;

      users
        .map((user) => {
          const rowDelta = Number(user.gridRow) - Number(station.gridRow);
          const colDelta = Number(user.gridCol) - Number(station.gridCol);
          return {
            user,
            distance: Math.sqrt(rowDelta * rowDelta + colDelta * colDelta),
          };
        })
        .filter((item) => item.distance <= radius && !assigned.has(item.user.id))
        .sort((a, b) => a.distance - b.distance)
        .slice(0, capacity)
        .forEach((item) => assigned.add(item.user.id));
    });

  return users.map((user) => {
    const online = assigned.has(user.id);
    return {
      ...user,
      tone: online ? "user-online" : "user-offline",
      color: online ? USER_MARKER_COLORS.online : USER_MARKER_COLORS.offline,
      opacity: online ? 0.82 : 0.7,
      title: online ? "在线用户" : "断联用户",
    };
  });
};

const disasterDeploymentMapMarkers = (deployments = disasterDeployments.value, grid = disasterGrid.value) => {
  const rows = grid.rows || gridBounds.value.maxX + 1;
  const cols = grid.cols || gridBounds.value.maxY + 1;
  return deployments.map((station, index) => {
    const position = station.grid_position || {};
    const markerKey = deploymentKey(station, index);
    const point = gridPointToMarker(position.row || 0, position.col || 0, rows, cols, `disaster-${markerKey}`, 0.62);
    const status = station.status || "unknown";
    const type = stationTypeInfo({
      key: station.station_type,
      label: disasterStationTypeLabel(station.station_type, station.station_label),
    });
    const baseStation = baseStationForDisasterDeployment(station);
    const commLabel = disasterCommunicationLabel(station.comm_type, station.comm_label);
    const mode = station.mode || disasterCommProfile(station.comm_type).mode;
    const gridText = disasterStationGridText(station);
    return {
      id: `disaster-station-${station.deployment_id || index}`,
      kind: "station",
      tone: "station-type",
      x: point.x,
      y: point.y,
      color: type.color,
      stationTypeKey: type.key,
      stationTypeLabel: type.label,
      communicationLabel: commLabel,
      gridText,
      status,
      statusLabel: STATION_STATUS_LABELS[status] || status,
      gridRow: point.gridRow,
      gridCol: point.gridCol,
      maxUsers: Number(station.cell_user_count || baseStation?.max_users || 0),
      coverageRadiusKm: Number(station.coverage_radius_km || 0),
      coverageRadiusGrid: stationCoverageRadiusGrid({
        coverageRadiusKm: station.coverage_radius_km,
        mode,
        stationTypeKey: type.key,
        stationTypeLabel: type.label,
      }),
      coversUsers: status !== "offline",
      title: `${type.label} / ${commLabel} / ${STATION_STATUS_LABELS[status] || status} / 网格 ${gridText}`,
    };
  });
};

const plannedDeviceMapMarkers = ({ includeImportRows = true } = {}) => {
  return appliedDeviceRows.value.filter((row) => includeImportRows || row.source !== "disaster-import").flatMap((row) =>
    Array.from({ length: Math.max(1, Number(row.quantity || 1)) }, (_, index) => {
      const importGrid = row.sourceImportId === activeDisasterImportId.value ? activeDisasterGrid.value : disasterGrid.value;
      const rows =
        row.source === "disaster-import"
          ? Math.max(1, Number(importGrid.rows) || gridBounds.value.maxX + 1)
          : gridBounds.value.maxX + 1;
      const cols =
        row.source === "disaster-import"
          ? Math.max(1, Number(importGrid.cols) || gridBounds.value.maxY + 1)
          : gridBounds.value.maxY + 1;
      const x = (Math.round(Number(row.x || 0)) + index) % Math.max(1, rows);
      const y = (Math.round(Number(row.y || 0)) + index) % Math.max(1, cols);
      const point = gridPointToMarker(x, y, rows, cols, `planned-${row.deviceId}-${index}`, 0.62);
      const type = stationTypeInfo({
        key: row.stationType || row.baseStationName || row.deviceType,
        label: row.stationLabel || row.deviceType || row.name,
        baseStationName: row.baseStationName,
      });
      const communicationLabel = row.commLabel || communicationCategoryLabel(row.communicationType);
      const gridText = row.gridText || `(${x}, ${y})`;
      const status = row.stationStatus || (row.enabled === false ? "offline" : "active");
      const statusLabel = STATION_STATUS_LABELS[row.stationStatus] || row.status || (row.enabled === false ? "未启用" : "已接入");
      return {
        id: `planned-station-${row.deviceId}-${index}`,
        kind: "station",
        tone: "station-type",
        x: point.x,
        y: point.y,
        color: type.color,
        stationTypeKey: type.key,
        stationTypeLabel: type.label,
        communicationLabel,
        gridText,
        status,
        statusLabel,
        gridRow: point.gridRow,
        gridCol: point.gridCol,
        maxUsers: Number(row.maxUsers || 0),
        coverageRadiusKm: Number(row.coverageRadiusKm || 0),
        coverageRadiusGrid: stationCoverageRadiusGrid({
          coverageRadiusKm: row.coverageRadiusKm,
          mode: row.mode,
          stationTypeKey: type.key,
          stationTypeLabel: type.label,
          baseStationName: row.baseStationName,
        }),
        coversUsers: row.stationStatus !== "offline",
        title: `${type.label} / ${communicationLabel} / ${statusLabel} / 网格 ${gridText}`,
      };
    })
  );
};

const sceneStationMapMarkers = ({ deployedOnly = false } = {}) => {
  const importedStationCount = (importedScene.value?.scene?.nodes || []).filter(isStationNode).length;
  return rawNodes.value
    .filter(isStationNode)
    .map((node, index) => {
      const deployed = activeSceneTab.value === "deployment" && index >= importedStationCount;
      if (deployedOnly && !deployed) return null;
      const point = jitteredSceneNodePoint(node, `scene-station-${node.id ?? index}`, 0.62);
      if (!point) return null;
      const type = stationTypeInfo({
        key: node.base_station || node.label || node.type,
        label: node.label,
        baseStationName: node.base_station,
      });
      const baseStation = scenarioBaseStations().find((item) => item.name === node.base_station);
      const mode = node.mode || defaultStationMode(baseStation);
      const grid = sceneNodeGridPoint(node);
      const roleDeployed = node.node_role === "planned_deployment";
      const roleRestored = Boolean(node.recovery_action && node.recovery_action !== "new_deployment");
      const status = node.status || (roleDeployed || deployed ? "active" : "active");
      const originalStatus = node.original_status || status;
      const baseStatusLabel = STATION_STATUS_LABELS[status] || status;
      const statusLabel =
        roleRestored && originalStatus && originalStatus !== status
          ? `${baseStatusLabel}（由${STATION_STATUS_LABELS[originalStatus] || originalStatus}恢复）`
          : baseStatusLabel;
      const communicationLabel = communicationModeLabel(mode, communicationCategoryLabel(stationCommunicationCategory(baseStation)));
      return {
        id: `station-${node.id ?? index}`,
        kind: "station",
        tone: roleDeployed || deployed ? "station-deployed" : roleRestored ? "station-restored" : "station-type",
        x: point.x,
        y: point.y,
        color: type.color,
        stationTypeKey: type.key,
        stationTypeLabel: type.label,
        communicationLabel,
        gridText: stationGridTextFromPoint(grid),
        status,
        statusLabel,
        gridRow: point.gridRow,
        gridCol: point.gridCol,
        maxUsers: stationMaxUsersForBase(node.base_station),
        coverageRadiusGrid: stationCoverageRadiusGrid({
          coverageRadiusGrid: node.coverage_radius || node.coverage_radius_grid,
          coverageRadiusKm: node.coverage_radius_km,
          mode: node.mode,
          stationTypeKey: type.key,
          stationTypeLabel: type.label,
          baseStationName: node.base_station,
        }),
        coversUsers: node.status !== "offline",
        sourceLabel: roleDeployed || deployed ? "新部署设备" : roleRestored ? "原始站点 / 策略恢复" : "原始导入站点",
        originalStatus,
        recoveryAction: node.recovery_action,
        recoveryStep: node.recovery_step,
        recoveryReason: node.recovery_reason,
        title: `${type.label} / ${communicationLabel} / ${statusLabel} / 网格 ${stationGridTextFromPoint(grid)}`,
      };
    })
    .filter(Boolean);
};

const sceneUserMapMarkers = () =>
  rawNodes.value
    .filter((node) => node.type === "USER")
    .map((node, index) => {
      const seed = `scene-user-${node.id ?? index}`;
      const point = jitteredSceneNodePoint(node, seed, USER_SCATTER_SPREAD);
      if (!point) return null;
      const online = Boolean(node.connected);
      return {
        id: `user-${node.id ?? index}`,
        kind: "user",
        tone: online ? "user-online" : "user-offline",
        x: point.x,
        y: point.y,
        gridRow: point.gridRow,
        gridCol: point.gridCol,
        color: online ? USER_MARKER_COLORS.online : USER_MARKER_COLORS.offline,
        size: userMarkerSize(seed),
        opacity: online ? 0.82 : 0.7,
        title: online ? "在线用户" : "断联用户",
      };
    })
    .filter(Boolean);

const disasterUserMapMarkers = (
  stationMarkers,
  importRecord = selectedDisasterImport.value,
  heatmap = disasterHeatmap.value,
  grid = disasterGrid.value
) => {
  const totalUsers = Number(importRecord?.unique_user_count || 0);
  if (!totalUsers) return [];

  const rows = Math.max(1, Number(grid.rows) || 1);
  const cols = Math.max(1, Number(grid.cols) || 1);
  const displayUsers = Math.max(1, Math.round(totalUsers));
  const heatCells = heatmap.length
    ? heatmap.map((cell) => ({
        row: Number(cell.grid_row || 0),
        col: Number(cell.grid_col || 0),
        weight: Math.max(0, Number(cell.user_count || 0)),
      }))
    : Array.from({ length: rows * cols }, (_, index) => ({
        row: Math.floor(index / cols),
        col: index % cols,
        weight: 1,
      }));

  const weightTotal = heatCells.reduce((sum, cell) => sum + cell.weight, 0) || heatCells.length || 1;
  const allocations = heatCells.map((cell) => {
    const exact = (cell.weight / weightTotal) * displayUsers;
    return {
      ...cell,
      exact,
      count: Math.floor(exact),
    };
  });
  let allocated = allocations.reduce((sum, cell) => sum + cell.count, 0);
  allocations
    .map((cell, index) => ({ index, remainder: cell.exact - cell.count }))
    .sort((a, b) => b.remainder - a.remainder)
    .slice(0, Math.max(0, displayUsers - allocated))
    .forEach(({ index }) => {
      allocations[index].count += 1;
      allocated += 1;
    });

  const markers = [];
  allocations.forEach((cell) => {
    for (let index = 0; index < cell.count; index += 1) {
      const seed = `import-user-${cell.row}-${cell.col}-${index}`;
      const point = gridPointToMarker(cell.row, cell.col, rows, cols, seed, USER_SCATTER_SPREAD);
      markers.push({
        id: `import-user-${cell.row}-${cell.col}-${index}`,
        kind: "user",
        tone: "user-offline",
        x: point.x,
        y: point.y,
        gridRow: point.gridRow,
        gridCol: point.gridCol,
        color: USER_MARKER_COLORS.offline,
        size: userMarkerSize(seed),
        opacity: 0.7,
        title: "断联用户",
      });
    }
  });
  return annotateUsersByCoverage(markers, stationMarkers, totalUsers);
};

const mapMarkers = computed(() => {
  if (isLoading.value || !mapDataReady.value) return [];

  const sceneStationMarkers = sceneStationMapMarkers();
  const hasSceneUsers = rawNodes.value.some((node) => node.type === "USER");
  if (sceneStationMarkers.length || hasSceneUsers) {
    const userMarkers = sceneUserMapMarkers();
    const stationSource = sceneStationMarkers.length ? sceneStationMarkers : plannedDeviceMapMarkers({ includeImportRows: false });
    const stationMarkers = constrainStationsToUsers(stationSource, userMarkers);
    return [...userMarkers, ...stationMarkers];
  }

  const markers = [];
  const disasterMarkers = disasterDeploymentMapMarkers(activeDisasterDeployments.value, activeDisasterGrid.value);
  const hasDisasterUsers = Boolean(activeDisasterImport.value) && Number(activeDisasterImport.value?.unique_user_count || 0) > 0;
  if (disasterMarkers.length || hasDisasterUsers) {
    const stationSource = [...disasterMarkers, ...plannedDeviceMapMarkers({ includeImportRows: false })];
    if (activeSceneTab.value === "deployment") {
      stationSource.push(...sceneStationMapMarkers({ deployedOnly: true }));
    }
    const userMarkers = disasterUserMapMarkers(stationSource, activeDisasterImport.value, activeDisasterHeatmap.value, activeDisasterGrid.value);
    const stationMarkers = constrainStationsToUsers(stationSource, userMarkers);
    markers.push(...disasterUserMapMarkers(stationMarkers, activeDisasterImport.value, activeDisasterHeatmap.value, activeDisasterGrid.value), ...stationMarkers);
    return markers;
  }

  return [];
});

const mapLegendItems = computed(() => {
  const byType = new Map();
  const hasUsers = mapMarkers.value.some((marker) => marker.kind === "user");
  mapMarkers.value
    .filter((marker) => marker.kind === "station")
    .forEach((marker) => {
      const type = stationTypeInfo({
        key: marker.stationTypeKey,
        label: marker.stationTypeLabel,
      });
      const legendKey = String(type.label || type.key || "").trim();
      if (legendKey && !byType.has(legendKey)) {
        byType.set(legendKey, {
          key: legendKey,
          label: type.label,
          shape: "circle",
          color: type.color,
        });
      }
    });
  const userEntries = hasUsers
    ? [
        { key: "user-offline", label: "断联用户", shape: "circle", color: USER_MARKER_COLORS.offline },
        { key: "user-online", label: "在线用户", shape: "circle", color: USER_MARKER_COLORS.online },
      ]
    : [];
  return [...userEntries, ...byType.values()];
});

const markerStyle = (marker) => {
  const style = {
    left: `${marker.x}px`,
    top: `${marker.y}px`,
    "--marker-color": marker.color || STATION_STATUS_COLORS.unknown,
    "--marker-opacity": marker.opacity ?? (marker.kind === "station" ? 1 : 0.72),
  };
  if (marker.size) {
    style["--marker-size"] = marker.size;
  }
  return style;
};

const stationTooltipMarker = computed(() => stationTooltip.value.marker);

const stationTooltipRows = computed(() => {
  const marker = stationTooltipMarker.value;
  if (!marker) return [];
  const gridText =
    marker.gridText ||
    `(${Math.max(0, Math.floor(Number(marker.gridRow || 0)))}, ${Math.max(0, Math.floor(Number(marker.gridCol || 0)))})`;
  return [
    { label: "类型", value: marker.stationTypeLabel || "基站" },
    { label: "通信方式", value: marker.communicationLabel || "--" },
    { label: "所属网格", value: gridText },
    ...(marker.sourceLabel ? [{ label: "来源", value: marker.sourceLabel }] : []),
    ...(marker.originalStatus && marker.originalStatus !== marker.status
      ? [{ label: "恢复前", value: STATION_STATUS_LABELS[marker.originalStatus] || marker.originalStatus }]
      : []),
    ...(marker.recoveryStep != null ? [{ label: "恢复步骤", value: `step ${marker.recoveryStep}` }] : []),
    ...(marker.recoveryReason ? [{ label: "恢复说明", value: marker.recoveryReason }] : []),
    { label: "状态", value: marker.statusLabel || STATION_STATUS_LABELS[marker.status] || marker.status || "未知", status: true },
  ];
});

const stationTooltipStyle = computed(() => {
  const marker = stationTooltipMarker.value;
  if (!marker) return {};
  const x = Number(marker.x || 0);
  const y = Number(marker.y || 0);
  const rightSpace = MAP_LAYER.left + MAP_LAYER.width - x;
  const left = rightSpace < STATION_TOOLTIP_WIDTH + STATION_TOOLTIP_GAP + 12
    ? x - STATION_TOOLTIP_WIDTH - STATION_TOOLTIP_GAP
    : x + STATION_TOOLTIP_GAP;
  const top = y - STATION_TOOLTIP_HEIGHT / 2;
  return {
    left: `${clampNumber(left, MAP_LAYER.left + 10, MAP_LAYER.left + MAP_LAYER.width - STATION_TOOLTIP_WIDTH - 10)}px`,
    top: `${clampNumber(top, MAP_LAYER.top + 10, MAP_LAYER.top + MAP_LAYER.height - STATION_TOOLTIP_HEIGHT - 10)}px`,
  };
});

const showStationTooltip = (marker) => {
  if (!marker || marker.kind !== "station") return;
  stationTooltip.value = {
    visible: true,
    marker,
  };
};

const hideStationTooltip = () => {
  stationTooltip.value = {
    visible: false,
    marker: null,
  };
};

const mapBounds = computed(() => {
  const importBounds = normalizeGeoBounds(simulationDisasterBounds.value);
  const activeBounds = normalizeGeoBounds(activeScene.value?.geo_bounds);
  const gridBounds = normalizeGeoBounds(regionGrid.value?.geo_bounds);
  const nodeBounds = expandGeoBounds(nodesGeoBounds(rawNodes.value), 0.12);
  return mergeGeoBounds(importBounds, activeBounds, gridBounds, nodeBounds);
});

const mapLabel = computed(
  () =>
    formatPlainDisasterName(
      activeDisasterImport.value?.disaster_scenario,
      activeDisasterImport.value?.disaster_scenario_label,
      regionGrid.value?.name,
      scenarioLabel(currentScenario.value)
    ) || scenarioLabel(currentScenario.value)
);

const mapSceneText = computed(() =>
  [
    activeDisasterImport.value?.disaster_scenario,
    activeDisasterImport.value?.disaster_scenario_label,
    activeDisasterImportDetail.value?.disaster_scenario,
    activeDisasterImportDetail.value?.disaster_scenario_label,
    activeScene.value?.name,
    activeScene.value?.disaster_type,
    regionGrid.value?.name,
    currentScenario.value?.name,
    currentScenario.value?.disaster_type,
    mapLabel.value,
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase()
);

const isRainstormMapScene = computed(() => /rainstorm|暴雨|内涝/.test(mapSceneText.value));
const isTyphoonMapScene = computed(() => /typhoon|台风|风暴潮/.test(mapSceneText.value));

const mapBasemapShiftX = computed(() => {
  if (isRainstormMapScene.value) return Math.round(MAP_LAYER.width * BASEMAP_SHIFT_RATIOS.rainstorm);
  if (isTyphoonMapScene.value) return Math.round(MAP_LAYER.width * BASEMAP_SHIFT_RATIOS.typhoon);
  return 0;
});

const mapTiles = computed(() => {
  if (!mapDataReady.value) return [];
  const viewport = mapViewport(MAP_LAYER.width, MAP_LAYER.height, mapBounds.value);
  if (!viewport) return [];
  const tileSize = 256;
  const maxTile = 2 ** viewport.zoom;
  const shiftedViewportLeft = viewport.left - mapBasemapShiftX.value;
  const minTileX = Math.floor(shiftedViewportLeft / tileSize) - 1;
  const maxTileX = Math.floor((shiftedViewportLeft + MAP_LAYER.width) / tileSize) + 1;
  const minTileY = Math.floor(viewport.top / tileSize) - 1;
  const maxTileY = Math.floor((viewport.top + MAP_LAYER.height) / tileSize) + 1;
  const tiles = [];

  for (let tileX = minTileX; tileX <= maxTileX; tileX += 1) {
    const wrappedX = ((tileX % maxTile) + maxTile) % maxTile;
    for (let tileY = minTileY; tileY <= maxTileY; tileY += 1) {
      if (tileY < 0 || tileY >= maxTile) continue;
      tiles.push({
        key: `${viewport.zoom}-${wrappedX}-${tileY}`,
        url: cartoTileUrl(viewport.zoom, wrappedX, tileY),
        left: Math.round(tileX * tileSize - shiftedViewportLeft),
        top: Math.round(tileY * tileSize - viewport.top),
      });
    }
  }
  return tiles;
});

const startButtonText = computed(() => {
  if (isRunning.value) return "测试中...";
  if (isLoading.value) return "同步中...";
  return simulationResult.value ? "重新测试" : "开始测试";
});

const evaluationProtocol = computed(() =>
  currentScenario.value?.disaster_type === "earthquake" ? "earthquake_stress" : "standard"
);

const formatMetric = (value, digits = 2) => {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : "--";
};

const formatInteger = (value) => {
  const number = Number(value);
  return Number.isFinite(number) ? Math.round(number).toLocaleString("zh-CN") : "--";
};

const formatPercent = (value) => {
  const number = Number(value);
  return Number.isFinite(number) ? `${(number * 100).toFixed(2)}%` : "--";
};

const formatDateTime = (value) => {
  if (!value) return "--";
  return new Date(value).toLocaleString("zh-CN", { hour12: false });
};

const normalizeGeoBounds = (bounds) => {
  if (!bounds) return null;
  const latMin = Number(bounds.lat_min);
  const latMax = Number(bounds.lat_max);
  const lonMin = Number(bounds.lon_min);
  const lonMax = Number(bounds.lon_max);
  if (![latMin, latMax, lonMin, lonMax].every(Number.isFinite)) return null;
  return {
    latMin: Math.min(latMin, latMax),
    latMax: Math.max(latMin, latMax),
    lonMin: Math.min(lonMin, lonMax),
    lonMax: Math.max(lonMin, lonMax),
  };
};

const nodesGeoBounds = (nodes) => {
  const points = nodes
    .map((node) => ({ lat: Number(node?.lat), lon: Number(node?.lon) }))
    .filter((point) => Number.isFinite(point.lat) && Number.isFinite(point.lon));
  if (!points.length) return null;
  return {
    latMin: Math.min(...points.map((point) => point.lat)),
    latMax: Math.max(...points.map((point) => point.lat)),
    lonMin: Math.min(...points.map((point) => point.lon)),
    lonMax: Math.max(...points.map((point) => point.lon)),
  };
};

const expandGeoBounds = (bounds, paddingRatio = 0.2) => {
  if (!bounds) return null;
  const latSpan = Math.max(0.0001, bounds.latMax - bounds.latMin);
  const lonSpan = Math.max(0.0001, bounds.lonMax - bounds.lonMin);
  return {
    latMin: bounds.latMin - latSpan * paddingRatio,
    latMax: bounds.latMax + latSpan * paddingRatio,
    lonMin: bounds.lonMin - lonSpan * paddingRatio,
    lonMax: bounds.lonMax + lonSpan * paddingRatio,
  };
};

const mergeGeoBounds = (...boundsList) => {
  const validBounds = boundsList.filter(Boolean);
  if (!validBounds.length) return null;
  return {
    latMin: Math.min(...validBounds.map((bounds) => bounds.latMin)),
    latMax: Math.max(...validBounds.map((bounds) => bounds.latMax)),
    lonMin: Math.min(...validBounds.map((bounds) => bounds.lonMin)),
    lonMax: Math.max(...validBounds.map((bounds) => bounds.lonMax)),
  };
};

const mercatorProject = (lat, lon, zoom) => {
  const size = 256 * 2 ** zoom;
  const safeLat = Math.max(-85.05112878, Math.min(85.05112878, Number(lat)));
  const sin = Math.sin((safeLat * Math.PI) / 180);
  return {
    x: ((Number(lon) + 180) / 360) * size,
    y: (0.5 - Math.log((1 + sin) / (1 - sin)) / (4 * Math.PI)) * size,
  };
};

const mapViewport = (width, height, bounds) => {
  if (!bounds) return null;
  let bestZoom = 5;
  for (let zoom = 5; zoom <= 14; zoom += 1) {
    const northWest = mercatorProject(bounds.latMax, bounds.lonMin, zoom);
    const southEast = mercatorProject(bounds.latMin, bounds.lonMax, zoom);
    const spanX = Math.abs(southEast.x - northWest.x);
    const spanY = Math.abs(southEast.y - northWest.y);
    if (spanX <= width * 0.82 && spanY <= height * 0.82) {
      bestZoom = zoom;
    }
  }
  const center = mercatorProject((bounds.latMin + bounds.latMax) / 2, (bounds.lonMin + bounds.lonMax) / 2, bestZoom);
  return {
    zoom: bestZoom,
    left: center.x - width / 2,
    top: center.y - height / 2,
  };
};

const cartoTileUrl = (zoom, x, y) => {
  const subdomains = ["a", "b", "c", "d"];
  const subdomain = subdomains[Math.abs(x + y) % subdomains.length];
  return `https://${subdomain}.basemaps.cartocdn.com/rastertiles/voyager/${zoom}/${x}/${y}.png`;
};

const disasterBoundsText = (bounds) => {
  const normalized = normalizeGeoBounds(bounds);
  if (!normalized) return "--";
  return `lat ${formatMetric(normalized.latMin, 3)}~${formatMetric(normalized.latMax, 3)}，lon ${formatMetric(
    normalized.lonMin,
    3
  )}~${formatMetric(normalized.lonMax, 3)}`;
};

const heatCellRatio = (cell) => {
  const maxHeat = Math.max(1, ...disasterHeatmap.value.map((item) => Number(item.user_count || 0)));
  return clampNumber(Number(cell?.user_count || 0) / maxHeat, 0, 1);
};

const heatCellTitle = (cell) => {
  const row = Number(cell?.grid_row || 0);
  const col = Number(cell?.grid_col || 0);
  const count = Number(cell?.user_count || 0);
  return `网格 (${row}, ${col}) / 断联用户 ${count}`;
};

const heatCellStyle = (cell) => {
  const rows = Math.max(1, Number(disasterGrid.value.rows) || 24);
  const cols = Math.max(1, Number(disasterGrid.value.cols) || 24);
  const row = Number(cell.grid_row || 0);
  const col = Number(cell.grid_col || 0);
  const ratio = heatCellRatio(cell);
  const heatScale = 2.9;
  const insetX = 5;
  const insetY = 7;
  const heatColors =
    ratio > 0.72
      ? ["rgba(239, 68, 68, 0.72)", "rgba(245, 158, 11, 0.46)"]
      : ratio > 0.38
        ? ["rgba(245, 158, 11, 0.54)", "rgba(34, 197, 94, 0.26)"]
        : ["rgba(37, 99, 235, 0.2)", "rgba(14, 165, 233, 0.24)"];
  return {
    top: `${insetY + ((row + 0.5) / rows) * (100 - insetY * 2)}%`,
    left: `${insetX + ((col + 0.5) / cols) * (100 - insetX * 2)}%`,
    width: `${(100 / cols) * heatScale}%`,
    height: `${(100 / rows) * heatScale}%`,
    background: `radial-gradient(circle, ${heatColors[0]} 0%, ${heatColors[1]} 42%, rgba(255, 255, 255, 0) 72%)`,
    opacity: `${(0.36 + ratio * 0.5).toFixed(3)}`,
  };
};

const deploymentMarkerStyle = (station) => {
  const rows = Math.max(1, Number(disasterGrid.value.rows) || 24);
  const cols = Math.max(1, Number(disasterGrid.value.cols) || 24);
  const position = station?.grid_position || {};
  const row = Number(position.row || 0);
  const col = Number(position.col || 0);
  const seed = deploymentKey(station, 0);
  const rowOffset = (seededUnit(seed, 1) - 0.5) * 0.52;
  const colOffset = (seededUnit(seed, 2) - 0.5) * 0.52;
  return {
    top: `${(clampNumber(row + 0.5 + rowOffset, 0.04, rows - 0.04) / rows) * 100}%`,
    left: `${(clampNumber(col + 0.5 + colOffset, 0.04, cols - 0.04) / cols) * 100}%`,
  };
};

const readStorage = (key, fallback) => {
  try {
    const raw = window.localStorage.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  } catch {
    return fallback;
  }
};

const writeStorage = (key, value) => {
  window.localStorage.setItem(key, JSON.stringify(value));
};

const communicationCategoryLabel = (type) =>
  COMMUNICATION_TYPE_OPTIONS.find((item) => item.value === type)?.label || type || "--";

const stationCommunicationCategory = (station) => {
  const text = `${station?.name || ""} ${station?.label || ""} ${(station?.supported_modes || []).join(" ")}`;
  if (/wifi/i.test(text)) return "wifi";
  if (/satellite|卫星/i.test(text)) return "satellite";
  if (/shortwave|hf|短波/i.test(text)) return "shortwave";
  if (/mesh|uav|无人机/i.test(text)) return "mesh";
  if (/5g|macro|mmwave|蜂窝|宏站|微站/i.test(text)) return "cellular";
  return "custom";
};

const ensureDeviceLibrarySeeded = () => {
  const current = readStorage(DEVICE_LIBRARY_KEY, []);
  if (Array.isArray(current) && current.length) return current;
  writeStorage(DEVICE_LIBRARY_KEY, DEFAULT_DEVICE_TEMPLATES);
  return [...DEFAULT_DEVICE_TEMPLATES];
};

const scenarioBaseStations = () => (Array.isArray(currentScenario.value?.base_stations) ? currentScenario.value.base_stations : []);

const scenarioBaseStationDeployments = () =>
  Array.isArray(currentScenario.value?.base_station_deployments)
    ? currentScenario.value.base_station_deployments
    : Array.isArray(currentScenario.value?.residual_base_stations)
      ? currentScenario.value.residual_base_stations
      : [];

const defaultStationMode = (station) =>
  Array.isArray(station?.supported_modes) && station.supported_modes.length ? station.supported_modes[0] : null;

const buildScenarioStationRow = (station, binding = {}) => ({
  deviceId: binding.deviceId || `station:${scenarioName.value}:${station.name}`,
  baseStationName: station.name,
  mode: binding.mode || defaultStationMode(station),
  name: binding.name || station.label || station.name,
  deviceType: binding.deviceType || station.label || station.name,
  communicationType: binding.communicationType || stationCommunicationCategory(station),
  quantity: Math.max(1, Number(binding.quantity || 1)),
  maxThroughput: Number(binding.maxThroughput ?? station.max_throughput ?? 0),
  maxUsers: Number(binding.maxUsers ?? station.max_users ?? 0),
  coverageRadiusKm: Number(binding.coverageRadiusKm || station.coverage_radius_km || 0),
  enabled: binding.enabled !== false,
  applied: Boolean(binding.applied),
  x: Number(binding.x || 0),
  y: Number(binding.y || 0),
  status: binding.status || "已导入",
  source: binding.source || "scenario-profile",
  sourceImportId: binding.sourceImportId || null,
  deploymentId: binding.deploymentId || null,
  stationType: binding.stationType || null,
  stationLabel: binding.stationLabel || null,
  commType: binding.commType || null,
  commLabel: binding.commLabel || null,
  stationStatus: binding.stationStatus || null,
  gridText: binding.gridText || null,
});

const buildScenarioDeploymentRow = (deployment, index, options = {}) => {
  const stations = scenarioBaseStations();
  const baseStationName = deployment.base_station || deployment.baseStationName || "";
  const station = stations.find((item) => item.name === baseStationName);
  const mode = deployment.mode || defaultStationMode(station);
  const x = Math.max(0, Math.min(gridBounds.value.maxX, Math.round(Number(deployment.x || 0))));
  const y = Math.max(0, Math.min(gridBounds.value.maxY, Math.round(Number(deployment.y || 0))));
  const stationTypeLabel = disasterStationTypeLabel(deployment.station_type, deployment.label || station?.label || baseStationName || "场景基站");
  const label = displayDeviceText(deployment.device_name || stationTypeLabel);
  const deploymentUid = deployment.device_uid || deployment.id || deployment.deployment_id || null;
  return {
    deviceId: deploymentUid || `scenario-deployment:${scenarioName.value}:${index}:${baseStationName}:${mode || "mode"}:${x}:${y}`,
    baseStationName,
    mode,
    name: label,
    deviceType: label,
    communicationType: station ? stationCommunicationCategory(station) : "custom",
    quantity: 1,
    maxThroughput: Number(deployment.max_throughput ?? station?.max_throughput ?? 0),
    maxUsers: Number(deployment.max_users ?? station?.max_users ?? 0),
    coverageRadiusKm: Number(deployment.source_coverage_radius_km || deployment.coverageRadiusKm || 0),
    enabled: true,
    applied: options.applied !== false,
    x,
    y,
    status: deployment.statusLabel || STATION_STATUS_LABELS[deployment.status] || "已接入",
    source: options.source || "scenario-deployment",
    sourceImportId: options.sourceImportId || null,
    deploymentId: deployment.deployment_id || deploymentUid,
    stationType: deployment.station_type || baseStationName,
    stationLabel: displayDeviceText(stationTypeLabel),
    commType: mode,
    commLabel: deployment.mode_label || communicationModeLabel(mode),
    stationStatus: deployment.status || "active",
    gridText: `(${x}, ${y})`,
  };
};

const loadScenarioDeviceRows = () => {
  ensureDeviceLibrarySeeded();
  const stations = scenarioBaseStations();

  if (stations.length) {
    const deploymentRows = scenarioBaseStationDeployments().map(buildScenarioDeploymentRow);
    const templateRows = stations.map((station) =>
      buildScenarioStationRow(station, {
        deviceId: `station-template:${scenarioName.value}:${station.name}`,
        applied: false,
        source: "scenario-profile",
      })
    );
    deviceRows.value = [...deploymentRows, ...templateRows];
    return;
  }

  const bindings = readStorage(DEVICE_BINDINGS_KEY, {});
  const currentBindings = Array.isArray(bindings[scenarioName.value]) ? bindings[scenarioName.value] : [];
  const bindingMap = Object.fromEntries(currentBindings.map((item) => [item.deviceId, item]));
  const library = ensureDeviceLibrarySeeded();
  deviceRows.value = library.map((device) => {
    const binding = bindingMap[device.id] || {};
    return {
      deviceId: device.id,
      name: device.name,
      deviceType: device.deviceType,
      communicationType: device.communicationType,
      quantity: Math.max(1, Number(binding.quantity || device.quantity || 1)),
      maxThroughput: Number(device.maxThroughput || 0),
      maxUsers: Number(device.maxUsers || 0),
      coverageRadiusKm: Number(binding.coverageRadiusKm || device.coverageRadiusKm || 0),
      enabled: binding.enabled !== false && device.enabled !== false,
      applied: Boolean(binding.applied),
      x: Number(binding.x || 0),
      y: Number(binding.y || 0),
      status: device.status || "已导入",
    };
  });
};

const applyDisasterDeviceRows = (detail) => {
  loadScenarioDeviceRows();
};

const saveScenarioDeviceRows = () => {
  const bindings = readStorage(DEVICE_BINDINGS_KEY, {});
  bindings[scenarioName.value] = deviceRows.value.map((row) => ({
    deviceId: row.deviceId,
    baseStationName: row.baseStationName || null,
    mode: row.mode || null,
    name: row.name,
    deviceType: row.deviceType || null,
    communicationType: row.communicationType || null,
    quantity: Math.max(1, Number(row.quantity || 1)),
    maxThroughput: Number(row.maxThroughput || 0),
    maxUsers: Number(row.maxUsers || 0),
    coverageRadiusKm: Number(row.coverageRadiusKm || 0),
    enabled: row.enabled !== false,
    applied: Boolean(row.applied),
    x: Math.max(0, Math.min(gridBounds.value.maxX, Math.round(Number(row.x || 0)))),
    y: Math.max(0, Math.min(gridBounds.value.maxY, Math.round(Number(row.y || 0)))),
    status: row.status || null,
    source: row.source || "scenario-profile",
    sourceImportId: row.sourceImportId || null,
    deploymentId: row.deploymentId || null,
    stationType: row.stationType || null,
    stationLabel: row.stationLabel || null,
    commType: row.commType || null,
    commLabel: row.commLabel || null,
    stationStatus: row.stationStatus || null,
    gridText: row.gridText || null,
  }));
  writeStorage(DEVICE_BINDINGS_KEY, bindings);
};

const applyScenarioBaseStationResponse = (payload) => {
  const deployments = Array.isArray(payload?.base_stations) ? payload.base_stations : [];
  scenarios.value = scenarios.value.map((scenario) =>
    scenario.name === scenarioName.value
      ? {
          ...scenario,
          base_station_deployments: deployments,
          residual_base_stations: deployments,
        }
      : scenario
  );
};

const applyScenarioDeviceStateResponse = (payload, options = {}) => {
  const targetScenarioName = payload?.scenario_name || scenarioName.value;
  const deployments = Array.isArray(payload?.devices)
    ? payload.devices
    : Array.isArray(payload?.base_stations)
      ? payload.base_stations
      : [];
  scenarios.value = scenarios.value.map((scenario) =>
    scenario.name === targetScenarioName
      ? {
          ...scenario,
          base_station_deployments: deployments,
          residual_base_stations: deployments,
        }
      : scenario
  );
  if (targetScenarioName !== scenarioName.value) return;
  const preservedImportRows = options.preserveDisasterImportRows === false
    ? []
    : deviceRows.value.filter((row) => row.source === "disaster-import");
  loadScenarioDeviceRows();
  if (preservedImportRows.length) {
    const baseRows = deviceRows.value.filter((row) => row.source !== "disaster-import");
    deviceRows.value = [...preservedImportRows, ...baseRows];
  }
};

const refreshScenarioDeviceState = async (options = {}) => {
  if (!scenarioName.value) return false;
  try {
    const { data } = await axios.get(
      `${API_BASE}/scenarios/${encodeURIComponent(scenarioName.value)}/device-state`,
      { timeout: SCENE_ACCESS_TIMEOUT_MS }
    );
    applyScenarioDeviceStateResponse(data, options);
    return true;
  } catch (error) {
    appendTerminalLine(`场景设备状态刷新失败：${error?.response?.data?.detail || error?.message || error}`);
    return false;
  }
};

const syncScenarioBaseStations = async () => {
  if (!scenarioName.value) return false;
  if (!scenarioBaseStations().length) {
    saveScenarioDeviceRows();
    return true;
  }
  try {
    const { data } = await axios.put(
      `${API_BASE}/scenarios/${encodeURIComponent(scenarioName.value)}/base-stations`,
      { base_stations: buildScenarioDeviceBaseStations() },
      { timeout: SCENE_ACCESS_TIMEOUT_MS }
    );
    applyScenarioBaseStationResponse(data);
    return true;
  } catch (error) {
    const message = error?.response?.data?.detail || error?.message || String(error);
    appendTerminalLine(`场景基站更新失败：${message}`);
    showStatus("场景基站更新失败，请检查后端服务。", "error");
    return false;
  }
};

const clearTestingResidualNetwork = async () => {
  if (!scenarioName.value || isRunning.value || isLoading.value || isClearingResidualNetwork.value) return;
  if (!activeAppliedDeviceRows.value.length) {
    appendTerminalLine("当前测试场景已经是无残余网络。");
    showStatus("当前测试场景已经是无残余网络。", "info", 2200);
    return;
  }
  if (!window.confirm("确认删除当前测试场景的所有残余基站并切换为无残余网络？可在设备管理中恢复原始场景基站。")) return;

  const shouldReloadScene = hasImportedScene.value;
  isClearingResidualNetwork.value = true;
  try {
    const { data } = await axios.put(
      `${API_BASE}/scenarios/${encodeURIComponent(scenarioName.value)}/device-state`,
      { base_stations: [], operation: "clear_residual_network" },
      { timeout: SCENE_ACCESS_TIMEOUT_MS }
    );
    applyScenarioDeviceStateResponse(data, { preserveDisasterImportRows: false });
    importedScene.value = null;
    simulationResult.value = null;
    activeSceneTab.value = "imported";
    appendTerminalLine("已删除所有残余基站，当前测试场景切换为无残余网络。");
    showStatus("已切换为无残余网络。", "success", 2400);
    if (shouldReloadScene) await importScene();
  } catch (error) {
    const message = error?.response?.data?.detail || error?.message || String(error);
    appendTerminalLine(`无残余网络切换失败：${message}`);
    showStatus("无残余网络切换失败，请检查后端服务。", "error", 4200);
  } finally {
    isClearingResidualNetwork.value = false;
  }
};

const defaultResidualGridPosition = () => {
  const fallback = {
    x: Math.round(gridBounds.value.maxX / 2),
    y: Math.round(gridBounds.value.maxY / 2),
  };
  const clusters = Array.isArray(currentScenario.value?.user_clusters) ? currentScenario.value.user_clusters : [];
  const best = clusters
    .filter((cluster) => Array.isArray(cluster.center))
    .sort((a, b) => Number(b.density || 0) * Number(b.demand_mbps || 1) - Number(a.density || 0) * Number(a.demand_mbps || 1))[0];
  if (!best) return fallback;
  return {
    x: Math.max(0, Math.min(gridBounds.value.maxX, Math.round(Number(best.center[0] || fallback.x)))),
    y: Math.max(0, Math.min(gridBounds.value.maxY, Math.round(Number(best.center[1] || fallback.y)))),
  };
};

const addDeviceSlot = async () => {
  if (!hasImportedScene.value) {
    showStatus("请先同步/导入场景后再接入设备。", "warning", 2200);
    return;
  }
  if (!deviceRows.value.length) loadScenarioDeviceRows();
  const position = defaultResidualGridPosition();
  const target = deviceRows.value.find((row) => !row.applied) || deviceRows.value[0];
  if (!target) return;
  if (target.applied) {
    deviceRows.value.push({
      ...target,
      deviceId: `${target.deviceId}-copy-${Date.now()}`,
      name: `${target.name} ${appliedDeviceRows.value.length + 1}`,
      quantity: 1,
      applied: true,
      enabled: true,
      x: position.x,
      y: position.y,
    });
  } else {
    target.applied = true;
    target.enabled = true;
    target.quantity = 1;
    target.x = position.x;
    target.y = position.y;
  }
  await syncScenarioBaseStations();
  appendTerminalLine(`已接入设备：${target.name}。`);
};

const removeDeviceSlot = async (deviceId) => {
  const row = deviceRows.value.find((item) => item.deviceId === deviceId);
  if (!row) return;
  const shouldReloadScene = hasImportedScene.value;
  if (row.source === "disaster-import" || row.source === "scenario-deployment") {
    deviceRows.value = deviceRows.value.filter((item) => item.deviceId !== deviceId);
    await syncScenarioBaseStations();
    appendTerminalLine(`已移除场景基站：${row.name} ${row.gridText || ""}。`);
    if (!activeAppliedDeviceRows.value.length) {
      importedScene.value = null;
      simulationResult.value = null;
      activeSceneTab.value = "imported";
      appendTerminalLine("当前测试场景已切换为无残余网络。");
      if (shouldReloadScene) await importScene();
    }
    return;
  }
  row.applied = false;
  await syncScenarioBaseStations();
  appendTerminalLine(`已移除设备接入：${row.name}。`);
  if (!activeAppliedDeviceRows.value.length) {
    importedScene.value = null;
    simulationResult.value = null;
    activeSceneTab.value = "imported";
    appendTerminalLine("当前测试场景已切换为无残余网络。");
    if (shouldReloadScene) await importScene();
  }
};

const formatDeviceParams = (row) => `吞吐 ${formatMetric(row.maxThroughput, 0)} Mbps / 用户 ${formatMetric(row.maxUsers, 0)}`;

const formatDeviceLocation = (device) => {
  const position = Array.isArray(device.position) ? device.position.join(", ") : "--";
  return `${position}${device.region_label ? ` / ${device.region_label}` : ""}`;
};

const buildScenarioDeviceBaseStations = () =>
  activeAppliedDeviceRows.value.flatMap((row) =>
    Array.from({ length: Math.max(1, Number(row.quantity || 1)) }, (_, index) => {
      const x = (Math.round(Number(row.x || 0)) + index) % Math.max(1, gridBounds.value.maxX + 1);
      const y = (Math.round(Number(row.y || 0)) + index) % Math.max(1, gridBounds.value.maxY + 1);
      return row.baseStationName
        ? {
            device_uid: row.deviceId || null,
            deployment_id: row.deploymentId || null,
            base_station: row.baseStationName,
            mode: row.mode || null,
            x,
            y,
            status: row.stationStatus || (row.enabled === false ? "offline" : "active"),
            device_name: row.name || null,
            station_type: row.stationType || null,
            station_label: row.stationLabel || row.name || null,
            cell_user_count: Number(row.maxUsers || 0),
            coverage_radius_km: Number(row.coverageRadiusKm || 0),
            max_throughput: Number(row.maxThroughput || 0),
            downlink_bandwidth_mbps: Number(row.maxThroughput || 0),
            max_users: Number(row.maxUsers || 0),
          }
        : null;
    }).filter(Boolean)
  );

const downloadJson = (payload, filename) => {
  if (!payload) return;
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
};

const downloadSceneExport = (key) => {
  if (!sceneExport.value?.[key]) return;
  const suffix =
    key === "disaster_scene"
      ? "disaster_scene"
      : key === "deployment_plan"
        ? "deployment_plan"
        : "deployment_scene";
  downloadJson(sceneExport.value[key], `${scenarioName.value || "scenario"}_${suffix}.json`);
  appendTerminalEvent(`前端操作：导出${suffix} 场景文件。`, { level: "ACTION" });
};

const downloadTerminalLog = () => {
  exportTerminalOutput(terminalHistoryLines.value, "rescuenet-strategy-terminal.log");
};

const clearTerminalLog = () => {
  terminalLines.value = [];
  clearTerminalOutput();
};

const showStatus = (message, tone = "info", timeout = 4200) => {
  statusMessage.value = message;
  statusTone.value = tone;
  if (statusTimer) {
    window.clearTimeout(statusTimer);
  }
  if (timeout) {
    statusTimer = window.setTimeout(() => {
      statusMessage.value = "";
    }, timeout);
  }
};

const appendTerminalLine = (message) => {
  if (!message) return;
  terminalLines.value = appendSyncedTerminalLine(terminalLines.value, message, { level: "INFO", source: "TEST" }, 240);
};

const appendTerminalEvent = (message, options = {}) => {
  if (!message) return;
  terminalLines.value = appendSyncedTerminalLine(
    terminalLines.value,
    message,
    { level: options.level || "INFO", source: options.source || "TEST", timestamp: options.timestamp },
    240
  );
};

const appendStrategyUserNodeCount = (prefix, ...sources) => {
  const key = userNodeCountLogKey(`strategy:${prefix}`, ...sources);
  if (key === lastStrategyUserNodeLogKey) return;
  lastStrategyUserNodeLogKey = key;
  appendTerminalEvent(buildUserNodeCountMessage(prefix, ...sources), { level: "SCENE" });
};

const readHistory = () => {
  try {
    const raw = window.localStorage.getItem(TEST_HISTORY_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
};

const writeHistory = (items) => {
  window.localStorage.setItem(TEST_HISTORY_KEY, JSON.stringify(items.slice(0, 50)));
  historyRows.value = readHistory();
};

const persistTestHistory = (result) => {
  const report = Array.isArray(result?.reports) ? result.reports[0] : null;
  const finalState = report?.final_state || {};
  const sceneExport = result?.scene_export || {};
  const record = {
    id: `${Date.now()}-${scenarioName.value}-${selectedAlgorithm.value}`,
    scenarioName: scenarioName.value,
    scenarioLabel: scenarioLabel(currentScenario.value),
    algorithm: selectedAlgorithm.value,
    algorithmLabel: algorithmLabel(selectedAlgorithm.value),
    checkpointPath: checkpointPath.value,
    createdAt: new Date().toISOString(),
    avgReward: result?.avg_reward,
    avgFinalCoverage: result?.avg_final_coverage,
    broadcastRatio: finalState.broadcast_ratio,
    userCount: finalState.total_users || summary.value.users,
    disasterScenePath: sceneExport.disaster_scene_path,
    deploymentScenePath: sceneExport.deployment_scene_path,
    deploymentPlanPath: sceneExport.deployment_plan_path,
    deviceRows: finalState.user_details || [],
  };
  writeHistory([record, ...readHistory()]);
};

const syncCheckpointPath = () => {
  checkpointPath.value = matchingCheckpoint.value?.checkpoint_path || "";
};

const syncSelectableAlgorithm = () => {
  if (matchingCheckpoint.value) {
    syncCheckpointPath();
    return;
  }
  const firstAvailable = algorithmOptions.value.find((option) => option.available);
  if (firstAvailable) {
    selectedAlgorithm.value = firstAvailable.value;
    checkpointPath.value = firstAvailable.checkpointPath;
    return;
  }
  checkpointPath.value = "";
};

const handleAlgorithmChange = () => {
  syncCheckpointPath();
  simulationResult.value = null;
  activeSceneTab.value = "imported";
};

const selectAlgorithmForTest = (option) => {
  if (!option?.available || isRunning.value || isLoading.value) return;
  selectedAlgorithm.value = option.value;
  checkpointPath.value = option.checkpointPath;
  handleAlgorithmChange();
  appendTerminalEvent(`切换策略算法：${option.label}，checkpoint=${option.checkpointPath || "未匹配"}`, { level: "ACTION" });
};

const fetchScenarios = async () => {
  appendTerminalEvent("前端操作：加载策略测试场景列表。", { level: "ACTION" });
  const { data } = await axios.get(`${API_BASE}/scenarios`, { timeout: SCENE_ACCESS_TIMEOUT_MS });
  scenarios.value = Array.isArray(data?.scenarios) ? data.scenarios : [];
  appendTerminalEvent(`后端响应：策略测试场景 ${scenarios.value.length} 个。`, { level: "BACKEND", source: "BACKEND" });
};

const fetchTrainingArtifacts = async () => {
  appendTerminalEvent("前端操作：加载训练权重列表。", { level: "ACTION" });
  const { data } = await axios.get(`${API_BASE}/train/artifacts`, { timeout: SCENE_ACCESS_TIMEOUT_MS });
  trainingArtifacts.value = Array.isArray(data?.artifacts) ? data.artifacts : [];
  appendTerminalEvent(`后端响应：训练权重 ${trainingArtifacts.value.length} 条。`, { level: "BACKEND", source: "BACKEND" });
};

const normalizeDisasterImport = (record) => {
  if (!record) return null;
  return {
    ...record,
    disaster_scenario_label:
      formatPlainDisasterName(record.disaster_scenario, record.disaster_scenario_label) ||
      record.disaster_scenario_label ||
      record.disaster_scenario,
    disaster_severity_label: record.disaster_severity_label || DISASTER_SEVERITY_LABELS[record.disaster_severity] || record.disaster_severity,
  };
};

const loadDisasterScenarioDetail = async (key) => {
  if (!key) return null;
  if (disasterScenarioDetails.value[key]) return disasterScenarioDetails.value[key];
  const { data } = await axios.get(`${API_BASE}/disaster-scenarios/${encodeURIComponent(key)}`, { timeout: SCENE_ACCESS_TIMEOUT_MS });
  disasterScenarioDetails.value = { ...disasterScenarioDetails.value, [key]: data };
  return data;
};

const loadDisasterSeverityOverview = async () => {
  if (!selectedDisasterScenario.value || !selectedDisasterSeverity.value) return null;
  const { data } = await axios.get(
    `${API_BASE}/disaster-scenarios/${encodeURIComponent(selectedDisasterScenario.value)}/severity-levels/${encodeURIComponent(
      selectedDisasterSeverity.value
    )}`,
    { timeout: SCENE_ACCESS_TIMEOUT_MS }
  );
  disasterSeverityOverview.value = data;
  return data;
};

const loadDisasterImportDetail = async (importId) => {
  if (!importId) return null;
  if (disasterImportDetails.value[importId]) return disasterImportDetails.value[importId];
  const { data } = await axios.get(`${API_BASE}/disaster-imports/${encodeURIComponent(importId)}`, { timeout: SCENE_ACCESS_TIMEOUT_MS });
  const normalized = normalizeDisasterImport(data);
  disasterImportDetails.value = { ...disasterImportDetails.value, [importId]: normalized };
  return normalized;
};

const loadDisasterCatalogAndImports = async () => {
  disasterLoading.value = true;
  disasterError.value = "";
  try {
    const [scenarioResponse, importsResponse] = await Promise.all([
      axios.get(`${API_BASE}/disaster-scenarios`, { timeout: SCENE_ACCESS_TIMEOUT_MS }),
      axios.get(`${API_BASE}/disaster-imports`, { timeout: SCENE_ACCESS_TIMEOUT_MS }),
    ]);
    const loadedScenarios = Array.isArray(scenarioResponse.data?.scenarios)
      ? scenarioResponse.data.scenarios
      : Array.isArray(scenarioResponse.data?.disaster_scenarios)
        ? scenarioResponse.data.disaster_scenarios
        : [];
    disasterScenarios.value = loadedScenarios.length ? loadedScenarios : FALLBACK_DISASTER_SCENARIOS;
    disasterImports.value = (Array.isArray(importsResponse.data?.imports) ? importsResponse.data.imports : [])
      .map(normalizeDisasterImport)
      .filter(Boolean)
      .reverse();

    if (!selectedDisasterScenario.value && disasterScenarioOptions.value.length) {
      selectedDisasterScenario.value =
        disasterScenarioOptions.value.find((item) => item.key === "extreme_rainstorm")?.key || disasterScenarioOptions.value[0].key;
    }
    if (!selectedDisasterScenario.value && disasterScenarioOptions.value.length) {
      selectedDisasterScenario.value = disasterScenarioOptions.value[0].key;
    }
    if (!selectedDisasterScenario.value) {
      throw new Error("灾害场景目录为空");
    }
    await loadDisasterScenarioDetail(selectedDisasterScenario.value);
    if (!disasterSeverityOptions.value.some((item) => item.key === selectedDisasterSeverity.value)) {
      selectedDisasterSeverity.value = preferredDisasterSeverityOptionKey();
    }
    await loadDisasterSeverityOverview();
    if (!selectedDisasterImportId.value && disasterImports.value.length) {
      selectedDisasterImportId.value = disasterImports.value[0].import_id;
      await loadDisasterImportDetail(selectedDisasterImportId.value);
    }
    if (!activeDisasterImportId.value && selectedDisasterImportId.value) {
      await selectDisasterImport(selectedDisasterImportId.value, true);
    }
  } catch (error) {
    disasterError.value = error?.response?.data?.detail || error?.message || String(error);
    appendTerminalLine(`灾害数据接入初始化失败：${disasterError.value}`);
  } finally {
    disasterLoading.value = false;
  }
};

const handleDisasterScenarioChange = async () => {
  disasterSeverityOverview.value = null;
  try {
    await loadDisasterScenarioDetail(selectedDisasterScenario.value);
    selectedDisasterSeverity.value = preferredDisasterSeverityOptionKey();
    await loadDisasterSeverityOverview();
  } catch (error) {
    disasterError.value = error?.response?.data?.detail || error?.message || String(error);
  }
};

const handleDisasterSeverityChange = async () => {
  try {
    await loadDisasterSeverityOverview();
  } catch (error) {
    disasterError.value = error?.response?.data?.detail || error?.message || String(error);
  }
};

const clearDisasterImportProgressTimers = () => {
  if (disasterImportProgressTimer) {
    window.clearInterval(disasterImportProgressTimer);
    disasterImportProgressTimer = null;
  }
  if (disasterImportProgressHideTimer) {
    window.clearTimeout(disasterImportProgressHideTimer);
    disasterImportProgressHideTimer = null;
  }
};

const startDisasterImportProgress = () => {
  clearDisasterImportProgressTimers();
  disasterImportProgress.value = 8;
  disasterImportProgressTone.value = "running";
  disasterImportStage.value = "读取灾害场景";
  disasterImportProgressTimer = window.setInterval(() => {
    const current = Number(disasterImportProgress.value) || 0;
    const next = Math.min(92, current + (current < 40 ? 9 : current < 72 ? 6 : 3));
    disasterImportProgress.value = next;
    if (next >= 82) {
      disasterImportStage.value = "等待服务返回";
    } else if (next >= 62) {
      disasterImportStage.value = "生成仿真场景";
    } else if (next >= 38) {
      disasterImportStage.value = "同步基站状态";
    } else if (next >= 18) {
      disasterImportStage.value = "解析网格热力图";
    }
  }, 420);
};

const finishDisasterImportProgress = (succeeded) => {
  clearDisasterImportProgressTimers();
  disasterImportProgress.value = 100;
  disasterImportProgressTone.value = succeeded ? "success" : "error";
  disasterImportStage.value = succeeded ? "导入完成" : "导入失败";
  disasterImportProgressHideTimer = window.setTimeout(() => {
    disasterImportProgress.value = 0;
    disasterImportProgressTone.value = "idle";
    disasterImportStage.value = "准备导入";
    disasterImportProgressHideTimer = null;
  }, succeeded ? 1200 : 1800);
};

const createDisasterImport = async () => {
  if (!selectedDisasterScenario.value || !selectedDisasterSeverity.value || disasterImporting.value) return;
  disasterImporting.value = true;
  disasterError.value = "";
  startDisasterImportProgress();
  let importSucceeded = false;
  let progressSettled = false;
  appendTerminalEvent(
    `前端操作：导入灾害数据 scenario=${selectedDisasterScenario.value} severity=${selectedDisasterSeverity.value} sample_limit=${disasterSessionSampleLimit.value}。`,
    { level: "ACTION" }
  );
  try {
    const { data } = await axios.post(
      `${API_BASE}/disaster-imports`,
      {
        disaster_scenario: selectedDisasterScenario.value,
        disaster_severity: selectedDisasterSeverity.value,
        session_sample_limit: Math.max(1, Math.min(500, Number(disasterSessionSampleLimit.value) || 100)),
      },
      { timeout: SCENE_ACCESS_TIMEOUT_MS }
    );
    const summary = normalizeDisasterImport(data);
    disasterImports.value = [summary, ...disasterImports.value.filter((item) => item.import_id !== summary.import_id)];
    appendTerminalEvent(`后端响应：灾害数据导入完成 ${summary.disaster_scenario}/${summary.disaster_severity}，import_id=${summary.import_id}。`, {
      level: "BACKEND",
      source: "BACKEND",
    });
    appendStrategyUserNodeCount(
      `策略测试导入灾害数据：${summary.disaster_scenario_label || summary.disaster_scenario}/${summary.disaster_severity_label || summary.disaster_severity}`,
      summary
    );
    importSucceeded = true;
    disasterImporting.value = false;
    finishDisasterImportProgress(true);
    progressSettled = true;
    await selectDisasterImport(summary.import_id, true);
  } catch (error) {
    disasterError.value = error?.response?.data?.detail || error?.message || String(error);
    appendTerminalEvent(`后端响应：灾害数据导入失败：${disasterError.value}`, { level: "ERROR", source: "BACKEND" });
  } finally {
    disasterImporting.value = false;
    if (!progressSettled) finishDisasterImportProgress(importSucceeded);
  }
};

const disasterScenarioNameForImport = (record) =>
  record?.disaster_scenario && record?.disaster_severity ? `${record.disaster_scenario}__${record.disaster_severity}` : "";

const applyDisasterImportToSimulation = async (record) => {
  if (!record) return;
  const normalized = normalizeDisasterImport(record);
  if (!normalized?.import_id) return;
  disasterImportDetails.value = {
    ...disasterImportDetails.value,
    [normalized.import_id]: {
      ...(disasterImportDetails.value[normalized.import_id] || {}),
      ...normalized,
    },
  };
  disasterImports.value = [
    normalized,
    ...disasterImports.value.filter((item) => item.import_id !== normalized.import_id),
  ];
  activeDisasterImportId.value = normalized.import_id;
  const nextScenarioName = disasterScenarioNameForImport(record);
  const matched = scenarios.value.find((scenario) => scenario.name === nextScenarioName);
  appendTerminalLine(`已选择灾害场景 ${record.import_id} 作为策略仿真输入。`);
  appendStrategyUserNodeCount(
    `策略测试接入灾害场景：${normalized.disaster_scenario_label || normalized.disaster_scenario}/${normalized.disaster_severity_label || normalized.disaster_severity}`,
    normalized
  );
  if (nextScenarioName && !matched && !scenarios.value.length) {
    importedScene.value = null;
    simulationResult.value = null;
    activeSceneTab.value = "imported";
    appendTerminalLine(`等待训练场景列表加载后再同步 ${nextScenarioName}。`);
    return;
  }
  if (matched) {
    scenarioName.value = nextScenarioName;
    importedScene.value = null;
    simulationResult.value = null;
    activeSceneTab.value = "imported";
    syncSelectableAlgorithm();
  } else if (nextScenarioName) {
    appendTerminalLine(`训练场景列表未找到 ${nextScenarioName}，当前附带 import_id 参与仿真，不自动切换场景。`);
  }
  importedScene.value = null;
  simulationResult.value = null;
  activeSceneTab.value = "imported";
  applyDisasterDeviceRows(record);
  await importScene();
};

const selectDisasterImport = async (importId, applyToSimulation) => {
  if (!importId) return;
  selectedDisasterImportId.value = importId;
  try {
    const detail = await loadDisasterImportDetail(importId);
    if (applyToSimulation) {
      await applyDisasterImportToSimulation(detail);
    }
  } catch (error) {
    disasterError.value = error?.response?.data?.detail || error?.message || String(error);
    appendTerminalLine(`读取灾害导入详情失败：${disasterError.value}`);
  }
};

const deleteDisasterImport = async (importId) => {
  if (!importId) return;
  try {
    await axios.delete(`${API_BASE}/disaster-imports/${encodeURIComponent(importId)}`, { timeout: SCENE_ACCESS_TIMEOUT_MS });
    disasterImports.value = disasterImports.value.filter((item) => item.import_id !== importId);
    const nextDetails = { ...disasterImportDetails.value };
    delete nextDetails[importId];
    disasterImportDetails.value = nextDetails;
    if (selectedDisasterImportId.value === importId) {
      selectedDisasterImportId.value = disasterImports.value[0]?.import_id || "";
      if (selectedDisasterImportId.value) await loadDisasterImportDetail(selectedDisasterImportId.value);
    }
    if (activeDisasterImportId.value === importId) {
      activeDisasterImportId.value = "";
      importedScene.value = null;
      simulationResult.value = null;
      activeSceneTab.value = "imported";
      loadScenarioDeviceRows();
      await importScene();
    }
    appendTerminalLine(`已移除灾害场景：${importId}。`);
  } catch (error) {
    disasterError.value = error?.response?.data?.detail || error?.message || String(error);
    appendTerminalLine(`移除灾害场景失败：${disasterError.value}`);
  }
};

const importScene = async () => {
  if (!scenarioName.value) return false;
  isLoading.value = true;
  terminalStatus.value = "importing";
  appendTerminalEvent(`开始同步策略测试场景：${comboText.value}`, { level: "ACTION" });
  try {
    await refreshScenarioDeviceState({ preserveDisasterImportRows: false });
    const customBaseStations = buildScenarioDeviceBaseStations();
    const { data } = await axios.post(
      `${API_BASE}/simulate/scene`,
      {
        scenario_name: scenarioName.value,
        env_type: "multimodal",
        evaluation_protocol: evaluationProtocol.value,
        dataset_import_ids: activeDatasetImportIds.value,
        custom_base_stations: customBaseStations,
      },
      { timeout: SCENE_ACCESS_TIMEOUT_MS }
    );
    importedScene.value = data;
    activeSceneTab.value = "imported";
    appendTerminalEvent(`后端已返回场景快照：${comboText.value}，设备 ${customBaseStations.length} 台。`, { level: "BACKEND" });
    appendStrategyUserNodeCount(`策略测试场景快照已接入：${comboText.value}`, data, activeDisasterImport.value, currentScenario.value);
    showStatus("场景就绪", "success", 2200);
    terminalStatus.value = "idle";
    return true;
  } catch (error) {
    appendTerminalEvent(`场景同步失败：${error?.response?.data?.detail || error?.message || error}`, { level: "ERROR" });
    showStatus("场景同步失败，请检查后端服务状态。", "error");
    terminalStatus.value = "failed";
    return false;
  } finally {
    isLoading.value = false;
  }
};

const selectDeploymentTab = () => {
  if (!simulationResult.value?.scene_export?.deployment_scene) {
    activeSceneTab.value = "imported";
    showStatus("当前还没有部署后场景，请先完成一次测试。", "warning", 2600);
    return;
  }
  activeSceneTab.value = "deployment";
};

const readErrorResponse = async (response) => {
  const rawText = await response.text();
  if (!rawText) return `请求失败 (${response.status})`;
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
    if (payload.state === "initializing") terminalStatus.value = "loading";
    if (payload.state === "running") terminalStatus.value = "running";
    if (payload.state === "completed") terminalStatus.value = "completed";
    if (payload.state === "failed") terminalStatus.value = "failed";
    appendTerminalEvent(`后端状态：${payload.state || "unknown"}`, { level: "STATUS", source: "BACKEND", timestamp: event.timestamp });
    return;
  }
  if (event.type === "log") {
    appendTerminalEvent(payload.message, { level: "BACKEND", source: "BACKEND", timestamp: event.timestamp });
    return;
  }
  if (event.type === "result") {
    simulationResult.value = payload;
    persistTestHistory(payload);
    appendStationRecoveryTerminal(recoverySummaryFromResult(payload));
    if (payload?.replay_session_id) {
      setActiveReplaySessionId(payload.replay_session_id);
      appendTerminalEvent(`后端回放会话已生成：${payload.replay_session_id}。`, { level: "REPLAY" });
    } else {
      saveReplaySessionFromSimulation({
        scenarioName: scenarioName.value,
        algorithm: selectedAlgorithm.value,
        result: payload,
      });
    }
    activeSceneTab.value = payload?.scene_export?.deployment_scene ? "deployment" : "imported";
    terminalStatus.value = "completed";
    appendTerminalEvent(`测试完成：平均奖励 ${formatMetric(payload.avg_reward, 2)}，覆盖率 ${formatPercent(payload.avg_final_coverage)}。`, { level: "RESULT" });
    refreshScenarioDeviceState({ preserveDisasterImportRows: false }).then((synced) => {
      if (synced) {
        appendTerminalEvent("设备接入列表已同步为策略测试终态。", { level: "DEVICE", source: "BACKEND" });
      }
    });
    showStatus(`测试完成：平均奖励 ${formatMetric(payload.avg_reward, 2)}，覆盖率 ${formatPercent(payload.avg_final_coverage)}。`, "success", 0);
    return;
  }
  if (event.type === "error") {
    terminalStatus.value = "failed";
    appendTerminalEvent(`测试失败：${payload.message || "未知错误"}`, { level: "ERROR", source: "BACKEND", timestamp: event.timestamp });
    showStatus(payload.message || "测试执行失败。", "error", 0);
  }
};

const processSseChunk = (chunk) => {
  const payloadText = chunk
    .split("\n")
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).trim())
    .join("\n");
  if (!payloadText) return;
  try {
    handleSimulationEvent(JSON.parse(payloadText));
  } catch {
    appendTerminalEvent(`无法解析流式结果：${payloadText}`, { level: "ERROR", source: "BACKEND" });
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
    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const chunk = buffer.slice(0, boundary).trim();
      buffer = buffer.slice(boundary + 2);
      if (chunk) processSseChunk(chunk);
      boundary = buffer.indexOf("\n\n");
    }
  }

  const tail = buffer.trim();
  if (tail) processSseChunk(tail);
};

const runSimulation = async () => {
  if (isRunning.value) return;
  syncCheckpointPath();
  if (!hasActiveDisasterImport.value) {
    showStatus("请先导入灾害数据并用于仿真。", "warning", 2600);
    appendTerminalEvent("启动被拦截：请先导入灾害数据并用于仿真。", { level: "WARN" });
    return;
  }
  if (!checkpointPath.value) {
    showStatus("当前场景与算法没有匹配的训练权重，请先完成训练。", "error", 0);
    appendTerminalEvent("启动被拦截：当前场景与算法没有匹配的训练权重。", { level: "ERROR" });
    return;
  }

  if (!importedScene.value) {
    const imported = await importScene();
    if (!imported) return;
  }
  await refreshScenarioDeviceState();
  const customBaseStations = buildScenarioDeviceBaseStations();

  isRunning.value = true;
  simulationResult.value = null;
  activeSceneTab.value = "imported";
  terminalLines.value = [];
  terminalStatus.value = "running";
  showStatus("测试中", "warning", 0);
  appendTerminalEvent(`准备启动 ${comboText.value} 的策略测试。`, { level: "ACTION" });
  if (activeDatasetImportIds.value.length) {
    appendTerminalEvent(`灾害数据来源 import_id=${activeDatasetImportIds.value.join(", ")}。`, { level: "CONFIG" });
  }
  appendTerminalEvent(`当前场景已应用设备：${customBaseStations.length} 台。`, { level: "CONFIG" });

  try {
    const response = await fetch(`${API_BASE}/simulate/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        scenario_name: scenarioName.value,
        algorithm: selectedAlgorithm.value,
        checkpoint_path: checkpointPath.value,
        env_type: "multimodal",
        reward_mode: matchingCheckpoint.value?.reward_mode || null,
        evaluation_protocol: evaluationProtocol.value,
        episodes: 1,
        stochastic_eval: true,
        eval_seed: 13,
        dataset_import_ids: activeDatasetImportIds.value,
        custom_base_stations: customBaseStations,
        custom_devices:
          importedScene.value?.initial_state?.user_details
            ?.filter((device) => Array.isArray(device.position) && device.position.length >= 2)
            .map((device) => ({
              x: Number(device.position[0]),
              y: Number(device.position[1]),
              demand: Number(device.demand || 10),
              connected: Boolean(device.connected),
              broadcast_served: Boolean(device.broadcast_served),
            })) || [],
      }),
    });

    if (!response.ok) {
      throw new Error(await readErrorResponse(response));
    }

    await consumeSimulationStream(response);
    if (!simulationResult.value) {
      throw new Error("测试结束但未收到结果数据。");
    }
  } catch (error) {
    appendTerminalEvent(`测试执行失败：${error?.message || error}`, { level: "ERROR" });
    terminalStatus.value = "failed";
    showStatus("测试执行失败，请检查后端接口与模型权重。", "error", 0);
  } finally {
    isRunning.value = false;
    if (terminalStatus.value === "running") terminalStatus.value = simulationResult.value ? "completed" : "idle";
  }
};

onMounted(async () => {
  historyRows.value = readHistory();
  isLoading.value = true;
  try {
    await Promise.all([fetchScenarios(), fetchTrainingArtifacts()]);
    const scenarioNames = new Set(scenarios.value.map((scenario) => scenario.name));
    const latestArtifact = trainingArtifacts.value.find((artifact) => scenarioNames.has(artifact.scenario_name));
    const fallbackScenario = scenarios.value[0] || fallbackScenarios[0];
    scenarioName.value = latestArtifact?.scenario_name || fallbackScenario.name;
    selectedAlgorithm.value = latestArtifact?.algorithm || "ppo";
    syncSelectableAlgorithm();
    loadScenarioDeviceRows();
    await loadDisasterCatalogAndImports();
    if (!activeDisasterImportId.value || !importedScene.value) {
      await importScene();
    }
  } catch (error) {
    showStatus("初始化失败，请确认 /api/scenarios 和 /api/train/artifacts 可访问。", "error");
    appendTerminalLine(`初始化失败：${error?.message || error}`);
  } finally {
    isLoading.value = false;
  }
});

onBeforeUnmount(() => {
  if (statusTimer) {
    window.clearTimeout(statusTimer);
  }
  clearDisasterImportProgressTimers();
});
</script>

<style scoped>
.strategy-tester {
  position: relative;
  width: 1920px;
  height: 1010px;
  min-height: 1010px;
  overflow: hidden;
  background: #eef5ff;
  color: #1f2d3d;
  font-family: "Microsoft YaHei", "PingFang SC", "Source Han Sans CN", sans-serif;
}

.strategy-tester__bg,
.strategy-tester__panel-shadow {
  position: absolute;
  display: block;
  border: 0;
  pointer-events: none;
  user-select: none;
  z-index: 0;
}

.strategy-tester__bg {
  left: 0;
  top: 0;
  width: 1920px;
  height: 1010px;
}

.strategy-tester__panel-shadow {
  left: 97px;
  top: 0;
  width: 1740px;
  height: 1027px;
  opacity: 0.5;
}

.strategy-panel {
  position: absolute;
  left: 140px;
  top: 44px;
  width: 1652px;
  height: 930px;
  min-height: 0;
  overflow-x: hidden;
  overflow-y: auto;
  scrollbar-color: rgba(57, 97, 246, 0.45) rgba(225, 236, 255, 0.72);
  scrollbar-width: thin;
  z-index: 2;
}

.strategy-panel::-webkit-scrollbar {
  width: 8px;
}

.strategy-panel::-webkit-scrollbar-track {
  background: rgba(225, 236, 255, 0.72);
  border-radius: 999px;
}

.strategy-panel::-webkit-scrollbar-thumb {
  background: rgba(57, 97, 246, 0.45);
  border-radius: 999px;
}

.strategy-panel__scroll {
  position: relative;
  width: 1640px;
  min-height: 100%;
}

.strategy-title {
  position: absolute;
  left: 0;
  top: 0;
  width: 157px;
  height: 68px;
}

.strategy-title__ribbon {
  position: absolute;
  left: -14px;
  top: 2px;
  width: 157px;
  height: 66px;
}

.strategy-title h1 {
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

.map-shell {
  position: absolute;
  left: 3px;
  top: 760px;
  width: 1637px;
  height: 798px;
  overflow: hidden;
}

.scene-tab {
  position: absolute;
  top: 2px;
  height: 40px;
  border: 1px solid rgba(183, 224, 254, 0.95);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.18);
  color: #1f2d3d;
  font-size: 16px;
  z-index: 20;
}

.scene-tab--imported {
  left: 1px;
  width: 102px;
}

.scene-tab--deployment {
  left: 124px;
  width: 110px;
}

.scene-tab--active {
  background: rgba(0, 121, 254, 0.12);
  color: #1f2d3d;
}

.info-pill {
  position: absolute;
  top: 0;
  height: 40px;
  box-sizing: border-box;
  display: block;
  padding: 0 12px;
  overflow: hidden;
  border: 1px solid #018ed3;
  border-radius: 10px;
  background: rgba(255, 255, 255, 0.08);
  color: #0066ff;
  font-size: 16px;
  line-height: 36px;
  text-align: center;
  text-overflow: ellipsis;
  white-space: nowrap;
  z-index: 20;
}

.info-pill--region {
  left: 254px;
  width: 594px;
}

.info-pill--span {
  left: 866px;
  top: 1px;
  width: 638px;
}

.satellite-map {
  position: absolute;
  left: 1px;
  top: 53px;
  width: 1618px;
  height: 745px;
  object-fit: cover;
  z-index: 1;
}

.tile-map {
  position: absolute;
  left: 1px;
  top: 53px;
  width: 1618px;
  height: 745px;
  overflow: hidden;
  z-index: 2;
  background: #dbeafe;
  pointer-events: none;
}

.tile-map img {
  position: absolute;
  width: 256px;
  height: 256px;
  border: 0;
}

.tile-map__label {
  position: absolute;
  z-index: 3;
  border-radius: 6px;
  font-size: 12px;
}

.tile-map__label {
  left: 18px;
  top: 18px;
  padding: 8px 12px;
  background: rgba(15, 23, 42, 0.62);
  color: #fff;
  font-size: 14px;
}

.node-layer {
  position: absolute;
  inset: 0;
  z-index: 8;
  pointer-events: none;
}

.map-empty {
  position: absolute;
  left: 1px;
  top: 53px;
  width: 1618px;
  height: 745px;
  z-index: 17;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 8px;
  border: 1px dashed rgba(96, 165, 250, 0.58);
  background: rgba(248, 250, 252, 0.92);
  color: #334155;
  text-align: center;
  pointer-events: none;
}

.map-empty strong {
  color: #17315d;
  font-size: 22px;
  font-weight: 700;
  line-height: 30px;
}

.map-empty span {
  max-width: 520px;
  color: #64748b;
  font-size: 15px;
  line-height: 24px;
}

.node-marker {
  position: absolute;
  width: var(--marker-size, 4px);
  height: var(--marker-size, 4px);
  transform: translate(-50%, -50%);
  border: 0;
  border-radius: 50%;
  background: var(--marker-color, #ef4444);
  opacity: var(--marker-opacity, 0.72);
  box-shadow:
    0 0 0 1px rgba(255, 255, 255, 0.28),
    0 0 4px color-mix(in srgb, var(--marker-color, #ef4444) 42%, transparent);
}

.node-marker--user-offline,
.node-marker--user-online {
  border: 0;
  box-shadow:
    0 0 0 1px rgba(255, 255, 255, 0.2),
    0 0 5px color-mix(in srgb, var(--marker-color, #ef4444) 48%, transparent);
}

.node-marker--station {
  width: 12px;
  height: 12px;
  transform: translate(-50%, -50%);
  border: 2px solid rgba(255, 255, 255, 0.92);
  border-radius: 50%;
  background: var(--marker-color, #2563eb);
  opacity: 1;
  pointer-events: auto;
  cursor: help;
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--marker-color, #2563eb) 24%, transparent), 0 3px 8px rgba(15, 23, 42, 0.28);
}

.node-marker--station::after {
  content: "";
  position: absolute;
  inset: -10px;
  border-radius: 50%;
}

.node-marker--station:focus-visible {
  outline: 3px solid rgba(14, 165, 233, 0.72);
  outline-offset: 6px;
}

.node-marker--station-deployed {
  border-color: rgba(224, 242, 254, 0.96);
}

.node-marker--station-restored {
  border-color: rgba(187, 247, 208, 0.98);
  box-shadow:
    0 0 0 3px rgba(34, 197, 94, 0.24),
    0 3px 8px rgba(15, 23, 42, 0.28);
}

.node-marker--station-planned {
  border-color: rgba(219, 234, 254, 0.96);
}

.station-tooltip {
  position: absolute;
  z-index: 32;
  width: 248px;
  min-height: 150px;
  padding: 12px 14px 13px;
  border: 1px solid rgba(148, 163, 184, 0.32);
  border-radius: 8px;
  background: rgba(15, 23, 42, 0.92);
  color: #f8fafc;
  box-shadow: 0 16px 34px rgba(15, 23, 42, 0.28);
  pointer-events: none;
  backdrop-filter: blur(8px);
}

.station-tooltip__title {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
  margin-bottom: 10px;
}

.station-tooltip__title strong {
  min-width: 0;
  overflow: hidden;
  color: #fff;
  font-size: 14px;
  font-weight: 700;
  line-height: 20px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.station-tooltip__dot {
  flex: 0 0 auto;
  width: 10px;
  height: 10px;
  border: 2px solid rgba(255, 255, 255, 0.82);
  border-radius: 50%;
}

.station-tooltip__list {
  display: grid;
  gap: 7px;
  margin: 0;
}

.station-tooltip__list div {
  display: grid;
  grid-template-columns: 66px minmax(0, 1fr);
  align-items: center;
  gap: 10px;
}

.station-tooltip__list dt {
  color: #cbd5e1;
  font-size: 12px;
  line-height: 18px;
}

.station-tooltip__list dd {
  min-width: 0;
  margin: 0;
  overflow: hidden;
  color: #f8fafc;
  font-size: 12px;
  font-weight: 600;
  line-height: 18px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.station-tooltip__status {
  justify-self: start;
  max-width: 100%;
  padding: 1px 8px;
  border-radius: 999px;
  background: rgba(148, 163, 184, 0.2);
  color: #e2e8f0;
}

.station-tooltip__status--active,
.station-tooltip__status--deployed,
.station-tooltip__status--planned {
  background: rgba(34, 197, 94, 0.2);
  color: #bbf7d0;
}

.station-tooltip__status--degraded {
  background: rgba(245, 158, 11, 0.22);
  color: #fde68a;
}

.station-tooltip__status--offline {
  background: rgba(148, 163, 184, 0.2);
  color: #cbd5e1;
}

.station-tooltip-fade-enter-active,
.station-tooltip-fade-leave-active {
  transition: opacity 120ms ease, transform 120ms ease;
}

.station-tooltip-fade-enter-from,
.station-tooltip-fade-leave-to {
  opacity: 0;
  transform: translateY(4px);
}

.map-legend {
  position: absolute;
  right: 30px;
  bottom: 34px;
  z-index: 22;
  display: grid;
  grid-template-columns: repeat(2, minmax(110px, max-content));
  gap: 8px 12px;
  padding: 10px 12px;
  border: 1px solid rgba(125, 211, 252, 0.5);
  border-radius: 6px;
  background: rgba(21, 63, 149, 0.72);
  color: #fff;
  font-size: 12px;
  line-height: 18px;
}

.map-legend__item {
  display: flex;
  align-items: center;
  gap: 7px;
  min-width: 0;
  white-space: nowrap;
}

.map-legend__mark {
  width: 12px;
  height: 12px;
  flex: 0 0 auto;
  border: 1px solid rgba(255, 255, 255, 0.82);
  box-shadow: 0 1px 3px rgba(15, 23, 42, 0.22);
}

.map-legend__mark--circle {
  border-radius: 50%;
}

.map-legend__mark--square {
  border-radius: 3px;
}

.metric-card {
  position: absolute;
  top: 69px;
  width: 130px;
  height: 80px;
  display: flex;
  flex-direction: column;
  align-items: center;
  background: rgba(21, 63, 149, 0.63);
  color: #fff;
  z-index: 18;
}

.metric-card--nodes {
  left: 1170px;
}

.metric-card--users {
  left: 1320px;
}

.metric-card--stations {
  left: 1471px;
}

.metric-card span {
  margin-top: 8px;
  height: 32px;
  font-size: 16px;
  line-height: 32px;
}

.metric-card strong {
  color: #ff9900;
  font-family: "Microsoft YaHei", sans-serif;
  font-size: 24px;
  font-weight: 400;
  line-height: 34px;
}

.status-toast {
  position: absolute;
  left: 8px;
  bottom: 14px;
  z-index: 40;
  min-width: 360px;
  max-width: 900px;
  min-height: 40px;
  padding: 9px 14px;
  border: 1px solid #b7e0fe;
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.94);
  color: #0066ff;
  font-size: 15px;
  box-shadow: 3px 3px 20px rgba(233, 233, 233, 0.8);
}

.status-toast--success {
  color: #15803d;
}

.status-toast--warning {
  color: #b45309;
}

.status-toast--error {
  color: #b91c1c;
}

.module-panel {
  position: absolute;
  left: 4px;
  width: 1628px;
  box-sizing: border-box;
  border: 1px solid rgba(233, 233, 233, 1);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.96);
  box-shadow: 3px 3px 20px rgba(233, 233, 233, 0.9);
  color: #334155;
  font-size: 14px;
}

.module-panel--scenario {
  top: 84px;
  height: 920px;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  padding: 14px;
}

.module-panel--device {
  top: 1518px;
  min-height: 300px;
  padding: 14px;
}

.module-panel--algorithm {
  min-height: 178px;
  padding: 14px;
}

.module-panel--result {
  top: 1840px;
  min-height: 470px;
  padding: 14px;
}

.module-heading {
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

.module-heading h2 {
  margin: 0;
  color: #1f2d3d;
  font-size: 16px;
  font-weight: 400;
}

.module-heading p {
  min-width: 0;
  max-width: 780px;
  margin: 0 0 0 8px;
  color: #64748b;
  font-size: 12px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.algorithm-controls {
  width: 100%;
  display: flex;
  align-items: center;
  gap: 14px;
}

.algorithm-card-grid {
  flex: 1 1 auto;
  min-width: 0;
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 14px;
}

.algorithm-card {
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

.algorithm-card:hover:not(:disabled) {
  border-color: #1890ff;
  background: rgba(231, 238, 255, 0.5);
  color: #333333;
}

.algorithm-card--active {
  border-color: #3961f6;
  background: rgba(231, 238, 255, 0.5);
  color: #3961f6;
  box-shadow: 0 0 6px rgba(57, 97, 246, 0.15);
}

.algorithm-card--disabled {
  cursor: not-allowed;
  opacity: 0.45;
  filter: grayscale(0.25);
}

.algorithm-card__name {
  max-width: 100%;
  font-size: 16px;
  font-weight: 700;
  line-height: 22px;
  text-align: center;
}

.algorithm-card__desc {
  max-width: 100%;
  overflow: hidden;
  font-size: 12px;
  line-height: 18px;
  opacity: 0.7;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.algorithm-start-button {
  min-width: 112px;
  height: 44px;
  align-self: center;
}

.module-actions {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.primary-button,
.ghost-button,
.replay-button {
  height: 34px;
  padding: 0 14px;
  border-radius: 6px;
  font-size: 14px;
  text-decoration: none;
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

.ghost-button--blue,
.replay-button {
  border: 1px solid #b7e0fe;
  background: #ebf5ff;
  color: #2563eb;
}

.ghost-button--danger {
  border-color: #fecaca;
  background: #fff7f7;
  color: #b91c1c;
}

.primary-button:disabled,
.ghost-button:disabled {
  opacity: 0.55;
}

.module-error {
  margin-bottom: 10px;
  border: 1px solid #fecaca;
  border-radius: 6px;
  background: #fef2f2;
  color: #991b1b;
  padding: 8px 10px;
}

.dataset-controls {
  flex: 0 0 auto;
  display: grid;
  grid-template-columns: minmax(0, 1fr) 420px;
  grid-template-areas:
    "scenario import"
    "severity import";
  gap: 12px;
  align-items: stretch;
  margin-bottom: 12px;
}

.dataset-choice--scenario {
  grid-area: scenario;
}

.dataset-choice--severity {
  grid-area: severity;
}

.dataset-import-card {
  grid-area: import;
}

.dataset-choice,
.dataset-import-card {
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

.dataset-option-grid--scenario {
  grid-template-columns: repeat(4, minmax(0, 1fr));
}

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

.dataset-option-card:hover:not(:disabled) {
  border-color: #1890ff;
  background: rgba(231, 238, 255, 0.62);
}

.dataset-option-card--active {
  border-color: #3961f6;
  background: rgba(231, 238, 255, 0.72);
  color: #3961f6;
  box-shadow: 0 0 6px rgba(57, 97, 246, 0.16);
}

.dataset-option-card:disabled {
  cursor: not-allowed;
  opacity: 0.6;
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

.dataset-import-card {
  display: grid;
  grid-template-rows: auto auto 1fr;
  gap: 12px;
}

.dataset-import-card label,
.device-table label {
  display: flex;
  flex-direction: column;
  gap: 6px;
  color: #334155;
  font-weight: 700;
  font-size: 13px;
}

.dataset-import-card input,
.device-table select,
.device-table input {
  width: 100%;
  height: 34px;
  border: 1px solid #d7e3f4;
  border-radius: 6px;
  background: #fff;
  color: #17315d;
  padding: 0 10px;
}

.dataset-import-card .primary-button {
  width: 100%;
}

.import-progress {
  align-self: end;
  min-height: 56px;
  border: 1px solid rgba(183, 224, 254, 0.72);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.74);
  padding: 10px;
}

.import-progress__meta {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  color: #334155;
  font-size: 12px;
  line-height: 16px;
}

.import-progress__meta strong {
  color: #3961f6;
  font-variant-numeric: tabular-nums;
}

.import-progress__track {
  height: 10px;
  margin-top: 9px;
  overflow: hidden;
  border-radius: 999px;
  background: #e2e8f0;
}

.import-progress__track span {
  display: block;
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, #3961f6, #06b6d4);
  transition: width 0.24s ease;
}

.import-progress--success .import-progress__track span {
  background: linear-gradient(90deg, #16a34a, #22c55e);
}

.import-progress--error {
  border-color: #fecaca;
}

.import-progress--error .import-progress__meta strong {
  color: #b91c1c;
}

.import-progress--error .import-progress__track span {
  background: linear-gradient(90deg, #ef4444, #f97316);
}

.dataset-main {
  flex: 1 1 auto;
  min-height: 0;
  display: grid;
  grid-template-columns: minmax(0, 1fr) 420px;
  gap: 12px;
}

.dataset-map-card,
.compact-card,
.result-box {
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  background: #fff;
  padding: 12px;
}

.dataset-map-card {
  min-height: 0;
  display: flex;
  flex-direction: column;
  align-items: stretch;
  overflow: hidden;
}

.dataset-summary {
  flex: 0 0 auto;
  display: grid;
  grid-template-columns: minmax(0, 1fr) 620px;
  gap: 12px;
  margin-bottom: 10px;
}

.dataset-summary h3 {
  margin: 0 0 6px;
  color: #111827;
  font-size: 17px;
}

.dataset-summary p {
  margin: 0 0 4px;
  color: #64748b;
  font-size: 13px;
  line-height: 1.5;
}

.mini-metrics {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 8px;
}

.mini-metrics--two {
  grid-template-columns: repeat(2, minmax(0, 1fr));
  margin-top: 10px;
}

.mini-metrics span,
.result-metrics span {
  display: block;
  min-width: 0;
  border: 1px solid #e2e8f0;
  border-radius: 7px;
  background: #f8fafc;
  padding: 9px 10px;
}

.mini-metrics small,
.result-metrics small {
  display: block;
  color: #64748b;
  font-size: 12px;
}

.mini-metrics strong,
.result-metrics strong {
  display: block;
  margin-top: 4px;
  color: #0f172a;
  font-size: 18px;
  font-weight: 700;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.recovery-card {
  display: flex;
  flex-direction: column;
  gap: 10px;
  border: 1px solid #bbf7d0;
  border-radius: 8px;
  background: #f0fdf4;
  padding: 12px;
}

.recovery-card__title {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  color: #14532d;
}

.recovery-card__title strong {
  font-size: 15px;
}

.recovery-card__title span {
  color: #166534;
  font-size: 12px;
}

.recovery-card__metrics {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 8px;
}

.recovery-card__metrics span {
  min-width: 0;
  border: 1px solid rgba(34, 197, 94, 0.22);
  border-radius: 7px;
  background: rgba(255, 255, 255, 0.82);
  padding: 8px 10px;
}

.recovery-card__metrics small {
  display: block;
  color: #166534;
  font-size: 12px;
}

.recovery-card__metrics strong {
  display: block;
  margin-top: 4px;
  color: #0f172a;
  font-size: 14px;
  line-height: 20px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.recovery-card__events {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 6px 10px;
}

.recovery-card__events span {
  overflow: hidden;
  color: #166534;
  font-size: 12px;
  line-height: 18px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.dataset-visual-toolbar {
  flex: 0 0 auto;
  min-height: 34px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 8px;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  background: #f8fafc;
  padding: 6px 10px;
}

.heat-legend,
.station-status-legend {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
  color: #475569;
  font-size: 12px;
  line-height: 16px;
}

.heat-legend span {
  color: #334155;
  font-weight: 700;
}

.heat-legend i {
  width: 128px;
  height: 8px;
  border-radius: 999px;
  background: linear-gradient(90deg, rgba(37, 99, 235, 0.26), rgba(34, 197, 94, 0.42), rgba(245, 158, 11, 0.7), rgba(239, 68, 68, 0.78));
  box-shadow: inset 0 0 0 1px rgba(15, 23, 42, 0.08);
}

.station-status-legend {
  justify-content: flex-end;
  flex-wrap: wrap;
}

.station-status-legend span {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  white-space: nowrap;
}

.station-status-dot {
  width: 9px;
  height: 9px;
  border-radius: 50%;
  box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.9), 0 1px 4px rgba(15, 23, 42, 0.22);
}

.station-status-dot--active {
  background: #16a34a;
}

.station-status-dot--degraded {
  background: #f59e0b;
}

.station-status-dot--offline {
  background: #dc2626;
}

.station-status-dot--unknown {
  background: #64748b;
}

.dataset-grid {
  position: relative;
  flex: 0 0 auto;
  align-self: center;
  width: min(100%, 700px);
  height: 400px;
  max-height: 400px;
  overflow: hidden;
  border: 1px solid #bfd4ee;
  border-radius: 10px;
  background:
    radial-gradient(circle at 18% 20%, rgba(255, 255, 255, 0.86), transparent 28%),
    radial-gradient(circle at 76% 68%, rgba(190, 224, 255, 0.28), transparent 32%),
    linear-gradient(180deg, #eaf3ff, #f8fafc 56%, #edf7f3);
  box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.72);
}

.dataset-grid::before {
  content: "";
  position: absolute;
  inset: 0;
  z-index: 0;
  pointer-events: none;
  background-image:
    linear-gradient(to right, rgba(51, 65, 85, 0.012) 1px, transparent 1px),
    linear-gradient(to bottom, rgba(51, 65, 85, 0.012) 1px, transparent 1px);
  background-size: calc(100% / var(--grid-cols)) calc(100% / var(--grid-rows));
}

.dataset-heat {
  position: absolute;
  z-index: 1;
  border-radius: 50%;
  filter: blur(14px);
  mix-blend-mode: multiply;
  transform: translate(-50%, -50%);
  transform-origin: center;
  pointer-events: none;
}

.dataset-station {
  position: absolute;
  z-index: 3;
  width: 18px;
  height: 18px;
  display: block;
  border: 3px solid #fff;
  border-radius: 50%;
  background: #fff;
  transform: translate(-50%, -50%);
  box-shadow: 0 2px 8px rgba(15, 23, 42, 0.28), 0 0 0 2px rgba(255, 255, 255, 0.62);
}

.dataset-station i {
  position: absolute;
  inset: 2px;
  border-radius: 50%;
  box-shadow: inset 0 0 0 1px rgba(15, 23, 42, 0.08);
}

.dataset-station--active i {
  background: #16a34a;
}

.dataset-station--degraded i {
  background: #f59e0b;
}

.dataset-station--offline i {
  background: #dc2626;
}

.dataset-station--unknown i {
  background: #64748b;
}

.dataset-station--active {
  color: #166534;
}

.dataset-station--degraded {
  color: #92400e;
}

.dataset-station--offline {
  color: #991b1b;
}

.dataset-station--unknown {
  color: #475569;
}

.dataset-empty {
  position: absolute;
  inset: 0;
  display: grid;
  place-items: center;
  color: #64748b;
}

.dataset-side {
  display: flex;
  flex-direction: column;
  gap: 10px;
  min-width: 0;
  min-height: 0;
}

.compact-card__title {
  display: flex;
  justify-content: space-between;
  gap: 8px;
}

.compact-card__title span {
  color: #2563eb;
  font-weight: 700;
}

.import-list {
  flex: 1 1 auto;
  min-height: 0;
  max-height: none;
  overflow: auto;
}

.import-row {
  display: flex;
  justify-content: space-between;
  gap: 8px;
  border: 1px solid #e2e8f0;
  border-radius: 7px;
  background: #f8fafc;
  padding: 9px;
  margin-top: 8px;
}

.import-row.active {
  border-color: #60a5fa;
  background: #eff6ff;
}

.import-row strong,
.import-row small {
  display: block;
}

.import-row small {
  margin-top: 3px;
  color: #64748b;
  font-size: 12px;
}

.row-actions {
  display: flex;
  align-items: center;
  gap: 6px;
  flex: 0 0 auto;
}

.row-actions button,
.danger-link {
  height: 28px;
  border: 1px solid #d7e3f4;
  border-radius: 6px;
  background: #fff;
  color: #2563eb;
  font-size: 12px;
}

.row-actions .danger,
.danger-link {
  border-color: #fecaca;
  color: #b91c1c;
}

.imported-station-panel {
  border: 1px solid #e2e8f0;
  border-radius: 7px;
  background: #f8fafc;
  margin-bottom: 10px;
  overflow: hidden;
}

.imported-station-panel__title {
  height: 38px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 0 12px;
  border-bottom: 1px solid #e2e8f0;
  color: #0f172a;
}

.imported-station-panel__title span {
  color: #64748b;
  font-size: 12px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.imported-station-table {
  max-height: 188px;
  overflow: auto;
}

.imported-station-table__head,
.imported-station-table__row {
  display: grid;
  grid-template-columns: minmax(180px, 1.35fr) 110px 92px 74px 64px;
  align-items: center;
  gap: 10px;
  padding: 0 12px;
}

.imported-station-table__head {
  height: 32px;
  color: #64748b;
  font-size: 12px;
  background: #fff;
}

.imported-station-table__row {
  min-height: 44px;
  border-top: 1px solid #edf2f7;
  color: #334155;
  font-size: 12px;
}

.imported-station-table__row strong,
.imported-station-table__row small {
  display: block;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.imported-station-table__row strong {
  color: #0f172a;
}

.imported-station-table__row small {
  margin-top: 2px;
  color: #64748b;
}

.station-status-pill {
  height: 24px;
  min-width: 52px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 999px;
  background: #e2e8f0;
  color: #475569;
  font-size: 12px;
}

.station-status-pill--active {
  background: rgba(34, 197, 94, 0.14);
  color: #15803d;
}

.station-status-pill--degraded {
  background: rgba(245, 158, 11, 0.16);
  color: #b45309;
}

.station-status-pill--offline {
  background: rgba(100, 116, 139, 0.16);
  color: #475569;
}

.ghost-link {
  height: 28px;
  border: 1px solid #bfdbfe;
  border-radius: 6px;
  background: #fff;
  color: #2563eb;
  font-size: 12px;
}

.device-table {
  max-height: 430px;
  overflow: auto;
}

.device-table__head,
.device-table__row {
  display: grid;
  grid-template-columns: 54px minmax(180px, 1.2fr) 110px minmax(220px, 1.2fr) 74px 86px 86px 86px 70px;
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
}

.device-name-cell {
  min-width: 0;
  overflow: hidden;
  color: #17315d;
  font-size: 13px;
  font-weight: 600;
  text-overflow: ellipsis;
  white-space: nowrap;
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

.hidden-file {
  display: none;
}

.run-state {
  color: #1890ff;
  font-weight: 700;
}

.result-layout {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 690px;
  gap: 12px;
}

.terminal-box {
  height: 390px;
  overflow: auto;
  border-radius: 3px;
  background: rgba(51, 51, 51, 0.9);
  padding: 12px;
  color: #dbeafe;
  font-family: Consolas, Monaco, monospace;
  font-size: 12px;
  line-height: 1.7;
  text-align: left;
}

.terminal-box p {
  margin: 0 0 2px;
  white-space: pre-wrap;
  word-break: break-word;
}

.result-metrics {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 8px;
  margin-bottom: 10px;
}

.export-card {
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  background: #f8fafc;
  padding: 10px;
  margin-bottom: 10px;
}

.export-card p {
  margin: 5px 0;
  color: #334155;
  font-size: 13px;
  word-break: break-all;
}

.replay-button {
  display: inline-flex;
  align-items: center;
}

.result-table {
  max-height: 188px;
  overflow: auto;
}

.result-table__head,
.result-table__row {
  display: grid;
  grid-template-columns: 82px minmax(0, 1fr) 120px 90px 90px;
  gap: 8px;
  align-items: center;
}

.result-table__head {
  height: 34px;
  background: rgba(247, 248, 250, 0.5);
  border-bottom: 1px solid #e4e4e4;
  color: #666;
  font-weight: 700;
}

.result-table__row {
  min-height: 38px;
  border-bottom: 1px solid #f1f5f9;
  color: #666;
}

.empty-note {
  margin: 10px 0 0;
  color: #64748b;
  font-size: 13px;
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

.prototype-dialog header {
  height: 61px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  border-bottom: 1px solid #e4e4e4;
  padding: 0 12px;
}

.prototype-dialog h2 {
  margin: 0;
  font-family: "Source Han Sans CN", "Microsoft YaHei", sans-serif;
  font-size: 18px;
  font-weight: 500;
  line-height: 28px;
}

.prototype-dialog header button {
  width: 50px;
  height: 42px;
  border: 0;
  background: transparent;
  color: #999;
  font-family: "FontAwesome", "Arial", sans-serif;
  font-size: 18px;
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

.prototype-dialog tbody tr {
  cursor: pointer;
}

.prototype-dialog tbody tr:hover,
.prototype-dialog tbody tr.active {
  background: rgba(231, 238, 255, 0.5);
}

.empty-row {
  height: 96px !important;
  color: #999 !important;
  text-align: center !important;
}

.status-fade-enter-active,
.status-fade-leave-active {
  transition: opacity 0.2s ease;
}

.status-fade-enter-from,
.status-fade-leave-to {
  opacity: 0;
}

.import-progress-fade-enter-active,
.import-progress-fade-leave-active {
  transition: opacity 0.18s ease, transform 0.18s ease;
}

.import-progress-fade-enter-from,
.import-progress-fade-leave-to {
  opacity: 0;
  transform: translateY(4px);
}
</style>
