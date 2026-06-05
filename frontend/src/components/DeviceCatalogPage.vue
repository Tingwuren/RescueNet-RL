<template>
  <div class="device-state-page">
    <img class="page-bg" :src="assetUrl('images/模型训练/u537.png')" alt="" />
    <img class="page-panel-shadow" :src="assetUrl('images/模型训练/u538.png')" alt="" />

    <main class="device-shell" aria-label="场景设备管理">
      <div class="device-shell__scroll" :style="{ height: `${pageHeight}px` }">
      <div class="page-title">
        <img class="page-title__ribbon" :src="assetUrl('images/模型训练/u541.png')" alt="" />
        <h1>设备管理</h1>
      </div>

      <transition name="fade">
        <div v-if="statusMessage" :class="['status-toast', `status-toast--${statusTone}`]">
          {{ statusMessage }}
        </div>
      </transition>
      <div v-if="errorMessage" class="module-error">{{ errorMessage }}</div>

      <section class="scenario-select-panel" aria-label="灾害场景选择">
        <header class="module-heading">
          <div>
            <i></i>
            <h2>灾害场景选择</h2>
            <p>{{ displayScenarioWithSeverity(currentScenario) }}</p>
          </div>
          <div class="module-actions">
            <button type="button" class="ghost-button" :disabled="loading || saving || !selectedScenarioName" @click="restoreOriginalScenarioBaseStations">
              恢复原始场景基站
            </button>
            <button type="button" class="ghost-button" :disabled="loading" @click="refreshAll">
              {{ loading ? "刷新中" : "刷新设备现状" }}
            </button>
          </div>
        </header>

        <div class="dataset-controls">
          <div class="dataset-choice dataset-choice--scenario">
            <span class="dataset-choice__label">灾害场景</span>
            <div class="dataset-option-grid dataset-option-grid--scenario" aria-label="灾害场景">
              <button
                v-for="option in disasterScenarioOptions"
                :key="option.key"
                type="button"
                :class="['dataset-option-card', { 'dataset-option-card--active': selectedDisasterKey === option.key }]"
                :disabled="loading"
                @click="selectDisasterScenario(option.key)"
              >
                <span class="dataset-option-card__name">{{ option.label }}</span>
                <span class="dataset-option-card__desc">{{ option.description }}</span>
              </button>
            </div>
          </div>

          <div class="dataset-choice dataset-choice--severity">
            <span class="dataset-choice__label">受灾等级</span>
            <div class="dataset-option-grid dataset-option-grid--severity" aria-label="受灾等级">
              <button
                v-for="option in severityOptions"
                :key="option.key"
                type="button"
                :class="['dataset-option-card', { 'dataset-option-card--active': selectedSeverityKey === option.key }]"
                :disabled="loading"
                @click="selectSeverity(option.key)"
              >
                <span class="dataset-option-card__name">{{ option.label }}</span>
                <span class="dataset-option-card__desc">{{ option.description }}</span>
              </button>
            </div>
          </div>
        </div>
      </section>

      <section class="device-map-panel" aria-label="场景设备地图">
        <header class="module-heading">
          <div>
            <i></i>
            <h2>场景设备地图</h2>
            <p>{{ mapSummaryText }}</p>
          </div>
          <div class="module-actions">
            <span class="sync-pill">{{ sceneLoading ? "同步中" : `${stationSourceNodes.length} 个基站` }}</span>
          </div>
        </header>

        <div class="device-map-shell" :style="deviceMapShellStyle">
          <img class="device-map-image" :src="assetUrl('images/首页/u127.jpg')" alt="" />
          <div v-if="mapTiles.length" class="device-tile-map" aria-hidden="true">
            <img
              v-for="tile in mapTiles"
              :key="tile.key"
              :src="tile.url"
              :style="{ left: `${tile.left}px`, top: `${tile.top}px` }"
              alt=""
              draggable="false"
            />
            <span class="device-tile-map__label">{{ mapLabel }}</span>
          </div>
          <div class="device-map-layer" aria-label="用户与基站节点">
            <span
              v-for="marker in userMapMarkers"
              :key="marker.id"
              class="device-map-marker device-map-marker--user"
              :style="markerStyle(marker)"
              @mouseenter="hoveredMapMarker = marker"
              @mouseleave="hoveredMapMarker = null"
            ></span>
            <button
              v-for="marker in stationMapMarkers"
              :key="marker.id"
              type="button"
              class="device-map-marker device-map-marker--station"
              :style="markerStyle(marker)"
              :aria-label="marker.title"
              @mouseenter="hoveredMapMarker = marker"
              @mouseleave="hoveredMapMarker = null"
              @focus="hoveredMapMarker = marker"
              @blur="hoveredMapMarker = null"
              @click="openMapDeviceDetail(marker)"
            ></button>
          </div>
          <div v-if="sceneLoading || sceneError || !mapMarkers.length" class="device-map-empty" role="status" aria-live="polite">
            <strong>{{ sceneLoading ? "正在加载场景地图" : sceneError ? "地图加载失败" : "暂无场景节点" }}</strong>
            <span>{{ sceneError || "当前场景没有可展示的用户或基站节点。" }}</span>
          </div>
          <div v-if="hoveredMapMarker" class="device-map-tooltip" :style="tooltipStyle(hoveredMapMarker)">
            <strong>{{ hoveredMapMarker.title }}</strong>
            <span>{{ hoveredMapMarker.subtitle }}</span>
            <dl>
              <template v-for="item in hoveredMapMarker.details" :key="item.label">
                <dt>{{ item.label }}</dt>
                <dd>{{ item.value }}</dd>
              </template>
            </dl>
          </div>
          <div v-if="mapLegendItems.length" class="device-map-legend" aria-label="地图图例">
            <span v-for="item in mapLegendItems" :key="item.label">
              <i :style="{ background: item.color }"></i>{{ item.label }}
            </span>
          </div>
        </div>
      </section>

      <section class="summary-grid" aria-label="场景设备统计">
        <article v-for="item in stats" :key="item.label">
          <small>{{ item.label }}</small>
          <strong>{{ item.value }}</strong>
          <span>{{ item.hint }}</span>
        </article>
      </section>

      <section class="device-param-panel" aria-label="设备参数配置">
        <header class="module-heading">
          <div>
            <i></i>
            <h2>设备参数配置</h2>
            <p>选择当前场景任意设备型号，配置该型号基站的默认能力参数</p>
          </div>
          <div class="module-actions">
            <span class="sync-pill">{{ selectedModel?.has_override ? "已覆盖默认参数" : "使用默认参数" }}</span>
          </div>
        </header>

        <div class="param-config-layout">
          <aside class="param-type-column" aria-label="设备类型列表">
            <button
              v-for="model in deviceModels"
              :key="model.model_key"
              type="button"
              :class="['param-type-row', { active: selectedModelKey === model.model_key }]"
              @click="selectedModelKey = model.model_key"
            >
              <span>
                <strong>{{ displayDeviceText(model.label) }}</strong>
                <small>{{ displayDeviceText(model.base_station_label) }} · {{ model.station_type || model.model_key }}</small>
              </span>
              <em>{{ formatNumber(model.counts?.total || 0, 0) }} 台</em>
            </button>
            <p v-if="!deviceModels.length" class="empty-note">当前场景暂无可配置设备型号。</p>
          </aside>

          <section class="param-editor">
            <div class="param-summary-strip">
              <span v-for="item in selectedModelSummary" :key="item.label">
                <small>{{ item.label }}</small>
                <strong>{{ item.value }}</strong>
              </span>
            </div>

            <form class="config-form param-config-form" @submit.prevent="saveModelConfig">
              <label>
                设备型号
                <input :value="displayDeviceText(selectedModel?.label) || selectedModelKey" readonly />
              </label>
              <label>
                设备类别
                <input v-model.trim="modelForm.device_category" />
              </label>
              <label>
                覆盖半径 km
                <input v-model.number="modelForm.coverage_radius_km" type="number" min="0" step="0.01" @blur="normalizeModelFormDecimals" />
              </label>
              <label>
                覆盖半径 grid
                <input v-model.number="modelForm.coverage_radius" type="number" min="0" step="0.01" @blur="normalizeModelFormDecimals" />
              </label>
              <label>
                下行能力 Mbps
                <input v-model.number="modelForm.downlink_bandwidth_mbps" type="number" min="0" step="any" @blur="normalizeModelFormDecimals" />
              </label>
              <label>
                上行能力 Mbps
                <input v-model.number="modelForm.uplink_bandwidth_mbps" type="number" min="0" step="any" @blur="normalizeModelFormDecimals" />
              </label>
              <label>
                最大用户数
                <input v-model.number="modelForm.max_users" type="number" min="0" step="1" @blur="normalizeModelFormDecimals" />
              </label>
              <label>
                发射功率 W
                <input v-model.number="modelForm.tx_power_watt" type="number" min="0" step="0.1" @blur="normalizeModelFormDecimals" />
              </label>
              <label>
                续航 h
                <input v-model.number="modelForm.battery_duration_h" type="number" min="0" step="0.1" @blur="normalizeModelFormDecimals" />
              </label>
              <label class="wide">
                备注
                <input v-model.trim="modelForm.notes" />
              </label>
              <div class="form-actions wide">
                <button type="button" class="ghost-button" :disabled="saving || !selectedModel?.has_override" @click="resetModelConfig">
                  恢复默认
                </button>
                <button type="submit" class="primary-button" :disabled="saving || !selectedModelKey">保存型号参数</button>
              </div>
            </form>
          </section>
        </div>
      </section>

      <section class="workbench">
        <section class="config-card config-card--block" aria-label="场景块基站数量配置">
          <aside class="config-card__list-pane">
            <div class="panel-heading">
              <div>
                <h2>场景块基站数量</h2>
                <p>按网格块、设备类型、状态聚合真实设备</p>
              </div>
              <button type="button" class="primary-button" @click="prepareNewBlock">新增块配置</button>
            </div>

            <div class="filter-row">
              <input v-model.trim="blockSearchText" type="search" placeholder="搜索块坐标 / 设备类型 / 状态" />
              <select v-model="blockStatusFilter">
                <option value="">全部状态</option>
                <option value="active">在线</option>
                <option value="degraded">降级</option>
                <option value="offline">离线</option>
                <option value="planned">计划</option>
              </select>
            </div>

            <div class="block-list">
              <button
                v-for="block in filteredBlocks"
                :key="blockKey(block)"
                type="button"
                :class="['block-row', { active: blockKey(block) === selectedBlockKey }]"
                @click="selectBlock(block)"
              >
                <span class="grid-cell">({{ block.x }}, {{ block.y }})</span>
                <span>
                  <strong>{{ displayDeviceText(block.label) || stationLabel(block.base_station) }}</strong>
                  <small>{{ modeLabel(block.mode) }} · {{ statusLabel(block.status) }}</small>
                </span>
                <b>{{ block.quantity }}</b>
              </button>
              <p v-if="!filteredBlocks.length" class="empty-note">当前场景没有匹配的块配置。</p>
            </div>
          </aside>

          <section class="config-card__editor">
            <div class="panel-heading">
              <div>
                <h2>块配置</h2>
                <p>{{ selectedBlockKey || "新增或选择一个场景块配置" }}</p>
              </div>
              <span class="sync-pill">{{ blockForm.quantity }} 台</span>
            </div>

            <form class="config-form config-form--block" @submit.prevent="saveBlockQuantity">
              <label>
                行
                <input v-model.number="blockForm.x" type="number" min="0" :max="gridRows - 1" step="1" />
              </label>
              <label>
                列
                <input v-model.number="blockForm.y" type="number" min="0" :max="gridCols - 1" step="1" />
              </label>
              <label>
                基站类型
                <select v-model="blockForm.base_station">
                  <option v-for="type in deviceTypes" :key="type.base_station" :value="type.base_station">
                    {{ displayDeviceText(type.label) }}
                  </option>
                </select>
              </label>
              <label>
                通信模式
                <select v-model="blockForm.mode">
                  <option v-for="mode in modesForType(blockForm.base_station)" :key="mode" :value="mode">
                    {{ modeLabel(mode) }}
                  </option>
                </select>
              </label>
              <label>
                状态
                <select v-model="blockForm.status">
                  <option value="active">在线</option>
                  <option value="degraded">降级</option>
                  <option value="offline">离线</option>
                  <option value="planned">计划</option>
                </select>
              </label>
              <label>
                数量
                <input v-model.number="blockForm.quantity" type="number" min="0" step="1" />
              </label>
              <div class="form-actions wide">
                <button type="button" class="ghost-button" @click="prepareNewBlock">清空新增</button>
                <button type="button" class="danger-button" :disabled="saving || !selectedBlockKey" @click="deleteSelectedBlock">删除块配置</button>
                <button type="submit" class="primary-button" :disabled="saving || !blockForm.base_station">保存块数量</button>
              </div>
            </form>
          </section>
        </section>

        <section class="config-card config-card--device" aria-label="单设备实例配置">
          <aside class="config-card__list-pane">
            <div class="panel-heading">
              <div>
                <h2>单设备实例</h2>
                <p>{{ filteredDevices.length }} 台设备，可单独覆盖类型参数</p>
              </div>
              <button type="button" class="primary-button" @click="prepareNewDevice">新增设备</button>
            </div>

            <div class="filter-row">
              <input v-model.trim="deviceSearchText" type="search" placeholder="搜索设备编号 / 坐标 / 类型 / 状态" />
              <select v-model="deviceStatusFilter">
                <option value="">全部状态</option>
                <option value="active">在线</option>
                <option value="degraded">降级</option>
                <option value="offline">离线</option>
                <option value="planned">计划</option>
              </select>
            </div>

            <div class="device-list">
              <button
                v-for="device in filteredDevices"
                :key="deviceId(device)"
                type="button"
                :class="['device-row', { active: deviceId(device) === selectedDeviceId }]"
                @click="selectDevice(device)"
              >
                <span :class="['status-dot', `status-dot--${device.status || 'unknown'}`]"></span>
                <span class="device-row__content">
                  <strong>{{ displayDeviceText(device.label || device.device_name) || stationLabel(device.base_station) }}</strong>
                  <small>{{ deviceId(device) }} · ({{ device.x }}, {{ device.y }})</small>
                </span>
                <em>{{ formatNumber(device.max_users, 0) }} 用户</em>
              </button>
              <p v-if="!filteredDevices.length" class="empty-note">当前场景没有匹配的设备实例。</p>
            </div>
          </aside>

          <section class="config-card__editor">
            <div class="panel-heading">
              <div>
                <h2>单设备配置</h2>
                <p>{{ selectedDeviceId || "新增或选择一个设备实例" }}</p>
              </div>
              <button v-if="selectedDeviceId" type="button" class="danger-button" @click="deleteSelectedDevice">删除设备</button>
            </div>

            <form class="config-form" @submit.prevent="saveDeviceConfig">
              <label>
                设备名称
                <input v-model.trim="deviceForm.device_name" placeholder="未命名则使用类型名称" />
              </label>
              <label>
                基站类型
                <select v-model="deviceForm.base_station">
                  <option v-for="type in deviceTypes" :key="type.base_station" :value="type.base_station">
                    {{ displayDeviceText(type.label) }}
                  </option>
                </select>
              </label>
              <label>
                通信模式
                <select v-model="deviceForm.mode">
                  <option v-for="mode in modesForType(deviceForm.base_station)" :key="mode" :value="mode">
                    {{ modeLabel(mode) }}
                  </option>
                </select>
              </label>
              <label>
                状态
                <select v-model="deviceForm.status">
                  <option value="active">在线</option>
                  <option value="degraded">降级</option>
                  <option value="offline">离线</option>
                  <option value="planned">计划</option>
                </select>
              </label>
              <label>
                行
                <input v-model.number="deviceForm.x" type="number" min="0" :max="gridRows - 1" step="1" />
              </label>
              <label>
                列
                <input v-model.number="deviceForm.y" type="number" min="0" :max="gridCols - 1" step="1" />
              </label>
              <label>
                覆盖半径 km
                <input v-model.number="deviceForm.coverage_radius_km" type="number" min="0" step="0.01" @blur="normalizeDeviceFormDecimals" />
              </label>
              <label>
                覆盖半径 grid
                <input v-model.number="deviceForm.coverage_radius" type="number" min="0" step="0.01" @blur="normalizeDeviceFormDecimals" />
              </label>
              <label>
                下行能力 Mbps
                <input v-model.number="deviceForm.downlink_bandwidth_mbps" type="number" min="0" step="any" @blur="normalizeDeviceFormDecimals" />
              </label>
              <label>
                上行能力 Mbps
                <input v-model.number="deviceForm.uplink_bandwidth_mbps" type="number" min="0" step="any" @blur="normalizeDeviceFormDecimals" />
              </label>
              <label>
                最大用户数
                <input v-model.number="deviceForm.max_users" type="number" min="0" step="1" @blur="normalizeDeviceFormDecimals" />
              </label>
              <label>
                续航 h
                <input v-model.number="deviceForm.battery_duration_h" type="number" min="0" step="0.1" @blur="normalizeDeviceFormDecimals" />
              </label>
              <label class="wide">
                备注
                <input v-model.trim="deviceForm.notes" />
              </label>
              <div class="form-actions wide">
                <button type="button" class="ghost-button" @click="prepareNewDevice">清空新增</button>
                <button type="submit" class="primary-button" :disabled="saving || !deviceForm.base_station">
                  {{ selectedDeviceId ? "保存单设备配置" : "新增设备实例" }}
                </button>
              </div>
            </form>
          </section>
        </section>
      </section>

      <section class="tracking-terminal-panel">
        <StreamingTerminal
          title="实时终端输出"
          subtitle="实时输出设备管理操作、后端保存响应和可供训练/测试读取的设备状态变更。"
          :lines="deviceTerminalLines"
          :status="terminalStatus"
          placeholder="暂无设备状态输出，保存一次配置后开始追踪。"
          exportable
          clearable
          @export="downloadTerminalLog"
          @clear="clearTerminalLog"
        />
      </section>

      <div v-if="mapDetailOpen" class="device-detail-modal" @click.self="closeMapDeviceDetail">
        <section class="device-detail-dialog" aria-label="地图设备详情">
          <header class="panel-heading">
            <div>
              <h2>设备详情</h2>
              <p>{{ selectedDeviceId || "未选择设备" }}</p>
            </div>
            <button type="button" class="ghost-button" @click="closeMapDeviceDetail">关闭</button>
          </header>

          <form class="config-form" @submit.prevent="saveMapDeviceConfig">
            <label>
              设备名称
              <input v-model.trim="deviceForm.device_name" placeholder="未命名则使用类型名称" />
            </label>
            <label>
              基站类型
              <select v-model="deviceForm.base_station">
                <option v-for="type in deviceTypes" :key="type.base_station" :value="type.base_station">
                  {{ displayDeviceText(type.label) }}
                </option>
              </select>
            </label>
            <label>
              通信模式
              <select v-model="deviceForm.mode">
                <option v-for="mode in modesForType(deviceForm.base_station)" :key="mode" :value="mode">
                  {{ modeLabel(mode) }}
                </option>
              </select>
            </label>
            <label>
              状态
              <select v-model="deviceForm.status">
                <option value="active">在线</option>
                <option value="degraded">降级</option>
                <option value="offline">离线</option>
                <option value="planned">计划</option>
              </select>
            </label>
            <label>
              行
              <input v-model.number="deviceForm.x" type="number" min="0" :max="gridRows - 1" step="1" />
            </label>
            <label>
              列
              <input v-model.number="deviceForm.y" type="number" min="0" :max="gridCols - 1" step="1" />
            </label>
            <label>
              覆盖半径 km
              <input v-model.number="deviceForm.coverage_radius_km" type="number" min="0" step="0.01" @blur="normalizeDeviceFormDecimals" />
            </label>
            <label>
              下行能力 Mbps
              <input v-model.number="deviceForm.downlink_bandwidth_mbps" type="number" min="0" step="any" @blur="normalizeDeviceFormDecimals" />
            </label>
            <label>
              最大用户数
              <input v-model.number="deviceForm.max_users" type="number" min="0" step="1" @blur="normalizeDeviceFormDecimals" />
            </label>
            <label class="wide">
              备注
              <input v-model.trim="deviceForm.notes" />
            </label>
            <div class="form-actions wide">
              <button type="button" class="danger-button" :disabled="saving || !selectedDeviceId" @click="deleteMapDevice">
                删除设备
              </button>
              <button type="submit" class="primary-button" :disabled="saving || !deviceForm.base_station">
                保存设备
              </button>
            </div>
          </form>
        </section>
      </div>
      </div>
    </main>
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, reactive, ref, watch } from "vue";
import axios from "axios";

import StreamingTerminal from "./StreamingTerminal.vue";
import { rescueApiBase } from "../utils/runtimeEndpoints";
import {
  appendSyncedTerminalLine,
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

const scenarios = ref([]);
const deviceState = ref(null);
const selectedScenarioName = ref("");
const selectedDisasterKey = ref("");
const selectedSeverityKey = ref("");
const selectedTypeKey = ref("");
const selectedModelKey = ref("");
const selectedBlockKey = ref("");
const selectedDeviceId = ref("");
const blockSearchText = ref("");
const blockStatusFilter = ref("");
const deviceSearchText = ref("");
const deviceStatusFilter = ref("");
const loading = ref(false);
const saving = ref(false);
const errorMessage = ref("");
const statusMessage = ref("");
const statusTone = ref("info");
const scenePreview = ref(null);
const sceneLoading = ref(false);
const sceneError = ref("");
const hoveredMapMarker = ref(null);
const mapDetailOpen = ref(false);
const terminalLines = ref([]);
const terminalStatus = ref("idle");
let statusTimer = null;
let lastDeviceUserNodeLogKey = "";

const USER_MARKER_COLORS = {
  online: "#38bdf8",
  offline: "#ef4444",
};
const USER_MARKER_SIZE_MIN = 3.4;
const USER_MARKER_SIZE_MAX = 4.7;

const STATION_MARKER_COLORS = {
  emergency_5g_700mhz_cell: "#2563eb",
  ka_satellite_terminal: "#7c3aed",
  wifi6_mesh_node: "#16a34a",
  shortwave_hf_station: "#f59e0b",
  default: "#0ea5e9",
};
const DEVICE_DISPLAY_TEXT_REPLACEMENTS = [
  ["5G 700MHz 应急小区", "5G 700MHz应急基站"],
  ["5G 700MHz应急小区", "5G 700MHz应急基站"],
  ["5G应急小区", "5G应急基站"],
];
const USER_SCATTER_SPREAD = 1.18;
const STATION_SCATTER_SPREAD = 0.62;
const DEVICE_MAP_CLIP_PADDING = 18;
const DEVICE_MAP_LAYER = {
  width: 1618,
  height: 745,
};
const BASEMAP_SHIFT_RATIOS = {
  rainstorm: 0.16,
  typhoon: 0.42,
};
const DEVICE_MARKER_AREA = {
  left: 90,
  top: 60,
  width: 1438,
  height: 610,
};
const DEVICE_MAP_FALLBACK_ZOOM = 1;
const DEVICE_MAP_SCATTER_ZOOM = 1;
const MIN_GEO_SPAN = 0.00001;
const MAX_PARAMETER_DECIMALS = 2;

const emptyTypeForm = () => ({
  device_category: "",
  coverage_radius_km: 0,
  coverage_radius: 0,
  downlink_bandwidth_mbps: 0,
  uplink_bandwidth_mbps: 0,
  max_users: 0,
  tx_power_watt: null,
  battery_duration_h: null,
  notes: "",
});

const emptyBlockForm = () => ({
  x: 0,
  y: 0,
  base_station: "",
  mode: "",
  status: "active",
  quantity: 1,
});

const emptyDeviceForm = () => ({
  device_name: "",
  base_station: "",
  mode: "",
  status: "active",
  x: 0,
  y: 0,
  coverage_radius_km: 0,
  coverage_radius: 0,
  downlink_bandwidth_mbps: 0,
  uplink_bandwidth_mbps: 0,
  max_users: 0,
  tx_power_watt: null,
  battery_duration_h: null,
  notes: "",
});

const typeForm = reactive(emptyTypeForm());
const modelForm = reactive(emptyTypeForm());
const blockForm = reactive(emptyBlockForm());
const deviceForm = reactive(emptyDeviceForm());

const assetUrl = (path) => `${import.meta.env.BASE_URL}prototype/${path}`;
const displayScenarioName = (scenario) => scenario?.display_name || formatScenarioName(scenario?.name);
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
const displayScenarioWithSeverity = (scenario) => (scenario ? `${scenarioDisasterLabel(scenario)} / ${severityLabel(scenario)}` : "--");
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
      disaster_type: item.disaster_type || item.type || source,
      grid_size: Math.max(rows, cols),
      region_grid: { ...(typeof grid === "object" ? grid : {}), rows, cols },
      num_users: numUsers,
      candidate_sites: Number(item.candidate_sites || rows * cols),
      base_stations: [],
      base_station_deployments: [],
      residual_base_stations: [],
    }));
  });
const isUserNode = (node) => String(node?.type || "").toUpperCase() === "USER";
const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
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
  return Number(size.toFixed(2));
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
  const left = quantileNumber(markers.map((marker) => Number(marker.left)), 0.02);
  const right = quantileNumber(markers.map((marker) => Number(marker.left)), 0.98);
  const top = quantileNumber(markers.map((marker) => Number(marker.top)), 0.02);
  const bottom = quantileNumber(markers.map((marker) => Number(marker.top)), 0.98);
  if (![left, right, top, bottom].every(Number.isFinite)) return null;
  const padX = Math.max(4, (right - left) * 0.1);
  const padY = Math.max(4, (bottom - top) * 0.1);
  return {
    left: clamp(left - padX, 0, 100),
    right: clamp(right + padX, 0, 100),
    top: clamp(top - padY, 0, 100),
    bottom: clamp(bottom + padY, 0, 100),
  };
};

const markerInsideEnvelope = (marker, envelope) => {
  if (!envelope) return true;
  const left = Number(marker?.left);
  const top = Number(marker?.top);
  return Number.isFinite(left) && Number.isFinite(top) && left >= envelope.left && left <= envelope.right && top >= envelope.top && top <= envelope.bottom;
};

const currentScenario = computed(() => scenarios.value.find((item) => item.name === selectedScenarioName.value) || null);
const deviceTypes = computed(() => deviceState.value?.device_types || []);
const deviceModels = computed(() => deviceState.value?.device_models || []);
const devices = computed(() => deviceState.value?.devices || []);
const blocks = computed(() => deviceState.value?.blocks || []);
const typeOverrides = computed(() => deviceState.value?.type_overrides || {});
const gridRows = computed(() => Number(deviceState.value?.grid?.rows || currentScenario.value?.region_grid?.rows || currentScenario.value?.grid_size || 1));
const gridCols = computed(() => Number(deviceState.value?.grid?.cols || currentScenario.value?.region_grid?.cols || currentScenario.value?.grid_size || 1));
const selectedType = computed(() => deviceTypes.value.find((item) => item.base_station === selectedTypeKey.value) || deviceTypes.value[0] || null);
const selectedModel = computed(() => deviceModels.value.find((item) => item.model_key === selectedModelKey.value) || deviceModels.value[0] || null);
const selectedDevice = computed(() => devices.value.find((item) => deviceId(item) === selectedDeviceId.value) || null);
const activeScene = computed(() => scenePreview.value?.scene || {});
const sceneNodes = computed(() => (Array.isArray(activeScene.value?.nodes) ? activeScene.value.nodes : []));
const sceneUsers = computed(() => sceneNodes.value.filter(isUserNode));
const sceneStations = computed(() => sceneNodes.value.filter((node) => !isUserNode(node)));
const sceneMapWidth = computed(() => Math.max(1, Number(activeScene.value?.map_width || 5000)));
const sceneMapHeight = computed(() => Math.max(1, Number(activeScene.value?.map_height || 5000)));
const stationSourceNodes = computed(() =>
  sceneStations.value.length ? sceneStations.value : devices.value.map((device, index) => deviceToSceneStationNode(device, index)).filter(Boolean)
);
const deviceLookup = computed(() => {
  const lookup = new Map();
  devices.value.forEach((device) => {
    [deviceId(device), device?.id, device?.device_uid, device?.deployment_id]
      .filter(Boolean)
      .forEach((key) => lookup.set(String(key), device));
  });
  return lookup;
});
const activeSceneGridShape = () => ({
  rows: Math.max(1, Number(gridRows.value || currentScenario.value?.grid_size || 22)),
  cols: Math.max(1, Number(gridCols.value || currentScenario.value?.grid_size || 22)),
});

const sceneNodeGridPoint = (node) => {
  const { rows, cols } = activeSceneGridShape();
  return {
    gridRow: clamp((Number(node?.y || 0) / sceneMapHeight.value) * rows, 0, rows),
    gridCol: clamp((Number(node?.x || 0) / sceneMapWidth.value) * cols, 0, cols),
    rows,
    cols,
  };
};

const deviceToSceneStationNode = (device, index = 0) => {
  if (!device?.base_station) return null;
  const { rows, cols } = activeSceneGridShape();
  const row = clamp(Number(device.x || 0), 0, Math.max(0, rows - 1));
  const col = clamp(Number(device.y || 0), 0, Math.max(0, cols - 1));
  return {
    ...device,
    id: deviceId(device) || `device-${index}`,
    type: device.base_station,
    visual_type: device.base_station,
    node_role: "residual_base_station",
    x: ((col + 0.5) / Math.max(1, cols)) * sceneMapWidth.value,
    y: ((row + 0.5) / Math.max(1, rows)) * sceneMapHeight.value,
    device_uid: device.device_uid || device.id,
    deployment_id: device.deployment_id,
    label: displayDeviceText(device.device_name || device.label) || stationLabel(device.base_station),
    mode: device.mode,
    status: device.status || "active",
  };
};

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

const severityOptions = computed(() =>
  scenarios.value
    .filter((scenario) => scenarioSourceKey(scenario) === selectedDisasterKey.value)
    .map((scenario) => ({
      key: scenarioSeverityKey(scenario),
      label: severityLabel(scenario),
      description: `${Number(scenario.region_grid?.rows || scenario.grid_size || 0)} x ${Number(scenario.region_grid?.cols || scenario.grid_size || 0)} 网格`,
      scenarioName: scenario.name,
    }))
);

const filteredBlocks = computed(() => {
  const keyword = blockSearchText.value.toLowerCase();
  return blocks.value.filter((block) => {
    const text = `${block.x} ${block.y} ${block.base_station} ${block.label} ${block.mode} ${block.status}`.toLowerCase();
    return (!blockStatusFilter.value || block.status === blockStatusFilter.value) && (!keyword || text.includes(keyword));
  });
});

const filteredDevices = computed(() => {
  const keyword = deviceSearchText.value.toLowerCase();
  return devices.value.filter((device) => {
    const text = `${deviceId(device)} ${device.label} ${device.device_name} ${device.base_station} ${device.mode} ${device.status} ${device.x} ${device.y}`.toLowerCase();
    return (!deviceStatusFilter.value || device.status === deviceStatusFilter.value) && (!keyword || text.includes(keyword));
  });
});

const historyRows = computed(() => [...(deviceState.value?.history || [])].reverse());
const deviceTerminalLines = computed(() => terminalHistoryLines.value.slice(-320));

const stats = computed(() => {
  const counts = deviceState.value?.status_counts || {};
  const typeCount = deviceTypes.value.filter((item) => Number(item.counts?.total || 0) > 0).length;
  return [
    {
      label: "设备总数",
      value: String(counts.total || devices.value.length || 0),
      hint: displayScenarioWithSeverity(currentScenario.value),
    },
    {
      label: "在线 / 降级 / 离线",
      value: `${counts.active || 0} / ${counts.degraded || 0} / ${counts.offline || 0}`,
      hint: "来自场景真实基站状态",
    },
    {
      label: "场景块",
      value: String(blocks.value.length),
      hint: `${gridRows.value} x ${gridCols.value} 网格`,
    },
    {
      label: "设备类型",
      value: String(typeCount),
      hint: formatDisasterType(currentScenario.value?.disaster_type),
    },
  ];
});

const selectedModelSummary = computed(() => {
  const counts = selectedModel.value?.counts || {};
  return [
    { label: "型号设备", value: `${Number(counts.total || 0)} 台` },
    { label: "在线", value: Number(counts.active || 0) },
    { label: "降级", value: Number(counts.degraded || 0) },
    { label: "离线", value: Number(counts.offline || 0) },
  ];
});

const mapBounds = computed(() =>
  mergeGeoBounds(
    normalizeGeoBounds(activeScene.value?.geo_bounds),
    normalizeGeoBounds(deviceState.value?.grid?.geo_bounds),
    normalizeGeoBounds(currentScenario.value?.region_grid?.geo_bounds),
    expandGeoBounds(nodesGeoBounds(sceneNodes.value), 0.12)
  )
);
const hasUsableMapBounds = computed(() => hasRenderableGeoBounds(mapBounds.value));
const mapLabel = computed(() => scenarioDisasterLabel(currentScenario.value));
const deviceMapShellStyle = computed(() => ({
  "--device-map-bg-scale": "1",
}));
const mapSceneText = computed(() =>
  [
    activeScene.value?.name,
    activeScene.value?.disaster_type,
    deviceState.value?.scenario_name,
    deviceState.value?.disaster_type,
    currentScenario.value?.name,
    currentScenario.value?.disaster_type,
    currentScenario.value?.display_name,
    currentScenario.value?.source_scenario,
    mapLabel.value,
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase()
);
const isRainstormMapScene = computed(() => /rainstorm|暴雨|内涝/.test(mapSceneText.value));
const isTyphoonMapScene = computed(() => /typhoon|台风|风暴潮/.test(mapSceneText.value));
const mapBasemapShiftX = computed(() => {
  if (isRainstormMapScene.value) return Math.round(DEVICE_MAP_LAYER.width * BASEMAP_SHIFT_RATIOS.rainstorm);
  if (isTyphoonMapScene.value) return Math.round(DEVICE_MAP_LAYER.width * BASEMAP_SHIFT_RATIOS.typhoon);
  return 0;
});
const activeMapViewport = computed(() =>
  hasUsableMapBounds.value ? mapViewport(DEVICE_MAP_LAYER.width, DEVICE_MAP_LAYER.height, mapBounds.value, 0) : null
);
const mapTiles = computed(() => {
  const viewport = activeMapViewport.value;
  if (!viewport) return [];
  const tileSize = 256;
  const maxTile = 2 ** viewport.zoom;
  const shiftedViewportLeft = viewport.left - mapBasemapShiftX.value;
  const minTileX = Math.floor(shiftedViewportLeft / tileSize) - 1;
  const maxTileX = Math.floor((shiftedViewportLeft + DEVICE_MAP_LAYER.width) / tileSize) + 1;
  const minTileY = Math.floor(viewport.top / tileSize) - 1;
  const maxTileY = Math.floor((viewport.top + DEVICE_MAP_LAYER.height) / tileSize) + 1;
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

const visibleUserMapMarkers = computed(() => {
  if (sceneLoading.value) return [];
  return sceneUsers.value
    .map((node, index) => {
      const seed = `scene-user-${node.id ?? index}`;
      const point = nodeMapPoint(node, seed, USER_SCATTER_SPREAD);
      if (!point) return null;
      const connected = Boolean(node.connected);
      return {
        id: `user-${node.id ?? index}`,
        kind: "user",
        title: connected ? "在线用户" : "断联用户",
        subtitle: `用户 ${node.id ?? index}`,
        left: point.left,
        top: point.top,
        color: connected ? USER_MARKER_COLORS.online : USER_MARKER_COLORS.offline,
        size: userMarkerSize(seed),
        opacity: connected ? 0.82 : 0.7,
        details: [
          { label: "连接状态", value: connected ? "已接入" : "未接入" },
          { label: "地图坐标", value: `${formatNumber(node.x, 0)}, ${formatNumber(node.y, 0)}` },
        ],
      };
    })
    .filter(Boolean);
});
const userMapMarkers = computed(() => visibleUserMapMarkers.value);
const userMarkerEnvelope = computed(() => markerEnvelope(visibleUserMapMarkers.value));

const stationMapMarkers = computed(() => {
  if (sceneLoading.value) return [];
  return stationSourceNodes.value
    .map((node, index) => {
      const point = nodeMapPoint(node, `scene-station-${node.id ?? index}`, STATION_SCATTER_SPREAD);
      if (!point) return null;
      const device = stationDeviceForNode(node);
      const gridCell = nodeGridCell(node);
      const baseStation = node.base_station || device?.base_station || node.type || "";
      const status = node.status || device?.status || "active";
      const id = device ? deviceId(device) : String(node.device_uid || node.deployment_id || node.id || index);
      const label = displayDeviceText(device?.device_name || device?.label || node.label) || stationLabel(baseStation);
      const gridText = device
        ? `(${device.x}, ${device.y})`
        : gridCell
          ? `(${gridCell.row}, ${gridCell.col})`
          : `${formatNumber(node.x, 0)}, ${formatNumber(node.y, 0)}`;
      return {
        id: `station-${id}-${index}`,
        kind: "station",
        deviceUid: id,
        device,
        title: label,
        subtitle: `${stationLabel(baseStation)} · ${modeLabel(node.mode || device?.mode)} · ${statusLabel(status)}`,
        left: point.left,
        top: point.top,
        color: STATION_MARKER_COLORS[baseStation] || STATION_MARKER_COLORS.default,
        size: 14,
        opacity: 1,
        details: [
          { label: "设备编号", value: id },
          { label: "网格", value: gridText },
          { label: "状态", value: statusLabel(status) },
          { label: "接入用户", value: formatNumber(node.connected_users ?? device?.cell_user_count ?? 0, 0) },
          { label: "覆盖半径", value: `${formatNumber(node.coverage_radius_km ?? device?.coverage_radius_km, 2)} km` },
        ],
      };
    })
    .filter(Boolean)
    .filter((marker) => markerInsideEnvelope(marker, userMarkerEnvelope.value));
});

const mapMarkers = computed(() => [...userMapMarkers.value, ...stationMapMarkers.value]);
const mapSummaryText = computed(() => {
  if (sceneLoading.value) return "正在同步场景用户与基站节点";
  if (sceneError.value) return sceneError.value;
  const userPrefix =
    visibleUserMapMarkers.value.length > userMapMarkers.value.length
      ? `用户 ${userMapMarkers.value.length}/${visibleUserMapMarkers.value.length}`
      : `用户 ${visibleUserMapMarkers.value.length}`;
  return `网格相对视图：${userPrefix}（总 ${sceneUsers.value.length}）/ 基站 ${stationMapMarkers.value.length}（总 ${stationSourceNodes.value.length}）`;
});
const mapLegendItems = computed(() => {
  if (!mapMarkers.value.length) return [];
  const stationTypes = new Map();
  stationMapMarkers.value.forEach((marker) => {
    const label = marker.subtitle.split(" · ")[0] || marker.title;
    if (!stationTypes.has(label)) stationTypes.set(label, { label, color: marker.color });
  });
  return [
    { label: "在线用户", color: USER_MARKER_COLORS.online },
    { label: "断联用户", color: USER_MARKER_COLORS.offline },
    ...stationTypes.values(),
  ];
});

const pageHeight = computed(() => 3180 + Math.max(0, Math.ceil(devices.value.length / 16)) * 28);

const appendDeviceTerminalLine = (message, options = {}) => {
  if (!message) return;
  terminalLines.value = appendSyncedTerminalLine(
    terminalLines.value,
    message,
    { level: options.level || "INFO", source: options.source || "DEVICE", timestamp: options.timestamp },
    220
  );
};

const appendDeviceUserNodeCount = (prefix, ...sources) => {
  const key = userNodeCountLogKey(`device:${selectedScenarioName.value || ""}:${prefix}`, ...sources);
  if (key === lastDeviceUserNodeLogKey) return;
  lastDeviceUserNodeLogKey = key;
  appendDeviceTerminalLine(buildUserNodeCountMessage(prefix, ...sources), { level: "SCENE" });
};

const downloadTerminalLog = () => {
  exportTerminalOutput(terminalHistoryLines.value, "rescuenet-device-terminal.log");
};

const clearTerminalLog = () => {
  terminalLines.value = [];
  clearTerminalOutput();
};

const showStatus = (message, tone = "info", timeout = 3600) => {
  statusMessage.value = message;
  statusTone.value = tone;
  if (tone === "error") terminalStatus.value = "failed";
  if (tone === "success") terminalStatus.value = "completed";
  appendDeviceTerminalLine(message, {
    level: tone === "error" ? "ERROR" : tone === "success" ? "OK" : tone === "warning" ? "WARN" : "INFO",
  });
  if (statusTimer) window.clearTimeout(statusTimer);
  if (timeout) {
    statusTimer = window.setTimeout(() => {
      statusMessage.value = "";
    }, timeout);
  }
};

const decimalDigits = (digits = MAX_PARAMETER_DECIMALS) => {
  const parsed = Number(digits);
  return Number.isFinite(parsed) ? Math.max(0, Math.min(MAX_PARAMETER_DECIMALS, Math.trunc(parsed))) : MAX_PARAMETER_DECIMALS;
};

const roundParameterNumber = (value, digits = MAX_PARAMETER_DECIMALS, fallback = 0) => {
  const number = Number(value);
  if (!Number.isFinite(number)) return fallback;
  const precision = decimalDigits(digits);
  return Number(number.toFixed(precision));
};

const roundOptionalParameterNumber = (value, digits = MAX_PARAMETER_DECIMALS) =>
  value === null || value === undefined || value === "" ? null : roundParameterNumber(value, digits, 0);

const formatNumber = (value, digits = MAX_PARAMETER_DECIMALS) => {
  const number = Number(value);
  if (!Number.isFinite(number)) return "--";
  const precision = decimalDigits(digits);
  return roundParameterNumber(number, precision).toLocaleString("zh-CN", {
    minimumFractionDigits: 0,
    maximumFractionDigits: precision,
  });
};

const formatTime = (value) => {
  const number = Number(value);
  if (!Number.isFinite(number) || number <= 0) return "--";
  return new Date(number * 1000).toLocaleString();
};

const statusLabel = (status) =>
  ({ active: "在线", degraded: "降级", offline: "离线", planned: "计划", deployed: "已部署", unknown: "未知" })[status] || status || "未知";

const operationLabel = (operation) =>
  ({
    add_device: "新增设备",
    delete_block_quantity: "删除块配置",
    delete_device: "删除设备",
    update_device: "修改单设备",
    update_block_quantity: "修改块数量",
    update_device_state: "保存设备状态",
    replace_base_stations: "替换基站清单",
    reset_base_stations: "恢复场景默认",
    clear_residual_network: "清空残余网络",
    update_type_config: "修改类型参数",
    reset_type_config: "恢复类型默认",
    update_device_model_config: "修改型号参数",
    reset_device_model_config: "恢复型号默认",
  })[operation] || operation || "--";

const deviceId = (device) => String(device?.id || device?.device_uid || device?.deployment_id || "");
const blockKey = (block) => `${block.x}:${block.y}:${block.base_station}:${block.mode}:${block.status}`;
const displayDeviceText = (value) => {
  if (value === null || value === undefined) return "";
  return DEVICE_DISPLAY_TEXT_REPLACEMENTS.reduce(
    (text, [source, target]) => text.replaceAll(source, target),
    String(value)
  );
};
const stationLabel = (baseStation) => displayDeviceText(deviceTypes.value.find((item) => item.base_station === baseStation)?.label) || baseStation || "--";
const modeLabel = (mode) => mode || "--";
const modesForType = (baseStation) => deviceTypes.value.find((item) => item.base_station === baseStation)?.supported_modes || [];
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

const hasRenderableGeoBounds = (bounds) =>
  Boolean(
    bounds &&
      Number.isFinite(bounds.latMin) &&
      Number.isFinite(bounds.latMax) &&
      Number.isFinite(bounds.lonMin) &&
      Number.isFinite(bounds.lonMax) &&
      Math.abs(bounds.latMax - bounds.latMin) >= MIN_GEO_SPAN &&
      Math.abs(bounds.lonMax - bounds.lonMin) >= MIN_GEO_SPAN
  );

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
  const safeLat = clamp(Number(lat), -85.05112878, 85.05112878);
  const sin = Math.sin((safeLat * Math.PI) / 180);
  return {
    x: ((Number(lon) + 180) / 360) * size,
    y: (0.5 - Math.log((1 + sin) / (1 - sin)) / (4 * Math.PI)) * size,
  };
};

const mapViewport = (width, height, bounds, zoomBoost = 0) => {
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
  const activeZoom = clamp(bestZoom + zoomBoost, 5, 18);
  const center = mercatorProject((bounds.latMin + bounds.latMax) / 2, (bounds.lonMin + bounds.lonMax) / 2, activeZoom);
  return {
    zoom: activeZoom,
    left: center.x - width / 2,
    top: center.y - height / 2,
  };
};

const cartoTileUrl = (zoom, x, y) => {
  const subdomains = ["a", "b", "c", "d"];
  const subdomain = subdomains[Math.abs(x + y) % subdomains.length];
  return `https://${subdomain}.basemaps.cartocdn.com/rastertiles/voyager/${zoom}/${x}/${y}.png`;
};

const isVisibleMapPoint = (x, y) =>
  x >= -DEVICE_MAP_CLIP_PADDING &&
  x <= DEVICE_MAP_LAYER.width + DEVICE_MAP_CLIP_PADDING &&
  y >= -DEVICE_MAP_CLIP_PADDING &&
  y <= DEVICE_MAP_LAYER.height + DEVICE_MAP_CLIP_PADDING;

const mapPointToPercent = (x, y) => {
  if (!isVisibleMapPoint(x, y)) return null;
  return {
    left: (x / DEVICE_MAP_LAYER.width) * 100,
    top: (y / DEVICE_MAP_LAYER.height) * 100,
  };
};

const zoomFallbackMapPoint = (point) => ({
  x: DEVICE_MAP_LAYER.width / 2 + (point.x - DEVICE_MAP_LAYER.width / 2) * DEVICE_MAP_FALLBACK_ZOOM,
  y: DEVICE_MAP_LAYER.height / 2 + (point.y - DEVICE_MAP_LAYER.height / 2) * DEVICE_MAP_FALLBACK_ZOOM,
});

const fallbackNodeMapPoint = (node) => {
  const x = Number(node?.x);
  const y = Number(node?.y);
  if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
  return zoomFallbackMapPoint({
    x: DEVICE_MARKER_AREA.left + (x / sceneMapWidth.value) * DEVICE_MARKER_AREA.width,
    y: DEVICE_MARKER_AREA.top + (y / sceneMapHeight.value) * DEVICE_MARKER_AREA.height,
  });
};

const nodeGridCell = (node) => {
  const { rows, cols } = activeSceneGridShape();
  const x = Number(node?.x);
  const y = Number(node?.y);
  if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
  return {
    row: Math.floor(clamp((y / sceneMapHeight.value) * rows, 0, rows - Number.EPSILON)),
    col: Math.floor(clamp((x / sceneMapWidth.value) * cols, 0, cols - Number.EPSILON)),
  };
};

const nodeMapPoint = (node, seed = "", spread = 0) => {
  let base = null;
  const lat = Number(node?.lat);
  const lon = Number(node?.lon);
  const viewport = activeMapViewport.value;
  if (Number.isFinite(lat) && Number.isFinite(lon) && viewport) {
    const point = mercatorProject(lat, lon, viewport.zoom);
    base = {
      x: point.x - viewport.left,
      y: point.y - viewport.top,
    };
  } else {
    base = fallbackNodeMapPoint(node);
  }
  if (!base) return null;
  let basePoint = mapPointToPercent(base.x, base.y);
  if (!basePoint) {
    const fallback = fallbackNodeMapPoint(node);
    if (!fallback) return null;
    base = fallback;
    basePoint = mapPointToPercent(base.x, base.y);
  }
  if (!basePoint) return null;
  if (!seed || !spread) {
    return basePoint;
  }
  const grid = sceneNodeGridPoint(node);
  const offset = scatteredOffset(seed, spread);
  const spreadZoom = DEVICE_MAP_SCATTER_ZOOM;
  const cellWidth = (DEVICE_MARKER_AREA.width / Math.max(1, grid.cols)) * spreadZoom;
  const cellHeight = (DEVICE_MARKER_AREA.height / Math.max(1, grid.rows)) * spreadZoom;
  const rawX = base.x + offset.col * cellWidth;
  const rawY = base.y + offset.row * cellHeight;
  const boundedX = softBoundNumber(rawX, 2, DEVICE_MAP_LAYER.width - 2, seed, 6, cellWidth * 0.9);
  const boundedY = softBoundNumber(rawY, 2, DEVICE_MAP_LAYER.height - 2, seed, 7, cellHeight * 0.9);
  const point = {
    left: (boundedX / DEVICE_MAP_LAYER.width) * 100,
    top: (boundedY / DEVICE_MAP_LAYER.height) * 100,
  };
  return {
    ...point,
    gridRow: softBoundGridCoord(grid.gridRow + offset.row, grid.rows, seed, 8),
    gridCol: softBoundGridCoord(grid.gridCol + offset.col, grid.cols, seed, 9),
  };
};

const stationDeviceForNode = (node) =>
  [node?.device_uid, node?.deployment_id, node?.id, node?.deployment_id ? `deployment:${node.deployment_id}` : ""]
    .filter(Boolean)
    .map((key) => deviceLookup.value.get(String(key)))
    .find(Boolean) || null;

const markerStyle = (marker) => ({
  left: `${marker.left}%`,
  top: `${marker.top}%`,
  "--marker-color": marker.color || STATION_MARKER_COLORS.default,
  "--marker-opacity": marker.opacity ?? 1,
  "--marker-size": `${marker.size || 8}px`,
});

const tooltipStyle = (marker) => {
  const horizontal = marker.left > 76 ? "calc(-100% - 14px)" : "14px";
  const vertical = marker.top < 18 ? "14px" : "calc(-100% - 14px)";
  return {
    left: `${marker.left}%`,
    top: `${marker.top}%`,
    transform: `translate(${horizontal}, ${vertical})`,
  };
};

const assignTypeForm = (type) => {
  Object.assign(typeForm, emptyTypeForm(), {
    device_category: displayDeviceText(type?.device_category || type?.label),
    coverage_radius_km: roundParameterNumber(type?.coverage_radius_km),
    coverage_radius: roundParameterNumber(type?.coverage_radius),
    downlink_bandwidth_mbps: roundParameterNumber(type?.downlink_bandwidth_mbps || type?.max_throughput),
    uplink_bandwidth_mbps: roundParameterNumber(type?.uplink_bandwidth_mbps),
    max_users: roundParameterNumber(type?.max_users, 0),
    tx_power_watt: roundOptionalParameterNumber(type?.tx_power_watt),
    battery_duration_h: roundOptionalParameterNumber(type?.battery_duration_h),
    notes: type?.notes || "",
  });
};

const assignModelForm = (model) => {
  Object.assign(modelForm, emptyTypeForm(), {
    device_category: displayDeviceText(model?.device_category || model?.label),
    coverage_radius_km: roundParameterNumber(model?.coverage_radius_km),
    coverage_radius: roundParameterNumber(model?.coverage_radius),
    downlink_bandwidth_mbps: roundParameterNumber(model?.downlink_bandwidth_mbps || model?.max_throughput),
    uplink_bandwidth_mbps: roundParameterNumber(model?.uplink_bandwidth_mbps),
    max_users: roundParameterNumber(model?.max_users, 0),
    tx_power_watt: roundOptionalParameterNumber(model?.tx_power_watt),
    battery_duration_h: roundOptionalParameterNumber(model?.battery_duration_h),
    notes: model?.notes || "",
  });
};

const assignDeviceForm = (device = null) => {
  const fallbackType = selectedType.value || deviceTypes.value[0] || {};
  const baseStation = device?.base_station || fallbackType.base_station || "";
  Object.assign(deviceForm, emptyDeviceForm(), {
    device_name: displayDeviceText(device?.device_name || device?.label),
    base_station: baseStation,
    mode: device?.mode || modesForType(baseStation)[0] || "",
    status: device?.status || "active",
    x: Number(device?.x || 0),
    y: Number(device?.y || 0),
    coverage_radius_km: roundParameterNumber(device?.coverage_radius_km || fallbackType.coverage_radius_km),
    coverage_radius: roundParameterNumber(device?.coverage_radius || fallbackType.coverage_radius),
    downlink_bandwidth_mbps: roundParameterNumber(device?.downlink_bandwidth_mbps || device?.max_throughput || fallbackType.downlink_bandwidth_mbps),
    uplink_bandwidth_mbps: roundParameterNumber(device?.uplink_bandwidth_mbps || fallbackType.uplink_bandwidth_mbps),
    max_users: roundParameterNumber(device?.max_users || fallbackType.max_users, 0),
    tx_power_watt: roundOptionalParameterNumber(device?.tx_power_watt ?? fallbackType.tx_power_watt),
    battery_duration_h: roundOptionalParameterNumber(device?.battery_duration_h ?? fallbackType.battery_duration_h),
    notes: device?.notes || "",
  });
};

const assignBlockForm = (block = null) => {
  const fallbackType = selectedType.value || deviceTypes.value[0] || {};
  const baseStation = block?.base_station || fallbackType.base_station || "";
  Object.assign(blockForm, emptyBlockForm(), {
    x: Number(block?.x || 0),
    y: Number(block?.y || 0),
    base_station: baseStation,
    mode: block?.mode || modesForType(baseStation)[0] || "",
    status: block?.status || "active",
    quantity: Number(block?.quantity ?? 1),
  });
};

const configPayload = (form) => ({
  device_category: form.device_category || undefined,
  coverage_radius_km: roundParameterNumber(form.coverage_radius_km),
  coverage_radius: roundParameterNumber(form.coverage_radius),
  max_throughput: roundParameterNumber(form.downlink_bandwidth_mbps || form.max_throughput),
  downlink_bandwidth_mbps: roundParameterNumber(form.downlink_bandwidth_mbps || form.max_throughput),
  uplink_bandwidth_mbps: roundParameterNumber(form.uplink_bandwidth_mbps),
  max_users: Math.max(0, Math.round(Number(form.max_users || 0))),
  tx_power_watt: form.tx_power_watt === null || form.tx_power_watt === "" ? undefined : roundParameterNumber(form.tx_power_watt),
  battery_duration_h: form.battery_duration_h === null || form.battery_duration_h === "" ? undefined : roundParameterNumber(form.battery_duration_h),
  notes: form.notes || undefined,
});

const normalizeParameterFormDecimals = (form) => {
  form.coverage_radius_km = roundParameterNumber(form.coverage_radius_km);
  form.coverage_radius = roundParameterNumber(form.coverage_radius);
  form.downlink_bandwidth_mbps = roundParameterNumber(form.downlink_bandwidth_mbps);
  form.uplink_bandwidth_mbps = roundParameterNumber(form.uplink_bandwidth_mbps);
  form.max_users = Math.max(0, Math.round(Number(form.max_users || 0)));
  form.tx_power_watt = roundOptionalParameterNumber(form.tx_power_watt);
  form.battery_duration_h = roundOptionalParameterNumber(form.battery_duration_h);
};

const normalizeModelFormDecimals = () => normalizeParameterFormDecimals(modelForm);
const normalizeDeviceFormDecimals = () => normalizeParameterFormDecimals(deviceForm);

const devicePayload = () => ({
  device_name: deviceForm.device_name || undefined,
  base_station: deviceForm.base_station,
  mode: deviceForm.mode || modesForType(deviceForm.base_station)[0] || null,
  status: deviceForm.status || "active",
  x: Math.max(0, Math.min(gridRows.value - 1, Math.round(Number(deviceForm.x || 0)))),
  y: Math.max(0, Math.min(gridCols.value - 1, Math.round(Number(deviceForm.y || 0)))),
  ...configPayload(deviceForm),
});

const applyState = (payload) => {
  deviceState.value = payload;
  if (!selectedTypeKey.value || !deviceTypes.value.some((item) => item.base_station === selectedTypeKey.value)) {
    selectedTypeKey.value = deviceTypes.value[0]?.base_station || "";
  }
  if (!selectedModelKey.value || !deviceModels.value.some((item) => item.model_key === selectedModelKey.value)) {
    selectedModelKey.value = deviceModels.value[0]?.model_key || "";
  }
  if (!selectedBlockKey.value && blocks.value.length) selectBlock(blocks.value[0]);
  if (!selectedDeviceId.value && devices.value.length) selectDevice(devices.value[0]);
};

const syncScenarioSelectors = () => {
  const scenario = currentScenario.value;
  if (!scenario) return;
  selectedDisasterKey.value = scenarioSourceKey(scenario);
  selectedSeverityKey.value = scenarioSeverityKey(scenario);
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

const selectDisasterScenario = (sourceKey) => {
  selectScenarioByParts(sourceKey);
};

const selectSeverity = (severityKey) => {
  selectScenarioByParts(selectedDisasterKey.value, severityKey);
};

const fetchScenarios = async () => {
  try {
    const { data } = await axios.get(`${API_BASE}/scenarios`, { timeout: SCENE_ACCESS_TIMEOUT_MS });
    scenarios.value = (Array.isArray(data?.scenarios) ? data.scenarios : []).slice().sort(compareScenarioRecords);
    if (!scenarios.value.length) {
      const catalogResponse = await axios.get(`${API_BASE}/disaster-scenarios`, { timeout: SCENE_ACCESS_TIMEOUT_MS });
      scenarios.value = scenarioRecordsFromDisasterCatalog(catalogResponse.data?.scenarios).sort(compareScenarioRecords);
    }
  } catch (error) {
    scenarios.value = scenarioRecordsFromDisasterCatalog([]).sort(compareScenarioRecords);
    appendDeviceTerminalLine(`后端场景接口暂不可用，已启用本地灾害场景兜底：${error?.message || error}`, { level: "WARN" });
  }
  if (scenarios.value.length && !scenarios.value.some((item) => item.name === selectedScenarioName.value)) {
    selectedScenarioName.value = preferredScenarioForSource(scenarioSourceKey(scenarios.value[0]))?.name || scenarios.value[0].name;
  }
  syncScenarioSelectors();
};

const fetchDeviceState = async () => {
  if (!selectedScenarioName.value) return;
  terminalStatus.value = "loading";
  appendDeviceTerminalLine(`请求后端设备状态：scenario=${selectedScenarioName.value}`, { level: "ACTION" });
  const { data } = await axios.get(`${API_BASE}/scenarios/${encodeURIComponent(selectedScenarioName.value)}/device-state`, {
    timeout: SCENE_ACCESS_TIMEOUT_MS,
  });
  applyState(data);
  const counts = data?.status_counts || {};
  appendDeviceTerminalLine(
    `后端返回设备状态：total=${counts.total || data?.devices?.length || 0} active=${counts.active || 0} degraded=${counts.degraded || 0} offline=${counts.offline || 0}`,
    { level: "BACKEND" }
  );
  appendDeviceUserNodeCount(`设备管理接入灾害场景：${displayScenarioWithSeverity(currentScenario.value)}`, currentScenario.value);
  await fetchScenePreview(data.devices || []);
  terminalStatus.value = "completed";
};

const fetchScenePreview = async (baseStations = devices.value) => {
  if (!selectedScenarioName.value) return;
  sceneLoading.value = true;
  sceneError.value = "";
  try {
    const { data } = await axios.post(
      `${API_BASE}/simulate/scene`,
      {
        scenario_name: selectedScenarioName.value,
        env_type: "multimodal",
        custom_base_stations: baseStations,
      },
      { timeout: SCENE_ACCESS_TIMEOUT_MS }
    );
    scenePreview.value = data;
    appendDeviceUserNodeCount(`设备管理接入灾害场景：${displayScenarioWithSeverity(currentScenario.value)}`, data, currentScenario.value);
  } catch (error) {
    console.error("Failed to load scenario map preview", error);
    scenePreview.value = null;
    sceneError.value = `场景地图加载失败：${error?.response?.data?.detail || error?.message || error}`;
  } finally {
    sceneLoading.value = false;
  }
};

const refreshAll = async () => {
  loading.value = true;
  errorMessage.value = "";
  terminalStatus.value = "loading";
  appendDeviceTerminalLine("刷新设备管理页面数据。", { level: "ACTION" });
  try {
    if (!scenarios.value.length) await fetchScenarios();
    await fetchDeviceState();
  } catch (error) {
    console.error("Failed to load scenario device state", error);
    errorMessage.value = `设备现状加载失败：${error?.response?.data?.detail || error?.message || error}`;
    terminalStatus.value = "failed";
    appendDeviceTerminalLine(errorMessage.value, { level: "ERROR" });
  } finally {
    loading.value = false;
  }
};

const restoreOriginalScenarioBaseStations = async () => {
  if (!selectedScenarioName.value || loading.value || saving.value) return false;
  if (!window.confirm("确认恢复当前场景的原始基站清单？这会覆盖当前设备管理中的基站增删改。")) return false;
  loading.value = true;
  errorMessage.value = "";
  terminalStatus.value = "running";
  appendDeviceTerminalLine(`恢复原始场景基站：scenario=${selectedScenarioName.value}`, { level: "ACTION" });
  try {
    await axios.delete(`${API_BASE}/scenarios/${encodeURIComponent(selectedScenarioName.value)}/base-stations`, {
      timeout: SCENE_ACCESS_TIMEOUT_MS,
    });
    await fetchScenarios();
    await fetchDeviceState();
    showStatus("已恢复原始场景基站。", "success");
    return true;
  } catch (error) {
    console.error("Failed to restore original scenario base stations", error);
    showStatus(`恢复原始场景失败：${error?.response?.data?.detail || error?.message || error}`, "error", 5200);
    return false;
  } finally {
    loading.value = false;
  }
};

const selectBlock = (block) => {
  selectedBlockKey.value = blockKey(block);
  assignBlockForm(block);
  appendDeviceTerminalLine(`选择场景块：${block?.base_station || "--"} (${block?.x ?? "-"}, ${block?.y ?? "-"})`, { level: "ACTION" });
};

const selectDevice = (device) => {
  selectedDeviceId.value = deviceId(device);
  assignDeviceForm(device);
  appendDeviceTerminalLine(`选择设备实例：${deviceId(device) || "--"} ${device?.base_station || ""}`, { level: "ACTION" });
};

const prepareNewBlock = () => {
  selectedBlockKey.value = "";
  assignBlockForm(null);
  appendDeviceTerminalLine("切换到新增场景块配置。", { level: "ACTION" });
};

const prepareNewDevice = () => {
  selectedDeviceId.value = "";
  assignDeviceForm(null);
  appendDeviceTerminalLine("切换到新增设备实例。", { level: "ACTION" });
};

const saveTypeConfig = async () => {
  if (!selectedTypeKey.value) return;
  saving.value = true;
  terminalStatus.value = "running";
  appendDeviceTerminalLine(`保存设备类型参数：${selectedTypeKey.value}`, { level: "ACTION" });
  try {
    const nextOverrides = { ...typeOverrides.value, [selectedTypeKey.value]: configPayload(typeForm) };
    const { data } = await axios.put(
      `${API_BASE}/scenarios/${encodeURIComponent(selectedScenarioName.value)}/device-state`,
      { type_overrides: nextOverrides, operation: "update_type_config" },
      { timeout: SCENE_ACCESS_TIMEOUT_MS }
    );
    applyState(data);
    await fetchScenePreview(data.devices || []);
    showStatus("设备类型参数已保存，训练和测试将使用新配置。", "success");
  } catch (error) {
    showStatus(`保存类型参数失败：${error?.response?.data?.detail || error?.message || error}`, "error", 5200);
  } finally {
    saving.value = false;
  }
};

const resetTypeConfig = async () => {
  if (!selectedTypeKey.value) return;
  saving.value = true;
  terminalStatus.value = "running";
  appendDeviceTerminalLine(`恢复设备类型默认参数：${selectedTypeKey.value}`, { level: "ACTION" });
  try {
    const nextOverrides = { ...typeOverrides.value };
    delete nextOverrides[selectedTypeKey.value];
    const { data } = await axios.put(
      `${API_BASE}/scenarios/${encodeURIComponent(selectedScenarioName.value)}/device-state`,
      { type_overrides: nextOverrides, operation: "reset_type_config" },
      { timeout: SCENE_ACCESS_TIMEOUT_MS }
    );
    applyState(data);
    await fetchScenePreview(data.devices || []);
    showStatus("设备类型参数已恢复默认。", "success");
  } catch (error) {
    showStatus(`恢复默认失败：${error?.response?.data?.detail || error?.message || error}`, "error", 5200);
  } finally {
    saving.value = false;
  }
};

const saveModelConfig = async () => {
  if (!selectedModelKey.value) return;
  saving.value = true;
  terminalStatus.value = "running";
  appendDeviceTerminalLine(`保存设备型号参数：${selectedModelKey.value}`, { level: "ACTION" });
  try {
    const nextOverrides = { ...typeOverrides.value, [selectedModelKey.value]: configPayload(modelForm) };
    const { data } = await axios.put(
      `${API_BASE}/scenarios/${encodeURIComponent(selectedScenarioName.value)}/device-state`,
      { type_overrides: nextOverrides, operation: "update_device_model_config" },
      { timeout: SCENE_ACCESS_TIMEOUT_MS }
    );
    applyState(data);
    await fetchScenePreview(data.devices || []);
    showStatus("设备型号参数已保存，同型号设备将使用新配置。", "success");
  } catch (error) {
    showStatus(`保存型号参数失败：${error?.response?.data?.detail || error?.message || error}`, "error", 5200);
  } finally {
    saving.value = false;
  }
};

const resetModelConfig = async () => {
  if (!selectedModelKey.value) return;
  saving.value = true;
  terminalStatus.value = "running";
  appendDeviceTerminalLine(`恢复设备型号默认参数：${selectedModelKey.value}`, { level: "ACTION" });
  try {
    const nextOverrides = { ...typeOverrides.value };
    delete nextOverrides[selectedModelKey.value];
    const { data } = await axios.put(
      `${API_BASE}/scenarios/${encodeURIComponent(selectedScenarioName.value)}/device-state`,
      { type_overrides: nextOverrides, operation: "reset_device_model_config" },
      { timeout: SCENE_ACCESS_TIMEOUT_MS }
    );
    applyState(data);
    await fetchScenePreview(data.devices || []);
    showStatus("设备型号参数已恢复默认。", "success");
  } catch (error) {
    showStatus(`恢复型号默认失败：${error?.response?.data?.detail || error?.message || error}`, "error", 5200);
  } finally {
    saving.value = false;
  }
};

const saveBlockQuantity = async () => {
  saving.value = true;
  terminalStatus.value = "running";
  appendDeviceTerminalLine(
    `保存场景块数量：${blockForm.base_station} (${blockForm.x}, ${blockForm.y}) quantity=${blockForm.quantity}`,
    { level: "ACTION" }
  );
  try {
    const selectedTypeConfig = deviceTypes.value.find((item) => item.base_station === blockForm.base_station) || {};
    const { data } = await axios.patch(
      `${API_BASE}/scenarios/${encodeURIComponent(selectedScenarioName.value)}/device-blocks`,
      {
        x: Math.max(0, Math.min(gridRows.value - 1, Math.round(Number(blockForm.x || 0)))),
        y: Math.max(0, Math.min(gridCols.value - 1, Math.round(Number(blockForm.y || 0)))),
        base_station: blockForm.base_station,
        mode: blockForm.mode || modesForType(blockForm.base_station)[0] || null,
        status: blockForm.status || "active",
        quantity: Math.max(0, Math.round(Number(blockForm.quantity || 0))),
        parameters: configPayload(selectedTypeConfig),
        operation: "update_block_quantity",
      },
      { timeout: SCENE_ACCESS_TIMEOUT_MS }
    );
    applyState(data);
    await fetchScenePreview(data.devices || []);
    showStatus("场景块基站数量已更新。", "success");
  } catch (error) {
    showStatus(`块数量保存失败：${error?.response?.data?.detail || error?.message || error}`, "error", 5200);
  } finally {
    saving.value = false;
  }
};

const deleteSelectedBlock = async () => {
  if (!selectedBlockKey.value || !blockForm.base_station) return false;
  const label = `${stationLabel(blockForm.base_station)} (${blockForm.x}, ${blockForm.y})`;
  if (!window.confirm(`确认删除块配置 ${label}？`)) return false;
  saving.value = true;
  terminalStatus.value = "running";
  appendDeviceTerminalLine(`删除场景块配置：${label}`, { level: "ACTION" });
  try {
    const selectedTypeConfig = deviceTypes.value.find((item) => item.base_station === blockForm.base_station) || {};
    const { data } = await axios.patch(
      `${API_BASE}/scenarios/${encodeURIComponent(selectedScenarioName.value)}/device-blocks`,
      {
        x: Math.max(0, Math.min(gridRows.value - 1, Math.round(Number(blockForm.x || 0)))),
        y: Math.max(0, Math.min(gridCols.value - 1, Math.round(Number(blockForm.y || 0)))),
        base_station: blockForm.base_station,
        mode: blockForm.mode || modesForType(blockForm.base_station)[0] || null,
        status: blockForm.status || "active",
        quantity: 0,
        parameters: configPayload(selectedTypeConfig),
        operation: "delete_block_quantity",
      },
      { timeout: SCENE_ACCESS_TIMEOUT_MS }
    );
    selectedBlockKey.value = "";
    assignBlockForm(null);
    applyState(data);
    await fetchScenePreview(data.devices || []);
    showStatus("场景块配置已删除。", "success");
    return true;
  } catch (error) {
    showStatus(`删除块配置失败：${error?.response?.data?.detail || error?.message || error}`, "error", 5200);
    return false;
  } finally {
    saving.value = false;
  }
};

const saveDeviceConfig = async () => {
  saving.value = true;
  const wasEditing = Boolean(selectedDeviceId.value);
  terminalStatus.value = "running";
  appendDeviceTerminalLine(
    `${wasEditing ? "保存设备实例" : "新增设备实例"}：${deviceForm.base_station} (${deviceForm.x}, ${deviceForm.y}) status=${deviceForm.status}`,
    { level: "ACTION" }
  );
  try {
    const payload = devicePayload();
    const url = `${API_BASE}/scenarios/${encodeURIComponent(selectedScenarioName.value)}/devices`;
    const { data } = selectedDeviceId.value
      ? await axios.patch(`${url}/${encodeURIComponent(selectedDeviceId.value)}`, payload, { timeout: SCENE_ACCESS_TIMEOUT_MS })
      : await axios.post(url, payload, { timeout: SCENE_ACCESS_TIMEOUT_MS });
    applyState(data);
    if (!wasEditing && data.devices?.length) {
      selectDevice(data.devices[data.devices.length - 1]);
    }
    await fetchScenePreview(data.devices || []);
    showStatus(wasEditing ? "单设备参数已保存。" : "设备实例已新增。", "success");
    return true;
  } catch (error) {
    showStatus(`设备保存失败：${error?.response?.data?.detail || error?.message || error}`, "error", 5200);
    return false;
  } finally {
    saving.value = false;
  }
};

const deleteSelectedDevice = async () => {
  if (!selectedDeviceId.value) return false;
  if (!window.confirm(`确认删除设备 ${selectedDeviceId.value}？`)) return false;
  saving.value = true;
  terminalStatus.value = "running";
  appendDeviceTerminalLine(`删除设备实例：${selectedDeviceId.value}`, { level: "ACTION" });
  try {
    const { data } = await axios.delete(
      `${API_BASE}/scenarios/${encodeURIComponent(selectedScenarioName.value)}/devices/${encodeURIComponent(selectedDeviceId.value)}`,
      { timeout: SCENE_ACCESS_TIMEOUT_MS }
    );
    selectedDeviceId.value = "";
    applyState(data);
    await fetchScenePreview(data.devices || []);
    showStatus("设备实例已删除。", "success");
    return true;
  } catch (error) {
    showStatus(`删除失败：${error?.response?.data?.detail || error?.message || error}`, "error", 5200);
    return false;
  } finally {
    saving.value = false;
  }
};

const openMapDeviceDetail = (marker) => {
  if (marker?.kind !== "station") return;
  const device = marker.device || deviceLookup.value.get(String(marker.deviceUid || ""));
  if (!device) {
    showStatus("该基站没有绑定设备记录，无法直接编辑。", "error", 3600);
    return;
  }
  selectDevice(device);
  mapDetailOpen.value = true;
};

const closeMapDeviceDetail = () => {
  mapDetailOpen.value = false;
};

const saveMapDeviceConfig = async () => {
  const ok = await saveDeviceConfig();
  if (ok) mapDetailOpen.value = false;
};

const deleteMapDevice = async () => {
  const ok = await deleteSelectedDevice();
  if (ok) mapDetailOpen.value = false;
};

watch(selectedScenarioName, async () => {
  syncScenarioSelectors();
  selectedTypeKey.value = "";
  selectedModelKey.value = "";
  selectedBlockKey.value = "";
  selectedDeviceId.value = "";
  deviceState.value = null;
  scenePreview.value = null;
  hoveredMapMarker.value = null;
  mapDetailOpen.value = false;
  if (selectedScenarioName.value) await refreshAll();
});

watch(selectedType, (type) => {
  if (type) assignTypeForm(type);
});

watch(selectedModel, (model) => {
  if (model) assignModelForm(model);
});

watch(
  () => deviceForm.base_station,
  (baseStation) => {
    if (!modesForType(baseStation).includes(deviceForm.mode)) {
      deviceForm.mode = modesForType(baseStation)[0] || "";
    }
  }
);

watch(
  () => blockForm.base_station,
  (baseStation) => {
    if (!modesForType(baseStation).includes(blockForm.mode)) {
      blockForm.mode = modesForType(baseStation)[0] || "";
    }
  }
);

onMounted(async () => {
  await fetchScenarios();
  await refreshAll();
});

onBeforeUnmount(() => {
  if (statusTimer) window.clearTimeout(statusTimer);
});
</script>

<style scoped>
.device-state-page {
  position: relative;
  width: 1920px;
  height: 1010px;
  min-height: 1010px;
  overflow: hidden;
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

.device-shell {
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

.device-shell::-webkit-scrollbar {
  width: 8px;
}

.device-shell::-webkit-scrollbar-track {
  background: rgba(225, 236, 255, 0.72);
  border-radius: 999px;
}

.device-shell::-webkit-scrollbar-thumb {
  background: rgba(57, 97, 246, 0.45);
  border-radius: 999px;
}

.device-shell__scroll {
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

.scenario-select-panel,
.device-map-panel,
.device-param-panel,
.config-card,
.summary-grid article {
  box-sizing: border-box;
  border: 1px solid rgba(233, 233, 233, 1);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.96);
  box-shadow: 3px 3px 20px rgba(233, 233, 233, 0.9);
}

.tracking-terminal-panel {
  width: 1628px;
  margin: 14px 0 0 4px;
}

.filter-row,
.form-actions {
  display: flex;
  align-items: end;
  gap: 10px;
}

label {
  display: flex;
  min-width: 0;
  flex-direction: column;
  gap: 6px;
  color: #334155;
  font-size: 12px;
  font-weight: 700;
}

select,
input {
  width: 100%;
  height: 34px;
  border: 1px solid #d7e3f4;
  border-radius: 6px;
  background: #fff;
  color: #17315d;
  font-size: 14px;
  padding: 0 10px;
}

button {
  cursor: pointer;
}

.primary-button,
.ghost-button,
.danger-button {
  height: 34px;
  border-radius: 6px;
  padding: 0 14px;
  font-size: 14px;
  font-weight: 700;
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

.danger-button {
  border: 1px solid #fecaca;
  background: #fff1f2;
  color: #b91c1c;
}

button:disabled {
  cursor: not-allowed;
  opacity: 0.56;
}

.status-toast,
.module-error {
  position: fixed;
  left: 150px;
  top: 92px;
  z-index: 50;
  min-width: 380px;
  max-width: 900px;
  padding: 9px 14px;
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.98);
  box-shadow: 3px 3px 20px rgba(204, 213, 226, 0.54);
}

.status-toast {
  border: 1px solid #bfdbfe;
  color: #1d4ed8;
}

.status-toast--success {
  color: #15803d;
}

.status-toast--error,
.module-error {
  color: #b91c1c;
}

.module-error {
  top: 140px;
  border: 1px solid #fecaca;
}

.scenario-select-panel {
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
  overflow: hidden;
  color: #64748b;
  font-size: 12px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.module-actions {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.dataset-controls {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  gap: 12px;
  align-items: stretch;
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

.device-map-panel {
  margin-top: 14px;
  padding: 14px;
}

.device-map-shell {
  position: relative;
  --device-map-bg-scale: 1;
  width: 1618px;
  max-width: calc(100% + 8px);
  height: 745px;
  overflow: hidden;
  border: 1px solid rgba(183, 224, 254, 0.58);
  border-radius: 8px;
  background: #dbeafe;
}

.device-map-image {
  position: absolute;
  inset: 0;
  z-index: 1;
  width: 100%;
  height: 100%;
  object-fit: cover;
  filter: saturate(1.04) contrast(0.98);
  transform: scale(var(--device-map-bg-scale));
  transform-origin: 50% 50%;
}

.device-tile-map {
  position: absolute;
  inset: 0;
  z-index: 2;
  overflow: hidden;
  background: #dbeafe;
  pointer-events: none;
}

.device-tile-map img {
  position: absolute;
  width: 256px;
  height: 256px;
  border: 0;
}

.device-tile-map__label {
  position: absolute;
  left: 18px;
  top: 18px;
  z-index: 3;
  max-width: 420px;
  overflow: hidden;
  border-radius: 6px;
  background: rgba(15, 23, 42, 0.62);
  color: #fff;
  font-size: 14px;
  line-height: 20px;
  padding: 8px 12px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.device-map-layer {
  position: absolute;
  inset: 0;
  z-index: 6;
  pointer-events: none;
}

.device-map-marker {
  position: absolute;
  width: var(--marker-size, 8px);
  height: var(--marker-size, 8px);
  border-radius: 50%;
  background: var(--marker-color, #0ea5e9);
  opacity: var(--marker-opacity, 1);
  transform: translate(-50%, -50%);
}

.device-map-marker--user {
  border: 0;
  box-shadow:
    0 0 0 1px rgba(255, 255, 255, 0.2),
    0 0 5px color-mix(in srgb, var(--marker-color, #ef4444) 48%, transparent);
  pointer-events: auto;
}

.device-map-marker--station {
  border: 2px solid rgba(255, 255, 255, 0.94);
  box-shadow:
    0 0 0 3px color-mix(in srgb, var(--marker-color, #2563eb) 24%, transparent),
    0 3px 8px rgba(15, 23, 42, 0.28);
  cursor: pointer;
  pointer-events: auto;
}

.device-map-marker--station::after,
.device-map-marker--user::after {
  content: "";
  position: absolute;
  inset: -8px;
  border-radius: 50%;
}

.device-map-marker--station:focus-visible {
  outline: 3px solid rgba(14, 165, 233, 0.72);
  outline-offset: 6px;
}

.device-map-empty {
  position: absolute;
  left: 50%;
  top: 50%;
  z-index: 8;
  display: flex;
  width: 360px;
  max-width: calc(100% - 48px);
  min-height: 92px;
  flex-direction: column;
  justify-content: center;
  gap: 8px;
  border: 1px solid rgba(183, 224, 254, 0.62);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.92);
  color: #334155;
  padding: 14px 18px;
  text-align: center;
  transform: translate(-50%, -50%);
}

.device-map-empty strong {
  color: #1f2d3d;
  font-size: 16px;
  font-weight: 700;
}

.device-map-empty span {
  overflow: hidden;
  color: #64748b;
  font-size: 12px;
  line-height: 18px;
  text-overflow: ellipsis;
}

.device-map-tooltip {
  position: absolute;
  z-index: 20;
  width: 274px;
  min-height: 118px;
  padding: 12px 14px;
  border: 1px solid rgba(148, 163, 184, 0.32);
  border-radius: 8px;
  background: rgba(15, 23, 42, 0.92);
  color: #f8fafc;
  box-shadow: 0 16px 34px rgba(15, 23, 42, 0.28);
  pointer-events: none;
  backdrop-filter: blur(8px);
}

.device-map-tooltip strong,
.device-map-tooltip span {
  display: block;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.device-map-tooltip strong {
  margin-bottom: 4px;
  font-size: 15px;
}

.device-map-tooltip span {
  color: #cbd5e1;
  font-size: 12px;
}

.device-map-tooltip dl {
  display: grid;
  grid-template-columns: 74px minmax(0, 1fr);
  gap: 5px 8px;
  margin: 10px 0 0;
  font-size: 12px;
}

.device-map-tooltip dt {
  color: #94a3b8;
}

.device-map-tooltip dd {
  min-width: 0;
  margin: 0;
  overflow: hidden;
  color: #f8fafc;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.device-map-legend {
  position: absolute;
  left: 14px;
  bottom: 14px;
  z-index: 9;
  display: flex;
  max-width: calc(100% - 28px);
  flex-wrap: wrap;
  gap: 8px;
}

.device-map-legend span {
  display: inline-flex;
  align-items: center;
  height: 26px;
  gap: 6px;
  border: 1px solid rgba(183, 224, 254, 0.55);
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.9);
  color: #334155;
  font-size: 12px;
  padding: 0 10px;
  box-shadow: 0 2px 8px rgba(15, 23, 42, 0.08);
}

.device-map-legend i {
  width: 9px;
  height: 9px;
  border-radius: 50%;
}

.summary-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin-top: 14px;
}

.summary-grid article {
  min-height: 82px;
  padding: 13px 14px;
}

.summary-grid small,
.summary-grid span,
.panel-heading p,
.block-row small,
.device-row small,
.empty-note {
  color: #64748b;
  font-size: 12px;
}

.summary-grid strong {
  display: block;
  margin: 5px 0;
  color: #1f2d3d;
  font-size: 24px;
}

.device-param-panel {
  margin-top: 14px;
  padding: 14px;
}

.param-config-layout {
  display: grid;
  grid-template-columns: 430px minmax(0, 1fr);
  gap: 14px;
  align-items: start;
}

.param-type-column {
  display: flex;
  min-width: 0;
  max-height: 356px;
  flex-direction: column;
  gap: 8px;
  overflow: auto;
  padding-right: 4px;
}

.param-type-row {
  display: grid;
  width: 100%;
  min-height: 58px;
  grid-template-columns: minmax(0, 1fr) 64px;
  align-items: center;
  gap: 10px;
  border: 1px solid rgba(183, 224, 254, 0.5);
  border-radius: 10px;
  background: rgba(255, 255, 255, 0.78);
  color: #333333;
  padding: 9px 10px;
  text-align: left;
  transition: all 0.2s ease;
}

.param-type-row:hover {
  border-color: #1890ff;
  background: rgba(231, 238, 255, 0.62);
}

.param-type-row.active {
  border-color: #3961f6;
  background: rgba(231, 238, 255, 0.72);
  color: #3961f6;
  box-shadow: 0 0 6px rgba(57, 97, 246, 0.16);
}

.param-type-row span {
  min-width: 0;
}

.param-type-row strong,
.param-type-row small {
  display: block;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.param-type-row strong {
  color: #0f172a;
  font-size: 14px;
}

.param-type-row small {
  margin-top: 3px;
  color: #64748b;
  font-size: 12px;
}

.param-type-row em {
  min-width: 0;
  overflow: hidden;
  color: #1d4ed8;
  font-size: 12px;
  font-style: normal;
  font-weight: 700;
  text-align: right;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.param-editor {
  min-width: 0;
}

.param-summary-strip {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 8px;
  margin-bottom: 12px;
}

.param-summary-strip span {
  min-width: 0;
  border: 1px solid rgba(183, 224, 254, 0.5);
  border-radius: 8px;
  background: rgba(248, 251, 255, 0.88);
  padding: 8px 10px;
}

.param-summary-strip small,
.param-summary-strip strong {
  display: block;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.param-summary-strip small {
  color: #64748b;
  font-size: 12px;
}

.param-summary-strip strong {
  margin-top: 4px;
  color: #1f2d3d;
  font-size: 18px;
}

.param-config-form {
  grid-template-columns: repeat(4, minmax(0, 1fr));
}

.workbench {
  display: flex;
  flex-direction: column;
  gap: 14px;
  margin-top: 14px;
}

.config-card {
  display: grid;
  grid-template-columns: 520px minmax(0, 1fr);
  gap: 14px;
  min-height: 420px;
  padding: 14px;
}

.config-card--device {
  min-height: 690px;
}

.config-card__list-pane,
.config-card__editor {
  min-width: 0;
}

.config-card__list-pane {
  border-right: 1px solid rgba(215, 227, 244, 0.78);
  padding-right: 14px;
}

.config-card__editor {
  padding-left: 2px;
}

.panel-heading {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 12px;
}

.panel-heading--compact {
  margin-top: 18px;
}

.panel-heading h2 {
  margin: 0;
  color: #1f2d3d;
  font-size: 16px;
  font-weight: 400;
}

.panel-heading p {
  margin: 4px 0 0;
}

.filter-row {
  align-items: center;
  margin-bottom: 10px;
}

.filter-row input {
  flex: 1;
}

.filter-row select {
  width: 126px;
}

.block-list,
.device-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
  overflow: auto;
  padding-right: 4px;
}

.block-list {
  max-height: 330px;
}

.device-list {
  max-height: 600px;
}

.block-row,
.device-row {
  display: grid;
  width: 100%;
  min-height: 58px;
  align-items: center;
  gap: 10px;
  border: 1px solid rgba(183, 224, 254, 0.5);
  border-radius: 10px;
  background: rgba(255, 255, 255, 0.78);
  color: #333333;
  text-align: left;
  padding: 9px 10px;
  transition: all 0.2s ease;
}

.block-row {
  grid-template-columns: 88px minmax(0, 1fr) 56px;
}

.device-row {
  grid-template-columns: 16px minmax(0, 1fr) 86px;
}

.device-row__content {
  min-width: 0;
  overflow: hidden;
}

.block-row.active,
.device-row.active {
  border-color: #3961f6;
  background: rgba(231, 238, 255, 0.72);
  color: #3961f6;
  box-shadow: 0 0 6px rgba(57, 97, 246, 0.16);
}

.block-row:hover,
.device-row:hover {
  border-color: #1890ff;
  background: rgba(231, 238, 255, 0.62);
}

.block-row strong,
.device-row strong {
  display: block;
  min-width: 0;
  overflow: hidden;
  color: #0f172a;
  font-size: 14px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.device-row small {
  display: block;
  max-width: 100%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.block-row b {
  color: #1d4ed8;
  font-size: 24px;
  text-align: center;
}

.device-row em {
  min-width: 0;
  overflow: hidden;
  color: #334155;
  font-size: 12px;
  font-style: normal;
  text-align: right;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.grid-cell,
.sync-pill {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  height: 28px;
  border-radius: 999px;
  background: #e8f4ff;
  color: #1d4ed8;
  font-size: 12px;
  font-weight: 700;
}

.status-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: #94a3b8;
}

.status-dot--active {
  background: #22c55e;
}

.status-dot--degraded {
  background: #f59e0b;
}

.status-dot--offline {
  background: #ef4444;
}

.config-form {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
}

.wide {
  grid-column: 1 / -1;
}

.form-actions {
  justify-content: flex-end;
  padding-top: 4px;
}

.tracking-panel {
  margin-top: 14px;
  padding: 14px;
}

.tracking-table {
  overflow: auto;
}

.tracking-head,
.tracking-row {
  display: grid;
  grid-template-columns: 220px 180px repeat(4, 100px);
  gap: 10px;
  align-items: center;
  min-width: 820px;
  padding: 0 10px;
}

.tracking-head {
  height: 34px;
  border-top: 1px solid rgba(183, 224, 254, 0.5);
  border-bottom: 1px solid rgba(183, 224, 254, 0.5);
  background: rgba(248, 251, 255, 0.88);
  color: #64748b;
  font-size: 13px;
}

.tracking-row {
  min-height: 42px;
  border-bottom: 1px solid #edf2f7;
  color: #334155;
  font-size: 13px;
}

.empty-note {
  margin: 12px 0;
}

.device-detail-modal {
  position: fixed;
  inset: 0;
  z-index: 80;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(15, 23, 42, 0.38);
}

.device-detail-dialog {
  box-sizing: border-box;
  width: 940px;
  max-width: calc(100vw - 64px);
  max-height: calc(100vh - 80px);
  overflow: auto;
  border: 1px solid rgba(233, 233, 233, 1);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.98);
  box-shadow: 0 24px 58px rgba(15, 23, 42, 0.26);
  padding: 18px;
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
