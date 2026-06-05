<template>
  <section class="replay-page">
    <div class="replay-page__header">
      <div>
        <h2>场景回放工作台</h2>
        <p>读取后端回放会话，逐帧查看训练或测试生成的恢复过程。</p>
      </div>
      <button type="button" :disabled="isLoadingSessions" @click="refreshSessions">
        {{ isLoadingSessions ? "刷新中..." : "刷新列表" }}
      </button>
    </div>

    <div v-if="errorMessage" class="replay-page__status">{{ errorMessage }}</div>

    <div class="replay-page__layout">
      <aside class="replay-page__list">
        <div class="replay-page__list-head">
          <strong>回放记录</strong>
          <span>{{ sessions.length }} 条</span>
        </div>

        <button
          v-for="session in sessions"
          :key="session.id"
          type="button"
          :class="['replay-session', { 'replay-session--active': session.id === activeSessionId }]"
          @click="selectSession(session.id)"
        >
          <strong>{{ session.title }}</strong>
          <small>{{ sourceLabel(session.source) }} · {{ formatTime(session.createdAt) }}</small>
          <span>覆盖 {{ percentageText(session.summary?.coverageRatio) }}</span>
          <span>帧 {{ session.frameCount }} · 节点 {{ formatNumber(session.nodeCountTotal) }}</span>
        </button>

        <div v-if="isLoadingSessions && !sessions.length" class="replay-page__empty">
          正在读取后端回放会话。
        </div>
        <div v-else-if="!sessions.length" class="replay-page__empty">
          当前没有后端回放记录。请先在训练中心或策略测试中心完成一次测试。
        </div>
      </aside>

      <section class="replay-page__stage">
        <template v-if="activeSession && currentFrame">
          <div class="replay-page__meta">
            <article v-for="item in summaryItems" :key="item.label">
              <small>{{ item.label }}</small>
              <strong>{{ item.value }}</strong>
            </article>
          </div>

          <div class="replay-page__controls">
            <label>
              <span>帧序号</span>
              <input
                v-model.number="frameIndex"
                type="range"
                min="0"
                :max="maxFrameIndex"
                step="1"
                :disabled="isLoadingFrame || !activeSession.frameCount"
              />
            </label>
            <div class="replay-page__frame-text">
              Frame {{ frameIndex + 1 }} / {{ activeSession.frameCount || 1 }}
              <small v-if="isLoadingFrame">载入中</small>
            </div>
            <div class="replay-page__actions">
              <button type="button" @click="togglePlayback">{{ isPlaying ? "暂停" : "播放" }}</button>
              <button type="button" @click="resetPlayback">重置</button>
              <button type="button" @click="downloadArtifact('log')">日志</button>
              <button type="button" @click="downloadArtifact('nodes')">节点</button>
            </div>
          </div>

          <SceneGraphPreview
            :scene="sceneForFrame"
            :title="activeSession.title"
            subtitle="后端回放帧视图"
            scene-kind="deployment"
            :show-header="true"
          />

          <div class="replay-page__details">
            <article>
              <small>吞吐量</small>
              <strong>{{ Number(currentFrame.tp || 0).toFixed(3) }}</strong>
            </article>
            <article>
              <small>丢包率</small>
              <strong>{{ Number(currentFrame.loss || 0).toFixed(4) }}</strong>
            </article>
            <article>
              <small>广播覆盖</small>
              <strong>{{ percentageText(currentFrame.broadcastRatio) }}</strong>
            </article>
            <article>
              <small>剩余预算</small>
              <strong>{{ Number(currentFrame.remainingBudget || 0).toFixed(1) }}</strong>
            </article>
            <article>
              <small>绘制节点</small>
              <strong>{{ formatNumber(currentFrame.nodesDrawn) }}/{{ formatNumber(currentFrame.nodesTotal) }}</strong>
            </article>
          </div>

          <div v-if="linkMetricItems.length" class="replay-page__details">
            <article v-for="item in linkMetricItems" :key="item.label">
              <small>{{ item.label }}</small>
              <strong>{{ item.value }}</strong>
            </article>
          </div>

          <div class="replay-page__logs">
            <div class="replay-page__logs-head">
              <strong>实时终端输出</strong>
              <div>
                <button type="button" @click="loadLogs(activeSessionId)">刷新后端日志</button>
                <button type="button" :disabled="!terminalHistoryLines.length" @click="downloadTerminalLog">导出终端输出</button>
                <button type="button" :disabled="!terminalHistoryLines.length" @click="clearTerminalLog">清空</button>
                <button type="button" :disabled="!activeSessionId" @click="downloadArtifact('log')">下载后端日志</button>
              </div>
            </div>
            <pre ref="replayTerminalRef">{{ replayTerminalText }}</pre>
          </div>
        </template>

        <div v-else class="replay-page__empty replay-page__empty--stage">
          选择一条后端回放记录后即可查看逐帧场景。
        </div>
      </section>
    </div>
  </section>
</template>

<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";

import SceneGraphPreview from "./SceneGraphPreview.vue";
import { rescueApiBase } from "../utils/runtimeEndpoints";
import {
  getActiveReplaySessionId,
  setActiveReplaySessionId,
} from "../utils/replaySessions";
import {
  appendSharedTerminalLine,
  clearTerminalOutput,
  exportTerminalOutput,
  terminalHistoryLines,
} from "../utils/terminalOutput";
import {
  buildUserNodeCountMessage,
  userNodeCountLogKey,
} from "../utils/scenarioNodeMetrics";

const API_BASE = rescueApiBase;
const FRAME_SAMPLE_RATIO = 30;

const sessions = ref([]);
const activeSessionId = ref(null);
const activeSessionDetail = ref(null);
const currentFrame = ref(null);
const frameIndex = ref(0);
const logs = ref([]);
const linkMetrics = ref(null);
const replayTerminalRef = ref(null);
const isLoadingSessions = ref(false);
const isLoadingFrame = ref(false);
const isPlaying = ref(false);
const errorMessage = ref("");

const frameCache = new Map();
const syncedReplayLogKeys = new Set();
let lastReplayUserNodeLogKey = "";
let playTimer = null;
let frameRequestToken = 0;

const percentageText = (value) => `${(Math.max(0, Math.min(1, Number(value || 0))) * 100).toFixed(1)}%`;
const formatNumber = (value) => Number(value || 0).toLocaleString("zh-CN");

const sourceLabel = (source) => {
  if (source === "training") return "训练回放";
  if (source === "manual") return "人工导入";
  return "测试回放";
};

const formatTime = (value) => {
  const timestamp = Number(value || 0);
  if (!timestamp) return "--";
  return new Date(timestamp * 1000).toLocaleString("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
};

const fetchJson = async (url) => {
  const response = await fetch(url);
  const rawText = await response.text();
  let payload = null;
  if (rawText) {
    try {
      payload = JSON.parse(rawText);
    } catch {
      payload = null;
    }
  }

  if (!response.ok) {
    const message = payload?.detail || payload?.message || rawText || `请求失败 (${response.status})`;
    throw new Error(message);
  }
  return payload ?? {};
};

const appendReplayTerminalLine = (message, options = {}) => {
  if (!message) return;
  appendSharedTerminalLine(message, {
    level: options.level || "INFO",
    source: options.source || "REPLAY",
    timestamp: options.timestamp,
  });
};

const appendReplayUserNodeCount = (prefix, ...sources) => {
  const key = userNodeCountLogKey(`replay-workbench:${activeSessionId.value || ""}:${prefix}`, ...sources);
  if (key === lastReplayUserNodeLogKey) return;
  lastReplayUserNodeLogKey = key;
  appendReplayTerminalLine(buildUserNodeCountMessage(prefix, ...sources), { level: "SCENE" });
};

const downloadTerminalLog = () => {
  exportTerminalOutput(terminalHistoryLines.value, "rescuenet-replay-terminal.log");
};

const clearTerminalLog = () => {
  clearTerminalOutput();
};

const syncBackendReplayLogLines = (lines = []) => {
  lines.slice(-120).forEach((line) => {
    const text = String(line || "");
    if (!text) return;
    const key = `${activeSessionId.value || "replay"}:${text}`;
    if (syncedReplayLogKeys.has(key)) return;
    syncedReplayLogKeys.add(key);
    appendReplayTerminalLine(text, { level: "BACKEND", source: "BACKEND" });
  });
  if (syncedReplayLogKeys.size > 700) syncedReplayLogKeys.clear();
};

const hashQueryReplayId = () => {
  if (typeof window === "undefined") return null;
  const queryText = window.location.hash.includes("?") ? window.location.hash.split("?").slice(1).join("?") : "";
  return new URLSearchParams(queryText).get("replay_id");
};

const normalizeSession = (session) => {
  const summary = session?.summary || {};
  return {
    ...session,
    id: session?.replay_id || session?.id,
    replayId: session?.replay_id || session?.id,
    title: session?.title || session?.replay_id || "未命名回放",
    source: session?.source || "test",
    createdAt: Number(session?.created_at || session?.createdAt || 0),
    scenarioName: session?.scenario_name || session?.scenarioName || "",
    algorithm: session?.algorithm || "--",
    frameCount: Number(session?.frame_count || session?.frameCount || 0),
    mapWidth: Number(session?.map_width || session?.mapWidth || 5000),
    mapHeight: Number(session?.map_height || session?.mapHeight || 5000),
    nodeCountTotal: Number(session?.node_count_total || session?.nodeCountTotal || 0),
    geoBounds: session?.geo_bounds || session?.geoBounds || null,
    summary: {
      totalReward: Number(summary.total_reward ?? summary.totalReward ?? 0),
      coverageRatio: Number(summary.coverage_ratio ?? summary.coverageRatio ?? 0),
      broadcastRatio: Number(summary.broadcast_ratio ?? summary.broadcastRatio ?? 0),
      stepsTaken: Number(summary.steps_taken ?? summary.stepsTaken ?? 0),
      totalUsers: Number(summary.total_users ?? summary.totalUsers ?? 0),
      initialStations: Number(summary.initial_stations ?? summary.initialStations ?? 0),
      finalStations: Number(summary.final_stations ?? summary.finalStations ?? 0),
    },
  };
};

const normalizeFrame = (frame) => {
  const metrics = frame?.metrics || {};
  return {
    ...frame,
    frameIndex: Number(frame?.frame_index ?? frame?.frameIndex ?? 0),
    mapWidth: Number(frame?.map_width || frame?.mapWidth || 5000),
    mapHeight: Number(frame?.map_height || frame?.mapHeight || 5000),
    geoBounds: frame?.geo_bounds || frame?.geoBounds || null,
    nodes: Array.isArray(frame?.nodes) ? frame.nodes : [],
    links: Array.isArray(frame?.links) ? frame.links : [],
    tp: Number(frame?.tp ?? metrics.avg_user_throughput ?? 0),
    loss: Number(frame?.loss ?? metrics.loss_ratio ?? 0),
    coverageRatio: Number(frame?.coverageRatio ?? frame?.coverage_ratio ?? metrics.coverage_ratio ?? 0),
    broadcastRatio: Number(frame?.broadcastRatio ?? frame?.broadcast_ratio ?? metrics.broadcast_ratio ?? 0),
    remainingBudget: Number(frame?.remainingBudget ?? frame?.remaining_budget ?? metrics.remaining_budget ?? 0),
    reward: Number(frame?.reward || 0),
    nodesTotal: Number(frame?.nodes_total || frame?.node_count_total || metrics.node_count_total || 0),
    nodesDrawn: Number(frame?.nodes_drawn || frame?.nodes?.length || 0),
    userCount: Number(frame?.user_count || metrics.user_count || 0),
    stationCount: Number(frame?.station_count || metrics.station_count || 0),
    connectedUsers: Number(frame?.connected_users || metrics.connected_users || 0),
    broadcastUsers: Number(frame?.broadcast_users || metrics.broadcast_users || 0),
  };
};

const refreshSessions = async () => {
  isLoadingSessions.value = true;
  errorMessage.value = "";
  appendReplayTerminalLine("前端操作：刷新场景回放会话列表。", { level: "ACTION" });
  try {
    const payload = await fetchJson(`${API_BASE}/replay/sessions?limit=50`);
    sessions.value = (payload?.sessions || []).map(normalizeSession);
    appendReplayTerminalLine(`后端响应：场景回放会话 ${sessions.value.length} 条。`, {
      level: "BACKEND",
      source: "BACKEND",
    });
    const preferredId = hashQueryReplayId() || getActiveReplaySessionId();
    const selected =
      sessions.value.find((session) => session.id === preferredId) ||
      sessions.value[0] ||
      null;
    if (selected) {
      await selectSession(selected.id);
    } else {
      activeSessionId.value = null;
      currentFrame.value = null;
      logs.value = [];
    }
  } catch (error) {
    errorMessage.value = `后端回放会话读取失败：${error?.message || error}`;
    appendReplayTerminalLine(errorMessage.value, { level: "ERROR", source: "BACKEND" });
  } finally {
    isLoadingSessions.value = false;
  }
};

const selectSession = async (id) => {
  if (!id) return;
  stopPlayback();
  activeSessionId.value = id;
  setActiveReplaySessionId(id);
  frameIndex.value = 0;
  currentFrame.value = null;
  linkMetrics.value = null;
  logs.value = [];
  frameCache.clear();
  appendReplayTerminalLine(`前端操作：选择场景回放 replay_id=${id}。`, { level: "ACTION" });
  await Promise.all([
    loadSessionDetail(id),
    loadFrame(id, 0),
    loadLogs(id),
  ]);
};

const loadSessionDetail = async (id) => {
  try {
    const detail = normalizeSession(await fetchJson(`${API_BASE}/replay/sessions/${encodeURIComponent(id)}`));
    activeSessionDetail.value = detail;
    const existingIndex = sessions.value.findIndex((session) => session.id === id);
    if (existingIndex >= 0) {
      sessions.value.splice(existingIndex, 1, detail);
    }
    appendReplayTerminalLine(`后端响应：回放元数据已加载，帧数=${detail.frameCount} 节点=${detail.nodeCountTotal}。`, {
      level: "BACKEND",
      source: "BACKEND",
    });
    appendReplayUserNodeCount(`场景回放接入灾害场景：${detail.title || id}`, detail);
  } catch (error) {
    errorMessage.value = `回放元数据读取失败：${error?.message || error}`;
    appendReplayTerminalLine(errorMessage.value, { level: "ERROR", source: "BACKEND" });
  }
};

const loadFrame = async (id, index) => {
  if (!id) return;
  const numericIndex = Math.max(0, Number(index || 0));
  const cacheKey = `${id}:${numericIndex}`;
  if (frameCache.has(cacheKey)) {
    currentFrame.value = frameCache.get(cacheKey);
    linkMetrics.value = null;
    appendReplayUserNodeCount(`场景回放帧数据已接入：${activeSession.value?.title || id}`, activeSession.value, currentFrame.value);
    void loadLinkMetrics(id, numericIndex);
    return;
  }

  const token = (frameRequestToken += 1);
  isLoadingFrame.value = true;
  try {
    const payload = await fetchJson(
      `${API_BASE}/replay/sessions/${encodeURIComponent(id)}/frames/${numericIndex}?sample_ratio=${FRAME_SAMPLE_RATIO}&include_links=true`
    );
    const frame = normalizeFrame(payload);
    frameCache.set(cacheKey, frame);
    if (token === frameRequestToken && activeSessionId.value === id) {
      currentFrame.value = frame;
      linkMetrics.value = null;
      appendReplayUserNodeCount(`场景回放帧数据已接入：${activeSession.value?.title || id}`, activeSession.value, frame);
    }
    if (numericIndex % 5 === 0) {
      appendReplayTerminalLine(
        `后端响应：回放帧 ${numericIndex}/${maxFrameIndex.value} 已加载，覆盖率=${percentageText(frame.coverageRatio)}。`,
        { level: "BACKEND", source: "BACKEND" }
      );
    }
    void loadLinkMetrics(id, numericIndex);
  } catch (error) {
    if (token === frameRequestToken) {
      errorMessage.value = `回放帧读取失败：${error?.message || error}`;
      appendReplayTerminalLine(errorMessage.value, { level: "ERROR", source: "BACKEND" });
    }
  } finally {
    if (token === frameRequestToken) {
      isLoadingFrame.value = false;
    }
  }
};

const loadLogs = async (id) => {
  if (!id) return;
  try {
    const payload = await fetchJson(`${API_BASE}/replay/sessions/${encodeURIComponent(id)}/logs?limit=160`);
    logs.value = Array.isArray(payload?.lines) ? payload.lines : [];
    appendReplayTerminalLine(`后端响应：回放日志已加载 ${logs.value.length} 行。`, {
      level: "BACKEND",
      source: "BACKEND",
    });
    syncBackendReplayLogLines(logs.value);
  } catch (error) {
    errorMessage.value = `回放日志读取失败：${error?.message || error}`;
    appendReplayTerminalLine(errorMessage.value, { level: "ERROR", source: "BACKEND" });
  }
};

const loadLinkMetrics = async (id, index) => {
  if (!id) return;
  try {
    linkMetrics.value = await fetchJson(
      `${API_BASE}/replay/sessions/${encodeURIComponent(id)}/link-metrics?frame_index=${Math.max(0, Number(index || 0))}`
    );
  } catch {
    linkMetrics.value = null;
  }
};

const togglePlayback = () => {
  if (isPlaying.value) {
    stopPlayback();
    appendReplayTerminalLine("前端操作：暂停场景回放。", { level: "ACTION" });
  } else {
    startPlayback();
  }
};

const startPlayback = () => {
  if (!activeSession.value || maxFrameIndex.value <= 0) return;
  stopPlayback();
  isPlaying.value = true;
  appendReplayTerminalLine(`前端操作：启动场景回放 replay_id=${activeSessionId.value}。`, { level: "ACTION" });
  playTimer = window.setInterval(() => {
    if (frameIndex.value >= maxFrameIndex.value) {
      stopPlayback();
      return;
    }
    frameIndex.value += 1;
  }, 900);
};

const stopPlayback = () => {
  isPlaying.value = false;
  if (playTimer) {
    window.clearInterval(playTimer);
    playTimer = null;
  }
};

const resetPlayback = () => {
  stopPlayback();
  frameIndex.value = 0;
  appendReplayTerminalLine("前端操作：复位场景回放。", { level: "ACTION" });
};

const downloadArtifact = (type) => {
  if (!activeSessionId.value || typeof window === "undefined") return;
  appendReplayTerminalLine(`前端操作：下载场景回放${type}文件 replay_id=${activeSessionId.value}。`, { level: "ACTION" });
  window.open(
    `${API_BASE}/replay/sessions/${encodeURIComponent(activeSessionId.value)}/download?type=${encodeURIComponent(type)}`,
    "_blank"
  );
  appendReplayTerminalLine("后端响应：已打开场景回放文件下载地址。", { level: "BACKEND", source: "BACKEND" });
};

const activeSession = computed(() => {
  if (activeSessionDetail.value?.id === activeSessionId.value) {
    return activeSessionDetail.value;
  }
  return sessions.value.find((session) => session.id === activeSessionId.value) || null;
});

const maxFrameIndex = computed(() => Math.max(0, Number(activeSession.value?.frameCount || 0) - 1));

const sceneForFrame = computed(() => {
  if (!activeSession.value || !currentFrame.value) return null;
  return {
    map_width: Number(currentFrame.value.mapWidth || activeSession.value.mapWidth || 5000),
    map_height: Number(currentFrame.value.mapHeight || activeSession.value.mapHeight || 5000),
    geo_bounds: currentFrame.value.geoBounds || activeSession.value.geoBounds || null,
    nodes: (currentFrame.value.nodes || []).map((node) => ({
      id: node.id,
      type: normalizeNodeType(node),
      x: Number(node.x || 0),
      y: Number(node.y || 0),
      lat: node.lat,
      lon: node.lon,
      connected: Boolean(node.connected),
      broadcast_served: Boolean(node.broadcast_served ?? node.broadcastServed),
      base_station: node.base_station,
      label: node.label || node.device_label || node.base_station,
      node_role: node.node_role,
      device_type: node.device_type,
      device_label: node.device_label,
    })),
  };
});

const normalizeNodeType = (node) => {
  if (typeof node.type === "string") return node.type;
  if (Number(node.type) === 0) return "USER";
  if (node.kind === "deployment") return "SMALL_CELL";
  return "MACRO_ENB";
};

const summaryItems = computed(() => {
  if (!activeSession.value) return [];
  return [
    { label: "算法", value: String(activeSession.value.algorithm || "--").toUpperCase() },
    { label: "总奖励", value: Number(activeSession.value.summary?.totalReward || 0).toFixed(2) },
    { label: "终态覆盖", value: percentageText(activeSession.value.summary?.coverageRatio) },
    { label: "步数", value: String(activeSession.value.summary?.stepsTaken || 0) },
  ];
});

const linkMetricItems = computed(() => {
  const groups = Array.isArray(linkMetrics.value?.groups) ? linkMetrics.value.groups : [];
  return groups.slice(0, 3).map((group) => ({
    label: group.label || group.link_type,
    value: `${formatNumber(group.active_links)} 条 / ${Number(group.avg_throughput_mbps || 0).toFixed(2)}Mbps`,
  }));
});

const replayTerminalText = computed(() =>
  terminalHistoryLines.value.length ? terminalHistoryLines.value.slice(-260).join("\n") : "暂无终端输出。"
);

const scrollReplayTerminalToBottom = async () => {
  await nextTick();
  const viewport = replayTerminalRef.value;
  if (!viewport) return;
  viewport.scrollTo({
    top: viewport.scrollHeight,
    behavior: "auto",
  });
};

watch(frameIndex, (index) => {
  if (!activeSessionId.value) return;
  void loadFrame(activeSessionId.value, index);
});

watch(
  () => terminalHistoryLines.value.length,
  () => {
    void scrollReplayTerminalToBottom();
  },
  { immediate: true, flush: "post" }
);

onMounted(() => {
  void refreshSessions();
  void scrollReplayTerminalToBottom();
});
onBeforeUnmount(stopPlayback);
</script>

<style scoped>
.replay-page {
  display: flex;
  flex-direction: column;
  gap: 18px;
}

.replay-page__header {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: end;
}

.replay-page__header h2,
.replay-page__header p {
  margin: 0;
}

.replay-page__header p {
  margin-top: 6px;
  color: #64748b;
}

.replay-page__header button,
.replay-page__actions button,
.replay-page__logs-head button {
  padding: 10px 14px;
  border-radius: 999px;
  border: 1px solid rgba(14, 165, 233, 0.24);
  background: rgba(224, 242, 254, 0.86);
  color: #075985;
}

.replay-page__header button:disabled,
.replay-page__actions button:disabled {
  cursor: not-allowed;
  opacity: 0.58;
}

.replay-page__status {
  padding: 12px 14px;
  border-radius: 16px;
  border: 1px solid rgba(248, 113, 113, 0.28);
  background: rgba(254, 226, 226, 0.74);
  color: #991b1b;
}

.replay-page__layout {
  display: grid;
  grid-template-columns: 320px minmax(0, 1fr);
  gap: 18px;
}

.replay-page__list,
.replay-page__stage {
  padding: 18px;
  border-radius: 20px;
  border: 1px solid rgba(148, 163, 184, 0.2);
  background: rgba(255, 255, 255, 0.92);
  box-shadow: 0 14px 28px rgba(15, 23, 42, 0.05);
}

.replay-page__list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.replay-page__list-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  color: #475569;
}

.replay-session {
  display: flex;
  flex-direction: column;
  gap: 4px;
  text-align: left;
  padding: 12px 14px;
  border-radius: 16px;
  border: 1px solid rgba(148, 163, 184, 0.18);
  background: rgba(248, 250, 252, 0.9);
  color: #0f172a;
}

.replay-session strong {
  font-size: 14px;
}

.replay-session small,
.replay-session span {
  color: #64748b;
}

.replay-session--active {
  border-color: rgba(14, 165, 233, 0.34);
  background: rgba(224, 242, 254, 0.9);
  color: #075985;
}

.replay-page__stage {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.replay-page__meta,
.replay-page__details {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 12px;
}

.replay-page__meta article,
.replay-page__details article,
.replay-page__empty,
.replay-page__logs {
  padding: 14px 16px;
  border-radius: 16px;
  border: 1px solid rgba(148, 163, 184, 0.18);
  background: rgba(248, 250, 252, 0.9);
}

.replay-page__meta small,
.replay-page__details small {
  display: block;
  color: #64748b;
  margin-bottom: 6px;
}

.replay-page__meta strong,
.replay-page__details strong {
  color: #075985;
  font-size: 20px;
}

.replay-page__controls {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: center;
}

.replay-page__controls label {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.replay-page__controls span,
.replay-page__frame-text {
  color: #64748b;
  font-size: 13px;
}

.replay-page__frame-text {
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 126px;
}

.replay-page__actions {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 8px;
}

.replay-page__empty {
  color: #64748b;
}

.replay-page__empty--stage {
  min-height: 420px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.replay-page__logs {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.replay-page__logs-head {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: center;
  color: #475569;
}

.replay-page__logs-head > div {
  display: flex;
  justify-content: flex-end;
  flex-wrap: wrap;
  gap: 8px;
}

.replay-page__logs pre {
  max-height: 220px;
  margin: 0;
  overflow: auto;
  white-space: pre-wrap;
  color: #334155;
  font-size: 12px;
  line-height: 1.65;
}

@media (max-width: 960px) {
  .replay-page__layout {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 720px) {
  .replay-page__header,
  .replay-page__controls {
    flex-direction: column;
    align-items: stretch;
  }

  .replay-page__actions {
    justify-content: flex-start;
  }
}
</style>
