<template>
  <section class="replay-page">
    <div class="replay-page__header">
      <div>
        <h2>场景回放工作台</h2>
        <p>读取浏览器本地回放记录，逐帧查看训练或测试生成的恢复过程。</p>
      </div>
      <button type="button" @click="refreshSessions">刷新列表</button>
    </div>

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
          <small>{{ session.source === "training" ? "训练回放" : "测试回放" }}</small>
          <span>覆盖 {{ percentageText(session.summary?.coverageRatio) }}</span>
        </button>

        <div v-if="!sessions.length" class="replay-page__empty">
          当前没有回放记录。请先在训练中心或策略测试中心生成回放。
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
                :max="Math.max(0, activeSession.frames.length - 1)"
                step="1"
              />
            </label>
            <div class="replay-page__frame-text">
              Frame {{ frameIndex + 1 }} / {{ activeSession.frames.length }}
            </div>
          </div>

          <SceneGraphPreview
            :scene="sceneForFrame"
            :title="activeSession.title"
            subtitle="回放帧视图"
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
          </div>
        </template>

        <div v-else class="replay-page__empty replay-page__empty--stage">
          选择一条回放记录后即可查看逐帧场景。
        </div>
      </section>
    </div>
  </section>
</template>

<script setup>
import { computed, ref } from "vue";

import SceneGraphPreview from "./SceneGraphPreview.vue";
import {
  getActiveReplaySessionId,
  listReplaySessions,
  setActiveReplaySessionId,
} from "../utils/replaySessions";

const sessions = ref([]);
const activeSessionId = ref(null);
const frameIndex = ref(0);

const percentageText = (value) => `${(Math.max(0, Math.min(1, Number(value || 0))) * 100).toFixed(1)}%`;

const refreshSessions = () => {
  sessions.value = listReplaySessions();
  const storedId = getActiveReplaySessionId();
  const fallbackId = sessions.value[0]?.id || null;
  activeSessionId.value =
    sessions.value.find((session) => session.id === storedId)?.id || fallbackId;
  frameIndex.value = 0;
};

const selectSession = (id) => {
  activeSessionId.value = id;
  frameIndex.value = 0;
  setActiveReplaySessionId(id);
};

const activeSession = computed(
  () => sessions.value.find((session) => session.id === activeSessionId.value) || null
);

const currentFrame = computed(() => activeSession.value?.frames?.[frameIndex.value] || null);

const sceneForFrame = computed(() => {
  if (!activeSession.value || !currentFrame.value) return null;
  return {
    map_width: Number(activeSession.value.mapWidth || 5000),
    map_height: Number(activeSession.value.mapHeight || 5000),
    nodes: (currentFrame.value.nodes || []).map((node) => ({
      id: node.id,
      type: Number(node.type) === 0 ? "USER" : node.kind === "deployment" ? "SMALL_CELL" : "MACRO_ENB",
      x: Number(node.x || 0),
      y: Number(node.y || 0),
      connected: Boolean(node.online),
      broadcast_served: Boolean(node.broadcastServed),
    })),
  };
});

const summaryItems = computed(() => {
  if (!activeSession.value) return [];
  return [
    { label: "算法", value: String(activeSession.value.algorithm || "--").toUpperCase() },
    { label: "总奖励", value: Number(activeSession.value.summary?.totalReward || 0).toFixed(2) },
    { label: "终态覆盖", value: percentageText(activeSession.value.summary?.coverageRatio) },
    { label: "步数", value: String(activeSession.value.summary?.stepsTaken || 0) },
  ];
});

refreshSessions();
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

.replay-page__header button {
  padding: 10px 14px;
  border-radius: 999px;
  border: 1px solid rgba(14, 165, 233, 0.24);
  background: rgba(224, 242, 254, 0.86);
  color: #075985;
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
.replay-page__empty {
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

.replay-page__empty {
  color: #64748b;
}

.replay-page__empty--stage {
  min-height: 420px;
  display: flex;
  align-items: center;
  justify-content: center;
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
}
</style>
