<template>
  <section class="link-page">
    <div v-if="errorMessage" class="link-page__error">{{ errorMessage }}</div>

    <section class="link-page__status-grid">
      <article>
        <small>ns-3 状态</small>
        <strong>{{ ns3Status?.running ? "运行中" : "空闲" }}</strong>
        <p>{{ ns3Status?.available ? "原生界面可访问" : "原生界面不可访问" }}</p>
      </article>
      <article>
        <small>实验记录</small>
        <strong>{{ Number(ns3Status?.experiment_count || 0).toLocaleString("zh-CN") }}</strong>
        <p>数据库中当前可回放实验数量。</p>
      </article>
      <article>
        <small>Mahimahi</small>
        <strong>{{ mahimahiStatus?.mahimahi_available ? "已接入" : "未接入" }}</strong>
        <p>链路容量模拟接口状态。</p>
      </article>
      <article>
        <small>最新完成</small>
        <strong>{{ latestFinishedText }}</strong>
        <p>最近一次 ns-3 任务完成时间。</p>
      </article>
    </section>

    <section class="link-page__control-panel">
      <div class="link-page__control-copy">
        <span class="link-page__eyebrow">ns-3 Control</span>
        <h3>链路实验控制区</h3>
        <p>这里直接接入 `/api/ns3/status`、`/api/ns3/run`、`/api/import` 和 `/api/experiments`。</p>
      </div>

      <div class="link-page__control-actions">
        <button type="button" class="is-primary" :disabled="busy" @click="runNs3">
          {{ ns3Status?.running ? "ns-3 运行中..." : "启动 ns-3 仿真" }}
        </button>
        <button type="button" :disabled="busy" @click="importTrace">导入最新 trace</button>
        <button type="button" :disabled="busy" @click="refreshStatus">刷新状态</button>
      </div>
    </section>

    <section class="link-page__grid">
      <article class="link-page__card">
        <div class="link-page__card-header">
          <div>
            <h3>Mahimahi 链路容量回放</h3>
            <p>基于真实 trace 文件列表和后端容量分析接口。</p>
          </div>
        </div>
        <MahimahiSimulator />
      </article>

      <article class="link-page__card">
        <div class="link-page__card-header">
          <div>
            <h3>ns-3 实验记录</h3>
            <p>展示数据库中的实验摘要，可配合下方原生界面回放。</p>
          </div>
          <span>{{ experiments.length }} 条</span>
        </div>

        <div class="link-page__table-wrap">
          <table class="link-page__table">
            <thead>
              <tr>
                <th>实验名</th>
                <th>时长</th>
                <th>节点数</th>
                <th>帧数</th>
                <th>时间</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="exp in experiments.slice(0, 8)" :key="exp.id">
                <td>{{ exp.name }}</td>
                <td>{{ Number(exp.duration || 0).toFixed(1) }}s</td>
                <td>{{ exp.total_nodes }}</td>
                <td>{{ exp.frames }}</td>
                <td>{{ exp.date }}</td>
              </tr>
              <tr v-if="!experiments.length">
                <td colspan="5" class="link-page__table-empty">暂无实验记录</td>
              </tr>
            </tbody>
          </table>
        </div>
      </article>
    </section>

    <section class="link-page__native-card">
      <div class="link-page__card-header">
        <div>
          <h3>ns-3 原生回放界面</h3>
          <p>保留原生交互界面，便于继续查看底层实验细节。</p>
        </div>
        <span>{{ ns3Status?.native_path || "/ns3-native/index.html" }}</span>
      </div>
      <Ns3ReplayPanel />
    </section>
  </section>
</template>

<script setup>
import { computed, onMounted, ref } from "vue";
import axios from "axios";

import MahimahiSimulator from "./MahimahiSimulator.vue";
import Ns3ReplayPanel from "./Ns3ReplayPanel.vue";
import { rescueApiBase } from "../utils/runtimeEndpoints";

const API_BASE = rescueApiBase;

const ns3Status = ref(null);
const experiments = ref([]);
const mahimahiStatus = ref(null);
const busy = ref(false);
const errorMessage = ref("");

const latestFinishedText = computed(() => {
  if (!ns3Status.value?.last_finished_at) return "暂无";
  return new Date(Number(ns3Status.value.last_finished_at) * 1000).toLocaleString("zh-CN", {
    hour12: false,
  });
});

const refreshStatus = async () => {
  errorMessage.value = "";
  try {
    const [ns3Resp, expResp, mahimahiResp] = await Promise.all([
      axios.get(`${API_BASE}/ns3/status`),
      axios.get(`${API_BASE}/experiments`),
      axios.get(`${API_BASE}/mahimahi/status`).catch(() => ({ data: null })),
    ]);
    ns3Status.value = ns3Resp.data;
    experiments.value = Array.isArray(expResp.data) ? expResp.data : [];
    mahimahiStatus.value = mahimahiResp.data;
  } catch (error) {
    console.error("Failed to refresh link simulation status", error);
    errorMessage.value = "链路仿真状态加载失败，请检查后端服务。";
  }
};

const withBusy = async (runner) => {
  busy.value = true;
  errorMessage.value = "";
  try {
    await runner();
    await refreshStatus();
  } catch (error) {
    console.error("Link simulation action failed", error);
    const detail = error?.response?.data?.detail || error?.message || "操作失败";
    errorMessage.value = typeof detail === "string" ? detail : JSON.stringify(detail);
  } finally {
    busy.value = false;
  }
};

const runNs3 = async () => {
  await withBusy(async () => {
    await axios.post(`${API_BASE}/ns3/run`);
  });
};

const importTrace = async () => {
  await withBusy(async () => {
    await axios.post(`${API_BASE}/import`);
  });
};

onMounted(() => {
  void refreshStatus();
});
</script>

<style scoped>
.link-page {
  display: flex;
  flex-direction: column;
  gap: 18px;
}

.link-page__error,
.link-page__status-grid article,
.link-page__control-panel,
.link-page__card,
.link-page__native-card {
  border: 1px solid rgba(148, 163, 184, 0.18);
  border-radius: 24px;
  background: rgba(255, 255, 255, 0.9);
  box-shadow: 0 18px 40px rgba(59, 130, 246, 0.08);
}

.link-page__error {
  padding: 14px 16px;
  color: #b91c1c;
  background: rgba(254, 242, 242, 0.92);
}

.link-page__status-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 14px;
}

.link-page__status-grid article {
  padding: 18px;
}

.link-page__status-grid small,
.link-page__card-header span,
.link-page__eyebrow {
  color: #728aac;
  font-size: 12px;
  letter-spacing: 0.08em;
}

.link-page__status-grid strong {
  display: block;
  margin-top: 8px;
  color: #17315d;
  font-size: 24px;
}

.link-page__status-grid p,
.link-page__control-copy p {
  margin: 8px 0 0;
  color: #6881a7;
  line-height: 1.7;
}

.link-page__control-panel {
  display: flex;
  justify-content: space-between;
  gap: 18px;
  align-items: center;
  padding: 20px 22px;
}

.link-page__eyebrow {
  display: inline-flex;
  padding: 7px 12px;
  border-radius: 999px;
  background: rgba(37, 99, 235, 0.08);
  color: #2563eb;
}

.link-page__control-copy h3,
.link-page__card-header h3 {
  margin: 10px 0 0;
  color: #17315d;
}

.link-page__control-actions {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 10px;
}

.link-page__control-actions button {
  height: 42px;
  padding: 0 16px;
  border-radius: 12px;
  border: 0;
  background: rgba(37, 99, 235, 0.1);
  color: #2563eb;
  font-weight: 700;
}

.link-page__control-actions .is-primary {
  background: linear-gradient(135deg, #38bdf8, #2563eb);
  color: #ffffff;
}

.link-page__grid {
  display: grid;
  grid-template-columns: minmax(0, 1.06fr) minmax(360px, 0.94fr);
  gap: 18px;
}

.link-page__card,
.link-page__native-card {
  padding: 18px;
}

.link-page__card-header {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: flex-start;
  margin-bottom: 14px;
}

.link-page__card-header p {
  margin: 6px 0 0;
  color: #6881a7;
}

.link-page__table-wrap {
  overflow: auto;
}

.link-page__table {
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
}

.link-page__table th,
.link-page__table td {
  padding: 12px 10px;
  border-bottom: 1px solid rgba(226, 232, 240, 0.84);
  text-align: left;
  white-space: nowrap;
}

.link-page__table thead th {
  color: #5d7699;
  font-weight: 600;
}

.link-page__table tbody td {
  color: #17315d;
}

.link-page__table-empty {
  text-align: center !important;
  color: #6881a7 !important;
}

@media (max-width: 1200px) {
  .link-page__status-grid,
  .link-page__grid {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 820px) {
  .link-page__control-panel,
  .link-page__card-header {
    flex-direction: column;
    align-items: flex-start;
  }

  .link-page__control-actions {
    justify-content: flex-start;
  }
}
</style>
