<template>
  <div class="terminal">
    <div class="terminal__header">
      <div>
        <h3>{{ title }}</h3>
        <p>{{ subtitle }}</p>
      </div>
      <div class="terminal__meta">
        <span>{{ lines.length }} lines</span>
        <span class="terminal__status" :class="`terminal__status--${normalizedStatus}`">
          {{ statusLabel }}
        </span>
        <button v-if="exportable" type="button" class="terminal__export" @click="emit('export')">
          导出
        </button>
        <button v-if="clearable" type="button" class="terminal__clear" @click="emit('clear')">
          清空
        </button>
      </div>
    </div>

    <div ref="viewportRef" class="terminal__viewport" role="log" aria-live="polite">
      <div v-if="!lines.length" class="terminal__placeholder">
        {{ placeholder }}
      </div>
      <div v-for="(line, index) in lines" :key="`${index}-${line}`" class="terminal__line">
        {{ line }}
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, nextTick, onMounted, ref, watch } from "vue";

const props = defineProps({
  title: {
    type: String,
    default: "实时终端",
  },
  subtitle: {
    type: String,
    default: "显示测试阶段的状态、动作和恢复结果。",
  },
  lines: {
    type: Array,
    default: () => [],
  },
  status: {
    type: String,
    default: "idle",
  },
  placeholder: {
    type: String,
    default: "等待实时输出...",
  },
  exportable: {
    type: Boolean,
    default: false,
  },
  clearable: {
    type: Boolean,
    default: false,
  },
});

const emit = defineEmits(["export", "clear"]);

const viewportRef = ref(null);

const normalizedStatus = computed(() => {
  const raw = String(props.status || "idle").toLowerCase();
  const map = {
    idle: "idle",
    initializing: "loading",
    starting: "loading",
    loading: "loading",
    importing: "importing",
    running: "running",
    completed: "completed",
    success: "completed",
    stopped: "stopped",
    disconnected: "disconnected",
    failed: "failed",
    error: "failed",
  };
  return map[raw] || "idle";
});

const statusLabel = computed(() => {
  const labelMap = {
    idle: "空闲",
    loading: "加载中",
    importing: "准备中",
    running: "运行中",
    completed: "已完成",
    stopped: "已停止",
    disconnected: "连接中断",
    failed: "失败",
  };
  return labelMap[normalizedStatus.value];
});

const scrollToBottom = async (behavior = "auto") => {
  await nextTick();
  const viewport = viewportRef.value;
  if (!viewport) return;
  viewport.scrollTo({
    top: viewport.scrollHeight,
    behavior,
  });
};

watch(
  () => props.lines.length,
  (length, previousLength) => {
    const delta = previousLength == null ? 0 : Math.abs(Number(length || 0) - Number(previousLength || 0));
    void scrollToBottom(previousLength == null || delta > 3 ? "auto" : "smooth");
  },
  { immediate: true, flush: "post" }
);

onMounted(() => {
  void scrollToBottom("auto");
});
</script>

<style scoped>
.terminal {
  display: flex;
  flex-direction: column;
  gap: 12px;
  min-width: 0;
  box-sizing: border-box;
  border: 1px solid rgba(203, 213, 225, 0.85);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.96);
  padding: 14px;
  box-shadow: 3px 3px 20px rgba(233, 233, 233, 0.78);
}

.terminal__header {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: flex-start;
  min-width: 0;
}

.terminal__header h3 {
  margin: 0;
  color: #1f2d3d;
  font-size: 16px;
  font-weight: 700;
}

.terminal__header p {
  margin: 4px 0 0;
  color: #64748b;
  font-size: 12px;
  line-height: 18px;
}

.terminal__meta {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  flex: 0 0 auto;
  color: #64748b;
  font-family: "Cascadia Code", "SFMono-Regular", Consolas, monospace;
  font-size: 12px;
}

.terminal__status {
  min-width: 76px;
  padding: 5px 10px;
  border-radius: 999px;
  border: 1px solid rgba(148, 163, 184, 0.25);
  background: rgba(100, 116, 139, 0.1);
  font-size: 12px;
  font-family: "Microsoft YaHei", "PingFang SC", sans-serif;
  color: #64748b;
  text-align: center;
}

.terminal__status--loading,
.terminal__status--importing {
  border-color: rgba(14, 165, 233, 0.28);
  background: rgba(14, 165, 233, 0.1);
  color: #0369a1;
}

.terminal__status--running {
  border-color: rgba(0, 121, 254, 0.28);
  background: rgba(0, 121, 254, 0.12);
  color: #0079fe;
}

.terminal__status--completed {
  border-color: rgba(34, 197, 94, 0.28);
  background: rgba(34, 197, 94, 0.12);
  color: #15803d;
}

.terminal__status--stopped,
.terminal__status--disconnected {
  border-color: rgba(245, 158, 11, 0.28);
  background: rgba(245, 158, 11, 0.12);
  color: #92400e;
}

.terminal__status--failed {
  border-color: rgba(239, 68, 68, 0.28);
  background: rgba(239, 68, 68, 0.12);
  color: #b91c1c;
}

.terminal__export {
  height: 28px;
  padding: 0 10px;
  border: 1px solid rgba(37, 99, 235, 0.18);
  border-radius: 6px;
  background: rgba(37, 99, 235, 0.08);
  color: #1d4ed8;
  font-size: 12px;
  font-family: "Microsoft YaHei", "PingFang SC", sans-serif;
  cursor: pointer;
  white-space: nowrap;
}

.terminal__export:hover {
  border-color: rgba(37, 99, 235, 0.32);
  background: rgba(37, 99, 235, 0.14);
}

.terminal__clear {
  height: 28px;
  padding: 0 10px;
  border: 1px solid rgba(239, 68, 68, 0.18);
  border-radius: 6px;
  background: rgba(239, 68, 68, 0.08);
  color: #b91c1c;
  font-size: 12px;
  font-family: "Microsoft YaHei", "PingFang SC", sans-serif;
  cursor: pointer;
  white-space: nowrap;
}

.terminal__clear:hover {
  border-color: rgba(239, 68, 68, 0.32);
  background: rgba(239, 68, 68, 0.14);
}

.terminal__viewport {
  height: 360px;
  overflow: auto;
  padding: 14px 16px;
  border-radius: 8px;
  border: 1px solid rgba(15, 23, 42, 0.78);
  background:
    linear-gradient(180deg, rgba(2, 6, 23, 0.98), rgba(15, 23, 42, 0.94)),
    radial-gradient(circle at top left, rgba(56, 189, 248, 0.12), transparent 34%);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05);
  font-family: "Cascadia Code", "SFMono-Regular", "Menlo", "Consolas", monospace;
  font-size: 12px;
  line-height: 1.72;
  color: #dbeafe;
  text-align: left;
  scrollbar-color: rgba(148, 163, 184, 0.5) rgba(15, 23, 42, 0.55);
  scrollbar-width: thin;
}

.terminal__placeholder {
  color: #64748b;
}

.terminal__line {
  white-space: pre-wrap;
  word-break: break-word;
  min-height: 20px;
}

@media (max-width: 720px) {
  .terminal__header {
    flex-direction: column;
  }

  .terminal__meta {
    flex-wrap: wrap;
  }

  .terminal__viewport {
    min-height: 220px;
  }
}
</style>
