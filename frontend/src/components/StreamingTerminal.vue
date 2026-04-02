<template>
  <div class="terminal">
    <div class="terminal__header">
      <div>
        <h3>{{ title }}</h3>
        <p>{{ subtitle }}</p>
      </div>
      <span class="terminal__status" :class="`terminal__status--${normalizedStatus}`">
        {{ statusLabel }}
      </span>
    </div>

    <div ref="viewportRef" class="terminal__viewport" role="log" aria-live="polite">
      <div v-if="!lines.length" class="terminal__placeholder">
        等待测试输出...
      </div>
      <div v-for="(line, index) in lines" :key="`${index}-${line}`" class="terminal__line">
        {{ line }}
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, nextTick, ref, watch } from "vue";

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
});

const viewportRef = ref(null);

const normalizedStatus = computed(() => {
  const allowed = new Set(["idle", "importing", "running", "completed", "failed"]);
  return allowed.has(props.status) ? props.status : "idle";
});

const statusLabel = computed(() => {
  const labelMap = {
    idle: "空闲",
    importing: "准备中",
    running: "运行中",
    completed: "已完成",
    failed: "失败",
  };
  return labelMap[normalizedStatus.value];
});

watch(
  () => props.lines.length,
  async () => {
    await nextTick();
    viewportRef.value?.scrollTo({
      top: viewportRef.value.scrollHeight,
      behavior: "smooth",
    });
  }
);
</script>

<style scoped>
.terminal {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.terminal__header {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: flex-start;
}

.terminal__header h3 {
  margin: 0;
  font-size: 18px;
}

.terminal__header p {
  margin: 6px 0 0;
  color: #94a3b8;
}

.terminal__status {
  padding: 8px 12px;
  border-radius: 999px;
  border: 1px solid rgba(148, 163, 184, 0.2);
  background: rgba(15, 23, 42, 0.5);
  font-size: 12px;
  color: #dbeafe;
}

.terminal__status--running {
  border-color: rgba(56, 189, 248, 0.35);
  color: #7dd3fc;
}

.terminal__status--completed {
  border-color: rgba(74, 222, 128, 0.3);
  color: #86efac;
}

.terminal__status--failed {
  border-color: rgba(248, 113, 113, 0.3);
  color: #fca5a5;
}

.terminal__viewport {
  min-height: 280px;
  max-height: 420px;
  overflow: auto;
  padding: 16px;
  border-radius: 18px;
  border: 1px solid rgba(15, 23, 42, 0.6);
  background:
    linear-gradient(180deg, rgba(2, 6, 23, 0.98), rgba(15, 23, 42, 0.92)),
    radial-gradient(circle at top left, rgba(34, 211, 238, 0.08), transparent 35%);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
  font-family: "SFMono-Regular", "Menlo", "Consolas", monospace;
  font-size: 13px;
  line-height: 1.7;
  color: #dbeafe;
}

.terminal__placeholder {
  color: #64748b;
}

.terminal__line {
  white-space: pre-wrap;
  word-break: break-word;
}

@media (max-width: 720px) {
  .terminal__header {
    flex-direction: column;
  }

  .terminal__viewport {
    min-height: 220px;
  }
}
</style>
