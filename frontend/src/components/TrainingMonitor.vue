<template>
  <div class="monitor">
    <div class="monitor__status">
      <div>
        <p class="monitor__label">当前状态</p>
        <p class="monitor__value">{{ status }}</p>
      </div>
      <div>
        <p class="monitor__label">最近更新</p>
        <p class="monitor__value">{{ latestUpdate }}</p>
      </div>
    </div>
    <div class="monitor__events">
      <p class="monitor__label">实时事件</p>
      <div class="monitor__event-list">
        <div v-for="(event, idx) in events" :key="idx" class="event">
          <p class="event__type">{{ event.type }}</p>
          <pre>{{ formatPayload(event.payload) }}</pre>
        </div>
        <p v-if="!events.length" class="monitor__placeholder">暂无事件，请启动训练。</p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from "vue";

const props = defineProps({
  events: {
    type: Array,
    default: () => [],
  },
  status: {
    type: String,
    default: "Idle",
  },
});

const latestUpdate = computed(() => {
  if (!props.events.length) {
    return "--";
  }
  const timestamp = props.events[props.events.length - 1].timestamp;
  return new Date(timestamp * 1000).toLocaleTimeString();
});

const formatPayload = (payload) => {
  if (!payload) {
    return "";
  }
  try {
    return JSON.stringify(payload, null, 2);
  } catch (err) {
    return String(payload);
  }
};
</script>

<style scoped>
.monitor {
  border: 1px solid rgba(148, 163, 184, 0.3);
  border-radius: 12px;
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 16px;
  background: rgba(15, 23, 42, 0.4);
}

.monitor__status {
  display: flex;
  justify-content: space-between;
  border-bottom: 1px solid rgba(148, 163, 184, 0.2);
  padding-bottom: 12px;
}

.monitor__label {
  margin: 0;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #94a3b8;
}

.monitor__value {
  margin: 2px 0 0;
  font-size: 18px;
  font-weight: 600;
}

.monitor__events {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.monitor__event-list {
  max-height: 200px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.monitor__placeholder {
  margin: 0;
  color: #64748b;
}

.event {
  border: 1px solid rgba(148, 163, 184, 0.2);
  border-radius: 8px;
  padding: 8px;
  background: rgba(30, 64, 175, 0.15);
}

.event__type {
  margin: 0 0 4px;
  font-weight: 600;
  color: #93c5fd;
}

pre {
  margin: 0;
  font-size: 12px;
  color: #e2e8f0;
  white-space: pre-wrap;
}
</style>
