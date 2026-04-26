<template>
  <section class="ns3-native-frame" aria-label="ns-3 原生真实回放界面">
    <iframe
      ref="frameRef"
      class="ns3-native-frame__iframe"
      :src="nativeReplayUrl"
      title="ns-3 原生真实回放"
      @load="handleFrameLoad"
    ></iframe>

    <div v-if="!frameReady" class="ns3-native-frame__loading">
      <strong>正在加载 ns-3 原生回放界面</strong>
      <span>{{ nativeReplayUrl }}</span>
      <a :href="nativeReplayUrl" target="_blank" rel="noreferrer">新窗口打开</a>
    </div>
  </section>
</template>

<script setup>
import { onBeforeUnmount, ref } from "vue";
import { ns3WebBase } from "../utils/runtimeEndpoints";

const nativeReplayUrl = ns3WebBase;
const frameReady = ref(false);
const frameRef = ref(null);

let lockTimer = null;

function lockNativeMapInteraction() {
  const frame = frameRef.value;
  const doc = frame?.contentDocument;

  if (!doc?.head) {
    return false;
  }

  const mapContainer =
    doc.getElementById("map-container") ||
    doc.getElementById("map") ||
    doc.querySelector(".leaflet-container");

  if (!mapContainer) {
    return false;
  }

  if (!doc.getElementById("rescuenet-native-map-lock")) {
    const style = doc.createElement("style");
    style.id = "rescuenet-native-map-lock";
    style.textContent = `
      #map-container,
      #map,
      .leaflet-container,
      .leaflet-pane,
      .leaflet-control-container,
      .leaflet-top,
      .leaflet-bottom {
        pointer-events: none !important;
        touch-action: none !important;
      }

      .leaflet-control-zoom {
        display: none !important;
      }
    `;
    doc.head.appendChild(style);
  }

  return true;
}

function startLockLoop() {
  if (lockTimer) {
    window.clearInterval(lockTimer);
    lockTimer = null;
  }

  let attempts = 0;
  const tryLock = () => {
    attempts += 1;
    const locked = lockNativeMapInteraction();
    if (locked || attempts >= 20) {
      window.clearInterval(lockTimer);
      lockTimer = null;
    }
  };

  const locked = lockNativeMapInteraction();
  if (!locked) {
    attempts = 1;
    lockTimer = window.setInterval(tryLock, 250);
  }
}

function handleFrameLoad() {
  frameReady.value = true;
  startLockLoop();
}

onBeforeUnmount(() => {
  if (lockTimer) {
    window.clearInterval(lockTimer);
  }
});
</script>

<style scoped>
.ns3-native-frame {
  position: relative;
  min-height: 100vh;
  overflow: hidden;
  border: none;
  background: #050810;
  box-shadow: none;
}

.ns3-native-frame__iframe {
  display: block;
  width: 100%;
  height: 100vh;
  min-height: 760px;
  border: 0;
  background: #050810;
}

.ns3-native-frame__loading {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 10px;
  padding: 24px;
  background: #050810;
  color: #f8fafc;
  text-align: center;
}

.ns3-native-frame__loading strong {
  font-size: 1rem;
}

.ns3-native-frame__loading span {
  max-width: 100%;
  color: #94a3b8;
  font-size: 12px;
  overflow-wrap: anywhere;
}

.ns3-native-frame__loading a {
  padding: 9px 13px;
  border: 1px solid rgba(0, 240, 255, 0.35);
  border-radius: 8px;
  color: #00f0ff;
  text-decoration: none;
  background: rgba(0, 240, 255, 0.1);
}

@media (max-width: 720px) {
  .ns3-native-frame {
    min-height: 100vh;
  }

  .ns3-native-frame__iframe {
    height: 100vh;
    min-height: 680px;
  }
}
</style>
