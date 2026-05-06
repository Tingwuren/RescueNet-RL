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

const cacheBuster = Date.now();
const nativeReplayUrl = `${ns3WebBase}${ns3WebBase.includes("?") ? "&" : "?"}v=${cacheBuster}`;
const frameReady = ref(false);
const frameRef = ref(null);

let lockTimer = null;
let stateTimer = null;

function ensureNativeRecoveryStyles(doc) {
  if (!doc?.head || doc.getElementById("rescuenet-native-recovery-bridge")) {
    return;
  }

  const style = doc.createElement("style");
  style.id = "rescuenet-native-recovery-bridge";
  style.textContent = `
    #map-container::after {
      content: '';
      position: absolute;
      inset: 0;
      z-index: 120;
      pointer-events: none;
      opacity: 0.18;
      transition: opacity 0.35s ease, background 0.35s ease;
      background: radial-gradient(circle at 50% 46%, rgba(0, 240, 255, 0.08) 0%, rgba(0, 240, 255, 0.03) 42%, transparent 74%);
    }

    #map-container.map-phase--normal::after {
      opacity: 0.18;
      background: radial-gradient(circle at 50% 46%, rgba(0, 240, 255, 0.08) 0%, rgba(0, 240, 255, 0.03) 42%, transparent 74%);
    }

    #map-container.map-phase--disaster::after {
      opacity: 0.52;
      background: radial-gradient(circle at 50% 46%, rgba(255, 0, 85, 0.22) 0%, rgba(255, 0, 85, 0.12) 38%, transparent 72%);
    }

    #map-container.map-phase--recovery::after {
      opacity: 0.34;
      background: radial-gradient(circle at 50% 46%, rgba(0, 255, 157, 0.14) 0%, rgba(0, 240, 255, 0.1) 38%, transparent 72%);
    }

    .status-card.recovery {
      background: rgba(0, 240, 255, 0.1) !important;
      border-color: rgba(0, 240, 255, 0.3) !important;
    }

    .status-dot.cyan {
      background: #00f0ff !important;
      color: #00f0ff !important;
      animation: rescuenet-native-pulse 1.4s ease-in-out infinite;
    }

    .disaster-alert.recovery {
      background: rgba(0, 240, 255, 0.12) !important;
      border-color: rgba(0, 240, 255, 0.38) !important;
      color: #bff8ff !important;
      box-shadow: 0 0 26px rgba(0, 240, 255, 0.22) !important;
      animation: none !important;
    }

    @keyframes rescuenet-native-pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.58; }
    }
  `;
  doc.head.appendChild(style);
}

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

  ensureNativeRecoveryStyles(doc);
  return true;
}

function inferNativePhase() {
  const frame = frameRef.value;
  const doc = frame?.contentDocument;
  const win = frame?.contentWindow;
  if (!doc || !win) return null;

  const slider = doc.getElementById("timelineSlider");
  if (!slider) return null;

  const currentTime = Number(slider.value || 0);
  const duration = Number(slider.max || 0);
  if (!Number.isFinite(currentTime) || !Number.isFinite(duration) || duration <= 0) {
    return null;
  }

  if (typeof win.getPhaseForTime === "function") {
    try {
      return win.getPhaseForTime(currentTime);
    } catch {
      return null;
    }
  }

  const disasterTime = Number(doc.getElementById("inpDisaster")?.value || 0);
  if (currentTime < disasterTime) return "normal";

  const remaining = Math.max(1, duration - disasterTime);
  const recoveryStart = Math.min(duration, disasterTime + Math.max(12, remaining * 0.22));
  const recoveryComplete = Math.min(duration, disasterTime + Math.max(32, remaining * 0.78));
  if (currentTime >= recoveryComplete) return "restored";
  if (currentTime > recoveryStart) return "recovery";
  return "disaster";
}

function applyNativeReplayChrome(phase) {
  const frame = frameRef.value;
  const doc = frame?.contentDocument;
  if (!doc || !phase) return false;

  ensureNativeRecoveryStyles(doc);

  const alertEl = doc.getElementById("disasterAlert");
  const statusCard = doc.getElementById("statusCard");
  const statusText = doc.getElementById("statusText");
  const statusDot = doc.getElementById("statusDot");
  const mapContainer = doc.getElementById("map-container");

  if (!alertEl || !statusCard || !statusText || !statusDot || !mapContainer) {
    return false;
  }

  mapContainer.classList.remove("map-phase--normal", "map-phase--disaster", "map-phase--recovery");
  alertEl.classList.remove("show", "recovery");

  if (phase === "normal") {
    mapContainer.classList.add("map-phase--normal");
    statusCard.className = "status-card normal";
    statusText.textContent = "通信枢纽运行正常";
    statusDot.className = "status-dot green";
    return true;
  }

  if (phase === "disaster") {
    mapContainer.classList.add("map-phase--disaster");
    alertEl.classList.add("show");
    alertEl.textContent = "⚠ 监测到极端灾害 | 宏基站受损 | 应急自组网协议已激活";
    statusCard.className = "status-card disaster";
    statusText.textContent = "极端灾害爆发 | 宏网熔断";
    statusDot.className = "status-dot red";
    return true;
  }

  mapContainer.classList.add("map-phase--recovery");
  alertEl.classList.add("show", "recovery");
  statusCard.className = "status-card recovery";
  statusDot.className = "status-dot cyan";

  if (phase === "restored") {
    alertEl.textContent = "✓ 主干链路已恢复 | 宏站回传重建 | 通信网络趋于稳定";
    statusText.textContent = "通信网络已恢复 | 覆盖稳定";
    return true;
  }

  alertEl.textContent = "↻ 应急链路接管中 | 回传骨干重建 | 覆盖持续回升";
  statusText.textContent = "应急链路重构中 | 网络恢复推进";
  return true;
}

function syncNativeReplayState() {
  const phase = inferNativePhase();
  if (!phase) return false;
  return applyNativeReplayChrome(phase);
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

function startStateLoop() {
  if (stateTimer) {
    window.clearInterval(stateTimer);
    stateTimer = null;
  }

  syncNativeReplayState();
  stateTimer = window.setInterval(() => {
    lockNativeMapInteraction();
    syncNativeReplayState();
  }, 300);
}

function handleFrameLoad() {
  frameReady.value = true;
  startLockLoop();
  startStateLoop();
}

onBeforeUnmount(() => {
  if (lockTimer) {
    window.clearInterval(lockTimer);
  }
  if (stateTimer) {
    window.clearInterval(stateTimer);
  }
});
</script>

<style scoped>
.ns3-native-frame {
  position: relative;
  width: 100%;
  height: 100%;
  min-height: 100%;
  overflow: hidden;
  border: none;
  background: #050810;
  box-shadow: none;
}

.ns3-native-frame__iframe {
  display: block;
  width: 100%;
  height: 100%;
  min-height: 100%;
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
    min-height: 100%;
  }

  .ns3-native-frame__iframe {
    height: 100%;
    min-height: 100%;
  }
}
</style>
