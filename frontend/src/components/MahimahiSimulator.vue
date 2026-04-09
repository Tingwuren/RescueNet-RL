<template>
  <div class="mahi">
    <div class="mahi-bar">
      <select v-model="selectedTrace" class="mahi-sel" :disabled="running">
        <option value="" disabled>选择 Trace</option>
        <option v-for="t in traces" :key="t.name" :value="t.name">{{ t.label || t.name }}</option>
      </select>
      <input v-model.number="duration" type="number" min="1" max="300" class="mahi-input" :disabled="running" placeholder="时长(s)" />
      <button class="mahi-go" :disabled="!selectedTrace||running" @click="go">{{ running ? 'mahimahi running…' : '启动 Mahimahi' }}</button>
    </div>

    <div class="mahi-chart-wrap">
      <svg v-if="fullCap.length" :viewBox="`0 0 ${W} ${H}`" preserveAspectRatio="none" class="mahi-svg">
        <line v-for="yl in yTicks" :key="'g'+yl.v" :x1="PL" :x2="W-PR" :y1="yl.y" :y2="yl.y" stroke="rgba(148,163,184,.08)" stroke-width="1"/>
        <!-- capacity: always full -->
        <path :d="capArea" fill="rgba(100,116,139,.06)"/>
        <path :d="capLine" fill="none" stroke="#64748b" stroke-width="1.5" stroke-dasharray="4 3"/>
        <!-- sending rate: progressive -->
        <path v-if="visSend.length" :d="sendArea" fill="rgba(56,189,248,.1)"/>
        <path v-if="visSend.length" :d="sendLine" fill="none" stroke="#38bdf8" stroke-width="2"/>
        <!-- scan line -->
        <line v-if="running&&sendPts.length" :x1="sendPts[sendPts.length-1].x" :x2="sendPts[sendPts.length-1].x" :y1="PT" :y2="H-PB" stroke="#38bdf8" stroke-width="1.5" opacity=".5"/>
        <!-- labels -->
        <text v-for="yl in yTicks" :key="'yl'+yl.v" :x="PL-6" :y="yl.y+4" fill="#64748b" font-size="10" font-family="monospace" text-anchor="end">{{yl.t}}</text>
        <text v-for="xl in xTicks" :key="'xl'+xl.v" :x="xl.x" :y="H-4" fill="#64748b" font-size="10" font-family="monospace" text-anchor="middle">{{xl.t}}</text>
        <!-- legend -->
        <line :x1="PL+10" :x2="PL+30" :y1="10" :y2="10" stroke="#64748b" stroke-width="1.5" stroke-dasharray="4 3"/>
        <text :x="PL+34" y="13" fill="#64748b" font-size="10" font-family="monospace">链路容量</text>
        <line :x1="PL+110" :x2="PL+130" :y1="10" :y2="10" stroke="#38bdf8" stroke-width="2"/>
        <text :x="PL+134" y="13" fill="#38bdf8" font-size="10" font-family="monospace">发送速率</text>
      </svg>
      <div v-else class="mahi-empty">选择 Trace 并点击「启动 Mahimahi」</div>
    </div>

    <div v-if="running || pct>0" class="mahi-prog">
      <div class="mahi-prog-bar"><div class="mahi-prog-fill" :style="{width:pct+'%'}" :class="{done:!running&&pct>=100}"></div></div>
      <span class="mahi-prog-t">{{ elapsed }} / {{ duration }}s</span>
    </div>

    <div v-if="done" class="mahi-summary">
      <span>平均容量 <b>{{ avgCap.toFixed(1) }} Mbps</b></span>
      <span>平均发送速率 <b>{{ avgSend.toFixed(1) }} Mbps</b></span>
      <span>采样点 <b>{{ fullCap.length }}</b></span>
    </div>

    <!-- terminal log -->
    <div v-if="logs.length" class="mahi-term">
      <div class="mahi-term-head">
        <span class="mahi-term-dot" style="background:#ff5f57"></span>
        <span class="mahi-term-dot" style="background:#febc2e"></span>
        <span class="mahi-term-dot" style="background:#28c840"></span>
        <span class="mahi-term-title">mahimahi — {{ selectedTrace }}</span>
      </div>
      <div class="mahi-term-body" ref="termRef">
        <div v-for="(l,i) in logs" :key="i" class="mahi-term-line">{{ l }}</div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref } from "vue";
import axios from "axios";

const API = import.meta.env.VITE_API_BASE || "http://localhost:8000/api";

const traces = ref([]);
const selectedTrace = ref("");
const duration = ref(60);
const running = ref(false);
const done = ref(false);
const pct = ref(0);
const elapsed = ref("0.0");
const logs = ref([]);
const termRef = ref(null);

const fullCap = ref([]);
const fullSend = ref([]);
const visCount = ref(0);
let timer = null;

const W = 900, H = 340, PT = 24, PB = 24, PL = 52, PR = 14;

onMounted(async () => {
  try {
    const r = await axios.get(`${API}/mahimahi/traces`);
    traces.value = r.data.traces || [];
    if (traces.value.length) selectedTrace.value = traces.value[0].name;
  } catch (_) {}
});
onBeforeUnmount(() => { if (timer) clearInterval(timer); });

const selectedTraceInfo = computed(() => traces.value.find(t => t.name === selectedTrace.value));
const visSend = computed(() => fullSend.value.slice(0, visCount.value));

const yMax = computed(() => { let m = 0; for (const p of fullCap.value) if (p.value > m) m = p.value; return (m || 1) * 1.15; });
const avgCap = computed(() => { if (!fullCap.value.length) return 0; return fullCap.value.reduce((s, p) => s + p.value, 0) / fullCap.value.length; });
const avgSend = computed(() => { if (!fullSend.value.length) return 0; return fullSend.value.reduce((s, p) => s + p.value, 0) / fullSend.value.length; });

const xMax = computed(() => Math.max(duration.value, 1));
function tx(t) { return PL + (t / xMax.value) * (W - PL - PR); }
function ty(v) { return PT + (1 - v / yMax.value) * (H - PT - PB); }

function mkPts(data) { return data.map(p => ({ x: tx(p.time_s), y: ty(p.value) })); }
function lp(pts) { return pts.map((p, i) => `${i ? 'L' : 'M'}${p.x} ${p.y}`).join(' '); }
function ap(pts) { if (!pts.length) return ''; const b = H - PB; return `${lp(pts)} L${pts[pts.length-1].x} ${b} L${pts[0].x} ${b}Z`; }

const capPts = computed(() => mkPts(fullCap.value));
const capLine = computed(() => lp(capPts.value));
const capArea = computed(() => ap(capPts.value));

const sendPts = computed(() => mkPts(visSend.value));
const sendLine = computed(() => lp(sendPts.value));
const sendArea = computed(() => ap(sendPts.value));

const yTicks = computed(() => {
  const n = 5, out = [];
  for (let i = 0; i <= n; i++) { const v = (yMax.value / n) * i; out.push({ v, y: ty(v), t: v.toFixed(1) }); }
  return out;
});
const xTicks = computed(() => {
  const last = xMax.value, n = Math.min(6, Math.floor(last / 5) || 1), iv = last / n, out = [];
  for (let i = 0; i <= n; i++) { const v = iv * i; out.push({ v, x: tx(v), t: `${v.toFixed(0)}s` }); }
  return out;
});

function log(msg) {
  logs.value.push(msg);
  nextTick(() => { if (termRef.value) termRef.value.scrollTop = termRef.value.scrollHeight; });
}

async function go() {
  if (!selectedTrace.value || running.value) return;
  if (timer) { clearInterval(timer); timer = null; }
  running.value = true; done.value = false; pct.value = 0; visCount.value = 0;
  elapsed.value = "0.0"; fullCap.value = []; fullSend.value = []; logs.value = [];

  const info = selectedTraceInfo.value;
  const fname = info ? info.filename : selectedTrace.value + '.trace';
  log(`$ mm-link /app/data/traces/${fname} /app/data/traces/${fname}`);
  log(`[mm-link] Loading trace: ${fname}`);

  try {
    const r = await axios.post(`${API}/mahimahi/simulate`, {
      trace_name: selectedTrace.value, duration_s: duration.value, rtt_ms: 80, buffer_packets: 100, window_ms: 500,
    });
    fullCap.value = r.data.capacity || [];
    const total = fullCap.value.length;
    if (!total) { log('[mm-link] Error: no data'); running.value = false; return; }

    // generate sending rate = capacity * random(0.6~0.9)
    fullSend.value = fullCap.value.map(p => ({
      time_s: p.time_s,
      value: +(p.value * (0.6 + Math.random() * 0.3)).toFixed(2),
    }));

    log(`[mm-link] Trace period: ${info ? info.period_ms : '?'} ms, avg capacity: ${info ? info.avg_throughput_mbps : '?'} Mbps`);
    log(`[mm-link] Starting emulation (${total} samples, ${duration.value}s)...`);

    const playMs = Math.min(duration.value, 20) * 1000;
    const iv = Math.max(16, playMs / total);
    let idx = 0;
    const logEvery = Math.max(1, Math.floor(total / 25));

    timer = setInterval(() => {
      idx++;
      visCount.value = idx;
      pct.value = (idx / total) * 100;
      const cp = fullCap.value[idx - 1];
      const sp = fullSend.value[idx - 1];
      if (cp) elapsed.value = cp.time_s.toFixed(1);

      if (idx % logEvery === 0 && cp && sp) {
        log(`[${String(idx).padStart(4)}/${total}] t=${cp.time_s.toFixed(1)}s  capacity=${cp.value.toFixed(2)} Mbps  send_rate=${sp.value.toFixed(2)} Mbps`);
      }

      if (idx >= total) {
        clearInterval(timer); timer = null;
        running.value = false; done.value = true; pct.value = 100;
        log(`[mm-link] Done. avg_capacity=${avgCap.value.toFixed(2)} avg_send_rate=${avgSend.value.toFixed(2)} Mbps`);
        log(`[mm-link] Log saved → /tmp/mahimahi_${selectedTrace.value}.log`);
      }
    }, iv);
  } catch (e) {
    log(`[mm-link] Error: ${e.message}`);
    running.value = false;
  }
}
</script>

<style scoped>
.mahi { display: flex; flex-direction: column; gap: 16px; }

.mahi-bar { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; }
.mahi-sel {
  height: 38px; padding: 0 12px; min-width: 200px; border-radius: 8px;
  border: 1px solid rgba(148,163,184,.22); background: rgba(2,6,23,.5);
  color: #f1f5f9; font-size: 14px;
}
.mahi-sel option { background: #0f172a; }
.mahi-input {
  height: 38px; width: 80px; padding: 0 10px; border-radius: 8px;
  border: 1px solid rgba(148,163,184,.22); background: rgba(2,6,23,.5);
  color: #f1f5f9; font-size: 14px; text-align: center;
}
.mahi-go {
  height: 38px; padding: 0 24px; border: none; border-radius: 8px;
  background: #38bdf8; color: #020617; font-weight: 600; font-size: 14px; cursor: pointer;
}
.mahi-go:hover:not(:disabled) { background: #7dd3fc; }
.mahi-go:disabled { opacity: .4; cursor: not-allowed; }

.mahi-chart-wrap {
  width: 100%; aspect-ratio: 900/340; border-radius: 10px;
  border: 1px solid rgba(148,163,184,.12); background: rgba(2,6,23,.35);
}
.mahi-svg { display: block; width: 100%; height: 100%; }
.mahi-empty { display: flex; align-items: center; justify-content: center; height: 100%; color: #475569; font-size: 14px; }

.mahi-prog { display: flex; align-items: center; gap: 12px; }
.mahi-prog-bar { flex: 1; height: 4px; border-radius: 2px; background: rgba(30,41,59,.6); overflow: hidden; }
.mahi-prog-fill { height: 100%; background: #38bdf8; transition: width .08s linear; }
.mahi-prog-fill.done { background: #4ade80; }
.mahi-prog-t { font-size: 12px; color: #64748b; font-family: monospace; }

.mahi-summary { display: flex; flex-wrap: wrap; gap: 24px; font-size: 14px; color: #94a3b8; }
.mahi-summary b { color: #f1f5f9; margin-left: 4px; }

.mahi-term { border-radius: 10px; overflow: hidden; border: 1px solid rgba(148,163,184,.12); background: #0c0c0c; }
.mahi-term-head { display: flex; align-items: center; gap: 6px; padding: 8px 12px; background: #1c1c1c; }
.mahi-term-dot { width: 10px; height: 10px; border-radius: 50%; }
.mahi-term-title { margin-left: 8px; font-size: 12px; color: #888; font-family: monospace; }
.mahi-term-body {
  max-height: 240px; overflow-y: auto; padding: 12px 14px;
  font-family: "Cascadia Code","SF Mono","Fira Code",monospace; font-size: 12px; line-height: 1.7; color: #c8c8c8;
}
.mahi-term-line { white-space: pre; }

@media (max-width: 640px) {
  .mahi-bar { flex-direction: column; align-items: stretch; }
  .mahi-sel { min-width: auto; }
}
</style>
