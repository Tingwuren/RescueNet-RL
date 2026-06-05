import { ref } from "vue";

const TERMINAL_STORAGE_KEY = "rescuenet:shared-terminal-output:v1";
const TERMINAL_BROADCAST_CHANNEL = "rescuenet-shared-terminal-output";
const MAX_SHARED_TERMINAL_LINES = 1200;
const TERMINAL_SYNC_DELAY_MS = 180;

export const formatTerminalTimestamp = (value = Date.now()) => {
  const numeric = Number(value);
  const timestamp = Number.isFinite(numeric)
    ? numeric < 1e12
      ? numeric * 1000
      : numeric
    : Date.now();
  return new Date(timestamp).toLocaleTimeString("zh-CN", { hour12: false });
};

export const buildTerminalLine = (message, options = {}) => {
  const level = String(options.level || "INFO").toUpperCase();
  const source = options.source ? ` [${String(options.source).toUpperCase()}]` : "";
  return `[${formatTerminalTimestamp(options.timestamp)}] [${level}]${source} ${String(message || "")}`;
};

const readStoredTerminalLines = () => {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(TERMINAL_STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed.slice(-MAX_SHARED_TERMINAL_LINES) : [];
  } catch {
    return [];
  }
};

const normalizeTerminalLines = (lines) =>
  (Array.isArray(lines) ? lines : [])
    .filter((line) => line != null)
    .map((line) => String(line))
    .slice(-MAX_SHARED_TERMINAL_LINES);

export const terminalHistoryLines = ref(readStoredTerminalLines());

let terminalBroadcast = null;
let terminalSyncTimer = null;
let terminalSyncShouldBroadcast = false;

const persistTerminalLines = () => {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(TERMINAL_STORAGE_KEY, JSON.stringify(terminalHistoryLines.value));
  } catch {
    // Ignore storage quota errors; the in-memory terminal remains available.
  }
};

const postTerminalSync = () => {
  try {
    terminalBroadcast?.postMessage({
      type: "terminal-sync",
      lines: normalizeTerminalLines(terminalHistoryLines.value),
    });
  } catch {
    // BroadcastChannel cannot clone Vue proxies in some browsers; localStorage still keeps tabs in sync.
  }
};

const flushTerminalSync = () => {
  if (typeof window !== "undefined" && terminalSyncTimer) {
    window.clearTimeout(terminalSyncTimer);
  }
  terminalSyncTimer = null;
  persistTerminalLines();
  if (terminalSyncShouldBroadcast) {
    postTerminalSync();
  }
  terminalSyncShouldBroadcast = false;
};

const scheduleTerminalSync = ({ broadcast = true } = {}) => {
  terminalSyncShouldBroadcast = terminalSyncShouldBroadcast || broadcast;
  if (typeof window === "undefined") {
    flushTerminalSync();
    return;
  }
  if (terminalSyncTimer) return;
  terminalSyncTimer = window.setTimeout(flushTerminalSync, TERMINAL_SYNC_DELAY_MS);
};

const ensureBroadcastChannel = () => {
  if (terminalBroadcast || typeof window === "undefined" || typeof window.BroadcastChannel === "undefined") return;
  terminalBroadcast = new window.BroadcastChannel(TERMINAL_BROADCAST_CHANNEL);
  terminalBroadcast.onmessage = (event) => {
    if (event?.data?.type !== "terminal-sync" || !Array.isArray(event.data.lines)) return;
    terminalHistoryLines.value = normalizeTerminalLines(event.data.lines);
    scheduleTerminalSync({ broadcast: false });
  };
};

ensureBroadcastChannel();

if (typeof window !== "undefined") {
  window.addEventListener("storage", (event) => {
    if (event.key !== TERMINAL_STORAGE_KEY) return;
    terminalHistoryLines.value = readStoredTerminalLines();
  });
  window.addEventListener("beforeunload", flushTerminalSync);
}

export const appendTerminalLine = (lines, message, options = {}, limit = 400) => {
  if (!message) return lines;
  const nextLine = buildTerminalLine(message, options);
  const maxLines = Math.max(1, Number(limit) || 400);
  return [...lines.slice(-(maxLines - 1)), nextLine];
};

export const appendSharedTerminalLine = (message, options = {}, limit = MAX_SHARED_TERMINAL_LINES) => {
  if (!message) return "";
  ensureBroadcastChannel();
  const nextLine = buildTerminalLine(message, options);
  const maxLines = Math.max(1, Number(limit) || MAX_SHARED_TERMINAL_LINES);
  terminalHistoryLines.value = normalizeTerminalLines([...terminalHistoryLines.value.slice(-(maxLines - 1)), nextLine]);
  scheduleTerminalSync();
  return nextLine;
};

export const appendSharedTerminalLines = (items, options = {}, limit = MAX_SHARED_TERMINAL_LINES) => {
  const entries = Array.isArray(items) ? items : [];
  if (!entries.length) return [];
  ensureBroadcastChannel();
  const nextLines = [];
  entries.forEach((item) => {
    const isObject = item && typeof item === "object";
    const message = isObject ? item.message ?? item.text : item;
    if (!message) return;
    nextLines.push(buildTerminalLine(message, {
      ...options,
      ...(isObject ? item.options || {} : {}),
      ...(isObject && item.level ? { level: item.level } : {}),
      ...(isObject && item.source ? { source: item.source } : {}),
      ...(isObject && item.timestamp !== undefined ? { timestamp: item.timestamp } : {}),
    }));
  });
  if (!nextLines.length) return [];
  const maxLines = Math.max(1, Number(limit) || MAX_SHARED_TERMINAL_LINES);
  terminalHistoryLines.value = normalizeTerminalLines([...terminalHistoryLines.value, ...nextLines].slice(-maxLines));
  scheduleTerminalSync();
  return nextLines;
};

export const appendSyncedTerminalLine = (lines, message, options = {}, limit = 400) => {
  const nextLine = appendSharedTerminalLine(message, options);
  if (!nextLine) return lines;
  const maxLines = Math.max(1, Number(limit) || 400);
  return [...lines.slice(-(maxLines - 1)), nextLine];
};

export const clearTerminalOutput = () => {
  ensureBroadcastChannel();
  terminalHistoryLines.value = [];
  terminalSyncShouldBroadcast = true;
  flushTerminalSync();
  return true;
};

export const exportTerminalOutput = (lines = terminalHistoryLines.value, filename = "") => {
  if (typeof document === "undefined") return false;
  const normalizedLines = Array.isArray(lines) ? lines : terminalHistoryLines.value;
  const timestamp = new Date()
    .toISOString()
    .replace(/[:.]/g, "-")
    .replace("T", "_")
    .slice(0, 19);
  const downloadName = filename || `rescuenet-terminal-${timestamp}.log`;
  const content = normalizedLines.length ? normalizedLines.join("\n") : "暂无终端输出";
  const blob = new Blob([`${content}\n`], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = downloadName;
  link.click();
  URL.revokeObjectURL(url);
  return true;
};
