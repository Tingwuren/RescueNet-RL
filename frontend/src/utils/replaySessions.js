const STORAGE_KEY = "rescuenet.replaySessions";
const ACTIVE_KEY = "rescuenet.activeReplaySession";
const SESSION_SCHEMA_VERSION = 2;
const MAX_PERSISTED_SESSIONS = 6;
const PRIMARY_FRAME_LIMIT = 96;
const FALLBACK_FRAME_LIMIT = 48;
const LAST_RESORT_FRAME_LIMIT = 24;

const MAP_WIDTH = 5000;
const MAP_HEIGHT = 5000;

const readStorage = () => {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
};

const writeStorage = (sessions) => {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
};

const isStorageQuotaError = (error) =>
  error?.name === "QuotaExceededError" ||
  error?.name === "NS_ERROR_DOM_QUOTA_REACHED" ||
  error?.code === 22 ||
  error?.code === 1014;

const roundMetric = (value, precision = 3) => {
  const numeric = Number(value || 0);
  if (!Number.isFinite(numeric)) return 0;
  const factor = 10 ** precision;
  return Math.round(numeric * factor) / factor;
};

const sampleFrames = (frames, limit) => {
  if (!Array.isArray(frames)) return [];
  if (frames.length <= limit) return frames;
  if (limit <= 2) return [frames[0], frames.at(-1)].filter(Boolean);
  const selected = new Set([0, frames.length - 1]);
  const span = frames.length - 1;
  for (let index = 1; index < limit - 1; index += 1) {
    selected.add(Math.round((index / (limit - 1)) * span));
  }
  return Array.from(selected)
    .sort((left, right) => left - right)
    .map((index) => frames[index])
    .filter(Boolean);
};

const compactNode = (node) => {
  if (Array.isArray(node)) {
    return {
      id: Number(node[0] ?? 0),
      type: Number(node[1] ?? 0),
      x: roundMetric(node[2], 1),
      y: roundMetric(node[3], 1),
      online: Number(node[5] ?? 1) === 1,
      broadcastServed: false,
      kind: null,
    };
  }
  return {
    id: Number(node?.id ?? 0),
    type: Number(node?.type ?? 0),
    x: roundMetric(node?.x, 1),
    y: roundMetric(node?.y, 1),
    online: Boolean(node?.online ?? true),
    broadcastServed: Boolean(node?.broadcastServed ?? false),
    kind: node?.kind || null,
    coverageRadius: node?.coverageRadius == null ? undefined : roundMetric(node.coverageRadius, 1),
    siteIndex: node?.siteIndex == null ? undefined : Number(node.siteIndex),
    commMode: node?.commMode || undefined,
    broadcastMode: node?.broadcastMode || undefined,
  };
};

const compactLink = (link) => {
  if (Array.isArray(link)) {
    return {
      src: Number(link[0] ?? 0),
      dst: Number(link[1] ?? 0),
      protocol: Number(link[2] ?? 0),
    };
  }
  return {
    src: Number(link?.src ?? link?.srcId ?? 0),
    dst: Number(link?.dst ?? link?.dstId ?? 0),
    protocol: Number(link?.protocol ?? 0),
  };
};

const compactActionDesc = (action) => {
  if (!action) return null;
  return {
    site_index: action.site_index,
    location: action.location,
    region_label: action.region_label,
    comm_mode: action.comm_mode,
    broadcast_mode: action.broadcast_mode,
  };
};

const compactFrame = (frame) => ({
  frameIndex: Number(frame?.frameIndex ?? 0),
  time: roundMetric(frame?.time, 2),
  tp: roundMetric(frame?.tp, 3),
  loss: roundMetric(frame?.loss, 4),
  disaster: Number(frame?.disaster ?? 1),
  nodes: (frame?.nodes || []).map(compactNode),
  links: (frame?.links || []).map(compactLink),
  coverageRatio: roundMetric(frame?.coverageRatio, 4),
  broadcastRatio: roundMetric(frame?.broadcastRatio, 4),
  remainingBudget: roundMetric(frame?.remainingBudget, 2),
  reward: roundMetric(frame?.reward, 3),
  label: frame?.label || "",
  actionDesc: compactActionDesc(frame?.actionDesc),
  latestDeploymentId: frame?.latestDeploymentId == null ? null : Number(frame.latestDeploymentId),
  userCount: Number(frame?.userCount ?? 0),
  stationCount: Number(frame?.stationCount ?? 0),
  connectedUsers: Number(frame?.connectedUsers ?? 0),
  broadcastUsers: Number(frame?.broadcastUsers ?? 0),
});

const compactSessionForStorage = (session, frameLimit = PRIMARY_FRAME_LIMIT) => ({
  id: session.id,
  source: session.source,
  schemaVersion: session.schemaVersion,
  createdAt: session.createdAt,
  scenarioName: session.scenarioName,
  algorithm: session.algorithm,
  artifactSignature: session.artifactSignature,
  title: session.title,
  mapWidth: session.mapWidth,
  mapHeight: session.mapHeight,
  frames: sampleFrames(session.frames, frameLimit).map(compactFrame),
  summary: session.summary,
});

const writeStorageWithQuotaFallback = (session, existingSessions) => {
  if (typeof window === "undefined") return true;
  const uniqueExisting = existingSessions.filter((item) => item.id !== session.id);
  const attempts = [
    {
      frameLimit: PRIMARY_FRAME_LIMIT,
      oldSessionLimit: MAX_PERSISTED_SESSIONS - 1,
    },
    {
      frameLimit: FALLBACK_FRAME_LIMIT,
      oldSessionLimit: 2,
    },
    {
      frameLimit: LAST_RESORT_FRAME_LIMIT,
      oldSessionLimit: 0,
    },
  ];

  for (const attempt of attempts) {
    const sessions = [
      compactSessionForStorage(session, attempt.frameLimit),
      ...uniqueExisting
        .slice(0, attempt.oldSessionLimit)
        .map((item) => compactSessionForStorage(item, Math.min(attempt.frameLimit, FALLBACK_FRAME_LIMIT))),
    ];
    try {
      writeStorage(sessions);
      return true;
    } catch (error) {
      if (!isStorageQuotaError(error)) {
        throw error;
      }
      console.warn("Replay session storage quota exceeded, retrying with smaller payload.", error);
    }
  }

  try {
    window.localStorage.removeItem(STORAGE_KEY);
    writeStorage([compactSessionForStorage(session, LAST_RESORT_FRAME_LIMIT)]);
    return true;
  } catch (error) {
    if (!isStorageQuotaError(error)) {
      throw error;
    }
  }

  return false;
};

const gridToCoords = (row, col, rows, cols) => ({
  x: ((Number(col) + 0.5) / Math.max(1, Number(cols))) * MAP_WIDTH,
  y: ((Number(row) + 0.5) / Math.max(1, Number(rows))) * MAP_HEIGHT,
});

const nodeTypeFromBase = (baseKey) => {
  const normalized = String(baseKey || "").toLowerCase();
  return normalized.includes("macro") ? 1 : 2;
};

const normalizeUserNode = (detail, rows, cols, previousNode = null) => {
  const hasPosition = Array.isArray(detail?.position) && detail.position.length >= 2;
  const row = hasPosition ? Number(detail.position[0]) : null;
  const col = hasPosition ? Number(detail.position[1]) : null;
  const coords =
    hasPosition && Number.isFinite(row) && Number.isFinite(col)
      ? gridToCoords(row, col, rows, cols)
      : {
          x: Number(previousNode?.x || 0),
          y: Number(previousNode?.y || 0),
        };
  return {
    id: Number(detail?.id ?? previousNode?.id ?? 0),
    type: 0,
    ...coords,
    rxBytes: Number(detail?.demand ?? previousNode?.rxBytes ?? 0),
    online: Boolean(detail?.connected ?? previousNode?.online ?? false),
    broadcastServed: Boolean(detail?.broadcast_served ?? previousNode?.broadcastServed ?? false),
    kind: "user",
  };
};

const buildUserNodeMap = (userDetails, rows, cols, previousMap = new Map()) => {
  const nextMap = new Map(previousMap);
  for (const detail of userDetails || []) {
    const id = Number(detail?.id ?? -1);
    if (!Number.isFinite(id) || id < 0) continue;
    nextMap.set(id, normalizeUserNode(detail, rows, cols, previousMap.get(id) || null));
  }
  return nextMap;
};

const buildResidualNodes = (stations, rows, cols) =>
  (stations || []).map((station, index) => ({
    id: 100000 + index,
    type: nodeTypeFromBase(station.base_station),
    ...gridToCoords(Number(station.x || 0), Number(station.y || 0), rows, cols),
    rxBytes: 0,
    online: true,
    kind: "residual",
    coverageRadius: Number(station.coverage_radius || 0),
  }));

const buildDeployedNodes = (steps, rows, cols, upToIndex) => {
  const seen = new Set();
  const nodes = [];
  let latestDeploymentId = null;
  (steps || []).slice(0, upToIndex).forEach((step, index) => {
    const action = step.action_desc || {};
    const location = action.location;
    if (!Array.isArray(location) || location.length < 2) return;
    const row = Number(location[0]);
    const col = Number(location[1]);
    const key = `${row}:${col}:${action.comm_mode || "unknown"}`;
    if (seen.has(key)) return;
    seen.add(key);
    const nodeId = 200000 + index;
    nodes.push({
      id: nodeId,
      type: nodeTypeFromBase(action.comm_mode),
      ...gridToCoords(row, col, rows, cols),
      rxBytes: 0,
      online: true,
      kind: "deployed",
      siteIndex: Number(action.site_index ?? -1),
      commMode: action.comm_mode || null,
      broadcastMode: action.broadcast_mode || null,
    });
    latestDeploymentId = nodeId;
  });
  return { nodes, latestDeploymentId };
};

const buildConnectivityLinks = (userNodes, stationNodes) =>
  userNodes
    .filter((user) => user.online || user.broadcastServed)
    .map((user) => {
      let best = null;
      let bestDistance = Infinity;
      for (const station of stationNodes) {
        const dx = station.x - user.x;
        const dy = station.y - user.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        if (distance < bestDistance) {
          best = station;
          bestDistance = distance;
        }
      }
      if (!best) return null;
      return {
        src: user.id,
        dst: best.id,
        protocol: user.online ? 1 : 0,
      };
    })
    .filter(Boolean);

const inferGridShape = (report) => {
  const scenario = report?.scenario || {};
  if (scenario.grid_rows && scenario.grid_cols) {
    return {
      rows: Number(scenario.grid_rows),
      cols: Number(scenario.grid_cols),
    };
  }

  let maxRow = 0;
  let maxCol = 0;
  const inspectPosition = (position) => {
    if (!Array.isArray(position) || position.length < 2) return;
    maxRow = Math.max(maxRow, Number(position[0]) || 0);
    maxCol = Math.max(maxCol, Number(position[1]) || 0);
  };

  (report?.initial_state?.user_details || []).forEach((detail) => inspectPosition(detail.position));
  (report?.initial_state?.residual_base_stations || []).forEach((station) =>
    inspectPosition([station.x, station.y])
  );
  (report?.steps || []).forEach((step) => inspectPosition(step?.action_desc?.location));

  return {
    rows: Math.max(1, maxRow + 1),
    cols: Math.max(1, maxCol + 1),
  };
};

const buildFramesFromReport = (report) => {
  const initialState = report?.initial_state || {};
  const steps = report?.steps || [];
  const { rows, cols } = inferGridShape(report);

  const residualNodes = buildResidualNodes(initialState.residual_base_stations, rows, cols);
  let userNodeMap = buildUserNodeMap(initialState.user_details, rows, cols);
  const initialUsers = Array.from(userNodeMap.values()).sort((left, right) => left.id - right.id);
  const initialStations = residualNodes;

  const frames = [
    {
      frameIndex: 0,
      time: 0,
      tp: Number(initialState.avg_user_throughput || initialState.recent_throughput || 0),
      loss: Math.max(0, 1 - Number(initialState.coverage_ratio || 0)),
      disaster: 1,
      mapWidth: MAP_WIDTH,
      mapHeight: MAP_HEIGHT,
      nodes: [...initialUsers, ...initialStations],
      links: buildConnectivityLinks(
        initialUsers,
        initialStations
      ),
      coverageRatio: Number(initialState.coverage_ratio || 0),
      broadcastRatio: Number(initialState.broadcast_ratio || 0),
      remainingBudget: Number(initialState.remaining_budget || 0),
      reward: 0,
      label: "初始受灾场景",
      userCount: initialUsers.length,
      stationCount: initialStations.length,
      connectedUsers: initialUsers.filter((node) => node.online).length,
      broadcastUsers: initialUsers.filter((node) => node.broadcastServed).length,
    },
  ];

  steps.forEach((step, index) => {
    const postState = step.post_state || {};
    userNodeMap = buildUserNodeMap(postState.user_details, rows, cols, userNodeMap);
    const userNodes = Array.from(userNodeMap.values()).sort((left, right) => left.id - right.id);
    const { nodes: deployedNodes, latestDeploymentId } = buildDeployedNodes(steps, rows, cols, index + 1);
    const stationNodes = [...residualNodes, ...deployedNodes];
    frames.push({
      frameIndex: index + 1,
      time: index + 1,
      tp: Number(postState.avg_user_throughput || postState.recent_throughput || 0),
      loss: Math.max(0, 1 - Number(postState.coverage_ratio || 0)),
      disaster: 1,
      mapWidth: MAP_WIDTH,
      mapHeight: MAP_HEIGHT,
      nodes: [...userNodes, ...stationNodes],
      links: buildConnectivityLinks(userNodes, stationNodes),
      coverageRatio: Number(postState.coverage_ratio || 0),
      broadcastRatio: Number(postState.broadcast_ratio || 0),
      remainingBudget: Number(postState.remaining_budget || 0),
      reward: Number(step.reward || 0),
      label: `Step ${step.step || index + 1}`,
      actionDesc: step.action_desc || null,
      latestDeploymentId,
      userCount: userNodes.length,
      stationCount: stationNodes.length,
      connectedUsers: userNodes.filter((node) => node.online).length,
      broadcastUsers: userNodes.filter((node) => node.broadcastServed).length,
    });
  });

  return frames;
};

const normalizeSession = (session) => {
  const frames = Array.isArray(session?.frames) ? session.frames : [];
  const firstFrame = frames[0] || {};
  const initialNodes = Array.isArray(firstFrame.nodes) ? firstFrame.nodes : [];
  const initialUsers = initialNodes.filter((node) => Number(node.type) === 0).length;
  const initialStations = initialNodes.filter((node) => Number(node.type) !== 0).length;
  return {
    ...session,
    source: session?.source || "test",
    artifactSignature: session?.artifactSignature || null,
    schemaVersion: Number(session?.schemaVersion || 1),
    mapWidth: Number(session?.mapWidth || firstFrame.mapWidth || MAP_WIDTH),
    mapHeight: Number(session?.mapHeight || firstFrame.mapHeight || MAP_HEIGHT),
    summary: {
      totalReward: Number(session?.summary?.totalReward || 0),
      coverageRatio: Number(session?.summary?.coverageRatio || 0),
      broadcastRatio: Number(session?.summary?.broadcastRatio || 0),
      stepsTaken: Number(session?.summary?.stepsTaken || Math.max(0, frames.length - 1)),
      totalUsers: Number(session?.summary?.totalUsers || initialUsers),
      initialStations: Number(session?.summary?.initialStations || initialStations),
      finalStations: Number(session?.summary?.finalStations || frames.at(-1)?.stationCount || initialStations),
    },
  };
};

export const listReplaySessions = () =>
  readStorage()
    .map(normalizeSession)
    .sort((a, b) => Number(b.createdAt || 0) - Number(a.createdAt || 0));

export const getActiveReplaySessionId = () => {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem(ACTIVE_KEY);
};

export const setActiveReplaySessionId = (id) => {
  if (typeof window === "undefined") return;
  try {
    if (!id) {
      window.localStorage.removeItem(ACTIVE_KEY);
      return;
    }
    window.localStorage.setItem(ACTIVE_KEY, id);
  } catch (error) {
    console.warn("Failed to update active replay session id.", error);
  }
};

export const saveReplaySessionFromSimulation = ({
  scenarioName,
  algorithm,
  result,
  sessionMeta = {},
  persist = true,
}) => {
  const report = result?.reports?.[0];
  if (!report) return null;
  const source = sessionMeta.source || (result?.source === "training" ? "training" : "test");
  const titlePrefix =
    sessionMeta.titlePrefix ?? (source === "training" ? "训练回放" : null);
  const artifactSignature = sessionMeta.artifactSignature || null;
  const sessionId = `${source}-${Date.now()}`;
  const scenario = report.scenario || {};
  const frames = buildFramesFromReport(report);
  const firstFrame = frames[0] || {};
  const lastFrame = frames.at(-1) || firstFrame;
  const session = {
    id: sessionId,
    source,
    schemaVersion: SESSION_SCHEMA_VERSION,
    createdAt: Date.now(),
    scenarioName,
    algorithm,
    artifactSignature,
    title: titlePrefix
      ? `${titlePrefix} / ${scenarioName} / ${algorithm.toUpperCase()} / Episode ${report.episode || 1}`
      : `${scenarioName} / ${algorithm.toUpperCase()} / Episode ${report.episode || 1}`,
    mapWidth: MAP_WIDTH,
    mapHeight: MAP_HEIGHT,
    frames,
    summary: {
      totalReward: Number(report.total_reward || 0),
      coverageRatio: Number(report.final_state?.coverage_ratio || 0),
      broadcastRatio: Number(report.final_state?.broadcast_ratio || 0),
      stepsTaken: Number(report.steps_taken || 0),
      totalUsers: Number(firstFrame.userCount || 0),
      initialStations: Number(firstFrame.stationCount || 0),
      finalStations: Number(lastFrame.stationCount || 0),
    },
  };
  const sessions = listReplaySessions()
    .filter((item) => item.id !== sessionId)
    .filter((item) => !(artifactSignature && item.source === source && item.artifactSignature === artifactSignature))
    .filter((item) => !(item.source === source && item.title === session.title))
    .slice(0, MAX_PERSISTED_SESSIONS - 1);
  let persisted = !persist;
  if (persist) {
    persisted = writeStorageWithQuotaFallback(session, sessions);
    if (persisted) {
      setActiveReplaySessionId(sessionId);
    } else {
      console.warn("Replay session was generated but not persisted because localStorage quota is exhausted.");
    }
  }
  return {
    ...session,
    persisted,
  };
};
