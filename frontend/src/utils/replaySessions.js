const STORAGE_KEY = "rescuenet.replaySessions";
const ACTIVE_KEY = "rescuenet.activeReplaySession";
const SESSION_SCHEMA_VERSION = 2;

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
  if (!id) {
    window.localStorage.removeItem(ACTIVE_KEY);
    return;
  }
  window.localStorage.setItem(ACTIVE_KEY, id);
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
    .slice(0, 19);
  if (persist) {
    writeStorage([session, ...sessions]);
    setActiveReplaySessionId(sessionId);
  }
  return session;
};
