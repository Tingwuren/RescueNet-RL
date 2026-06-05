const USER_COUNT_KEYS = [
  "user_node_count",
  "userNodeCount",
  "user_nodes",
  "userNodes",
  "unique_user_count",
  "uniqueUserCount",
  "total_users",
  "totalUsers",
  "num_users",
  "numUsers",
  "user_count",
  "userCount",
];

const toCount = (value) => {
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric <= 0) return null;
  return Math.round(numeric);
};

export const isUserNodeRecord = (node) => {
  const type = String(node?.type ?? "").toUpperCase();
  return type === "USER" || Number(node?.type) === 0 || node?.node_role === "user";
};

const countUserNodes = (nodes) => {
  if (!Array.isArray(nodes)) return null;
  const count = nodes.filter(isUserNodeRecord).length;
  return count > 0 ? count : null;
};

const countUserDetails = (details) => (Array.isArray(details) && details.length ? details.length : null);

const sumHeatmapUsers = (heatmap) => {
  if (!Array.isArray(heatmap) || !heatmap.length) return null;
  const total = heatmap.reduce((sum, cell) => {
    const count = toCount(cell?.user_count ?? cell?.userCount ?? cell?.count);
    return sum + (count ?? 0);
  }, 0);
  return total > 0 ? total : null;
};

const directUserCount = (source) => {
  if (!source || typeof source !== "object") return null;
  for (const key of USER_COUNT_KEYS) {
    const count = toCount(source[key]);
    if (count != null) return count;
  }
  return null;
};

const nestedUserCount = (source) => {
  if (!source || typeof source !== "object") return null;
  return (
    directUserCount(source.summary) ??
    directUserCount(source.metrics) ??
    directUserCount(source.scenario) ??
    directUserCount(source.initial_state) ??
    directUserCount(source.initialState) ??
    directUserCount(source.final_state) ??
    directUserCount(source.finalState) ??
    countUserDetails(source.user_details) ??
    countUserDetails(source.userDetails) ??
    countUserDetails(source.initial_state?.user_details) ??
    countUserDetails(source.initialState?.userDetails) ??
    countUserDetails(source.final_state?.user_details) ??
    countUserDetails(source.finalState?.userDetails) ??
    sumHeatmapUsers(source.user_heatmap) ??
    sumHeatmapUsers(source.userHeatmap) ??
    countUserNodes(source.nodes) ??
    resolveUserNodeCount(source.scene) ??
    resolveUserNodeCount(source.scene_export?.disaster_scene) ??
    resolveUserNodeCount(source.scene_export?.deployment_scene) ??
    resolveUserNodeCount(source.sceneExport?.disasterScene) ??
    resolveUserNodeCount(source.sceneExport?.deploymentScene)
  );
};

export function resolveUserNodeCount(...sources) {
  for (const source of sources.flat().filter((item) => item != null)) {
    if (Array.isArray(source)) {
      const count = countUserNodes(source) ?? countUserDetails(source);
      if (count != null) return count;
      continue;
    }
    const count = directUserCount(source) ?? nestedUserCount(source);
    if (count != null) return count;
  }
  return null;
}

export const formatUserNodeCount = (value) => {
  const count = toCount(value);
  return count == null ? "未知" : count.toLocaleString("zh-CN");
};

export const buildUserNodeCountMessage = (prefix, ...sources) => {
  const count = resolveUserNodeCount(...sources);
  const countText = count == null ? "未知" : `${formatUserNodeCount(count)} 个`;
  return `${prefix}，用户节点数量=${countText}。`;
};

export const userNodeCountLogKey = (scope, ...sources) => {
  const count = resolveUserNodeCount(...sources);
  return `${scope || "user-nodes"}:${count == null ? "unknown" : count}`;
};
