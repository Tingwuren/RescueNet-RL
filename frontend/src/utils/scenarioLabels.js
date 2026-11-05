const scenarioNameMap = {
  typhoon_residual: "台风灾后残余网络",
  flood_no_residual: "洪水孤岛通信恢复",
  earthquake_residual: "地震灾后断链恢复",
  super_typhoon__level_1: "台风灾后残余网络",
  super_typhoon__level_2: "台风灾后残余网络",
  super_typhoon__level_3: "台风灾后残余网络",
  super_typhoon__level_4: "台风灾后残余网络",
  extreme_rainstorm__level_1: "洪水孤岛通信恢复",
  extreme_rainstorm__level_2: "洪水孤岛通信恢复",
  extreme_rainstorm__level_3: "洪水孤岛通信恢复",
  extreme_rainstorm__level_4: "洪水孤岛通信恢复",
  water_disaster__level_1: "洪水孤岛通信恢复",
  water_disaster__level_2: "洪水孤岛通信恢复",
  water_disaster__level_3: "洪水孤岛通信恢复",
  water_disaster__level_4: "洪水孤岛通信恢复",
  destructive_earthquake__level_1: "地震灾后断链恢复",
  destructive_earthquake__level_2: "地震灾后断链恢复",
  destructive_earthquake__level_3: "地震灾后断链恢复",
  destructive_earthquake__level_4: "地震灾后断链恢复",
};

const disasterTypeMap = {
  typhoon: "台风灾害",
  earthquake: "地震灾害",
  rainstorm: "暴雨灾害",
  flood: "洪水灾害",
  wildfire: "山火灾害",
};

const plainDisasterNameMap = {
  typhoon: "台风",
  super_typhoon: "台风",
  earthquake: "地震",
  destructive_earthquake: "地震",
  rainstorm: "暴雨",
  extreme_rainstorm: "暴雨",
  flood: "暴雨",
};

export const normalizeDisasterDisplayText = (value) =>
  String(value || "")
    .replace(/water_disaster/gi, "extreme_rainstorm")
    .replace(/water[\s-]+disaster/gi, "extreme rainstorm")
    .replace(/超强水灾/g, "超强暴雨")
    .replace(/水灾/g, "暴雨");

export const formatScenarioName = (name) => {
  if (!name) return "未选择场景";
  const text = String(name || "").trim();
  const lower = text.toLowerCase();
  if (scenarioNameMap[lower]) return scenarioNameMap[lower];
  if (lower.includes("typhoon") || text.includes("台风")) return "台风灾后残余网络";
  if (lower.includes("earthquake") || text.includes("地震")) return "地震灾后断链恢复";
  if (
    lower.includes("rainstorm") ||
    lower.includes("flood") ||
    lower.includes("water_disaster") ||
    lower.includes("water disaster") ||
    text.includes("暴雨") ||
    text.includes("洪水") ||
    text.includes("水灾")
  ) {
    return "洪水孤岛通信恢复";
  }
  return normalizeDisasterDisplayText(scenarioNameMap[text] || text.replace(/_/g, " "));
};

export const formatDisasterType = (type) => {
  if (!type) return "灾害场景";
  return normalizeDisasterDisplayText(disasterTypeMap[type] || String(type));
};

export const formatPlainDisasterName = (...values) => {
  for (const value of values) {
    const text = String(value || "").trim();
    if (!text) continue;
    const lower = text.toLowerCase();
    if (plainDisasterNameMap[lower]) return plainDisasterNameMap[lower];
    if (lower.includes("earthquake") || text.includes("地震")) return "地震";
    if (lower.includes("rainstorm") || lower.includes("water_disaster") || lower.includes("water disaster") || text.includes("暴雨") || text.includes("水灾")) return "暴雨";
    if (lower.includes("flood") || text.includes("洪水")) return "暴雨";
    if (lower.includes("typhoon") || text.includes("台风")) return "台风";
  }
  return "";
};

const disasterScenarioOrderMap = {
  "暴雨": 0,
  "台风": 1,
  "地震": 2,
};

export const disasterScenarioSortRank = (...values) => {
  const label = formatPlainDisasterName(...values);
  return Object.prototype.hasOwnProperty.call(disasterScenarioOrderMap, label)
    ? disasterScenarioOrderMap[label]
    : Number.MAX_SAFE_INTEGER;
};

export const compareDisasterScenarioOrder = (leftValues = [], rightValues = []) => {
  const leftItems = Array.isArray(leftValues) ? leftValues : [leftValues];
  const rightItems = Array.isArray(rightValues) ? rightValues : [rightValues];
  const rankDelta = disasterScenarioSortRank(...leftItems) - disasterScenarioSortRank(...rightItems);
  if (rankDelta) return rankDelta;
  const leftText = String(leftItems.find(Boolean) || "");
  const rightText = String(rightItems.find(Boolean) || "");
  return leftText.localeCompare(rightText, "zh-CN");
};

export const preferredDisasterSeverityKey = (items = [], fallbackKey = "") => {
  const options = (Array.isArray(items) ? items : [])
    .map((item) => (typeof item === "string" ? { key: item, label: item } : item))
    .filter((item) => item?.key);
  const preferred = options.find((item) => {
    const key = String(item.key || "").toLowerCase();
    const label = String(item.label || "");
    return key === "level_4" || key.includes("level_4") || key.includes("extreme") || label.includes("特别严重");
  });
  if (preferred?.key) return preferred.key;
  if (fallbackKey && options.some((item) => item.key === fallbackKey)) return fallbackKey;
  return options[0]?.key || "";
};
