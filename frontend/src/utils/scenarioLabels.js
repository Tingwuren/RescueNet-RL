const scenarioNameMap = {
  typhoon_residual: "台风灾后残余网络",
  flood_no_residual: "洪涝孤岛通信恢复",
  earthquake_residual: "地震灾后断链恢复",
};

const disasterTypeMap = {
  typhoon: "台风灾害",
  earthquake: "地震灾害",
  flood: "洪涝灾害",
  wildfire: "山火灾害",
};

export const formatScenarioName = (name) => {
  if (!name) return "未选择场景";
  return scenarioNameMap[name] || String(name).replace(/_/g, " ");
};

export const formatDisasterType = (type) => {
  if (!type) return "灾害场景";
  return disasterTypeMap[type] || String(type);
};
