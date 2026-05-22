const COMMUNICATION_TYPE_OPTIONS = [
  { value: "cellular", label: "蜂窝通信" },
  { value: "wifi", label: "WiFi 通信" },
  { value: "satellite", label: "卫星通信" },
  { value: "shortwave", label: "短波通信" },
];

const DEFAULT_DEVICE_TEMPLATES = [
  {
    id: "tester-cellular-macro",
    name: "测试蜂窝宏站",
    deviceType: "宏基站",
    communicationType: "cellular",
    quantity: 1,
    maxThroughput: 240,
    maxUsers: 180,
    enabled: true,
    status: "已导入"
  },
  {
    id: "tester-wifi-hotspot",
    name: "测试 WiFi6 热点",
    deviceType: "背负式基站",
    communicationType: "wifi",
    quantity: 1,
    maxThroughput: 160,
    maxUsers: 96,
    enabled: true,
    status: "已导入"
  },
  {
    id: "tester-satellite-relay",
    name: "测试卫星中继",
    deviceType: "中继设备",
    communicationType: "satellite",
    quantity: 1,
    maxThroughput: 150,
    maxUsers: 120,
    enabled: true,
    status: "已导入"
  },
  {
    id: "tester-shortwave-station",
    name: "测试短波台",
    deviceType: "临时设备/车载设备",
    communicationType: "shortwave",
    quantity: 1,
    maxThroughput: 24,
    maxUsers: 220,
    enabled: true,
    status: "已导入"
  }
];

const buildInjectionScript = (apiBase, communicationTypes, defaultTemplates) => `
(function () {
  var API = ${JSON.stringify(apiBase)};
  var TEST_HISTORY_KEY = "prototype-tester-history";
  var DEVICE_LIBRARY_KEY = "prototype-tester-device-library-v1";
  var DEVICE_BINDINGS_KEY = "prototype-tester-device-bindings-v1";
  var COMM_TYPES = ${JSON.stringify(communicationTypes)};
  var DEFAULT_DEVICE_TEMPLATES = ${JSON.stringify(defaultTemplates)};
  var ALGORITHMS = [
    { key: "ppo", label: "PPO（基线）" },
    { key: "dqn", label: "DQN（大动作空间）" },
    { key: "a3c", label: "A3C（多目标）" },
    { key: "mppo", label: "MPPO（多头策略）" },
    { key: "hmarl", label: "HMARL（层次协同）" }
  ];
  var STATION_COLORS = ["#f59e0b", "#a78bfa", "#14b8a6", "#f97316", "#22c55e", "#38bdf8", "#eab308"];
  var state = {
    scenarios: [],
    artifacts: [],
    scenarioName: "",
    algorithm: "ppo",
    checkpointPath: "",
    importedScene: null,
    simulationResult: null,
    activeSceneTab: "imported",
    loadingScene: false,
    running: false,
    scenarioDeviceRows: []
  };

  function byId(id) {
    return document.getElementById(id);
  }

  function textHolder(id) {
    var node = byId(id);
    if (!node) return null;
    return node.querySelector("span") || node;
  }

  function setText(id, value) {
    var holder = textHolder(id);
    if (holder) {
      holder.textContent = value;
    }
  }

  function setPanelVisible(id, visible) {
    var panel = byId(id);
    if (!panel) return;
    panel.style.display = visible ? "block" : "none";
    panel.style.visibility = visible ? "visible" : "hidden";
    if (visible) {
      panel.classList.remove("ax_default_hidden");
    } else {
      panel.classList.add("ax_default_hidden");
    }
  }

  function setElementVisible(id, visible) {
    var node = byId(id);
    if (!node) return;
    node.style.display = visible ? "" : "none";
    node.style.visibility = visible ? "visible" : "hidden";
    if (visible) {
      node.classList.remove("ax_default_hidden");
    } else {
      node.classList.add("ax_default_hidden");
    }
  }

  function algorithmLabel(key) {
    var match = ALGORITHMS.find(function (item) {
      return item.key === key;
    });
    return match ? match.label : key.toUpperCase();
  }

  function disasterLabel(type) {
    var mapping = {
      flood: "洪水孤岛通信恢复",
      earthquake: "地震灾后断链恢复",
      landslide: "泥石流滑坡通信阻断恢复",
      typhoon: "台风灾后残余网络"
    };
    return mapping[type] || type || "灾害场景";
  }

  function scenarioLabel(name) {
    var scenario = state.scenarios.find(function (item) {
      return item.name === name;
    });
    if (!scenario) return name || "未选择场景";
    return disasterLabel(scenario.disaster_type);
  }

  function comboLabel(scenarioName, algorithm) {
    return scenarioLabel(scenarioName) + " + " + algorithmLabel(algorithm);
  }

  function currentScenario() {
    return state.scenarios.find(function (item) {
      return item.name === state.scenarioName;
    }) || null;
  }

  function evaluationProtocol() {
    var scenario = currentScenario();
    return scenario && scenario.disaster_type === "earthquake" ? "earthquake_stress" : "standard";
  }

  function findMatchingArtifact() {
    return state.artifacts.find(function (artifact) {
      return artifact.scenario_name === state.scenarioName &&
        artifact.algorithm === state.algorithm &&
        artifact.checkpoint_path;
    }) || null;
  }

  function escapeHtml(value) {
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function formatNumber(value, digits) {
    if (typeof value !== "number" || !isFinite(value)) return "--";
    return value.toFixed(digits == null ? 2 : digits);
  }

  function formatPercent(value) {
    if (typeof value !== "number" || !isFinite(value)) return "--";
    return (value * 100).toFixed(2) + "%";
  }

  function setPercentText(id, value) {
    var number = typeof value === "number" && isFinite(value) ? value : null;
    var node = byId(id);
    if (!node) return;
    var spans = node.querySelectorAll("span");
    if (spans.length >= 2) {
      spans[0].textContent = number == null ? "--" : (number * 100).toFixed(2);
      spans[1].textContent = number == null ? "" : "%";
      return;
    }
    setText(id, number == null ? "--" : formatPercent(number));
  }

  function formatMetric(value, digits) {
    if (typeof value !== "number" || !isFinite(value)) return "--";
    return value.toFixed(digits == null ? 2 : digits);
  }

  function sceneExportHasFile(sceneExport, key) {
    if (!sceneExport) return false;
    var pathKey = key === "disaster_scene" ? "disaster_scene_path" : "deployment_scene_path";
    return Boolean(sceneExport[pathKey] && sceneExport[key]);
  }

  function updateSceneExportButtons(result) {
    var sceneExport = result && result.scene_export ? result.scene_export : null;
    var hasDisasterFile = sceneExportHasFile(sceneExport, "disaster_scene");
    var hasDeploymentFile = sceneExportHasFile(sceneExport, "deployment_scene");
    ["u2999", "u3337"].forEach(function (id) {
      setElementVisible(id, hasDisasterFile);
    });
    ["u3002", "u3340"].forEach(function (id) {
      setElementVisible(id, hasDeploymentFile);
    });
  }

  function removePrototypeMoreButton() {
    var moreButton = byId("u3025");
    if (moreButton) moreButton.remove();
  }

  function numberValue(value, fallback) {
    var next = Number(value);
    return Number.isFinite(next) ? next : fallback;
  }

  function integerValue(value, fallback) {
    var next = Number(value);
    return Number.isFinite(next) ? Math.max(0, Math.round(next)) : fallback;
  }

  function formatDateTime(value) {
    if (!value) return "--";
    try {
      return new Date(value).toLocaleString("zh-CN");
    } catch (error) {
      return "--";
    }
  }

  function readStorage(key, fallback) {
    try {
      var raw = window.localStorage.getItem(key);
      return raw ? JSON.parse(raw) : fallback;
    } catch (error) {
      return fallback;
    }
  }

  function writeStorage(key, value) {
    window.localStorage.setItem(key, JSON.stringify(value));
  }

  function communicationLabel(type) {
    var found = COMM_TYPES.find(function (item) {
      return item.value === type;
    });
    return found ? found.label : type || "--";
  }

  function ensureDeviceLibrarySeeded() {
    var current = readStorage(DEVICE_LIBRARY_KEY, []);
    if (Array.isArray(current) && current.length) return current;
    writeStorage(DEVICE_LIBRARY_KEY, DEFAULT_DEVICE_TEMPLATES);
    return DEFAULT_DEVICE_TEMPLATES.slice();
  }

  function loadDeviceLibrary() {
    var items = readStorage(DEVICE_LIBRARY_KEY, []);
    return Array.isArray(items) ? items : [];
  }

  function stationCommunicationCategory(station) {
    var text = ((station && station.name) || "") + " " + ((station && station.label) || "") + " " + ((station && station.supported_modes || []).join(" "));
    if (/wifi/i.test(text)) return "wifi";
    if (/satellite|卫星/i.test(text)) return "satellite";
    if (/shortwave|hf|短波/i.test(text)) return "shortwave";
    if (/mesh|uav|无人机/i.test(text)) return "mesh";
    if (/5g|macro|mmwave|蜂窝|宏站|微站/i.test(text)) return "cellular";
    return "custom";
  }

  function communicationCategoryLabel(type) {
    if (type === "mesh") return "Mesh/UAV";
    if (type === "custom") return "专用通信";
    return communicationLabel(type);
  }

  function scenarioBaseStations() {
    var scenario = currentScenario();
    return scenario && Array.isArray(scenario.base_stations) ? scenario.base_stations : [];
  }

  function scenarioBaseStationByName(name) {
    return scenarioBaseStations().find(function (station) {
      return station.name === name;
    }) || null;
  }

  function defaultStationMode(station) {
    return station && Array.isArray(station.supported_modes) && station.supported_modes.length ? station.supported_modes[0] : null;
  }

  function buildScenarioStationRow(station, binding) {
    var category = stationCommunicationCategory(station);
    return {
      deviceId: "station:" + state.scenarioName + ":" + station.name,
      baseStationName: station.name,
      mode: binding && binding.mode ? binding.mode : defaultStationMode(station),
      name: station.label || station.name,
      deviceType: station.label || station.name,
      communicationType: category,
      quantity: binding && binding.quantity != null ? Math.max(1, integerValue(binding.quantity, 1)) : 1,
      maxThroughput: numberValue(station.max_throughput, 0),
      maxUsers: integerValue(station.max_users, 0),
      enabled: binding && binding.enabled != null ? binding.enabled !== false : true,
      applied: Boolean(binding && binding.applied),
      x: integerValue(binding && binding.x, 0),
      y: integerValue(binding && binding.y, 0),
      status: "已导入"
    };
  }

  function loadScenarioDeviceRows() {
    var baseStations = scenarioBaseStations();
    var bindings = readStorage(DEVICE_BINDINGS_KEY, {});
    var current = Array.isArray(bindings[state.scenarioName]) ? bindings[state.scenarioName] : [];
    var bindingMap = {};
    current.forEach(function (item) {
      bindingMap[item.deviceId] = item;
    });
    if (baseStations.length) {
      var baseRows = baseStations.map(function (station) {
        var id = "station:" + state.scenarioName + ":" + station.name;
        return buildScenarioStationRow(station, bindingMap[id]);
      });
      var baseIds = {};
      baseRows.forEach(function (row) {
        baseIds[row.deviceId] = true;
      });
      var cloneRows = current.filter(function (item) {
        return item && item.deviceId && !baseIds[item.deviceId] && scenarioBaseStationByName(item.baseStationName);
      }).map(function (item) {
        var station = scenarioBaseStationByName(item.baseStationName);
        var row = buildScenarioStationRow(station, item);
        row.deviceId = item.deviceId;
        row.name = item.name || row.name;
        row.applied = Boolean(item.applied);
        return row;
      });
      state.scenarioDeviceRows = baseRows.concat(cloneRows);
      return;
    }

    var library = loadDeviceLibrary();
    state.scenarioDeviceRows = library.map(function (device) {
      var binding = bindingMap[device.id] || {};
      return {
        deviceId: device.id,
        name: device.name,
        deviceType: device.deviceType,
        communicationType: device.communicationType,
        quantity: binding.quantity != null ? Math.max(1, integerValue(binding.quantity, device.quantity || 1)) : Math.max(1, integerValue(device.quantity, 1)),
        maxThroughput: numberValue(device.maxThroughput, 0),
        maxUsers: integerValue(device.maxUsers, 0),
        enabled: binding.enabled != null ? binding.enabled !== false : device.enabled !== false,
        applied: Boolean(binding.applied),
        x: integerValue(binding.x, 0),
        y: integerValue(binding.y, 0),
        status: device.status || "已导入"
      };
    });
  }

  function saveScenarioDeviceRows() {
    var bindings = readStorage(DEVICE_BINDINGS_KEY, {});
    bindings[state.scenarioName] = state.scenarioDeviceRows.map(function (row) {
      return {
        deviceId: row.deviceId,
        baseStationName: row.baseStationName || null,
        mode: row.mode || null,
        name: row.name,
        quantity: Math.max(1, integerValue(row.quantity, 1)),
        enabled: row.enabled !== false,
        applied: Boolean(row.applied),
        x: integerValue(row.x, 0),
        y: integerValue(row.y, 0)
      };
    });
    writeStorage(DEVICE_BINDINGS_KEY, bindings);
  }

  function deviceSummaryLabel() {
    var active = state.scenarioDeviceRows.filter(function (row) {
      return row.applied && row.enabled && Number(row.quantity) > 0;
    });
    if (!active.length) return "未应用设备";
    return active.map(function (row) {
      return (row.name || communicationCategoryLabel(row.communicationType)) + " " + row.quantity + " 台";
    }).join(" / ");
  }

  function updateDeviceSummaryBadge() {
    ["tester-device-summary", "tester-device-inline-summary"].forEach(function (id) {
      var summary = byId(id);
      if (summary) {
        summary.textContent = "设备接入：" + deviceSummaryLabel();
      }
    });
  }

  function syncDeviceRowsFromStorage() {
    ensureDeviceLibrarySeeded();
    loadScenarioDeviceRows();
    updateDeviceSummaryBadge();
    renderTesterDeviceAccessModule();
  }

  function readTestHistory() {
    try {
      var raw = window.localStorage.getItem(TEST_HISTORY_KEY);
      var parsed = raw ? JSON.parse(raw) : [];
      return Array.isArray(parsed) ? parsed : [];
    } catch (error) {
      return [];
    }
  }

  function writeTestHistory(items) {
    try {
      window.localStorage.setItem(TEST_HISTORY_KEY, JSON.stringify(items));
    } catch (error) {
      appendTerminalLine("测试记录写入失败：" + (error && error.message ? error.message : error), "warning");
    }
  }

  function finalBroadcastRatio(result) {
    var reports = result && Array.isArray(result.reports) ? result.reports : [];
    var finalState = reports.length ? reports[0].final_state || {} : {};
    var ratio = Number(finalState.broadcast_ratio);
    return isFinite(ratio) ? ratio : null;
  }

  function buildHistoryEntry(result) {
    var reports = result && Array.isArray(result.reports) ? result.reports : [];
    var finalState = reports.length ? reports[0].final_state || {} : {};
    var userDetails = Array.isArray(finalState.user_details) ? finalState.user_details : [];
    return {
      id: "test-" + Date.now() + "-" + Math.random().toString(16).slice(2, 8),
      createdAt: Date.now(),
      scenarioName: state.scenarioName,
      scenarioLabel: scenarioLabel(state.scenarioName),
      algorithm: state.algorithm,
      algorithmLabel: algorithmLabel(state.algorithm),
      checkpointPath: state.checkpointPath,
      avgReward: Number(result && result.avg_reward),
      avgFinalCoverage: Number(result && result.avg_final_coverage),
      broadcastRatio: finalBroadcastRatio(result),
      disasterScenePath: result && result.scene_export ? result.scene_export.disaster_scene_path : "",
      deploymentScenePath: result && result.scene_export ? result.scene_export.deployment_scene_path : "",
      userCount: userDetails.length,
      deviceRows: userDetails.slice(0, 64).map(function (device) {
        return {
          id: device.id,
          position: Array.isArray(device.position) ? device.position.slice(0, 2) : [],
          region_label: device.region_label || "",
          demand: Number(device.demand || 0),
          connected: Boolean(device.connected),
          broadcast_served: Boolean(device.broadcast_served)
        };
      })
    };
  }

  function persistTestHistory(result) {
    var entry = buildHistoryEntry(result);
    var history = readTestHistory().filter(function (item) {
      return item && item.id !== entry.id;
    });
    history.unshift(entry);
    writeTestHistory(history.slice(0, 40));
  }

  function setTabVisual(activeKey) {
    activeKey = activeKey === "deployment" && hasDeploymentScene() ? "deployment" : "imported";
    state.activeSceneTab = activeKey;
    var imported = byId("u2844");
    var importedImg = byId("u2844_img");
    var deployment = byId("u2843");
    var deploymentImg = byId("u2843_img");
    setDeploymentTabVisible(hasDeploymentScene());

    if (imported) imported.classList.toggle("selected", activeKey === "imported");
    if (deployment) deployment.classList.toggle("selected", activeKey === "deployment");
    if (importedImg) {
      importedImg.classList.toggle("selected", activeKey === "imported");
      importedImg.src = activeKey === "imported" ? "images/模型训练/u781_selected.png" : "images/模型训练/u781.png";
    }
    if (deploymentImg) {
      deploymentImg.classList.toggle("selected", activeKey === "deployment");
      deploymentImg.src = activeKey === "deployment" ? "images/策略测试/u2843_selected.png" : "images/策略测试/u2843.png";
    }
    setPrototypeMapPanelState(activeKey);
    renderSceneGraphOverlay(activeKey);
  }

  function hasDeploymentScene() {
    return Boolean(state.simulationResult &&
      state.simulationResult.scene_export &&
      state.simulationResult.scene_export.deployment_scene);
  }

  function setDeploymentTabVisible(visible) {
    var tab = byId("u2843");
    if (!tab) return;
    tab.style.display = visible ? "block" : "none";
    tab.style.visibility = visible ? "visible" : "hidden";
    tab.style.pointerEvents = visible ? "auto" : "none";
  }

  function setPrototypeMapPanelState(activeKey) {
    var isDeployment = activeKey === "deployment";
    [
      { imported: "u2695_state0", deployment: "u2695_state1" },
      { imported: "u3042_state1", deployment: "u3042_state0" }
    ].forEach(function (group) {
      var imported = byId(group.imported);
      var deployment = byId(group.deployment);
      if (imported) {
        imported.style.display = isDeployment ? "none" : "block";
        imported.style.visibility = isDeployment ? "hidden" : "visible";
      }
      if (deployment) {
        deployment.style.display = isDeployment ? "block" : "none";
        deployment.style.visibility = isDeployment ? "visible" : "hidden";
      }
    });
  }

  function activeScenePayload(activeKey) {
    if (activeKey === "deployment") {
      return state.simulationResult &&
        state.simulationResult.scene_export &&
        state.simulationResult.scene_export.deployment_scene
        ? state.simulationResult.scene_export.deployment_scene
        : null;
    }
    return state.importedScene && state.importedScene.scene
      ? state.importedScene.scene
      : state.simulationResult && state.simulationResult.scene_export
        ? state.simulationResult.scene_export.disaster_scene
        : null;
  }

  function sceneNodes(scene) {
    return scene && Array.isArray(scene.nodes) ? scene.nodes : [];
  }

  function stationNodeCount(scene) {
    return sceneNodes(scene).filter(function (node) {
      return node && node.type !== "USER";
    }).length;
  }

  function sceneSummary(scene, activeKey) {
    var stats = sceneStats(scene);
    return (activeKey === "deployment" ? "部署后场景" : "导入的场景") +
      "：用户 " + stats.userCount +
      "，站点 " + stats.stationCount +
      "，覆盖 " + stats.coverage +
      "，广播 " + stats.broadcast;
  }

  function sceneStats(scene) {
    var nodes = sceneNodes(scene);
    var userCount = 0;
    var stationCount = 0;
    var connectedCount = 0;
    var broadcastCount = 0;
    var stationTypes = {};
    nodes.forEach(function (node) {
      if (!node) return;
      if (node.type === "USER") {
        userCount += 1;
        if (node.connected) connectedCount += 1;
        if (node.broadcast_served) broadcastCount += 1;
      } else {
        stationCount += 1;
        var key = node.base_station || node.label || node.type || "UNKNOWN";
        if (!stationTypes[key]) {
          stationTypes[key] = {
            key: key,
            label: node.label || stationLabelByName(key),
            count: 0,
            color: stationColorForName(key)
          };
        }
        stationTypes[key].count += 1;
      }
    });
    return {
      userCount: userCount,
      stationCount: stationCount,
      connectedCount: connectedCount,
      broadcastCount: broadcastCount,
      stationTypes: Object.keys(stationTypes).map(function (key) { return stationTypes[key]; }),
      coverage: userCount ? formatPercent(connectedCount / userCount) : "--",
      broadcast: userCount ? formatPercent(broadcastCount / userCount) : "--"
    };
  }

  function stationLabelByName(name) {
    var station = scenarioBaseStationByName(name);
    return station ? (station.label || station.name) : (name || "未知基站");
  }

  function visibleSceneNodes(nodes) {
    var users = nodes.filter(function (node) { return node && node.type === "USER"; });
    var stations = nodes.filter(function (node) { return node && node.type !== "USER"; });
    var maxUsers = 260;
    if (users.length <= maxUsers) return users.concat(stations);
    var step = Math.max(1, Math.ceil(users.length / maxUsers));
    return users.filter(function (_, index) {
      return index % step === 0;
    }).slice(0, maxUsers).concat(stations);
  }

  function sceneNodeStyle(node, isDeployedStation) {
    if (!node || node.type === "USER") {
      if (node && node.connected) {
        return { fill: "#38bdf8", stroke: "rgba(219,234,254,0.92)", radius: 4.2, opacity: 0.95 };
      }
      if (node && node.broadcast_served) {
        return { fill: "#facc15", stroke: "rgba(254,249,195,0.9)", radius: 4.2, opacity: 0.95 };
      }
      return { fill: "#ef4444", stroke: "rgba(254,226,226,0.84)", radius: 3.8, opacity: 0.9 };
    }
    var stationColor = stationColorForNode(node);
    if (isDeployedStation) {
      return { fill: stationColor, stroke: "rgba(220,252,231,0.94)", radius: 9.2, opacity: 1 };
    }
    return { fill: stationColor, stroke: "rgba(255,255,255,0.88)", radius: 8, opacity: 1 };
  }

  function stationColorForName(baseStationName) {
    var stations = scenarioBaseStations();
    var index = stations.findIndex(function (station) {
      return station.name === baseStationName;
    });
    if (index >= 0) return STATION_COLORS[index % STATION_COLORS.length];
    var text = String(baseStationName || "");
    var hash = 0;
    for (var i = 0; i < text.length; i += 1) {
      hash = (hash * 31 + text.charCodeAt(i)) % 9973;
    }
    return STATION_COLORS[hash % STATION_COLORS.length];
  }

  function stationColorForNode(node) {
    var key = node && (node.base_station || node.label || node.type);
    return stationColorForName(key);
  }

  function scaleSceneNode(node, scene, width, height) {
    var geoPoint = scaleSceneGeoNode(node, scene, width, height);
    if (geoPoint) return geoPoint;

    var x = Number(node && node.x);
    var y = Number(node && node.y);
    if (!isFinite(x) || !isFinite(y)) return null;
    var extent = sceneExtent(scene);
    var spanX = Math.max(1, extent.maxX - extent.minX);
    var spanY = Math.max(1, extent.maxY - extent.minY);
    var padding = Math.max(64, Math.round(Math.min(width, height) * 0.14));
    var availableWidth = Math.max(1, width - padding * 2);
    var availableHeight = Math.max(1, height - padding * 2);
    var scale = Math.min(availableWidth / spanX, availableHeight / spanY);
    var scaledWidth = spanX * scale;
    var scaledHeight = spanY * scale;
    var originX = (width - scaledWidth) / 2;
    var originY = (height - scaledHeight) / 2;
    return {
      x: originX + (x - extent.minX) * scale,
      y: originY + (y - extent.minY) * scale
    };
  }

  function scaleSceneGeoNode(node, scene, width, height) {
    var lat = Number(node && node.lat);
    var lon = Number(node && node.lon);
    if (!isFinite(lat) || !isFinite(lon)) return null;
    var viewport = mapViewport(width, height, scene);
    if (!viewport) return null;
    var point = mercatorProject(lat, lon, viewport.zoom);
    return {
      x: point.x - viewport.left,
      y: point.y - viewport.top
    };
  }

  function sceneExtent(scene) {
    var xs = [];
    var ys = [];
    sceneNodes(scene).forEach(function (node) {
      var x = Number(node && node.x);
      var y = Number(node && node.y);
      if (isFinite(x) && isFinite(y)) {
        xs.push(x);
        ys.push(y);
      }
    });
    if (!xs.length || !ys.length) {
      return { minX: 0, maxX: 1, minY: 0, maxY: 1 };
    }
    var minX = Math.min.apply(Math, xs);
    var maxX = Math.max.apply(Math, xs);
    var minY = Math.min.apply(Math, ys);
    var maxY = Math.max.apply(Math, ys);
    if (maxX === minX) {
      minX -= 1;
      maxX += 1;
    }
    if (maxY === minY) {
      minY -= 1;
      maxY += 1;
    }
    return { minX: minX, maxX: maxX, minY: minY, maxY: maxY };
  }

  function buildRestorationLinks(nodes, scene, width, height) {
    var users = nodes.filter(function (node) {
      return node.type === "USER" && (node.connected || node.broadcast_served);
    });
    var stations = nodes.filter(function (node) { return node.type !== "USER"; });
    if (!users.length || !stations.length) return "";

    var scaledUsers = users.map(function (node) {
      var point = scaleSceneNode(node, scene, width, height);
      return point ? { node: node, x: point.x, y: point.y } : null;
    }).filter(Boolean);
    var scaledStations = stations.map(function (node) {
      var point = scaleSceneNode(node, scene, width, height);
      return point ? { node: node, x: point.x, y: point.y } : null;
    }).filter(Boolean);
    if (!scaledUsers.length || !scaledStations.length) return "";

    var links = [];
    scaledUsers.slice(0, 150).forEach(function (user) {
      var best = null;
      var bestDistance = Infinity;
      scaledStations.forEach(function (station) {
        var dx = user.x - station.x;
        var dy = user.y - station.y;
        var distance = dx * dx + dy * dy;
        if (distance < bestDistance) {
          bestDistance = distance;
          best = station;
        }
      });
      if (best) {
        links.push("<line x1='" + best.x.toFixed(1) + "' y1='" + best.y.toFixed(1) +
          "' x2='" + user.x.toFixed(1) + "' y2='" + user.y.toFixed(1) +
          "' stroke='rgba(56,189,248,0.28)' stroke-width='1.2' />");
      }
    });
    return links.join("");
  }

  function renderSceneGraphOverlay(activeKey) {
    var key = activeKey || state.activeSceneTab || "imported";
    applyScenarioMapSkin(key);
    hidePrototypeMapArtifacts();
    renderSceneGraphInto("u2692_state0_content", key);
    renderSceneGraphInto("u3039_state0_content", key);
    renderSceneLegend(activeScenePayload(key), key);
  }

  function renderSceneGraphInto(hostId, activeKey) {
    var host = byId(hostId);
    if (!host) return;
    host.style.pointerEvents = "none";
    var overlayId = "tester-scene-graph-overlay-" + hostId;
    var overlay = byId(overlayId);
    if (!overlay) {
      overlay = document.createElement("div");
      overlay.id = overlayId;
      host.appendChild(overlay);
    }
    var width = 1618;
    var height = 1090;
    overlay.style.cssText = [
      "position:absolute",
      "left:1px",
      "top:53px",
      "width:" + width + "px",
      "height:" + height + "px",
      "z-index:90",
      "pointer-events:none",
      "overflow:hidden",
      "font-family:'Microsoft YaHei','PingFang SC',sans-serif"
    ].join(";");

    var scene = activeScenePayload(activeKey);
    if (!scene) {
      overlay.innerHTML = "";
      return;
    }

    var rawNodes = sceneNodes(scene);
    var nodes = visibleSceneNodes(rawNodes);
    var importedStationCount = stationNodeCount(state.importedScene && state.importedScene.scene);
    var stationIndex = 0;
    var nodeMarkup = nodes.map(function (node) {
      var point = scaleSceneNode(node, scene, width, height);
      if (!point) return "";
      var isStation = node.type !== "USER";
      var isDeployedStation = false;
      if (isStation) {
        isDeployedStation = activeKey === "deployment" && stationIndex >= importedStationCount;
        stationIndex += 1;
      }
      var style = sceneNodeStyle(node, isDeployedStation);
      var pulse = isDeployedStation
        ? "<circle cx='" + point.x.toFixed(1) + "' cy='" + point.y.toFixed(1) + "' r='" + (style.radius + 6) + "' fill='none' stroke='rgba(34,197,94,0.36)' stroke-width='2' />"
        : "";
      return pulse +
        "<circle cx='" + point.x.toFixed(1) + "' cy='" + point.y.toFixed(1) + "' r='" + (style.radius + 3) + "' fill='" + style.fill + "' opacity='0.16' />" +
        "<circle cx='" + point.x.toFixed(1) + "' cy='" + point.y.toFixed(1) + "' r='" + style.radius + "' fill='" + style.fill + "' stroke='" + style.stroke + "' stroke-width='1.5' opacity='" + style.opacity + "' />";
    }).join("");

    var linkMarkup = activeKey === "deployment"
      ? buildRestorationLinks(nodes, scene, width, height)
      : "";
    overlay.innerHTML =
      "<svg width='" + width + "' height='" + height + "' viewBox='0 0 " + width + " " + height + "' preserveAspectRatio='none' style='position:absolute;left:0;top:0;'>" +
      linkMarkup +
      nodeMarkup +
      "</svg>";
  }

  function hidePrototypeMapArtifacts() {
    ["u2694", "u2695", "u3041", "u3042"].forEach(function (id) {
      var node = byId(id);
      if (node) {
        node.style.display = "none";
        node.style.visibility = "hidden";
      }
    });
  }

  function renderSceneLegend(scene, activeKey) {
    for (var id = 2816; id <= 2833; id += 1) {
      var item = byId("u" + id);
      if (item) {
        item.style.display = "none";
        item.style.visibility = "hidden";
      }
    }

    var host = byId("u2815");
    if (!host) return;
    host.style.position = "absolute";
    host.style.zIndex = "130";
    host.style.pointerEvents = "none";
    var panel = byId("tester-scene-legend-panel");
    if (!panel) {
      panel = document.createElement("div");
      panel.id = "tester-scene-legend-panel";
      host.appendChild(panel);
    }
    panel.style.cssText = [
      "position:absolute",
      "left:14px",
      "top:14px",
      "width:182px",
      "height:268px",
      "box-sizing:border-box",
      "font-family:'Microsoft YaHei','PingFang SC',sans-serif",
      "color:#fff",
      "font-size:13px",
      "line-height:1.5",
      "pointer-events:none"
    ].join(";");

    if (!scene) {
      panel.innerHTML =
        "<div style='font-size:16px;font-weight:700;margin-bottom:10px;'>场景图例</div>" +
        "<div style='color:rgba(255,255,255,0.78);'>正在等待真实场景数据</div>";
      return;
    }

    var stats = sceneStats(scene);
    var title = activeKey === "deployment" ? "部署后场景" : "导入场景";
    var stationLegend = stats.stationTypes.length
      ? stats.stationTypes.map(function (item) {
        return legendRow(item.color, item.label + " " + item.count);
      }).join("")
      : "<div style='margin-top:8px;color:rgba(255,255,255,0.72);'>暂无基站节点</div>";
    panel.innerHTML =
      "<div style='font-size:16px;font-weight:700;margin-bottom:8px;'>" + title + "</div>" +
      "<div style='display:grid;grid-template-columns:1fr 1fr;gap:4px 8px;margin-bottom:12px;color:rgba(255,255,255,0.86);'>" +
      "<span>用户 " + escapeHtml(stats.userCount) + "</span>" +
      "<span>站点 " + escapeHtml(stats.stationCount) + "</span>" +
      "<span>覆盖 " + escapeHtml(stats.coverage) + "</span>" +
      "<span>广播 " + escapeHtml(stats.broadcast) + "</span>" +
      "</div>" +
      legendRow("#ef4444", "断联用户节点") +
      legendRow("#38bdf8", "恢复/正常用户") +
      stationLegend +
      "<div style='display:flex;align-items:center;gap:10px;margin-top:8px;'><i style='display:inline-block;width:28px;height:2px;background:rgba(56,189,248,0.72);'></i><span>通信链路</span></div>";
  }

  function legendRow(color, label) {
    return "<div style='display:flex;align-items:center;gap:10px;margin-top:8px;'>" +
      "<i style='display:inline-block;width:12px;height:12px;border-radius:50%;background:" + color + ";box-shadow:0 0 0 4px rgba(255,255,255,0.10);'></i>" +
      "<span>" + label + "</span>" +
      "</div>";
  }

  function setStatus(stateText, tone) {
    setText("u2855_text", stateText);
    var pill = byId("tester-live-status");
    if (pill) {
      pill.textContent = stateText;
      pill.dataset.tone = tone || "idle";
    }
  }

  function setStartButton(text, tone, disabled) {
    setText("u2834_text", text);
    var button = byId("u2834");
    var buttonDiv = byId("u2834_div");
    if (button) {
      button.style.cursor = disabled ? "not-allowed" : "pointer";
      button.style.pointerEvents = "auto";
      button.style.opacity = disabled ? "0.55" : "1";
    }
    if (buttonDiv) {
      var colors = {
        idle: "#03b4f5",
        running: "#1d4ed8",
        success: "#22c55e",
        error: "#ef4444",
        disabled: "#94a3b8"
      };
      buttonDiv.style.background = colors[tone] || colors.idle;
      buttonDiv.style.borderColor = "transparent";
      buttonDiv.style.boxShadow = tone === "running" ? "0 12px 24px rgba(29, 78, 216, 0.28)" : "0 10px 22px rgba(3, 180, 245, 0.24)";
    }
  }

  function currentScenarioGridBounds() {
    var scenario = currentScenario();
    var rows = scenario && scenario.region_grid && scenario.region_grid.rows
      ? Number(scenario.region_grid.rows)
      : Number(scenario && scenario.grid_size);
    var cols = scenario && scenario.region_grid && scenario.region_grid.cols
      ? Number(scenario.region_grid.cols)
      : Number(scenario && scenario.grid_size);
    return {
      maxX: Math.max(0, Math.round(rows || 1) - 1),
      maxY: Math.max(0, Math.round(cols || rows || 1) - 1)
    };
  }

  function clampGridIndex(value, max, fallback) {
    var next = integerValue(value, fallback == null ? 0 : fallback);
    return Math.max(0, Math.min(Math.max(0, max), next));
  }

  function defaultResidualGridPosition() {
    var bounds = currentScenarioGridBounds();
    var fallback = {
      x: Math.round(bounds.maxX / 2),
      y: Math.round(bounds.maxY / 2)
    };
    var scenario = currentScenario();
    var clusters = scenario && Array.isArray(scenario.user_clusters) ? scenario.user_clusters : [];
    var bestCluster = null;
    var bestScore = -Infinity;
    clusters.forEach(function (cluster) {
      if (!cluster || !Array.isArray(cluster.center) || cluster.center.length < 2) return;
      var density = numberValue(cluster.density, 0);
      var demand = numberValue(cluster.demand_mbps, 1);
      var score = density * Math.max(1, demand);
      if (score > bestScore) {
        bestScore = score;
        bestCluster = cluster;
      }
    });
    if (bestCluster) {
      return {
        x: clampGridIndex(bestCluster.center[0], bounds.maxX, fallback.x),
        y: clampGridIndex(bestCluster.center[1], bounds.maxY, fallback.y)
      };
    }
    var preview = scenario && Array.isArray(scenario.candidate_site_preview) ? scenario.candidate_site_preview : [];
    var candidate = preview.find(function (site) { return site.category === "核心覆盖"; }) || preview[0];
    if (candidate) {
      return {
        x: clampGridIndex(candidate.x, bounds.maxX, fallback.x),
        y: clampGridIndex(candidate.y, bounds.maxY, fallback.y)
      };
    }
    return fallback;
  }

  function isScenarioDeviceSupported(row) {
    var scenario = currentScenario();
    if (!scenario) return true;
    if (row && row.baseStationName) {
      return Boolean(scenarioBaseStationByName(row.baseStationName));
    }
    var baseStations = Array.isArray(scenario.base_stations) ? scenario.base_stations : [];
    return Boolean(matchBaseStationForCommunicationType(baseStations, row && row.communicationType));
  }

  function activeScenarioDeviceRows() {
    return state.scenarioDeviceRows
      .map(function (row, index) {
        return { row: row, index: index };
      })
      .filter(function (entry) {
        return entry.row && entry.row.applied && entry.row.enabled !== false;
      });
  }

  function supportedScenarioDeviceRows() {
    return state.scenarioDeviceRows.filter(function (row) {
      return row && isScenarioDeviceSupported(row);
    });
  }

  function testerDeviceOptions(currentDeviceId) {
    var supported = supportedScenarioDeviceRows();
    var source = supported.length ? supported : state.scenarioDeviceRows;
    if (currentDeviceId && !source.some(function (row) { return row.deviceId === currentDeviceId; })) {
      var current = state.scenarioDeviceRows.find(function (row) {
        return row.deviceId === currentDeviceId;
      });
      if (current) source = [current].concat(source);
    }
    return source;
  }

  function supportedCommunicationText() {
    var baseStations = scenarioBaseStations();
    if (baseStations.length) {
      return baseStations.map(function (station) {
        return station.label || station.name;
      }).join(" / ");
    }
    var supported = COMM_TYPES.filter(function (type) {
      return state.scenarioDeviceRows.some(function (row) {
        return row.communicationType === type.value && isScenarioDeviceSupported(row);
      });
    });
    if (!supported.length) return "等待场景加载";
    return supported.map(function (item) { return item.label.replace(" 通信", ""); }).join(" / ");
  }

  function persistDeviceLibraryRow(row) {
    if (!row || !row.deviceId) return;
    var library = loadDeviceLibrary();
    if (!library.some(function (item) { return item.id === row.deviceId; })) {
      library.push({
        id: row.deviceId,
        name: row.name,
        deviceType: row.deviceType,
        communicationType: row.communicationType,
        quantity: Math.max(1, integerValue(row.quantity, 1)),
        maxThroughput: numberValue(row.maxThroughput, 0),
        maxUsers: integerValue(row.maxUsers, 0),
        enabled: row.enabled !== false,
        status: row.status || "已导入"
      });
      writeStorage(DEVICE_LIBRARY_KEY, library);
    }
  }

  function ensureToolbarDeviceEntry() {
    var anchor = byId("u2834");
    var host = anchor && anchor.parentElement ? anchor.parentElement : document.body;
    if (!host) return;

    var button = byId("tester-device-entry");
    if (button && button.parentElement !== host) {
      button.remove();
      button = null;
    }
    if (!button) {
      button = document.createElement("button");
      button.id = "tester-device-entry";
      button.type = "button";
      button.textContent = "导入设备";
      host.appendChild(button);
    }
    button.style.cssText = [
      "position:absolute",
      "left:248px",
      "top:2px",
      "width:95px",
      "height:40px",
      "display:flex",
      "align-items:center",
      "justify-content:center",
      "white-space:nowrap",
      "padding:0 12px",
      "border:1px solid #b7e0fe",
      "border-radius:10px",
      "background:#3961f6",
      "color:#ffffff",
      "font-size:16px",
      "line-height:1",
      "font-family:'思源黑体 CN Regular','思源黑体 CN',sans-serif",
      "cursor:pointer",
      "z-index:30",
      "pointer-events:auto"
    ].join(";");

    var summary = byId("tester-device-summary");
    if (summary && summary.parentElement !== host) {
      summary.remove();
      summary = null;
    }
    if (!summary) {
      summary = document.createElement("div");
      summary.id = "tester-device-summary";
      host.appendChild(summary);
    }
    summary.style.cssText = [
      "position:absolute",
      "left:358px",
      "top:10px",
      "width:235px",
      "font-size:12px",
      "line-height:1.6",
      "color:#64748b",
      "z-index:30",
      "white-space:nowrap",
      "overflow:hidden",
      "text-overflow:ellipsis",
      "pointer-events:none"
    ].join(";");
    updateDeviceSummaryBadge();
  }

  function ensureTesterDeviceAccessModule() {
    var host = byId("u2852_state0_content");
    if (!host) return null;
    host.style.position = "relative";
    host.style.pointerEvents = "auto";
    var module = byId("tester-device-access-module");
    if (!module) {
      module = document.createElement("div");
      module.id = "tester-device-access-module";
      host.appendChild(module);
    }
    module.style.cssText = [
      "position:absolute",
      "left:8px",
      "top:122px",
      "width:1588px",
      "min-height:116px",
      "box-sizing:border-box",
      "z-index:28",
      "font-size:13px",
      "font-family:'Microsoft YaHei','PingFang SC',sans-serif",
      "color:#334155",
      "pointer-events:auto"
    ].join(";");
    return module;
  }

  function storeBaseTop(node) {
    if (!node) return 0;
    if (node.dataset.testerBaseTop == null) {
      var computed = window.getComputedStyle ? window.getComputedStyle(node) : null;
      var top = parseFloat(node.style.top || (computed && computed.top) || "0");
      node.dataset.testerBaseTop = String(Number.isFinite(top) ? top : 0);
    }
    return Number(node.dataset.testerBaseTop) || 0;
  }

  function shouldShiftResultNode(node) {
    if (!node || !node.id) return false;
    if (node.id === "tester-device-access-module" || node.id === "u2856") return false;
    if (node.closest && (node.closest("#tester-device-access-module") || node.closest("#u2856"))) return false;
    if (node.id !== "u2857" && node.closest && node.closest("#u2857")) return false;
    return true;
  }

  function relayoutTesterResultPanel() {
    var content = byId("u2852_state0_content");
    var module = byId("tester-device-access-module");
    if (!content || !module) return;
    var moduleHeight = Math.max(116, module.scrollHeight || module.offsetHeight || 116);
    module.style.height = moduleHeight + "px";

    var terminal = byId("u2856");
    var terminalBaseTop = terminal ? storeBaseTop(terminal) : 122;
    var delta = moduleHeight + 16;
    if (terminal) {
      terminal.style.top = (terminalBaseTop + delta) + "px";
    }

    Array.prototype.forEach.call(content.querySelectorAll("[id]"), function (node) {
      if (!shouldShiftResultNode(node)) return;
      var baseTop = storeBaseTop(node);
      if (baseTop >= 512) {
        node.style.top = (baseTop + delta) + "px";
      }
    });

    var statePanel = byId("u2852_state0");
    var resultPanel = byId("u2852");
    var height = 1388 + delta;
    if (statePanel) statePanel.style.height = height + "px";
    if (resultPanel) resultPanel.style.height = height + "px";
    if (document.body && resultPanel) {
      document.body.style.minHeight = Math.max(document.body.scrollHeight || 0, resultPanel.offsetTop + height + 48) + "px";
    }
  }

  function renderTesterDeviceAccessModule() {
    var module = ensureTesterDeviceAccessModule();
    if (!module) return;
    if (!Array.isArray(state.scenarioDeviceRows) || !state.scenarioDeviceRows.length) {
      ensureDeviceLibrarySeeded();
      loadScenarioDeviceRows();
    }
    var bounds = currentScenarioGridBounds();
    var rows = activeScenarioDeviceRows();
    var gridColumns = "44px minmax(260px,1.2fr) 140px minmax(210px,1fr) 92px 92px 92px 112px 76px";
    var body = rows.length ? rows.map(function (entry, order) {
      var row = entry.row;
      var index = entry.index;
      var supported = isScenarioDeviceSupported(row);
      var options = testerDeviceOptions(row.deviceId).map(function (option) {
        return "<option value='" + escapeHtml(option.deviceId) + "'" + (option.deviceId === row.deviceId ? " selected" : "") + ">" +
          escapeHtml(option.name) +
        "</option>";
      }).join("");
      var station = row.baseStationName ? scenarioBaseStationByName(row.baseStationName) : null;
      var modeText = row.mode || defaultStationMode(station) || "--";
      var keyParams = row.deviceType + " / " + formatMetric(row.maxThroughput, 0) + "Mbps / " + row.maxUsers + "用户";
      return "<div data-tester-device-row='" + index + "' style='display:grid;grid-template-columns:" + gridColumns + ";align-items:center;gap:10px;min-height:48px;border-top:1px solid #edf2f7;'>" +
        "<div style='text-align:center;color:#64748b;'>" + (order + 1) + "</div>" +
        "<select data-tester-device-field='deviceId' style='width:100%;height:34px;border:1px solid #d7e3f4;border-radius:6px;padding:0 10px;background:#fff;color:#17315d;box-sizing:border-box;'>" + options + "</select>" +
        "<div title='" + escapeHtml(modeText) + "' style='height:34px;line-height:34px;padding:0 10px;border:1px solid #edf2f7;border-radius:6px;background:#f8fafc;color:#475569;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>" + escapeHtml(communicationCategoryLabel(row.communicationType) + " / " + modeText) + "</div>" +
        "<div style='height:34px;line-height:34px;padding:0 10px;border:1px solid #edf2f7;border-radius:6px;background:#f8fafc;color:#475569;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>" + escapeHtml(keyParams) + "</div>" +
        "<input data-tester-device-field='quantity' type='number' min='1' step='1' value='" + escapeHtml(row.quantity) + "' style='width:100%;height:34px;border:1px solid #d7e3f4;border-radius:6px;padding:0 10px;box-sizing:border-box;'>" +
        "<input data-tester-device-field='x' type='number' min='0' max='" + escapeHtml(bounds.maxX) + "' step='1' value='" + escapeHtml(clampGridIndex(row.x, bounds.maxX, 0)) + "' title='x 为区域网格行索引，范围 0-" + escapeHtml(bounds.maxX) + "' style='width:100%;height:34px;border:1px solid #d7e3f4;border-radius:6px;padding:0 10px;box-sizing:border-box;'>" +
        "<input data-tester-device-field='y' type='number' min='0' max='" + escapeHtml(bounds.maxY) + "' step='1' value='" + escapeHtml(clampGridIndex(row.y, bounds.maxY, 0)) + "' title='y 为区域网格列索引，范围 0-" + escapeHtml(bounds.maxY) + "' style='width:100%;height:34px;border:1px solid #d7e3f4;border-radius:6px;padding:0 10px;box-sizing:border-box;'>" +
        "<div style='height:28px;line-height:28px;text-align:center;border-radius:999px;background:" + (supported ? "rgba(34,197,94,0.12)" : "rgba(245,158,11,0.16)") + ";color:" + (supported ? "#15803d" : "#b45309") + ";white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>" + (supported ? "已接入" : "场景不支持") + "</div>" +
        "<button type='button' data-tester-device-action='remove' style='height:32px;border:1px solid #f1c7c7;border-radius:6px;background:#fff;color:#b42318;cursor:pointer;'>移除</button>" +
      "</div>";
    }).join("") : "<div style='height:38px;line-height:38px;color:#64748b;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;border-top:1px solid #edf2f7;'>暂无设备接入，点击“导入设备”或“+ 添加设备”按当前场景默认接入 1 台设备，并配置 x/y 网格位置</div>";

    module.innerHTML =
      "<div style='display:flex;align-items:center;justify-content:space-between;height:38px;'>" +
      "<div style='display:flex;align-items:center;gap:10px;min-width:0;'>" +
      "<div style='width:8px;height:22px;background:#3961f6;flex:0 0 auto;'></div>" +
      "<div style='font-size:16px;color:#111827;font-weight:600;white-space:nowrap;'>设备接入模块</div>" +
      "<div id='tester-device-inline-summary' style='font-size:12px;color:#64748b;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>设备接入：" + escapeHtml(deviceSummaryLabel()) + "</div>" +
      "<div style='font-size:12px;color:#64748b;white-space:nowrap;'>支持：" + escapeHtml(supportedCommunicationText()) + "；x 0-" + escapeHtml(bounds.maxX) + "，y 0-" + escapeHtml(bounds.maxY) + "</div>" +
      "</div>" +
      "<div style='display:flex;align-items:center;gap:8px;'>" +
      "<button type='button' data-tester-device-action='add' style='height:34px;padding:0 16px;border:1px solid #b7e0fe;border-radius:6px;background:#3961f6;color:#fff;cursor:pointer;'>+ 添加设备</button>" +
      "<button type='button' data-tester-device-action='json' style='height:34px;padding:0 14px;border:1px solid #d7e3f4;border-radius:6px;background:#fff;color:#17315d;cursor:pointer;'>导入JSON</button>" +
      "<button type='button' data-tester-device-action='apply' style='height:34px;padding:0 14px;border:1px solid #b7e0fe;border-radius:6px;background:#ebf5ff;color:#2563eb;cursor:pointer;'>应用到当前测试</button>" +
      "</div></div>" +
      "<input id='tester-device-json-importer' type='file' accept='.json,application/json' style='display:none;'>" +
      "<div style='display:grid;grid-template-columns:" + gridColumns + ";align-items:center;gap:10px;height:34px;margin-top:8px;color:#64748b;background:#f8fafc;border:1px solid #edf2f7;border-left:0;border-right:0;'>" +
      "<div style='text-align:center;'>序号</div><div>接入设备</div><div>通信方式</div><div>关键参数</div><div>数量</div><div>x（行）</div><div>y（列）</div><div>状态</div><div>操作</div>" +
      "</div>" +
      body;

    Array.prototype.forEach.call(module.querySelectorAll("input,select,button"), function (node) {
      node.style.pointerEvents = "auto";
    });
    bindTesterDeviceAccessEvents(module);
    updateDeviceSummaryBadge();
    relayoutTesterResultPanel();
  }

  function addTesterDeviceSlot() {
    if (!state.scenarioName) {
      appendTerminalLine("场景配置尚未加载完成，暂不能接入设备。", "warning");
      return;
    }
    if (!Array.isArray(state.scenarioDeviceRows) || !state.scenarioDeviceRows.length) {
      ensureDeviceLibrarySeeded();
      loadScenarioDeviceRows();
    }
    var bounds = currentScenarioGridBounds();
    var defaultPosition = defaultResidualGridPosition();
    var supported = supportedScenarioDeviceRows();
    var source = supported.length ? supported : state.scenarioDeviceRows;
    var template = source.find(function (row) { return !row.applied; }) || source[0];
    if (!template) {
      appendTerminalLine("当前没有可接入设备，请先导入设备 JSON。", "warning");
      return;
    }
    var addedName = template.name;
    if (template.applied) {
      var clone = {
        deviceId: template.deviceId + "-copy-" + Date.now() + "-" + Math.random().toString(16).slice(2, 6),
        baseStationName: template.baseStationName || null,
        mode: template.mode || null,
        name: template.name + " " + (state.scenarioDeviceRows.filter(function (row) { return row.name.indexOf(template.name) === 0; }).length + 1),
        deviceType: template.deviceType,
        communicationType: template.communicationType,
        quantity: 1,
        maxThroughput: template.maxThroughput,
        maxUsers: template.maxUsers,
        enabled: true,
        applied: true,
        x: defaultPosition.x,
        y: defaultPosition.y,
        status: "已导入"
      };
      state.scenarioDeviceRows.push(clone);
      addedName = clone.name;
    } else {
      template.applied = true;
      template.enabled = true;
      template.quantity = 1;
      template.x = clampGridIndex(defaultPosition.x, bounds.maxX, 0);
      template.y = clampGridIndex(defaultPosition.y, bounds.maxY, 0);
      template.status = "已导入";
    }
    saveScenarioDeviceRows();
    renderTesterDeviceAccessModule();
    appendTerminalLine("已接入设备：" + addedName + "，默认数量 1。", "success");
    var module = byId("tester-device-access-module");
    if (module && module.scrollIntoView) {
      module.scrollIntoView({ block: "nearest" });
    }
  }

  function removeTesterDeviceSlot(index) {
    var row = state.scenarioDeviceRows[index];
    if (!row) return;
    row.applied = false;
    saveScenarioDeviceRows();
    renderTesterDeviceAccessModule();
    appendTerminalLine("已移除设备接入：" + row.name + "。", "info");
  }

  function updateTesterDeviceField(index, field) {
    var row = state.scenarioDeviceRows[index];
    if (!row || !field) return;
    var key = field.getAttribute("data-tester-device-field");
    if (key === "deviceId") {
      var targetIndex = state.scenarioDeviceRows.findIndex(function (item) {
        return item.deviceId === field.value;
      });
      if (targetIndex >= 0 && targetIndex !== index) {
        var target = state.scenarioDeviceRows[targetIndex];
        target.applied = true;
        target.enabled = true;
        target.quantity = Math.max(1, integerValue(row.quantity, 1));
        target.x = row.x;
        target.y = row.y;
        row.applied = false;
        saveScenarioDeviceRows();
        renderTesterDeviceAccessModule();
      }
      return;
    }
    if (key === "quantity") {
      row.quantity = Math.max(1, integerValue(field.value, 1));
      field.value = String(row.quantity);
    } else if (key === "x" || key === "y") {
      var bounds = currentScenarioGridBounds();
      var max = key === "x" ? bounds.maxX : bounds.maxY;
      row[key] = clampGridIndex(field.value, max, 0);
      field.value = String(row[key]);
    }
    saveScenarioDeviceRows();
    updateDeviceSummaryBadge();
  }

  function bindTesterDeviceAccessEvents(module) {
    Array.prototype.forEach.call(module.querySelectorAll("[data-tester-device-action]"), function (button) {
      button.addEventListener("click", function (event) {
        event.preventDefault();
        event.stopPropagation();
        var action = button.getAttribute("data-tester-device-action");
        if (action === "add") {
          addTesterDeviceSlot();
          return;
        }
        if (action === "json") {
          var input = byId("tester-device-json-importer");
          if (input) input.click();
          return;
        }
        if (action === "apply") {
          saveScenarioDeviceRows();
          updateDeviceSummaryBadge();
          appendTerminalLine("已应用当前测试场景设备接入配置：" + deviceSummaryLabel() + "。", "success");
          importScene();
          return;
        }
        if (action === "remove") {
          var row = button.closest("[data-tester-device-row]");
          if (row) removeTesterDeviceSlot(Number(row.getAttribute("data-tester-device-row")));
        }
      }, true);
    });

    var importer = byId("tester-device-json-importer");
    if (importer) {
      importer.addEventListener("change", function (event) {
        var file = event.target.files && event.target.files[0];
        if (!file) return;
        var reader = new FileReader();
        reader.onload = function () {
          try {
            var parsed = JSON.parse(String(reader.result || "[]"));
            var devices = Array.isArray(parsed) ? parsed : (Array.isArray(parsed.devices) ? parsed.devices : []);
            persistDeviceLibraryMerge(devices);
            syncDeviceRowsFromStorage();
            appendTerminalLine("已导入设备库 JSON，共 " + devices.length + " 条。", "success");
          } catch (error) {
            window.alert("设备导入失败：" + (error && error.message ? error.message : error));
          }
        };
        reader.readAsText(file, "utf-8");
      });
    }

    Array.prototype.forEach.call(module.querySelectorAll("[data-tester-device-row]"), function (rowNode) {
      var index = Number(rowNode.getAttribute("data-tester-device-row"));
      Array.prototype.forEach.call(rowNode.querySelectorAll("[data-tester-device-field]"), function (field) {
        field.addEventListener("input", function () {
          updateTesterDeviceField(index, field);
        });
        field.addEventListener("change", function () {
          updateTesterDeviceField(index, field);
        });
      });
    });
  }

  function ensureOverlay() {
	    var panel = byId("u2852");
	    var content = byId("u2852_state0_content");
	    if (panel && content) {
	      setPanelVisible("u2852", true);
	      setPanelVisible("u2857", true);
	      content.style.position = "relative";
	      content.style.pointerEvents = "auto";
	      removePrototypeMoreButton();
	      ensureToolbarDeviceEntry();
	      renderTesterDeviceAccessModule();
	      updateSceneExportButtons(state.simulationResult);
	    }
    var terminalHost = byId("u2856");
    if (terminalHost) {
      terminalHost.style.position = "relative";
      terminalHost.style.pointerEvents = "auto";
      var terminal = byId("tester-live-terminal");
      if (!terminal) {
        terminal = document.createElement("div");
        terminal.id = "tester-live-terminal";
        terminal.style.cssText = [
          "position:absolute",
          "left:12px",
          "top:12px",
          "right:12px",
          "bottom:12px",
          "overflow:auto",
          "color:#dbeafe",
          "font-family:Consolas,Monaco,monospace",
          "font-size:12px",
          "line-height:1.7",
          "text-align:left",
          "white-space:pre-wrap",
          "word-break:break-word",
          "pointer-events:auto"
        ].join(";");
        terminalHost.appendChild(terminal);
      }
    }

    ["u2999", "u3002", "u3005"].forEach(function (id) {
      var node = byId(id);
      if (!node || node.dataset.liveBound) return;
      node.dataset.liveBound = "true";
      node.style.pointerEvents = "auto";
      node.style.cursor = "pointer";
      node.addEventListener("click", function (event) {
        event.preventDefault();
        event.stopPropagation();
        if (id === "u3005") {
          if (window.top) window.top.location.hash = "/replay";
          else window.location.hash = "/replay";
          return;
        }
        if (!state.simulationResult || !state.simulationResult.scene_export) {
          appendTerminalLine("当前没有可导出的真实测试场景文件。", "warning");
          return;
        }
        var exportKey = id === "u2999" ? "disaster_scene" : "deployment_scene";
        if (!sceneExportHasFile(state.simulationResult.scene_export, exportKey)) {
          appendTerminalLine("当前没有可下载的" + (exportKey === "disaster_scene" ? "受灾场景" : "部署后场景") + "文件。", "warning");
          return;
        }
        var fileName = (state.scenarioName || "scenario") + (exportKey === "disaster_scene" ? "_disaster_scene.json" : "_deployment_scene.json");
        var payload = state.simulationResult.scene_export[exportKey];
        var blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
        var url = URL.createObjectURL(blob);
        var link = document.createElement("a");
        link.href = url;
        link.download = fileName;
        link.click();
        URL.revokeObjectURL(url);
      }, true);
    });
  }

  function appendTerminalLine(message, tone) {
    if (!message) return;
    var terminal = byId("tester-live-terminal");
    if (!terminal) return;
    var time = new Date().toLocaleTimeString("zh-CN", { hour12: false });
    var colors = {
      info: "#7dd3fc",
      success: "#4ade80",
      warning: "#fbbf24",
      error: "#f87171",
      status: "#e2e8f0"
    };
    var line = document.createElement("div");
    line.style.cssText = "padding:1px 0;white-space:pre-wrap;word-break:break-word;text-align:left;";
    line.innerHTML =
      '<span style="color:#64748b">[' + escapeHtml(time) + ']</span> ' +
      '<span style="color:' + (colors[tone] || "#cbd5e1") + '">[' + escapeHtml(tone || "info") + "]</span> " +
      '<span style="color:#dbeafe">' + escapeHtml(message) + "</span>";
    terminal.appendChild(line);
    terminal.scrollTop = terminal.scrollHeight;
    while (terminal.childNodes.length > 240) {
      terminal.removeChild(terminal.firstChild);
    }
  }

  function clearTerminal() {
    var terminal = byId("tester-live-terminal");
    if (terminal) {
      terminal.innerHTML = "";
    }
  }

  function updateSummaryLine(message) {
    return message;
  }

  function updateMetrics(result) {
    setPanelVisible("u2852", true);
    setPanelVisible("u2857", true);
    var reports = result && Array.isArray(result.reports) ? result.reports : [];
    var finalState = reports.length ? reports[0].final_state || {} : {};
    var deviceRows = Array.isArray(finalState.user_details) ? finalState.user_details : [];
    var totalReward = reports.length && typeof reports[0].total_reward === "number" ? reports[0].total_reward : (result ? result.avg_reward : null);
    var finalCoverage = typeof finalState.coverage_ratio === "number" ? finalState.coverage_ratio : (result ? result.avg_final_coverage : null);
    var finalBroadcast = typeof finalState.broadcast_ratio === "number" ? finalState.broadcast_ratio : null;
    var remainingBudget = typeof finalState.remaining_budget === "number" ? finalState.remaining_budget : null;

    setText("u2955_text", result ? formatNumber(result.avg_reward, 2) : "--");
    setPercentText("u2962_text", result ? result.avg_final_coverage : null);
    setText("u2969_text", totalReward != null ? formatNumber(totalReward, 2) : "--");
    setPercentText("u2976_text", finalCoverage);
    setPercentText("u2983_text", finalBroadcast);
    setText("u2990_text", remainingBudget != null ? formatNumber(remainingBudget, 1) : "--");

    if (result && result.scene_export) {
      setText("u2997_text", "受灾场景文件：" + (result.scene_export.disaster_scene_path || "--"));
      setText("u2998_text", "部署后场景文件：" + (result.scene_export.deployment_scene_path || "--"));
    } else {
      setText("u2997_text", "受灾场景文件：--");
      setText("u2998_text", "部署后场景文件：--");
    }
    updateSceneExportButtons(result);

    var rowIds = {
      id: ["u2883","u2884","u2885","u2886","u2887","u2888","u2889","u2890","u2891","u2892"],
      type: ["u2895","u2896","u2897","u2898","u2899","u2900","u2901","u2902","u2903","u2904"],
      location: ["u2907","u2908","u2909","u2910","u2911","u2912","u2913","u2914","u2915","u2916"],
      demand: ["u2871","u2872","u2873","u2874","u2875","u2876","u2877","u2878","u2879","u2880"],
      status: ["u2919","u2920","u2921","u2922","u2923","u2924","u2925","u2926","u2927","u2928"],
      broadcast: ["u2931","u2932","u2933","u2934","u2935","u2936","u2937","u2938","u2939","u2940"]
    };

    for (var i = 0; i < 10; i += 1) {
      var item = deviceRows[i];
      var position = item && Array.isArray(item.position) ? item.position.join(", ") : "--";
      var locationText = item ? position + (item.region_label ? " / " + item.region_label : "") : "--";
      setText(rowIds.id[i] + "_text", item ? String(item.id) : "--");
      setText(rowIds.type[i] + "_text", item ? "终端用户" : "--");
      setText(rowIds.location[i] + "_text", locationText);
      setText(rowIds.demand[i] + "_text", item ? formatNumber(Number(item.demand || 0), 1) + " Mbps" : "--");
      setText(rowIds.status[i] + "_text", item ? (item.connected ? "在线" : "离线") : "--");
      setText(rowIds.broadcast[i] + "_text", item ? (item.broadcast_served ? "已覆盖" : "未覆盖") : "--");
    }

    setText("u3014_text", String(finalState.total_users || state.importedScene && state.importedScene.initial_state && state.importedScene.initial_state.total_users || 0));
    setText("u3018_text", String(finalState.total_users || state.importedScene && state.importedScene.initial_state && state.importedScene.initial_state.total_users || 0));
    setText("u3022_text", String((finalState.residual_base_stations || state.importedScene && state.importedScene.initial_state && state.importedScene.initial_state.residual_base_stations || []).length || 0));
  }

  function normalizeGeoBounds(bounds) {
    if (!bounds) return null;
    var latMin = Number(bounds.lat_min);
    var latMax = Number(bounds.lat_max);
    var lonMin = Number(bounds.lon_min);
    var lonMax = Number(bounds.lon_max);
    if (![latMin, latMax, lonMin, lonMax].every(isFinite)) return null;
    return {
      latMin: Math.min(latMin, latMax),
      latMax: Math.max(latMin, latMax),
      lonMin: Math.min(lonMin, lonMax),
      lonMax: Math.max(lonMin, lonMax)
    };
  }

  function scenarioGeoBounds(scene) {
    var sceneBounds = normalizeGeoBounds(scene && scene.geo_bounds);
    if (sceneBounds) return sceneBounds;

    var activeScene = activeScenePayload(state.activeSceneTab || "imported");
    var activeSceneBounds = normalizeGeoBounds(activeScene && activeScene.geo_bounds);
    if (activeSceneBounds) return activeSceneBounds;

    var scenario = currentScenario();
    return normalizeGeoBounds(scenario && scenario.region_grid && scenario.region_grid.geo_bounds);
  }

  function mercatorProject(lat, lon, zoom) {
    var size = 256 * Math.pow(2, zoom);
    var safeLat = Math.max(-85.05112878, Math.min(85.05112878, Number(lat)));
    var sin = Math.sin(safeLat * Math.PI / 180);
    return {
      x: (Number(lon) + 180) / 360 * size,
      y: (0.5 - Math.log((1 + sin) / (1 - sin)) / (4 * Math.PI)) * size
    };
  }

  function mapViewport(width, height, scene) {
    var bounds = scenarioGeoBounds(scene);
    if (!bounds) return null;
    var bestZoom = 5;
    for (var zoom = 5; zoom <= 14; zoom += 1) {
      var northWest = mercatorProject(bounds.latMax, bounds.lonMin, zoom);
      var southEast = mercatorProject(bounds.latMin, bounds.lonMax, zoom);
      var spanX = Math.abs(southEast.x - northWest.x);
      var spanY = Math.abs(southEast.y - northWest.y);
      if (spanX <= width * 0.82 && spanY <= height * 0.82) {
        bestZoom = zoom;
      }
    }
    var centerLat = (bounds.latMin + bounds.latMax) / 2;
    var centerLon = (bounds.lonMin + bounds.lonMax) / 2;
    var center = mercatorProject(centerLat, centerLon, bestZoom);
    return {
      zoom: bestZoom,
      left: center.x - width / 2,
      top: center.y - height / 2
    };
  }

  function cartoTileUrl(zoom, x, y) {
    var subdomains = ["a", "b", "c", "d"];
    var subdomain = subdomains[Math.abs(x + y) % subdomains.length];
    return "https://" + subdomain + ".basemaps.cartocdn.com/rastertiles/voyager/" + zoom + "/" + x + "/" + y + ".png";
  }

  function renderRealMapLayer(hostId, activeKey) {
    var host = byId(hostId);
    if (!host) return;
    host.style.position = "relative";
    var width = 1618;
    var height = 1090;
    var scene = activeScenePayload(activeKey || state.activeSceneTab || "imported");
    var viewport = mapViewport(width, height, scene);
    var layerId = "tester-real-map-layer-" + hostId;
    var layer = byId(layerId);
    if (!layer) {
      layer = document.createElement("div");
      layer.id = layerId;
      host.appendChild(layer);
    }
    layer.style.cssText = [
      "position:absolute",
      "left:1px",
      "top:53px",
      "width:" + width + "px",
      "height:" + height + "px",
      "z-index:20",
      "overflow:hidden",
      "background:#e5e7eb",
      "pointer-events:none"
    ].join(";");

    if (!viewport) {
      layer.innerHTML = "";
      return;
    }

    var tileSize = 256;
    var maxTile = Math.pow(2, viewport.zoom);
    var minTileX = Math.floor(viewport.left / tileSize) - 1;
    var maxTileX = Math.floor((viewport.left + width) / tileSize) + 1;
    var minTileY = Math.floor(viewport.top / tileSize) - 1;
    var maxTileY = Math.floor((viewport.top + height) / tileSize) + 1;
    var tiles = "";

    for (var tileX = minTileX; tileX <= maxTileX; tileX += 1) {
      var wrappedX = ((tileX % maxTile) + maxTile) % maxTile;
      for (var tileY = minTileY; tileY <= maxTileY; tileY += 1) {
        if (tileY < 0 || tileY >= maxTile) continue;
        var left = Math.round(tileX * tileSize - viewport.left);
        var top = Math.round(tileY * tileSize - viewport.top);
        tiles += "<img src='" + cartoTileUrl(viewport.zoom, wrappedX, tileY) + "' style='position:absolute;left:" + left + "px;top:" + top + "px;width:" + tileSize + "px;height:" + tileSize + "px;' draggable='false' />";
      }
    }

    var scenario = currentScenario() || {};
    var regionName = scenario.region_grid && scenario.region_grid.name ? scenario.region_grid.name : scenarioLabel(state.scenarioName);
    layer.innerHTML = tiles +
      "<div style='position:absolute;left:18px;top:18px;padding:8px 12px;border-radius:8px;background:rgba(15,23,42,0.62);color:#fff;font-size:14px;font-family:Microsoft YaHei, PingFang SC, sans-serif;'>" +
      escapeHtml(regionName) +
      "</div>" +
      "<div style='position:absolute;right:14px;bottom:10px;padding:4px 8px;border-radius:6px;background:rgba(255,255,255,0.78);color:#334155;font-size:11px;font-family:Arial,sans-serif;'>© OpenStreetMap © CARTO</div>";
  }

  function applyScenarioMapSkin(activeKey) {
    renderRealMapLayer("u2692_state0_content", activeKey);
    renderRealMapLayer("u3039_state0_content", activeKey);
    ["u2693", "u2693_img", "u3040", "u3040_img"].forEach(function (id) {
      var node = byId(id);
      if (!node) return;
      node.style.display = "none";
      node.style.visibility = "hidden";
    });
  }

  function hideStaticScenarioRegionCards() {
    ["u2835", "u3181"].forEach(function (id) {
      setElementVisible(id, false);
    });
  }

  function updateScenarioDecorations() {
    var scenario = currentScenario();
    if (!scenario) return;

    var regionText = "区域：" + ((scenario.region_grid && scenario.region_grid.name) || scenarioLabel(state.scenarioName)) +
      "（离散网格 " + (scenario.region_grid && scenario.region_grid.rows || scenario.grid_size || "--") + " × " +
      (scenario.region_grid && scenario.region_grid.cols || scenario.grid_size || "--") + "）";

    hideStaticScenarioRegionCards();
    setText("u3034_text", comboLabel(state.scenarioName, state.algorithm));
    setText("u3038_text", comboLabel(state.scenarioName, state.algorithm) + " 策略测试");
    setText("u2851_text", regionText);
    setText("u3197_text", regionText);
    applyScenarioMapSkin();
  }

  function closeModal(id) {
    var node = byId(id);
    if (node) node.remove();
  }

  function persistDeviceLibraryMerge(items) {
    var library = loadDeviceLibrary();
    var map = {};
    library.concat(items || []).forEach(function (item, index) {
      var id = item.id || ("device-" + Date.now() + "-" + index);
      map[id] = {
        id: id,
        name: item.name || "未命名设备",
        deviceType: item.deviceType || "专用设备",
        communicationType: item.communicationType || "cellular",
        quantity: Math.max(1, integerValue(item.quantity, 1)),
        maxThroughput: numberValue(item.maxThroughput, 0),
        maxUsers: integerValue(item.maxUsers, 0),
        enabled: item.enabled !== false,
        status: item.status || "已导入"
      };
    });
    writeStorage(DEVICE_LIBRARY_KEY, Object.keys(map).map(function (key) { return map[key]; }));
  }

  function openDeviceModal() {
    closeModal("tester-device-modal");
    renderTesterDeviceAccessModule();
    addTesterDeviceSlot();
  }

  function matchBaseStationForCommunicationType(baseStations, communicationType) {
    var items = Array.isArray(baseStations) ? baseStations : [];
    var matchers = {
      cellular: function (station) {
        return /5g|macro|mmwave|蜂窝|宏站|微站/i.test(station.name || "") ||
          /5g|macro|mmwave|蜂窝|宏站|微站/i.test(station.label || "") ||
          (station.supported_modes || []).some(function (mode) { return /5g/i.test(mode); });
      },
      wifi: function (station) {
        return /wifi/i.test(station.name || "") ||
          /wifi/i.test(station.label || "") ||
          (station.supported_modes || []).some(function (mode) { return /wifi/i.test(mode); });
      },
      satellite: function (station) {
        return /satellite|卫星/i.test(station.name || "") ||
          /satellite|卫星/i.test(station.label || "") ||
          (station.supported_modes || []).some(function (mode) { return /satellite/i.test(mode); });
      },
      shortwave: function (station) {
        return /shortwave|hf|短波/i.test(station.name || "") ||
          /shortwave|hf|短波/i.test(station.label || "") ||
          (station.supported_modes || []).some(function (mode) { return /shortwave|hf/i.test(mode); });
      }
    };
    var predicate = matchers[communicationType];
    return predicate ? (items.find(predicate) || null) : null;
  }

  function buildScenarioDeviceBaseStations() {
    var scenario = currentScenario();
    var baseStations = scenario && Array.isArray(scenario.base_stations) ? scenario.base_stations : [];
    var rows = scenario && scenario.region_grid && scenario.region_grid.rows ? scenario.region_grid.rows : (scenario && scenario.grid_size ? scenario.grid_size : 1);
    var cols = scenario && scenario.region_grid && scenario.region_grid.cols ? scenario.region_grid.cols : (scenario && scenario.grid_size ? scenario.grid_size : 1);
    return state.scenarioDeviceRows.flatMap(function (row) {
      if (!row.applied || !row.enabled || Number(row.quantity) <= 0) return [];
      var matched = row.baseStationName
        ? scenarioBaseStationByName(row.baseStationName)
        : matchBaseStationForCommunicationType(baseStations, row.communicationType);
      if (!matched) return [];
      var supportedModes = Array.isArray(matched.supported_modes) ? matched.supported_modes : [];
      var mode = row.mode && supportedModes.indexOf(row.mode) !== -1 ? row.mode : (supportedModes.length ? supportedModes[0] : null);
      return Array.from({ length: Math.max(1, integerValue(row.quantity, 1)) }, function (_, index) {
        return {
          base_station: matched.name,
          mode: mode,
          x: (integerValue(row.x, 0) + index) % Math.max(1, rows),
          y: (integerValue(row.y, 0) + index) % Math.max(1, cols)
        };
      });
    });
  }

  function comboRows() {
    var rows = [];
    state.scenarios.forEach(function (scenario) {
      ALGORITHMS.forEach(function (algorithm) {
        var artifact = state.artifacts.find(function (item) {
          return item.scenario_name === scenario.name && item.algorithm === algorithm.key && item.checkpoint_path;
        }) || null;
        rows.push({
          scenarioName: scenario.name,
          scenarioLabel: disasterLabel(scenario.disaster_type),
          algorithm: algorithm.key,
          algorithmLabel: algorithm.label,
          checkpointPath: artifact ? artifact.checkpoint_path : ""
        });
      });
    });
    return rows;
  }

  function openComboModal() {
    closeModal("tester-combo-modal");
    var items = comboRows();
    var modal = document.createElement("div");
    modal.id = "tester-combo-modal";
    modal.style.cssText = "position:fixed;inset:0;background:rgba(2,6,23,0.5);z-index:100001;display:flex;align-items:center;justify-content:center;";
    var rows = items.map(function (item, index) {
      var active = item.scenarioName === state.scenarioName && item.algorithm === state.algorithm;
      return "<tr data-index='" + index + "' style='cursor:pointer;background:" + (active ? "#eff6ff" : "#fff") + ";'>" +
        "<td style='padding:11px 12px;border-bottom:1px solid #eef2f7;'>" + (index + 1) + "</td>" +
        "<td style='padding:11px 12px;border-bottom:1px solid #eef2f7;'>" + escapeHtml(item.scenarioLabel) + "</td>" +
        "<td style='padding:11px 12px;border-bottom:1px solid #eef2f7;'>" + escapeHtml(item.algorithmLabel) + "</td>" +
        "<td style='padding:11px 12px;border-bottom:1px solid #eef2f7;color:" + (item.checkpointPath ? "#16a34a" : "#ef4444") + ";'>" + (item.checkpointPath ? "已匹配权重" : "无权重") + "</td>" +
        "</tr>";
    }).join("");
    modal.innerHTML =
      "<div style='width:920px;max-height:680px;overflow:auto;background:#fff;border-radius:16px;padding:24px;box-shadow:0 24px 60px rgba(15,23,42,0.24);'>" +
      "<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;'>" +
      "<strong style='font-size:18px;color:#0f172a;'>选择场景与算法</strong>" +
      "<button type='button' style='border:0;background:none;font-size:16px;color:#64748b;cursor:pointer;' onclick='this.closest(\\\"#tester-combo-modal\\\").remove()'>关闭</button>" +
      "</div>" +
      "<table style='width:100%;border-collapse:collapse;font-size:14px;color:#334155;'>" +
      "<thead><tr style='background:#f8fafc;'><th style='padding:10px 12px;text-align:left;'>序号</th><th style='padding:10px 12px;text-align:left;'>场景</th><th style='padding:10px 12px;text-align:left;'>算法</th><th style='padding:10px 12px;text-align:left;'>状态</th></tr></thead>" +
      "<tbody>" + (rows || "<tr><td colspan='4' style='padding:28px 0;text-align:center;color:#94a3b8;'>暂无可用组合</td></tr>") + "</tbody>" +
      "</table>" +
      "</div>";
    document.body.appendChild(modal);
    modal.addEventListener("click", function (event) {
      if (event.target === modal) modal.remove();
    });
    Array.prototype.forEach.call(modal.querySelectorAll("tbody tr[data-index]"), function (row) {
      row.addEventListener("click", function () {
        var selected = items[Number(row.getAttribute("data-index"))];
        if (!selected) return;
        state.scenarioName = selected.scenarioName;
        state.algorithm = selected.algorithm;
        state.simulationResult = null;
        state.importedScene = null;
        updateMetrics(null);
        setTabVisual("imported");
        state.activeSceneTab = "imported";
        syncDeviceRowsFromStorage();
        syncCheckpoint();
        appendTerminalLine("已切换到 " + comboLabel(state.scenarioName, state.algorithm) + "。", "info");
        modal.remove();
        importScene();
      });
    });
  }

  function openTestHistoryModal() {
    closeModal("tester-history-modal");
    var history = readTestHistory();
    var modal = document.createElement("div");
    modal.id = "tester-history-modal";
    modal.style.cssText = "position:fixed;inset:0;background:rgba(2,6,23,0.5);z-index:100001;display:flex;align-items:center;justify-content:center;";
    var rows = history.map(function (item, index) {
      return "<tr>" +
        "<td style='padding:10px 12px;border-bottom:1px solid #eef2f7;'>" + (index + 1) + "</td>" +
        "<td style='padding:10px 12px;border-bottom:1px solid #eef2f7;'>" + escapeHtml(item.scenarioLabel || item.scenarioName || "--") + "</td>" +
        "<td style='padding:10px 12px;border-bottom:1px solid #eef2f7;'>" + escapeHtml(item.algorithmLabel || item.algorithm || "--") + "</td>" +
        "<td style='padding:10px 12px;border-bottom:1px solid #eef2f7;'>" + escapeHtml(formatDateTime(item.createdAt)) + "</td>" +
        "<td style='padding:10px 12px;border-bottom:1px solid #eef2f7;'>" + escapeHtml(formatMetric(item.avgReward, 2)) + "</td>" +
        "<td style='padding:10px 12px;border-bottom:1px solid #eef2f7;'>" + escapeHtml(formatPercent(item.avgFinalCoverage)) + "</td>" +
        "<td style='padding:10px 12px;border-bottom:1px solid #eef2f7;display:flex;gap:8px;'>" +
          "<button type='button' data-action='view' data-id='" + escapeHtml(item.id) + "' style='padding:6px 12px;border:1px solid #dbe4ff;border-radius:8px;background:#eef4ff;color:#1d4ed8;cursor:pointer;'>查看</button>" +
          "<button type='button' data-action='delete' data-id='" + escapeHtml(item.id) + "' style='padding:6px 12px;border:1px solid #fee2e2;border-radius:8px;background:#fff1f2;color:#e11d48;cursor:pointer;'>删除</button>" +
        "</td>" +
        "</tr>";
    }).join("");
    modal.innerHTML =
      "<div style='width:1120px;max-height:720px;overflow:auto;background:#fff;border-radius:16px;padding:24px;box-shadow:0 24px 60px rgba(15,23,42,0.24);'>" +
      "<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;'>" +
      "<strong style='font-size:18px;color:#0f172a;'>策略测试记录</strong>" +
      "<button type='button' style='border:0;background:none;font-size:16px;color:#64748b;cursor:pointer;' onclick='this.closest(\\\"#tester-history-modal\\\").remove()'>关闭</button>" +
      "</div>" +
      "<table style='width:100%;border-collapse:collapse;font-size:14px;color:#334155;'>" +
      "<thead><tr style='background:#f8fafc;'><th style='padding:10px 12px;text-align:left;'>序号</th><th style='padding:10px 12px;text-align:left;'>场景</th><th style='padding:10px 12px;text-align:left;'>算法</th><th style='padding:10px 12px;text-align:left;'>时间</th><th style='padding:10px 12px;text-align:left;'>平均奖励</th><th style='padding:10px 12px;text-align:left;'>覆盖率</th><th style='padding:10px 12px;text-align:left;'>操作</th></tr></thead>" +
      "<tbody>" + (rows || "<tr><td colspan='7' style='padding:28px 0;text-align:center;color:#94a3b8;'>暂无真实测试记录，请先执行一次策略测试。</td></tr>") + "</tbody>" +
      "</table>" +
      "</div>";
    document.body.appendChild(modal);
    modal.addEventListener("click", function (event) {
      if (event.target === modal) modal.remove();
    });
    Array.prototype.forEach.call(modal.querySelectorAll("button[data-action]"), function (button) {
      button.addEventListener("click", function (event) {
        event.preventDefault();
        event.stopPropagation();
        var id = button.getAttribute("data-id");
        if (button.getAttribute("data-action") === "view") {
          var item = readTestHistory().find(function (entry) { return entry.id === id; });
          if (item) openTestHistoryDetailModal(item);
          return;
        }
        writeTestHistory(readTestHistory().filter(function (entry) { return entry.id !== id; }));
        modal.remove();
        openTestHistoryModal();
      });
    });
  }

  function openTestHistoryDetailModal(item) {
    closeModal("tester-history-detail-modal");
    var rows = (item.deviceRows || []).map(function (device) {
      var position = Array.isArray(device.position) ? device.position.join(", ") : "--";
      return "<tr>" +
        "<td style='padding:10px 12px;border-bottom:1px solid #f1f5f9;'>" + escapeHtml(device.id) + "</td>" +
        "<td style='padding:10px 12px;border-bottom:1px solid #f1f5f9;'>" + escapeHtml(position) + "</td>" +
        "<td style='padding:10px 12px;border-bottom:1px solid #f1f5f9;'>" + escapeHtml(device.region_label || "--") + "</td>" +
        "<td style='padding:10px 12px;border-bottom:1px solid #f1f5f9;'>" + escapeHtml(formatMetric(device.demand, 1)) + " Mbps</td>" +
        "<td style='padding:10px 12px;border-bottom:1px solid #f1f5f9;color:" + (device.connected ? "#16a34a" : "#ef4444") + ";'>" + (device.connected ? "在线" : "离线") + "</td>" +
        "<td style='padding:10px 12px;border-bottom:1px solid #f1f5f9;color:" + (device.broadcast_served ? "#16a34a" : "#f59e0b") + ";'>" + (device.broadcast_served ? "已覆盖" : "未覆盖") + "</td>" +
        "</tr>";
    }).join("");
    var modal = document.createElement("div");
    modal.id = "tester-history-detail-modal";
    modal.style.cssText = "position:fixed;inset:0;background:rgba(2,6,23,0.58);z-index:100002;display:flex;align-items:center;justify-content:center;";
    modal.innerHTML =
      "<div style='width:1080px;max-height:760px;overflow:auto;background:#fff;border-radius:16px;padding:24px;box-shadow:0 24px 60px rgba(15,23,42,0.28);'>" +
      "<div style='display:flex;justify-content:space-between;align-items:flex-start;gap:16px;margin-bottom:18px;'>" +
      "<div><div style='font-size:22px;line-height:1.4;color:#0f172a;font-weight:700;'>" + escapeHtml((item.scenarioLabel || item.scenarioName || "--") + " + " + (item.algorithmLabel || item.algorithm || "--")) + "</div>" +
      "<div style='margin-top:6px;font-size:13px;color:#64748b;'>测试时间 " + escapeHtml(formatDateTime(item.createdAt)) + "</div></div>" +
      "<button type='button' style='border:0;background:none;font-size:16px;cursor:pointer;color:#64748b;' onclick='this.closest(\\\"#tester-history-detail-modal\\\").remove()'>关闭</button>" +
      "</div>" +
      "<div style='display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-bottom:18px;'>" +
      "<div style='padding:14px 16px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;'><div style='font-size:12px;color:#64748b;'>平均奖励</div><div style='margin-top:8px;font-size:20px;color:#0f172a;font-weight:700;'>" + escapeHtml(formatMetric(item.avgReward, 2)) + "</div></div>" +
      "<div style='padding:14px 16px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;'><div style='font-size:12px;color:#64748b;'>平均覆盖率</div><div style='margin-top:8px;font-size:20px;color:#0f172a;font-weight:700;'>" + escapeHtml(formatPercent(item.avgFinalCoverage)) + "</div></div>" +
      "<div style='padding:14px 16px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;'><div style='font-size:12px;color:#64748b;'>广播覆盖</div><div style='margin-top:8px;font-size:20px;color:#0f172a;font-weight:700;'>" + escapeHtml(formatPercent(item.broadcastRatio)) + "</div></div>" +
      "<div style='padding:14px 16px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;'><div style='font-size:12px;color:#64748b;'>终端数量</div><div style='margin-top:8px;font-size:20px;color:#0f172a;font-weight:700;'>" + escapeHtml(String(item.userCount || 0)) + "</div></div>" +
      "</div>" +
      "<div style='padding:16px;border:1px solid #e2e8f0;border-radius:14px;margin-bottom:16px;'>" +
      "<div style='font-size:16px;font-weight:700;color:#0f172a;margin-bottom:10px;'>导出路径</div>" +
      "<div style='font-size:13px;line-height:1.9;color:#334155;word-break:break-all;'>" +
      "<div>Checkpoint：<strong>" + escapeHtml(item.checkpointPath || "--") + "</strong></div>" +
      "<div>受灾场景：<strong>" + escapeHtml(item.disasterScenePath || "--") + "</strong></div>" +
      "<div>部署后场景：<strong>" + escapeHtml(item.deploymentScenePath || "--") + "</strong></div>" +
      "</div></div>" +
      "<div style='padding:16px;border:1px solid #e2e8f0;border-radius:14px;'>" +
      "<div style='font-size:16px;font-weight:700;color:#0f172a;margin-bottom:12px;'>设备恢复明细</div>" +
      "<div style='max-height:360px;overflow:auto;'><table style='width:100%;border-collapse:collapse;font-size:13px;color:#334155;'>" +
      "<thead><tr style='background:#f8fafc;'><th style='padding:10px 12px;text-align:left;'>ID</th><th style='padding:10px 12px;text-align:left;'>位置</th><th style='padding:10px 12px;text-align:left;'>区域</th><th style='padding:10px 12px;text-align:left;'>需求</th><th style='padding:10px 12px;text-align:left;'>连接状态</th><th style='padding:10px 12px;text-align:left;'>广播</th></tr></thead>" +
      "<tbody>" + (rows || "<tr><td colspan='6' style='padding:24px 0;text-align:center;color:#94a3b8;'>暂无明细</td></tr>") + "</tbody></table></div></div>" +
      "</div>";
    document.body.appendChild(modal);
    modal.addEventListener("click", function (event) {
      if (event.target === modal) modal.remove();
    });
  }

  function syncCheckpoint() {
    var artifact = findMatchingArtifact();
    state.checkpointPath = artifact ? artifact.checkpoint_path : "";
    updateScenarioDecorations();
    if (!state.checkpointPath) {
      setStartButton("开始测试", "disabled", true);
      setStatus("缺少权重", "error");
    } else if (!state.running) {
      setStartButton("开始测试", "idle", false);
      setStatus("场景就绪", "success");
    }
  }

  function bindComboPicker() {
    ["u3027", "u3034", "u3034_div", "u3034_text"].forEach(function (id) {
      var node = byId(id);
      if (!node || node.dataset.liveBound) return;
      node.dataset.liveBound = "true";
      node.style.pointerEvents = "auto";
      node.style.cursor = "pointer";
      node.addEventListener("click", function (event) {
        event.preventDefault();
        event.stopPropagation();
        openComboModal();
      }, true);
    });
  }

  function buildImportedDevices() {
    var details = state.importedScene && state.importedScene.initial_state && Array.isArray(state.importedScene.initial_state.user_details)
      ? state.importedScene.initial_state.user_details
      : [];
    return details
      .filter(function (device) {
        return Array.isArray(device.position) && device.position.length >= 2;
      })
      .map(function (device) {
        return {
          x: Number(device.position[0]),
          y: Number(device.position[1]),
          demand: Number(device.demand || 10),
          connected: Boolean(device.connected),
          broadcast_served: Boolean(device.broadcast_served)
        };
      });
  }

  async function importScene() {
    if (!state.scenarioName || state.loadingScene) return;
    state.loadingScene = true;
    if (!state.running) {
      state.simulationResult = null;
    }
    setStatus("同步场景中", "warning");
    syncDeviceRowsFromStorage();

    try {
      var response = await fetch(API + "/simulate/scene", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          scenario_name: state.scenarioName,
          env_type: "multimodal",
          evaluation_protocol: evaluationProtocol(),
          custom_base_stations: buildScenarioDeviceBaseStations()
        })
      });

      if (!response.ok) {
        throw new Error(await response.text() || "场景导入失败");
      }

      state.importedScene = await response.json();
      var initialState = state.importedScene.initial_state || {};
      setText("u3014_text", String(initialState.total_users || 0));
      setText("u3018_text", String(initialState.total_users || 0));
      setText("u3022_text", String((initialState.residual_base_stations || []).length || 0));
      appendTerminalLine(
        "已同步场景：" + scenarioLabel(state.scenarioName) +
          "，用户 " + (initialState.total_users || 0) +
          "，残余基站 " + (((initialState.residual_base_stations || []).length)) + " 个，设备接入 " + deviceSummaryLabel() + "。",
        "success"
      );
      updateSummaryLine(
        comboLabel(state.scenarioName, state.algorithm) +
        "，场景已同步，用户 " + (initialState.total_users || 0) +
        "，残余基站 " + (((initialState.residual_base_stations || []).length)) + " 个。"
      );
      if (!state.running) {
        setStatus(state.checkpointPath ? "场景就绪" : "缺少权重", state.checkpointPath ? "success" : "error");
      }
      state.activeSceneTab = "imported";
      setTabVisual("imported");
    } catch (error) {
      appendTerminalLine("场景同步失败：" + (error && error.message ? error.message : error), "error");
      updateSummaryLine("场景同步失败，请检查后端服务状态。");
      setStatus("同步失败", "error");
    } finally {
      state.loadingScene = false;
    }
  }

  function readErrorResponse(response) {
    return response.text().then(function (text) {
      if (!text) return "请求失败 (" + response.status + ")";
      try {
        var parsed = JSON.parse(text);
        return parsed.detail || parsed.message || text;
      } catch (error) {
        return text;
      }
    });
  }

  function processSimulationEvent(event) {
    var payload = event && event.payload ? event.payload : {};

    if (event.type === "status") {
      if (payload.state === "initializing") {
        setStatus("初始化中", "warning");
      } else if (payload.state === "running") {
        setStatus("测试中", "warning");
      } else if (payload.state === "completed") {
        setStatus("已完成", "success");
      } else if (payload.state === "failed") {
        setStatus("执行失败", "error");
      }
      return;
    }

    if (event.type === "log") {
      appendTerminalLine(payload.message, payload.event_type === "error" ? "error" : "info");
      return;
    }

    if (event.type === "result") {
      state.simulationResult = payload;
      updateMetrics(payload);
      persistTestHistory(payload);
      state.activeSceneTab = payload.scene_export && payload.scene_export.deployment_scene ? "deployment" : "imported";
      setTabVisual(state.activeSceneTab);
      var note = byId("tester-scene-note");
      if (note) {
        note.textContent = state.activeSceneTab === "deployment" ? "部署后场景" : "导入的场景";
      }
      appendTerminalLine(
        "测试完成：平均奖励 " + formatNumber(payload.avg_reward, 2) +
          "，平均覆盖率 " + formatPercent(payload.avg_final_coverage) + "。",
        "success"
      );
      return;
    }

    if (event.type === "error") {
      appendTerminalLine("测试失败：" + (payload.message || "未知错误"), "error");
      setStatus("执行失败", "error");
      return;
    }
  }

  function processSseChunk(chunk) {
    var payloadText = chunk
      .split("\\n")
      .filter(function (line) { return line.indexOf("data:") === 0; })
      .map(function (line) { return line.slice(5).trim(); })
      .join("\\n");
    if (!payloadText) return;

    try {
      var event = JSON.parse(payloadText);
      processSimulationEvent(event);
    } catch (error) {
      appendTerminalLine("无法解析流式结果：" + payloadText, "warning");
    }
  }

  async function consumeSimulationStream(response) {
    if (!response.body) {
      throw new Error("当前浏览器不支持流式返回。");
    }

    var reader = response.body.getReader();
    var decoder = new TextDecoder("utf-8");
    var buffer = "";

    while (true) {
      var result = await reader.read();
      if (result.done) break;
      buffer += decoder.decode(result.value, { stream: true }).replace(/\\r\\n/g, "\\n");
      var boundary = buffer.indexOf("\\n\\n");
      while (boundary !== -1) {
        var chunk = buffer.slice(0, boundary).trim();
        buffer = buffer.slice(boundary + 2);
        if (chunk) {
          processSseChunk(chunk);
        }
        boundary = buffer.indexOf("\\n\\n");
      }
    }

    var tail = buffer.trim();
    if (tail) {
      processSseChunk(tail);
    }
  }

  async function startSimulation() {
    if (state.running || !state.checkpointPath) {
      if (!state.checkpointPath) {
        appendTerminalLine("当前场景与算法没有匹配的训练权重，请先完成训练。", "error");
      }
      return;
    }

    clearTerminal();
    updateMetrics(null);
    state.simulationResult = null;
    state.activeSceneTab = "imported";
    setTabVisual("imported");
    appendTerminalLine("准备启动 " + comboLabel(state.scenarioName, state.algorithm) + " 的策略测试。", "info");

    appendTerminalLine("正在按当前设备接入配置同步场景。", "info");
    await importScene();
    if (!state.importedScene) {
      return;
    }

    state.running = true;
    setStartButton("测试中...", "running", false);
    setStatus("测试中", "warning");
    syncDeviceRowsFromStorage();
    updateSummaryLine(comboLabel(state.scenarioName, state.algorithm) + "，测试执行中，实时日志已接入真实后端流。");
    appendTerminalLine("当前场景已应用设备：" + deviceSummaryLabel() + "。", "info");

    try {
      var response = await fetch(API + "/simulate/stream", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          scenario_name: state.scenarioName,
          checkpoint_path: state.checkpointPath,
          env_type: "multimodal",
          algorithm: state.algorithm,
          evaluation_protocol: evaluationProtocol(),
          episodes: 1,
          stochastic_eval: true,
          eval_seed: 13,
          custom_devices: buildImportedDevices(),
          custom_base_stations: buildScenarioDeviceBaseStations()
        })
      });

      if (!response.ok) {
        throw new Error(await readErrorResponse(response));
      }

      await consumeSimulationStream(response);

      if (state.simulationResult) {
        setStartButton("重新测试", "success", false);
        updateSummaryLine(comboLabel(state.scenarioName, state.algorithm) + "，测试已完成，可查看设备恢复与导出结果。");
      } else {
        throw new Error("测试结束但未收到结果数据。");
      }
    } catch (error) {
      appendTerminalLine("测试执行失败：" + (error && error.message ? error.message : error), "error");
      setStatus("执行失败", "error");
      setStartButton("重新测试", "error", false);
      updateSummaryLine("测试执行失败，请检查后端接口与模型权重。");
    } finally {
      state.running = false;
      if (state.checkpointPath && !state.simulationResult && textHolder("u2834_text") && textHolder("u2834_text").textContent !== "重新测试") {
        setStartButton("开始测试", "idle", false);
      }
    }
  }

  function wireButtons() {
    var button = byId("u2834");
    if (button && !button.dataset.liveBound) {
      button.dataset.liveBound = "true";
      button.style.pointerEvents = "auto";
      button.addEventListener("click", function (event) {
        event.preventDefault();
        event.stopPropagation();
        startSimulation();
      }, true);
    }

    [
      { ids: ["u2844", "u2844_img", "u2844_text"], key: "imported", label: "导入的场景" },
      { ids: ["u2843", "u2843_img", "u2843_text"], key: "deployment", label: "部署后场景" }
    ].forEach(function (tab) {
      tab.ids.forEach(function (id) {
        var node = byId(id);
        if (!node || node.dataset.liveBound) return;
        node.dataset.liveBound = "true";
        node.style.pointerEvents = "auto";
        node.style.cursor = "pointer";
        node.addEventListener("click", function (event) {
          event.preventDefault();
          event.stopPropagation();
          if (tab.key === "deployment" && !hasDeploymentScene()) {
            state.activeSceneTab = "imported";
            setTabVisual("imported");
            return;
          }
          state.activeSceneTab = tab.key;
          setTabVisual(tab.key);
          var note = byId("tester-scene-note");
          if (note) note.textContent = tab.label;
        }, true);
      });
    });

    ["u2418", "u2418_div", "u2418_text"].forEach(function (id) {
      var historyTrigger = byId(id);
      if (!historyTrigger || historyTrigger.dataset.liveBound) return;
      historyTrigger.dataset.liveBound = "true";
      historyTrigger.style.pointerEvents = "auto";
      historyTrigger.style.cursor = "pointer";
      historyTrigger.addEventListener("click", function (event) {
        event.preventDefault();
        event.stopPropagation();
        openTestHistoryModal();
      }, true);
    });

    var deviceTrigger = byId("tester-device-entry");
    if (deviceTrigger && !deviceTrigger.dataset.liveBound) {
      deviceTrigger.dataset.liveBound = "true";
      deviceTrigger.style.pointerEvents = "auto";
      deviceTrigger.addEventListener("click", function (event) {
        event.preventDefault();
        event.stopPropagation();
        openDeviceModal();
      }, true);
    }
  }

  async function bootstrap() {
    ensureDeviceLibrarySeeded();
    ensureOverlay();
    wireButtons();
    setTabVisual("imported");
    updateMetrics(null);
    setStartButton("开始测试", "disabled", true);
    setStatus("加载中", "warning");
    appendTerminalLine("正在加载场景配置与训练产物清单。", "info");

    try {
      var responses = await Promise.all([
        fetch(API + "/scenarios"),
        fetch(API + "/train/artifacts")
      ]);

      if (!responses[0].ok) {
        throw new Error(await readErrorResponse(responses[0]));
      }
      if (!responses[1].ok) {
        throw new Error(await readErrorResponse(responses[1]));
      }

      var scenariosPayload = await responses[0].json();
      var artifactsPayload = await responses[1].json();
      state.scenarios = Array.isArray(scenariosPayload.scenarios) ? scenariosPayload.scenarios : [];
      state.artifacts = Array.isArray(artifactsPayload.artifacts) ? artifactsPayload.artifacts : [];

      var latestArtifact = state.artifacts[0] || null;
      var fallbackScenario = state.scenarios[0] || null;
      state.scenarioName = latestArtifact && latestArtifact.scenario_name ? latestArtifact.scenario_name : (fallbackScenario ? fallbackScenario.name : "");
      state.algorithm = latestArtifact && latestArtifact.algorithm ? latestArtifact.algorithm : "ppo";

      bindComboPicker();
      syncDeviceRowsFromStorage();
      syncCheckpoint();
      await importScene();
      appendTerminalLine("策略测试页已切换为真实后端驱动模式。", "success");
    } catch (error) {
      appendTerminalLine("初始化失败：" + (error && error.message ? error.message : error), "error");
      updateSummaryLine("初始化失败，请确认前端可访问 /api/scenarios 和 /api/train/artifacts。");
      setStatus("初始化失败", "error");
      setStartButton("开始测试", "disabled", true);
    }
  }

  bootstrap();
})();
`;

export function injectPrototypeTester(doc) {
  if (!doc) return;

  var previousScript = doc.getElementById("tester-api-inject");
  if (previousScript) previousScript.remove();

  var previousOverlay = doc.getElementById("tester-live-overlay");
  if (previousOverlay) previousOverlay.remove();

  var script = doc.createElement("script");
  script.id = "tester-api-inject";
  script.textContent = buildInjectionScript("/api", COMMUNICATION_TYPE_OPTIONS, DEFAULT_DEVICE_TEMPLATES);
  doc.head && doc.head.appendChild(script);
}
