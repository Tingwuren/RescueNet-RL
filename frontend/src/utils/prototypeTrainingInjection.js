const COMMUNICATION_TYPE_OPTIONS = [
  { value: "cellular", label: "蜂窝通信" },
  { value: "wifi", label: "WiFi 通信" },
  { value: "satellite", label: "卫星通信" },
  { value: "shortwave", label: "短波通信" },
];

const DEFAULT_DEVICE_TEMPLATES = [
  {
    id: "train-cellular-macro",
    name: "蜂窝宏站",
    deviceType: "宏基站",
    communicationType: "cellular",
    quantity: 2,
    maxThroughput: 240,
    maxUsers: 180,
    enabled: true,
    status: "已导入",
  },
  {
    id: "train-wifi-hotspot",
    name: "WiFi6 热点",
    deviceType: "背负式基站",
    communicationType: "wifi",
    quantity: 3,
    maxThroughput: 160,
    maxUsers: 96,
    enabled: true,
    status: "已导入",
  },
  {
    id: "train-satellite-relay",
    name: "卫星中继",
    deviceType: "中继设备",
    communicationType: "satellite",
    quantity: 1,
    maxThroughput: 150,
    maxUsers: 120,
    enabled: true,
    status: "已导入",
  },
  {
    id: "train-shortwave-station",
    name: "短波台",
    deviceType: "临时设备/车载设备",
    communicationType: "shortwave",
    quantity: 2,
    maxThroughput: 24,
    maxUsers: 220,
    enabled: true,
    status: "已导入",
  },
];

function trainingInjector(apiBase, communicationTypes, defaultTemplates) {
  var API = apiBase;
  var LOCAL_SCENES_KEY = "prototype-training-scenes";
  var COMM_TYPES = communicationTypes;
  var DEFAULT_DEVICE_TEMPLATES = defaultTemplates;
  var eventSource = null;
  var state = {
    scenarios: [],
    scenarioName: "",
    disasterType: "typhoon",
    disasterSeverity: "severe",
    scenarioTitle: "",
    adminDivision: "",
    affectedGridCount: 24,
    impactedPopulation: 320,
    disasterNotes: "",
    priorityArea: "",
    priorityEquipment: "背负式基站 + 多跳中继",
    coverageRange: "",
    cellGranularity: "",
    candidateSiteCount: 24,
    dispatchUnit: "前线应急通信保障队",
    teamCount: 4,
    budgetLimit: 300,
    accessSlotCount: 0,
    accessDevices: [],
    residualType1: "",
    residualDevice1: "",
    residualLocation1: "",
    residualCount1: 0,
    residualType2: "",
    residualDevice2: "",
    residualLocation2: "",
    residualCount2: 0,
    candidateType1: "",
    candidateDevice1: "",
    candidateCount1: 0,
    candidateType2: "",
    candidateDevice2: "",
    candidateCount2: 0,
    algorithm: "ppo",
    rewardMode: "coverage_balance",
    envType: "multimodal",
    stochasticEval: true,
    totalTimesteps: 12000,
    learningRate: 0.0003,
    discountFactor: 0.99,
    batchSize: 256,
    rolloutSteps: 1024,
    entropyCoef: 0.01,
    clipRange: 0.2,
    simulationWindowHours: 6,
    coverageTarget: 85,
    logWindow: 50,
    trafficLoadProfile: "high",
    priorityObjective: "coverage_first",
    evalInterval: 4,
    autoReplay: true,
    running: false,
    chartPoints: []
  };
  var editors = {};
  var DEVICE_ACCESS_MAX = 50;
  var SCENARIO_DESCRIPTION_TOP = 420;
  var SCENARIO_DESCRIPTION_HEIGHT = 170;
  var DEVICE_ACCESS_TOP = SCENARIO_DESCRIPTION_TOP + SCENARIO_DESCRIPTION_HEIGHT + 24;
  var ALGORITHM_PANEL_COMPACT_HEIGHT = 204;
  var ALGORITHM_CARD_LEFT = 66;
  var ALGORITHM_CARD_TOP = 64;
  var ALGORITHM_CARD_WIDTH = 284;
  var ALGORITHM_CARD_HEIGHT = 112;
  var ALGORITHM_CARD_GAP = 18;
  var REWARD_CONFIG_PANEL_HEIGHT = 196;
  var SECTION_GAP = 20;
  var SCENE_DEVICE_TYPE_OPTIONS = [
    { value: "", label: "请选择类型" },
    { value: "macro_station", label: "宏基站" },
    { value: "backpack_station", label: "背负式基站" },
    { value: "relay", label: "中继设备" },
    { value: "vehicle_station", label: "临时设备/车载设备" },
  ];
  var SCENE_DEVICE_OPTIONS = buildDeviceOptions();
  var PARAMETER_TAB_IMAGES = {
    u780: {
      normal: "images/模型训练/u780.png",
      selected: "images/模型训练/u780_selected.png",
    },
    u781: {
      normal: "images/模型训练/u781.png",
      selected: "images/模型训练/u781_selected.png",
    },
    u782: {
      normal: "images/模型训练/u782.png",
      selected: "images/模型训练/u782_selected.png",
    },
  };
  var ALGORITHM_CARD_IMAGE = {
    normal: "images/模型训练/u1124.png",
    selected: "images/模型训练/u1124_selected.png",
  };
  var ALGORITHM_CARD_CONFIG = [
    {
      key: "ppo",
      cardId: "u1124",
      titleId: "u1125",
      descId: "u1126",
      title: "基于 PPO 的覆盖恢复策略优化方案",
      desc: "覆盖优先 / 稳定基线",
    },
    {
      key: "dqn",
      cardId: "u1127",
      titleId: "u1128",
      descId: "u1129",
      title: "基于 DQN 的离散站点部署决策方案",
      desc: "离散动作 / 快速推演",
    },
    {
      key: "a3c",
      cardId: "u1130",
      titleId: "u1131",
      descId: "u1132",
      title: "基于 A3C 的多目标协同训练方案",
      desc: "异步更新 / 多目标",
    },
    {
      key: "mppo",
      cardId: "u1133",
      titleId: "u1134",
      descId: "u1135",
      title: "基于 MPPO 的多头策略组网方案",
      desc: "多头策略 / 资源协同",
    },
    {
      key: "hmarl",
      cardId: "u1136",
      titleId: "u1137",
      descId: "u1138",
      title: "层次化多智能体通信资源配置与组网方案",
      desc: "自研方案 / 分层协同",
    },
  ];
  var SCENARIO_MODELING_TEMPLATES = {
    earthquake_residual: {
      scenarioBackground: "地震场景面向震后山区断链恢复。重点关注县城核心区、山脊村落和河谷安置点之间的通信重建，要求在余震持续、道路通行受限的条件下恢复基础覆盖和关键业务回传。",
      disasterTraits: "震后地形破坏明显，受灾点位分散，安置点与山地村落之间距离拉大，局部区域会持续出现余震和临时交通阻断。",
      networkDamage: "原有宏站存在残余可用能力，但部分站点退服、链路中断、回传不稳定，需要依靠卫星中继、短波和补盲热点进行分层接入恢复。",
      modelingMethod: "将受灾区域离散为山区网格，把用户、候选站点、残余站和机动设备统一建模为图上的节点与边；重点保障区、边缘补盲区和中继转发区采用不同的覆盖收益和部署成本。",
      stateDef: "状态包含网格覆盖率、残余基站存活情况、候选站点可用性、关键区域服务缺口、设备剩余预算和链路容量余量。",
      actionDef: "动作定义为在候选位置部署或切换抗震宏站、mmWave 微站、卫星 Ku 中继、WiFi6 热点、短波超视距台等设备，并调整补盲与中继组合。",
      rewardDef: "奖励以覆盖恢复优先为主，同时加入链路吞吐、广播可达性、设备开销和带宽成本惩罚，适合余震期先补盲、再稳链路的恢复策略。",
      target: "优先恢复重点保障区域和山地聚落的连续覆盖，兼顾关键业务回传能力，尽量减少高成本设备的过量投入。",
      difference: "与洪水场景相比，地震场景存在残余网络可复用，重点在断链修复和山区中继；与台风场景相比，地震更强调离散山地节点的补盲和多跳回传，而不是沿海大范围连续覆盖。"
    },
    flood_no_residual: {
      scenarioBackground: "洪水场景面向孤岛通信恢复。受灾区域以城郊易涝区块和低洼片区为主，原有地面网络大量失效，需要从零开始搭建应急接入和回传链路。",
      disasterTraits: "洪水导致道路中断、局部区域被水体分割，用户集中在安置点、堤坝值守点和临时指挥点，通信需求呈现明显的孤岛化和阶段性波动。",
      networkDamage: "该场景默认无残余网络，地面站点不可直接复用，必须依靠卫星 Ka 中继、便携热点、短波和 Mesh UAV 快速拉起临时网络。",
      modelingMethod: "将易涝区、安置点、指挥点抽象为孤岛网格与保障节点，把设备投送能力、可部署位点和区域隔离度纳入环境约束，强调从无到有的应急建网。",
      stateDef: "状态包含孤岛区域的用户密度、未覆盖网格数、可部署候选站点、设备库存、预算约束和当前临时链路可达性。",
      actionDef: "动作定义为选择候选位点投放防洪宏站、卫星 Ka 中继、WiFi6 便携热点、短波增程台和 Mesh UAV，并决定覆盖优先还是成本优先的投送策略。",
      rewardDef: "奖励函数更强调成本与覆盖平衡，既要求尽快打通孤岛通信，也要抑制高成本设备的冗余部署，适合验证零残余网络条件下的应急组网效率。",
      target: "在缺乏可复用基础设施的情况下，以尽可能少的设备和部署成本恢复安置点、指挥点和抢险作业面的通信可达性。",
      difference: "与地震、台风场景不同，洪水场景默认没有残余网络，因此核心难点不是修复已有链路，而是从零构建临时网络；同时比台风更强调孤岛分区和投送成本控制。"
    },
    typhoon_residual: {
      scenarioBackground: "台风场景面向沿海灾后残余网络恢复。主要覆盖沿海城区、港区和转移安置区域，目标是在大面积受损但仍有部分基站可用的条件下快速恢复连续通信服务。",
      disasterTraits: "台风影响范围大、覆盖连续，风暴潮和大风会造成沿海多片区同时受损，业务需求既包括居民通信恢复，也包括港区、码头和应急指挥的连续保障。",
      networkDamage: "场景存在残余网络，但部分站点降级运行、回传链路受损、供电不稳，需通过应急宏站、卫星 Ka 中继和快速热点对沿海带状区域进行连续补强。",
      modelingMethod: "将沿海灾区建模为连续网格带，用户与重点区域沿海岸线分布，设备部署既要考虑覆盖半径，也要考虑主干链路恢复和区域连续性。",
      stateDef: "状态包含沿海网格覆盖缺口、残余站点工作状态、候选位点容量、用户密度热区、业务负载等级和链路恢复进度。",
      actionDef: "动作定义为在沿海候选点部署应急宏站、卫星 Ka 中继、WiFi6 快速热点和短波应急台，并对连续覆盖与容量恢复进行动态权衡。",
      rewardDef: "奖励默认采用覆盖与容量平衡，既奖励连续覆盖恢复，也奖励吞吐和广播可达性提升，同时对设备成本与带宽成本进行约束。",
      target: "尽快恢复沿海带状区域的连续覆盖和关键链路容量，保障指挥调度、港区作业和居民安置点的通信稳定性。",
      difference: "与地震场景相比，台风场景范围更连续、覆盖带更长，重点在大范围连续补强；与洪水场景相比，台风仍可复用残余网络，因此更强调残余站协同和容量恢复，而不是从零建网。"
    }
  };

  function byId(id) {
    return document.getElementById(id);
  }

  function markInteractive(node) {
    if (node) node.dataset.liveInteractive = "true";
  }

  function numericCss(value, fallback) {
    var parsed = parseFloat(value);
    return Number.isFinite(parsed) ? parsed : (fallback || 0);
  }

  function visibleNode(node) {
    return !!(node && node.style.display !== "none" && node.style.visibility !== "hidden");
  }

  function measurePanelHeight(panelId, fallback) {
    var panel = byId(panelId);
    if (!panel) return fallback || 0;
    var visibleStates = Array.prototype.filter.call(panel.children || [], function (child) {
      return child.classList && child.classList.contains("panel_state") && visibleNode(child);
    });
    var measured = [panel.scrollHeight, panel.offsetHeight];
    visibleStates.forEach(function (stateNode) {
      measured.push(stateNode.scrollHeight, stateNode.offsetHeight);
      Array.prototype.forEach.call(stateNode.children || [], function (contentNode) {
        measured.push(contentNode.scrollHeight, contentNode.offsetHeight);
      });
    });
    var result = measured.reduce(function (maxValue, current) {
      return Math.max(maxValue, Number(current) || 0);
    }, 0);
    return result || fallback || 0;
  }

  function forcePanelState(visibleId, hiddenId) {
    var visibleNode = byId(visibleId);
    var hiddenNode = byId(hiddenId);
    if (visibleNode) {
      visibleNode.style.display = "block";
      visibleNode.style.visibility = "visible";
    }
    if (hiddenNode) {
      hiddenNode.style.display = "none";
      hiddenNode.style.visibility = "hidden";
    }
  }

  function getDeviceAccessModuleHeight() {
    var module = byId("training-device-access-module");
    if (module && module.offsetHeight) return module.offsetHeight;
    var rowCount = Array.isArray(state.accessDevices) ? state.accessDevices.length : 0;
    return rowCount ? 104 + rowCount * 48 : 86;
  }

  function getScenePanelHeight() {
    return Math.max(650, DEVICE_ACCESS_TOP + getDeviceAccessModuleHeight() + 96);
  }

  function resizeScenarioDescriptionBox() {
    [
      "u550",
      "u551",
      "u551_div",
    ].forEach(function (id) {
      var node = byId(id);
      if (!node) return;
      node.style.height = SCENARIO_DESCRIPTION_HEIGHT + "px";
    });
    var wrapper = byId("u550");
    if (wrapper) wrapper.setAttribute("data-height", String(SCENARIO_DESCRIPTION_HEIGHT));
    var box = byId("u551");
    if (box) box.style.position = "relative";
    if (editors.u551) {
      editors.u551.style.height = "100%";
      editors.u551.style.minHeight = SCENARIO_DESCRIPTION_HEIGHT + "px";
      editors.u551.style.resize = "none";
    }
    var counter = byId("u579");
    if (counter) {
      counter.style.top = (SCENARIO_DESCRIPTION_TOP + SCENARIO_DESCRIPTION_HEIGHT - 28) + "px";
      counter.style.zIndex = "20";
    }
  }

  function rewardModeOptions() {
    return [
      { value: "bandwidth_priority", label: "带宽优先" },
      { value: "cost_priority", label: "设备开销最小优先" },
      { value: "coverage_balance", label: "考虑覆盖" },
      { value: "coverage_priority", label: "覆盖优先" },
    ];
  }

  function rewardModeHint(value) {
    var hints = {
      bandwidth_priority: "吞吐保障",
      cost_priority: "设备开销",
      coverage_balance: "覆盖均衡",
      coverage_priority: "覆盖恢复",
    };
    return hints[value] || "奖励模式";
  }

  function hideAlgorithmSourceAndNotes() {
    [
      "u1139",
      "u1140",
      "u1147",
      "u1148",
      "u1149",
      "u1150",
    ].forEach(function (id) {
      var node = byId(id);
      if (!node) return;
      node.style.display = "none";
      node.style.visibility = "hidden";
      node.setAttribute("aria-hidden", "true");
    });
  }

  function compactAlgorithmPanel() {
    [
      byId("u1118"),
      byId("u1118_state0"),
    ].forEach(function (node) {
      if (!node) return;
      node.style.height = ALGORITHM_PANEL_COMPACT_HEIGHT + "px";
      node.style.maxHeight = ALGORITHM_PANEL_COMPACT_HEIGHT + "px";
      node.style.overflow = "visible";
    });
  }

  function ensureRewardConfigPanel() {
    var existing = byId("training-reward-config-panel");
    if (existing) return existing;

    var container = byId("u544_state0_content") || byId("u544_state0");
    if (!container) return null;

    var panel = document.createElement("div");
    panel.id = "training-reward-config-panel";
    panel.style.cssText = [
      "position:absolute",
      "left:3px",
      "top:0",
      "width:1630px",
      "height:" + REWARD_CONFIG_PANEL_HEIGHT + "px",
      "pointer-events:auto",
      "z-index:1"
    ].join(";");
    markInteractive(panel);

    var options = rewardModeOptions().map(function (option) {
      return "<button type='button' data-reward-mode='" + escapeHtml(option.value) + "' style='" +
        "width:" + ALGORITHM_CARD_WIDTH + "px;height:" + ALGORITHM_CARD_HEIGHT + "px;border:0;padding:12px 16px;background:transparent url(images/模型训练/u1124.png) center/" + ALGORITHM_CARD_WIDTH + "px " + ALGORITHM_CARD_HEIGHT + "px no-repeat;" +
        "appearance:none;-webkit-appearance:none;" +
        "color:#333;font-family:思源黑体 CN,Microsoft YaHei,sans-serif;text-align:left;cursor:pointer;box-sizing:border-box;'>" +
        "<span style='display:block;font-weight:700;font-size:18px;line-height:28px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>" + escapeHtml(option.label) + "</span>" +
        "<span style='display:block;margin-top:22px;font-size:16px;line-height:22px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>" + escapeHtml(rewardModeHint(option.value)) + "</span>" +
        "</button>";
    }).join("");

    panel.innerHTML =
      "<div style='position:absolute;left:6px;top:39px;width:112px;height:2px;background:linear-gradient(90deg,#03b4f5,#05b7df);'></div>" +
      "<div style='position:absolute;left:14px;top:6px;width:126px;height:33px;font-family:思源黑体 CN,Microsoft YaHei,sans-serif;font-weight:700;font-size:16px;color:#0f172a;line-height:33px;'>奖励配置设置</div>" +
      "<div id='training-reward-options' style='position:absolute;left:58px;top:64px;width:" + ((ALGORITHM_CARD_WIDTH * 4) + (ALGORITHM_CARD_GAP * 3)) + "px;height:" + ALGORITHM_CARD_HEIGHT + "px;display:flex;gap:" + ALGORITHM_CARD_GAP + "px;'>" + options + "</div>";

    container.appendChild(panel);
    Array.prototype.forEach.call(panel.querySelectorAll("[data-reward-mode]"), function (button) {
      markInteractive(button);
      button.addEventListener("click", function (event) {
        event.preventDefault();
        event.stopPropagation();
        state.rewardMode = button.getAttribute("data-reward-mode") || "coverage_balance";
        syncRewardConfigPanel();
        addConsole("info", "已切换奖励配置：" + rewardModeLabel(state.rewardMode));
      }, true);
    });
    syncRewardConfigPanel();
    return panel;
  }

  function syncRewardConfigPanel() {
    var valid = rewardModeOptions().some(function (option) {
      return option.value === state.rewardMode;
    });
    if (!valid) state.rewardMode = "coverage_balance";
    setChoiceText("u571", rewardModeLabel(state.rewardMode));

    var panel = byId("training-reward-config-panel");
    if (!panel) return;
    Array.prototype.forEach.call(panel.querySelectorAll("[data-reward-mode]"), function (button) {
      var selected = button.getAttribute("data-reward-mode") === state.rewardMode;
      button.style.backgroundImage = selected ? "url(images/模型训练/u1124_selected.png)" : "url(images/模型训练/u1124.png)";
    });
  }

  function relayoutSections() {
    var scenePanel = byId("u545");
    var algoPanel = byId("u1118");
    var parameterPanel = byId("u775");
    var resultPanel = byId("u734");
    var rewardPanel = ensureRewardConfigPanel();
    var foldContainer = byId("u544_state0");
    var foldContent = byId("u544_state0_content");
    var scrollContent = byId("u543_state0_content");
    var scrollPanel = byId("u543_state0");
    if (!scenePanel || !algoPanel || !parameterPanel || !resultPanel) return;

    hideAlgorithmSourceAndNotes();
    compactAlgorithmPanel();
    syncRewardConfigPanel();

    var sceneHeight = getScenePanelHeight();
    var algoTop = sceneHeight + 28;
    var rewardTop = algoTop + ALGORITHM_PANEL_COMPACT_HEIGHT + SECTION_GAP;
    var parameterTop = rewardTop + REWARD_CONFIG_PANEL_HEIGHT + SECTION_GAP;
    var resultTop = parameterTop + 314;
    var resultVisible = visibleNode(byId("u735"));
    var resultHeight = resultVisible ? 824 : 49;
    var totalHeight = resultTop + resultHeight + 40;

    [
      { node: scenePanel, left: 2, top: 0, zIndex: 1 },
      { node: algoPanel, left: 3, top: algoTop, zIndex: 1 },
      { node: rewardPanel, left: 3, top: rewardTop, zIndex: 1 },
      { node: parameterPanel, left: 2, top: parameterTop, zIndex: 1 },
      { node: resultPanel, left: 2, top: resultTop, zIndex: 1 },
    ].forEach(function (section) {
      if (!section.node) return;
      section.node.style.position = "absolute";
      section.node.style.left = section.left + "px";
      section.node.style.top = section.top + "px";
      section.node.style.margin = "0";
      section.node.style.zIndex = String(section.zIndex);
    });
    scenePanel.style.height = sceneHeight + "px";

    if (foldContainer) {
      foldContainer.style.height = totalHeight + "px";
      foldContainer.style.position = "relative";
    }
    if (foldContent) {
      foldContent.style.display = "block";
      foldContent.style.height = totalHeight + "px";
      foldContent.style.position = "absolute";
    }
    if (scrollContent) {
      scrollContent.style.height = totalHeight + "px";
      scrollContent.style.position = "relative";
    }
    if (scrollPanel) {
      scrollPanel.style.height = "848px";
      scrollPanel.style.position = "relative";
    }
  }

  function protectPanelState(node, visibleId, hiddenId) {
    if (!node || !visibleId || !hiddenId || node.dataset.livePanelProtected) return;
    node.dataset.livePanelProtected = "true";
    ["pointerdown", "mousedown", "click", "focusin"].forEach(function (eventName) {
      node.addEventListener(eventName, function () {
        setTimeout(function () {
          forcePanelState(visibleId, hiddenId);
          relayoutSections();
        }, 0);
      }, true);
    });
  }

  function stopBubbleOnly(node) {
    if (!node || node.dataset.liveBubbleStopped) return;
    node.dataset.liveBubbleStopped = "true";
    ["mousedown", "mouseup", "click", "dblclick", "focusin", "pointerdown", "pointerup"].forEach(function (eventName) {
      node.addEventListener(eventName, function (event) {
        event.stopPropagation();
      }, false);
    });
  }

  function disableToggleClick(node) {
    if (!node || node.dataset.liveToggleDisabled) return;
    node.dataset.liveToggleDisabled = "true";
    ["mousedown", "mouseup", "click", "dblclick", "pointerdown", "pointerup"].forEach(function (eventName) {
      node.addEventListener(eventName, function (event) {
        event.preventDefault();
        event.stopPropagation();
      }, false);
    });
  }

  function unbindAxureHandlers(ids) {
    var jq = window.jQuery || window.$;
    ids.forEach(function (id) {
      var node = byId(id);
      if (!node || node.dataset.liveAxureUnbound) return;
      node.dataset.liveAxureUnbound = "true";
      node.onclick = null;
      node.onmousedown = null;
      node.onmouseup = null;
      if (jq) {
        try {
          jq(node).off();
        } catch (error) {
        }
      }
    });
  }

  function unbindAxureSubtree(root) {
    var jq = window.jQuery || window.$;
    if (!root || root.dataset.liveAxureSubtreeUnbound) return;
    root.dataset.liveAxureSubtreeUnbound = "true";
    var nodes = [root].concat(Array.prototype.slice.call(root.querySelectorAll("*")));
    nodes.forEach(function (node) {
      node.onclick = null;
      node.onmousedown = null;
      node.onmouseup = null;
      if (jq) {
        try {
          jq(node).off();
        } catch (error) {
        }
      }
    });
  }

  function lockPanelExpanded(panelId, visibleId, hiddenId, toggleIds) {
    var panel = byId(panelId);
    var visibleNode = byId(visibleId);
    var hiddenNode = byId(hiddenId);
    forcePanelState(visibleId, hiddenId);
    if (hiddenNode) hiddenNode.style.pointerEvents = "none";
    (toggleIds || []).forEach(function (id) {
      disableToggleClick(byId(id));
    });
    if (!panel || panel.dataset.liveExpandLocked) return;
    panel.dataset.liveExpandLocked = "true";
    var observer = new MutationObserver(function () {
      forcePanelState(visibleId, hiddenId);
      relayoutSections();
    });
    [panel, visibleNode, hiddenNode].forEach(function (node) {
      if (!node) return;
      observer.observe(node, {
        attributes: true,
        attributeFilter: ["style", "class"],
      });
    });
  }

  function stabilizeEditablePanel(contentId, visibleId, hiddenId) {
    forcePanelState(visibleId, hiddenId);
    var content = byId(contentId);
    if (!content) return;
    content.style.pointerEvents = "auto";
    stopBubbleOnly(content);
    protectPanelState(content, visibleId, hiddenId);
  }

  function installBlankAreaGuard(contentId, visibleId, hiddenId) {
    var content = byId(contentId);
    if (!content || content.dataset.liveBlankGuard) return;
    content.dataset.liveBlankGuard = "true";
    ["pointerdown", "mousedown", "click", "dblclick"].forEach(function (eventName) {
      content.addEventListener(eventName, function (event) {
        var target = event.target;
        if (!target) return;
        if (target.closest("[data-live-interactive='true']")) return;
        if (target.closest("input, textarea, select, button")) return;
        event.preventDefault();
        event.stopImmediatePropagation();
        forcePanelState(visibleId, hiddenId);
        relayoutSections();
      }, true);
    });
  }

  function isolatePointerEvents(node) {
    if (!node || node.dataset.liveIsolated) return;
    node.dataset.liveIsolated = "true";
    ["mousedown", "mouseup", "click", "dblclick", "focus", "pointerdown", "pointerup"].forEach(function (eventName) {
      node.addEventListener(eventName, function (event) {
        event.stopPropagation();
      }, true);
    });
  }

  function ensureLiveSelectStyle() {
    if (document.getElementById("training-live-select-style")) return;
    var style = document.createElement("style");
    style.id = "training-live-select-style";
    style.textContent = [
      "[data-live-select-arrow='true']{pointer-events:none!important;}",
      "[data-live-select-arrow='true'] [id$='_div']{background:transparent!important;border:none!important;box-shadow:none!important;}",
      "[data-live-select-arrow-text='true']{position:absolute!important;left:0!important;top:0!important;width:100%!important;height:100%!important;display:flex!important;align-items:center!important;justify-content:center!important;}",
      "[data-live-select-arrow-text='true']>*{display:none!important;}",
      "[data-live-select-arrow-text='true']::before{content:'';display:block;width:15px;height:15px;background-repeat:no-repeat;background-position:center;background-size:15px 15px;background-image:url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23999999' stroke-width='2.4' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='m6 9 6 6 6-6'/%3E%3C/svg%3E\");}"
    ].join("\n");
    document.head && document.head.appendChild(style);
  }

  function isDropdownArrowGlyph(value) {
    return /^[\s\u00a0]*[\uf0dc\uf107\uf106\uf078\uf077\uf0d7\uf0d8\uf0dd\uf0de][\s\u00a0]*$/.test(value || "");
  }

  function markDropdownArrowNode(node, textNode) {
    if (!node || !textNode) return;
    node.dataset.liveSelectArrow = "true";
    node.style.pointerEvents = "none";
    node.style.zIndex = "14";
    textNode.dataset.liveSelectArrowText = "true";
  }

  function replaceSelectArrowGlyphs(panel, choice, fakeMenu) {
    if (!panel || !choice) return;
    ensureLiveSelectStyle();
    Array.prototype.forEach.call(panel.querySelectorAll("[id]"), function (node) {
      if (!node || node === panel || node === choice || node.contains(choice) || choice.contains(node)) return;
      if (fakeMenu && fakeMenu.contains(node)) return;
      if (/_text$|_div$|_state\d/.test(node.id)) return;
      var textNode = node.querySelector("[id$='_text']");
      if (!textNode) return;
      var rawText = textNode.textContent || "";
      var styleText = (node.getAttribute("style") || "") + " " + (textNode.getAttribute("style") || "");
      var fontFamily = "";
      try {
        fontFamily = (window.getComputedStyle(node).fontFamily || "") + " " + (window.getComputedStyle(textNode).fontFamily || "");
      } catch (error) {}
      var looksLikeIcon = isDropdownArrowGlyph(rawText) || /Font\s*Awesome|FontAwesome/i.test(styleText + " " + fontFamily);
      if (!looksLikeIcon) return;
      markDropdownArrowNode(node, textNode);
    });
  }

  function replaceAllDropdownArrowGlyphs() {
    ensureLiveSelectStyle();
    Array.prototype.forEach.call(document.querySelectorAll("[id]"), function (node) {
      if (!node || /_text$|_div$|_state\d/.test(node.id)) return;
      var textNode = node.querySelector("[id$='_text']");
      if (!textNode) return;
      if (!isDropdownArrowGlyph(textNode.textContent || "")) return;
      markDropdownArrowNode(node, textNode);
    });
  }

  function setPanelVisible(id, visible) {
    var node = byId(id);
    if (!node) return;
    node.style.display = visible ? "block" : "none";
    node.style.visibility = visible ? "visible" : "hidden";
    node.classList.toggle("ax_default_hidden", !visible);
    relayoutSections();
  }

  function setChoiceText(id, text) {
    var node = byId(id);
    if (!node) return;
    var textWrap = node.querySelector('[id$="_text"]');
    if (textWrap) {
      textWrap.textContent = text;
      return;
    }
    node.textContent = text;
  }

  function scenarioDisplayName(scenario) {
    if (!scenario) return "未选择场景";
    var typeMap = {
      flood: "洪水孤岛通信恢复",
      earthquake: "地震灾后断链恢复",
      typhoon: "台风灾后残余网络",
    };
    return typeMap[scenario.disaster_type] || scenario.name;
  }

  function disasterTypeLabel(type) {
    var labelMap = {
      flood: "洪水",
      earthquake: "地震",
      typhoon: "台风",
    };
    return labelMap[type] || type || "--";
  }

  function rewardModeLabel(key) {
    var labelMap = {
      bandwidth_priority: "带宽优先",
      cost_priority: "设备开销最小优先",
      coverage_balance: "考虑覆盖",
      coverage_priority: "覆盖优先",
    };
    return labelMap[key] || key || "--";
  }

  function scenarioGridShape(scenario) {
    var rows = scenario && scenario.region_grid && scenario.region_grid.rows
      ? Number(scenario.region_grid.rows)
      : Number(scenario && scenario.grid_size ? scenario.grid_size : 1);
    var cols = scenario && scenario.region_grid && scenario.region_grid.cols
      ? Number(scenario.region_grid.cols)
      : Number(scenario && scenario.grid_size ? scenario.grid_size : 1);
    rows = Number.isFinite(rows) && rows > 0 ? Math.round(rows) : 1;
    cols = Number.isFinite(cols) && cols > 0 ? Math.round(cols) : 1;
    return { rows: rows, cols: cols, count: rows * cols };
  }

  function scenarioGeoBoundsText(scenario) {
    var bounds = scenario && scenario.region_grid && scenario.region_grid.geo_bounds;
    if (!bounds) return "未配置经纬度边界";
    return "纬度 " + bounds.lat_min + "-" + bounds.lat_max + "，经度 " + bounds.lon_min + "-" + bounds.lon_max;
  }

  function scenarioCellLabelsText(scenario) {
    var labels = scenario && scenario.region_grid && scenario.region_grid.cell_labels;
    if (!labels || typeof labels !== "object") return "未配置重点保障网格";
    var items = Object.keys(labels).map(function (key) {
      return labels[key] + "（" + key + "）";
    });
    return items.length ? items.join("、") : "未配置重点保障网格";
  }

  function scenarioRewardProfilesText(scenario) {
    var profiles = scenario && scenario.reward_profiles ? scenario.reward_profiles : [];
    if (!Array.isArray(profiles)) {
      profiles = Object.keys(profiles).map(function (key) {
        return Object.assign({ key: key }, profiles[key] || {});
      });
    }
    return profiles.map(function (profile) {
      return rewardModeLabel(profile.key || profile.label);
    }).filter(Boolean).join(" / ") || "--";
  }

  function scenarioDeviceLibraryText(scenario) {
    return baseStationProfilesForScenario(scenario).map(function (profile) {
      return deviceProfileLabel(profile) + "（" + deviceModesLabel(profile) + "）";
    }).filter(Boolean).join("、") || "--";
  }

  function scenarioBasicNotes(scenario) {
    if (!scenario) return "";
    var detail = buildScenarioModelingDetail(scenario);
    var grid = scenarioGridShape(scenario);
    return [
      "场景背景：" + (detail ? detail.scenarioBackground : scenarioDisplayName(scenario)),
      "网络特征：" + (scenario.has_residual_network ? "存在可复用残余网络" : "无残余网络，需要从零构建临时通信能力") +
        "；候选站点 " + Number(scenario.candidate_sites || 0).toLocaleString("zh-CN") +
        " 个；训练最大步长 " + Number(scenario.max_steps || 0).toLocaleString("zh-CN") + "。",
      "区域网格：" + grid.rows + " 行 × " + grid.cols + " 列，共 " + grid.count.toLocaleString("zh-CN") +
        " 个网格；用户规模 " + Number(scenario.num_users || 0).toLocaleString("zh-CN") + "。",
      "设备库：" + scenarioDeviceLibraryText(scenario),
      "奖励配置：默认 考虑覆盖；可选 " + scenarioRewardProfilesText(scenario)
    ].join("\n");
  }

  function syncScenarioBasicInfoLabels() {
    setRequiredLabelText("u573", "区域网格数");
    setRequiredLabelText("u576", "用户规模");
    setRequiredLabelText("u563", "默认奖励模式");
    setRequiredLabelText("u652", "地理覆盖范围");
    setRequiredLabelText("u655", "网格规模");
    setRequiredLabelText("u635", "重点保障区域");
    setRequiredLabelText("u552", "场景说明");
  }

  function currentScenario() {
    return state.scenarios.find(function (item) {
      return item.name === state.scenarioName;
    }) || null;
  }

  function currentRewardMode() {
    return state.rewardMode || "coverage_balance";
  }

  function currentScenarioForType(type) {
    return state.scenarios.find(function (item) {
      return item.disaster_type === type;
    }) || state.scenarios[0] || null;
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
    if (found) return found.label;
    if (!type) return "--";
    var label = String(type);
    if (/^5g/i.test(label)) return "蜂窝 " + label.replace(/_/g, " ");
    if (/^satellite/i.test(label)) return label.replace(/^Satellite/i, "卫星").replace(/_/g, " ");
    if (/^shortwave|hf/i.test(label)) return label.replace(/^Shortwave/i, "短波").replace(/_/g, " ");
    return label.replace(/_/g, " ");
  }

  function numericValue(value, fallback) {
    var next = Number(value);
    return Number.isFinite(next) ? next : fallback;
  }

  function integerValue(value, fallback) {
    var next = Number(value);
    return Number.isFinite(next) ? Math.max(0, Math.round(next)) : fallback;
  }

  function deviceTypeValue(label) {
    var mapping = {
      "宏基站": "macro_station",
      "背负式基站": "backpack_station",
      "中继设备": "relay",
      "临时设备/车载设备": "vehicle_station",
    };
    return mapping[label] || "macro_station";
  }

  function baseStationProfilesForScenario(scenario) {
    var baseStations = scenario && scenario.base_stations ? scenario.base_stations : [];
    if (Array.isArray(baseStations)) {
      return baseStations.filter(Boolean);
    }
    if (baseStations && typeof baseStations === "object") {
      return Object.keys(baseStations).map(function (key) {
        var profile = baseStations[key] || {};
        return Object.assign({ name: profile.name || key }, profile);
      });
    }
    return [];
  }

  function fallbackDeviceProfiles() {
    return DEFAULT_DEVICE_TEMPLATES.map(function (item) {
      return {
        name: item.id || item.name,
        label: item.name,
        deviceType: item.deviceType,
        max_throughput: item.maxThroughput,
        max_users: item.maxUsers,
        supported_modes: item.communicationType ? [item.communicationType] : [],
      };
    });
  }

  function currentScenarioDeviceProfiles() {
    var profiles = baseStationProfilesForScenario(currentScenario());
    if (profiles.length) return profiles;
    return state.scenarios.length ? [] : fallbackDeviceProfiles();
  }

  function deviceProfileValue(profile) {
    return profile ? String(profile.name || profile.id || profile.label || "") : "";
  }

  function deviceProfileLabel(profile) {
    return profile ? String(profile.label || profile.name || profile.id || "") : "";
  }

  function deviceProfileModes(profile) {
    if (!profile) return [];
    if (Array.isArray(profile.supported_modes)) return profile.supported_modes.filter(Boolean);
    if (profile.communicationType) return [profile.communicationType];
    return [];
  }

  function deviceModesLabel(profile) {
    var modes = deviceProfileModes(profile);
    return modes.length ? modes.map(communicationLabel).join(" / ") : "--";
  }

  function deviceTypeForProfile(profile) {
    if (!profile) return "";
    if (profile.deviceType) return deviceTypeValue(profile.deviceType);
    var text = [profile.name, profile.label].concat(deviceProfileModes(profile)).join(" ");
    if (/wifi/i.test(text)) return "backpack_station";
    if (/shortwave|hf|短波/i.test(text)) return "vehicle_station";
    if (/satellite|relay|mesh|uav|卫星|中继|无人机/i.test(text)) return "relay";
    return "macro_station";
  }

  function buildDeviceOptions() {
    return [{ value: "", label: "请选择设备" }].concat(currentScenarioDeviceProfiles().map(function (item) {
      return {
        value: deviceProfileValue(item),
        label: deviceProfileLabel(item) + "（" + deviceModesLabel(item) + "）",
      };
    }));
  }

  function findDeviceTemplateByName(name) {
    if (!name) return null;
    var value = String(name);
    return currentScenarioDeviceProfiles().find(function (item) {
      return deviceProfileValue(item) === value || deviceProfileLabel(item) === value || item.name === value || item.id === value;
    });
  }

  function defaultDeviceForSlot(index) {
    var profiles = currentScenarioDeviceProfiles();
    return profiles[0] || null;
  }

  function currentScenarioGridBounds() {
    var scenario = currentScenario();
    var rows = scenario && scenario.region_grid && scenario.region_grid.rows
      ? Number(scenario.region_grid.rows)
      : Number(scenario && scenario.grid_size ? scenario.grid_size : 1);
    var cols = scenario && scenario.region_grid && scenario.region_grid.cols
      ? Number(scenario.region_grid.cols)
      : Number(scenario && scenario.grid_size ? scenario.grid_size : 1);
    rows = Number.isFinite(rows) && rows > 0 ? Math.round(rows) : 1;
    cols = Number.isFinite(cols) && cols > 0 ? Math.round(cols) : 1;
    return {
      rows: rows,
      cols: cols,
      maxX: Math.max(0, rows - 1),
      maxY: Math.max(0, cols - 1),
    };
  }

  function clampGridIndex(value, maxValue, fallback) {
    var next = integerValue(value, fallback == null ? 0 : fallback);
    return Math.min(Math.max(0, next), Math.max(0, integerValue(maxValue, 0)));
  }

  function parseLocationToGrid(location) {
    if (!location) return null;
    var parts = String(location).match(/-?\d+/g);
    if (!parts || parts.length < 2) return null;
    return {
      x: Number(parts[0]),
      y: Number(parts[1]),
    };
  }

  function accessSlotKeys(index) {
    var prefix = index < 2 ? "residual" : "candidate";
    var suffix = index < 2 ? index + 1 : index - 1;
    return {
      type: prefix + "Type" + suffix,
      device: prefix + "Device" + suffix,
      count: prefix + "Count" + suffix,
      location: prefix === "residual" ? prefix + "Location" + suffix : null,
    };
  }

  function readAccessSlot(index) {
    var rows = Array.isArray(state.accessDevices) ? state.accessDevices : [];
    var row = rows[index] || {};
    var bounds = currentScenarioGridBounds();
    var parsedLocation = parseLocationToGrid(row.location);
    return {
      type: row.type || "",
      device: row.device || "",
      count: integerValue(row.count, 0),
      x: clampGridIndex(row.x != null ? row.x : (parsedLocation ? parsedLocation.x : 0), bounds.maxX, 0),
      y: clampGridIndex(row.y != null ? row.y : (parsedLocation ? parsedLocation.y : 0), bounds.maxY, 0),
      location: row.location || "",
    };
  }

  function writeAccessSlot(index, slot) {
    var template = findDeviceTemplateByName(slot && slot.device);
    var rows = Array.isArray(state.accessDevices) ? state.accessDevices.slice() : [];
    var bounds = currentScenarioGridBounds();
    var parsedLocation = parseLocationToGrid(slot && slot.location);
    var x = clampGridIndex(slot && slot.x != null ? slot.x : (parsedLocation ? parsedLocation.x : 0), bounds.maxX, 0);
    var y = clampGridIndex(slot && slot.y != null ? slot.y : (parsedLocation ? parsedLocation.y : 0), bounds.maxY, 0);
    rows[index] = {
      type: template ? deviceTypeForProfile(template) : (slot && slot.type ? slot.type : ""),
      device: template ? deviceProfileValue(template) : (slot && slot.device ? slot.device : ""),
      count: Math.max(1, integerValue(slot && slot.count, 1)),
      x: x,
      y: y,
      location: x + "," + y,
    };
    state.accessDevices = rows.filter(function (row) {
      return row && row.device;
    }).slice(0, DEVICE_ACCESS_MAX);
  }

  function normalizeTrainingDeviceSlot(index) {
    var row = readAccessSlot(index);
    var template = findDeviceTemplateByName(row.device);
    if (template) {
      writeAccessSlot(index, {
        type: deviceTypeForProfile(template),
        device: deviceProfileValue(template),
        count: row.count,
        x: row.x,
        y: row.y,
        location: row.location,
      });
    }
  }

  function syncLegacyAccessFields() {
    [0, 1, 2, 3].forEach(function (index) {
      var keys = accessSlotKeys(index);
      var row = index < state.accessDevices.length ? readAccessSlot(index) : { type: "", device: "", count: 0, x: 0, y: 0, location: "" };
      state[keys.type] = row.type || "";
      state[keys.device] = row.device || "";
      state[keys.count] = integerValue(row.count, 0);
      if (keys.location) state[keys.location] = row.device ? (row.x + "," + row.y) : "";
    });
  }

  function normalizeTrainingDeviceState() {
    var rows = Array.isArray(state.accessDevices) ? state.accessDevices : [];
    var bounds = currentScenarioGridBounds();
    state.accessDevices = rows.map(function (row) {
      if (!row || !row.device) return null;
      var template = findDeviceTemplateByName(row.device);
      if (!template) return null;
      var parsedLocation = parseLocationToGrid(row.location);
      var x = clampGridIndex(row.x != null ? row.x : (parsedLocation ? parsedLocation.x : 0), bounds.maxX, 0);
      var y = clampGridIndex(row.y != null ? row.y : (parsedLocation ? parsedLocation.y : 0), bounds.maxY, 0);
      return {
        type: deviceTypeForProfile(template),
        device: deviceProfileValue(template),
        count: Math.max(1, integerValue(row.count, 1)),
        x: x,
        y: y,
        location: x + "," + y,
      };
    }).filter(Boolean).slice(0, DEVICE_ACCESS_MAX);
    state.accessSlotCount = state.accessDevices.length;
    syncLegacyAccessFields();
  }

  function syncTrainingDeviceSlot(index, typeEditorId) {
    var keys = accessSlotKeys(index);
    normalizeTrainingDeviceSlot(index);
    setEditorValue(typeEditorId, state[keys.type], "change");
    syncPriorityEquipmentText();
    syncDeviceSectionVisibility();
  }

  function summarizeTrainingDevice(deviceName, count) {
    var template = findDeviceTemplateByName(deviceName);
    if (!template) return "";
    return deviceProfileLabel(template) + "（" + deviceModesLabel(template) + "）x" + Math.max(1, integerValue(count, 1));
  }

  function activeDeviceSummaries() {
    return state.accessDevices.map(function (row, index) {
      var slot = readAccessSlot(index);
      return summarizeTrainingDevice(slot.device, slot.count);
    }).filter(Boolean);
  }

  function trainingDeviceSummaryLabel() {
    var devices = activeDeviceSummaries();
    return "接入设备：" + (devices.length ? devices.join("、") : "未配置");
  }

  function syncPriorityEquipmentText() {
    var devices = activeDeviceSummaries();
    var summary = devices.length ? devices.join(" + ") : "未配置接入设备";
    state.priorityEquipment = summary;
    if (editors.u651 && !editors.u651.dataset.userEdited) {
      setEditorValue("u651", summary, "input");
    }
  }

  function syncTrainingDeviceFields() {
    normalizeTrainingDeviceState();
    [
      ["u592", state.residualType1],
      ["u603", state.residualDevice1],
      ["u607", state.residualLocation1],
      ["u726", state.residualCount1],
      ["u624", state.residualType2],
      ["u633", state.residualDevice2],
      ["u613", state.residualLocation2],
      ["u729", state.residualCount2],
      ["u673", state.candidateType1],
      ["u684", state.candidateDevice1],
      ["u689", state.candidateCount1],
      ["u698", state.candidateType2],
      ["u708", state.candidateDevice2],
      ["u712", state.candidateCount2],
    ].forEach(function (entry) {
      if (editors[entry[0]]) setEditorValue(entry[0], entry[1], editors[entry[0]].tagName === "SELECT" ? "change" : "input");
    });
    syncDeviceSectionVisibility();
    syncPriorityEquipmentText();
  }

  function setNodesVisible(ids, visible) {
    ids.forEach(function (id) {
      var node = byId(id);
      if (!node) return;
      node.style.display = visible ? "block" : "none";
      node.style.visibility = visible ? "visible" : "hidden";
      node.classList.toggle("ax_default_hidden", !visible);
    });
  }

  function hideObsoleteSceneFields() {
    setNodesVisible([
      "u580", "u581", "u582", "u583",
      "u584", "u585", "u595", "u596", "u605", "u606", "u608", "u724", "u725", "u615",
      "u609", "u617", "u610", "u626", "u611", "u612", "u614", "u727", "u728", "u616",
      "u661", "u662", "u663", "u664", "u675",
      "u665", "u666", "u676", "u677", "u687", "u688",
      "u690", "u691", "u700", "u701", "u710", "u711", "u686",
      "u646", "u647", "u648",
      "u649", "u650", "u651",
      "u715", "u716", "u717",
      "u718", "u719", "u720",
      "u721", "u722", "u723"
    ], false);
  }

  function ensureDeviceAccessModule() {
    var host = byId("u545_state0_content");
    if (!host) return null;
    var module = byId("training-device-access-module");
    if (module) {
      module.style.top = DEVICE_ACCESS_TOP + "px";
      return module;
    }
    module = document.createElement("div");
    module.id = "training-device-access-module";
    module.style.cssText = [
      "position:absolute",
      "left:7px",
      "top:" + DEVICE_ACCESS_TOP + "px",
      "width:1593px",
      "min-height:78px",
      "box-sizing:border-box",
      "z-index:28",
      "font-size:13px",
      "font-family:'Microsoft YaHei','PingFang SC',sans-serif",
      "color:#334155"
    ].join(";");
    host.appendChild(module);
    markInteractive(module);
    stopBubbleOnly(module);
    return module;
  }

  function renderDeviceAccessModule() {
    var module = ensureDeviceAccessModule();
    if (!module) return;
    normalizeTrainingDeviceState();
    var rows = state.accessDevices;
    var deviceOptions = buildDeviceOptions();
    var supportedDeviceCount = Math.max(0, deviceOptions.length - 1);
    var bounds = currentScenarioGridBounds();
    var gridColumns = "48px minmax(280px,1.2fr) 170px minmax(210px,1fr) 100px 110px 110px 74px";
    var body = rows.length ? rows.map(function (row, index) {
      var template = findDeviceTemplateByName(row.device);
      var communication = template ? deviceModesLabel(template) : "--";
      var typeLabel = SCENE_DEVICE_TYPE_OPTIONS.find(function (item) { return item.value === row.type; });
      return "<div data-device-access-row='" + index + "' style='display:grid;grid-template-columns:" + gridColumns + ";align-items:center;gap:10px;min-height:48px;border-top:1px solid #edf2f7;'>" +
        "<div style='color:#64748b;text-align:center;'>" + (index + 1) + "</div>" +
        "<select data-access-field='device' style='width:100%;height:34px;border:1px solid #d7e3f4;border-radius:6px;padding:0 10px;background:#fff;color:#17315d;'>" +
        deviceOptions.map(function (option) {
          return "<option value='" + escapeHtml(option.value) + "'" + (option.value === row.device ? " selected" : "") + ">" + escapeHtml(option.label) + "</option>";
        }).join("") +
        "</select>" +
        "<div style='height:34px;line-height:34px;padding:0 10px;border:1px solid #edf2f7;border-radius:6px;background:#f8fafc;color:#475569;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>" + escapeHtml(typeLabel ? typeLabel.label : "--") + "</div>" +
        "<div style='height:34px;line-height:34px;padding:0 10px;border:1px solid #edf2f7;border-radius:6px;background:#f8fafc;color:#475569;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>" + escapeHtml(communication) + "</div>" +
        "<input data-access-field='count' type='number' min='1' step='1' value='" + escapeHtml(row.count) + "' style='width:100%;height:34px;border:1px solid #d7e3f4;border-radius:6px;padding:0 10px;box-sizing:border-box;'>" +
        "<input data-access-field='x' type='number' min='0' max='" + escapeHtml(bounds.maxX) + "' step='1' value='" + escapeHtml(row.x) + "' title='x 为区域网格行索引，范围 0-" + escapeHtml(bounds.maxX) + "' style='width:100%;height:34px;border:1px solid #d7e3f4;border-radius:6px;padding:0 10px;box-sizing:border-box;'>" +
        "<input data-access-field='y' type='number' min='0' max='" + escapeHtml(bounds.maxY) + "' step='1' value='" + escapeHtml(row.y) + "' title='y 为区域网格列索引，范围 0-" + escapeHtml(bounds.maxY) + "' style='width:100%;height:34px;border:1px solid #d7e3f4;border-radius:6px;padding:0 10px;box-sizing:border-box;'>" +
        "<button type='button' data-access-action='remove' style='height:32px;border:1px solid #f1c7c7;border-radius:6px;background:#fff;color:#b42318;cursor:pointer;'>移除</button>" +
      "</div>";
    }).join("") : "<div id='training-device-empty' style='height:38px;line-height:38px;color:#64748b;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;border-top:1px solid #edf2f7;'>暂无设备接入，点击右侧“添加设备”按当前场景默认接入 1 台设备，并配置 x/y 网格位置</div>";

    module.innerHTML =
      "<div style='display:flex;align-items:center;justify-content:space-between;height:38px;'>" +
      "<div style='display:flex;align-items:center;gap:10px;'>" +
      "<div style='width:8px;height:22px;background:#3961f6;'></div>" +
      "<div style='font-size:16px;color:#111827;font-weight:600;'>设备接入模块</div>" +
      "<div style='font-size:12px;color:#64748b;'>当前场景支持 " + supportedDeviceCount + " 类设备，已接入 " + rows.length + " 类；x 0-" + bounds.maxX + "，y 0-" + bounds.maxY + "</div>" +
      "</div>" +
      "<button type='button' data-access-action='add' style='height:34px;padding:0 16px;border:1px solid #b7e0fe;border-radius:6px;background:#3961f6;color:#fff;cursor:pointer;'>+ 添加设备</button>" +
      "</div>" +
      "<div style='display:grid;grid-template-columns:" + gridColumns + ";align-items:center;gap:10px;height:34px;margin-top:8px;color:#64748b;background:#f8fafc;border:1px solid #edf2f7;border-left:0;border-right:0;'>" +
      "<div style='text-align:center;'>序号</div><div>接入设备</div><div>设备类型</div><div>通信方式</div><div>数量</div><div>x（行）</div><div>y（列）</div><div>操作</div>" +
      "</div>" +
      body;

    Array.prototype.forEach.call(module.querySelectorAll("input,select,button"), function (node) {
      markInteractive(node);
      isolatePointerEvents(node);
    });
    bindDeviceAccessModuleEvents(module);
    positionSceneActionButtons();
  }

  function bindDeviceAccessModuleEvents(module) {
    Array.prototype.forEach.call(module.querySelectorAll("[data-access-action='add']"), function (button) {
      button.addEventListener("click", function (event) {
        event.preventDefault();
        event.stopPropagation();
        addTrainingDeviceSlot();
      }, true);
    });
    Array.prototype.forEach.call(module.querySelectorAll("[data-device-access-row]"), function (rowNode) {
      var index = Number(rowNode.getAttribute("data-device-access-row"));
      var remove = rowNode.querySelector("[data-access-action='remove']");
      if (remove) {
        remove.addEventListener("click", function (event) {
          event.preventDefault();
          event.stopPropagation();
          removeTrainingDeviceSlot(index);
        }, true);
      }
      Array.prototype.forEach.call(rowNode.querySelectorAll("[data-access-field]"), function (field) {
        field.addEventListener("input", function () {
          updateAccessRowFromField(index, field);
        });
        field.addEventListener("change", function () {
          updateAccessRowFromField(index, field);
        });
      });
    });
  }

  function updateAccessRowFromField(index, field) {
    var row = readAccessSlot(index);
    var key = field.getAttribute("data-access-field");
    if (key === "device") {
      var template = findDeviceTemplateByName(field.value);
      row.device = template ? deviceProfileValue(template) : "";
      row.type = template ? deviceTypeForProfile(template) : "";
      if (template && !row.count) row.count = 1;
      writeAccessSlot(index, row);
      renderDeviceAccessModule();
    } else if (key === "count") {
      row.count = Math.max(1, integerValue(field.value, 1));
      field.value = String(row.count);
      state.accessDevices[index] = row;
      syncLegacyAccessFields();
      syncPriorityEquipmentText();
    } else if (key === "x" || key === "y") {
      var bounds = currentScenarioGridBounds();
      row[key] = clampGridIndex(field.value, key === "x" ? bounds.maxX : bounds.maxY, 0);
      field.value = String(row[key]);
      row.location = clampGridIndex(row.x, bounds.maxX, 0) + "," + clampGridIndex(row.y, bounds.maxY, 0);
      state.accessDevices[index] = row;
      syncLegacyAccessFields();
    }
    relayoutSections();
  }

  function positionSceneActionButtons() {
    [
      { id: "u713", left: 1282 },
      { id: "u714", left: 1428 },
    ].forEach(function (item) {
      var node = byId(item.id);
      if (!node) return;
      styleSceneActionButton(node);
      node.style.position = "absolute";
      node.style.left = item.left + "px";
      node.style.top = "8px";
      node.style.width = "118px";
      node.style.height = "36px";
      node.style.zIndex = "35";
    });
  }

  function styleSceneActionButton(node, tone) {
    if (!node) return;
    node.style.display = "block";
    node.style.boxSizing = "border-box";
    node.style.overflow = "hidden";
    node.style.fontFamily = "'思源黑体 CN Regular','思源黑体 CN','Microsoft YaHei','PingFang SC',sans-serif";
    node.style.fontSize = "16px";
    node.style.lineHeight = "1";
    node.style.color = "#fff";

    var card = byId(node.id + "_div");
    if (card) {
      card.style.left = "0px";
      card.style.top = "0px";
      card.style.width = "100%";
      card.style.height = "100%";
      card.style.boxSizing = "border-box";
      card.style.borderRadius = "10px";
      card.style.backgroundColor = tone || card.style.backgroundColor || "#3961f6";
    }

    var text = byId(node.id + "_text");
    if (text) {
      text.style.left = "0px";
      text.style.top = "0px";
      text.style.width = "100%";
      text.style.height = "100%";
      text.style.display = "flex";
      text.style.alignItems = "center";
      text.style.justifyContent = "center";
      text.style.textAlign = "center";
      text.style.lineHeight = "1";
      text.style.padding = "0";
      text.style.margin = "0";
      text.style.transform = "none";
      text.style.transformOrigin = "center center";
      text.style.pointerEvents = "none";
      Array.prototype.forEach.call(text.querySelectorAll("p, span"), function (child) {
        child.style.margin = "0";
        child.style.padding = "0";
        child.style.lineHeight = "1";
      });
    }
  }

  function setRequiredLabelText(id, label) {
    var node = byId(id);
    if (!node) return;
    var textWrap = node.querySelector('[id$="_text"]');
    if (!textWrap) {
      node.textContent = "* " + label;
      return;
    }
    textWrap.innerHTML = "<p><span style=\"color:#FF0000;\">*</span><span> " + label + "</span></p>";
  }

  function updateDeviceSectionTitles() {
    setChoiceText("u582", "设备接入模块");
    setChoiceText("u663", "设备接入配置");
    ["u584", "u609", "u665", "u690"].forEach(function (id) {
      setRequiredLabelText(id, "设备类型");
    });
    ["u595", "u610", "u676", "u700"].forEach(function (id) {
      setRequiredLabelText(id, "选择设备");
    });
    ["u605", "u611"].forEach(function (id) {
      setRequiredLabelText(id, "接入位置");
    });
    ["u724", "u727", "u687", "u710"].forEach(function (id) {
      setRequiredLabelText(id, "接入数量");
    });
  }

  function syncDeviceSectionVisibility() {
    hideObsoleteSceneFields();
    renderDeviceAccessModule();
    relayoutSections();
  }

  function addTrainingDeviceSlot() {
    normalizeTrainingDeviceState();
    var currentCount = state.accessDevices.length;
    if (currentCount >= DEVICE_ACCESS_MAX) {
      addConsole("warn", "当前原型支持 " + DEVICE_ACCESS_MAX + " 行设备接入配置，可通过每行数量字段接入多台同类设备。");
      return;
    }
    var template = defaultDeviceForSlot(currentCount);
    if (!template) {
      addConsole("warn", "当前场景没有可接入设备，请先导入或切换场景。");
      return;
    }
    writeAccessSlot(currentCount, {
      type: deviceTypeForProfile(template),
      device: deviceProfileValue(template),
      count: 1,
      x: 0,
      y: 0,
      location: "0,0",
    });
    state.accessSlotCount = state.accessDevices.length;
    syncTrainingDeviceFields();
    addConsole("success", "已按当前场景添加接入设备：" + deviceProfileLabel(template) + "，默认数量 1。");
  }

  function removeTrainingDeviceSlot(index) {
    normalizeTrainingDeviceState();
    var currentCount = state.accessDevices.length;
    if (!currentCount) return;
    var slots = state.accessDevices.slice();
    slots.splice(Math.min(index, slots.length - 1), 1);
    state.accessDevices = slots;
    normalizeTrainingDeviceState();
    syncTrainingDeviceFields();
    addConsole("info", "已移除一行设备接入配置。");
  }

  function buildScenarioModelingDetail(scenario) {
    if (!scenario) return null;
    var template =
      SCENARIO_MODELING_TEMPLATES[scenario.name] ||
      SCENARIO_MODELING_TEMPLATES[scenario.disaster_type + "_residual"] ||
      null;
    var regionName = scenario.region_grid && scenario.region_grid.name ? scenario.region_grid.name : "--";
    var devices = Array.isArray(scenario.base_stations)
      ? scenario.base_stations.map(function (item) {
          return item.label || item.name;
        }).filter(Boolean).join(" / ")
      : "--";
    var rows = scenario.region_grid && scenario.region_grid.rows ? scenario.region_grid.rows : (scenario.grid_size || "--");
    var cols = scenario.region_grid && scenario.region_grid.cols ? scenario.region_grid.cols : (scenario.grid_size || "--");

    return {
      displayName: scenarioDisplayName(scenario),
      disasterType: disasterTypeLabel(scenario.disaster_type),
      regionName: regionName,
      userScale: Number(scenario.num_users || 0).toLocaleString("zh-CN"),
      candidateSites: Number(scenario.candidate_sites || 0).toLocaleString("zh-CN"),
      maxSteps: Number(scenario.max_steps || 0).toLocaleString("zh-CN"),
      defaultReward: rewardModeLabel(scenario.default_reward_profile),
      rewardProfiles: Array.isArray(scenario.reward_profiles)
        ? scenario.reward_profiles.map(function (item) {
            return rewardModeLabel(item.key);
          }).join(" / ")
        : "--",
      deviceSummary: devices || "--",
      gridSummary: "离散网格 " + rows + " × " + cols,
      scenarioBackground: template ? template.scenarioBackground : "当前场景用于灾后应急通信恢复训练，重点关注用户覆盖、链路可达性和设备投放成本之间的权衡。",
      disasterTraits: template ? template.disasterTraits : "受灾区域存在空间分布不均、关键区域优先保障和通信需求动态波动等特征。",
      networkDamage: template ? template.networkDamage : "网络基础设施在灾后出现不同程度退服、断链或容量下降，需要通过机动设备补网恢复。",
      modelingMethod: template ? template.modelingMethod : "将受灾区域离散为网格，把用户、站点、候选位点和机动设备统一映射到资源调度环境中。",
      stateDef: template ? template.stateDef : "状态包含覆盖率、用户分布、设备库存、候选位点和当前网络容量等信息。",
      actionDef: template ? template.actionDef : "动作定义为在候选站点上部署、切换或组合应急通信设备。",
      rewardDef: template ? template.rewardDef : "奖励综合覆盖恢复、吞吐提升、广播可达性和设备成本惩罚。",
      target: template ? template.target : "目标是在预算约束下尽快提升关键区域通信恢复效果。",
      difference: template ? template.difference : "该场景与其他场景的区别主要体现在受灾机理、残余网络可复用程度和资源恢复目标上。"
    };
  }

  function renderScenarioModelingContent(host, scenarioName) {
    if (!host) return;
    var scenario = state.scenarios.find(function (item) {
      return item.name === scenarioName;
    }) || currentScenario();
    var detail = buildScenarioModelingDetail(scenario);
    if (!detail) {
      host.innerHTML = "<div style='padding:40px 0;text-align:center;color:#94a3b8;'>暂无场景建模说明。</div>";
      return;
    }

    host.innerHTML =
      "<div style='display:flex;justify-content:space-between;align-items:flex-start;gap:16px;margin-bottom:18px;'>" +
      "<div>" +
      "<div style='font-size:24px;line-height:1.4;color:#0f172a;font-weight:700;'>" + escapeHtml(detail.displayName) + "</div>" +
      "<div style='margin-top:6px;font-size:13px;line-height:1.7;color:#64748b;'>" +
      "灾害类型：" + escapeHtml(detail.disasterType) + " · 区域：" + escapeHtml(detail.regionName) + " · 默认奖励：" + escapeHtml(detail.defaultReward) +
      "</div>" +
      "</div>" +
      "<button type='button' data-scene-name='" + escapeHtml(scenario.name) + "' id='scene-modeling-use-current' style='padding:10px 16px;border:1px solid #c7d2fe;border-radius:10px;background:#eef2ff;color:#3730a3;font-size:13px;cursor:pointer;white-space:nowrap;'>切换为当前训练场景</button>" +
      "</div>" +
      "<div style='display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-bottom:18px;'>" +
      "<div style='padding:14px 16px;border:1px solid #e2e8f0;border-radius:14px;background:#f8fbff;'><div style='font-size:12px;color:#64748b;'>用户规模</div><div style='margin-top:8px;font-size:22px;font-weight:700;color:#0f172a;'>" + escapeHtml(detail.userScale) + "</div></div>" +
      "<div style='padding:14px 16px;border:1px solid #e2e8f0;border-radius:14px;background:#f8fbff;'><div style='font-size:12px;color:#64748b;'>候选站点</div><div style='margin-top:8px;font-size:22px;font-weight:700;color:#0f172a;'>" + escapeHtml(detail.candidateSites) + "</div></div>" +
      "<div style='padding:14px 16px;border:1px solid #e2e8f0;border-radius:14px;background:#f8fbff;'><div style='font-size:12px;color:#64748b;'>训练最大步长</div><div style='margin-top:8px;font-size:22px;font-weight:700;color:#0f172a;'>" + escapeHtml(detail.maxSteps) + "</div></div>" +
      "<div style='padding:14px 16px;border:1px solid #e2e8f0;border-radius:14px;background:#f8fbff;'><div style='font-size:12px;color:#64748b;'>空间离散粒度</div><div style='margin-top:8px;font-size:16px;font-weight:700;color:#0f172a;'>" + escapeHtml(detail.gridSummary) + "</div></div>" +
      "</div>" +
      "<div style='display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:18px;'>" +
      "<div style='padding:16px;border:1px solid #e2e8f0;border-radius:14px;'>" +
      "<div style='font-size:16px;font-weight:700;color:#0f172a;margin-bottom:10px;'>受灾特点</div>" +
      "<div style='font-size:13px;line-height:1.9;color:#334155;'>" + escapeHtml(detail.disasterTraits) + "</div>" +
      "</div>" +
      "<div style='padding:16px;border:1px solid #e2e8f0;border-radius:14px;'>" +
      "<div style='font-size:16px;font-weight:700;color:#0f172a;margin-bottom:10px;'>网络 / 资源受损特征</div>" +
      "<div style='font-size:13px;line-height:1.9;color:#334155;'>" + escapeHtml(detail.networkDamage) + "</div>" +
      "</div>" +
      "</div>" +
      "<div style='display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:18px;'>" +
      "<div style='padding:16px;border:1px solid #e2e8f0;border-radius:14px;'>" +
      "<div style='font-size:16px;font-weight:700;color:#0f172a;margin-bottom:10px;'>场景背景</div>" +
      "<div style='font-size:13px;line-height:1.9;color:#334155;'>" + escapeHtml(detail.scenarioBackground) + "</div>" +
      "</div>" +
      "<div style='padding:16px;border:1px solid #e2e8f0;border-radius:14px;'>" +
      "<div style='font-size:16px;font-weight:700;color:#0f172a;margin-bottom:10px;'>建模方式</div>" +
      "<div style='font-size:13px;line-height:1.9;color:#334155;'>" + escapeHtml(detail.modelingMethod) + "</div>" +
      "<div style='margin-top:10px;font-size:12px;line-height:1.8;color:#64748b;'>设备库：" + escapeHtml(detail.deviceSummary) + "</div>" +
      "<div style='font-size:12px;line-height:1.8;color:#64748b;'>可选奖励模式：" + escapeHtml(detail.rewardProfiles) + "</div>" +
      "</div>" +
      "</div>" +
      "<div style='padding:16px;border:1px solid #e2e8f0;border-radius:14px;margin-bottom:18px;'>" +
      "<div style='font-size:16px;font-weight:700;color:#0f172a;margin-bottom:12px;'>强化学习设计</div>" +
      "<div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;'>" +
      "<div style='padding:14px;border-radius:12px;background:#f8fafc;'><div style='font-size:13px;font-weight:700;color:#0f172a;margin-bottom:8px;'>状态</div><div style='font-size:13px;line-height:1.8;color:#334155;'>" + escapeHtml(detail.stateDef) + "</div></div>" +
      "<div style='padding:14px;border-radius:12px;background:#f8fafc;'><div style='font-size:13px;font-weight:700;color:#0f172a;margin-bottom:8px;'>动作</div><div style='font-size:13px;line-height:1.8;color:#334155;'>" + escapeHtml(detail.actionDef) + "</div></div>" +
      "<div style='padding:14px;border-radius:12px;background:#f8fafc;'><div style='font-size:13px;font-weight:700;color:#0f172a;margin-bottom:8px;'>奖励</div><div style='font-size:13px;line-height:1.8;color:#334155;'>" + escapeHtml(detail.rewardDef) + "</div></div>" +
      "</div>" +
      "</div>" +
      "<div style='display:grid;grid-template-columns:1fr 1fr;gap:16px;'>" +
      "<div style='padding:16px;border:1px solid #e2e8f0;border-radius:14px;'>" +
      "<div style='font-size:16px;font-weight:700;color:#0f172a;margin-bottom:10px;'>预期目标</div>" +
      "<div style='font-size:13px;line-height:1.9;color:#334155;'>" + escapeHtml(detail.target) + "</div>" +
      "</div>" +
      "<div style='padding:16px;border:1px solid #e2e8f0;border-radius:14px;'>" +
      "<div style='font-size:16px;font-weight:700;color:#0f172a;margin-bottom:10px;'>与另外两个场景的核心差异</div>" +
      "<div style='font-size:13px;line-height:1.9;color:#334155;'>" + escapeHtml(detail.difference) + "</div>" +
      "</div>" +
      "</div>";

    var useButton = byId("scene-modeling-use-current");
    if (useButton) {
      useButton.addEventListener("click", function () {
        updateScenarioByType(scenario.disaster_type);
        var modal = byId("scene-modeling-modal");
        if (modal) modal.remove();
        addConsole("success", "已切换到场景：" + scenarioDisplayName(scenario));
      });
    }
  }

  function openSceneModelingModal(initialScenarioName) {
    var available = state.scenarios.slice();
    var existing = byId("scene-modeling-modal");
    if (existing) existing.remove();

    var scenarioName = initialScenarioName || state.scenarioName || (available[0] && available[0].name) || "";
    var modal = document.createElement("div");
    modal.id = "scene-modeling-modal";
    modal.style.cssText = "position:fixed;inset:0;background:rgba(2,6,23,0.58);z-index:100000;display:flex;align-items:center;justify-content:center;";
    modal.innerHTML =
      "<div style='width:1260px;max-height:820px;overflow:hidden;background:#fff;border-radius:16px;box-shadow:0 24px 60px rgba(15,23,42,0.28);display:flex;flex-direction:column;'>" +
      "<div style='display:flex;justify-content:space-between;align-items:flex-start;gap:16px;padding:24px 24px 18px;border-bottom:1px solid #e2e8f0;'>" +
      "<div>" +
      "<div style='font-size:22px;line-height:1.4;color:#0f172a;font-weight:700;'>三种场景建模说明</div>" +
      "<div style='margin-top:6px;font-size:13px;line-height:1.8;color:#64748b;'>点击下方场景查看其受灾特点、建模方式、强化学习设计和与其他场景的差异。</div>" +
      "</div>" +
      "<button type='button' style='border:0;background:none;font-size:16px;cursor:pointer;color:#64748b;' onclick='this.closest(\"#scene-modeling-modal\").remove()'>关闭</button>" +
      "</div>" +
      "<div style='padding:16px 24px 0;display:flex;gap:12px;flex-wrap:wrap;' id='scene-modeling-tabs'></div>" +
      "<div id='scene-modeling-content' style='padding:20px 24px 24px;overflow:auto;'></div>" +
      "</div>";
    document.body.appendChild(modal);
    modal.addEventListener("click", function (event) {
      if (event.target === modal) modal.remove();
    });

    var tabs = byId("scene-modeling-tabs");
    available.forEach(function (scenario) {
      var button = document.createElement("button");
      button.type = "button";
      button.setAttribute("data-scene-name", scenario.name);
      button.style.cssText = [
        "padding:10px 16px",
        "border:1px solid #dbe4ff",
        "border-radius:999px",
        "background:" + (scenario.name === scenarioName ? "#1d4ed8" : "#f8fbff"),
        "color:" + (scenario.name === scenarioName ? "#ffffff" : "#1e3a8a"),
        "font-size:13px",
        "cursor:pointer"
      ].join(";");
      button.textContent = scenarioDisplayName(scenario);
      button.addEventListener("click", function () {
        openSceneModelingModal(scenario.name);
      });
      tabs.appendChild(button);
    });

    renderScenarioModelingContent(byId("scene-modeling-content"), scenarioName);
  }

  function readLocalScenes() {
    try {
      var raw = window.localStorage.getItem(LOCAL_SCENES_KEY);
      var parsed = raw ? JSON.parse(raw) : [];
      return Array.isArray(parsed) ? parsed : [];
    } catch (error) {
      return [];
    }
  }

  function writeLocalScenes(items) {
    try {
      window.localStorage.setItem(LOCAL_SCENES_KEY, JSON.stringify(items));
    } catch (error) {
      addConsole("warn", "本地场景保存失败：" + (error && error.message ? error.message : error));
    }
  }

  function setEditorValue(editorId, value, eventType) {
    var editor = editors[editorId];
    if (!editor) return;
    editor.value = value == null ? "" : String(value);
    if (eventType) {
      editor.dispatchEvent(new Event(eventType, { bubbles: false }));
    }
  }

  function snapshotAccessSlots() {
    normalizeTrainingDeviceState();
    return state.accessDevices.map(function (row, index) {
      var slot = readAccessSlot(index);
      return {
        type: slot.type,
        device: slot.device,
        x: slot.x,
        y: slot.y,
        location: slot.x + "," + slot.y,
        count: slot.count,
      };
    }).filter(function (slot) {
      return Boolean(slot.device);
    });
  }

  function applyAccessSnapshotList(rows) {
    var normalized = (Array.isArray(rows) ? rows : []).map(function (item) {
      if (!item || !item.device) return null;
      var template = findDeviceTemplateByName(item.device);
      if (!template) return null;
      var bounds = currentScenarioGridBounds();
      var parsedLocation = parseLocationToGrid(item.location);
      var x = clampGridIndex(item.x != null ? item.x : (parsedLocation ? parsedLocation.x : 0), bounds.maxX, 0);
      var y = clampGridIndex(item.y != null ? item.y : (parsedLocation ? parsedLocation.y : 0), bounds.maxY, 0);
      return {
        type: deviceTypeForProfile(template),
        device: deviceProfileValue(template),
        x: x,
        y: y,
        location: x + "," + y,
        count: Math.max(1, integerValue(item.count, 1)),
      };
    }).filter(Boolean).slice(0, DEVICE_ACCESS_MAX);
    state.accessDevices = normalized;
    normalizeTrainingDeviceState();
  }

  function currentSceneSnapshot() {
    return {
      name: state.scenarioTitle || state.scenarioName || "未命名场景",
      scenario_name: state.scenarioName,
      disaster_type: state.disasterType,
      disaster_severity: state.disasterSeverity,
      disaster_notes: state.disasterNotes,
      affected_grid_count: state.affectedGridCount,
      impacted_population: state.impactedPopulation,
      priority_area: state.priorityArea,
      coverage_range: state.coverageRange,
      cell_granularity: state.cellGranularity,
      admin_division: state.adminDivision,
      candidate_site_count: state.candidateSiteCount,
      dispatch_unit: state.dispatchUnit,
      team_count: state.teamCount,
      budget_limit: state.budgetLimit,
      priority_equipment: state.priorityEquipment,
      device_access: snapshotAccessSlots(),
      residual_devices: [],
      candidate_devices: snapshotAccessSlots(),
      saved_at: Date.now(),
      source: "local",
    };
  }

  function applySceneSnapshot(snapshot) {
    if (!snapshot) return;
    state.disasterType = snapshot.disaster_type || state.disasterType;
    var matchedScenario = currentScenarioForType(state.disasterType);
    state.scenarioName = matchedScenario ? matchedScenario.name : (snapshot.scenario_name || snapshot.name || state.scenarioName);
    state.scenarioTitle = snapshot.name || state.scenarioTitle || state.scenarioName;
    state.disasterSeverity = snapshot.disaster_severity || state.disasterSeverity;
    state.disasterNotes = snapshot.disaster_notes || "";
    state.affectedGridCount = Number(snapshot.affected_grid_count || state.affectedGridCount || 1);
    state.impactedPopulation = Number(snapshot.impacted_population || state.impactedPopulation || 1);
    state.priorityArea = snapshot.priority_area || "";
    state.coverageRange = snapshot.coverage_range || "";
    state.cellGranularity = snapshot.cell_granularity || "";
    state.adminDivision = snapshot.admin_division || state.adminDivision;
    state.candidateSiteCount = Number(snapshot.candidate_site_count || state.candidateSiteCount || 1);
    state.dispatchUnit = snapshot.dispatch_unit || "";
    state.teamCount = Number(snapshot.team_count || state.teamCount || 1);
    state.budgetLimit = Number(snapshot.budget_limit || state.budgetLimit || 1);
    state.priorityEquipment = snapshot.priority_equipment || "";
    var accessDevices = Array.isArray(snapshot.device_access)
      ? snapshot.device_access
      : (Array.isArray(snapshot.residual_devices) ? snapshot.residual_devices : []).concat(Array.isArray(snapshot.candidate_devices) ? snapshot.candidate_devices : []);
    applyAccessSnapshotList(accessDevices);
    normalizeTrainingDeviceState();

    setEditorValue("u561", state.disasterType, "change");
    setEditorValue("u571", state.disasterSeverity, "change");
    [
      ["u551", state.disasterNotes],
      ["u575", state.affectedGridCount],
      ["u578", state.impactedPopulation],
      ["u637", state.priorityArea],
      ["u641", state.scenarioTitle],
      ["u654", state.coverageRange],
      ["u657", state.cellGranularity],
      ["u648", state.candidateSiteCount],
      ["u651", state.priorityEquipment],
      ["u720", state.dispatchUnit],
      ["u723", state.teamCount],
      ["u717", state.budgetLimit],
      ["u607", state.residualLocation1],
      ["u726", state.residualCount1],
      ["u613", state.residualLocation2],
      ["u729", state.residualCount2],
      ["u689", state.candidateCount1],
      ["u712", state.candidateCount2],
    ].forEach(function (entry) {
      setEditorValue(entry[0], entry[1], "input");
    });
    setEditorValue("u644", state.adminDivision, "change");
    setEditorValue("u592", state.residualType1, "change");
    setEditorValue("u603", state.residualDevice1, "change");
    setEditorValue("u624", state.residualType2, "change");
    setEditorValue("u633", state.residualDevice2, "change");
    setEditorValue("u673", state.candidateType1, "change");
    setEditorValue("u684", state.candidateDevice1, "change");
    setEditorValue("u698", state.candidateType2, "change");
    setEditorValue("u708", state.candidateDevice2, "change");
    syncTrainingDeviceFields();
    updateScenarioFields();
  }

  function ensureScrollPanel() {
    ["u543_state0", "u543_state0_content", "u544_state0", "u544_state0_content", "u772", "u773", "u780", "u781", "u782", "u735", "u736"].forEach(function (id) {
      var node = byId(id);
      if (node) node.style.pointerEvents = "auto";
    });
    var scrollPanel = byId("u543_state0");
    if (scrollPanel) {
      scrollPanel.style.pointerEvents = "auto";
      scrollPanel.style.overflow = "auto";
      scrollPanel.style.scrollbarWidth = "thin";
    }
  }

  function mountInput(boxId, key, options) {
    var box = byId(boxId);
    if (!box || editors[boxId]) return;
    var type = options && options.type ? options.type : "text";
    var tagName = options && options.tagName ? String(options.tagName).toLowerCase() : "input";
    box.style.position = "relative";
    box.style.pointerEvents = "auto";
    markInteractive(box);
    isolatePointerEvents(box);
    protectPanelState(box, options && options.panelVisibleId, options && options.panelHiddenId);
    var textWrap = box.querySelector('[id$="_text"]');
    if (textWrap) textWrap.style.display = "none";

    var input = document.createElement(tagName === "textarea" ? "textarea" : "input");
    if (tagName !== "textarea") input.type = type;
    input.value = state[key] == null ? "" : String(state[key]);
    input.style.cssText = [
      "position:absolute",
      "left:0",
      "top:0",
      "width:100%",
      "height:100%",
      tagName === "textarea" ? "padding:10px 14px" : "padding:0 14px",
      "margin:0",
      "border:none",
      "border-radius:0",
      "background:transparent",
      "font-size:14px",
      "color:#333",
      "font-family:'Microsoft YaHei','PingFang SC',sans-serif",
      "outline:none",
      "z-index:15",
      "box-sizing:border-box",
      "line-height:1.4",
      tagName === "textarea" ? "resize:none" : "",
      tagName === "textarea" ? "line-height:1.6" : "",
      tagName === "textarea" ? "overflow:auto" : ""
    ].join(";");
    if (options && options.min != null) input.min = String(options.min);
    if (options && options.max != null) input.max = String(options.max);
    if (options && options.step != null) input.step = String(options.step);
    markInteractive(input);
    isolatePointerEvents(input);
    protectPanelState(input, options && options.panelVisibleId, options && options.panelHiddenId);
    input.addEventListener("input", function () {
      var value = input.value;
      if (type === "number") {
        state[key] = value === "" ? null : Number(value);
      } else {
        state[key] = value;
      }
    });
    box.appendChild(input);
    editors[boxId] = input;
  }

  function mountSelect(panelId, choiceId, key, options, onChange) {
    var panel = byId(panelId);
    var choice = byId(choiceId);
    if (!panel || !choice || editors[choiceId]) return;
    unbindAxureSubtree(panel);
    panel.style.pointerEvents = "auto";
    choice.style.position = "relative";
    choice.style.pointerEvents = "auto";
    markInteractive(panel);
    markInteractive(choice);
    isolatePointerEvents(panel);
    isolatePointerEvents(choice);
    var panelVisibleId = null;
    var panelHiddenId = null;
    if (options && !Array.isArray(options)) {
      panelVisibleId = options.panelVisibleId;
      panelHiddenId = options.panelHiddenId;
      options = options.items || [];
    }
    var fakeMenu = panel.querySelector('[data-label="下拉菜单"]');
    if (fakeMenu) {
      fakeMenu.style.display = "none";
      fakeMenu.style.visibility = "hidden";
      fakeMenu.style.pointerEvents = "none";
    }
    replaceSelectArrowGlyphs(panel, choice, fakeMenu);
    protectPanelState(panel, panelVisibleId, panelHiddenId);
    protectPanelState(choice, panelVisibleId, panelHiddenId);

    var select = document.createElement("select");
    select.style.cssText = [
      "position:absolute",
      "left:0",
      "top:0",
      "width:100%",
      "height:100%",
      "margin:0",
      "padding:0",
      "border:none",
      "background:transparent",
      "opacity:0",
      "cursor:pointer",
      "appearance:none",
      "-webkit-appearance:none",
      "-moz-appearance:none",
      "z-index:16"
    ].join(";");
    markInteractive(select);
    isolatePointerEvents(select);
    protectPanelState(select, panelVisibleId, panelHiddenId);

    options.forEach(function (option) {
      var item = document.createElement("option");
      item.value = String(option.value);
      item.textContent = option.label;
      select.appendChild(item);
    });

    function syncText(value) {
      var match = options.find(function (option) {
        return String(option.value) === String(value);
      });
      if (match) setChoiceText(choiceId, match.label);
    }

    select.value = String(state[key]);
    syncText(select.value);
    select.addEventListener("change", function () {
      var value = select.value;
      if (value === "true") state[key] = true;
      else if (value === "false") state[key] = false;
      else state[key] = value;
      syncText(value);
      if (typeof onChange === "function") onChange(state[key]);
      replaceSelectArrowGlyphs(panel, choice, fakeMenu);
    });
    choice.appendChild(select);
    window.setTimeout(function () {
      replaceSelectArrowGlyphs(panel, choice, fakeMenu);
    }, 0);
    editors[choiceId] = select;
  }

  function algorithmDisplayName(key) {
    var match = ALGORITHM_CARD_CONFIG.find(function (item) {
      return item.key === key;
    });
    return match ? match.title : String(key || "--").toUpperCase();
  }

  function backendAlgorithmKey(key) {
    return key === "hmarl" ? "mppo" : key;
  }

  function styleAlgorithmText(id, left, top, width, height, fontSize, fontWeight, color, lineHeight) {
    var node = byId(id);
    if (!node) return;
    node.style.left = left + "px";
    node.style.top = top + "px";
    node.style.width = width + "px";
    node.style.height = height + "px";
    node.style.fontSize = fontSize + "px";
    node.style.fontWeight = String(fontWeight);
    node.style.color = color;
    node.style.lineHeight = lineHeight + "px";
    node.style.pointerEvents = "none";

    var div = byId(id + "_div");
    if (div) {
      div.style.width = width + "px";
      div.style.height = height + "px";
      div.style.fontSize = fontSize + "px";
      div.style.fontWeight = String(fontWeight);
      div.style.color = color;
      div.style.lineHeight = lineHeight + "px";
      div.style.background = "transparent";
      div.style.border = "none";
      div.style.boxShadow = "none";
    }

    var text = byId(id + "_text");
    if (text) {
      text.style.left = "0px";
      text.style.top = "0px";
      text.style.width = width + "px";
      text.style.height = height + "px";
      text.style.fontSize = fontSize + "px";
      text.style.fontWeight = String(fontWeight);
      text.style.color = color;
      text.style.lineHeight = lineHeight + "px";
      text.style.whiteSpace = "normal";
      text.style.overflow = "hidden";
      text.style.wordBreak = "break-word";
      text.style.wordWrap = "break-word";
    }
  }

  function configureAlgorithmCards() {
    ALGORITHM_CARD_CONFIG.forEach(function (card, index) {
      var left = ALGORITHM_CARD_LEFT + index * (ALGORITHM_CARD_WIDTH + ALGORITHM_CARD_GAP);
      var cardNode = byId(card.cardId);
      var img = byId(card.cardId + "_img");
      if (cardNode) {
        cardNode.style.left = left + "px";
        cardNode.style.top = ALGORITHM_CARD_TOP + "px";
        cardNode.style.width = ALGORITHM_CARD_WIDTH + "px";
        cardNode.style.height = ALGORITHM_CARD_HEIGHT + "px";
      }
      if (img) {
        img.style.width = ALGORITHM_CARD_WIDTH + "px";
        img.style.height = ALGORITHM_CARD_HEIGHT + "px";
      }

      styleAlgorithmText(
        card.titleId,
        left + 14,
        ALGORITHM_CARD_TOP + 12,
        ALGORITHM_CARD_WIDTH - 28,
        44,
        15,
        700,
        "#0f172a",
        20
      );
      styleAlgorithmText(
        card.descId,
        left + 14,
        ALGORITHM_CARD_TOP + 74,
        ALGORITHM_CARD_WIDTH - 28,
        22,
        13,
        400,
        "#64748b",
        18
      );
      setChoiceText(card.titleId, card.title);
      setChoiceText(card.descId, card.desc);
    });
  }

  function mountAlgorithmCards() {
    configureAlgorithmCards();
    var cards = {};
    ALGORITHM_CARD_CONFIG.forEach(function (card) {
      cards[card.key] = [card.cardId, card.titleId, card.descId];
    });
    function bindAlgorithmSelection(algorithm, event) {
      if (event) {
        event.preventDefault();
        event.stopPropagation();
      }
      state.algorithm = algorithm;
      syncAlgorithmCards();
      addConsole("info", "已切换训练算法：" + algorithmDisplayName(state.algorithm));
    }

    Object.keys(cards).forEach(function (algorithm) {
      cards[algorithm].forEach(function (id) {
        var node = byId(id);
        if (!node || node.dataset.liveBound) return;
        node.dataset.liveBound = "true";
        node.style.cursor = "pointer";
        node.style.pointerEvents = "auto";
        markInteractive(node);
        isolatePointerEvents(node);
        protectPanelState(node, "u1118_state0", "u1118_state1");

        Array.prototype.forEach.call(node.children || [], function (child) {
          child.style.pointerEvents = "none";
          Array.prototype.forEach.call(child.querySelectorAll("*"), function (descendant) {
            descendant.style.pointerEvents = "none";
          });
        });

        ["pointerdown", "click"].forEach(function (eventName) {
          node.addEventListener(eventName, function (event) {
            bindAlgorithmSelection(algorithm, event);
          }, true);
        });
      });
    });
    syncAlgorithmCards();
  }

  function syncAlgorithmCards() {
    ALGORITHM_CARD_CONFIG.forEach(function (card) {
      var id = card.cardId;
      var node = byId(id);
      if (!node) return;
      var selected = card.key === state.algorithm;
      node.classList.toggle("selected", selected);
      node.style.outline = "";
      node.style.outlineOffset = "";
      var img = byId(id + "_img");
      if (img) {
        img.classList.toggle("selected", selected);
        img.src = selected ? ALGORITHM_CARD_IMAGE.selected : ALGORITHM_CARD_IMAGE.normal;
        img.style.width = ALGORITHM_CARD_WIDTH + "px";
        img.style.height = ALGORITHM_CARD_HEIGHT + "px";
      }
    });
  }

  function mountParameterTabs() {
    [
      { id: "u780", key: "goal" },
      { id: "u781", key: "algo" },
      { id: "u782", key: "sim" },
    ].forEach(function (tab) {
      var node = byId(tab.id);
      if (!node || node.dataset.liveBound) return;
      node.dataset.liveBound = "true";
      node.style.cursor = "pointer";
      node.style.pointerEvents = "auto";
      markInteractive(node);
      isolatePointerEvents(node);
      protectPanelState(node, "u775_state0", "u775_state1");
      node.addEventListener("click", function (event) {
        event.preventDefault();
        event.stopPropagation();
        setParameterTab(tab.key);
      }, true);
    });
    setParameterTab("algo");
  }

  function setParameterTab(key) {
    var stateMap = {
      goal: "u783_state1",
      algo: "u783_state0",
      sim: "u783_state2",
    };
    Object.keys(stateMap).forEach(function (tabKey) {
      var visible = tabKey === key;
      var panel = byId(stateMap[tabKey]);
      if (panel) panel.style.visibility = visible ? "visible" : "hidden";
      if (panel) panel.style.display = visible ? "block" : "none";
    });
    [
      { id: "u780", key: "goal" },
      { id: "u781", key: "algo" },
      { id: "u782", key: "sim" },
    ].forEach(function (tab) {
      var node = byId(tab.id);
      if (!node) return;
      var selected = tab.key === key;
      node.classList.toggle("selected", selected);
      var img = byId(tab.id + "_img");
      if (img) {
        img.classList.toggle("selected", selected);
        var imageSet = PARAMETER_TAB_IMAGES[tab.id];
        if (imageSet) {
          img.src = selected ? imageSet.selected : imageSet.normal;
        }
      }
      var div = byId(tab.id + "_div");
      if (div) div.classList.toggle("selected", selected);
    });
  }

  function updateScenarioFields() {
    var scenario = currentScenario();
    if (!scenario) return;
    var grid = scenarioGridShape(scenario);
    var displayName = scenarioDisplayName(scenario);
    var regionName = scenario.region_grid && scenario.region_grid.name ? scenario.region_grid.name : displayName;
    SCENE_DEVICE_OPTIONS = buildDeviceOptions();
    syncScenarioBasicInfoLabels();

    state.scenarioTitle = displayName + "（" + scenario.name + "）";
    state.adminDivision = regionName;
    state.affectedGridCount = grid.count;
    state.impactedPopulation = Number(scenario.num_users || 0);
    state.coverageRange = scenarioGeoBoundsText(scenario);
    state.cellGranularity = grid.rows + " 行 × " + grid.cols + " 列，共 " + grid.count.toLocaleString("zh-CN") + " 个网格";
    state.priorityArea = scenarioCellLabelsText(scenario);
    state.disasterNotes = scenarioBasicNotes(scenario);
    state.candidateSiteCount = Number(scenario.candidate_sites || state.candidateSiteCount);

    if (editors.u561) editors.u561.value = state.disasterType;
    if (editors.u571) editors.u571.disabled = true;
    setChoiceText("u561", disasterTypeLabel(state.disasterType));
    setChoiceText("u571", rewardModeLabel(state.rewardMode));
    setChoiceText("u644", state.adminDivision || "请选择行政区划");
    syncRewardConfigPanel();

    [
      ["u551", state.disasterNotes, "input"],
      ["u575", state.affectedGridCount, "input"],
      ["u578", state.impactedPopulation, "input"],
      ["u637", state.priorityArea, "input"],
      ["u641", state.scenarioTitle, "input"],
      ["u648", state.candidateSiteCount, "input"],
      ["u654", state.coverageRange, "input"],
      ["u657", state.cellGranularity, "input"],
    ].forEach(function (entry) {
      setEditorValue(entry[0], entry[1], entry[2]);
    });
  }

  function updateScenarioByType(type) {
    var scenario = currentScenarioForType(type);
    if (!scenario) return;
    var changedScenario = state.scenarioName !== scenario.name;
    state.scenarioName = scenario.name;
    state.disasterType = scenario.disaster_type || type;
    if (changedScenario) {
      state.accessDevices = [];
      state.accessSlotCount = 0;
    }
    updateScenarioFields();
    syncTrainingDeviceFields();
  }

  function mountEditableFields() {
    var scenePanel = { panelVisibleId: "u545_state0", panelHiddenId: "u545_state1" };
    mountSelect("u554", "u561", "disasterType", {
      panelVisibleId: scenePanel.panelVisibleId,
      panelHiddenId: scenePanel.panelHiddenId,
      items: [
        { value: "flood", label: "洪水" },
        { value: "earthquake", label: "地震" },
        { value: "typhoon", label: "台风" },
      ],
    }, updateScenarioByType);
    mountSelect("u564", "u571", "disasterSeverity", {
      panelVisibleId: scenePanel.panelVisibleId,
      panelHiddenId: scenePanel.panelHiddenId,
      items: [
        { value: "general", label: "一般" },
        { value: "moderate", label: "中等" },
        { value: "severe", label: "严重" },
        { value: "critical", label: "特别严重" },
      ],
    });
    mountInput("u551", "disasterNotes", { tagName: "textarea", panelVisibleId: scenePanel.panelVisibleId, panelHiddenId: scenePanel.panelHiddenId });
    mountInput("u575", "affectedGridCount", { type: "number", min: 1, step: 1, panelVisibleId: scenePanel.panelVisibleId, panelHiddenId: scenePanel.panelHiddenId });
    mountInput("u578", "impactedPopulation", { type: "number", min: 1, step: 10, panelVisibleId: scenePanel.panelVisibleId, panelHiddenId: scenePanel.panelHiddenId });
    mountInput("u637", "priorityArea", { type: "text", panelVisibleId: scenePanel.panelVisibleId, panelHiddenId: scenePanel.panelHiddenId });
    mountInput("u641", "scenarioTitle", { type: "text", panelVisibleId: scenePanel.panelVisibleId, panelHiddenId: scenePanel.panelHiddenId });
    mountSelect("u643", "u644", "adminDivision", {
      panelVisibleId: scenePanel.panelVisibleId,
      panelHiddenId: scenePanel.panelHiddenId,
      items: [
        { value: "川西震后山区网格", label: "川西震后山区网格" },
        { value: "珠海-沿海台风救灾区块", label: "珠海-沿海台风救灾区块" },
        { value: "洞庭湖-城郊易涝区块", label: "洞庭湖-城郊易涝区块" },
      ],
    });
    mountInput("u648", "candidateSiteCount", { type: "number", min: 1, step: 1, panelVisibleId: scenePanel.panelVisibleId, panelHiddenId: scenePanel.panelHiddenId });
    mountInput("u651", "priorityEquipment", { type: "text", panelVisibleId: scenePanel.panelVisibleId, panelHiddenId: scenePanel.panelHiddenId });
    mountInput("u654", "coverageRange", { type: "text", panelVisibleId: scenePanel.panelVisibleId, panelHiddenId: scenePanel.panelHiddenId });
    mountInput("u657", "cellGranularity", { type: "text", panelVisibleId: scenePanel.panelVisibleId, panelHiddenId: scenePanel.panelHiddenId });
    mountInput("u717", "budgetLimit", { type: "number", min: 1, step: 1, panelVisibleId: scenePanel.panelVisibleId, panelHiddenId: scenePanel.panelHiddenId });
    mountInput("u720", "dispatchUnit", { type: "text", panelVisibleId: scenePanel.panelVisibleId, panelHiddenId: scenePanel.panelHiddenId });
    mountInput("u723", "teamCount", { type: "number", min: 1, step: 1, panelVisibleId: scenePanel.panelVisibleId, panelHiddenId: scenePanel.panelHiddenId });
    syncScenarioBasicInfoLabels();
    mountSelect("u585", "u592", "residualType1", {
      panelVisibleId: scenePanel.panelVisibleId,
      panelHiddenId: scenePanel.panelHiddenId,
      items: SCENE_DEVICE_TYPE_OPTIONS,
    });
    mountSelect("u596", "u603", "residualDevice1", {
      panelVisibleId: scenePanel.panelVisibleId,
      panelHiddenId: scenePanel.panelHiddenId,
      items: SCENE_DEVICE_OPTIONS,
    }, function () {
      syncTrainingDeviceSlot(0, "u592");
    });
    mountInput("u607", "residualLocation1", { type: "text", panelVisibleId: scenePanel.panelVisibleId, panelHiddenId: scenePanel.panelHiddenId });
    mountInput("u726", "residualCount1", { type: "number", min: 0, step: 1, panelVisibleId: scenePanel.panelVisibleId, panelHiddenId: scenePanel.panelHiddenId });
    mountSelect("u617", "u624", "residualType2", {
      panelVisibleId: scenePanel.panelVisibleId,
      panelHiddenId: scenePanel.panelHiddenId,
      items: SCENE_DEVICE_TYPE_OPTIONS,
    });
    mountSelect("u626", "u633", "residualDevice2", {
      panelVisibleId: scenePanel.panelVisibleId,
      panelHiddenId: scenePanel.panelHiddenId,
      items: SCENE_DEVICE_OPTIONS,
    }, function () {
      syncTrainingDeviceSlot(1, "u624");
    });
    mountInput("u613", "residualLocation2", { type: "text", panelVisibleId: scenePanel.panelVisibleId, panelHiddenId: scenePanel.panelHiddenId });
    mountInput("u729", "residualCount2", { type: "number", min: 0, step: 1, panelVisibleId: scenePanel.panelVisibleId, panelHiddenId: scenePanel.panelHiddenId });
    mountSelect("u666", "u673", "candidateType1", {
      panelVisibleId: scenePanel.panelVisibleId,
      panelHiddenId: scenePanel.panelHiddenId,
      items: SCENE_DEVICE_TYPE_OPTIONS,
    });
    mountSelect("u677", "u684", "candidateDevice1", {
      panelVisibleId: scenePanel.panelVisibleId,
      panelHiddenId: scenePanel.panelHiddenId,
      items: SCENE_DEVICE_OPTIONS,
    }, function () {
      syncTrainingDeviceSlot(2, "u673");
    });
    mountInput("u689", "candidateCount1", { type: "number", min: 0, step: 1, panelVisibleId: scenePanel.panelVisibleId, panelHiddenId: scenePanel.panelHiddenId });
    mountSelect("u691", "u698", "candidateType2", {
      panelVisibleId: scenePanel.panelVisibleId,
      panelHiddenId: scenePanel.panelHiddenId,
      items: SCENE_DEVICE_TYPE_OPTIONS,
    });
    mountSelect("u701", "u708", "candidateDevice2", {
      panelVisibleId: scenePanel.panelVisibleId,
      panelHiddenId: scenePanel.panelHiddenId,
      items: SCENE_DEVICE_OPTIONS,
    }, function () {
      syncTrainingDeviceSlot(3, "u698");
    });
    mountInput("u712", "candidateCount2", { type: "number", min: 0, step: 1, panelVisibleId: scenePanel.panelVisibleId, panelHiddenId: scenePanel.panelHiddenId });

    ["u551", "u575", "u578", "u637", "u641", "u648", "u651", "u654", "u657", "u717", "u720", "u723", "u607", "u726", "u613", "u729", "u689", "u712"].forEach(function (id) {
      var editor = editors[id];
      if (editor) {
        editor.addEventListener("change", function () {
          editor.dataset.userEdited = "true";
        });
      }
    });
    ["u689", "u712"].forEach(function (id) {
      var editor = editors[id];
      if (!editor) return;
      editor.addEventListener("input", syncPriorityEquipmentText);
      editor.addEventListener("change", syncPriorityEquipmentText);
    });
    syncTrainingDeviceFields();
  }

  function bindSceneActionButtons() {
    [
      { id: "u713", handler: saveCurrentScene, tone: "#3961f6" },
      { id: "u714", handler: openSceneImportModal, tone: "#1890ff" },
    ].forEach(function (item) {
      var node = byId(item.id);
      if (!node || node.dataset.liveBound) return;
      node.dataset.liveBound = "true";
      node.style.pointerEvents = "auto";
      node.style.cursor = "pointer";
      styleSceneActionButton(node, item.tone);
      markInteractive(node);
      isolatePointerEvents(node);
      protectPanelState(node, "u545_state0", "u545_state1");
      ["mouseenter", "mouseover", "mouseleave", "mouseout", "pointerenter", "pointerleave"].forEach(function (eventName) {
        node.addEventListener(eventName, function () {
          styleSceneActionButton(node, item.tone);
        }, true);
      });
      node.addEventListener("click", function (event) {
        event.preventDefault();
        event.stopPropagation();
        item.handler();
      }, true);
    });
  }

  function bindIconButton(id, handler, title) {
    var node = byId(id);
    if (!node || node.dataset.liveBound) return;
    node.dataset.liveBound = "true";
    node.title = title || "";
    node.style.pointerEvents = "auto";
    node.style.cursor = "pointer";
    markInteractive(node);
    isolatePointerEvents(node);
    Array.prototype.forEach.call(node.children || [], function (child) {
      child.style.pointerEvents = "none";
    });
    ["pointerdown", "click"].forEach(function (eventName) {
      node.addEventListener(eventName, function (event) {
        event.preventDefault();
        event.stopPropagation();
        handler();
      }, true);
    });
  }

  function bindDeviceSlotActions() {
    bindIconButton("u594", addTrainingDeviceSlot, "添加接入设备");
    bindIconButton("u675", addTrainingDeviceSlot, "继续添加接入设备");
    bindIconButton("u615", function () { removeTrainingDeviceSlot(0); }, "移除接入设备");
    bindIconButton("u616", function () { removeTrainingDeviceSlot(1); }, "移除接入设备");
    bindIconButton("u686", function () { removeTrainingDeviceSlot(Math.max(0, state.accessSlotCount - 1)); }, "移除接入设备");
  }

  function saveCurrentScene() {
    var snapshot = currentSceneSnapshot();
    var localScenes = readLocalScenes().filter(function (item) {
      return item && item.name !== snapshot.name;
    });
    localScenes.unshift(snapshot);
    writeLocalScenes(localScenes.slice(0, 20));
    addConsole("success", "场景已保存到本地，可通过“导入场景”再次载入：" + snapshot.name);
  }

  function openSceneImportModal() {
    var localScenes = readLocalScenes();
    var builtinScenes = state.scenarios.map(function (item) {
      return {
        name: item.name,
        scenario_name: item.name,
        disaster_type: item.disaster_type,
        admin_division: item.region_grid && item.region_grid.name ? item.region_grid.name : scenarioDisplayName(item),
        affected_grid_count: item.candidate_sites || state.affectedGridCount,
        candidate_site_count: item.candidate_sites || state.candidateSiteCount,
        coverage_range: item.grid_size ? String(item.grid_size) : "",
        cell_granularity: item.region_grid && item.region_grid.rows && item.region_grid.cols ? ("离散网格 " + item.region_grid.rows + " × " + item.region_grid.cols) : "",
        source: "builtin",
      };
    });
    function sceneImportKey(scene) {
      if (!scene) return "";
      if (scene.scenario_name) return scene.scenario_name;
      if (state.scenarios.some(function (item) { return item.name === scene.name; })) return scene.name;
      var sameType = state.scenarios.filter(function (item) {
        return item.disaster_type === scene.disaster_type;
      });
      return sameType.length === 1 ? sameType[0].name : (scene.name || scene.disaster_type || "");
    }
    var builtinKeys = {};
    builtinScenes.forEach(function (scene) {
      builtinKeys[sceneImportKey(scene)] = true;
    });
    var scenes = builtinScenes.concat(localScenes.filter(function (scene) {
      return !builtinKeys[sceneImportKey(scene)];
    }));
    var existing = byId("scene-import-modal");
    if (existing) existing.remove();

    var modal = document.createElement("div");
    modal.id = "scene-import-modal";
    modal.style.cssText = "position:fixed;inset:0;background:rgba(2,6,23,0.48);z-index:99999;display:flex;align-items:center;justify-content:center;";
    var rows = scenes.map(function (scene, index) {
      var typeLabel = disasterTypeLabel(scene.disaster_type);
      var modelingAction = scene.source === "builtin"
        ? "<button type='button' data-model-index='" + index + "' style='padding:6px 12px;border:1px solid #dbe4ff;border-radius:8px;background:#eef4ff;color:#1d4ed8;cursor:pointer;'>查看建模</button>"
        : "<span style='color:#94a3b8;'>仅标准场景支持</span>";
      return "<tr data-index='" + index + "' style='cursor:pointer;'>" +
        "<td style='padding:10px 12px;border-bottom:1px solid #eef2f7;'>" + (index + 1) + "</td>" +
        "<td style='padding:10px 12px;border-bottom:1px solid #eef2f7;'>" + escapeHtml(scene.source === "builtin" ? scenarioDisplayName(scene) : (scene.name || "--")) + "</td>" +
        "<td style='padding:10px 12px;border-bottom:1px solid #eef2f7;'>" + typeLabel + "</td>" +
        "<td style='padding:10px 12px;border-bottom:1px solid #eef2f7;'>" + (scene.source === "local" ? "本地保存" : "后端场景") + "</td>" +
        "<td style='padding:10px 12px;border-bottom:1px solid #eef2f7;'>" + modelingAction + "</td>" +
        "</tr>";
    }).join("");
    modal.innerHTML =
      "<div style='width:920px;max-height:640px;overflow:auto;background:#fff;border-radius:14px;padding:24px;box-shadow:0 24px 60px rgba(15,23,42,0.2);'>" +
      "<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;'>" +
      "<strong style='font-size:18px;color:#0f172a;'>导入场景</strong>" +
      "<button type='button' style='border:0;background:none;font-size:16px;cursor:pointer;color:#64748b;' onclick='this.closest(\"#scene-import-modal\").remove()'>关闭</button>" +
      "</div>" +
      "<table style='width:100%;border-collapse:collapse;font-size:14px;color:#334155;'>" +
      "<thead><tr style='background:#f8fafc;'><th style='padding:10px 12px;text-align:left;'>序号</th><th style='padding:10px 12px;text-align:left;'>场景</th><th style='padding:10px 12px;text-align:left;'>灾害类型</th><th style='padding:10px 12px;text-align:left;'>来源</th><th style='padding:10px 12px;text-align:left;'>建模说明</th></tr></thead>" +
      "<tbody>" + (rows || "<tr><td colspan='5' style='padding:28px 0;text-align:center;color:#94a3b8;'>暂无可导入场景</td></tr>") + "</tbody>" +
      "</table>" +
      "</div>";
    document.body.appendChild(modal);
    modal.addEventListener("click", function (event) {
      if (event.target === modal) modal.remove();
    });
    Array.prototype.forEach.call(modal.querySelectorAll("button[data-model-index]"), function (button) {
      button.addEventListener("click", function (event) {
        event.preventDefault();
        event.stopPropagation();
        var selectedScene = scenes[Number(button.getAttribute("data-model-index"))];
        if (!selectedScene) return;
        openSceneModelingModal(selectedScene.name);
      });
    });
    Array.prototype.forEach.call(modal.querySelectorAll("tbody tr[data-index]"), function (row) {
      row.addEventListener("click", function () {
        var selected = scenes[Number(row.getAttribute("data-index"))];
        if (selected) applySceneSnapshot(selected);
        modal.remove();
        addConsole("success", "已导入场景：" + (selected && selected.name ? selected.name : "--"));
      });
    });
  }

  function mountParameterFields() {
    var parameterPanel = { panelVisibleId: "u775_state0", panelHiddenId: "u775_state1" };
    mountInput("u786", "totalTimesteps", { type: "number", min: 1000, step: 1000, panelVisibleId: parameterPanel.panelVisibleId, panelHiddenId: parameterPanel.panelHiddenId });
    mountInput("u790", "learningRate", { type: "number", min: 0.00001, step: 0.00001, panelVisibleId: parameterPanel.panelVisibleId, panelHiddenId: parameterPanel.panelHiddenId });
    mountInput("u794", "discountFactor", { type: "number", min: 0.8, max: 0.999, step: 0.001, panelVisibleId: parameterPanel.panelVisibleId, panelHiddenId: parameterPanel.panelHiddenId });
    mountInput("u798", "batchSize", { type: "number", min: 1, step: 1, panelVisibleId: parameterPanel.panelVisibleId, panelHiddenId: parameterPanel.panelHiddenId });
    mountInput("u802", "rolloutSteps", { type: "number", min: 1, step: 64, panelVisibleId: parameterPanel.panelVisibleId, panelHiddenId: parameterPanel.panelHiddenId });
    mountInput("u806", "entropyCoef", { type: "number", min: 0, step: 0.001, panelVisibleId: parameterPanel.panelVisibleId, panelHiddenId: parameterPanel.panelHiddenId });
    mountInput("u810", "clipRange", { type: "number", min: 0.05, max: 0.5, step: 0.01, panelVisibleId: parameterPanel.panelVisibleId, panelHiddenId: parameterPanel.panelHiddenId });

    mountSelect("u837", "u842", "envType", {
      panelVisibleId: parameterPanel.panelVisibleId,
      panelHiddenId: parameterPanel.panelHiddenId,
      items: [
        { value: "multimodal", label: "多模融合环境" },
        { value: "baseline", label: "基线环境" },
      ],
    });
    mountSelect("u858", "u863", "stochasticEval", {
      panelVisibleId: parameterPanel.panelVisibleId,
      panelHiddenId: parameterPanel.panelHiddenId,
      items: [
        { value: "true", label: "随机策略评估" },
        { value: "false", label: "确定性策略评估" },
      ],
    });
    mountInput("u845", "simulationWindowHours", { type: "number", min: 1, step: 1, panelVisibleId: parameterPanel.panelVisibleId, panelHiddenId: parameterPanel.panelHiddenId });
    mountInput("u849", "coverageTarget", { type: "number", min: 1, max: 100, step: 1, panelVisibleId: parameterPanel.panelVisibleId, panelHiddenId: parameterPanel.panelHiddenId });
    mountInput("u853", "logWindow", { type: "number", min: 1, step: 1, panelVisibleId: parameterPanel.panelVisibleId, panelHiddenId: parameterPanel.panelHiddenId });
    mountSelect("u867", "u873", "trafficLoadProfile", {
      panelVisibleId: parameterPanel.panelVisibleId,
      panelHiddenId: parameterPanel.panelHiddenId,
      items: [
        { value: "low", label: "低负载" },
        { value: "medium", label: "中负载" },
        { value: "high", label: "高负载" },
      ],
    });
    mountSelect("u877", "u884", "priorityObjective", {
      panelVisibleId: parameterPanel.panelVisibleId,
      panelHiddenId: parameterPanel.panelHiddenId,
      items: [
        { value: "bandwidth_first", label: "带宽优先" },
        { value: "device_cost_first", label: "设备开销最小优先" },
        { value: "coverage_considered", label: "考虑覆盖" },
        { value: "coverage_first", label: "覆盖优先" },
      ],
    });
    mountInput("u887", "evalInterval", { type: "number", min: 1, step: 1, panelVisibleId: parameterPanel.panelVisibleId, panelHiddenId: parameterPanel.panelHiddenId });
    mountSelect("u892", "u897", "autoReplay", {
      panelVisibleId: parameterPanel.panelVisibleId,
      panelHiddenId: parameterPanel.panelHiddenId,
      items: [
        { value: "true", label: "自动生成回放" },
        { value: "false", label: "仅保留训练日志" },
      ],
    });
  }

  function ensureDashboard() {
    setPanelVisible("u735", true);
    ensureTrainingChart();
    var panel = byId("u736");
    if (!panel) return null;
    panel.style.position = "relative";
    panel.style.pointerEvents = "auto";

    var body = byId("training-console-body");
    if (body) return body;

    body = document.createElement("div");
    body.id = "training-console-body";
    body.style.cssText = [
      "position:absolute",
      "left:0",
      "top:0",
      "right:0",
      "bottom:0",
      "padding:12px 14px",
      "overflow:auto",
      "color:#dbeafe",
      "font-family:Consolas,'Courier New',monospace",
      "font-size:12px",
      "line-height:1.75",
      "z-index:2"
    ].join(";");
    panel.appendChild(body);
    return body;
  }

  function ensureTrainingChart() {
    var chartHost = byId("u735_state0_content") || byId("u739");
    if (!chartHost) return null;
    chartHost.style.pointerEvents = "auto";

    var staticBlock = byId("u739");
    var staticImg = byId("u739_img");
    if (staticBlock) {
      staticBlock.style.display = "none";
      staticBlock.style.visibility = "hidden";
    }
    if (staticImg) {
      staticImg.style.display = "none";
      staticImg.style.visibility = "hidden";
    }

    var existing = byId("training-live-chart");
    if (existing) return existing;

    var chart = document.createElement("div");
    chart.id = "training-live-chart";
    chart.style.cssText = [
      "position:absolute",
      "left:0px",
      "top:125px",
      "width:1579px",
      "height:219px",
      "max-height:219px",
      "padding:8px 10px 10px",
      "box-sizing:border-box",
      "background:linear-gradient(180deg,rgba(14,23,36,0.16),rgba(14,23,36,0.06))",
      "border-radius:3px",
      "pointer-events:none",
      "overflow:hidden",
      "z-index:1"
    ].join(";");
    chart.innerHTML =
      "<div style='display:flex;justify-content:space-between;align-items:center;height:24px;margin-bottom:6px;color:#dbeafe;font-family:Microsoft YaHei,PingFang SC,sans-serif;font-size:12px;'>" +
      "<div style='display:flex;gap:14px;align-items:center;'>" +
      "<span style='display:inline-flex;gap:6px;align-items:center;'><i style='display:inline-block;width:10px;height:2px;background:#38bdf8;'></i>覆盖率</span>" +
      "<span style='display:inline-flex;gap:6px;align-items:center;'><i style='display:inline-block;width:10px;height:2px;background:#fbbf24;'></i>广播覆盖</span>" +
      "</div>" +
      "<span id='training-chart-caption' style='color:rgba(219,234,254,0.72);'>等待训练数据...</span>" +
      "</div>" +
      "<svg id='training-live-chart-svg' viewBox='0 0 1000 250' preserveAspectRatio='none' style='display:block;width:100%;height:170px;overflow:hidden;'></svg>";
    chartHost.appendChild(chart);
    renderTrainingChart();
    return chart;
  }

  function pushChartPoint(point) {
    state.chartPoints.push(point);
    if (state.chartPoints.length > 80) {
      state.chartPoints = state.chartPoints.slice(-80);
    }
    renderTrainingChart();
  }

  function clearChart() {
    state.chartPoints = [];
    renderTrainingChart();
  }

  function renderTrainingChart() {
    var chart = ensureTrainingChart();
    if (!chart) return;
    var svg = byId("training-live-chart-svg");
    var caption = byId("training-chart-caption");
    if (!svg) return;

    var width = 1000;
    var height = 250;
    var padding = { left: 48, right: 18, top: 14, bottom: 28 };
    var innerWidth = width - padding.left - padding.right;
    var innerHeight = height - padding.top - padding.bottom;
    var points = state.chartPoints;

    if (!points.length) {
      svg.innerHTML =
        "<rect x='0' y='0' width='" + width + "' height='" + height + "' fill='rgba(15,23,42,0.18)' rx='6' />" +
        "<text x='500' y='128' text-anchor='middle' fill='rgba(219,234,254,0.35)' font-size='16'>训练开始后将在这里绘制实时曲线</text>";
      if (caption) caption.textContent = "等待训练数据...";
      return;
    }

    function xAt(index) {
      return padding.left + (points.length === 1 ? innerWidth / 2 : (innerWidth * index) / (points.length - 1));
    }

    function yAt(value) {
      var ratio = Math.max(0, Math.min(1, Number(value) || 0));
      return padding.top + (1 - ratio) * innerHeight;
    }

    function buildPath(key) {
      return points.map(function (point, index) {
        var x = xAt(index).toFixed(2);
        var y = yAt(point[key]).toFixed(2);
        return (index === 0 ? "M" : "L") + x + " " + y;
      }).join(" ");
    }

    function buildArea(key) {
      var line = points.map(function (point, index) {
        return (index === 0 ? "M" : "L") + xAt(index).toFixed(2) + " " + yAt(point[key]).toFixed(2);
      }).join(" ");
      return line + " L " + xAt(points.length - 1).toFixed(2) + " " + (padding.top + innerHeight).toFixed(2) +
        " L " + xAt(0).toFixed(2) + " " + (padding.top + innerHeight).toFixed(2) + " Z";
    }

    var horizontalGuides = [0, 0.25, 0.5, 0.75, 1].map(function (value) {
      var y = yAt(value).toFixed(2);
      var label = Math.round(value * 100) + "%";
      return (
        "<line x1='" + padding.left + "' y1='" + y + "' x2='" + (padding.left + innerWidth) + "' y2='" + y + "' stroke='rgba(148,163,184,0.18)' stroke-dasharray='4 4' />" +
        "<text x='8' y='" + (Number(y) + 4) + "' fill='rgba(219,234,254,0.52)' font-size='11'>" + label + "</text>"
      );
    }).join("");

    var verticalGuides = points.filter(function (_, index) {
      if (points.length <= 6) return true;
      return index === 0 || index === points.length - 1 || index % Math.ceil(points.length / 5) === 0;
    }).map(function (point, index) {
      var realIndex = points.indexOf(point);
      var x = xAt(realIndex).toFixed(2);
      var label = point.label || ("#" + realIndex);
      return (
        "<line x1='" + x + "' y1='" + padding.top + "' x2='" + x + "' y2='" + (padding.top + innerHeight) + "' stroke='rgba(148,163,184,0.1)' />" +
        "<text x='" + x + "' y='" + (height - 6) + "' text-anchor='middle' fill='rgba(219,234,254,0.42)' font-size='10'>" + label + "</text>"
      );
    }).join("");

    var coveragePath = buildPath("coverage");
    var broadcastPath = buildPath("broadcast");
    var coverageArea = buildArea("coverage");
    var lastPoint = points[points.length - 1];
    var lastX = xAt(points.length - 1).toFixed(2);
    var lastCoverageY = yAt(lastPoint.coverage).toFixed(2);
    var lastBroadcastY = yAt(lastPoint.broadcast).toFixed(2);

    svg.innerHTML =
      "<defs>" +
      "<linearGradient id='training-coverage-fill' x1='0' y1='0' x2='0' y2='1'>" +
      "<stop offset='0%' stop-color='rgba(56,189,248,0.28)' />" +
      "<stop offset='100%' stop-color='rgba(56,189,248,0.02)' />" +
      "</linearGradient>" +
      "</defs>" +
      "<rect x='0' y='0' width='" + width + "' height='" + height + "' fill='rgba(15,23,42,0.2)' rx='6' />" +
      horizontalGuides +
      verticalGuides +
      "<path d='" + coverageArea + "' fill='url(#training-coverage-fill)' />" +
      "<path d='" + coveragePath + "' fill='none' stroke='#38bdf8' stroke-width='3' stroke-linecap='round' stroke-linejoin='round' />" +
      "<path d='" + broadcastPath + "' fill='none' stroke='#fbbf24' stroke-width='3' stroke-linecap='round' stroke-linejoin='round' />" +
      "<circle cx='" + lastX + "' cy='" + lastCoverageY + "' r='4.5' fill='#38bdf8' />" +
      "<circle cx='" + lastX + "' cy='" + lastBroadcastY + "' r='4.5' fill='#fbbf24' />" +
      "<text x='" + (padding.left + innerWidth - 4) + "' y='20' text-anchor='end' fill='rgba(219,234,254,0.72)' font-size='11'>最新覆盖 " + Math.round((lastPoint.coverage || 0) * 1000) / 10 + "%</text>" +
      "<text x='" + (padding.left + innerWidth - 4) + "' y='36' text-anchor='end' fill='rgba(251,191,36,0.82)' font-size='11'>最新广播 " + Math.round((lastPoint.broadcast || 0) * 1000) / 10 + "%</text>";

    if (caption) {
      caption.textContent = "已采样 " + points.length + " 个点，最新标签 " + (lastPoint.label || "--");
    }
  }

  function addConsole(type, msg) {
    var body = ensureDashboard();
    if (!body) return;
    body.style.textAlign = "left";
    var time = new Date().toLocaleTimeString("zh-CN", { hour12: false });
    var colors = {
      info: "#38bdf8",
      success: "#22c55e",
      error: "#ef4444",
      warn: "#f59e0b",
      episode: "#22d3ee",
      update: "#fbbf24",
      evaluation: "#a78bfa",
      status: "#e2e8f0",
      request: "#94a3b8",
      completed: "#d8b4fe",
    };
    var line = document.createElement("div");
    line.style.whiteSpace = "pre-wrap";
    line.style.wordBreak = "break-all";
    line.style.textAlign = "left";
    line.innerHTML =
      '<span style="color:#64748b;">[' + time + ']</span> ' +
      '<span style="color:' + (colors[type] || "#cbd5e1") + ';">[' + type + ']</span> ' +
      '<span>' + String(msg) + "</span>";
    body.appendChild(line);
    body.scrollTop = body.scrollHeight;
    while (body.children.length > 160) body.removeChild(body.firstChild);
  }

  function escapeHtml(value) {
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function formatPercent(value) {
    if (value == null || value === "") return "--";
    return (Number(value) * 100).toFixed(1) + "%";
  }

  function formatMetric(value, digits) {
    if (value == null || value === "") return "--";
    return Number(value).toFixed(digits == null ? 3 : digits);
  }

  function updateStatus(text) {
    setChoiceText("u746", text);
  }

  function updateTime(text) {
    setChoiceText("u753", text);
  }

  function formatLivePercentText(value) {
    if (value == null || value === "" || value === "--") return "--";
    var number = Number(value);
    return Number.isFinite(number) ? number.toFixed(1) + "%" : String(value);
  }

  function updateCoverage(value) {
    setChoiceText("u760", formatLivePercentText(value));
  }

  function updateBroadcast(value) {
    setChoiceText("u767", formatLivePercentText(value));
  }

  function updateTrainButton(text, tone) {
    setChoiceText("u773", text);
    setChoiceText("u774", text);
    var node = byId("u773_div") || byId("u774_div");
    if (node) {
      node.style.backgroundColor = tone === "running" ? "#dc7274" : "#3961f6";
    }
  }

  function resetTrainingResultState() {
    state.running = false;
    state.chartPoints = [];
    updateTrainButton("启动训练", "idle");
    updateStatus("未开始");
    updateTime("--:--:--");
    updateCoverage("--");
    updateBroadcast("--");
    renderTrainingChart();
  }

  function closeStream() {
    if (eventSource) {
      eventSource.close();
      eventSource = null;
    }
  }

  function stopTraining() {
    closeStream();
    state.running = false;
    updateTrainButton("启动训练", "idle");
    updateStatus("已停止查看");
    addConsole("warn", "已停止前端监听；后端训练任务如果已启动，仍会继续执行直到结束。");
  }

  function bindTrainingButton() {
    ["u772", "u773", "u774"].forEach(function (id) {
      var node = byId(id);
      if (!node || node.dataset.liveBound) return;
      node.dataset.liveBound = "true";
      node.style.pointerEvents = "auto";
      node.style.cursor = "pointer";
      markInteractive(node);
      isolatePointerEvents(node);
      node.addEventListener("click", function (event) {
        event.preventDefault();
        event.stopPropagation();
        if (state.running) stopTraining();
        else startTraining();
      }, true);
    });
  }

  function bindHistoryButton() {
    var trigger = byId("u1155");
    if (!trigger || trigger.dataset.liveBound) return;
    trigger.dataset.liveBound = "true";
    trigger.style.pointerEvents = "auto";
    trigger.style.cursor = "pointer";
    markInteractive(trigger);
    isolatePointerEvents(trigger);
    trigger.addEventListener("click", function (event) {
      event.preventDefault();
      event.stopPropagation();
      loadTrainingHistory();
    }, true);
  }

  async function loadTrainingHistory() {
    try {
      var response = await fetch(API + "/train/artifacts");
      if (!response.ok) throw new Error(await response.text());
      var payload = await response.json();
      var artifacts = Array.isArray(payload.artifacts) ? payload.artifacts : [];
      var existing = byId("training-history-modal");
      if (existing) existing.remove();

      var modal = document.createElement("div");
      modal.id = "training-history-modal";
      modal.style.cssText = "position:fixed;inset:0;background:rgba(2,6,23,0.48);z-index:99999;display:flex;align-items:center;justify-content:center;";
      var rows = artifacts.slice(0, 16).map(function (item, index) {
        var timeText = item.updated_at ? new Date(item.updated_at * 1000).toLocaleString("zh-CN") : "--";
        return "<tr>" +
          "<td style='padding:10px 12px;border-bottom:1px solid #eef2f7;'>" + (index + 1) + "</td>" +
          "<td style='padding:10px 12px;border-bottom:1px solid #eef2f7;'>" + escapeHtml(item.scenario_name || "--") + "</td>" +
          "<td style='padding:10px 12px;border-bottom:1px solid #eef2f7;'>" + escapeHtml(String(item.algorithm || "").toUpperCase() || "--") + "</td>" +
          "<td style='padding:10px 12px;border-bottom:1px solid #eef2f7;'>" + escapeHtml(timeText) + "</td>" +
          "<td style='padding:10px 12px;border-bottom:1px solid #eef2f7;'>" +
            "<button type='button' data-run-dir='" + escapeHtml(item.run_dir || "") + "' style='padding:6px 12px;border:1px solid #dbe4ff;border-radius:8px;background:#eef4ff;color:#1d4ed8;cursor:pointer;'>查看</button>" +
          "</td>" +
          "</tr>";
      }).join("");
      modal.innerHTML =
        "<div style='width:920px;max-height:640px;overflow:auto;background:#fff;border-radius:14px;padding:24px;box-shadow:0 24px 60px rgba(15,23,42,0.2);'>" +
        "<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;'>" +
        "<strong style='font-size:18px;color:#0f172a;'>训练记录</strong>" +
        "<button type='button' style='border:0;background:none;font-size:16px;cursor:pointer;color:#64748b;' onclick='this.closest(\"#training-history-modal\").remove()'>关闭</button>" +
        "</div>" +
        "<table style='width:100%;border-collapse:collapse;font-size:14px;color:#334155;'>" +
        "<thead><tr style='background:#f8fafc;'><th style='padding:10px 12px;text-align:left;'>序号</th><th style='padding:10px 12px;text-align:left;'>场景</th><th style='padding:10px 12px;text-align:left;'>算法</th><th style='padding:10px 12px;text-align:left;'>更新时间</th><th style='padding:10px 12px;text-align:left;'>操作</th></tr></thead>" +
        "<tbody>" + (rows || "<tr><td colspan='5' style='padding:28px 0;text-align:center;color:#94a3b8;'>暂无训练记录</td></tr>") + "</tbody>" +
        "</table>" +
        "</div>";
      document.body.appendChild(modal);
      modal.addEventListener("click", function (event) {
        if (event.target === modal) modal.remove();
      });
      Array.prototype.forEach.call(modal.querySelectorAll("button[data-run-dir]"), function (button) {
        button.addEventListener("click", function (event) {
          event.preventDefault();
          event.stopPropagation();
          viewTrainingHistoryDetail(button.getAttribute("data-run-dir"));
        });
      });
    } catch (error) {
      addConsole("error", "加载训练记录失败：" + (error && error.message ? error.message : error));
    }
  }

  async function viewTrainingHistoryDetail(runDir) {
    if (!runDir) {
      addConsole("warn", "当前训练记录缺少运行目录，无法查看详情。");
      return;
    }
    try {
      var response = await fetch(API + "/train/artifacts/detail?run_dir=" + encodeURIComponent(runDir));
      if (!response.ok) throw new Error(await response.text());
      var detail = await response.json();
      openTrainingHistoryDetailModal(detail);
    } catch (error) {
      addConsole("error", "加载训练记录详情失败：" + (error && error.message ? error.message : error));
    }
  }

  function openTrainingHistoryDetailModal(detail) {
    var existing = byId("training-history-detail-modal");
    if (existing) existing.remove();

    var modal = document.createElement("div");
    modal.id = "training-history-detail-modal";
    modal.style.cssText = "position:fixed;inset:0;background:rgba(2,6,23,0.58);z-index:100000;display:flex;align-items:center;justify-content:center;";

    var evalRows = Array.isArray(detail.eval_history) && detail.eval_history.length
      ? detail.eval_history.map(function (item, index) {
        return "<tr>" +
          "<td style='padding:10px 12px;border-bottom:1px solid #eef2f7;'>" + (index + 1) + "</td>" +
          "<td style='padding:10px 12px;border-bottom:1px solid #eef2f7;'>" + escapeHtml(String(item.step == null ? "--" : item.step)) + "</td>" +
          "<td style='padding:10px 12px;border-bottom:1px solid #eef2f7;'>" + escapeHtml(formatMetric(item.avg_reward, 3)) + "</td>" +
          "<td style='padding:10px 12px;border-bottom:1px solid #eef2f7;'>" + escapeHtml(formatPercent(item.avg_coverage)) + "</td>" +
          "<td style='padding:10px 12px;border-bottom:1px solid #eef2f7;'>" + escapeHtml(formatPercent(item.avg_broadcast)) + "</td>" +
          "</tr>";
      }).join("")
      : "<tr><td colspan='5' style='padding:24px 0;text-align:center;color:#94a3b8;'>暂无评估记录</td></tr>";

    var config = detail.config || {};
    var experimentCfg = config.experiment || {};
    var trainCfg = config.train || {};
    var envCfg = config.multimodal_env || {};
    var algoCfg = config.algorithm || {};
    var timeText = detail.updated_at ? new Date(detail.updated_at * 1000).toLocaleString("zh-CN") : "--";
    var summaryCards = [
      { label: "训练轮次", value: detail.episode_count || 0 },
      { label: "总步数", value: detail.total_timesteps || "--" },
      { label: "最佳奖励", value: formatMetric(detail.best_reward, 3) },
      { label: "最终奖励", value: formatMetric(detail.last_reward, 3) },
      { label: "最佳覆盖率", value: formatPercent(detail.best_coverage) },
      { label: "最终广播覆盖", value: formatPercent(detail.last_broadcast) },
    ].map(function (item) {
      return "<div style='padding:14px 16px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;'>" +
        "<div style='font-size:12px;color:#64748b;margin-bottom:8px;'>" + escapeHtml(item.label) + "</div>" +
        "<div style='font-size:20px;color:#0f172a;font-weight:700;'>" + escapeHtml(String(item.value)) + "</div>" +
      "</div>";
    }).join("");

    modal.innerHTML =
      "<div style='width:1080px;max-height:760px;overflow:auto;background:#fff;border-radius:16px;padding:24px;box-shadow:0 24px 60px rgba(15,23,42,0.28);'>" +
      "<div style='display:flex;justify-content:space-between;align-items:flex-start;gap:16px;margin-bottom:18px;'>" +
      "<div>" +
      "<div style='font-size:22px;line-height:1.4;color:#0f172a;font-weight:700;'>" + escapeHtml(detail.scenario_name || "训练详情") + "</div>" +
      "<div style='margin-top:6px;font-size:13px;color:#64748b;'>算法 " + escapeHtml(String(detail.algorithm || "").toUpperCase()) + " · 环境 " + escapeHtml(detail.env_type || "--") + " · 更新时间 " + escapeHtml(timeText) + "</div>" +
      "</div>" +
      "<button type='button' style='border:0;background:none;font-size:16px;cursor:pointer;color:#64748b;' onclick='this.closest(\"#training-history-detail-modal\").remove()'>关闭</button>" +
      "</div>" +
      "<div style='display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin-bottom:18px;'>" + summaryCards + "</div>" +
      "<div style='display:grid;grid-template-columns:1.2fr 1fr;gap:16px;margin-bottom:18px;'>" +
      "<div style='padding:16px;border:1px solid #e2e8f0;border-radius:14px;'>" +
      "<div style='font-size:16px;font-weight:700;color:#0f172a;margin-bottom:12px;'>训练配置</div>" +
      "<div style='font-size:13px;line-height:1.9;color:#334155;'>" +
      "<div>场景：<strong>" + escapeHtml(detail.scenario_name || "--") + "</strong></div>" +
      "<div>奖励模式：<strong>" + escapeHtml(detail.reward_mode || "--") + "</strong></div>" +
      "<div>训练总步数：<strong>" + escapeHtml(String(trainCfg.total_timesteps || detail.total_timesteps || "--")) + "</strong></div>" +
      "<div>评估间隔：<strong>" + escapeHtml(String(trainCfg.eval_interval || "--")) + "</strong></div>" +
      "<div>学习率：<strong>" + escapeHtml(String(algoCfg.learning_rate || "--")) + "</strong></div>" +
      "<div>折扣因子：<strong>" + escapeHtml(String(algoCfg.gamma || "--")) + "</strong></div>" +
      "<div>熵系数：<strong>" + escapeHtml(String(algoCfg.entropy_coef || "--")) + "</strong></div>" +
      "<div>Rollout 步长：<strong>" + escapeHtml(String(trainCfg.rollout_steps || "--")) + "</strong></div>" +
      "<div>场景预算上限：<strong>" + escapeHtml(String(envCfg.max_base_stations || "--")) + "</strong></div>" +
      "</div>" +
      "</div>" +
      "<div style='padding:16px;border:1px solid #e2e8f0;border-radius:14px;'>" +
      "<div style='font-size:16px;font-weight:700;color:#0f172a;margin-bottom:12px;'>产物信息</div>" +
      "<div style='font-size:13px;line-height:1.9;color:#334155;word-break:break-all;'>" +
      "<div>Checkpoint：<strong>" + escapeHtml(detail.checkpoint_path || "--") + "</strong></div>" +
      "<div>运行目录：<strong>" + escapeHtml(detail.run_dir || "--") + "</strong></div>" +
      "<div>实验环境：<strong>" + escapeHtml(experimentCfg.env_type || detail.env_type || "--") + "</strong></div>" +
      "<div>算法：<strong>" + escapeHtml(experimentCfg.algorithm || detail.algorithm || "--") + "</strong></div>" +
      "</div>" +
      "</div>" +
      "</div>" +
      "<div style='padding:16px;border:1px solid #e2e8f0;border-radius:14px;'>" +
      "<div style='font-size:16px;font-weight:700;color:#0f172a;margin-bottom:12px;'>评估结果</div>" +
      "<table style='width:100%;border-collapse:collapse;font-size:13px;color:#334155;'>" +
      "<thead><tr style='background:#f8fafc;'><th style='padding:10px 12px;text-align:left;'>序号</th><th style='padding:10px 12px;text-align:left;'>Step</th><th style='padding:10px 12px;text-align:left;'>平均奖励</th><th style='padding:10px 12px;text-align:left;'>平均覆盖率</th><th style='padding:10px 12px;text-align:left;'>平均广播覆盖</th></tr></thead>" +
      "<tbody>" + evalRows + "</tbody>" +
      "</table>" +
      "</div>" +
      "</div>";

    document.body.appendChild(modal);
    modal.addEventListener("click", function (event) {
      if (event.target === modal) modal.remove();
    });
  }

  function handleTrainingEvent(event) {
    var payload = event && event.payload ? event.payload : {};
    if (event.type === "status") {
      var stateText = payload.state || "unknown";
      addConsole("status", "状态：" + stateText + (payload.step != null ? " step=" + payload.step : ""));
      if (stateText === "completed") {
        updateStatus("已完成");
        updateTrainButton("启动训练", "idle");
        updateTime(new Date().toLocaleTimeString("zh-CN", { hour12: false }));
        state.running = false;
        closeStream();
      } else if (stateText === "running") {
        updateStatus("运行中");
      } else if (stateText === "initializing") {
        updateStatus("初始化中");
      } else if (stateText === "failed") {
        updateStatus("失败");
      }
      return;
    }
    if (event.type === "episode") {
      pushChartPoint({
        label: "E" + (payload.episode || "?"),
        coverage: Number(payload.coverage || 0),
        broadcast: Number(payload.broadcast || 0),
        reward: Number(payload.reward || 0),
      });
      updateCoverage((Number(payload.coverage || 0) * 100).toFixed(1));
      updateBroadcast((Number(payload.broadcast || 0) * 100).toFixed(1));
      updateTime(new Date().toLocaleTimeString("zh-CN", { hour12: false }));
      addConsole(
        "episode",
        "episode=" + (payload.episode || "?") +
          " reward=" + Number(payload.reward || 0).toFixed(3) +
          " coverage=" + (Number(payload.coverage || 0) * 100).toFixed(1) + "%" +
          " broadcast=" + (Number(payload.broadcast || 0) * 100).toFixed(1) + "%"
      );
      return;
    }
    if (event.type === "update") {
      pushChartPoint({
        label: payload.update != null ? "U" + payload.update : "S" + (payload.step || "?"),
        coverage: Number(payload.mean_coverage || 0),
        broadcast: Number(payload.mean_broadcast || 0),
        reward: Number(payload.mean_reward || 0),
      });
      updateCoverage((Number(payload.mean_coverage || 0) * 100).toFixed(1));
      updateBroadcast((Number(payload.mean_broadcast || 0) * 100).toFixed(1));
      updateTime(new Date().toLocaleTimeString("zh-CN", { hour12: false }));
      addConsole(
        "update",
        "step=" + (payload.step || "?") +
          (payload.update != null ? " update=" + payload.update : "") +
          " mean_reward=" + Number(payload.mean_reward || 0).toFixed(3) +
          " mean_coverage=" + (Number(payload.mean_coverage || 0) * 100).toFixed(1) + "%" +
          " mean_broadcast=" + (Number(payload.mean_broadcast || 0) * 100).toFixed(1) + "%"
      );
      return;
    }
    if (event.type === "evaluation") {
      pushChartPoint({
        label: "V" + (payload.step || "?"),
        coverage: Number(payload.avg_coverage || 0),
        broadcast: Number(payload.avg_broadcast || 0),
        reward: Number(payload.avg_reward || 0),
      });
      updateCoverage((Number(payload.avg_coverage || 0) * 100).toFixed(1));
      updateBroadcast((Number(payload.avg_broadcast || 0) * 100).toFixed(1));
      updateTime(new Date().toLocaleTimeString("zh-CN", { hour12: false }));
      addConsole(
        "evaluation",
        "eval step=" + (payload.step || "?") +
          " avg_reward=" + Number(payload.avg_reward || 0).toFixed(3) +
          " avg_coverage=" + (Number(payload.avg_coverage || 0) * 100).toFixed(1) + "%" +
          " avg_broadcast=" + (Number(payload.avg_broadcast || 0) * 100).toFixed(1) + "%"
      );
      return;
    }
    if (event.type === "completed") {
      addConsole(
        "completed",
        "训练完成：step=" + (payload.step || "?") +
          " episodes=" + (payload.episodes || 0) +
          " total_timesteps=" + (payload.total_timesteps || 0)
      );
      updateStatus("已完成");
      updateTrainButton("启动训练", "idle");
      state.running = false;
      return;
    }
    if (event.type === "error") {
      addConsole("error", payload.message || "训练失败");
      updateStatus("失败");
      updateTrainButton("启动训练", "idle");
      state.running = false;
    }
  }

  function subscribeTrainingStream(runId) {
    closeStream();
    eventSource = new EventSource(API + "/train/" + runId + "/stream");
    eventSource.onopen = function () {
      addConsole("info", "已建立训练事件流连接。");
    };
    eventSource.onmessage = function (rawEvent) {
      if (!rawEvent.data) return;
      try {
        handleTrainingEvent(JSON.parse(rawEvent.data));
      } catch (error) {
        addConsole("warn", "无法解析训练事件：" + rawEvent.data);
      }
    };
    eventSource.onerror = function () {
      addConsole("warn", "训练事件流中断。");
      closeStream();
      if (state.running) {
        state.running = false;
        updateTrainButton("启动训练", "idle");
        updateStatus("连接断开");
      }
    };
  }

  async function fetchScenarios() {
    var response = await fetch(API + "/scenarios");
    if (!response.ok) throw new Error(await response.text());
    var payload = await response.json();
    state.scenarios = Array.isArray(payload.scenarios) ? payload.scenarios : [];
    updateScenarioByType(state.disasterType);
  }

  async function startTraining() {
    if (!state.scenarioName) {
      addConsole("error", "当前没有可用场景，无法启动训练。");
      return;
    }

    var scenario = currentScenario();
    var payload = {
      scenario_name: state.scenarioName,
      env_type: state.envType,
      algorithm: backendAlgorithmKey(state.algorithm),
      total_timesteps: Number(state.totalTimesteps || 12000),
      stochastic_eval: Boolean(state.stochasticEval),
      reward_mode: currentRewardMode(),
      learning_rate: Number(state.learningRate || 0.0003),
      discount_factor: Number(state.discountFactor || 0.99),
      batch_size: Number(state.batchSize || 256),
      rollout_steps: Number(state.rolloutSteps || 1024),
      entropy_coef: Number(state.entropyCoef || 0.01),
      clip_range: Number(state.clipRange || 0.2),
      eval_interval: Number(state.evalInterval || 4),
    };

    state.running = true;
    ensureDashboard();
    clearChart();
    updateTrainButton("停止训练", "running");
    updateStatus("初始化中");
    updateTime("--:--:--");
    syncTrainingDeviceFields();
    addConsole("info", "准备启动训练：" + (scenario ? scenarioDisplayName(scenario) : state.scenarioName) + " / " + algorithmDisplayName(state.algorithm));
    addConsole("info", "当前训练场景设备配置：" + trainingDeviceSummaryLabel() + "。");
    addConsole("request", "POST /api/train " + JSON.stringify(payload));

    try {
      var response = await fetch(API + "/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!response.ok) throw new Error(await response.text());
      var result = await response.json();
      addConsole("success", "训练任务已启动，run_id=" + result.run_id);
      subscribeTrainingStream(result.run_id);
    } catch (error) {
      addConsole("error", "启动训练失败：" + (error && error.message ? error.message : error));
      updateStatus("失败");
      updateTrainButton("启动训练", "idle");
      state.running = false;
    }
  }

  function initialize() {
    unbindAxureHandlers([
      "u545", "u545_state0", "u545_state1", "u546", "u547", "u548", "u549", "u730", "u731", "u732", "u733",
      "u775", "u775_state0", "u775_state1", "u776", "u777", "u778", "u779",
      "u1118", "u1118_state0", "u1118_state1", "u1119", "u1120", "u1121", "u1122"
    ]);
    lockPanelExpanded("u545", "u545_state0", "u545_state1", ["u546", "u547", "u548", "u549", "u730", "u731", "u732", "u733"]);
    stabilizeEditablePanel("u545_state0_content", "u545_state0", "u545_state1");
    stabilizeEditablePanel("u775_state0_content", "u775_state0", "u775_state1");
    installBlankAreaGuard("u545_state0_content", "u545_state0", "u545_state1");
    installBlankAreaGuard("u775_state0_content", "u775_state0", "u775_state1");
    installBlankAreaGuard("u1118_state0_content", "u1118_state0", "u1118_state1");
    ensureScrollPanel();
    relayoutSections();
    mountEditableFields();
    resizeScenarioDescriptionBox();
    syncDeviceSectionVisibility();
    bindSceneActionButtons();
    bindDeviceSlotActions();
    mountParameterFields();
    mountParameterTabs();
    mountAlgorithmCards();
    replaceAllDropdownArrowGlyphs();
    bindTrainingButton();
    bindHistoryButton();
    ensureDashboard();
    resetTrainingResultState();
    relayoutSections();
    window.setTimeout(replaceAllDropdownArrowGlyphs, 0);
    window.setTimeout(replaceAllDropdownArrowGlyphs, 300);
    addConsole("info", "训练页已切换为真实训练联调模式，正在同步场景配置。");
    fetchScenarios()
      .then(function () {
        addConsole("success", "场景配置同步完成，可直接调整参数并启动训练。");
      })
      .catch(function (error) {
        addConsole("error", "加载场景配置失败：" + (error && error.message ? error.message : error));
      });
  }

  window.__protoTrainingCleanup = function () {
    closeStream();
  };

  initialize();
}

export function injectPrototypeTraining(doc) {
  if (!doc) return;
  if (doc.defaultView && doc.defaultView.__protoTrainingBootstrapped) return;

  if (doc.defaultView && typeof doc.defaultView.__protoTrainingCleanup === "function") {
    try {
      doc.defaultView.__protoTrainingCleanup();
    } catch (error) {
      console.warn("prototype training cleanup failed", error);
    }
  }

  const previousScript = doc.getElementById("training-api-inject");
  if (previousScript) previousScript.remove();

  const script = doc.createElement("script");
  script.id = "training-api-inject";
  script.textContent = `window.__protoTrainingBootstrapped = true;(${trainingInjector.toString()})(${JSON.stringify("/api")},${JSON.stringify(COMMUNICATION_TYPE_OPTIONS)},${JSON.stringify(DEFAULT_DEVICE_TEMPLATES)});`;
  doc.head?.appendChild(script);
}
