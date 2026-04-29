function trainingInjector(apiBase) {
  var API = apiBase;
  var LOCAL_SCENES_KEY = "prototype-training-scenes";
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
    residualType1: "backpack_station",
    residualDevice1: "设备1",
    residualLocation1: "",
    residualCount1: 1,
    residualType2: "relay",
    residualDevice2: "设备2",
    residualLocation2: "",
    residualCount2: 1,
    candidateType1: "backpack_station",
    candidateDevice1: "设备1",
    candidateCount1: 2,
    candidateType2: "relay",
    candidateDevice2: "设备2",
    candidateCount2: 2,
    algorithm: "ppo",
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
    chartPoints: [],
  };
  var editors = {};
  var SCENE_DEVICE_TYPE_OPTIONS = [
    { value: "macro_station", label: "宏基站" },
    { value: "backpack_station", label: "背负式基站" },
    { value: "relay", label: "中继设备" },
    { value: "vehicle_station", label: "临时设备/车载设备" },
  ];
  var SCENE_DEVICE_OPTIONS = [
    { value: "设备1", label: "设备1" },
    { value: "设备2", label: "设备2" },
    { value: "设备3", label: "设备3" },
    { value: "设备4", label: "设备4" },
  ];
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

  function relayoutSections() {
    var scenePanel = byId("u545");
    var algoPanel = byId("u1118");
    var parameterPanel = byId("u775");
    var resultPanel = byId("u734");
    var foldContainer = byId("u544_state0");
    var foldContent = byId("u544_state0_content");
    var scrollContent = byId("u543_state0_content");
    var scrollPanel = byId("u543_state0");
    if (!scenePanel || !algoPanel || !parameterPanel || !resultPanel) return;

    var resultVisible = visibleNode(byId("u735"));
    var resultHeight = resultVisible ? 824 : 49;
    var totalHeight = 1747 + resultHeight + 40;

    [
      { node: scenePanel, left: 2, top: 0, zIndex: 1 },
      { node: algoPanel, left: 3, top: 1085, zIndex: 1 },
      { node: parameterPanel, left: 2, top: 1433, zIndex: 1 },
      { node: resultPanel, left: 2, top: 1747, zIndex: 1 },
    ].forEach(function (section) {
      section.node.style.position = "absolute";
      section.node.style.left = section.left + "px";
      section.node.style.top = section.top + "px";
      section.node.style.margin = "0";
      section.node.style.zIndex = String(section.zIndex);
    });

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
      flood: "洪涝孤岛通信恢复",
      earthquake: "地震灾后断链恢复",
      landslide: "泥石流滑坡通信阻断恢复",
      typhoon: "台风灾后残余网络",
    };
    return typeMap[scenario.disaster_type] || scenario.name;
  }

  function currentScenario() {
    return state.scenarios.find(function (item) {
      return item.name === state.scenarioName;
    }) || null;
  }

  function currentRewardMode() {
    var scenario = currentScenario();
    return scenario ? (scenario.default_reward_profile || null) : null;
  }

  function currentScenarioForType(type) {
    return state.scenarios.find(function (item) {
      return item.disaster_type === type;
    }) || state.scenarios[0] || null;
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

  function currentSceneSnapshot() {
    return {
      name: state.scenarioTitle || state.scenarioName || "未命名场景",
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
      residual_devices: [
        { type: state.residualType1, device: state.residualDevice1, location: state.residualLocation1, count: state.residualCount1 },
        { type: state.residualType2, device: state.residualDevice2, location: state.residualLocation2, count: state.residualCount2 },
      ],
      candidate_devices: [
        { type: state.candidateType1, device: state.candidateDevice1, count: state.candidateCount1 },
        { type: state.candidateType2, device: state.candidateDevice2, count: state.candidateCount2 },
      ],
      saved_at: Date.now(),
      source: "local",
    };
  }

  function applySceneSnapshot(snapshot) {
    if (!snapshot) return;
    state.scenarioTitle = snapshot.name || state.scenarioTitle;
    state.scenarioName = snapshot.name || state.scenarioName;
    state.disasterType = snapshot.disaster_type || state.disasterType;
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
    var residualDevices = Array.isArray(snapshot.residual_devices) ? snapshot.residual_devices : [];
    var candidateDevices = Array.isArray(snapshot.candidate_devices) ? snapshot.candidate_devices : [];
    state.residualType1 = residualDevices[0] && residualDevices[0].type ? residualDevices[0].type : state.residualType1;
    state.residualDevice1 = residualDevices[0] && residualDevices[0].device ? residualDevices[0].device : state.residualDevice1;
    state.residualLocation1 = residualDevices[0] && residualDevices[0].location ? residualDevices[0].location : "";
    state.residualCount1 = Number(residualDevices[0] && residualDevices[0].count ? residualDevices[0].count : state.residualCount1);
    state.residualType2 = residualDevices[1] && residualDevices[1].type ? residualDevices[1].type : state.residualType2;
    state.residualDevice2 = residualDevices[1] && residualDevices[1].device ? residualDevices[1].device : state.residualDevice2;
    state.residualLocation2 = residualDevices[1] && residualDevices[1].location ? residualDevices[1].location : "";
    state.residualCount2 = Number(residualDevices[1] && residualDevices[1].count ? residualDevices[1].count : state.residualCount2);
    state.candidateType1 = candidateDevices[0] && candidateDevices[0].type ? candidateDevices[0].type : state.candidateType1;
    state.candidateDevice1 = candidateDevices[0] && candidateDevices[0].device ? candidateDevices[0].device : state.candidateDevice1;
    state.candidateCount1 = Number(candidateDevices[0] && candidateDevices[0].count ? candidateDevices[0].count : state.candidateCount1);
    state.candidateType2 = candidateDevices[1] && candidateDevices[1].type ? candidateDevices[1].type : state.candidateType2;
    state.candidateDevice2 = candidateDevices[1] && candidateDevices[1].device ? candidateDevices[1].device : state.candidateDevice2;
    state.candidateCount2 = Number(candidateDevices[1] && candidateDevices[1].count ? candidateDevices[1].count : state.candidateCount2);

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
    });
    choice.appendChild(select);
    editors[choiceId] = select;
  }

  function mountAlgorithmCards() {
    var cards = {
      ppo: ["u1124", "u1125", "u1126"],
      dqn: ["u1127", "u1128", "u1129"],
      a3c: ["u1130", "u1131", "u1132"],
      mppo: ["u1133", "u1134", "u1135"],
    };
    function bindAlgorithmSelection(algorithm, event) {
      if (event) {
        event.preventDefault();
        event.stopPropagation();
      }
      state.algorithm = algorithm;
      syncAlgorithmCards();
      addConsole("info", "已切换训练算法：" + state.algorithm.toUpperCase());
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
    var cards = {
      u1124: "ppo",
      u1127: "dqn",
      u1130: "a3c",
      u1133: "mppo",
    };
    Object.keys(cards).forEach(function (id) {
      var node = byId(id);
      if (!node) return;
      var selected = cards[id] === state.algorithm;
      node.classList.toggle("selected", selected);
      node.style.outline = "";
      node.style.outlineOffset = "";
      var img = byId(id + "_img");
      if (img) {
        img.classList.toggle("selected", selected);
        img.src = selected ? ALGORITHM_CARD_IMAGE.selected : ALGORITHM_CARD_IMAGE.normal;
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
    if (editors.u641 && !editors.u641.dataset.userEdited) editors.u641.value = scenario.name;
    if (editors.u644) state.adminDivision = scenario.region_grid && scenario.region_grid.name ? scenario.region_grid.name : scenarioDisplayName(scenario);
    setChoiceText("u644", state.adminDivision || "请选择行政区划");
    if (editors.u648 && (!editors.u648.value || !editors.u648.dataset.userEdited)) editors.u648.value = String(scenario.candidate_sites || 24);
    if (editors.u654 && (!editors.u654.value || !editors.u654.dataset.userEdited)) {
      var rows = scenario.region_grid && scenario.region_grid.rows ? scenario.region_grid.rows : (scenario.grid_size || "--");
      var cols = scenario.region_grid && scenario.region_grid.cols ? scenario.region_grid.cols : (scenario.grid_size || "--");
      editors.u654.value = "离散网格 " + rows + " × " + cols;
    }
    if (editors.u657 && (!editors.u657.value || !editors.u657.dataset.userEdited)) {
      editors.u657.value = "按当前场景区域网格粒度自动映射";
    }
    state.scenarioTitle = scenario.name;
    state.adminDivision = state.adminDivision || scenarioDisplayName(scenario);
    state.candidateSiteCount = Number(scenario.candidate_sites || state.candidateSiteCount);
  }

  function updateScenarioByType(type) {
    var scenario = currentScenarioForType(type);
    if (!scenario) return;
    state.scenarioName = scenario.name;
    state.disasterType = scenario.disaster_type || type;
    updateScenarioFields();
  }

  function mountEditableFields() {
    var scenePanel = { panelVisibleId: "u545_state0", panelHiddenId: "u545_state1" };
    mountSelect("u554", "u561", "disasterType", {
      panelVisibleId: scenePanel.panelVisibleId,
      panelHiddenId: scenePanel.panelHiddenId,
      items: [
        { value: "flood", label: "洪涝" },
        { value: "earthquake", label: "地震" },
        { value: "landslide", label: "滑坡" },
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
        { value: "沿海台风残余网络区", label: "沿海台风残余网络区" },
        { value: "洪涝孤岛网格区", label: "洪涝孤岛网格区" },
        { value: "滑坡阻断山区网格", label: "滑坡阻断山区网格" },
      ],
    });
    mountInput("u648", "candidateSiteCount", { type: "number", min: 1, step: 1, panelVisibleId: scenePanel.panelVisibleId, panelHiddenId: scenePanel.panelHiddenId });
    mountInput("u651", "priorityEquipment", { type: "text", panelVisibleId: scenePanel.panelVisibleId, panelHiddenId: scenePanel.panelHiddenId });
    mountInput("u654", "coverageRange", { type: "text", panelVisibleId: scenePanel.panelVisibleId, panelHiddenId: scenePanel.panelHiddenId });
    mountInput("u657", "cellGranularity", { type: "text", panelVisibleId: scenePanel.panelVisibleId, panelHiddenId: scenePanel.panelHiddenId });
    mountInput("u717", "budgetLimit", { type: "number", min: 1, step: 1, panelVisibleId: scenePanel.panelVisibleId, panelHiddenId: scenePanel.panelHiddenId });
    mountInput("u720", "dispatchUnit", { type: "text", panelVisibleId: scenePanel.panelVisibleId, panelHiddenId: scenePanel.panelHiddenId });
    mountInput("u723", "teamCount", { type: "number", min: 1, step: 1, panelVisibleId: scenePanel.panelVisibleId, panelHiddenId: scenePanel.panelHiddenId });
    mountSelect("u585", "u592", "residualType1", {
      panelVisibleId: scenePanel.panelVisibleId,
      panelHiddenId: scenePanel.panelHiddenId,
      items: SCENE_DEVICE_TYPE_OPTIONS,
    });
    mountSelect("u596", "u603", "residualDevice1", {
      panelVisibleId: scenePanel.panelVisibleId,
      panelHiddenId: scenePanel.panelHiddenId,
      items: SCENE_DEVICE_OPTIONS,
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
      markInteractive(node);
      isolatePointerEvents(node);
      protectPanelState(node, "u545_state0", "u545_state1");
      node.addEventListener("click", function (event) {
        event.preventDefault();
        event.stopPropagation();
        item.handler();
      }, true);
      var card = byId(item.id + "_div");
      if (card) card.style.backgroundColor = item.tone;
    });
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
        disaster_type: item.disaster_type,
        admin_division: item.region_grid && item.region_grid.name ? item.region_grid.name : scenarioDisplayName(item),
        affected_grid_count: item.candidate_sites || state.affectedGridCount,
        candidate_site_count: item.candidate_sites || state.candidateSiteCount,
        coverage_range: item.grid_size ? String(item.grid_size) : "",
        cell_granularity: item.region_grid && item.region_grid.rows && item.region_grid.cols ? ("离散网格 " + item.region_grid.rows + " × " + item.region_grid.cols) : "",
        source: "builtin",
      };
    });
    var scenes = localScenes.concat(builtinScenes.filter(function (scene) {
      return !localScenes.some(function (localScene) { return localScene.name === scene.name; });
    }));
    var existing = byId("scene-import-modal");
    if (existing) existing.remove();

    var modal = document.createElement("div");
    modal.id = "scene-import-modal";
    modal.style.cssText = "position:fixed;inset:0;background:rgba(2,6,23,0.48);z-index:99999;display:flex;align-items:center;justify-content:center;";
    var rows = scenes.map(function (scene, index) {
      var typeLabel = {
        flood: "洪涝",
        earthquake: "地震",
        landslide: "滑坡",
        typhoon: "台风",
      }[scene.disaster_type] || "--";
      return "<tr data-index='" + index + "' style='cursor:pointer;'>" +
        "<td style='padding:10px 12px;border-bottom:1px solid #eef2f7;'>" + (index + 1) + "</td>" +
        "<td style='padding:10px 12px;border-bottom:1px solid #eef2f7;'>" + (scene.name || "--") + "</td>" +
        "<td style='padding:10px 12px;border-bottom:1px solid #eef2f7;'>" + typeLabel + "</td>" +
        "<td style='padding:10px 12px;border-bottom:1px solid #eef2f7;'>" + (scene.source === "local" ? "本地保存" : "后端场景") + "</td>" +
        "</tr>";
    }).join("");
    modal.innerHTML =
      "<div style='width:920px;max-height:640px;overflow:auto;background:#fff;border-radius:14px;padding:24px;box-shadow:0 24px 60px rgba(15,23,42,0.2);'>" +
      "<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;'>" +
      "<strong style='font-size:18px;color:#0f172a;'>导入场景</strong>" +
      "<button type='button' style='border:0;background:none;font-size:16px;cursor:pointer;color:#64748b;' onclick='this.closest(\"#scene-import-modal\").remove()'>关闭</button>" +
      "</div>" +
      "<table style='width:100%;border-collapse:collapse;font-size:14px;color:#334155;'>" +
      "<thead><tr style='background:#f8fafc;'><th style='padding:10px 12px;text-align:left;'>序号</th><th style='padding:10px 12px;text-align:left;'>场景</th><th style='padding:10px 12px;text-align:left;'>灾害类型</th><th style='padding:10px 12px;text-align:left;'>来源</th></tr></thead>" +
      "<tbody>" + (rows || "<tr><td colspan='4' style='padding:28px 0;text-align:center;color:#94a3b8;'>暂无可导入场景</td></tr>") + "</tbody>" +
      "</table>" +
      "</div>";
    document.body.appendChild(modal);
    modal.addEventListener("click", function (event) {
      if (event.target === modal) modal.remove();
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

  function updateCoverage(value) {
    setChoiceText("u760", value + "%");
  }

  function updateBroadcast(value) {
    setChoiceText("u767", value + "%");
  }

  function updateTrainButton(text, tone) {
    setChoiceText("u773", text);
    setChoiceText("u774", text);
    var node = byId("u773_div") || byId("u774_div");
    if (node) {
      node.style.backgroundColor = tone === "running" ? "#dc7274" : "#3961f6";
    }
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
      algorithm: state.algorithm,
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
    addConsole("info", "准备启动训练：" + (scenario ? scenarioDisplayName(scenario) : state.scenarioName) + " / " + state.algorithm.toUpperCase());
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
    bindSceneActionButtons();
    mountParameterFields();
    mountParameterTabs();
    mountAlgorithmCards();
    bindTrainingButton();
    bindHistoryButton();
    ensureDashboard();
    relayoutSections();
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
  script.textContent = `window.__protoTrainingBootstrapped = true;(${trainingInjector.toString()})(${JSON.stringify("/api")});`;
  doc.head?.appendChild(script);
}
