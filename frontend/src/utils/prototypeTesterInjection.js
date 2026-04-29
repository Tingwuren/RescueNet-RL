const buildInjectionScript = (apiBase) => `
(function () {
  var API = ${JSON.stringify(apiBase)};
  var TEST_HISTORY_KEY = "prototype-tester-history";
  var ALGORITHMS = [
    { key: "ppo", label: "PPO（基线）" },
    { key: "dqn", label: "DQN（大动作空间）" },
    { key: "a3c", label: "A3C（多目标）" },
    { key: "mppo", label: "MPPO（多头策略）" }
  ];
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
    running: false
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

  function algorithmLabel(key) {
    var match = ALGORITHMS.find(function (item) {
      return item.key === key;
    });
    return match ? match.label : key.toUpperCase();
  }

  function disasterLabel(type) {
    var mapping = {
      flood: "洪涝孤岛通信恢复",
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

  function findMatchingArtifact() {
    return state.artifacts.find(function (artifact) {
      return artifact.scenario_name === state.scenarioName && artifact.algorithm === state.algorithm && artifact.checkpoint_path;
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

  function formatMetric(value, digits) {
    if (typeof value !== "number" || !isFinite(value)) return "--";
    return value.toFixed(digits == null ? 2 : digits);
  }

  function formatDateTime(value) {
    if (!value) return "--";
    try {
      return new Date(value).toLocaleString("zh-CN");
    } catch (error) {
      return "--";
    }
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
    var imported = byId("u2844");
    var importedImg = byId("u2844_img");
    var deployment = byId("u2843");
    var deploymentImg = byId("u2843_img");

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

  function ensureOverlay() {
    var panel = byId("u2852");
    var content = byId("u2852_state0_content");
    if (panel && content) {
      setPanelVisible("u2852", true);
      setPanelVisible("u2857", true);
      content.style.position = "relative";
      content.style.pointerEvents = "auto";
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
    setText("u2962_text", result ? formatPercent(result.avg_final_coverage) : "--");
    setText("u2969_text", totalReward != null ? formatNumber(totalReward, 2) : "--");
    setText("u2976_text", finalCoverage != null ? formatPercent(finalCoverage) : "--");
    setText("u2983_text", finalBroadcast != null ? formatPercent(finalBroadcast) : "--");
    setText("u2990_text", remainingBudget != null ? formatNumber(remainingBudget, 1) : "--");

    if (result && result.scene_export) {
      setText("u2997_text", "受灾场景文件：" + (result.scene_export.disaster_scene_path || "--"));
      setText("u2998_text", "部署后场景文件：" + (result.scene_export.deployment_scene_path || "--"));
    } else {
      setText("u2997_text", "受灾场景文件：--");
      setText("u2998_text", "部署后场景文件：--");
    }

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

  function updateScenarioDecorations() {
    var scenario = currentScenario();
    if (!scenario) return;

    setText("u3034_text", comboLabel(state.scenarioName, state.algorithm));
    setText("u3038_text", comboLabel(state.scenarioName, state.algorithm) + " 策略测试");
    setText("u2851_text", "区域：" + ((scenario.region_grid && scenario.region_grid.name) || scenarioLabel(state.scenarioName)) +
      "（离散网格 " + (scenario.region_grid && scenario.region_grid.rows || scenario.grid_size || "--") + " × " +
      (scenario.region_grid && scenario.region_grid.cols || scenario.grid_size || "--") + "）");
    setText("u3038_text", comboLabel(state.scenarioName, state.algorithm) + " 策略测试");
  }

  function closeModal(id) {
    var node = byId(id);
    if (node) node.remove();
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
    setStatus("同步场景中", "warning");

    try {
      var response = await fetch(API + "/simulate/scene", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          scenario_name: state.scenarioName,
          env_type: "multimodal"
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
          "，残余基站 " + (((initialState.residual_base_stations || []).length)) + " 个。",
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
      setTabVisual(payload.scene_export && payload.scene_export.deployment_scene ? "deployment" : "imported");
      state.activeSceneTab = payload.scene_export && payload.scene_export.deployment_scene ? "deployment" : "imported";
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

    if (!state.importedScene) {
      appendTerminalLine("场景尚未同步，先拉取导入场景。", "warning");
      await importScene();
      if (!state.importedScene) {
        return;
      }
    }

    state.running = true;
    setStartButton("测试中...", "running", false);
    setStatus("测试中", "warning");
    updateSummaryLine(comboLabel(state.scenarioName, state.algorithm) + "，测试执行中，实时日志已接入真实后端流。");

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
          episodes: 1,
          stochastic_eval: true,
          eval_seed: 13,
          custom_devices: buildImportedDevices(),
          custom_base_stations: null
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

    var importedTab = byId("u2844");
    if (importedTab && !importedTab.dataset.liveBound) {
      importedTab.dataset.liveBound = "true";
      importedTab.style.pointerEvents = "auto";
      importedTab.addEventListener("click", function (event) {
        event.preventDefault();
        event.stopPropagation();
        state.activeSceneTab = "imported";
        setTabVisual("imported");
        var note = byId("tester-scene-note");
        if (note) note.textContent = "导入的场景";
      }, true);
    }

    var deploymentTab = byId("u2843");
    if (deploymentTab && !deploymentTab.dataset.liveBound) {
      deploymentTab.dataset.liveBound = "true";
      deploymentTab.style.pointerEvents = "auto";
      deploymentTab.addEventListener("click", function (event) {
        event.preventDefault();
        event.stopPropagation();
        state.activeSceneTab = "deployment";
        setTabVisual("deployment");
        var note = byId("tester-scene-note");
        if (note) note.textContent = "部署后场景";
      }, true);
    }

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
  }

  async function bootstrap() {
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
  script.textContent = buildInjectionScript("/api");
  doc.head && doc.head.appendChild(script);
}
