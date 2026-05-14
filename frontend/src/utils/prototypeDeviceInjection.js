const DEVICE_DRAFT_KEY = "rescuenet-prototype-device-drafts-v1";

const stationImageUrls = {
  macro_bs: new URL("../assets/base-stations/photos/cell-tower-macro-real.jpg", import.meta.url).href,
  delta_macro: new URL("../assets/base-stations/photos/cellular-base-station-real.jpg", import.meta.url).href,
  mmwave_micro: new URL("../assets/base-stations/photos/mmwave-micro-real.jpg", import.meta.url).href,
  wifi_hotspot: new URL("../assets/base-stations/photos/wifi-hotspot-real.jpg", import.meta.url).href,
  shortwave_node: new URL("../assets/base-stations/photos/backpack-station-real.jpg", import.meta.url).href,
  orbital_relay: new URL("../assets/base-stations/photos/ku-ka-vsat-real.jpg", import.meta.url).href,
  satellite_relay: new URL("../assets/base-stations/photos/mobile-vsat-real.jpg", import.meta.url).href,
  satellite_ku: new URL("../assets/base-stations/photos/marine-vsat-real.jpg", import.meta.url).href,
  mesh_uav: new URL("../assets/base-stations/photos/mesh-uav-real.jpg", import.meta.url).href,
};

const fallbackStationImageUrl = new URL("../assets/base-stations/photos/compact-station-real.jpg", import.meta.url).href;

const scenarioNameMap = {
  typhoon_residual: "台风灾后残余网络",
  flood_no_residual: "洪水孤岛通信恢复",
  earthquake_residual: "地震灾后断链恢复",
};

const disasterTypeMap = {
  typhoon: "台风灾害",
  earthquake: "地震灾害",
  flood: "洪水灾害",
  wildfire: "山火灾害",
};

const ORIGINAL_DEVICE_CARD_IDS = ["u3900", "u3910", "u3920", "u3929", "u3938", "u3947", "u3956", "u3965"];

const formatScenarioName = (name) => scenarioNameMap[name] || String(name || "未选择场景").replace(/_/g, " ");
const formatDisasterType = (type) => disasterTypeMap[type] || String(type || "灾害场景");

const resolveDeviceClass = (station) => {
  const text = `${station.name || ""} ${station.label || ""} ${(station.supported_modes || []).join(" ")}`.toLowerCase();
  if (/macro|宏站/.test(text)) return "宏基站";
  if (/mmwave|微站|small/.test(text)) return "微型基站";
  if (/satellite|relay|中继|mesh/.test(text)) return "中继设备";
  if (/shortwave|短波/.test(text)) return "临时设备/车载设备";
  return "背负式基站";
};

const copyStation = (station) => ({
  name: station.name || "",
  label: station.label || station.name || "",
  device_class: station.device_class || resolveDeviceClass(station),
  max_throughput: Number(station.max_throughput || 0),
  max_users: Number(station.max_users || 0),
  device_cost: Number(station.device_cost || 0),
  bandwidth_cost: Number(station.bandwidth_cost || 0),
  supported_modes: Array.isArray(station.supported_modes) ? [...station.supported_modes] : [],
});

const numberValue = (value) => {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
};

const modeText = (modes) => (Array.isArray(modes) && modes.length ? modes.join(" / ") : "未配置");

const resolveStationImage = (station) => stationImageUrls[station?.name] || fallbackStationImageUrl;

const parseModes = (value) =>
  String(value || "")
    .split(/[,/，、|]/)
    .flatMap((item) => item.split(" / "))
    .map((item) => item.trim())
    .filter(Boolean);

const readDrafts = (storage) => {
  try {
    const raw = storage.getItem(DEVICE_DRAFT_KEY);
    return raw ? JSON.parse(raw) : {};
  } catch (error) {
    return {};
  }
};

const writeDrafts = (storage, value) => {
  storage.setItem(DEVICE_DRAFT_KEY, JSON.stringify(value));
};

const downloadJson = (doc, filename, payload) => {
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = doc.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
};

const removePrototypeDeviceCards = (doc) => {
  for (const id of ORIGINAL_DEVICE_CARD_IDS) {
    doc.getElementById(id)?.remove();
  }
  doc.getElementById("u3899")?.remove();
};

const getPrototypeDeviceListContainer = (doc) =>
  doc.getElementById("u3894_state0_content") || doc.getElementById("u3894_state0") || doc.body;

const keepPrototypePanelOpen = (doc) => {
  const listState = doc.getElementById("u3894_state0");
  const createState = doc.getElementById("u3894_state1");
  const detailState = doc.getElementById("u3894_state2");

  if (listState) {
    listState.style.visibility = "visible";
    listState.style.display = "block";
  }
  for (const state of [createState, detailState]) {
    if (state) {
      state.style.visibility = "hidden";
      state.style.display = "none";
    }
  }

};

const injectStyles = (doc) => {
  const old = doc.getElementById("prototype-device-workbench-style");
  if (old) old.remove();

  const style = doc.createElement("style");
  style.id = "prototype-device-workbench-style";
  style.textContent = `
    .prototype-device-workbench {
      position: absolute;
      z-index: 2;
      left: 0;
      top: 72px;
      width: 1640px;
      height: 855px;
      box-sizing: border-box;
      display: flex;
      flex-direction: column;
      gap: 12px;
      overflow: hidden;
      color: #14213d;
      font-family: "Microsoft YaHei", "PingFang SC", "Segoe UI", sans-serif;
      background: rgba(247, 251, 255, 0.96);
      border: 1px solid rgba(24, 144, 255, 0.28);
      border-radius: 8px;
      box-shadow: 0 18px 48px rgba(26, 61, 114, 0.16);
    }

    .proto-device-toolbar {
      display: flex;
      gap: 16px;
      align-items: end;
      justify-content: flex-end;
      padding: 14px 18px 12px;
      background: #ffffff;
      border-bottom: 1px solid rgba(120, 144, 180, 0.18);
    }

    .proto-device-field span,
    .proto-device-stat small,
    .proto-device-list__meta,
    .proto-device-empty {
      color: #667893;
      font-size: 13px;
    }

    .proto-device-controls {
      display: flex;
      flex-wrap: wrap;
      justify-content: flex-end;
      gap: 10px;
      align-items: end;
    }

    .proto-device-field {
      display: flex;
      min-width: 160px;
      flex-direction: row;
      align-items: center;
      gap: 8px;
      font-size: 13px;
    }

    .proto-device-field span {
      flex: 0 0 auto;
      white-space: nowrap;
    }

    .proto-device-field input,
    .proto-device-field select {
      height: 38px;
      border: 1px solid rgba(120, 144, 180, 0.26);
      border-radius: 6px;
      padding: 0 11px;
      color: #16345f;
      background: #ffffff;
      font-size: 14px;
      outline: none;
    }

    .proto-device-field input:read-only {
      background: #f3f6fa;
      color: #64748b;
    }

    .proto-device-button {
      height: 38px;
      border-radius: 6px;
      border: 1px solid rgba(37, 99, 235, 0.22);
      padding: 0 13px;
      font-weight: 700;
      font-size: 13px;
      cursor: pointer;
      transition: transform .15s ease, border-color .15s ease, background .15s ease;
    }

    .proto-device-button:hover {
      transform: translateY(-1px);
      border-color: rgba(37, 99, 235, 0.42);
    }

    .proto-device-button--primary {
      background: #2563eb;
      border-color: #2563eb;
      color: #ffffff;
    }

    .proto-device-button--secondary {
      background: #eef5ff;
      color: #2457b7;
    }

    .proto-device-button--ghost {
      background: #ffffff;
      color: #52657e;
    }

    .proto-device-stats {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      padding: 0 20px;
    }

    .proto-device-stat {
      min-height: 64px;
      padding: 11px 14px;
      border: 1px solid rgba(120, 144, 180, 0.18);
      border-radius: 8px;
      background: #ffffff;
    }

    .proto-device-stat strong {
      display: block;
      margin-top: 5px;
      color: #16345f;
      font-size: 20px;
      line-height: 1.1;
    }

    .proto-device-main {
      display: grid;
      grid-template-columns: 430px minmax(0, 1fr);
      gap: 12px;
      min-height: 0;
      padding: 0 20px 14px;
      flex: 1;
    }

    .proto-device-panel {
      min-height: 0;
      border: 1px solid rgba(120, 144, 180, 0.18);
      border-radius: 8px;
      background: #ffffff;
      overflow: hidden;
    }

    .proto-device-panel__header {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 13px 16px 11px;
      border-bottom: 1px solid rgba(120, 144, 180, 0.16);
    }

    .proto-device-panel__header h2 {
      margin: 0;
      color: #16345f;
      font-size: 18px;
    }

    .proto-device-panel__header p {
      margin: 5px 0 0;
      color: #667893;
      font-size: 13px;
    }

    .proto-device-list {
      display: flex;
      flex-direction: column;
      gap: 8px;
      padding: 12px;
      max-height: calc(100% - 78px);
      overflow: auto;
    }

    .proto-device-card {
      width: 100%;
      border: 1px solid rgba(120, 144, 180, 0.18);
      border-left: 4px solid #7aa7f8;
      border-radius: 7px;
      padding: 10px;
      background: #ffffff;
      text-align: left;
      cursor: pointer;
      display: grid;
      grid-template-columns: 118px minmax(0, 1fr);
      gap: 10px;
      align-items: stretch;
      transition: background .15s ease, border-color .15s ease, transform .15s ease;
    }

    .proto-device-card:hover,
    .proto-device-card--active {
      background: #f4f8ff;
      border-color: rgba(37, 99, 235, 0.42);
      transform: translateY(-1px);
    }

    .proto-device-card__media,
    .proto-device-active-media {
      overflow: hidden;
      border-radius: 6px;
      background: #e8eef7;
    }

    .proto-device-card__media {
      height: 82px;
    }

    .proto-device-card__media img,
    .proto-device-active-media img {
      width: 100%;
      height: 100%;
      display: block;
      object-fit: cover;
    }

    .proto-device-card__body {
      min-width: 0;
    }

    .proto-device-card strong {
      display: block;
      color: #16345f;
      font-size: 16px;
    }

    .proto-device-card span {
      display: inline-flex;
      margin-top: 8px;
      margin-right: 6px;
      padding: 4px 7px;
      border-radius: 999px;
      background: #edf4ff;
      color: #2f5da8;
      font-size: 12px;
    }

    .proto-device-card p {
      margin: 8px 0 0;
      color: #667893;
      font-size: 13px;
      line-height: 1.45;
    }

    .proto-device-editor {
      display: flex;
      flex-direction: column;
      min-height: 0;
    }

    .proto-device-editor__body {
      min-height: 0;
      overflow: auto;
      padding: 16px;
    }

    .proto-device-detail-hero {
      display: grid;
      grid-template-columns: minmax(320px, 46%) minmax(0, 1fr);
      gap: 16px;
      align-items: stretch;
      margin-bottom: 16px;
    }

    .proto-device-active-media {
      height: 282px;
      aspect-ratio: 16 / 9;
      overflow: hidden;
      border-radius: 8px;
      background: #e8eef7;
      border: 1px solid rgba(120, 144, 180, 0.18);
    }

    .proto-device-detail-summary {
      display: grid;
      align-content: start;
      gap: 10px;
      padding: 4px 0;
    }

    .proto-device-detail-summary h3 {
      margin: 0;
      color: #16345f;
      font-size: 20px;
    }

    .proto-device-detail-summary p {
      margin: 0;
      color: #667893;
      font-size: 13px;
      line-height: 1.6;
    }

    .proto-device-detail-kpis {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-top: 2px;
    }

    .proto-device-detail-kpis span {
      display: grid;
      gap: 4px;
      padding: 10px 12px;
      border: 1px solid rgba(120, 144, 180, 0.16);
      border-radius: 8px;
      background: #f7fbff;
      color: #667893;
      font-size: 12px;
    }

    .proto-device-detail-kpis strong {
      color: #16345f;
      font-size: 16px;
    }

    .proto-device-form-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
    }

    .proto-device-field--wide {
      grid-column: span 2;
    }

    .proto-device-table-wrap {
      margin-top: 16px;
      overflow: auto;
      border: 1px solid rgba(120, 144, 180, 0.16);
      border-radius: 8px;
    }

    .proto-device-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      min-width: 820px;
    }

    .proto-device-table th,
    .proto-device-table td {
      padding: 10px 9px;
      border-bottom: 1px solid rgba(120, 144, 180, 0.14);
      text-align: left;
      color: #16345f;
      white-space: nowrap;
    }

    .proto-device-table th {
      background: #f3f7fd;
      color: #52657e;
      font-weight: 700;
    }

    .proto-device-footer {
      display: flex;
      justify-content: flex-start;
      align-items: center;
      gap: 12px;
      padding: 12px 16px;
      border-top: 1px solid rgba(120, 144, 180, 0.16);
      background: #fbfdff;
      color: #667893;
      font-size: 13px;
    }

    .proto-device-loading,
    .proto-device-empty {
      display: grid;
      place-items: center;
      min-height: 280px;
      padding: 28px;
      text-align: center;
    }

    @media (max-width: 1200px) {
      .prototype-device-workbench {
        left: 0;
        width: 1640px;
      }
      .proto-device-toolbar,
      .proto-device-main,
      .proto-device-detail-hero {
        grid-template-columns: 1fr;
      }
      .proto-device-controls {
        justify-content: flex-start;
      }
      .proto-device-stats,
      .proto-device-form-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
    }
  `;
  doc.head?.appendChild(style);
};

const renderWorkbench = (doc, root, apiBase = "/api") => {
  const storage = doc.defaultView?.localStorage || window.localStorage;
  const state = {
    scenarios: [],
    selectedScenarioName: "",
    activeStationName: "",
    drafts: readDrafts(storage),
    dirty: false,
    loading: true,
    error: "",
  };

  const getCurrentScenario = () =>
    state.scenarios.find((scenario) => scenario.name === state.selectedScenarioName) || null;

  const getDefaultStations = (scenario) => (scenario?.base_stations || []).map(copyStation);

  const getDraftStations = (scenario) => {
    if (!scenario) return [];
    const draft = state.drafts[scenario.name];
    if (Array.isArray(draft) && draft.length) return draft.map(copyStation);
    return getDefaultStations(scenario);
  };

  const setDraftStations = (scenarioName, stations) => {
    state.drafts = {
      ...state.drafts,
      [scenarioName]: stations.map(copyStation),
    };
  };

  const getStations = () => getDraftStations(getCurrentScenario());
  const getActiveStation = () => {
    const stations = getStations();
    return stations.find((station) => station.name === state.activeStationName) || stations[0] || null;
  };

  const updateActiveStation = (patch) => {
    const scenario = getCurrentScenario();
    if (!scenario) return;
    const stations = getStations();
    const active = getActiveStation();
    if (!active) return;
    const next = stations.map((station) => (station.name === active.name ? { ...station, ...patch } : station));
    setDraftStations(scenario.name, next);
    state.dirty = true;
    draw();
  };

  const renderStats = (scenario, stations) => {
    const modeCount = stations.reduce((total, station) => total + (station.supported_modes || []).length, 0);
    const stats = [
      ["灾害类型", formatDisasterType(scenario?.disaster_type)],
      ["设备类型", String(stations.length)],
      ["支持模式", String(modeCount)],
      ["候选站点", Number(scenario?.candidate_sites || 0).toLocaleString("zh-CN")],
    ];
    return stats
      .map(
        ([label, value]) => `
          <article class="proto-device-stat">
            <small>${label}</small>
            <strong>${value}</strong>
          </article>
        `
      )
      .join("");
  };

  const renderDeviceList = (stations, active) =>
    stations
      .map(
        (station) => `
          <button type="button" class="proto-device-card ${
            active?.name === station.name ? "proto-device-card--active" : ""
          }" data-device-name="${station.name}">
            <div class="proto-device-card__media">
              <img src="${resolveStationImage(station)}" alt="${station.label || station.name}" loading="lazy" />
            </div>
            <div class="proto-device-card__body">
              <strong>${station.label || station.name}</strong>
              <div>
                <span>${station.device_class || resolveDeviceClass(station)}</span>
                <span>${modeText(station.supported_modes)}</span>
              </div>
              <p>峰值吞吐 ${Number(station.max_throughput || 0).toFixed(1)} Mbps，最大接入 ${Number(
                station.max_users || 0
              )} 用户。</p>
            </div>
          </button>
        `
      )
      .join("");

  const renderTable = (stations) => `
    <div class="proto-device-table-wrap">
      <table class="proto-device-table">
        <thead>
          <tr>
            <th>设备 key</th>
            <th>设备名称</th>
            <th>类型</th>
            <th>峰值吞吐</th>
            <th>最大用户</th>
            <th>设备成本</th>
            <th>带宽成本</th>
            <th>支持模式</th>
          </tr>
        </thead>
        <tbody>
          ${stations
            .map(
              (station) => `
                <tr>
                  <td>${station.name}</td>
                  <td>${station.label}</td>
                  <td>${station.device_class || resolveDeviceClass(station)}</td>
                  <td>${Number(station.max_throughput || 0).toFixed(1)} Mbps</td>
                  <td>${Number(station.max_users || 0)}</td>
                  <td>${Number(station.device_cost || 0).toFixed(2)}</td>
                  <td>${Number(station.bandwidth_cost || 0).toFixed(3)}</td>
                  <td>${modeText(station.supported_modes)}</td>
                </tr>
              `
            )
            .join("")}
        </tbody>
      </table>
    </div>
  `;

  const mountControls = () => {
    const scenario = getCurrentScenario();
    const stations = getStations();
    const active = getActiveStation();

    const scenarioSelect = root.querySelector("[data-role='scenario-select']");
    if (scenarioSelect) {
      scenarioSelect.addEventListener("change", (event) => {
        state.selectedScenarioName = event.target.value;
        const nextScenario = getCurrentScenario();
        state.activeStationName = getDraftStations(nextScenario)[0]?.name || "";
        state.dirty = false;
        draw();
      });
    }

    root.querySelector("[data-role='refresh']")?.addEventListener("click", () => {
      void loadScenarios(true);
    });

    root.querySelector("[data-role='reset']")?.addEventListener("click", () => {
      if (!scenario) return;
      const nextDrafts = { ...state.drafts };
      delete nextDrafts[scenario.name];
      state.drafts = nextDrafts;
      writeDrafts(storage, state.drafts);
      state.activeStationName = getDefaultStations(scenario)[0]?.name || "";
      state.dirty = false;
      draw();
    });

    root.querySelector("[data-role='apply']")?.addEventListener("click", () => {
      writeDrafts(storage, state.drafts);
      state.dirty = false;
      draw();
    });

    root.querySelector("[data-role='export']")?.addEventListener("click", () => {
      if (!scenario) return;
      downloadJson(doc, `${scenario.name}-devices.json`, {
        scenario_name: scenario.name,
        disaster_type: scenario.disaster_type,
        base_stations: stations,
      });
    });

    root.querySelectorAll("[data-device-name]").forEach((button) => {
      button.addEventListener("click", () => {
        state.activeStationName = button.getAttribute("data-device-name") || "";
        draw();
      });
    });

    if (!active) return;

    root.querySelector("[data-field='label']")?.addEventListener("change", (event) => {
      updateActiveStation({ label: event.target.value });
    });
    root.querySelector("[data-field='device_class']")?.addEventListener("change", (event) => {
      updateActiveStation({ device_class: event.target.value });
    });
    root.querySelector("[data-field='max_throughput']")?.addEventListener("change", (event) => {
      updateActiveStation({ max_throughput: numberValue(event.target.value) });
    });
    root.querySelector("[data-field='max_users']")?.addEventListener("change", (event) => {
      updateActiveStation({ max_users: numberValue(event.target.value) });
    });
    root.querySelector("[data-field='device_cost']")?.addEventListener("change", (event) => {
      updateActiveStation({ device_cost: numberValue(event.target.value) });
    });
    root.querySelector("[data-field='bandwidth_cost']")?.addEventListener("change", (event) => {
      updateActiveStation({ bandwidth_cost: numberValue(event.target.value) });
    });
    root.querySelector("[data-field='supported_modes']")?.addEventListener("change", (event) => {
      updateActiveStation({ supported_modes: parseModes(event.target.value) });
    });
  };

  const draw = () => {
    if (state.loading) {
      root.innerHTML = `<div class="proto-device-loading">正在加载灾害场景设备库...</div>`;
      return;
    }

    if (state.error) {
      root.innerHTML = `<div class="proto-device-loading">${state.error}</div>`;
      return;
    }

    const scenario = getCurrentScenario();
    const stations = getStations();
    const active = getActiveStation();
    const options = state.scenarios
      .map(
        (item) =>
          `<option value="${item.name}" ${item.name === state.selectedScenarioName ? "selected" : ""}>${formatScenarioName(
            item.name
          )}</option>`
      )
      .join("");

    root.innerHTML = `
      <header class="proto-device-toolbar">
        <div class="proto-device-controls">
          <label class="proto-device-field">
            <span>灾害场景</span>
            <select data-role="scenario-select">${options}</select>
          </label>
          <button type="button" class="proto-device-button proto-device-button--ghost" data-role="refresh">刷新场景</button>
          <button type="button" class="proto-device-button proto-device-button--ghost" data-role="reset">重置默认</button>
          <button type="button" class="proto-device-button proto-device-button--primary" data-role="apply">应用本地配置</button>
          <button type="button" class="proto-device-button proto-device-button--secondary" data-role="export">导出 JSON</button>
        </div>
      </header>

      <section class="proto-device-stats">${renderStats(scenario, stations)}</section>

      <main class="proto-device-main">
        <aside class="proto-device-panel">
          <div class="proto-device-panel__header">
            <div>
              <h2>${formatScenarioName(scenario?.name)}</h2>
              <p>${formatDisasterType(scenario?.disaster_type)}，${stations.length} 类设备。</p>
            </div>
          </div>
          <div class="proto-device-list">
            ${stations.length ? renderDeviceList(stations, active) : `<div class="proto-device-empty">当前场景没有设备配置。</div>`}
          </div>
        </aside>

        <section class="proto-device-panel proto-device-editor">
          <div class="proto-device-panel__header">
            <div>
              <h2>${active ? active.label : "未选择设备"}</h2>
              <p>${active ? `${active.name} / ${modeText(active.supported_modes)}` : "请选择左侧设备类型。"}</p>
            </div>
          </div>

          <div class="proto-device-editor__body">
            ${
              active
                ? `
              <section class="proto-device-detail-hero">
                <div class="proto-device-active-media">
                  <img src="${resolveStationImage(active)}" alt="${active.label || active.name}" />
                </div>
                <div class="proto-device-detail-summary">
                  <h3>${active.label || active.name}</h3>
                  <p>${active.device_class || resolveDeviceClass(active)}，支持 ${modeText(active.supported_modes)}。</p>
                  <div class="proto-device-detail-kpis">
                    <span>峰值吞吐<strong>${Number(active.max_throughput || 0).toFixed(1)} Mbps</strong></span>
                    <span>最大接入<strong>${Number(active.max_users || 0)} 用户</strong></span>
                    <span>设备成本<strong>${Number(active.device_cost || 0).toFixed(2)}</strong></span>
                    <span>带宽成本<strong>${Number(active.bandwidth_cost || 0).toFixed(3)}</strong></span>
                  </div>
                </div>
              </section>
              <div class="proto-device-form-grid">
                <label class="proto-device-field">
                  <span>设备 key</span>
                  <input value="${active.name}" readonly />
                </label>
                <label class="proto-device-field">
                  <span>设备显示名称</span>
                  <input data-field="label" value="${active.label}" />
                </label>
                <label class="proto-device-field">
                  <span>设备类型</span>
                  <select data-field="device_class">
                    ${["宏基站", "微型基站", "背负式基站", "中继设备", "临时设备/车载设备"]
                      .map(
                        (item) =>
                          `<option value="${item}" ${
                            (active.device_class || resolveDeviceClass(active)) === item ? "selected" : ""
                          }>${item}</option>`
                      )
                      .join("")}
                  </select>
                </label>
                <label class="proto-device-field">
                  <span>峰值吞吐 Mbps</span>
                  <input data-field="max_throughput" type="number" min="0" step="0.1" value="${active.max_throughput}" />
                </label>
                <label class="proto-device-field">
                  <span>最大接入用户</span>
                  <input data-field="max_users" type="number" min="0" step="1" value="${active.max_users}" />
                </label>
                <label class="proto-device-field">
                  <span>设备成本</span>
                  <input data-field="device_cost" type="number" min="0" step="0.01" value="${active.device_cost}" />
                </label>
                <label class="proto-device-field">
                  <span>带宽成本</span>
                  <input data-field="bandwidth_cost" type="number" min="0" step="0.001" value="${active.bandwidth_cost}" />
                </label>
                <label class="proto-device-field proto-device-field--wide">
                  <span>支持模式，使用 / 或逗号分隔</span>
                  <input data-field="supported_modes" value="${modeText(active.supported_modes).replace("未配置", "")}" />
                </label>
              </div>
              ${renderTable(stations)}
            `
                : `<div class="proto-device-empty">请选择左侧设备后编辑参数。</div>`
            }
          </div>

          <footer class="proto-device-footer">
            <span>${state.dirty ? "存在未应用的本地修改" : "当前配置已应用到本地页面状态"}</span>
          </footer>
        </section>
      </main>
    `;
    mountControls();
  };

  const loadScenarios = async (forceDefaultSelection = false) => {
    state.loading = true;
    state.error = "";
    draw();
    try {
      const response = await fetch(`${apiBase}/scenarios`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      state.scenarios = Array.isArray(data?.scenarios) ? data.scenarios : [];
      if (!state.scenarios.length) {
        state.selectedScenarioName = "";
        state.activeStationName = "";
      } else if (
        forceDefaultSelection ||
        !state.scenarios.some((scenario) => scenario.name === state.selectedScenarioName)
      ) {
        state.selectedScenarioName = state.scenarios[0].name;
      }
      if (!getStations().some((station) => station.name === state.activeStationName)) {
        state.activeStationName = getStations()[0]?.name || "";
      }
      state.loading = false;
      draw();
    } catch (error) {
      state.loading = false;
      state.error = "设备库加载失败，请检查后端服务。";
      draw();
    }
  };

  void loadScenarios();
};

export function injectPrototypeDevice(doc) {
  if (!doc) return;

  doc.getElementById("prototype-device-workbench")?.remove();
  injectStyles(doc);
  keepPrototypePanelOpen(doc);
  removePrototypeDeviceCards(doc);

  const root = doc.createElement("section");
  root.id = "prototype-device-workbench";
  root.className = "prototype-device-workbench";
  getPrototypeDeviceListContainer(doc)?.appendChild(root);

  renderWorkbench(doc, root);
}
