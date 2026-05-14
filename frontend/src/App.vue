<template>
  <div class="proto-app">
    <div class="proto-stage-shell">
      <div class="proto-stage-wrap" :style="stageWrapStyle">
        <div class="proto-stage" :style="stageTransformStyle">
          <template v-if="currentView.withMenu">
            <div class="proto-menu-viewport">
              <iframe
                ref="menuFrameRef"
                class="proto-menu-frame"
                :src="prototypeUrl('菜单.html')"
                title="原型顶部菜单"
                tabindex="-1"
                scrolling="no"
                @load="handleFrameLoad"
              ></iframe>
              <div class="proto-menu-hotspots">
                <a
                  v-for="item in menuHotspots"
                  :key="item.key"
                  :href="item.href"
                  class="proto-menu-hotspot"
                  :style="hotspotStyle(item)"
                  :aria-label="item.label"
                  @mouseenter="setMenuHover(item.key)"
                  @mouseleave="clearMenuHover"
                >
                  <span class="proto-menu-hotspot__label">{{ item.label }}</span>
                </a>
              </div>
            </div>

            <div class="proto-content-viewport" :style="{ top: `${currentView.contentTop}px` }">
              <template v-if="!currentView.useLiveContent">
                <iframe
                  ref="contentFrameRef"
                  :class="[
                    'proto-content-frame',
                    {
                      'proto-content-frame--live': currentView.enableApiInjection,
                      'proto-content-frame--pending': currentView.enableApiInjection && !contentFrameReady,
                    }
                  ]"
                  :src="prototypeUrl(currentView.prototypePage)"
                  :title="currentView.title"
                  tabindex="-1"
                  :scrolling="currentView.allowContentScroll ? 'auto' : 'no'"
                  @load="handleContentFrameLoad"
                ></iframe>
                <div
                  v-if="currentView.enableApiInjection && !contentFrameReady"
                  class="proto-content-loading"
                >
                  <span>{{ currentRoute === "train" ? "正在加载真实训练界面" : "正在加载真实联调界面" }}</span>
                </div>
              </template>
              <div v-else class="proto-live-content">
                <PrototypeTrainingPage v-if="currentRoute === 'train'" />
                <Ns3ReplayPanel v-else-if="currentRoute === 'replay'" />
              </div>
            </div>
          </template>

          <template v-else>
            <iframe
              class="proto-page-frame"
              :src="prototypeUrl(currentView.prototypePage)"
              :title="currentView.title"
              tabindex="-1"
              scrolling="no"
              @load="handleFrameLoad"
            ></iframe>
          </template>

          <button
            v-for="spot in currentHotspots"
            :key="spot.key"
            type="button"
            class="proto-hotspot proto-hotspot--button"
            :style="hotspotStyle(spot)"
            @click="handleHotspot(spot)"
          ></button>
        </div>
      </div>
    </div>

    <transition name="drawer-fade">
      <div v-if="drawerOpen" class="live-backdrop" @click.self="closeDrawer">
        <aside class="live-drawer">
          <header class="live-drawer__header">
            <div>
              <span class="live-drawer__eyebrow">Live API Panel</span>
              <h2>{{ drawerTitle }}</h2>
              <p>{{ drawerDescription }}</p>
            </div>
            <button type="button" @click="closeDrawer">关闭</button>
          </header>

          <div class="live-drawer__body">
            <ScenarioTrainingPanel v-if="drawerType === 'train'" />
            <CustomEnvironmentTester v-else-if="drawerType === 'tester'" />
            <ReplayWorkbench v-else-if="drawerType === 'replay'" />
            <LinkSimulationPage v-else-if="drawerType === 'link'" />
            <HomeOverview v-else-if="drawerType === 'home'" />
          </div>
        </aside>
      </div>
    </transition>
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from "vue";

import CustomEnvironmentTester from "./components/CustomEnvironmentTester.vue";
import HomeOverview from "./components/HomeOverview.vue";
import LinkSimulationPage from "./components/LinkSimulationPage.vue";
import Ns3ReplayPanel from "./components/Ns3ReplayPanel.vue";
import PrototypeTrainingPage from "./components/PrototypeTrainingPage.vue";
import ReplayWorkbench from "./components/ReplayWorkbench.vue";
import ScenarioTrainingPanel from "./components/ScenarioTrainingPanel.vue";
import { injectPrototypeDevice } from "./utils/prototypeDeviceInjection";
import { injectPrototypeLink } from "./utils/prototypeLinkInjection";
import { injectPrototypeTester } from "./utils/prototypeTesterInjection";
import { injectPrototypeTraining } from "./utils/prototypeTrainingInjection";

const menuFrameRef = ref(null);
const contentFrameRef = ref(null);

const views = {
  home: {
    title: "首页",
    prototypePage: "首页.html",
    withMenu: true,
    stageWidth: 1920,
    stageHeight: 1080,
    contentTop: 70,
    drawerType: "home",
    drawerTitle: "首页真实总览",
    drawerDescription: "首页原型保持不动，真实场景总览通过侧边实时面板提供。",
  },
  train: {
    title: "模型训练",
    prototypePage: "模型训练.html",
    withMenu: true,
    useLiveContent: false,
    enableApiInjection: true,
    stageWidth: 1920,
    stageHeight: 1080,
    contentTop: 70,
    drawerType: null,
    drawerTitle: "",
    drawerDescription: "",
  },
  tester: {
    title: "策略测试",
    prototypePage: "策略测试.html",
    withMenu: true,
    useLiveContent: false,
    enableApiInjection: true,
    stageWidth: 1920,
    stageHeight: 1080,
    contentTop: 70,
    drawerType: null,
    drawerTitle: "",
    drawerDescription: "",
  },
  replay: {
    title: "场景回放",
    prototypePage: "场景回放.html",
    withMenu: true,
    useLiveContent: true,
    stageWidth: 1920,
    stageHeight: 1080,
    contentTop: 70,
    drawerType: "replay",
    drawerTitle: "场景回放真实联调面板",
    drawerDescription: "此面板直接消费训练或测试生成的真实回放数据。",
  },
  link: {
    title: "链路仿真",
    prototypePage: "链路仿真.html",
    withMenu: true,
    useLiveContent: false,
    enableApiInjection: true,
    stageWidth: 1920,
    stageHeight: 1080,
    contentTop: 70,
    drawerType: null,
    drawerTitle: "",
    drawerDescription: "",
  },
  device: {
    title: "设备管理",
    prototypePage: "设备管理.html",
    withMenu: true,
    useLiveContent: false,
    enableApiInjection: true,
    stageWidth: 1920,
    stageHeight: 1080,
    contentTop: 70,
    drawerType: null,
    drawerTitle: "",
    drawerDescription: "",
  },
  login: {
    title: "登录",
    prototypePage: "登录.html",
    withMenu: false,
    useLiveContent: false,
    stageWidth: 1920,
    stageHeight: 1010,
    contentTop: 0,
    drawerType: null,
    drawerTitle: "",
    drawerDescription: "",
  },
};

const menuHotspots = [
  { key: "home", label: "首页", href: "#/home", x: 113, y: 18, width: 150, height: 40 },
  { key: "train", label: "模型训练", href: "#/train", x: 297, y: 18, width: 150, height: 40 },
  { key: "tester", label: "策略测试", href: "#/tester", x: 482, y: 18, width: 150, height: 40 },
  { key: "replay", label: "场景回放", href: "#/replay", x: 1290, y: 18, width: 150, height: 40 },
  { key: "link", label: "链路仿真", href: "#/link", x: 1464, y: 18, width: 150, height: 40 },
  { key: "device", label: "设备管理", href: "#/device", x: 1636, y: 18, width: 150, height: 40 },
];

const menuVisualItems = [
  { key: "home", nodeId: "u63", imageId: "u63_img", defaultFile: "u63.png", activeFile: "u63_mouseOver.png" },
  { key: "train", nodeId: "u65", imageId: "u65_img", defaultFile: "u63.png", activeFile: "u63_mouseOver.png" },
  { key: "tester", nodeId: "u67", imageId: "u67_img", defaultFile: "u63.png", activeFile: "u63_mouseOver.png" },
  { key: "replay", nodeId: "u59", imageId: "u59_img", defaultFile: "u59.png", activeFile: "u59_mouseOver.png" },
  { key: "link", nodeId: "u60", imageId: "u60_img", defaultFile: "u59.png", activeFile: "u59_mouseOver.png" },
  { key: "device", nodeId: "u61", imageId: "u61_img", defaultFile: "u59.png", activeFile: "u59_mouseOver.png" },
];

const pageHotspots = {
  home: [
    { key: "open-home-live", action: "drawer", x: 1239, y: 141, width: 669, height: 100 },
    { key: "go-train", action: "route", route: "train", x: 1302, y: 143, width: 580, height: 88 },
  ],
  login: [
    { key: "login-enter", action: "route", route: "train", x: 821, y: 712, width: 278, height: 56 },
  ],
  train: [
    { key: "train-open-drawer", action: "drawer", x: 1640, y: 24, width: 180, height: 50 },
  ],
  tester: [],
  replay: [
    { key: "replay-open", action: "drawer", x: 293, y: 128, width: 150, height: 40 },
  ],
  link: [],
  device: [],
};

const currentRoute = ref("home");
const drawerOpen = ref(false);
const drawerType = ref(null);
const hoveredMenuKey = ref(null);
const contentFrameReady = ref(false);
const viewportWidth = ref(typeof window === "undefined" ? 1440 : window.innerWidth);
const viewportHeight = ref(typeof window === "undefined" ? 900 : window.innerHeight);

const normalizeRoute = (hash) => {
  const raw = String(hash || "").replace(/^#\/?/, "").trim().toLowerCase();
  if (!raw) return "home";
  if (["home", "index"].includes(raw)) return "home";
  if (["train", "training", "algorithm"].includes(raw)) return "train";
  if (["tester", "test", "strategy"].includes(raw)) return "tester";
  if (["replay", "scene", "scene-replay"].includes(raw)) return "replay";
  if (["link", "network", "mahimahi"].includes(raw)) return "link";
  if (["device", "devices"].includes(raw)) return "device";
  if (["login", "signin"].includes(raw)) return "login";
  return "home";
};

const currentView = computed(() => views[currentRoute.value] || views.home);
const currentHotspots = computed(() => pageHotspots[currentRoute.value] || []);

const stageScaleX = computed(() => viewportWidth.value / currentView.value.stageWidth);
const stageScaleY = computed(() => viewportHeight.value / currentView.value.stageHeight);

const stageTransformStyle = computed(() => ({
  width: `${currentView.value.stageWidth}px`,
  height: `${currentView.value.stageHeight}px`,
  transform: `scale(${stageScaleX.value}, ${stageScaleY.value})`,
  transformOrigin: "top left",
}));

const stageWrapStyle = computed(() => ({
  width: `${viewportWidth.value}px`,
  height: `${viewportHeight.value}px`,
}));

const drawerTitle = computed(() => currentView.value.drawerTitle || "");
const drawerDescription = computed(() => currentView.value.drawerDescription || "");

const prototypeUrl = (page) => `${import.meta.env.BASE_URL}prototype/${page}`;
const prototypeMenuImageUrl = (file) => prototypeUrl(`images/菜单/${file}`);

const hotspotStyle = (spot) => ({
  left: `${spot.x}px`,
  top: `${spot.y}px`,
  width: `${spot.width}px`,
  height: `${spot.height}px`,
});

const syncRoute = () => {
  const nextRoute = normalizeRoute(window.location.hash);
  currentRoute.value = nextRoute;
  contentFrameReady.value = !views[nextRoute]?.enableApiInjection;
  drawerOpen.value = false;
  drawerType.value = null;
  syncMenuHighlight();
};

const updateViewport = () => {
  viewportWidth.value = window.innerWidth;
  viewportHeight.value = window.innerHeight;
};

const openDrawer = (type) => {
  if (!type) return;
  drawerType.value = type;
  drawerOpen.value = true;
};

const closeDrawer = () => {
  drawerOpen.value = false;
};

const setMenuHover = (key) => {
  hoveredMenuKey.value = key;
  syncMenuHighlight();
};

const clearMenuHover = () => {
  hoveredMenuKey.value = null;
  syncMenuHighlight();
};

const navigateTo = (route) => {
  window.location.hash = `/${route}`;
};

const handleHotspot = (spot) => {
  if (spot.action === "route" && spot.route) {
    navigateTo(spot.route);
    return;
  }
  if (spot.action === "drawer") {
    openDrawer(currentView.value.drawerType);
  }
};

const syncMenuHighlight = () => {
  const doc = menuFrameRef.value?.contentDocument;
  if (!doc) return;

  for (const item of menuVisualItems) {
    const node = doc.getElementById(item.nodeId);
    const img = doc.getElementById(item.imageId);
    const highlighted = currentRoute.value === item.key || hoveredMenuKey.value === item.key;

    node?.classList.toggle("selected", currentRoute.value === item.key);
    node?.classList.toggle("mouseOver", hoveredMenuKey.value === item.key);
    img?.classList.toggle("selected", currentRoute.value === item.key);
    img?.classList.toggle("mouseOver", hoveredMenuKey.value === item.key);

    if (img) {
      img.src = prototypeMenuImageUrl(highlighted ? item.activeFile : item.defaultFile);
    }
  }
};

const handleFrameLoad = (event) => {
  const frame = event?.target;
  const doc = frame?.contentDocument;
  if (!doc) return;

  const html = doc.documentElement;
  const body = doc.body;

  if (html) {
    html.style.overflow = "hidden";
    html.style.width = "100%";
    html.style.height = "100%";
    html.style.margin = "0";
    html.style.padding = "0";
    html.style.scrollbarWidth = "none";
  }

  if (body) {
    body.style.overflow = "hidden";
    body.style.width = "100%";
    body.style.height = "100%";
    body.style.margin = "0";
    body.style.padding = "0";
    body.style.scrollbarWidth = "none";
  }

  if (!doc.getElementById("proto-scroll-lock-style")) {
    const style = doc.createElement("style");
    style.id = "proto-scroll-lock-style";
    style.textContent = `
      html, body {
        overflow: hidden !important;
        scrollbar-width: none !important;
        -ms-overflow-style: none !important;
      }
      html::-webkit-scrollbar,
      body::-webkit-scrollbar {
        display: none !important;
        width: 0 !important;
        height: 0 !important;
      }
    `;
    doc.head?.appendChild(style);
  }

  if (frame === menuFrameRef.value) {
    syncMenuHighlight();
  }
};

const handleContentFrameLoad = (event) => {
  const frame = event?.target;
  const doc = frame?.contentDocument;
  if (!doc) return;
  const routeAtLoad = currentRoute.value;
  const allowContentScroll = Boolean(views[routeAtLoad]?.allowContentScroll);
  const revealInjectedFrame = () => {
    if (frame === contentFrameRef.value && currentRoute.value === routeAtLoad) {
      contentFrameReady.value = true;
    }
  };
  const injectNow = (handler) => {
    handler(doc);
    requestAnimationFrame(() => requestAnimationFrame(revealInjectedFrame));
    setTimeout(() => handler(doc), 120);
  };

  const html = doc.documentElement;
  const body = doc.body;
  if (html) {
    html.style.width = "100%";
    html.style.height = allowContentScroll ? "auto" : "100%";
    html.style.margin = "0";
    html.style.padding = "0";
    html.style.overflowX = "hidden";
    html.style.overflowY = allowContentScroll ? "auto" : "hidden";
  }
  if (body) {
    body.style.width = "100%";
    body.style.height = allowContentScroll ? "auto" : "100%";
    body.style.margin = "0";
    body.style.padding = "0";
    body.style.overflowX = "hidden";
    body.style.overflowY = allowContentScroll ? "auto" : "hidden";
  }

  if (allowContentScroll) {
    doc.getElementById("proto-scroll-lock-style")?.remove();
    if (!doc.getElementById("proto-scroll-unlock-style")) {
      const style = doc.createElement("style");
      style.id = "proto-scroll-unlock-style";
      style.textContent = `
        html, body {
          overflow-x: hidden !important;
          overflow-y: auto !important;
          scrollbar-width: auto !important;
          -ms-overflow-style: auto !important;
        }
      `;
      doc.head?.appendChild(style);
    }
  } else if (!doc.getElementById("proto-scroll-lock-style")) {
    const style = doc.createElement("style");
    style.id = "proto-scroll-lock-style";
    style.textContent = `
      html, body {
        overflow: hidden !important;
        scrollbar-width: none !important;
        -ms-overflow-style: none !important;
      }
      html::-webkit-scrollbar,
      body::-webkit-scrollbar {
        display: none !important;
        width: 0 !important;
        height: 0 !important;
      }
    `;
    doc.head?.appendChild(style);
  }

  if (currentRoute.value === "train") {
    injectNow(injectPrototypeTraining);
    return;
  }

  if (currentRoute.value === "tester") {
    injectNow(injectPrototypeTester);
    return;
  }

  if (currentRoute.value === "link") {
    injectNow(injectPrototypeLink);
    return;
  }

  if (currentRoute.value === "device") {
    injectNow(injectPrototypeDevice);
    return;
  }

  contentFrameReady.value = true;
};

onMounted(() => {
  syncRoute();
  updateViewport();
  window.addEventListener("hashchange", syncRoute);
  window.addEventListener("resize", updateViewport);
});

onBeforeUnmount(() => {
  window.removeEventListener("hashchange", syncRoute);
  window.removeEventListener("resize", updateViewport);
});
</script>

<style>
.proto-app {
  width: 100vw;
  height: 100vh;
  overflow: hidden;
  background: #020817;
}

.proto-stage-shell {
  position: relative;
  width: 100%;
  height: 100%;
  overflow: hidden;
}

.proto-stage-wrap {
  position: relative;
  overflow: hidden;
}

.proto-stage {
  position: relative;
  overflow: hidden;
}

.proto-menu-viewport {
  position: absolute;
  left: 0;
  top: 0;
  width: 1920px;
  height: 70px;
}

.proto-menu-frame,
.proto-page-frame {
  width: 100%;
  height: 100%;
  border: 0;
  display: block;
}

.proto-content-viewport {
  position: absolute;
  left: 0;
  width: 1920px;
  height: calc(100% - 70px);
}

.proto-content-frame,
.proto-live-content {
  width: 100%;
  height: 100%;
  border: 0;
  display: block;
}

.proto-content-frame--pending {
  opacity: 0;
  pointer-events: none;
}

.proto-content-loading {
  position: absolute;
  inset: 0;
  z-index: 5;
  display: grid;
  place-items: center;
  background: #eef5ff;
  color: #17315d;
  font-family: "Microsoft YaHei", "PingFang SC", sans-serif;
  font-size: 18px;
}

.proto-content-loading span {
  padding: 10px 18px;
  border: 1px solid rgba(57, 97, 246, 0.2);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.78);
}

.proto-menu-hotspots {
  position: absolute;
  inset: 0;
  pointer-events: none;
}

.proto-menu-hotspot,
.proto-hotspot--button {
  position: absolute;
  display: block;
  background: transparent;
  border: 0;
  padding: 0;
  cursor: pointer;
  pointer-events: auto;
}

.proto-menu-hotspot__label {
  position: absolute;
  width: 1px;
  height: 1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
}

.proto-page-frame {
  position: absolute;
  left: 0;
  top: 0;
}

.proto-live-content {
  overflow: auto;
  background: #eef5ff;
}

.live-backdrop {
  position: fixed;
  inset: 0;
  background: rgba(15, 23, 42, 0.48);
  display: flex;
  justify-content: flex-end;
  z-index: 50;
}

.live-drawer {
  width: min(1080px, 88vw);
  height: 100vh;
  background: #f8fbff;
  box-shadow: -24px 0 48px rgba(15, 23, 42, 0.24);
  display: flex;
  flex-direction: column;
}

.live-drawer__header {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: flex-start;
  padding: 20px 24px;
  border-bottom: 1px solid rgba(148, 163, 184, 0.18);
  background: linear-gradient(180deg, #ffffff, #f3f8ff);
}

.live-drawer__eyebrow {
  display: inline-flex;
  margin-bottom: 8px;
  font-size: 12px;
  letter-spacing: 0.08em;
  color: #2563eb;
}

.live-drawer__header h2 {
  margin: 0;
  color: #17315d;
}

.live-drawer__header p {
  margin: 8px 0 0;
  color: #6881a7;
}

.live-drawer__header button {
  border: 0;
  border-radius: 10px;
  padding: 10px 14px;
  background: rgba(37, 99, 235, 0.1);
  color: #2563eb;
  cursor: pointer;
}

.live-drawer__body {
  flex: 1;
  min-height: 0;
  overflow: auto;
  padding: 20px 24px 24px;
}

.drawer-fade-enter-active,
.drawer-fade-leave-active {
  transition: opacity 0.2s ease;
}

.drawer-fade-enter-from,
.drawer-fade-leave-to {
  opacity: 0;
}
</style>
