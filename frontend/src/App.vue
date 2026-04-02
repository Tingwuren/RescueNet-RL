<template>
  <div class="app-shell">
    <div class="app-shell__glow app-shell__glow--left"></div>
    <div class="app-shell__glow app-shell__glow--right"></div>

    <header class="topbar">
      <a class="brand" href="#/">
        <span class="brand__eyebrow">应急广播与通信资源智能决策</span>
        <strong>智能应急广播与通信资源配置与组网模块</strong>
      </a>
      <nav class="topbar__nav" aria-label="主导航">
        <a
          v-for="item in navItems"
          :key="item.key"
          :href="item.href"
          :class="['nav-link', { 'nav-link--active': currentRoute === item.key }]"
        >
          {{ item.label }}
        </a>
      </nav>
    </header>

    <main class="app-main">
      <section v-if="currentRoute === 'home'" class="home-view">
        <div class="hero">
          <div class="hero__content panel-surface panel-surface--hero">
            <span class="eyebrow">Disaster Recovery RL</span>
            <h1>面向灾后通信恢复的强化学习实验台</h1>
            <p class="hero__summary">
              围绕灾害场景建模、组网策略训练与残余设施测试，比较不同强化学习算法在覆盖率、带宽利用和部署成本之间的权衡表现。
            </p>
            <div class="hero__actions">
              <a class="primary-link" href="#/training">进入训练中心</a>
              <a class="secondary-link" href="#/testing">进入测试中心</a>
            </div>
          </div>

          <div class="hero__metrics">
            <article v-for="item in metrics" :key="item.label" class="metric-card panel-surface">
              <span>{{ item.label }}</span>
              <strong>{{ item.value }}</strong>
              <p>{{ item.description }}</p>
            </article>
          </div>
        </div>

        <div class="feature-grid">
          <article v-for="card in homeCards" :key="card.title" class="feature-card panel-surface">
            <span class="feature-card__tag">{{ card.tag }}</span>
            <h2>{{ card.title }}</h2>
            <p>{{ card.description }}</p>
            <a class="feature-card__link" :href="card.href">{{ card.cta }}</a>
          </article>
        </div>
      </section>

      <section v-else-if="currentRoute === 'training'" class="workspace-view">
        <div class="workspace-hero">
          <div>
            <span class="eyebrow">Training Center</span>
            <h1>灾害场景训练中心</h1>
            <p>
              选择场景、奖励函数与算法组合，启动训练后在同一界面实时观察事件流和状态变化。
            </p>
          </div>
          <div class="workspace-meta">
            <span>场景驱动配置</span>
            <span>实时事件流监控</span>
            <span>多算法实验对比</span>
          </div>
        </div>

        <div class="workspace-layout">
          <aside class="workspace-aside panel-surface">
            <h2>训练流程</h2>
            <ol>
              <li>选择灾害场景并确认区域规模。</li>
              <li>挑选奖励函数策略，明确优化目标。</li>
              <li>指定算法与训练步数，启动训练。</li>
              <li>通过事件流观察状态、结果与异常。</li>
            </ol>
          </aside>
          <section class="workspace-content panel-surface">
            <ScenarioTrainingPanel />
          </section>
        </div>
      </section>

      <section v-else class="workspace-view">
        <div class="workspace-hero">
          <div>
            <span class="eyebrow">Testing Center</span>
            <h1>自定义环境测试中心</h1>
            <p>
              配置残余基站和 checkpoint，复现基础设施受损后的网络恢复条件，独立评估策略表现。
            </p>
          </div>
          <div class="workspace-meta">
            <span>残余基站配置</span>
            <span>Checkpoint 快速切换</span>
            <span>结果与场景导出</span>
          </div>
        </div>

        <div class="workspace-layout">
          <aside class="workspace-aside panel-surface">
            <h2>测试流程</h2>
            <ol>
              <li>选择待验证的灾害场景和算法。</li>
              <li>指定 checkpoint 路径，按需添加残余基站。</li>
              <li>执行单轮模拟，查看覆盖率与奖励输出。</li>
              <li>导出受灾前后场景 JSON 进行复盘。</li>
            </ol>
          </aside>
          <section class="workspace-content panel-surface">
            <CustomEnvironmentTester />
          </section>
        </div>
      </section>
    </main>
  </div>
</template>

<script setup>
import { onBeforeUnmount, onMounted, ref } from "vue";
import ScenarioTrainingPanel from "./components/ScenarioTrainingPanel.vue";
import CustomEnvironmentTester from "./components/CustomEnvironmentTester.vue";

const navItems = [
  { key: "home", label: "首页", href: "#/" },
  { key: "training", label: "训练", href: "#/training" },
  { key: "testing", label: "测试", href: "#/testing" },
];

const metrics = [
  { label: "Algorithms", value: "4 Models", description: "支持 PPO、DQN、A3C、MPPO 的统一训练与测试。" },
  { label: "Scenarios", value: "Scenario-Driven", description: "场景配置覆盖灾害类型、区域网格、候选站点与奖励函数。" },
  { label: "Outputs", value: "Reports", description: "训练侧提供实时事件流，测试侧输出评估报告与场景导出。" },
];

const homeCards = [
  {
    tag: "Train",
    title: "训练工作台",
    description: "按灾害场景配置奖励函数、算法与训练步数，持续观察策略收敛过程和状态事件流。",
    href: "#/training",
    cta: "打开训练页",
  },
  {
    tag: "Test",
    title: "测试工作台",
    description: "配置残余基站和模型 checkpoint，验证灾后恢复策略在指定条件下的覆盖效果。",
    href: "#/testing",
    cta: "打开测试页",
  },
  {
    tag: "Scenario",
    title: "场景与指标",
    description: "围绕用户覆盖、吞吐、广播能力、设备成本和带宽成本等核心指标评估策略效果。",
    href: "#/training",
    cta: "查看训练配置",
  },
];

const normalizeRoute = (hash) => {
  const route = hash.replace(/^#\/?/, "").replace(/\/+$/, "").trim();
  if (!route) return "home";
  return navItems.some((item) => item.key === route) ? route : "home";
};

const currentRoute = ref(normalizeRoute(window.location.hash));

const syncRoute = () => {
  currentRoute.value = normalizeRoute(window.location.hash);
};

onMounted(() => {
  syncRoute();
  window.addEventListener("hashchange", syncRoute);
});

onBeforeUnmount(() => {
  window.removeEventListener("hashchange", syncRoute);
});
</script>

<style scoped>
.app-shell {
  position: relative;
  max-width: 1440px;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  gap: 28px;
  isolation: isolate;
}

.app-shell__glow {
  position: fixed;
  width: 30rem;
  height: 30rem;
  border-radius: 999px;
  pointer-events: none;
  filter: blur(70px);
  opacity: 0.28;
  z-index: -1;
}

.app-shell__glow--left {
  top: -8rem;
  left: -10rem;
  background: rgba(14, 165, 233, 0.55);
}

.app-shell__glow--right {
  top: 8rem;
  right: -12rem;
  background: rgba(251, 191, 36, 0.28);
}

.topbar {
  position: sticky;
  top: 24px;
  z-index: 20;
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 20px;
  padding: 18px 22px;
  border-radius: 24px;
  border: 1px solid rgba(125, 211, 252, 0.18);
  background: rgba(7, 15, 31, 0.72);
  backdrop-filter: blur(18px);
  box-shadow: 0 18px 60px rgba(15, 23, 42, 0.28);
}

.brand {
  display: inline-flex;
  flex-direction: column;
  gap: 4px;
  color: inherit;
  text-decoration: none;
  max-width: 34rem;
}

.brand strong {
  font-size: clamp(1.05rem, 2vw, 1.45rem);
  line-height: 1.25;
  letter-spacing: 0.04em;
}

.brand__eyebrow,
.eyebrow {
  font-size: 11px;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: #7dd3fc;
}

.topbar__nav {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.nav-link {
  padding: 10px 16px;
  border-radius: 999px;
  border: 1px solid transparent;
  color: #cbd5e1;
  text-decoration: none;
  transition: all 0.2s ease;
}

.nav-link:hover {
  color: #f8fafc;
  border-color: rgba(125, 211, 252, 0.24);
  background: rgba(14, 165, 233, 0.1);
}

.nav-link--active {
  color: #f8fafc;
  border-color: rgba(56, 189, 248, 0.35);
  background: linear-gradient(135deg, rgba(14, 165, 233, 0.24), rgba(37, 99, 235, 0.16));
}

.app-main {
  display: flex;
  flex-direction: column;
  gap: 24px;
}

.home-view,
.workspace-view {
  display: flex;
  flex-direction: column;
  gap: 24px;
}

.hero {
  display: grid;
  grid-template-columns: minmax(0, 1.3fr) minmax(300px, 0.9fr);
  gap: 24px;
}

.panel-surface {
  border-radius: 28px;
  border: 1px solid rgba(148, 163, 184, 0.18);
  background:
    linear-gradient(180deg, rgba(15, 23, 42, 0.92), rgba(15, 23, 42, 0.78)),
    radial-gradient(circle at top right, rgba(14, 165, 233, 0.2), transparent 38%);
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.06),
    0 18px 50px rgba(2, 6, 23, 0.3);
}

.panel-surface--hero {
  padding: 36px;
  min-height: 360px;
  justify-content: space-between;
}

.hero__content {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.hero__content h1,
.workspace-hero h1 {
  margin: 0;
  font-size: clamp(2.2rem, 5vw, 4.2rem);
  line-height: 0.95;
}

.hero__summary,
.workspace-hero p {
  margin: 0;
  max-width: 56rem;
  font-size: 1rem;
  line-height: 1.75;
  color: #bfd0e5;
}

.hero__actions {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
}

.primary-link,
.secondary-link,
.feature-card__link {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-height: 46px;
  padding: 0 18px;
  border-radius: 999px;
  text-decoration: none;
  transition: transform 0.2s ease, border-color 0.2s ease, background 0.2s ease;
}

.primary-link {
  color: #04111f;
  background: linear-gradient(135deg, #7dd3fc, #facc15);
}

.secondary-link,
.feature-card__link {
  color: #f8fafc;
  border: 1px solid rgba(148, 163, 184, 0.28);
  background: rgba(15, 23, 42, 0.36);
}

.primary-link:hover,
.secondary-link:hover,
.feature-card__link:hover {
  transform: translateY(-1px);
}

.hero__metrics {
  display: grid;
  gap: 16px;
}

.metric-card {
  padding: 22px 24px;
}

.metric-card span {
  display: inline-block;
  margin-bottom: 12px;
  font-size: 12px;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: #94a3b8;
}

.metric-card strong {
  display: block;
  font-size: 1.9rem;
  color: #f8fafc;
}

.metric-card p,
.feature-card p,
.workspace-aside li {
  color: #bfd0e5;
  line-height: 1.65;
}

.feature-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 20px;
}

.feature-card {
  padding: 28px;
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.feature-card h2,
.workspace-aside h2 {
  margin: 0;
  font-size: 1.25rem;
}

.feature-card__tag {
  display: inline-flex;
  width: fit-content;
  padding: 6px 10px;
  border-radius: 999px;
  background: rgba(125, 211, 252, 0.12);
  color: #7dd3fc;
  font-size: 12px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.workspace-hero {
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  gap: 20px;
}

.workspace-meta {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 10px;
}

.workspace-meta span {
  padding: 9px 12px;
  border-radius: 999px;
  background: rgba(15, 23, 42, 0.55);
  border: 1px solid rgba(148, 163, 184, 0.18);
  color: #dbeafe;
  font-size: 13px;
}

.workspace-layout {
  display: grid;
  grid-template-columns: minmax(260px, 320px) minmax(0, 1fr);
  gap: 24px;
  align-items: start;
}

.workspace-aside {
  position: sticky;
  top: 120px;
  padding: 24px;
}

.workspace-aside ol {
  margin: 18px 0 0;
  padding-left: 20px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.workspace-content {
  padding: 28px;
}

@media (max-width: 1100px) {
  .hero,
  .workspace-layout {
    grid-template-columns: 1fr;
  }

  .feature-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .workspace-aside {
    position: static;
  }
}

@media (max-width: 720px) {
  .topbar,
  .workspace-hero {
    flex-direction: column;
    align-items: stretch;
  }

  .topbar {
    top: 12px;
    border-radius: 20px;
    padding: 16px;
  }

  .feature-grid {
    grid-template-columns: 1fr;
  }

  .panel-surface--hero,
  .workspace-content,
  .workspace-aside,
  .feature-card {
    padding: 20px;
  }

  .hero__content h1,
  .workspace-hero h1 {
    line-height: 1.02;
  }
}
</style>
