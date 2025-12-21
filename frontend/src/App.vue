<template>
  <div class="dashboard">
    <header class="dashboard__header">
      <div>
        <h1>RescueNet-RL 仪表盘</h1>
        <p>选择灾害场景进行训练，并在自定义环境中测试组网策略</p>
      </div>
      <div class="header__links">
        <a href="https://github.com/" target="_blank" rel="noreferrer">Repo</a>
      </div>
    </header>
    <main class="dashboard__grid">
      <section class="panel">
        <ScenarioTrainingPanel />
      </section>
      <section class="panel">
        <CustomEnvironmentTester />
      </section>
    </main>
  </div>
</template>

<script setup>
import ScenarioTrainingPanel from "./components/ScenarioTrainingPanel.vue";
import CustomEnvironmentTester from "./components/CustomEnvironmentTester.vue";
</script>

<style scoped>
.dashboard {
  display: flex;
  flex-direction: column;
  gap: 24px;
}

.dashboard__header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  border: 1px solid rgba(226, 232, 240, 0.1);
  padding: 20px;
  border-radius: 12px;
  background: rgba(15, 23, 42, 0.6);
  box-shadow: 0 10px 40px rgba(8, 112, 184, 0.1);
}

.dashboard__grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
  gap: 24px;
}

.panel {
  border-radius: 16px;
  background: rgba(15, 23, 42, 0.65);
  border: 1px solid rgba(148, 163, 184, 0.25);
  padding: 24px;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05);
}

h1 {
  margin: 0;
  font-size: 28px;
}

p {
  margin: 4px 0 0;
  color: #94a3b8;
}

.header__links a {
  text-decoration: none;
  border: 1px solid rgba(147, 197, 253, 0.5);
  padding: 8px 12px;
  border-radius: 999px;
  transition: background 0.2s ease;
}

.header__links a:hover {
  background: rgba(56, 189, 248, 0.15);
}
</style>
