import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";
import fs from "node:fs";

const apiProxyTarget = process.env.VITE_PROXY_TARGET || "http://127.0.0.1:8000";

const copyPrototypeAssets = () => ({
  name: "copy-prototype-assets",
  apply: "build",
  closeBundle() {
    const source = new URL("./public/prototype", import.meta.url);
    const target = new URL("./dist/prototype", import.meta.url);
    if (!fs.existsSync(source)) return;
    fs.rmSync(target, { recursive: true, force: true });
    fs.cpSync(source, target, { recursive: true });
  },
});

export default defineConfig({
  plugins: [vue(), copyPrototypeAssets()],
  server: {
    port: 5173,
    host: "0.0.0.0",
    allowedHosts: ["nomanda-envoy.me"],
    proxy: {
      "/api": {
        target: apiProxyTarget,
        changeOrigin: true,
      },
      "/ns3-native": {
        target: apiProxyTarget,
        changeOrigin: true,
      },
    },
  },
  build: {
    copyPublicDir: false,
  },
  base: "./",
});
