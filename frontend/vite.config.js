import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5173,
    host: "0.0.0.0",
    allowedHosts: ["nomanda-envoy.me"],
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/ns3-native": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
  base: "./",
});
