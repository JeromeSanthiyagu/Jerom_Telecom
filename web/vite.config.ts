/// <reference types="vitest/config" />
import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

// Built files are served by FastAPI at /ui (see app/main.py). In development the Vite server
// proxies API calls to the backend, so the browser always talks to a single origin.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, ".", "");
  const target = env.API_TARGET || "http://localhost:8000";
  return {
    base: "/ui/",
    plugins: [react()],
    server: {
      port: Number(env.UI_PORT || 5173),
      proxy: { "/api": target, "/health": target },
    },
    test: { environment: "node" },
  };
});
