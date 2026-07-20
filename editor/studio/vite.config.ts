/// <reference types="vitest/config" />
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "vite";

// Tauri expects a fixed dev port (see src-tauri/tauri.conf.json build.devUrl).
export default defineConfig({
  plugins: [react(), tailwindcss()],
  clearScreen: false,
  server: {
    port: 1440,
    strictPort: true,
  },
  build: {
    target: "chrome110",
    outDir: "dist",
  },
  test: {
    // Node by default; DOM-dependent tests opt in per-file with
    // `// @vitest-environment jsdom`.
    environment: "node",
  },
});
