import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// BASE=/prometheus/ for GitHub Pages; "/" locally
export default defineConfig({
  base: process.env.BASE ?? "/",
  plugins: [react()],
  test: { environment: "node" },
} as never);
