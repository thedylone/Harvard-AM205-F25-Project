import { defineConfig } from "vite";

export default defineConfig({
  // Served from https://projects.thedylone.com/Harvard-AM205-F25-Project/
  base: "/Harvard-AM205-F25-Project/",
  worker: { format: "es" },
  build: { target: "es2022" },
});
