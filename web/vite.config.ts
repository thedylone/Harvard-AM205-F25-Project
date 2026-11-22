import { defineConfig } from "vite";

export default defineConfig({
  // Served from https://projects.thedylone.com/image-compression/
  base: "/image-compression/",
  worker: { format: "es" },
  build: { target: "es2022" },
});
