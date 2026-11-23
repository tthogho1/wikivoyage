import type { NextConfig } from "next";

// The repo-root .env is loaded at runtime by lib/env.ts (server only).
const nextConfig: NextConfig = {
  images: { unoptimized: true },
  devIndicators: false,
  // Enables the minimal, self-contained `.next/standalone` output used by
  // infra/ecs/terraform's Docker image (see webapp/Dockerfile).
  output: "standalone",
};

export default nextConfig;
