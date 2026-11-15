import type { NextConfig } from "next";

// The repo-root .env is loaded at runtime by lib/env.ts (server only).
const nextConfig: NextConfig = {
  images: { unoptimized: true },
};

export default nextConfig;
