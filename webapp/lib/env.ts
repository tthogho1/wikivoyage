import path from "node:path";
import { loadEnvConfig } from "@next/env";

// Server only. Share the repo-root .env (OPENAI_API_KEY, ...) with the Python side. Values already in
// the environment or in webapp/.env.local win; next.config.ts alone does not reach the route runtime.
loadEnvConfig(path.resolve(process.cwd(), ".."), process.env.NODE_ENV !== "production");

export const env = {
  get openaiKey() {
    return process.env.OPENAI_API_KEY ?? "";
  },
  get openaiModel() {
    return process.env.OPENAI_MODEL || "gpt-4o-mini";
  },
  get retrievalUrl() {
    return process.env.RETRIEVAL_URL || "http://127.0.0.1:8001";
  },
};
