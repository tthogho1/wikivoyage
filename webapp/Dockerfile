# Multi-stage build producing the Next.js "standalone" server (next.config.ts: output: "standalone").
# Build from the repo root:  docker build -f webapp/Dockerfile -t wikivoyage-webapp .

FROM node:22-slim AS deps
WORKDIR /app
COPY webapp/package.json webapp/package-lock.json* ./
RUN npm install --no-audit --no-fund

FROM node:22-slim AS build
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY webapp/ ./
# webapp/ has no public/ dir; create it so the runtime stage's COPY of it doesn't fail.
RUN mkdir -p public && npm run build

FROM node:22-slim AS runtime
ENV NODE_ENV=production \
    PORT=3000
WORKDIR /app

RUN useradd --create-home --home-dir /home/appuser appuser

# Standalone output bundles only the files/deps needed to run `node server.js`.
COPY --from=build /app/.next/standalone ./
COPY --from=build /app/.next/static ./.next/static
COPY --from=build /app/public ./public

USER appuser
EXPOSE 3000
ENTRYPOINT ["node", "server.js"]
