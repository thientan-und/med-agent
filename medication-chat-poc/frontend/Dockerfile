# Medical Chat App Dockerfile for Dokploy Deployment
# Multi-stage build for optimized production image

# Stage 1: Dependencies
FROM node:22-alpine AS deps
WORKDIR /app

# Install pnpm
RUN npm install -g pnpm@9.0.0

# Copy package files
COPY package.json pnpm-lock.yaml* ./

# Install dependencies
RUN pnpm install --frozen-lockfile

# Stage 2: Builder
FROM node:22-alpine AS builder
WORKDIR /app

# Install pnpm
RUN npm install -g pnpm@9.0.0

# Copy dependencies from deps stage
COPY --from=deps /app/node_modules ./node_modules
COPY . .

# Set environment variables for build
ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

# Build the application
RUN pnpm build

# Stage 3: Runner
FROM node:22-alpine AS runner
WORKDIR /app

# Create non-root user for security
RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 nextjs

# Install pnpm
RUN npm install -g pnpm@9.0.0

# Copy necessary files
COPY --from=builder /app/public ./public
COPY --from=builder /app/package.json ./package.json
COPY --from=builder /app/pnpm-lock.yaml ./pnpm-lock.yaml

# Copy built application
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

# Copy additional files needed for production
COPY --from=builder /app/websocket-server.js ./websocket-server.js
COPY --from=builder /app/start-sync.sh ./start-sync.sh
COPY --from=builder /app/data ./data

# Make scripts executable
RUN chmod +x ./start-sync.sh

# Install production dependencies only
RUN pnpm install --prod --frozen-lockfile

# Set environment variables
ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1
ENV PORT=3000
ENV WEBSOCKET_PORT=3003

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:3000/api/health || exit 1

# Expose ports
EXPOSE 3000 3003

# Switch to non-root user
USER nextjs

# Start command
CMD ["sh", "./start-sync.sh"]