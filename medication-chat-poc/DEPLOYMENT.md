# Deployment Guide - Dokploy

## Quick Deploy

```bash
./deploy.sh
```

## Manual Deployment Steps

### 1. Build and Start Services

```bash
docker-compose -f dokploy-compose.yml up -d --build
```

### 2. Check Service Status

```bash
docker-compose -f dokploy-compose.yml ps
```

### 3. View Logs

```bash
# All services
docker-compose -f dokploy-compose.yml logs -f

# Specific service
docker-compose -f dokploy-compose.yml logs -f ollama
docker-compose -f dokploy-compose.yml logs -f backend
docker-compose -f dokploy-compose.yml logs -f frontend
```

### 4. Stop Services

```bash
docker-compose -f dokploy-compose.yml down
```

### 5. Stop and Remove Volumes

```bash
docker-compose -f dokploy-compose.yml down -v
```

## Service Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Frontend (Next.js)               │
│                 Port: 3000, 3003                    │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│               Backend (FastAPI)                     │
│                    Port: 8000                       │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│            Ollama (AI Models)                       │
│  Models: medllama2, seallm-7b-v2                   │
│                Port: 11434                          │
└─────────────────────────────────────────────────────┘
```

## Service Details

### Ollama Service
- **Port:** 11434
- **Models:** medllama2, seallm-7b-v2
- **Health Check:** `/api/tags`
- **Volume:** `ollama_data` for persistent model storage
- **Startup Time:** ~2-3 minutes (downloads models on first run)

### Backend Service
- **Port:** 8000
- **Health Check:** `/api/v1/health/`
- **Dependencies:** Requires Ollama to be healthy
- **Volumes:**
  - `backend_logs` for application logs
  - `backend_data` for RAG knowledge base

### Frontend Service
- **Ports:** 3000 (HTTP), 3003 (WebSocket)
- **Health Check:** `/api/health`
- **Dependencies:** Requires Backend to be healthy

## Environment Variables

### Backend
- `ENVIRONMENT=production`
- `OLLAMA_URL=http://ollama:11434`
- `SEALLM_MODEL=nxphi47/seallm-7b-v2-q4_0:latest`
- `MEDLLAMA_MODEL=medllama2:latest`

### Frontend
- `NODE_ENV=production`
- `OLLAMA_URL=http://ollama:11434`
- `API_BASE_URL=http://backend:8000`

## Dokploy Configuration

The repository includes `dokploy.json` for automated deployment via Dokploy platform.

### Import to Dokploy

1. In Dokploy dashboard, create new project
2. Choose "Docker Compose" deployment
3. Point to this repository
4. Dokploy will automatically use `dokploy-compose.yml`

### Manual Configuration in Dokploy

Alternatively, use the `dokploy.json` file:

```bash
dokploy import dokploy.json
```

## Troubleshooting

### Ollama Models Not Loading

```bash
# Check Ollama logs
docker-compose -f dokploy-compose.yml logs ollama

# Manually pull models
docker-compose -f dokploy-compose.yml exec ollama ollama pull medllama2
docker-compose -f dokploy-compose.yml exec ollama ollama pull nxphi47/seallm-7b-v2-q4_0
```

### Backend Cannot Connect to Ollama

```bash
# Check network connectivity
docker-compose -f dokploy-compose.yml exec backend curl http://ollama:11434/api/tags

# Restart backend
docker-compose -f dokploy-compose.yml restart backend
```

### Frontend Cannot Connect to Backend

```bash
# Check backend health
curl http://localhost:8000/api/v1/health/

# Check backend logs
docker-compose -f dokploy-compose.yml logs backend
```

## Production Considerations

### 1. Update CORS Origins

Edit `dokploy-compose.yml`:

```yaml
backend:
  environment:
    - CORS_ORIGINS=["https://your-production-domain.com"]
```

### 2. Add Reverse Proxy (Nginx/Traefik)

Add labels for Traefik or configure Nginx to handle SSL/TLS.

### 3. Resource Limits

Add resource constraints in `dokploy-compose.yml`:

```yaml
services:
  ollama:
    deploy:
      resources:
        limits:
          memory: 8G
        reservations:
          memory: 4G
```

### 4. Monitoring

Add health monitoring services like Prometheus/Grafana.

## Backup and Recovery

### Backup Volumes

```bash
# Backup Ollama models
docker run --rm -v medical-chat-poc_ollama_data:/data -v $(pwd):/backup alpine tar czf /backup/ollama-backup.tar.gz /data

# Backup backend data
docker run --rm -v medical-chat-poc_backend_data:/data -v $(pwd):/backup alpine tar czf /backup/backend-backup.tar.gz /data
```

### Restore Volumes

```bash
# Restore Ollama models
docker run --rm -v medical-chat-poc_ollama_data:/data -v $(pwd):/backup alpine tar xzf /backup/ollama-backup.tar.gz -C /

# Restore backend data
docker run --rm -v medical-chat-poc_backend_data:/data -v $(pwd):/backup alpine tar xzf /backup/backend-backup.tar.gz -C /
```
