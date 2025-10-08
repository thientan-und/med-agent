# Dokploy Deployment Guide
## Medical Chat Application with AI Models

This guide provides step-by-step instructions for deploying the Medical Chat Application on Dokploy, including frontend, backend, and Ollama with medical AI models.

---

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend       │    │   Ollama        │
│   (Next.js)     │    │   (FastAPI)     │    │   (AI Models)   │
│   Port: 3000    │◄──►│   Port: 8000    │◄──►│   Port: 11434   │
│   Port: 3003 WS │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Services:
- **Frontend**: Next.js with Thai medical chat interface
- **Backend**: FastAPI with precision medical AI (no VitalSigns)
- **Ollama**: AI model server with MedLlama2 + SeaLLM models

---

## 📋 Prerequisites

### System Requirements:
- **RAM**: 16GB minimum (32GB recommended)
- **Storage**: 50GB free space (for AI models)
- **CPU**: 4+ cores (8+ recommended)
- **GPU**: Optional but recommended for faster inference

### Software Requirements:
- Docker Engine 20.10+
- Docker Compose 2.0+
- Dokploy CLI (latest version)
- 16GB+ available RAM

---

## 🚀 Quick Deployment

### 1. Clone and Prepare
```bash
# Clone the repository
git clone <repository-url>
cd medical-chat-app

# Make deployment script executable
chmod +x deploy-dokploy.sh
```

### 2. Configure Environment
```bash
# Edit production environment
cp .env.production .env.local
# Update CORS_ORIGINS with your domain
```

### 3. Deploy with Script
```bash
# Run automated deployment
./deploy-dokploy.sh
```

### 4. Verify Deployment
```bash
# Check services
docker-compose -f dokploy-compose.yml ps

# Test endpoints
curl http://localhost:3000/api/health    # Frontend
curl http://localhost:8000/api/v1/health/ # Backend
curl http://localhost:11434/api/tags     # Ollama
```

---

## 🔧 Manual Dokploy Deployment

### Step 1: Dokploy Project Setup
```bash
# Create new Dokploy project
dokploy project create medical-chat-app

# Set project directory
cd medical-chat-app
```

### Step 2: Service Configuration

#### A. Ollama Service
```bash
dokploy service create ollama \
  --type docker \
  --build-context ./ollama \
  --dockerfile Dockerfile \
  --port 11434:11434 \
  --volume ollama_data:/root/.ollama \
  --memory 8GB \
  --env OLLAMA_HOST=0.0.0.0 \
  --env OLLAMA_PORT=11434
```

#### B. Backend Service
```bash
dokploy service create backend \
  --type docker \
  --build-context ./backend \
  --dockerfile Dockerfile \
  --port 8000:8000 \
  --volume backend_logs:/app/logs \
  --memory 2GB \
  --env ENVIRONMENT=production \
  --env OLLAMA_URL=http://ollama:11434 \
  --depends-on ollama
```

#### C. Frontend Service
```bash
dokploy service create frontend \
  --type docker \
  --build-context ./frontend \
  --dockerfile Dockerfile \
  --port 3000:3000 \
  --port 3003:3003 \
  --memory 1GB \
  --env NODE_ENV=production \
  --env API_BASE_URL=http://backend:8000 \
  --depends-on backend
```

### Step 3: Deploy Stack
```bash
dokploy deploy --all
```

---

## 🌐 Domain Configuration

### 1. Configure Domain in Dokploy
```bash
dokploy domain add medical-chat.your-domain.com \
  --service frontend \
  --port 3000 \
  --ssl-enabled
```

### 2. Update Environment Variables
```bash
# Update CORS origins in backend
dokploy env set backend CORS_ORIGINS='["https://medical-chat.your-domain.com"]'

# Update API base URL in frontend
dokploy env set frontend API_BASE_URL=https://medical-chat.your-domain.com/api
```

### 3. Redeploy
```bash
dokploy deploy backend frontend
```

---

## 🔒 Security Configuration

### Environment Variables
```bash
# Production security settings
CORS_ORIGINS=["https://your-domain.com"]
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_PER_HOUR=1000
SESSION_TIMEOUT=3600
```

### Firewall Rules
```bash
# Allow only necessary ports
ufw allow 80/tcp      # HTTP
ufw allow 443/tcp     # HTTPS
ufw deny 11434/tcp    # Block direct Ollama access
ufw deny 8000/tcp     # Block direct API access
```

---

## 📊 Monitoring & Logging

### Health Checks
```bash
# Service health
curl https://your-domain.com/api/health
curl https://your-domain.com/api/v1/health/

# Model availability
docker exec medical-chat-ollama ollama list
```

### Logs
```bash
# View all logs
dokploy logs --all --follow

# Service-specific logs
dokploy logs frontend --follow
dokploy logs backend --follow
dokploy logs ollama --follow
```

### Metrics
```bash
# Resource usage
dokploy stats

# Service status
dokploy status --all
```

---

## 🧪 Testing the Deployment

### 1. Frontend Test
```bash
curl -X GET https://your-domain.com/api/health
# Expected: {"status": "healthy"}
```

### 2. Backend API Test
```bash
curl -X GET https://your-domain.com/api/v1/health/
# Expected: {"status": "healthy", "precision_architecture": "active"}
```

### 3. AI Models Test
```bash
curl -X POST https://your-domain.com/api/v1/medical/chat/ \
  -H "Content-Type: application/json" \
  -d '{
    "message": "ปวดหัว เป็นไข้",
    "patient_info": {"age": 30, "gender": "female"},
    "session_id": "test"
  }'
```

### 4. Load Test
```bash
# Install artillery for load testing
npm install -g artillery

# Run load test
artillery quick --count 10 --num 2 https://your-domain.com
```

---

## 🔧 Maintenance

### Model Updates
```bash
# Update AI models
dokploy exec ollama "ollama pull medllama2:latest"
dokploy exec ollama "ollama pull nxphi47/seallm-7b-v2-q4_0:latest"
```

### Service Updates
```bash
# Update specific service
dokploy deploy frontend

# Update all services
dokploy deploy --all
```

### Backup
```bash
# Backup volumes
dokploy backup create --volumes ollama_data,backend_logs

# Backup configuration
dokploy config export medical-chat-app
```

---

## 🚨 Troubleshooting

### Common Issues

#### 1. Ollama Models Not Loading
```bash
# Check model download status
dokploy logs ollama | grep "download"

# Manually download models
dokploy exec ollama "ollama pull medllama2:latest"
```

#### 2. Backend Connection Issues
```bash
# Check network connectivity
dokploy exec backend "curl -f http://ollama:11434/api/tags"

# Restart backend
dokploy restart backend
```

#### 3. Frontend Build Failures
```bash
# Check build logs
dokploy logs frontend --since 1h

# Rebuild frontend
dokploy build frontend --no-cache
```

#### 4. Memory Issues
```bash
# Check resource usage
dokploy stats

# Increase memory limits
dokploy service update ollama --memory 12GB
```

### Performance Optimization

#### 1. Enable Model Caching
```bash
# Set Ollama cache settings
dokploy env set ollama OLLAMA_KEEP_ALIVE=24h
dokploy env set ollama OLLAMA_MAX_LOADED_MODELS=2
```

#### 2. Frontend Optimization
```bash
# Enable Next.js caching
dokploy env set frontend NEXT_CACHE_STRATEGY=revalidate
```

#### 3. Database Connection Pooling
```bash
# If using database
dokploy env set backend DATABASE_POOL_SIZE=20
dokploy env set backend DATABASE_MAX_OVERFLOW=30
```

---

## 📈 Scaling

### Horizontal Scaling
```bash
# Scale frontend
dokploy scale frontend --replicas 3

# Scale backend
dokploy scale backend --replicas 2

# Note: Ollama typically runs as single instance
```

### Auto-scaling Configuration
```json
{
  "scaling": {
    "frontend": {
      "min": 1,
      "max": 5,
      "cpu_threshold": 70,
      "memory_threshold": 80
    },
    "backend": {
      "min": 1,
      "max": 3,
      "cpu_threshold": 80,
      "memory_threshold": 85
    }
  }
}
```

---

## 🎯 Production Checklist

### Pre-Deployment
- [ ] Update CORS origins
- [ ] Configure SSL certificates
- [ ] Set up monitoring
- [ ] Configure backups
- [ ] Test all endpoints
- [ ] Load test application

### Post-Deployment
- [ ] Verify all services healthy
- [ ] Test AI model responses
- [ ] Check logs for errors
- [ ] Monitor resource usage
- [ ] Set up alerts
- [ ] Document any customizations

---

## 📞 Support

### Resources:
- [Dokploy Documentation](https://dokploy.com/docs)
- [Application Repository](https://github.com/your-repo)
- [Medical AI Models Documentation](./AGENTIC_AI_FLOW.md)

### Emergency Contacts:
- System Admin: admin@your-domain.com
- Medical Team: medical@your-domain.com

---

*This deployment guide ensures a production-ready Medical Chat Application with proper AI model integration and security configurations.*