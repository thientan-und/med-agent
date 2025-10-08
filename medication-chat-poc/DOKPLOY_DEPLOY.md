# วิธี Deploy ขึ้น Dokploy (Manual Upload)

## เตรียมไฟล์

### 1. Compress โปรเจค

```bash
cd /Users/plug-und/Development/chat-agent
tar -czf med-agent.tar.gz medication-chat-poc/
```

### 2. Upload ไปยัง Server

```bash
# ถ้ามี SSH access
scp med-agent.tar.gz user@your-dokploy-server:/path/to/upload/

# หรือใช้ SFTP/FTP ตาม Dokploy server ของคุณ
```

### 3. Extract บน Server

```bash
ssh user@your-dokploy-server
cd /path/to/upload
tar -xzf med-agent.tar.gz
cd medication-chat-poc
```

## Deploy ด้วย Docker Compose

### ตัวเลือกที่ 1: Deploy ทั้งหมด (Ollama + Backend + Frontend)

```bash
docker compose -f dokploy-compose.yml up -d --build
```

### ตัวเลือกที่ 2: Deploy ทีละ Service

```bash
# 1. Start Ollama ก่อน (ใช้เวลา 2-3 นาที download models)
docker compose -f dokploy-compose.yml up -d ollama

# 2. รอให้ Ollama พร้อม แล้ว start Backend
docker compose -f dokploy-compose.yml up -d backend

# 3. รอให้ Backend พร้อม แล้ว start Frontend
docker compose -f dokploy-compose.yml up -d frontend
```

## ตรวจสอบ Status

```bash
# ดู logs
docker compose -f dokploy-compose.yml logs -f

# ดู status ทุก service
docker compose -f dokploy-compose.yml ps

# ตรวจสอบ health แต่ละ service
curl http://localhost:11434/api/tags          # Ollama
curl http://localhost:8000/api/v1/health/     # Backend
curl http://localhost:3000                     # Frontend
```

## ถ้า Dokploy มี Web Interface

### Create Services Manually

**Service 1: Ollama**
- Name: `medical-chat-ollama`
- Type: Docker
- Build Context: `./ollama`
- Dockerfile: `Dockerfile`
- Port: `11434`
- Volume: `ollama_data:/root/.ollama`

**Service 2: Backend**
- Name: `medical-chat-backend`
- Type: Docker
- Build Context: `./backend`
- Dockerfile: `Dockerfile`
- Port: `8000`
- Environment:
  - `OLLAMA_URL=http://medical-chat-ollama:11434`
  - `ENVIRONMENT=production`
- Depends on: `medical-chat-ollama`

**Service 3: Frontend**
- Name: `medical-chat-frontend`
- Type: Docker
- Build Context: `./frontend`
- Dockerfile: `Dockerfile`
- Port: `3000`, `3003`
- Environment:
  - `OLLAMA_URL=http://medical-chat-ollama:11434`
  - `API_BASE_URL=http://medical-chat-backend:8000`
- Depends on: `medical-chat-backend`

## Alternative: Deploy แบบ Standalone (ไม่ใช้ Dokploy)

หากต้องการ deploy โดยตรงบน server:

```bash
# 1. Clone repo
git clone https://github.com/thientan-und/med-agent.git
cd med-agent/medication-chat-poc

# 2. Deploy
./deploy.sh
```

## Production URLs

หลัง deploy เสร็จ ให้เปิด:
- Frontend: `http://your-server-ip:3000`
- Backend API: `http://your-server-ip:8000`
- Ollama: `http://your-server-ip:11434`

## Update CORS

แก้ไข `dokploy-compose.yml` ให้ตรงกับ domain จริง:

```yaml
backend:
  environment:
    - CORS_ORIGINS=["http://your-domain.com", "https://your-domain.com"]
```

## Troubleshooting

### Ollama ไม่ download models
```bash
docker compose -f dokploy-compose.yml exec ollama ollama pull medllama2
docker compose -f dokploy-compose.yml exec ollama ollama pull nxphi47/seallm-7b-v2-q4_0
```

### Backend ติดต่อ Ollama ไม่ได้
```bash
# ตรวจสอบ network
docker compose -f dokploy-compose.yml exec backend curl http://ollama:11434/api/tags
```

### Memory ไม่พอ
Ollama ต้องการ RAM อย่างน้อย 8GB, แนะนำ 16GB+
