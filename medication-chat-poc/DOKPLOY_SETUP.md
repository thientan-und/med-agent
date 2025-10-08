# Deploy ขึ้น Dokploy Dashboard

URL: http://172.30.103.242:3000/dashboard/project/lAVf_UVOjluz1hvnmojmq/environment/Hj7QPSks7wc96_VImswvK

## วิธีที่ 1: Deploy ด้วย Docker Compose (แนะนำ)

### ขั้นตอน:

1. **ใน Dokploy Dashboard** → คลิก "Create New Application" หรือ "Add Service"

2. **เลือก Application Type:**
   - เลือก "Docker Compose" หรือ "Compose"

3. **Upload Docker Compose File:**
   - Upload ไฟล์ `dokploy-compose.yml` จากโปรเจค
   - หรือ copy-paste เนื้อหาจาก `dokploy-compose.yml`

4. **กด Deploy**

---

## วิธีที่ 2: สร้าง 3 Applications แยกกัน

### Application 1: Ollama Service

**ขั้นตอน:**
1. คลิก "+ Create Application" หรือ "New Service"
2. ตั้งค่า:
   - **Name:** `medical-chat-ollama`
   - **Type:** Docker / Dockerfile
   - **Repository:** Manual Upload
   - **Build Context:** Upload โฟลเดอร์ `ollama/`
   - **Dockerfile Path:** `Dockerfile`

3. **Ports:**
   - Container Port: `11434`
   - Host Port: `11434`

4. **Volumes:**
   - Mount: `/root/.ollama`
   - Type: Persistent Volume
   - Name: `ollama_data`

5. **Environment Variables:** (ไม่จำเป็น)
   - `OLLAMA_HOST=0.0.0.0`

6. **Health Check:**
   - Endpoint: `http://localhost:11434/api/tags`
   - Interval: 30s

7. **กด Deploy**

---

### Application 2: Backend Service

**ขั้นตอน:**
1. คลิก "+ Create Application"
2. ตั้งค่า:
   - **Name:** `medical-chat-backend`
   - **Type:** Docker / Dockerfile
   - **Build Context:** Upload โฟลเดอร์ `backend/`
   - **Dockerfile Path:** `Dockerfile`

3. **Ports:**
   - Container Port: `8000`
   - Host Port: `8000`

4. **Volumes:**
   - Volume 1: `/app/logs` → `backend_logs`
   - Volume 2: `/app/data` → `backend_data`

5. **Environment Variables:**
   ```
   ENVIRONMENT=production
   HOST=0.0.0.0
   PORT=8000
   OLLAMA_URL=http://medical-chat-ollama:11434
   SEALLM_MODEL=nxphi47/seallm-7b-v2-q4_0:latest
   MEDLLAMA_MODEL=medllama2:latest
   CORS_ORIGINS=["http://172.30.103.242:3000"]
   ```

6. **Dependencies:**
   - Depends on: `medical-chat-ollama`
   - Wait for health check: Yes

7. **Health Check:**
   - Endpoint: `http://localhost:8000/api/v1/health/`
   - Interval: 30s

8. **กด Deploy**

---

### Application 3: Frontend Service

**ขั้นตอน:**
1. คลิก "+ Create Application"
2. ตั้งค่า:
   - **Name:** `medical-chat-frontend`
   - **Type:** Docker / Dockerfile
   - **Build Context:** Upload โฟลเดอร์ `frontend/`
   - **Dockerfile Path:** `Dockerfile`

3. **Ports:**
   - Container Port: `3000` → Host Port: `3000`
   - Container Port: `3003` → Host Port: `3003`

4. **Environment Variables:**
   ```
   NODE_ENV=production
   NEXT_TELEMETRY_DISABLED=1
   PORT=3000
   WEBSOCKET_PORT=3003
   OLLAMA_URL=http://medical-chat-ollama:11434
   API_BASE_URL=http://medical-chat-backend:8000
   ```

5. **Dependencies:**
   - Depends on: `medical-chat-backend`
   - Wait for health check: Yes

6. **Health Check:**
   - Endpoint: `http://localhost:3000/api/health`
   - Interval: 30s

7. **กด Deploy**

---

## วิธีที่ 3: Upload via Git Repository

ถ้า Dokploy รองรับ Git:

1. **คลิก "+ Create Application"**
2. **เลือก Source:**
   - Git Repository: `https://github.com/thientan-und/med-agent`
   - Branch: `main`
   - Path: `medication-chat-poc`

3. **Build Method:**
   - Docker Compose
   - Compose File: `dokploy-compose.yml`

4. **กด Deploy**

---

## วิธีที่ 4: Upload ไฟล์โดยตรง (Manual Upload)

### สร้างไฟล์ compress ก่อน:

```bash
cd /Users/plug-und/Development/chat-agent/medication-chat-poc
tar -czf ../med-agent-deploy.tar.gz .
```

### ใน Dokploy Dashboard:

1. **หา Upload Function** (File Upload / Import)
2. **Upload:** `med-agent-deploy.tar.gz`
3. **Extract Path:** ระบุ path ที่ต้องการ
4. **เลือก Deploy Method:** Docker Compose
5. **Compose File:** `dokploy-compose.yml`
6. **กด Deploy**

---

## ตรวจสอบหลัง Deploy

### ใน Dokploy Dashboard:

1. ดู **Logs** ของแต่ละ service
2. ตรวจสอบ **Status** ต้องเป็น "Running" หรือ "Healthy"
3. ดู **Resource Usage** (CPU, Memory)

### ทดสอบ Services:

```bash
# Ollama
curl http://172.30.103.242:11434/api/tags

# Backend
curl http://172.30.103.242:8000/api/v1/health/

# Frontend
เปิดเบราว์เซอร์: http://172.30.103.242:3000
```

---

## หมายเหตุสำคัญ

### Resource Requirements:
- **Ollama:** RAM อย่างน้อย 8GB (แนะนำ 16GB)
- **Backend:** RAM 2GB
- **Frontend:** RAM 1GB
- **รวม:** ต้องการ RAM อย่างน้อย 11GB+

### Startup Time:
- **Ollama:** 2-5 นาที (download models ครั้งแรก)
- **Backend:** 30 วินาที (รอ Ollama)
- **Frontend:** 30 วินาที (รอ Backend)

### Network:
- Services ต้องอยู่ใน network เดียวกัน
- Dokploy จะจัดการ internal networking อัตโนมัติ
