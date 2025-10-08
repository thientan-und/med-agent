# Dokploy - Deploy แบบแยก Service (Simple Method)

ถ้า Compose ไม่ work ให้ลอง deploy แยก 3 services:

## Service 1: Ollama (ต้อง deploy ก่อน)

**ใน Dokploy Dashboard:**

1. คลิก "+ Create Service" → **Application**
2. กรอก:
   - **Name:** `ollama`
   - **Source:** GitHub
   - **Repository:** `med-agent`
   - **Branch:** `main`
   - **Build Type:** Dockerfile
   - **Dockerfile Path:** `medication-chat-poc/ollama/Dockerfile`
   - **Context Path:** `medication-chat-poc/ollama`

3. **Ports:**
   - Container Port: `11434` → Host Port: `11434`

4. **Volumes:**
   - Mount Path: `/root/.ollama`
   - Volume Name: `ollama_data`

5. **Environment:** (optional)
   ```
   OLLAMA_HOST=0.0.0.0
   ```

6. **Deploy** และรอ 3-5 นาที (download models)

---

## Service 2: Backend (deploy หลัง Ollama พร้อม)

**ใน Dokploy Dashboard:**

1. คลิก "+ Create Service" → **Application**
2. กรอก:
   - **Name:** `backend`
   - **Source:** GitHub
   - **Repository:** `med-agent`
   - **Branch:** `main`
   - **Build Type:** Dockerfile
   - **Dockerfile Path:** `medication-chat-poc/backend/Dockerfile`
   - **Context Path:** `medication-chat-poc/backend`

3. **Ports:**
   - Container Port: `8000` → Host Port: `8000`

4. **Volumes:**
   - Volume 1: `/app/logs` → `backend_logs`
   - Volume 2: `/app/data` → `backend_data`

5. **Environment:**
   ```
   ENVIRONMENT=production
   HOST=0.0.0.0
   PORT=8000
   OLLAMA_URL=http://ollama:11434
   SEALLM_MODEL=nxphi47/seallm-7b-v2-q4_0:latest
   MEDLLAMA_MODEL=medllama2:latest
   CORS_ORIGINS=["http://172.30.103.242:3000"]
   ```

6. **Deploy**

---

## Service 3: Frontend (deploy หลัง Backend พร้อม)

**ใน Dokploy Dashboard:**

1. คลิก "+ Create Service" → **Application**
2. กรอก:
   - **Name:** `frontend`
   - **Source:** GitHub
   - **Repository:** `med-agent`
   - **Branch:** `main`
   - **Build Type:** Dockerfile
   - **Dockerfile Path:** `medication-chat-poc/frontend/Dockerfile`
   - **Context Path:** `medication-chat-poc/frontend`

3. **Ports:**
   - Port 1: `3000` → `3000`
   - Port 2: `3003` → `3003`

4. **Environment:**
   ```
   NODE_ENV=production
   NEXT_TELEMETRY_DISABLED=1
   PORT=3000
   WEBSOCKET_PORT=3003
   OLLAMA_URL=http://ollama:11434
   API_BASE_URL=http://backend:8000
   ```

5. **Deploy**

---

## ตรวจสอบ

หลัง deploy เสร็จ:

```bash
# Ollama
curl http://172.30.103.242:11434/api/tags

# Backend
curl http://172.30.103.242:8000/api/v1/health/

# Frontend
เปิดเบราว์เซอร์: http://172.30.103.242:3000
```

---

## หมายเหตุ

- **Ollama ใช้เวลา 2-5 นาที** download models ครั้งแรก
- **Backend ต้องรอ Ollama พร้อม** ถึงจะ start ได้
- **Frontend ต้องรอ Backend พร้อม** ถึงจะใช้งานได้
- **Network:** Dokploy จะจัดการ internal networking อัตโนมัติ
