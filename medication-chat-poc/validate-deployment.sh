#!/bin/bash
# Deployment Validation Script for Medical Chat Application

set -e

echo "🔍 VALIDATING DOKPLOY DEPLOYMENT"
echo "=================================="

# Function to check HTTP endpoint
check_endpoint() {
    local url=$1
    local name=$2
    local expected=$3

    echo -n "Checking $name... "
    if response=$(curl -s -w "%{http_code}" "$url" -o /tmp/response 2>/dev/null); then
        if [[ "$response" == "200" ]]; then
            if [[ -n "$expected" ]]; then
                if grep -q "$expected" /tmp/response; then
                    echo "✅ PASS"
                    return 0
                else
                    echo "❌ FAIL (wrong response)"
                    cat /tmp/response
                    return 1
                fi
            else
                echo "✅ PASS"
                return 0
            fi
        else
            echo "❌ FAIL (HTTP $response)"
            return 1
        fi
    else
        echo "❌ FAIL (connection error)"
        return 1
    fi
}

# Function to check Docker service
check_docker_service() {
    local service=$1
    echo -n "Checking Docker service $service... "
    if docker-compose -f dokploy-compose.yml ps | grep -q "$service.*Up"; then
        echo "✅ RUNNING"
        return 0
    else
        echo "❌ NOT RUNNING"
        return 1
    fi
}

# Check if deployment files exist
echo "📁 Checking deployment files..."
required_files=(
    "dokploy-compose.yml"
    "dokploy.json"
    ".env.production"
    "backend/Dockerfile"
    "frontend/Dockerfile"
    "ollama/Dockerfile"
    "ollama/download-models.sh"
)

missing_files=()
for file in "${required_files[@]}"; do
    if [[ ! -f "$file" ]]; then
        missing_files+=("$file")
    fi
done

if [[ ${#missing_files[@]} -eq 0 ]]; then
    echo "✅ All deployment files present"
else
    echo "❌ Missing files:"
    printf '  - %s\n' "${missing_files[@]}"
    exit 1
fi

# Check Docker services
echo ""
echo "🐳 Checking Docker services..."
services=("ollama" "backend" "frontend")
failed_services=()

for service in "${services[@]}"; do
    if ! check_docker_service "$service"; then
        failed_services+=("$service")
    fi
done

if [[ ${#failed_services[@]} -gt 0 ]]; then
    echo "❌ Failed services: ${failed_services[*]}"
    echo "Try running: docker-compose -f dokploy-compose.yml up -d"
    exit 1
fi

# Check health endpoints
echo ""
echo "🏥 Checking health endpoints..."
health_checks=(
    "http://localhost:11434/api/tags|ollama|models"
    "http://localhost:8000/api/v1/health/|backend|healthy"
    "http://localhost:3000/api/health|frontend|"
)

failed_health=()
for check in "${health_checks[@]}"; do
    IFS='|' read -r url name expected <<< "$check"
    if ! check_endpoint "$url" "$name" "$expected"; then
        failed_health+=("$name")
    fi
done

if [[ ${#failed_health[@]} -gt 0 ]]; then
    echo "❌ Failed health checks: ${failed_health[*]}"
fi

# Check AI models
echo ""
echo "🧠 Checking AI models..."
if ollama_response=$(curl -s http://localhost:11434/api/tags 2>/dev/null); then
    if echo "$ollama_response" | grep -q "medllama2"; then
        echo "✅ MedLlama2 model available"
    else
        echo "❌ MedLlama2 model missing"
    fi

    if echo "$ollama_response" | grep -q "seallm"; then
        echo "✅ SeaLLM model available"
    else
        echo "❌ SeaLLM model missing"
    fi
else
    echo "❌ Cannot check models (Ollama not responding)"
fi

# Test medical AI functionality
echo ""
echo "🩺 Testing medical AI functionality..."
test_payload='{
    "message": "ปวดหัว ไข้",
    "patient_info": {"age": 30, "gender": "female"},
    "session_id": "validation_test"
}'

if ai_response=$(curl -s -X POST "http://localhost:8000/api/v1/medical/chat/" \
    -H "Content-Type: application/json" \
    -d "$test_payload" 2>/dev/null); then

    if echo "$ai_response" | grep -q "type"; then
        echo "✅ Medical AI responding"

        # Check for VitalSigns contamination
        if echo "$ai_response" | grep -qi "vital"; then
            echo "⚠️ WARNING: VitalSigns references detected in response"
        else
            echo "✅ No VitalSigns contamination detected"
        fi
    else
        echo "❌ Medical AI not responding correctly"
        echo "Response: $ai_response"
    fi
else
    echo "❌ Cannot test Medical AI"
fi

# Resource usage check
echo ""
echo "📊 Checking resource usage..."
if command -v docker &> /dev/null; then
    echo "Memory usage by service:"
    docker stats --no-stream --format "table {{.Name}}\t{{.MemUsage}}\t{{.CPUPerc}}" | grep medical-chat

    # Check if any container is using too much memory
    high_memory=$(docker stats --no-stream --format "{{.Name}} {{.MemUsage}}" | grep medical-chat | while read -r name usage; do
        mem_val=$(echo "$usage" | cut -d'/' -f1 | sed 's/[^0-9.]//g')
        if (( $(echo "$mem_val > 1024" | bc -l) )); then
            echo "$name: ${usage}"
        fi
    done)

    if [[ -n "$high_memory" ]]; then
        echo "⚠️ High memory usage detected:"
        echo "$high_memory"
    fi
fi

# Network connectivity check
echo ""
echo "🌐 Checking internal network connectivity..."
if docker-compose -f dokploy-compose.yml exec -T backend curl -f http://ollama:11434/api/tags > /dev/null 2>&1; then
    echo "✅ Backend can reach Ollama"
else
    echo "❌ Backend cannot reach Ollama"
fi

if docker-compose -f dokploy-compose.yml exec -T frontend curl -f http://backend:8000/api/v1/health/ > /dev/null 2>&1; then
    echo "✅ Frontend can reach Backend"
else
    echo "❌ Frontend cannot reach Backend"
fi

# Security check
echo ""
echo "🔒 Basic security validation..."
if [[ -f ".env.production" ]]; then
    if grep -q "your-domain.com" .env.production; then
        echo "⚠️ Update CORS_ORIGINS in .env.production with your actual domain"
    else
        echo "✅ CORS configuration appears updated"
    fi
fi

# Final summary
echo ""
echo "📋 VALIDATION SUMMARY"
echo "===================="

all_passed=true

if [[ ${#failed_services[@]} -gt 0 ]]; then
    echo "❌ Docker services: FAILED"
    all_passed=false
else
    echo "✅ Docker services: PASSED"
fi

if [[ ${#failed_health[@]} -gt 0 ]]; then
    echo "❌ Health checks: FAILED"
    all_passed=false
else
    echo "✅ Health checks: PASSED"
fi

if [[ "$all_passed" == true ]]; then
    echo ""
    echo "🎉 DEPLOYMENT VALIDATION SUCCESSFUL!"
    echo "Your Medical Chat Application is ready for production use."
    echo ""
    echo "Access URLs:"
    echo "  Frontend: http://localhost:3000"
    echo "  Backend:  http://localhost:8000"
    echo "  API Docs: http://localhost:8000/docs"
    echo ""
    echo "Next steps:"
    echo "1. Update domain configuration"
    echo "2. Set up SSL certificates"
    echo "3. Configure monitoring"
    echo "4. Run load tests"
else
    echo ""
    echo "❌ DEPLOYMENT VALIDATION FAILED"
    echo "Please fix the issues above before proceeding to production."
    exit 1
fi