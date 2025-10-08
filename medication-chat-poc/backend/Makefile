# Medical Chat API - FastAPI Backend Makefile
# Easy commands for development and deployment

.PHONY: help run dev prod check install test clean logs setup

# Default target
help:
	@echo "🏥 Medical Chat API - FastAPI Backend"
	@echo "====================================="
	@echo ""
	@echo "Available commands:"
	@echo "  make run        - Run development server with auto-reload"
	@echo "  make dev        - Same as 'run' (alias)"
	@echo "  make prod       - Run production server"
	@echo "  make check      - Check configuration without starting"
	@echo "  make install    - Install dependencies"
	@echo "  make test       - Run API tests"
	@echo "  make setup      - Setup backend environment"
	@echo "  make clean      - Clean cache and temporary files"
	@echo "  make logs       - View recent logs"
	@echo "  make help       - Show this help message"
	@echo ""

# Development server with auto-reload
run:
	@echo "🚀 Starting FastAPI development server..."
	@echo "📍 API will be available at: http://localhost:8000"
	@echo "📍 API documentation: http://localhost:8000/docs"
	@echo "📍 Interactive docs: http://localhost:8000/redoc"
	@echo ""
	python start.py --reload

# Alias for run
dev: run

# Production server
prod:
	@echo "🏭 Starting FastAPI production server..."
	@echo "📍 API will be available at: http://localhost:8000"
	@echo ""
	python start.py --prod

# Check configuration
check:
	@echo "🔍 Checking FastAPI configuration..."
	python start.py --check-only

# Install dependencies
install:
	@echo "📦 Installing Python dependencies..."
	pip install -r requirements.txt
	@echo "✅ Dependencies installed successfully!"

# Setup backend environment
setup:
	@echo "🔧 Setting up backend environment..."
	chmod +x setup_backend.sh
	./setup_backend.sh
	@echo "✅ Backend setup complete!"

# Run API tests
test:
	@echo "🧪 Running API tests..."
	@if [ -f "../test-patient-scenarios.js" ]; then \
		cd .. && node test-patient-scenarios.js; \
	else \
		echo "❌ Test file not found. Please run from project root."; \
	fi

# Clean cache and temporary files
clean:
	@echo "🧹 Cleaning cache and temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf temp/* 2>/dev/null || true
	rm -rf logs/*.log.* 2>/dev/null || true
	@echo "✅ Cleanup complete!"

# View recent logs
logs:
	@echo "📋 Recent API logs:"
	@echo "=================="
	@if [ -f "logs/medical_api.log" ]; then \
		tail -50 logs/medical_api.log; \
	else \
		echo "📝 No log files found. Start the server to generate logs."; \
	fi

# Quick status check
status:
	@echo "📊 FastAPI Backend Status"
	@echo "========================"
	@echo "📁 Working directory: $(PWD)"
	@echo "🐍 Python version: $(shell python --version 2>/dev/null || echo 'Not found')"
	@echo "📦 Pip version: $(shell pip --version 2>/dev/null || echo 'Not found')"
	@echo "🔧 FastAPI installed: $(shell python -c 'import fastapi; print(fastapi.__version__)' 2>/dev/null || echo 'Not installed')"
	@echo "📝 Log files: $(shell ls -la logs/ 2>/dev/null | wc -l || echo '0') files"
	@echo "💾 Cache size: $(shell du -sh __pycache__ 2>/dev/null | cut -f1 || echo '0B')"

# Development shortcuts
restart: clean run

# Docker commands (if needed)
docker-build:
	@echo "🐳 Building Docker image..."
	docker build -t medical-chat-api .

docker-run:
	@echo "🐳 Running Docker container..."
	docker run -p 8000:8000 medical-chat-api

# Environment info
env-info:
	@echo "🌍 Environment Information"
	@echo "========================="
	@echo "📁 Current directory: $(PWD)"
	@echo "🔧 Environment file: $(shell [ -f .env ] && echo '✅ Found' || echo '❌ Missing')"
	@echo "📋 Requirements file: $(shell [ -f requirements.txt ] && echo '✅ Found' || echo '❌ Missing')"
	@echo "🚀 Start script: $(shell [ -f start.py ] && echo '✅ Found' || echo '❌ Missing')"
	@echo ""
	@echo "📊 Dependencies:"
	@python -c "import pkg_resources; print('\n'.join([f'  {pkg.key}=={pkg.version}' for pkg in pkg_resources.working_set if pkg.key in ['fastapi', 'uvicorn', 'pydantic']]))" 2>/dev/null || echo "  Dependencies not installed"