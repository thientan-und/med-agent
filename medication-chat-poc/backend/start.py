#!/usr/bin/env python3
"""
Medical Chat AI Backend Startup Script
Launch the FastAPI server with proper configuration
"""

import os
import sys
import asyncio
import argparse
import uvicorn
from pathlib import Path

# Add backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from app.util.config import get_settings, get_settings_for_environment
from main import app


def setup_environment():
    """Setup environment variables and paths"""

    # Set default environment if not specified
    if not os.getenv("ENVIRONMENT"):
        os.environ["ENVIRONMENT"] = "development"

    # Set Python path
    os.environ["PYTHONPATH"] = str(backend_dir)

    print(f"🔧 Environment: {os.getenv('ENVIRONMENT')}")
    print(f"🔧 Python path: {backend_dir}")


async def check_dependencies():
    """Check if required services are available"""

    print("🔍 Checking dependencies...")

    settings = get_settings()

    # Check Ollama server
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{settings.ollama_url}/api/tags", timeout=5) as response:
                if response.status == 200:
                    models = await response.json()
                    model_count = len(models.get("models", []))
                    print(f"✅ Ollama server connected ({model_count} models available)")
                else:
                    print(f"⚠️  Ollama server responded with status {response.status}")
    except Exception as e:
        print(f"⚠️  Ollama server check failed: {e}")
        print("   Make sure Ollama is running: ./setup-ollama.sh")

    # Check medical data files
    data_files = [
        ("Medicine data", settings.medicine_data_path),
        ("Diagnosis data", settings.diagnosis_data_path),
        ("Treatment data", settings.treatment_data_path)
    ]

    for name, path in data_files:
        if os.path.exists(path):
            print(f"✅ {name} found")
        else:
            print(f"⚠️  {name} not found at {path}")

    print()


def create_directories():
    """Create necessary directories"""

    directories = [
        "logs",
        "training-data",
        "temp"
    ]

    for directory in directories:
        dir_path = backend_dir / directory
        dir_path.mkdir(exist_ok=True)
        print(f"📁 Created directory: {directory}")


def main():
    """Main startup function"""

    parser = argparse.ArgumentParser(description="Medical Chat AI Backend")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--log-level", default="info", help="Log level")
    parser.add_argument("--check-only", action="store_true", help="Only check dependencies")

    args = parser.parse_args()

    # Setup
    setup_environment()
    create_directories()

    # Get settings
    settings = get_settings_for_environment()

    # Override with command line args
    host = args.host or settings.host
    port = args.port or settings.port

    print(f"""
🏥 Medical Chat AI Backend
==========================
Version: {settings.version}
Environment: {os.getenv('ENVIRONMENT')}
Host: {host}
Port: {port}
Debug: {settings.debug}
Ollama URL: {settings.ollama_url}
""")

    # Check dependencies
    asyncio.run(check_dependencies())

    if args.check_only:
        print("✅ Dependency check completed")
        return

    # Configure uvicorn
    uvicorn_config = {
        "app": "main:app",
        "host": host,
        "port": port,
        "log_level": args.log_level,
        "access_log": True,
    }

    # Development vs Production settings
    if settings.debug or args.reload:
        uvicorn_config.update({
            "reload": True,
            "reload_dirs": [str(backend_dir)],
            "reload_includes": ["*.py"],
        })
    else:
        uvicorn_config.update({
            "workers": args.workers,
        })

    print(f"🚀 Starting Medical AI Backend on http://{host}:{port}")
    print(f"📚 API Documentation: http://{host}:{port}/docs")
    print(f"📖 Alternative Docs: http://{host}:{port}/redoc")
    print()

    try:
        uvicorn.run(**uvicorn_config)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()