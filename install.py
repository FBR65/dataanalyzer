#!/usr/bin/env python3
"""
Enhanced Data Analyzer Installation Script
==========================================

This script installs and configures the Enhanced Data Analyzer with all components.
"""

import sys
import subprocess
import json
from pathlib import Path


def run_command(command, description=""):
    """Run a command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(
            command, shell=True, check=True, capture_output=True, text=True
        )
        print(f"✅ {description} completed successfully")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return None


def check_system_requirements():
    """Check system requirements"""
    print("🔍 Checking system requirements...")

    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 10):
        print(
            f"❌ Python 3.10+ required, found {python_version.major}.{python_version.minor}"
        )
        return False

    print(f"✅ Python {python_version.major}.{python_version.minor} detected")

    # Check uv
    uv_check = run_command("uv --version", "Checking uv")
    if not uv_check:
        print("⚠️  uv not found, installing...")
        pip_install = run_command("pip install uv", "Installing uv")
        if not pip_install:
            print("❌ Failed to install uv")
            return False

    # Check Docker (optional)
    docker_check = run_command("docker --version", "Checking Docker")
    if docker_check:
        print("✅ Docker available")
    else:
        print("⚠️  Docker not available - containerization features will be limited")

    return True


def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")

    directories = [
        "excel_files",
        "databases",
        "ducklake",
        "generated_code",
        "logs",
        "temp",
        "notebooks",
        "exports",
    ]

    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")


def install_dependencies():
    """Install Python dependencies"""
    print("📦 Installing dependencies...")

    # Install main dependencies
    install_result = run_command("uv sync", "Installing main dependencies")
    if not install_result:
        print("❌ Failed to install dependencies")
        return False

    # Install additional packages
    additional_packages = [
        "pygwalker",
        "streamlit",
        "watchdog",
        "schedule",
        "apscheduler",
    ]

    for package in additional_packages:
        run_command(f"uv add {package}", f"Installing {package}")

    return True


def create_configuration():
    """Create configuration files"""
    print("⚙️  Creating configuration files...")

    # Create .env file
    env_content = """# Enhanced Data Analyzer Configuration
# Database Settings
DUCKDB_PATH=./databases/main.duckdb
DUCK_LAKE_PATH=./ducklake

# LLM Settings
OPENAI_API_KEY=ollama
OPENAI_API_BASE_URL=http://localhost:11434/v1
LLM_MODEL=qwen2.5:latest
CODING_MODEL=qwen2.5-coder:latest

# Interface Settings
GRADIO_SERVER_NAME=0.0.0.0
GRADIO_SERVER_PORT=7860
GRADIO_SHARE=false

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/dataanalyzer.log

# Processing Settings
MAX_FILE_SIZE_MB=100
AUTO_CLEAN_DATA=true
PARALLEL_PROCESSING=true
"""

    with open(".env", "w") as f:
        f.write(env_content)

    print("✅ Created .env file")

    # Create database configs
    db_configs = {
        "mongodb_local": {
            "type": "mongodb",
            "host": "localhost",
            "port": 27017,
            "username": "dataanalyzer",
            "password": "dataanalyzer_pwd",
            "database": "my_test_db",
            "auth_source": "test",
        },
        "mariadb_local": {
            "type": "mariadb",
            "host": "localhost",
            "port": 3306,
            "username": "dataanalyzer",
            "password": "dataanalyzer_pwd",
            "database": "my_test_db",
        },
    }

    with open("database_configs.json", "w") as f:
        json.dump(db_configs, f, indent=2)

    print("✅ Created database configurations")


def setup_ollama():
    """Setup Ollama for local LLM"""
    print("🤖 Setting up Ollama...")

    # Check if Ollama is installed
    ollama_check = run_command("ollama --version", "Checking Ollama")
    if not ollama_check:
        print("⚠️  Ollama not found. Please install from https://ollama.com/")
        return False

    # Pull required models
    models = ["qwen2.5:latest", "qwen2.5-coder:latest"]
    for model in models:
        run_command(f"ollama pull {model}", f"Pulling {model}")

    return True


def create_sample_data():
    """Create sample Excel files for testing"""
    print("📊 Creating sample data...")

    try:
        import pandas as pd

        # Sample sales data
        sales_data = {
            "Date": pd.date_range("2024-01-01", periods=100, freq="D"),
            "Product": ["Product A", "Product B", "Product C"] * 34,
            "Sales": [100, 150, 200] * 34,
            "Region": ["North", "South", "East", "West"] * 25,
        }

        sales_df = pd.DataFrame(sales_data)
        sales_df.to_excel("excel_files/sample_sales.xlsx", index=False)

        # Sample customer data
        customer_data = {
            "CustomerID": range(1, 51),
            "Name": [f"Customer {i}" for i in range(1, 51)],
            "Age": [25 + i % 40 for i in range(50)],
            "City": ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"] * 10,
            "Spend": [1000 + i * 100 for i in range(50)],
        }

        customer_df = pd.DataFrame(customer_data)
        customer_df.to_excel("excel_files/sample_customers.xlsx", index=False)

        print("✅ Created sample Excel files")

    except ImportError:
        print("⚠️  pandas not available, skipping sample data creation")


def run_tests():
    """Run basic tests"""
    print("🧪 Running basic tests...")

    try:
        # Test imports
        from tools.excel_processor import ExcelProcessor
        from tools.duckdb_connector import DuckDBConnector

        print("✅ All imports successful")

        # Test Excel processor
        ExcelProcessor("./excel_files", "./ducklake", "./databases/test.duckdb")
        print("✅ Excel processor initialized")

        # Test DuckDB connector
        DuckDBConnector("./databases/test.duckdb", "./ducklake")
        print("✅ DuckDB connector initialized")

        # Cleanup test files
        test_db = Path("./databases/test.duckdb")
        if test_db.exists():
            test_db.unlink()

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def main():
    """Main installation function"""
    print("🚀 Enhanced Data Analyzer Installation")
    print("=" * 50)

    # Check system requirements
    if not check_system_requirements():
        print("❌ System requirements not met")
        return False

    # Create directories
    create_directories()

    # Install dependencies
    if not install_dependencies():
        print("❌ Dependency installation failed")
        return False

    # Create configuration
    create_configuration()

    # Setup Ollama
    setup_ollama()

    # Create sample data
    create_sample_data()

    # Run tests
    if not run_tests():
        print("❌ Tests failed")
        return False

    print("\n🎉 Installation completed successfully!")
    print("\nNext steps:")
    print("1. Start the enhanced frontend:")
    print("   uv run python enhanced_frontend.py")
    print("\n2. Or start with Docker:")
    print("   docker-compose up -d")
    print("\n3. Access the application at:")
    print("   http://localhost:7860")
    print("\n4. For scheduled processing:")
    print("   uv run python scheduler.py --config scheduler_config.json")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
