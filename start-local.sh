#!/bin/bash

# Start script for local development with external databases
# This script uses your local Ollama, MongoDB, and MariaDB instances

echo "🚀 Starting Data Analyzer with local databases..."

# Check if local services are running
echo "🔍 Checking local services..."

# Check Ollama
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✅ Ollama is running locally"
else
    echo "❌ Ollama is not running. Please start it first."
    exit 1
fi

# Check MongoDB
if nc -z localhost 27017 > /dev/null 2>&1; then
    echo "✅ MongoDB is running locally"
else
    echo "❌ MongoDB is not running. Please start it first."
    exit 1
fi

# Check MariaDB
if nc -z localhost 3306 > /dev/null 2>&1; then
    echo "✅ MariaDB is running locally"
else
    echo "❌ MariaDB is not running. Please start it first."
    exit 1
fi

# Load environment variables
if [ -f .env.local ]; then
    echo "📁 Loading .env.local configuration..."
    export $(cat .env.local | grep -v '^#' | xargs)
else
    echo "⚠️  .env.local not found, using defaults"
fi

# Start only the dataanalyzer service
echo "🐳 Starting Data Analyzer container..."
docker-compose -f docker-compose.local.yml up -d dataanalyzer

# Optional: Start additional services
read -p "🤔 Do you want to start Jupyter Notebook? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📊 Starting Jupyter Notebook..."
    docker-compose -f docker-compose.local.yml --profile jupyter up -d jupyter
fi

read -p "🤔 Do you want to start Metabase? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📈 Starting Metabase..."
    docker-compose -f docker-compose.local.yml --profile metabase up -d metabase
fi

echo "🎉 Services started successfully!"
echo ""
echo "🌐 Access points:"
echo "   - Data Analyzer: http://localhost:7860"
echo "   - Jupyter (if started): http://localhost:8888 (Token: dataanalyzer123)"
echo "   - Metabase (if started): http://localhost:3000"
echo ""
echo "📋 To check logs: docker-compose -f docker-compose.local.yml logs -f dataanalyzer"
echo "🛑 To stop: docker-compose -f docker-compose.local.yml down"
