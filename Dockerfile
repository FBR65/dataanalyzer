# Use Python 3.10 as base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster Python package management
RUN pip install uv

# Copy project files
COPY pyproject.toml uv.lock ./
COPY . .

# Install Python dependencies
RUN uv sync --frozen

# Create necessary directories
RUN mkdir -p /app/databases \
    /app/ducklake \
    /app/excel_files \
    /app/generated_code \
    /app/logs

# Set permissions
RUN chmod +x /app/enhanced_frontend.py

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Run the enhanced frontend
CMD ["uv", "run", "python", "enhanced_frontend.py"]
