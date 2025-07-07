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
    default-libmysqlclient-dev \
    libmariadb-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster Python package management
RUN pip install uv

# Copy project configuration files first
COPY pyproject.toml ./
COPY uv.lock ./

# Initialize project and install dependencies
RUN uv venv && \
    uv sync --frozen

# Set virtual environment path
ENV PATH="/app/.venv/bin:$PATH"

# Copy all project files
COPY . .

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

# Activate virtual environment and run the enhanced frontend
CMD ["/app/.venv/bin/python", "enhanced_frontend.py"]
