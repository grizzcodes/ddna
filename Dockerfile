FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install package
RUN pip install -e .

# Create necessary directories
RUN mkdir -p /app/models /app/output /app/logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEVICE=cuda

# Default command
CMD ["python", "-m", "ddna.cli", "--help"]