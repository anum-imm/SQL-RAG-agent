# ========================
# Backend - FastAPI + SQL+RAG
# ========================
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Prevent apt interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies in one step (update + install)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsqlite3-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy only requirements.txt for dependency caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code, database, and FAISS index
COPY . .

# Validate FAISS and DB exist
RUN test -f Chinook.db || (echo "❌ Missing Chinook.db in backend folder" && exit 1)
RUN test -d faiss_index || (echo "❌ Missing faiss_index directory in backend folder" && exit 1)

# Expose FastAPI port
EXPOSE 8000

# Run backend app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
