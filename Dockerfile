FROM python:3.11-slim

WORKDIR /app

# System libraries required by Pillow and scipy
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

# Suppress any CUDA memory-caching overhead if torch ever enters the image transitively
ENV PYTORCH_NO_CUDA_MEMORY_CACHING=1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
