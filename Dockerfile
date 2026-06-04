FROM python:3.11-slim

WORKDIR /app

# System libraries required by Pillow and scipy
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch first to avoid pulling the CUDA variant (~3 GB)
RUN pip install --no-cache-dir \
    torch>=2.0 torchvision>=0.15 \
    --index-url https://download.pytorch.org/whl/cpu

# Remaining dependencies (torch already satisfied, pip will skip it)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python", "app/main.py"]
