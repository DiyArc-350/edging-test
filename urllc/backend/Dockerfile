# Use a slim Python footprint as our foundation
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install light decoding libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# STABLE ENDPOINT: Downloads the exact nanodet-plus-m_320.onnx model structure
RUN curl -L -o nanodet-plus-m_320.onnx "https://sourceforge.net/projects/nanodet-plus.mirror/files/v1.0.0-alpha-1/nanodet-plus-m_320.onnx/download"

COPY . .

EXPOSE 80

CMD ["python", "main.py"]
