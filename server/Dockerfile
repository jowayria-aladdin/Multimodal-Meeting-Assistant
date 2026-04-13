FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --timeout 300 --retries 10 \
    fastapi==0.111.0 \
    uvicorn==0.30.0 \
    python-multipart==0.0.9

RUN pip install --no-cache-dir --timeout 300 --retries 10 \
    tensorflow==2.17.0

RUN pip install --no-cache-dir --timeout 300 --retries 10 \
    mediapipe==0.10.14 \
    opencv-python-headless==4.10.0.84 \
    protobuf==4.25.3 \
    numpy==1.26.4

COPY server.py .
COPY sign_to_class.pkl .
COPY model/ ./model/

EXPOSE 5001

CMD ["python", "server.py"]