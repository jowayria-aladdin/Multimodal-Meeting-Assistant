FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    httpx \
    langgraph \
    pydantic \
    python-dotenv

COPY server/ ./server
COPY pipeline_scripts/ ./pipeline_scripts

WORKDIR /app/server

EXPOSE 5000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "5000"]