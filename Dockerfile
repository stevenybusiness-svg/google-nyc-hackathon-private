FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

# Single worker (required for in-memory room state + WebSocket affinity).
# High concurrency limit lets asyncio handle many connections per worker.
# Timeout extended for long-running Veo video generation (up to 5 min).
CMD ["uvicorn", "server:app", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--workers", "1", \
     "--timeout-keep-alive", "120", \
     "--ws-max-size", "16777216", \
     "--limit-concurrency", "200"]
