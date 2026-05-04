FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY pyproject.toml .

ENV PYTHONPATH=/app/backend:/app

EXPOSE 8000

CMD ["uvicorn", "methylcurate.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
