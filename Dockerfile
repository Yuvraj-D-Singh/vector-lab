FROM node:20-alpine AS client-build

WORKDIR /app/client
COPY client/package.json client/package-lock.json ./
RUN npm ci
COPY client/ ./
RUN npm run build

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY server/requirements.txt ./server/requirements.txt
RUN HNSWLIB_NO_NATIVE=1 pip install --no-cache-dir -r server/requirements.txt

COPY server/app ./server/app
COPY --from=client-build /app/client/dist ./client/dist

CMD ["sh", "-c", "uvicorn app.main:app --app-dir server --host 0.0.0.0 --port ${PORT:-8000}"]
