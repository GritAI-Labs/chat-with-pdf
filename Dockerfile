# Chat with your PDF — runs on Hugging Face Spaces, Render, Railway, Fly.io, etc.
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# fastembed caches its model here; keep it writable
ENV FASTEMBED_CACHE_PATH=/app/.fastembed_cache
ENV PORT=7860
EXPOSE 7860
CMD ["python", "app.py"]
