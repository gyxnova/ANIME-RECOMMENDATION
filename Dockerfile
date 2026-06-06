FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

# install dependencies first to leverage layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy only application code (exclude models, data, notebooks via .dockerignore)
COPY app.py .
COPY app.yaml .
COPY src/ ./src/

# expose Hugging Face port
EXPOSE 7860

# start server
CMD ["uvicorn", "src.models.api:app", "--host", "0.0.0.0", "--port", "7860"]