FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

# install dependencies first to leverage layer caching
COPY requirements.txt .
COPY setup.py .
# copy source package so editable install (-e .) works
COPY src/ ./src/
RUN pip install --no-cache-dir -r requirements.txt

# copy only application entry files
COPY app.py .
COPY app.yaml .
COPY configs/ ./configs/

# expose Hugging Face port
EXPOSE 7860

# start server
CMD ["uvicorn", "src.models.api:app", "--host", "0.0.0.0", "--port", "7860"]