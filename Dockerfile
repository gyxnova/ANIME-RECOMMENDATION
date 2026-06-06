FROM python:3.11-slim

WORKDIR /app

# install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy everything
COPY . .

# expose HuggingFace port
EXPOSE 7860

# start server
CMD ["uvicorn", "src.models.api:app", "--host", "0.0.0.0", "--port", "7860"]