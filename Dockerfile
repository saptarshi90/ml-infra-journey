FROM python:3.10-slim
WORKDIR /app
RUN pip install --no-cache-dir fastapi uvicorn pika requests prometheus_client
COPY *.py ./
EXPOSE 80
CMD ["uvicorn", "metric_logger_v2:app", "--host", "0.0.0.0", "--port", "80"]
