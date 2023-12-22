# Используйте официальный образ Python 3.8
FROM python:3.11.6

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py ./
COPY src ./src

ENV HOST_IP=0.0.0.0
ENV HOST_PORT=8080

CMD uvicorn main:app --host $HOST_IP --port $HOST_PORT