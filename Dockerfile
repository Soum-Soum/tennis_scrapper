FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
COPY uv.lock .
COPY tennis_scrapper .
COPY conf .
COPY resources .

RUN uv sync

