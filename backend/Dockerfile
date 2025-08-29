FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    PATH="/root/.local/bin:/root/.cargo/bin:${PATH}"

WORKDIR /app

# System deps for installer and TLS
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv (https://astral.sh/uv)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh -s -- -y

# Cache deps layer
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

# Copy app sources
COPY tennis_scrapper ./tennis_scrapper
COPY conf ./conf
COPY resources ./resources
COPY script ./script

RUN chmod +x script/run_all.sh

CMD [ "bash", "script/run_all.sh" ]