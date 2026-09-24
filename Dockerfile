# Chess Arena web app and API. Stockfish comes from Debian; models and games
# persist in the /data volume.
FROM python:3.12-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends stockfish \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:${PATH}" \
    CHESS_ARENA_HOME=/data \
    CHESS_ARENA_STOCKFISH=/usr/games/stockfish \
    CHESS_ARENA_HOST=0.0.0.0
VOLUME ["/data"]
EXPOSE 8744
CMD ["chess-arena", "serve"]
