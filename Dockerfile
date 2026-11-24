FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY pyproject.toml README.md ./
COPY pyharness ./pyharness
COPY skills ./skills
COPY policies ./policies

RUN pip install --no-cache-dir -e ".[postgres]" \
    && useradd --uid 10001 --create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/data \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000
CMD ["uvicorn", "pyharness.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
