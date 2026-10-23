FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /srv

# CPU-only torch. Dependencies come from PyPI; the PyTorch index alone cannot build them.
RUN pip install torch==2.14.1+cpu \
        --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY facerestore facerestore
COPY scripts/fetch_weights.py scripts/convert_legacy.py scripts/
# Weights already in ./weights are used as-is; the rest is downloaded and checksummed.
COPY weights weights
ARG WEIGHTS="detector facerestore upstream"
RUN python scripts/fetch_weights.py --dest weights --only ${WEIGHTS}
COPY app app

RUN useradd --system --uid 10001 facerestore
USER facerestore
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')"
CMD ["gunicorn", "app.server:create_app()", "--bind", "0.0.0.0:8000", \
     "--workers", "1", "--threads", "4", "--timeout", "120"]
