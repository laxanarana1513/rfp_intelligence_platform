# Streamlit only calls the API, so it does not need Docling or PyTorch.
FROM python:3.12-slim-bookworm AS ui

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY ui ./ui

RUN pip install --no-cache-dir \
        "streamlit>=1.40" \
        "httpx>=0.28" \
        "python-dotenv>=1.0" \
    && rm -rf /root/.cache

EXPOSE 8501

CMD ["streamlit", "run", "ui/streamlit_app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]


# API image. The CPU PyTorch index avoids the multi-gigabyte CUDA wheels.
FROM python:3.12-slim-bookworm AS api

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src:/app \
    HF_HOME=/cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && python -c "import importlib.metadata as m; open('/tmp/constraints.txt','w').write(''.join(f'{name}=={m.version(name)}\n' for name in ('torch', 'torchvision')))" \
    && grep -Ev '^(pytest|pandas)([<>=]|$)' requirements.txt | tr -d '\r' > /tmp/requirements.txt \
    && pip install --no-cache-dir -c /tmp/constraints.txt -r /tmp/requirements.txt \
    && python -c "import torch; version=torch.__version__; assert '+cu' not in version, version; print('torch', version)" \
    && rm -rf /tmp/constraints.txt /tmp/requirements.txt /root/.cache \
    && find /usr/local/lib/python3.12/site-packages -type d -name '__pycache__' -prune -exec rm -rf {} +

COPY src ./src
COPY api ./api
COPY migrations ./migrations
COPY main.py ./

EXPOSE 8000

CMD ["sh", "-c", "python -m rfp_intel.db.migrate && exec python main.py serve --host 0.0.0.0 --port 8000"]
