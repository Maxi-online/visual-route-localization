FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VLOC_DEVICE=cuda

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt

RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio \
    && pip install -r requirements.txt \
    && pip install "git+https://github.com/cvg/LightGlue.git"

COPY app ./app
COPY web ./web
COPY README.md ./README.md
COPY VERSION.txt ./VERSION.txt

EXPOSE 8002

CMD ["python", "-m", "app.docker_entrypoint"]
