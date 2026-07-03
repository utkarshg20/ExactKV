# CPU-only reproduction container for ExactKV research release (git tag v-release).
# GPU benchmark panels require a CUDA host + HF model access; not included here.
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /work

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    bash \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY exactkv ./exactkv
COPY scripts ./scripts
COPY tests ./tests
COPY reports ./reports
COPY paper ./paper
COPY site ./site
COPY docs/CLAIM_BOUNDARIES.md docs/METRIC_DEFINITIONS.md docs/EVALUATOR_GUIDE.md docs/VERSIONING.md ./docs/
COPY REPRODUCE.md ./

RUN pip install -U pip && pip install -e ".[dev]"

CMD ["bash", "-lc", "bash scripts/smoke_test.sh && python3 scripts/exactkv_repro.py --reports-only && python3 scripts/check_site_claims.py"]
