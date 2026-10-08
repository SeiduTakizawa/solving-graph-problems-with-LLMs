# Sandbox image for run_python: slim Python plus the packages the graph tools need, nothing else.
# Versions are pinned to the project's (uv.lock), so tools behave the same inside and outside.
# The harness code is NOT copied in: it is mounted read-only at run time (only harness/, never eval/ or results/).
FROM python:3.12-slim

RUN pip install --no-cache-dir \
        networkx==3.7 \
        pydantic==2.13.5 \
        pydantic-core==2.46.5 \
        annotated-types==0.8.0 \
        typing-extensions==4.16.0 \
        typing-inspection==0.4.4 \
    && useradd --uid 1000 --create-home sandbox

USER sandbox
WORKDIR /tmp
ENV PYTHONPATH=/opt PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
