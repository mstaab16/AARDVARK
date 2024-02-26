# syntax=docker/dockerfile:1

# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/go/dockerfile-reference/

# ARG PYTHON_VERSION=3.10
# FROM python:${PYTHON_VERSION}-slim as base
from nvcr.io/nvidia/rapidsai/base:24.02-cuda12.0-py3.10 as base

USER root

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/go/dockerfile-user-best-practices/
# ARG UID=10001
# RUN adduser \
#     --disabled-password \
#     --gecos "" \
#     --home "/nonexistent" \
#     --shell "/sbin/nologin" \
#     --no-create-home \
#     --uid "${UID}" \
#     appuser

# Download dependencies as a separate step to take advantage of Docker's caching.
# Leverage a cache mount to /root/.cache/pip to speed up subsequent builds.
# Leverage a bind mount to requirements.txt to avoid having to copy them into
# into this layer.
# RUN --mount=type=cache,target=/root/.cache/pip \
#     --mount=type=bind,source=celery_requirements.txt,target=celery_requirements.txt \ 

# RUN --mount=type=cache,target=/root/.cache/pip \
#     --mount=type=bind,source=celery_requirements.txt,target=celery_requirements.txt \ 
#     python -m pip install \
#     --extra-index-url=https://pypi.nvidia.com \
#     cuml-cu12==24.2.*
    # cudf-cu12==24.2.* dask-cudf-cu12==24.2.* cuml-cu12==24.2.* \
    # cugraph-cu12==24.2.* cuspatial-cu12==24.2.* cuproj-cu12==24.2.* \
    # cuxfilter-cu12==24.2.* cucim-cu12==24.2.* pylibraft-cu12==24.2.* \
    # raft-dask-cu12==24.2.*

# RUN --mount=type=bind,source=celery_requirements.txt,target=celery_requirements.txt \ 
#     python -m pip install torch

RUN --mount=type=bind,source=celery_requirements.txt,target=celery_requirements.txt \ 
    python -m pip install -r celery_requirements.txt

# Switch to the non-privileged user to run the application.
# USER appuser

# Copy the source code into the container.
# COPY . .

# Expose the port that the application listens on.
EXPOSE 80

# Run the application.
# CMD uvicorn 'app.main:app' --host=0.0.0.0 --port=80 --reload
