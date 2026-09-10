# Voyage Analytics - Streamlit app serving flight price, hotel recommendation,
# and classification models. Build context is the repo root (Voyage-Analytics/).

FROM python:3.11-slim

# Prevents Python from writing .pyc files and buffers stdout (cleaner container logs)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# curl is needed for the HEALTHCHECK below; not present in the slim base image by default
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first so this layer is cached unless requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code + models + data.
# NOTE: this bakes app/models/*.pkl and app/data/*.csv into the image itself.
# That's the simplest path for free hosts (Streamlit Cloud, HF Spaces, Render) since
# they don't give free persistent volumes -- if your models grow large enough that
# baking them in becomes impractical, switch to pulling them from external storage
# (e.g. S3/GCS) in an entrypoint script instead.
COPY app/ ./app/

EXPOSE 8501

# Basic container healthcheck so orchestrators (Render, Fly.io, etc.) know when it's ready
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

WORKDIR /app/app
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]