FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY demo/requirements.txt /tmp/demo-requirements.txt
RUN pip install -r /tmp/demo-requirements.txt
# The lightweight image serves the UI and reports missing inference assets honestly.
# Select an inference image only after checking the VPS hardware and cache files.
COPY demo /app/demo
COPY src /app/src
COPY .streamlit /app/.streamlit
RUN useradd --create-home --uid 10001 neurosteer \
    && mkdir -p /app/results/demo/demo_v1 \
    && chown -R neurosteer:neurosteer /app
USER neurosteer
CMD ["python", "-m", "uvicorn", "demo.backend.main_fastapp_backend:app", "--host", "0.0.0.0", "--port", "8000"]
