FROM python:3.10-slim

WORKDIR /app

# System libraries required at runtime:
#   libglib2.0-0  — GLib, needed by opencv-python-headless
#   libgl1        — OpenGL stub, needed by some opencv operations even in headless mode
#   libgomp1      — OpenMP, needed by numpy, scikit-image, and tensorflow-cpu
# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies before copying source so this layer is cached
# on code-only changes. Note: tensorflow-cpu adds ~1 GB to the image size.
COPY requirements.txt .
# hadolint ignore=DL3013
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Run as non-root for security
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501"]
