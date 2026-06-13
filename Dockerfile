FROM python:3.12-slim

# Install system utilities and FFmpeg video processing binaries
RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set up a non-root user to satisfy Hugging Face security protocols
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy dependencies first for better layer caching
COPY --chown=user multimodal-engine/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire multimodal-engine folder
COPY --chown=user multimodal-engine/ .

# Hugging Face Spaces requires port 7860
EXPOSE 7860

# Launch Streamlit
ENTRYPOINT ["streamlit", "run", "app/app.py", "--server.port=7860", "--server.address=0.0.0.0"]
