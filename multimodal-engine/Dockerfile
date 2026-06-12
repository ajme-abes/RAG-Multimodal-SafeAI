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

# Since app_dir enters the subfolder, local paths are relative to multimodal-engine/
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the specific app folder contents safely
COPY --chown=user . .

# Hugging Face Spaces strictly routes traffic over port 7860
EXPOSE 7860

# Launch Streamlit on the specified port
ENTRYPOINT ["streamlit", "run", "app/app.py", "--server.port=7860", "--server.address=0.0.0.0"]