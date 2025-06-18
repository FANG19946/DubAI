# Start with a lightweight Python 3.10 base
FROM python:3.10-slim

# Install system-level dependencies
RUN apt-get update && apt-get install -y \
    git wget parallel cmake build-essential \
    libprotobuf-dev protobuf-compiler \
    && apt-get clean

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip && \
    pip install sentencepiece

# Set working directory inside the container
WORKDIR /app/IndicTrans2

# Copy IndicTrans2 source code (submodule) into the container
COPY IndicTrans2 /app/IndicTrans2

# Make install.sh executable and run it
RUN chmod +x install.sh && ./install.sh

# Default command when container starts
CMD ["bash"]
