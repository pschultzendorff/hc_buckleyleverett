FROM python:3.12-slim
LABEL title="hc_buckleyleverett"
LABEL description="Docker image to reproduce results in 'Efficient design of continuation methods for hyperbolic transport problems in porous media"
LABEL version="0.1"
LABEL maintainer="Peter von Schultzendorff"
LABEL email="peter.schultzendorff@uib.no"

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Copy the repository files into the container
COPY . .

# Install the package
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e .

# Matplotlib backend for .png files
ENV MPLBACKEND=Agg

CMD ["python", "scripts/buckley_leverett/viscous.py"]
