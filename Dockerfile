# Base image
ARG PYTHON_VERSION="3.11"
FROM python:${PYTHON_VERSION}

# Set environment variables for non-interactive installs
ENV DEBIAN_FRONTEND=noninteractive

# Update and install essential packages
RUN apt update -y && apt install -y --no-install-recommends git && apt clean && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/RokasEl/MACE-Models /workspace/MACE-Models
RUN cd /workspace/MACE-Models && pip install . && dvc pull

RUN pip install zndraw
RUN pip install git+https://github.com/RokasEl/hydromace.git@develop

COPY ./ /workspace/simgen
RUN cd /workspace/simgen pip install .
WORKDIR /workspace/MACE-Models

CMD [ "simgen", "connect" ]
