FROM python:3.13-slim

# Setup
USER root
RUN useradd -ms /bin/bash jupyteruser
WORKDIR /home/jupyteruser/work
COPY requirements.txt .
COPY scripts/getdata.py .

# Install
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && apt-get autoremove --purge && apt-get clean \
    && pip install --no-cache-dir -r requirements.txt \
    && chown -R jupyteruser:jupyteruser /home/jupyteruser

USER jupyteruser
RUN python /home/jupyteruser/work/getdata.py

# Run Jupyter
EXPOSE 8888
CMD ["bash", "-c", "set -e && \
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser && \
#rm -rf /home/jupyteruser/work/data && \
find /home/jupyteruser/work -type d -name '__pycache__' -exec rm -rf {} + && \
find /home/jupyteruser/work -type d -name '.ipynb_checkpoints' -exec rm -rf {} +"]
