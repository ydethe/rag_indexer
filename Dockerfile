# Stage 1: Build
FROM python:3.13-slim AS builder

RUN apt update && apt install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-fra \
    poppler-utils \
    libgl1 \
    fonts-freefont-ttf

WORKDIR /app

RUN python3 -m venv venv --system-site-packages && \
    ./venv/bin/python -m pip install --no-cache-dir -U torch --index-url https://download.pytorch.org/whl/cpu && \
    find /app/venv \( -type d -a -name test -o -name tests \) -o \( -type f -a -name '*.pyc' -o -name '*.pyo' \) -exec rm -rf '{}' \+

COPY requirements.txt .
RUN ./venv/bin/python -m pip install --no-cache-dir -U -r requirements.txt && \
    find /app/venv \( -type d -a -name test -o -name tests \) -o \( -type f -a -name '*.pyc' -o -name '*.pyo' \) -exec rm -rf '{}' \+

COPY dist/*.whl .
RUN ./venv/bin/python -m pip install --no-cache-dir *.whl && \
    rm -f *.whl && \
    find /app/venv \( -type d -a -name test -o -name tests \) -o \( -type f -a -name '*.pyc' -o -name '*.pyo' \) -exec rm -rf '{}' \+

# # Stage 2: Production
# FROM python:3.13-slim

# # Set the working directory
# WORKDIR /app

# # Copy only the necessary files from the build stage
# COPY --from=builder /app /app

# Expose the port the app will run on
EXPOSE 7860

CMD ["/app/venv/bin/python", "-m" , "ingestwatch"]
