# Stage 1: Build
FROM python:3.13 AS builder

RUN apt update && python3 -m pip install --upgrade pip && apt install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-fra \
    poppler-utils \
    libgl1

WORKDIR /code

COPY requirements.txt .
RUN python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt

COPY dist/*.whl .
RUN .venv/bin/python -m pip install -U *.whl && rm -f *.whl

# Stage 2: Production
FROM python:3.13

# Set the working directory
WORKDIR /code

# Copy only the necessary files from the build stage
COPY --from=builder . .

# Expose the port the app will run on
EXPOSE 7860

CMD [ ".venv/bin/python3", "-m" , "ingestwatch"]
