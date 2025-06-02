FROM python:3.13-alpine

RUN apt update && apt install -y \
    tesseract-ocr \
    poppler-utils \
    libgl1

WORKDIR /code

COPY dist/*.whl /code
RUN pip install -U /code/*.whl && rm -f /code/*.whl

CMD ["ingest_watch"]
