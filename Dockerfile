FROM python:3.13

RUN apt update && python3 -m pip install --upgrade pip && apt install -y \
    tesseract-ocr \
    poppler-utils \
    libgl1

WORKDIR /code

COPY dist/*.whl /code
RUN pip install -U /code/*.whl && rm -f /code/*.whl

CMD ["ingest_watch"]
