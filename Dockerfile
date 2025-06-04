FROM python:3.13

RUN apt update && python3 -m pip install --upgrade pip && apt install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-fra \
    poppler-utils \
    libgl1 \
    python3-nltk

WORKDIR /code

COPY dist/*.whl /code
COPY requirements.txt /code
RUN pip install -r requirements.txt
RUN pip install -U /code/*.whl && rm -f /code/*.whl

CMD ["ingestwatch"]
