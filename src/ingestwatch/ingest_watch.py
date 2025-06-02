import time
import hashlib
import mimetypes
from pathlib import Path
import uuid

import fitz  # PyMuPDF for PDFs
import docx
import openpyxl
import pytesseract
from PIL import Image
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    PointStruct,
    Distance,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)

from .config import config


qdrant = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
openai = OpenAI(api_key=config.OPENAI_API_KEY)

# === INIT VECTOR COLLECTION ===
def init_collection():
    if not qdrant.collection_exists(config.COLLECTION_NAME):
        qdrant.recreate_collection(
            collection_name=config.COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )


# === TEXT EXTRACTION ===
def extract_text(file_path: Path) -> str:
    mime, _ = mimetypes.guess_type(file_path)
    ext = file_path.suffix.lower()

    try:
        if ext == ".pdf":
            text = ""
            with fitz.open(file_path) as doc:
                for page in doc:
                    content = page.get_text()
                    if content.strip():
                        text += content
                    else:
                        # OCR fallback
                        pix = page.get_pixmap()
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        text += pytesseract.image_to_string(img)
            return text

        elif ext == ".docx":
            return "\n".join([p.text for p in docx.Document(file_path).paragraphs])

        elif ext == ".xlsx":
            wb = openpyxl.load_workbook(file_path, data_only=True)
            text = ""
            for sheet in wb.worksheets:
                for row in sheet.iter_rows(values_only=True):
                    text += " ".join(str(cell or "") for cell in row) + "\n"
            return text

        elif ext in [".md", ".txt"]:
            return file_path.read_text()

    except Exception as e:
        print(f"[ERROR] Failed to parse {file_path}: {e}")
    return ""


# === TEXT CHUNKING ===
def chunk_text(text: str):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + config.CHUNK_SIZE, len(text))
        chunks.append(text[start:end])
        start += config.CHUNK_SIZE - config.CHUNK_OVERLAP
    return chunks


# === HASH FOR DOC TRACKING ===
def file_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


# === INDEX FILE ===
def index_file(path: Path):
    print(f"[INDEX] {path}")
    # doc_id = file_hash(path)
    text = extract_text(path)
    chunks = chunk_text(text)

    points = []
    for i, chunk in enumerate(chunks):
        embedding = (
            openai.embeddings.create(input=chunk, model=config.EMBEDDING_MODEL).data[0].embedding
        )
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                payload={"doc_path": str(path), "chunk_index": i, "text": chunk},
                vector=embedding,
            )
        )

    # Remove previous vectors
    qdrant.delete(
        collection_name=config.COLLECTION_NAME,
        points_selector=Filter(
            must=[FieldCondition(key="doc_path", match=MatchValue(value=str(path)))]
        ),
    )
    qdrant.upsert(collection_name=config.COLLECTION_NAME, points=points)


# === REMOVE FILE ===
def remove_file(path: Path):
    print(f"[REMOVE] {path}")
    qdrant.delete(
        collection_name=config.COLLECTION_NAME,
        points_selector=Filter(
            must=[FieldCondition(key="doc_path", match=MatchValue(value=str(path)))]
        ),
    )


# === FILE WATCHER ===
class DocsHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            path = Path(event.src_path)
            index_file(path)

    def on_deleted(self, event):
        if not event.is_directory:
            path = Path(event.src_path)
            remove_file(path)

    def on_modified(self, event):
        if not event.is_directory:
            path = Path(event.src_path)
            index_file(path)


# === MAIN ===
def initial_index():
    for path in config.DOCS_PATH.glob("**/*"):
        if path.is_file():
            index_file(path)


def main():
    print("[START] Initial indexing...")
    init_collection()
    initial_index()
    print("[WATCHING] Waiting for changes...")

    event_handler = DocsHandler()
    observer = Observer()
    observer.schedule(event_handler, path=config.DOCS_PATH, recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


# if __name__ == "__main__":
#     main()
