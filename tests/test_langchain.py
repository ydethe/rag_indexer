from pathlib import Path
from langchain.chat_models import init_chat_model
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_community.document_loaders.parsers import TesseractBlobParser

from ragindexer.config import config


def test_langchain_model():
    model = init_chat_model(
        model=config.OPEN_MODEL_PREF,
        model_provider="openai",
        api_key=config.OPEN_MODEL_API_KEY,
        base_url=config.OPEN_MODEL_ENDPOINT,
        temperature=config.OPEN_MODEL_TEMPERATURE,
    )

    print(model.invoke("Hello, world!"))


def test_langchain_pdf_loader():
    loader = PyMuPDF4LLMLoader(
        file_path=Path("tests/inputs/docs/Marina Robledo.pdf"),
        mode="page",
        extract_images=True,
        images_parser=TesseractBlobParser(),
        table_strategy="lines",
    )

    docs = loader.load()

    output_folder = Path("tests/output")
    output_folder.mkdir(exist_ok=True, parents=True)
    with open(output_folder / "Marina Robledo.md", "w") as f:
        for d in docs[:3]:
            f.write(d.page_content)


if __name__ == "__main__":
    # test_langchain_model()
    test_langchain_pdf_loader()
