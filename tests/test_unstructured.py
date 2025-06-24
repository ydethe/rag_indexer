from pathlib import Path

from unstructured.partition.pdf import partition_pdf


def test_unstructured_pdf_loader():
    pdf_pth = Path("tests/inputs/docs/Marina Robledo.pdf")
    elements = partition_pdf(filename=pdf_pth, infer_table_structure=True, languages=["fra", "eng"])
    print("\n\n".join([str(el) for el in elements]))


if __name__ == "__main__":
    test_unstructured_pdf_loader()
