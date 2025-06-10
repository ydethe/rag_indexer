import pytesseract
import layoutparser as lp
from pdf2image import convert_from_path
import cv2
import numpy as np
from PIL import Image


# pip install pdfminer.six pytesseract pdf2image layoutparser[ocr] opencv-python-headless numpy Pillow
# sudo apt-get install tesseract-ocr poppler-utils  # for Linux


# Setup LayoutParser model (CPU-friendly)
model = lp.Detectron2LayoutModel(
    config_path="lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
    model_path=None,
    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
    label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
    use_gpu=False,
)


def remove_light_gray(img):
    """Remove very light gray watermarks using thresholding."""
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)  # Tune threshold as needed
    return Image.fromarray(thresh)


def extract_clean_text_from_pdf(pdf_path):
    images = convert_from_path(pdf_path, dpi=300)
    all_text = []

    for idx, img in enumerate(images):
        print(f"Processing page {idx + 1}")
        cleaned_img = remove_light_gray(img)

        # Detect layout
        layout = model.detect(cleaned_img)

        # Filter only text blocks
        text_blocks = [b for b in layout if b.type == "Text" or b.type == "Title"]

        # Remove headers and footers (top/bottom 10% of the page)
        height = cleaned_img.height
        margin = 0.1 * height
        filtered_blocks = [
            b for b in text_blocks if b.block.y_1 > margin and b.block.y_2 < (height - margin)
        ]

        # Sort top to bottom, left to right
        filtered_blocks.sort(key=lambda b: (b.block.y_1, b.block.x_1))

        page_text = ""
        for block in filtered_blocks:
            segment = cleaned_img.crop(block.block.to_tuple())
            text = pytesseract.image_to_string(segment, lang="eng")
            page_text += text.strip() + "\n"

        all_text.append(page_text)

    return "\n\n".join(all_text)


# Example use
pdf_path = "your_file.pdf"
text = extract_clean_text_from_pdf(pdf_path)
with open("clean_text_output.txt", "w", encoding="utf-8") as f:
    f.write(text)
