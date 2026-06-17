import re
import shutil
from pathlib import Path

import fitz  # PyMuPDF
import pytesseract
from PIL import Image, ImageOps


DATA_DIR = Path("data")
OCR_DIR = DATA_DIR / "_ocr_text"

TESSERACT_EXE = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")

if TESSERACT_EXE.exists():
    pytesseract.pytesseract.tesseract_cmd = str(TESSERACT_EXE)


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def preprocess_image(image: Image.Image) -> Image.Image:
    image = image.convert("L")
    image = ImageOps.autocontrast(image)
    return image


def ocr_image(image: Image.Image) -> str:
    image = preprocess_image(image)
    return pytesseract.image_to_string(image, config="--psm 6")


def process_pdf(pdf_path: Path):
    print(f"OCR PDF: {pdf_path.name}")

    pdf = fitz.open(pdf_path)

    for page_number, page in enumerate(pdf, start=1):
        selectable_text = page.get_text("text") or ""

        pix = page.get_pixmap(
            matrix=fitz.Matrix(3, 3),
            alpha=False,
        )

        image = Image.frombytes(
            "RGB",
            [pix.width, pix.height],
            pix.samples,
        )

        ocr_text = ocr_image(image)

        final_text = f"""
Source file: {pdf_path.name}
Page: {page_number}

Selectable PDF text:
{selectable_text}

OCR text from page image:
{ocr_text}
"""

        output_file = OCR_DIR / f"{safe_name(pdf_path.stem)}_page_{page_number}.txt"
        output_file.write_text(final_text, encoding="utf-8")


def process_image(image_path: Path):
    print(f"OCR Image: {image_path.name}")

    image = Image.open(image_path)
    ocr_text = ocr_image(image)

    final_text = f"""
Source image: {image_path.name}

OCR text from image:
{ocr_text}
"""

    output_file = OCR_DIR / f"{safe_name(image_path.stem)}.txt"
    output_file.write_text(final_text, encoding="utf-8")


def main():
    DATA_DIR.mkdir(exist_ok=True)

    if OCR_DIR.exists():
        shutil.rmtree(OCR_DIR)

    OCR_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = list(DATA_DIR.rglob("*.pdf"))
    image_files = []

    for ext in ["*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp", "*.tif", "*.tiff"]:
        image_files.extend(DATA_DIR.rglob(ext))

    for pdf_path in pdf_files:
        if OCR_DIR in pdf_path.parents:
            continue
        process_pdf(pdf_path)

    for image_path in image_files:
        if OCR_DIR in image_path.parents:
            continue
        process_image(image_path)

    txt_count = len(list(OCR_DIR.rglob("*.txt")))
    print(f"\nDone. Created {txt_count} OCR text files in: {OCR_DIR}")


if __name__ == "__main__":
    main()