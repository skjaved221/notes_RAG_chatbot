from pathlib import Path

import fitz  # PyMuPDF
import pytesseract
from PIL import Image, ImageOps

from llama_index.core import Document, SimpleDirectoryReader


# Change this only if Tesseract is installed somewhere else
TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

if Path(TESSERACT_EXE).exists():
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
PDF_EXTENSIONS = {".pdf"}
TEXT_EXTENSIONS = {".txt", ".md"}
OFFICE_EXTENSIONS = {".docx", ".pptx"}

SUPPORTED_EXTENSIONS = (
    IMAGE_EXTENSIONS
    | PDF_EXTENSIONS
    | TEXT_EXTENSIONS
    | OFFICE_EXTENSIONS
)


def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Improves OCR accuracy for screenshots, scanned notes, and PDF slide images.
    """
    image = image.convert("L")
    image = ImageOps.autocontrast(image)
    return image


def image_to_text(image: Image.Image) -> str:
    image = preprocess_image(image)

    # psm 6 = assume a block of text. Good for notes, slides, screenshots.
    return pytesseract.image_to_string(image, config="--psm 6")


def load_image_file(file_path: Path) -> list[Document]:
    try:
        image = Image.open(file_path)
        text = image_to_text(image)
    except Exception as e:
        text = f"OCR failed for image {file_path.name}: {e}"

    if not text.strip():
        text = "[No readable text found in this image.]"

    return [
        Document(
            text=text,
            metadata={
                "file_name": file_path.name,
                "file_path": str(file_path),
                "source_type": "image_ocr",
            },
        )
    ]


def load_pdf_with_ocr(file_path: Path) -> list[Document]:
    documents = []

    pdf = fitz.open(file_path)

    for page_index, page in enumerate(pdf, start=1):
        # Normal selectable PDF text
        normal_text = page.get_text("text") or ""

        # OCR text from page image
        ocr_text = ""

        try:
            # Higher zoom = better OCR, but slower.
            zoom = 3.0
            matrix = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=matrix, alpha=False)

            image = Image.frombytes(
                "RGB",
                [pix.width, pix.height],
                pix.samples,
            )

            ocr_text = image_to_text(image)

        except Exception as e:
            ocr_text = f"OCR failed on page {page_index}: {e}"

        combined_text = f"""
PDF file: {file_path.name}
Page: {page_index}

Selectable PDF text:
{normal_text}

OCR text from page image:
{ocr_text}
"""

        if combined_text.strip():
            documents.append(
                Document(
                    text=combined_text,
                    metadata={
                        "file_name": file_path.name,
                        "file_path": str(file_path),
                        "page_label": str(page_index),
                        "source_type": "pdf_text_plus_ocr",
                    },
                )
            )

    return documents


def load_text_file(file_path: Path) -> list[Document]:
    text = file_path.read_text(encoding="utf-8", errors="ignore")

    return [
        Document(
            text=text,
            metadata={
                "file_name": file_path.name,
                "file_path": str(file_path),
                "source_type": "text",
            },
        )
    ]


def load_office_files(files: list[Path]) -> list[Document]:
    if not files:
        return []

    return SimpleDirectoryReader(
        input_files=[str(file) for file in files],
    ).load_data()


def load_notes_with_ocr(data_dir: str | Path) -> list[Document]:
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)

    documents = []
    office_files = []

    for file_path in data_dir.rglob("*"):
        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()

        if suffix not in SUPPORTED_EXTENSIONS:
            continue

        print(f"Reading: {file_path}")

        if suffix in PDF_EXTENSIONS:
            documents.extend(load_pdf_with_ocr(file_path))

        elif suffix in IMAGE_EXTENSIONS:
            documents.extend(load_image_file(file_path))

        elif suffix in TEXT_EXTENSIONS:
            documents.extend(load_text_file(file_path))

        elif suffix in OFFICE_EXTENSIONS:
            office_files.append(file_path)

    documents.extend(load_office_files(office_files))

    return documents