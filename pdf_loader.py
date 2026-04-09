import io
import re
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from unstructured.partition.pdf import partition_pdf

from config import DATA_DIR

try:
    import fitz
except ImportError:  # pragma: no cover
    fitz = None

try:
    import pytesseract
except ImportError:  # pragma: no cover
    pytesseract = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None


TEXT_ELEMENT_TYPES = {
    "Title",
    "NarrativeText",
    "ListItem",
    "Text",
    "Header",
    "Footer",
    "UncategorizedText",
}
TABLE_ELEMENT_TYPES = {"Table"}
IMAGE_ELEMENT_TYPES = {"Image", "Figure"}
TEXT_MIN_CHARS = 80
IMAGE_TEXT_MIN_CHARS = 20


def clean_text(text: str) -> str:
    text = text or ""
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\x00", " ")
    return text.strip()


def _metadata_attr(metadata, attr: str, default=None):
    return getattr(metadata, attr, default) if metadata is not None else default


def _build_text_documents(elements, filename: str) -> list[Document]:
    documents = []

    for index, element in enumerate(elements):
        category = getattr(element, "category", "Text")
        if category not in TEXT_ELEMENT_TYPES:
            continue

        text = clean_text(getattr(element, "text", ""))
        if len(text) < TEXT_MIN_CHARS:
            continue

        metadata = {
            "source": filename,
            "source_pdf": filename,
            "page": _metadata_attr(getattr(element, "metadata", None), "page_number"),
            "content_type": "text",
            "element_type": category,
            "element_id": index,
        }
        documents.append(Document(page_content=text, metadata=metadata))

    return documents


def _build_table_documents(elements, filename: str) -> list[Document]:
    documents = []

    for index, element in enumerate(elements):
        category = getattr(element, "category", "")
        if category not in TABLE_ELEMENT_TYPES:
            continue

        raw_text = clean_text(getattr(element, "text", ""))
        html_text = clean_text(_metadata_attr(getattr(element, "metadata", None), "text_as_html", ""))
        table_text = raw_text or html_text

        if not table_text:
            continue

        page = _metadata_attr(getattr(element, "metadata", None), "page_number")
        table_id = f"{Path(filename).stem}_table_{index}"
        prefix = f"Table {table_id} from {filename}"
        if page:
            prefix += f", page {page}"

        metadata = {
            "source": filename,
            "source_pdf": filename,
            "page": page,
            "content_type": "table",
            "element_type": category,
            "table_id": table_id,
            "element_id": index,
        }
        documents.append(
            Document(
                page_content=f"{prefix}\n\n{table_text}",
                metadata=metadata,
            )
        )

    return documents


def _extract_image_documents(pdf_path: str, filename: str) -> list[Document]:
    if fitz is None or Image is None:
        return []

    output_root = DATA_DIR / "extracted_images" / Path(filename).stem
    output_root.mkdir(parents=True, exist_ok=True)

    documents = []
    pdf = fitz.open(pdf_path)

    try:
        for page_index, page in enumerate(pdf, start=1):
            for image_index, image in enumerate(page.get_images(full=True), start=1):
                xref = image[0]
                base_image = pdf.extract_image(xref)
                image_bytes = base_image["image"]
                extension = base_image.get("ext", "png")
                image_name = f"{Path(filename).stem}_p{page_index}_img{image_index}.{extension}"
                image_path = output_root / image_name

                pil_image = Image.open(io.BytesIO(image_bytes))
                pil_image.save(image_path)

                ocr_text = ""
                if pytesseract is not None:
                    try:
                        ocr_text = clean_text(pytesseract.image_to_string(pil_image, lang="eng"))
                    except Exception:
                        ocr_text = ""

                if len(ocr_text) < IMAGE_TEXT_MIN_CHARS:
                    ocr_text = (
                        f"Figure/image extracted from {filename}, page {page_index}. "
                        f"Image file: {image_name}."
                    )

                metadata = {
                    "source": filename,
                    "source_pdf": filename,
                    "page": page_index,
                    "content_type": "image",
                    "element_type": "Image",
                    "image_id": f"{Path(filename).stem}_p{page_index}_img{image_index}",
                    "image_path": str(image_path),
                }
                documents.append(Document(page_content=ocr_text, metadata=metadata))
    finally:
        pdf.close()

    return documents


def load_pdf_documents(pdf_path: str, filename: str) -> List[Document]:
    """
    Load a PDF into multimodal LangChain documents.
    Returns separate text, table, and image documents with modality metadata.
    """
    elements = partition_pdf(
        filename=pdf_path,
        strategy="hi_res",
        infer_table_structure=True,
    )

    text_docs = _build_text_documents(elements, filename)
    table_docs = _build_table_documents(elements, filename)
    image_docs = _extract_image_documents(pdf_path, filename)

    figure_docs = []
    for index, element in enumerate(elements):
        category = getattr(element, "category", "")
        if category not in IMAGE_ELEMENT_TYPES:
            continue

        figure_text = clean_text(getattr(element, "text", ""))
        if len(figure_text) < IMAGE_TEXT_MIN_CHARS:
            continue

        metadata = {
            "source": filename,
            "source_pdf": filename,
            "page": _metadata_attr(getattr(element, "metadata", None), "page_number"),
            "content_type": "image",
            "element_type": category,
            "image_id": f"{Path(filename).stem}_figure_{index}",
        }
        figure_docs.append(Document(page_content=figure_text, metadata=metadata))

    return text_docs + table_docs + image_docs + figure_docs
