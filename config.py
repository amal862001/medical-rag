from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT
SOURCE_PDF_DIR = DATA_DIR / "raw_pdfs"
CHROMA_DIR = DATA_DIR / "chroma1"

COLLECTION_NAME = "chunk_store"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150
