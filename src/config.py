# src/config.py

from pathlib import Path

# ----------------------------
# Rutas principales del proyecto
# ----------------------------
BASE_DIR         = Path(__file__).parent.parent
DATA_DIR         = BASE_DIR / "data"
STRUCTURED_DIR   = DATA_DIR / "structured"   # JSONs generados por gpt_structurer.py
INDEX_DIR        = BASE_DIR / "index"        # Donde guardaremos embeddings y metadatos

# ----------------------------
# Parámetros del modelo de embeddings
# ----------------------------
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ----------------------------
# Archivos de salida (índice / metadatos)
# ----------------------------
EMBEDDINGS_FILE = INDEX_DIR / "embeddings.npy"
METADATA_FILE   = INDEX_DIR / "metadata.json"
