import os
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sentence_transformers import SentenceTransformer

from config import STRUCTURED_DIR, INDEX_DIR, EMBEDDING_MODEL_NAME, EMBEDDINGS_FILE, METADATA_FILE

def ensure_index_dir():
    """
    Crea la carpeta INDEX_DIR si no existe.
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

def load_structured_jsons():
    """
    Retorna una lista de Path a todos los archivos .json dentro de STRUCTURED_DIR.
    """
    return sorted(STRUCTURED_DIR.glob("*.json"))

def generate_embeddings():
    """
    - Carga todos los JSON de STRUCTURED_DIR.
    - Para cada bloque en cada archivo JSON, obtiene el campo "content" y genera un embedding.
    - Construye:
      * embeddings_list: lista de vectores (NumPy)
      * metadata_list: lista de dicts con metadatos mínimos por bloque
    - Guarda embeddings como un array en EMBEDDINGS_FILE (.npy).
    - Guarda metadata_list en METADATA_FILE (.json).
    """
    # 1) Asegurarnos de que exista la carpeta de índice
    ensure_index_dir()

    # 2) Cargar modelo de SentenceTransformers
    print(f"⌛ Cargando modelo de embeddings '{EMBEDDING_MODEL_NAME}' …")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    embeddings_list = []
    metadata_list   = []

    # 3) Recorrer todos los JSON estructurados
    all_jsons = load_structured_jsons()
    if not all_jsons:
        print("⚠️ No se encontraron archivos en data/structured/. Verifica que gpt_structurer.py haya corrido antes.")
        return

    print(f"🔍 Procesando {len(all_jsons)} archivos estructurados…")
    for json_path in all_jsons:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        filename = data.get("filename", json_path.stem + ".pdf")
        blocks   = data.get("blocks", [])

        if not blocks:
            continue

        # 4) Para cada bloque, generar embedding y guardar metadatos
        for idx, block in enumerate(blocks, start=1):
            content = block.get("content", "").strip()
            if not content:
                # Si no hay texto, saltar
                continue

            # Generar embedding (1 x dim_embedding)
            vector = model.encode(content, show_progress_bar=False)

            embeddings_list.append(vector.tolist())

            # Metadatos mínimos para recuperar o mostrar posteriormente
            metadata_list.append({
                "source_file": filename,
                "block_index": idx - 1,          # índice dentro del array "blocks"
                "title": block.get("title", ""),
                "tags": block.get("tags", []),
                "roles": block.get("roles", []),
                "content_preview": content[:200]  # solo un preview corto opcional
            })

    # 5) Convertir lista de embeddings a array NumPy y guardarlo
    embeddings_arr = np.array(embeddings_list, dtype=np.float32)
    np.save(EMBEDDINGS_FILE, embeddings_arr)
    print(f"✅ Guardado embeddings ({embeddings_arr.shape}) en:\n   {EMBEDDINGS_FILE}")

    # 6) Guardar metadata_list en JSON
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=2, ensure_ascii=False)
    print(f"✅ Guardados metadatos ({len(metadata_list)} entradas) en:\n   {METADATA_FILE}\n")


if __name__ == "__main__":
    generate_embeddings()
