# src/indexer.py

import os
import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

from config import (
    INDEX_DIR,
    EMBEDDINGS_FILE,
    METADATA_FILE,
    EMBEDDING_MODEL_NAME,
)

# Archivo donde guardaremos el índice FAISS
FAISS_INDEX_PATH = INDEX_DIR / "index.faiss"


def ensure_index_dir():
    """
    Crea INDEX_DIR si no existe.
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)


def build_faiss_index():
    """
    1. Carga la matriz de embeddings desde index/embeddings.npy.
    2. Construye un índice FAISS de tipo Flat (IndexFlatL2).
    3. Agrega todos los vectores al índice.
    4. Guarda el índice serializado en index/index.faiss.
    """
    ensure_index_dir()

    if not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(f"No se encontró {EMBEDDINGS_FILE}, ejecuta antes embeddings.py")

    # 1) Cargar embeddings
    embeddings = np.load(EMBEDDINGS_FILE)
    num_vectors, dim = embeddings.shape
    print(f"⚙️  Construyendo índice FAISS con {num_vectors} vectores de dimensión {dim}...")

    # 2) Crear un índice L2 plano
    index = faiss.IndexFlatL2(dim)

    # 3) Agregar vectores
    index.add(embeddings)
    print(f"✅ Índice entrenado y poblado (ntotal = {index.ntotal}).")

    # 4) Guardar el índice
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    print(f"✅ Índice FAISS guardado en:\n   {FAISS_INDEX_PATH}")


def load_faiss_index() -> faiss.Index:
    """
    Carga y retorna el índice FAISS desde disk. 
    Lanza error si no está construido.
    """
    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(f"No existe {FAISS_INDEX_PATH}. Ejecuta antes indexer.build_faiss_index().")
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    return index


def load_metadata() -> list[dict]:
    """
    Carga y retorna la lista de metadatos desde index/metadata.json.
    """
    if not METADATA_FILE.exists():
        raise FileNotFoundError(f"No existe {METADATA_FILE}. Ejecuta antes embeddings.py.")
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def search_index(query: str, top_k: int = 5) -> list[dict]:
    """
    Dada una consulta de texto, genera su embedding con el mismo modelo empleado
    para indexar y recupera los top_k bloques más similares:
      - Calcula embedding de `query`.
      - Realiza index.search para obtener distancias e índices.
      - Devuelve una lista de diccionarios con:
          {
            "metadata": <diccionario de metadata.json correspondiente>,
            "distance": <distancia Euclídea>
          }
    """
    # 1) Cargar modelo de embeddings
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # 2) Generar embedding para la query
    query_vec = model.encode(query)
    # Asegurarse de que tenga forma (1, dim)
    query_vec = np.array([query_vec]).astype(np.float32)

    # 3) Cargar índice y metadata
    index = load_faiss_index()
    metadata_list = load_metadata()

    # 4) Buscar los top_k más cercanos
    distances, indices = index.search(query_vec, top_k)
    distances = distances[0]  # forma (top_k,)
    indices   = indices[0]    # forma (top_k,)

    results = []
    for dist, idx in zip(distances, indices):
        # idx puede ser -1 si hay menos vectores que top_k
        if idx < 0 or idx >= len(metadata_list):
            continue
        result = {
            "metadata": metadata_list[idx],
            "distance": float(dist),
        }
        results.append(result)

    return results


if __name__ == "__main__":
    """
    Uso por defecto:
      1) python src/indexer.py build
         => Construye (o reconstuye) el índice FAISS.
      2) python src/indexer.py search "tu consulta aquí" --k 5
         => Muestra en consola los top 5 resultados.
    """

    import argparse

    parser = argparse.ArgumentParser(
        description="Construye y/o consulta el índice FAISS de embeddings."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcomando 'build'
    build_parser = subparsers.add_parser("build", help="Construir/reconstruir el índice FAISS.")
    # No necesita argumentos adicionales

    # Subcomando 'search'
    search_parser = subparsers.add_parser("search", help="Buscar en el índice FAISS.")
    search_parser.add_argument(
        "query",
        type=str,
        help="Texto de la consulta a buscar."
    )
    search_parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Número de resultados a recuperar (por defecto 5)."
    )

    args = parser.parse_args()

    if args.command == "build":
        build_faiss_index()
    elif args.command == "search":
        resultados = search_index(args.query, top_k=args.k)
        if not resultados:
            print("⚠️ No se obtuvieron resultados.")
        else:
            print(f"🔎 Top {len(resultados)} resultados para: \"{args.query}\"")
            for i, entry in enumerate(resultados, start=1):
                md = entry["metadata"]
                print(f"\n[{i}] #{md['source_file']} — block #{md['block_index']}")
                print(f"    Title: {md['title']}")
                print(f"    Tags: {md['tags']}  Roles: {md['roles']}")
                print(f"    Distance: {entry['distance']:.4f}")
                # Opcional: mostrar un preview del contenido
                preview = md.get("content_preview", "")
                if preview:
                    print(f"    Preview: {preview}...")
