# src/gpt_structurer.py

import os
import json
import hashlib
import openai
import re
from pathlib import Path

# Importamos del mismo paquete (pdf_loader.py debe definir MANIFEST_PATH)
from pdf_loader import compute_file_hash, MANIFEST_PATH as PDF_MANIFEST_PATH

# Directorios base relativos a la raíz del proyecto
BASE_DIR        = Path(__file__).parent
PROJECT_DIR     = BASE_DIR.parent
PROCESSED_DIR   = PROJECT_DIR / "data" / "processed"
STRUCTURED_DIR  = PROJECT_DIR / "data" / "structured"
STRUCT_MANIFEST = STRUCTURED_DIR / "manifest.json"

# Ajustable: máximo de caracteres por fragmento
MAX_CHARS_PER_CHUNK = 3000

# Modelo a usar para estructurar (puedes cambiarlo en config/env)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")


def ensure_dirs():
    """
    Crea las carpetas y archivos manifest si no existían.
    """
    STRUCTURED_DIR.mkdir(parents=True, exist_ok=True)
    if not STRUCT_MANIFEST.exists():
        STRUCT_MANIFEST.write_text(json.dumps({}, indent=2), encoding="utf-8")


def load_manifest(path: Path) -> dict:
    """
    Carga un manifest JSON desde la ruta dada o devuelve {} si no existe.
    """
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(path: Path, manifest: dict):
    """
    Guarda el manifiesto (hashes) en la ruta indicada.
    """
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def split_text_into_chunks(text: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> list[str]:
    """
    Divide el texto completo en fragmentos (chunks) de tamaño aproximado <= max_chars,
    intentando cortar en doble salto de línea ("\n\n") para no partir párrafos en medio.
    """
    paragraphs = text.split("\n\n")
    chunks = []
    current = ""

    for para in paragraphs:
        # Si agregando este párrafo excede max_chars, cerrar chunk anterior
        if len(current) + len(para) + 2 > max_chars:
            if current.strip():
                chunks.append(current.strip())
            current = para + "\n\n"
        else:
            current += para + "\n\n"

    # Agregar el último chunk si queda contenido
    if current.strip():
        chunks.append(current.strip())

    return chunks


def _strip_markdown_fences(text: str) -> str:
    """
    Si el texto comienza y/o termina con fences Markdown (```json…```), 
    los elimina. Devuelve únicamente la sección JSON interior.
    """
    # Patrón para capturar todo lo que esté entre ``` y ```.
    fence_pattern = r"```(?:json)?\s*(?P<json_body>.*?)[\s]*```"
    match = re.search(fence_pattern, text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        content = match.group("json_body").strip()
    else:
        content = text.strip()

    # A veces el modelo no usa fences pero incluye comentarios; 
    # extraemos desde la primera llave "{" hasta la última "}"
    start = content.find("{")
    end   = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        return content[start : end + 1]
    return content


def call_openai_to_structure(chunk: str, pdf_filename: str) -> list[dict]:
    """
    Llama a la API de OpenAI (nueva interfaz) para que transforme un fragmento de texto
    en un array de bloques JSON con la estructura deseada.
    """
    prompt = f"""
Eres un asistente experto en documentación técnica de Jenkins. Recibes un
fragmento de un manual de Jenkins y debes devolver **solo** un JSON válido
con la clave "blocks", donde cada bloque tenga esta forma:

{{
  "title": "Título breve del bloque",
  "file": "{pdf_filename}",
  "tags": ["palabra_clave1", "palabra_clave2", ...],
  "roles": ["system" o "gcei" o ambos],
  "content": "Texto completo del bloque"
}}

Devuelve únicamente el JSON. Ahora procesa este fragmento (entre comillas triples):
\"\"\"{chunk}\"\"\"
"""

    response = openai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "Eres un experto en DevOps y Jenkins."},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.0,
        max_tokens=1500
    )

    raw_content = response.choices[0].message.content.strip()
    # Primero, eliminamos fences Markdown si los hubiera y extraemos solo el JSON
    json_candidate = _strip_markdown_fences(raw_content)

    try:
        parsed = json.loads(json_candidate)
        return parsed.get("blocks", [])
    except json.JSONDecodeError as e:
        print("⚠️ Error al parsear JSON de OpenAI:", e)
        print("Respuesta cruda:\n", raw_content)
        print("Texto tras limpiar fences:\n", json_candidate)
        return []


def process_all_structures():
    """
    Recorre cada .txt en data/processed/, compara hashes con data/processed/manifest.json
    para ver cuáles PDFs cambiaron, y para esos llama a OpenAI para generar JSON estructurado.
    """
    ensure_dirs()

    # Cargo el manifest de PDFs (contiene hashes calculados en pdf_loader.py)
    pdf_manifest = load_manifest(PDF_MANIFEST_PATH)
    # Cargo (o inicializo) el manifest de estructuras
    struct_manifest = load_manifest(STRUCT_MANIFEST)

    # Itero sobre cada archivo .txt generado por pdf_loader.py
    for txt_path in sorted(PROCESSED_DIR.glob("*.txt")):
        pdf_name = txt_path.stem + ".pdf"
        current_hash = pdf_manifest.get(pdf_name)

        if current_hash is None:
            print(f"⚠️ No hay hash para {pdf_name} en data/processed/manifest.json. Saltando.")
            continue

        old_hash = struct_manifest.get(pdf_name)
        if old_hash == current_hash:
            print(f"— {pdf_name}: sin cambios (hash coincide). [Skipped]")
            continue

        print(f"— {pdf_name}: hash nuevo o modificado. Estructurando…")
        texto = txt_path.read_text(encoding="utf-8")
        chunks = split_text_into_chunks(texto)

        all_blocks = []
        for idx, chunk in enumerate(chunks, 1):
            print(f"   • Chunk {idx}/{len(chunks)} ({len(chunk)} chars)…")
            bloques = call_openai_to_structure(chunk, pdf_name)
            all_blocks.extend(bloques)

        # Armo el JSON final
        output_data = {
            "filename": pdf_name,
            "hash": current_hash,
            "blocks": all_blocks
        }
        out_path = STRUCTURED_DIR / f"{txt_path.stem}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        # Actualizo el manifest de estructuras
        struct_manifest[pdf_name] = current_hash
        save_manifest(STRUCT_MANIFEST, struct_manifest)

        print(f"   → Guardado: data/structured/{txt_path.stem}.json ({len(all_blocks)} bloques)\n")

    print("✅ Estructuración completa. Manifest de estructuras actualizado.")


if __name__ == "__main__":
    # Verifico que exista la variable de entorno
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("❌ Exporta OPENAI_API_KEY antes de ejecutar.")
    openai.api_key = os.getenv("OPENAI_API_KEY")

    process_all_structures()
