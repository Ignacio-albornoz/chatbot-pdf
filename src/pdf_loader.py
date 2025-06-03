# src/pdf_loader.py

import os
import json
import hashlib
from pathlib import Path
import fitz  # PyMuPDF

# Directorios base
BASE_DIR       = Path(__file__).parent.parent
RAW_DIR        = BASE_DIR / "data" / "raw"
PROCESSED_DIR  = BASE_DIR / "data" / "processed"
MANIFEST_PATH  = PROCESSED_DIR / "manifest.json"

def ensure_dirs():
    """Crea las carpetas necesarias si no existen."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if not MANIFEST_PATH.exists():
        # Si no existe el manifest, crear uno vacío
        MANIFEST_PATH.write_text(json.dumps({}), encoding="utf-8")

def compute_file_hash(path: Path) -> str:
    """
    Calcula el SHA256 de un archivo binario (el PDF).
    Devuelve el hash en formato hexadecimal.
    """
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def load_manifest() -> dict:
    """
    Carga el JSON de seguimiento de hashes.
    Estructura esperada: { "nombre.pdf": "hashAnterior", ... }
    """
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_manifest(manifest: dict):
    """
    Guarda el dict de tracking de hashes en MANIFEST_PATH.
    """
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

def list_pdfs() -> list[Path]:
    """Devuelve una lista de Paths a todos los archivos .pdf en RAW_DIR."""
    return sorted(RAW_DIR.glob("*.pdf"))

def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Abre un PDF con PyMuPDF y extrae todo el texto.
    Devuelve un string con el contenido completo.
    """
    text = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text.append(page.get_text())
    return "\n".join(text)

def save_text(pdf_name: str, content: str):
    """
    Guarda el texto extraído en PROCESSED_DIR con el mismo nombre que el PDF,
    pero con extensión .txt.
    """
    output_path = PROCESSED_DIR / f"{pdf_name}.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

def process_all_pdfs():
    """
    Recorre cada PDF en data/raw, compara hashes con el manifest,
    y solo extrae texto si el PDF cambió (o no estaba procesado antes).
    """
    ensure_dirs()
    manifest = load_manifest()

    for pdf_path in list_pdfs():
        name      = pdf_path.name            # e.g. "MT037.3.pdf"
        name_stem = pdf_path.stem            # e.g. "MT037.3"
        print(f"--- Procesando {name} ---")

        # 1️⃣ Calcular hash del PDF
        current_hash = compute_file_hash(pdf_path)

        # 2️⃣ Verificar en el manifest si ya existe el mismo hash
        old_hash = manifest.get(name)
        if old_hash == current_hash:
            print(f"    • Sin cambios (hash coincide). [Skipped]")
            continue

        # 3️⃣ Si llegó hasta acá, significa que hubo cambios o es nuevo
        print(f"    • Hash nuevo o diferente (guardando texto).")

        # Extraer texto y guardar
        texto = extract_text_from_pdf(pdf_path)
        save_text(name_stem, texto)

        # 4️⃣ Actualizar manifest con el nuevo hash
        manifest[name] = current_hash
        print(f"    → Guardado: data/processed/{name_stem}.txt")

    # 5️⃣ Al finalizar, persistir el manifest actualizado
    save_manifest(manifest)
    print("\nManifest actualizado con hashes de PDFs procesados.")

if __name__ == "__main__":
    process_all_pdfs()
