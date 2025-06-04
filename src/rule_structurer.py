import os
import json
import re
from pathlib import Path

from pdf_loader import compute_file_hash, MANIFEST_PATH as PDF_MANIFEST_PATH

# Directorios base
BASE_DIR        = Path(__file__).parent
PROJECT_DIR     = BASE_DIR.parent
PROCESSED_DIR   = PROJECT_DIR / "data" / "processed"   # Contiene .txt generados por pdf_loader.py
STRUCTURED_DIR  = PROJECT_DIR / "data" / "structured"  # Aquí guardaremos los JSON rule-based
STRUCT_MANIFEST = STRUCTURED_DIR / "manifest.json"

# Patrón para detectar “títulos” estilo “TODO EN MAYÚSCULAS” o “Encabezado:”
#  - LÍNEA COMPLETA EN MAYÚSCULAS Y ESPACIOS (mínimo 3 caracteres)
#  - O línea que termine en “:”
HEADING_PATTERNS = [
    re.compile(r'^[A-ZÁÉÍÓÚÑ0-9\s]{3,}$'),   # todo mayúsculas (o dígitos) y al menos 3 caracteres
    re.compile(r'.+:$')                      # cualquier texto que termine en “:”
]

# Máximo número de saltos de línea en blanco que separan bloques
SPLIT_PATTERN = re.compile(r'\n{2,}')  # dos o más saltos = nuevo bloque


def ensure_dirs():
    """
    Crea la carpeta STRUCTURED_DIR y el manifest si no existen.
    """
    STRUCTURED_DIR.mkdir(parents=True, exist_ok=True)
    if not STRUCT_MANIFEST.exists():
        STRUCT_MANIFEST.write_text(json.dumps({}, indent=2), encoding="utf-8")


def load_manifest(path: Path) -> dict:
    """
    Carga un manifest JSON (PDF → hash) o retorna {} si no existe.
    """
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(path: Path, manifest: dict):
    """
    Guarda el manifest JSON en la ruta dada.
    """
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def is_heading(line: str) -> bool:
    """
    Retorna True si la línea coincide con alguno de los patrones de encabezado.
    """
    texto = line.strip()
    if not texto:
        return False
    for pat in HEADING_PATTERNS:
        if pat.match(texto):
            return True
    return False


def split_into_blocks(text: str) -> list[dict]:
    """
    Divide el texto completo en bloques basados en detectores de encabezado y saltos de párrafo.
    Cada bloque es un dict con keys: 'title', 'content'.
    """
    blocks = []
    # Dividimos primero por dobles saltos de línea para tener “párrafos largos”
    raw_chunks = SPLIT_PATTERN.split(text)

    current_title = None
    current_content_lines = []

    def flush_block():
        """
        Interna: cuando detectamos que un bloque termina, creamos el entry en blocks.
        """
        content = "\n".join(current_content_lines).strip()
        if content:
            blocks.append({
                "title": current_title if current_title else "Sin título",
                "content": content
            })

    for chunk in raw_chunks:
        lines = chunk.splitlines()
        # Si el chunk completo parece un encabezado (todas las líneas son “heading”), lo tomamos como título
        if len(lines) == 1 and is_heading(lines[0]):
            # Guardar el bloque anterior si existía
            if current_content_lines:
                flush_block()
            # Iniciar nuevo bloque con este título
            current_title = lines[0].strip()
            current_content_lines = []
        else:
            # Este chunk no es “solo un heading”; lo agregamos al contenido actual
            current_content_lines.extend(lines)
    # Al final, flush último bloque
    if current_content_lines:
        flush_block()

    # Si no se detectó nunca título, recomponer todo como un solo bloque
    if not blocks:
        text_strip = text.strip()
        if text_strip:
            blocks.append({
                "title": "Sin título",
                "content": text_strip
            })
    return blocks


def process_all_rules():
    """
    Recorre cada .txt en data/processed/, verifica el hash en PDF_MANIFEST_PATH,
    y para cada PDF nuevo/modificado genera un JSON rule-based en data/structured/.
    """
    ensure_dirs()

    # Cargar manifest de hashes (generado por pdf_loader.py)
    pdf_manifest = load_manifest(PDF_MANIFEST_PATH)
    # Manifest de estructuras rule-based
    struct_manifest = load_manifest(STRUCT_MANIFEST)

    for txt_path in sorted(PROCESSED_DIR.glob("*.txt")):
        pdf_name = txt_path.stem + ".pdf"
        current_hash = pdf_manifest.get(pdf_name)

        if current_hash is None:
            print(f"⚠️ No existe hash para {pdf_name} en data/processed/manifest.json. Skipping.")
            continue

        old_hash = struct_manifest.get(pdf_name)
        if old_hash == current_hash:
            print(f"— {pdf_name}: sin cambios (hash coincide). [Skipped]")
            continue

        print(f"— {pdf_name}: hash nuevo o modificado. Estructurando rule-based…")

        texto = txt_path.read_text(encoding="utf-8")
        raw_blocks = split_into_blocks(texto)

        # Construir la lista final de “blocks” con campos file, tags, roles, content
        final_blocks = []
        for blk in raw_blocks:
            title = blk["title"]
            content = blk["content"]
            final_blocks.append({
                "title": title,
                "file": pdf_name,
                "tags": [],               # Podrías agregar heurísticas para extraer tags simples
                "roles": ["system"],      # O ["system","gcei"] si quieres abarcar ambos
                "content": content
            })

        # Armar JSON completo
        output_data = {
            "filename": pdf_name,
            "blocks": final_blocks
        }

        # Guardar JSON en data/structured
        out_path = STRUCTURED_DIR / f"{txt_path.stem}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        # Actualizar manifest de estructuras
        struct_manifest[pdf_name] = current_hash
        save_manifest(STRUCT_MANIFEST, struct_manifest)

        print(f"   → Guardado rule-based: data/structured/{txt_path.stem}.json ({len(final_blocks)} bloques)\n")

    print("✅ Estructuración rule-based completada. Manifest actualizado.")


if __name__ == "__main__":
    process_all_rules()
