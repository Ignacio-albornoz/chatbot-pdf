# src/chat.py

import os
import json
import openai
import argparse
from pathlib import Path
from typing import List, Dict

from indexer import search_index
from config import EMBEDDING_MODEL_NAME

# Palabras clave para detectar usuario GCEI
CLAVES_GCEI = ["gcei", "aprobación", "aceptar despliegue", "gestión de cambio", "rechazo", "arcgcei"]

# Máximo de intercambios que guardaremos (5 interacciones → 5 preguntas + 5 respuestas = 10 mensajes)
MAX_HISTORY_TURNS = 5

def detectar_usuario(pregunta: str) -> str:
    """
    Devuelve "gcei" si la pregunta contiene alguna palabra clave de GCEI; en caso contrario "desarrollador".
    """
    texto = pregunta.lower()
    for clave in CLAVES_GCEI:
        if clave.lower() in texto:
            return "gcei"
    return "desarrollador"

def build_context_blocks(results: List[Dict], max_chars_per_block: int = 1000) -> str:
    """
    A partir de la lista de resultados devueltos por search_index (cada uno con 'metadata' y 'distance'),
    genera un bloque de texto que concatena los contenidos más relevantes (recortados a max_chars_per_block).
    Retorna un string formateado para colocarlo en el prompt.
    """
    context = []
    for rank, entry in enumerate(results, start=1):
        md = entry["metadata"]
        preview = md.get("content_preview", "").strip()
        if not preview:
            continue

        bloque = (
            f"[Bloque {rank}] Fuente: {md['source_file']}  |  "
            f"Title: {md['title']}  |  Tags: {md['tags']}  |  Roles: {md['roles']}\n"
            f"{preview[:max_chars_per_block]}...\n"
        )
        context.append(bloque)

    if not context:
        return "No se encontraron bloques relevantes.\n"

    return "\n".join(context)

def build_system_message() -> str:
    """
    Mensaje sistémico fijo que da instrucciones generales a ChatGPT.
    """
    return (
        "Eres un asistente técnico experto en Jenkins.\n"
        "Tienes a disposición documentación extraída de manuales de Jenkins.\n"
        "Responde sólo con la información que esté en el contexto proporcionado.\n"
        "No añadas nada fuera de esos datos.\n"
        "Adapta tu respuesta según el rol del usuario:\n"
        "- Si el usuario es GCEI, enfócate en flujos de aprobación y gestión de cambios.\n"
        "- Si el usuario es desarrollador, enfócate en pasos técnicos y ejemplos de Jenkinsfile.\n"
    )

def truncate_history(history: List[Dict]) -> List[Dict]:
    """
    Mantiene en 'history' sólo los últimos MAX_HISTORY_TURNS * 2 mensajes
    (cada interacción consta de un mensaje de usuario y uno de asistente).
    """
    max_messages = MAX_HISTORY_TURNS * 2
    return history[-max_messages:]

def chat_loop(api_model: str, top_k: int):
    """
    Entramos en un bucle REPL. El usuario puede escribir preguntas, recibir respuestas
    y 'repreguntar' tantas veces quiera. Se mantiene un historial de las últimas 5 interacciones.
    """
    # Lista de mensajes en formato OpenAI Chat API: [{"role": "...", "content": "..."}, ...]
    history: List[Dict] = []

    print("🤖 Chat interactivo iniciado. Escribe tu pregunta y presiona ENTER.")
    print("   Para salir, escribe 'exit' o presiona Ctrl+C.\n")

    while True:
        try:
            pregunta = input("Tú: ").strip()
            if pregunta.lower() in ("exit", "quit"):
                print("👋 ¡Hasta luego!")
                break

            # 1) Detectar rol
            rol = detectar_usuario(pregunta)

            # 2) Recuperar bloques relevantes en FAISS
            resultados = search_index(pregunta, top_k=top_k)

            # 3) Construir contexto para esta pregunta
            bloques_contexto = build_context_blocks(resultados)

            # 3.1) Si no hay contexto, devolvemos un mensaje breve y no consultamos a la API
            if "No se encontraron bloques relevantes" in bloques_contexto:
                respuesta_corta = (
                    "Lo siento, pero no dispongo de suficiente información para responder "
                    "de forma clara y precisa. "
                    "Revisa el manual o consulta al equipo de Jenkins/GCEI para más detalles."
                )
                print(f"Chatbot: {respuesta_corta}\n")
                # **Guardar esta interacción en el historial** (pregunta + respuesta breve)
                history.append({"role": "user", "content": pregunta})
                history.append({"role": "assistant", "content": respuesta_corta})
                # Limitar la longitud del historial
                history = truncate_history(history)
                continue

            # 4) Armar mensajes para la API, incluyendo:
            #    - Mensaje sistémico fijo
            #    - Historial de conversación (hasta las últimas 5 interacciones)
            #    - Nuevo mensaje de usuario que incluye rol + contexto + pregunta
            system_msg = build_system_message()

            # Empezamos a construir la lista de mensajes
            messages: List[Dict] = []
            messages.append({"role": "system", "content": system_msg})

            # 4.1) Incluir el historial (últimos 5 diálogos, cada uno con user + assistant)
            for m in history:
                messages.append(m)

            # 4.2) Crear el contenido del nuevo mensaje de usuario con contexto
            user_msg_content = (
                f"ROL_USUARIO: {rol}\n\n"
                "CONTEXTO:\n"
                f"{bloques_contexto}\n"
                "PREGUNTA:\n"
                f"{pregunta}\n"
            )
            messages.append({"role": "user", "content": user_msg_content})

            # 5) Llamada a la API de ChatGPT
            response = openai.chat.completions.create(
                model=api_model,
                messages=messages,
                temperature=0.0,
                max_tokens=1000
            )
            respuesta = response.choices[0].message.content.strip()

            # 6) Imprimir respuesta y guardarla en el historial
            print(f"Chatbot: {respuesta}\n")
            history.append({"role": "user", "content": pregunta})
            history.append({"role": "assistant", "content": respuesta})

            # 7) Limitar el historial a las últimas 5 interacciones
            history = truncate_history(history)

        except KeyboardInterrupt:
            print("\n👋 ¡Hasta luego!")
            break
        except Exception as e:
            print(f"❌ Ocurrió un error: {e}")
            break

def main():
    parser = argparse.ArgumentParser(
        description="Modo interactivo del chatbot técnico sobre Jenkins."
    )
    parser.add_argument(
        "--k",
        "-k",
        type=int,
        default=5,
        help="Cantidad de bloques relevantes a recuperar del índice FAISS (por defecto 5)."
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="gpt-3.5-turbo",
        help="Modelo de ChatGPT a usar (por defecto gpt-3.5-turbo)."
    )

    args = parser.parse_args()
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Debes exportar OPENAI_API_KEY antes de ejecutar este script.")
        return

    openai.api_key = os.getenv("OPENAI_API_KEY")

    # Iniciamos el bucle interactivo
    chat_loop(api_model=args.model, top_k=args.k)

if __name__ == "__main__":
    main()
