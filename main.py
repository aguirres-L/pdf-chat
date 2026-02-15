import io
import os
import threading
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from PyPDF2 import PdfReader

# Carga `.env` relativo a este archivo (más robusto que depender del cwd).
_ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(_ENV_PATH)


def _normalizar_modelo(nombre_modelo: str) -> str:
    nombre_modelo = (nombre_modelo or "").strip()
    if nombre_modelo.startswith("models/"):
        return nombre_modelo[len("models/") :]
    return nombre_modelo


def _normalizar_origin(origin: str) -> str:
    origin = (origin or "").strip()
    # El Origin del browser nunca trae "/" final; si lo configuran así en env, no matchea.
    while origin.endswith("/"):
        origin = origin[:-1]
    return origin


def _get_env_int(nombre: str, por_defecto: int) -> int:
    valor = os.getenv(nombre)
    if not valor:
        return por_defecto
    try:
        return int(valor)
    except ValueError:
        return por_defecto


def _get_env_list(nombre: str, por_defecto: list[str]) -> list[str]:
    valor = os.getenv(nombre)
    if not valor:
        return [_normalizar_origin(x) for x in por_defecto if _normalizar_origin(x)]
    # Soporta: "a,b,c" o JSON-like no; mantenemos simple.
    items = [_normalizar_origin(x) for x in valor.split(",")]
    return [x for x in items if x]


_CORS_ORIGINS = _get_env_list(
    "CORS_ORIGINS",
    [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
)
_MAX_CONTEXT_CHARS = _get_env_int("MAX_CONTEXT_CHARS", 60_000)
_GEMINI_MODEL = _normalizar_modelo(os.getenv("GEMINI_MODEL") or "gemini-flash-latest")

_lock = threading.Lock()
_estado_pdf: dict[str, Any] = {"texto": None, "nombre": None, "paginas": 0, "chars": 0}


def _extraer_texto_pdf(pdf_bytes: bytes) -> tuple[str, int]:
    """
    Extrae texto con PyPDF2.
    Nota: algunos PDFs son escaneados (imagen) y NO traen texto; ahí necesitarías OCR.
    """
    reader = PdfReader(io.BytesIO(pdf_bytes))
    textos: list[str] = []
    for pagina in reader.pages:
        try:
            txt = pagina.extract_text() or ""
        except Exception:
            txt = ""
        if txt.strip():
            textos.append(txt)
    return ("\n\n".join(textos)).strip(), len(reader.pages)


def _obtener_llm() -> ChatGoogleGenerativeAI:
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise HTTPException(
            status_code=500,
            detail="Falta GOOGLE_API_KEY. Configurala en tu .env o variables de entorno.",
        )

    # ChatGoogleGenerativeAI usa GOOGLE_API_KEY también por env,
    # pero pasarlo explícito deja más claro el setup.
    return ChatGoogleGenerativeAI(
        model=_GEMINI_MODEL,
        temperature=0.2,
        google_api_key=google_api_key,
    )

def _coerce_text(valor: Any) -> str:
    """
    Normaliza distintas formas de "content" a string.
    Algunos modelos/SDKs devuelven content como lista de partes (multimodal).
    """
    if valor is None:
        return ""

    if isinstance(valor, str):
        return valor

    if isinstance(valor, list):
        partes: list[str] = []
        for item in valor:
            if item is None:
                continue
            if isinstance(item, str):
                partes.append(item)
                continue
            if isinstance(item, dict):
                # Formato común: {"type": "...", "text": "..."}
                txt = item.get("text")
                if isinstance(txt, str) and txt.strip():
                    partes.append(txt)
                continue
            # Fallback
            partes.append(str(item))
        return "\n".join([p for p in partes if p.strip()])

    return str(valor)


app = FastAPI(title="PDF Chat API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class UploadPdfResponse(BaseModel):
    nombreArchivo: str | None = None
    paginas: int
    chars: int


class ChatRequest(BaseModel):
    pregunta: str = Field(..., min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    respuesta: str
    modelo: str
    charsContextoUsados: int


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/pdf/status")
def pdf_status() -> dict[str, Any]:
    with _lock:
        return {
            "nombreArchivo": _estado_pdf["nombre"],
            "paginas": _estado_pdf["paginas"],
            "chars": _estado_pdf["chars"],
            "isCargado": bool(_estado_pdf["texto"]),
        }


@app.post("/api/pdf", response_model=UploadPdfResponse)
async def upload_pdf(file: UploadFile = File(...)) -> UploadPdfResponse:
    if not file:
        raise HTTPException(status_code=400, detail="Falta el archivo.")

    # Nota: algunos navegadores no envían content-type correcto; igual intentamos.
    if file.content_type and file.content_type.lower() != "application/pdf":
        raise HTTPException(status_code=400, detail="El archivo debe ser un PDF.")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="El PDF está vacío.")

    try:
        texto, paginas = _extraer_texto_pdf(pdf_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="No se pudo leer el PDF.")

    if not texto:
        raise HTTPException(
            status_code=400,
            detail="No se encontró texto en el PDF. Si es un escaneo, necesitarás OCR.",
        )

    with _lock:
        _estado_pdf["texto"] = texto
        _estado_pdf["nombre"] = file.filename
        _estado_pdf["paginas"] = paginas
        _estado_pdf["chars"] = len(texto)

    return UploadPdfResponse(nombreArchivo=file.filename, paginas=paginas, chars=len(texto))


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    with _lock:
        texto_pdf = _estado_pdf["texto"]

    if not texto_pdf:
        raise HTTPException(
            status_code=400,
            detail="Primero subí un PDF a /api/pdf antes de chatear.",
        )

    contexto = texto_pdf[:_MAX_CONTEXT_CHARS]

    system = (
        "Sos un asistente B2B. Respondé usando SOLO el contenido del PDF provisto como contexto. "
        "Si la respuesta no está en el PDF, decí explícitamente que no la encontrás en el documento."
    )

    prompt = (
        "### Contexto (PDF)\n"
        f"{contexto}\n\n"
        "### Pregunta\n"
        f"{req.pregunta}\n"
    )

    llm = _obtener_llm()
    try:
        resultado = llm.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
        contenido = getattr(resultado, "content", resultado)
        respuesta = _coerce_text(contenido).strip()
        if not respuesta:
            respuesta = "No pude generar una respuesta en este momento."
    except Exception as e:
        # Mensaje acotado para debugging (sin exponer secrets).
        msg = str(e) or "Error desconocido"
        if len(msg) > 600:
            msg = msg[:600] + "…"
        if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
            msg = (
                msg
                + " (probable cuota/plan: revisá límites en ai.google.dev y el link de rate limits)"
            )
        raise HTTPException(status_code=502, detail=f"Error consultando el modelo (Gemini): {msg}")

    return ChatResponse(
        respuesta=respuesta,
        modelo=_GEMINI_MODEL,
        charsContextoUsados=len(contexto),
    )

