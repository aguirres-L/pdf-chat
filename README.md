# PDF Chat (Backend)

Backend en FastAPI para:
- Subir un PDF, extraer texto y mantenerlo en memoria (global).
- Consultar a Google Gemini usando ese texto como contexto.

## Requisitos
- Python 3.10+ (recomendado 3.11/3.12)

## Setup local (PowerShell)

Desde `pdf-chat/`:

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Crear `.env` (podés copiar `.env.example`):

```powershell
copy .env.example .env
```

Ejecutar:

```powershell
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## CORS (Netlify / Render)
Si tu UI en Netlify te muestra un error tipo “blocked by CORS policy”, asegurate de que en Render
estén configuradas estas variables (y redeploy luego de cambiarlas):

- `CORS_ORIGINS`: lista separada por comas, sin `/` al final. Ejemplo:
  - `https://pdf-chat-ui.netlify.app,http://localhost:5173`
- (Opcional) `CORS_ORIGIN_REGEX`: útil para previews de Netlify. Ejemplo:
  - `^https://.*\\.netlify\\.app$`

## Endpoints
- `GET /health`
- `POST /api/pdf` (form-data `file`)
- `GET /api/pdf/status`
- `POST /api/chat` (JSON `{ "pregunta": "...", "usar_ia_adicional": false }`)

## Notas importantes (limitaciones fase 1)
- El PDF se guarda en memoria (solo “el último PDF”).
- Si tu PDF es escaneado (imagen), PyPDF2 no extrae texto → necesitás OCR.
- Para PDFs largos, se recorta el contexto a `MAX_CONTEXT_CHARS`.

