# OSA Vision API

Backend FastAPI del PoC OSA Vision (La Comer / EY). Expone autenticación
(login + JWT) y detección de huecos de anaquel vía VLM (`/vision/analizar`).
Los endpoints de negocio restantes (`/inventario/cruzar`, `/osa/dashboard`)
se agregan sobre esta misma base.

## Cómo correrlo

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt

copy .env.example .env        # define SECRET_KEY y GROQ_API_KEY
uvicorn app.main:app --reload --port 8000
```

Documentación interactiva: http://localhost:8000/docs

`GROQ_API_KEY` se obtiene en https://console.groq.com/keys. El modelo de
visión usado es `qwen/qwen3.6-27b` (VLM en preview en GroqCloud a jul-2026;
Llama 4 Scout/Maverick, que se usaban antes para esto, ya fueron deprecados).

## Usuarios demo

En memoria, uno por rol (mismos roles que usa el frontend):

| username    | password      | rol       |
|-------------|---------------|-----------|
| operativo   | operativo123  | operativo |
| tienda      | tienda123     | tienda    |
| ejecutivo   | ejecutivo123  | ejecutivo |

## Endpoints

- `POST /api/v1/auth/login` — `{ username, password }` → `{ access_token, token_type, user }`
- `GET  /api/v1/auth/me` — requiere `Authorization: Bearer <token>` → usuario actual
- `GET  /api/v1/vision/secciones` — requiere Bearer token → lista de planogramas disponibles (secciones demo: `lacteos-4b`, `lacteos-4c`, `abarrotes-7a`)
- `POST /api/v1/vision/analizar` — requiere Bearer token; multipart `imagen` (JPEG/PNG/WebP, máx. 20MB) + campo `seccion_id` → `{ seccion_id, resumen, huecos: [{ posicion_id, posicion, sku, producto, facings_esperados, estado, confianza }] }`
- `GET  /api/v1/health` — chequeo de salud

La VLM compara la foto contra el **planograma** de la sección indicada (lo que
debería tener cada posición) en vez de adivinar el producto solo por contexto
visual: el modelo únicamente decide qué posiciones están vacías/parciales
(por su `id`), y el `sku`/`producto`/`facings_esperados` de la respuesta
salen siempre del planograma, no del modelo — así no hay alucinación de SKU.

## Planograma

Datos en memoria (`app/services/planograma_service.py`), un planograma por
sección con sus posiciones y el SKU/producto/cantidad esperada en cada una.
Es el equivalente al catálogo de referencia contra el que se audita el
anaquel; en un siguiente paso esto se movería a una fuente real (el sistema
de planogramas de la tienda) en vez de datos hardcodeados.

## Estructura

```
app/
  main.py              → FastAPI app, CORS, routers
  core/config.py        → settings (SECRET_KEY, CORS, expiración de tokens, Groq)
  core/security.py      → hash/verify de password (bcrypt), crear/decodificar JWT
  models/user.py        → esquemas Pydantic de auth
  models/planograma.py   → esquemas Pydantic de planograma (posición → SKU esperado)
  models/vision.py       → esquemas Pydantic de detección de huecos
  services/auth_service.py → usuarios en memoria + autenticación
  services/planograma_service.py → planogramas demo en memoria
  services/vision_service.py → llamada a Groq (qwen/qwen3.6-27b), compara foto vs. planograma
  api/deps.py            → dependencia get_current_user (valida el Bearer token)
  api/routes/auth.py     → /login, /me
  api/routes/vision.py    → /vision/secciones, /vision/analizar
```
