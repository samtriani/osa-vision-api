# OSA Vision API

Backend FastAPI del PoC OSA Vision (La Comer / EY). Por ahora expone solo
autenticación (login + JWT); los endpoints de negocio (`/vision/analizar`,
`/inventario/cruzar`, `/osa/dashboard`) se agregan sobre esta misma base.

## Cómo correrlo

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt

copy .env.example .env        # opcional: define tu propio SECRET_KEY
uvicorn app.main:app --reload --port 8000
```

Documentación interactiva: http://localhost:8000/docs

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
- `GET  /api/v1/health` — chequeo de salud

## Estructura

```
app/
  main.py              → FastAPI app, CORS, routers
  core/config.py        → settings (SECRET_KEY, CORS, expiración de tokens)
  core/security.py      → hash/verify de password (bcrypt), crear/decodificar JWT
  models/user.py        → esquemas Pydantic
  services/auth_service.py → usuarios en memoria + autenticación
  api/deps.py            → dependencia get_current_user (valida el Bearer token)
  api/routes/auth.py     → /login, /me
```
