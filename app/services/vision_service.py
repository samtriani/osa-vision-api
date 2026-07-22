import base64
import json
from typing import Any

from groq import Groq

from app.core.config import settings
from app.models.planograma import Planograma
from app.models.vision import AnalizarImagenResponse, HuecoDetectado

_SYSTEM_PROMPT_TEMPLATE = """Eres un asistente experto en auditoría de anaquel (on-shelf \
availability) para tiendas de autoservicio. Vas a comparar una foto de anaquel contra su \
planograma (lo que DEBERÍA tener cada posición) y reportar qué posiciones están vacías o \
mal surtidas.

Planograma de la sección "{nombre}":
{posiciones}

Para cada posición de la lista de arriba, revisa en la foto si el producto está presente \
en la cantidad esperada. Reporta solo las posiciones vacías o parciales.

Responde ÚNICAMENTE con un JSON con esta forma exacta, sin texto adicional:
{{
  "resumen": "una frase breve describiendo el estado general del anaquel",
  "huecos": [
    {{
      "posicion_id": "el id EXACTO tal como aparece en el planograma de arriba, ej. 'P3'",
      "estado": "vacio o parcial",
      "confianza": 0.0
    }}
  ]
}}

No inventes ids que no estén en la lista del planograma. "confianza" es un número entre \
0.0 y 1.0. Si todas las posiciones están completas, responde con "huecos": [].
"""


def _client() -> Groq:
    if not settings.groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY no está configurada. Define GROQ_API_KEY en tu archivo .env."
        )
    return Groq(api_key=settings.groq_api_key)


def _prompt_planograma(planograma: Planograma) -> str:
    return "\n".join(
        f"- id={p.id} | {p.posicion} | debería tener: {p.producto} (SKU {p.sku}), "
        f"{p.facings_esperados} piezas visibles"
        for p in planograma.posiciones
    )


def analizar_imagen(
    image_bytes: bytes, content_type: str, planograma: Planograma
) -> AnalizarImagenResponse:
    data_url = f"data:{content_type};base64,{base64.b64encode(image_bytes).decode('utf-8')}"
    system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
        nombre=planograma.nombre, posiciones=_prompt_planograma(planograma)
    )

    completion = _client().chat.completions.create(
        model=settings.groq_vision_model,
        response_format={"type": "json_object"},
        # qwen3.6-27b es un modelo "thinking" por defecto: sin esto, gasta el
        # presupuesto de tokens razonando y el JSON final sale vacío/truncado.
        reasoning_effort="none",
        reasoning_format="hidden",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Compara esta foto del anaquel contra el planograma y detecta los huecos.",
                    },
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
    )

    payload: dict[str, Any] = json.loads(completion.choices[0].message.content)
    posiciones_por_id = {p.id: p for p in planograma.posiciones}

    huecos: list[HuecoDetectado] = []
    for h in payload.get("huecos", []):
        posicion = posiciones_por_id.get(h.get("posicion_id"))
        if posicion is None:
            # El modelo alucinó un id que no existe en el planograma; se descarta
            # en vez de inventar un sku/producto que no podemos respaldar.
            continue
        huecos.append(
            HuecoDetectado(
                posicion_id=posicion.id,
                posicion=posicion.posicion,
                sku=posicion.sku,
                producto=posicion.producto,
                facings_esperados=posicion.facings_esperados,
                estado=h.get("estado", "vacio"),
                confianza=h.get("confianza", 0.5),
            )
        )

    return AnalizarImagenResponse(
        seccion_id=planograma.seccion_id,
        resumen=payload.get("resumen", ""),
        huecos=huecos,
    )
