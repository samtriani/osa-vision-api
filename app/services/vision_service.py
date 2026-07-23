import base64
import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from groq import Groq

from app.core.config import settings
from app.models.planograma import Planograma
from app.models.vision import (
    AnalizarConReferenciaResponse,
    AnalizarImagenResponse,
    CategoriaVisual,
    EstadoPosicion,
    HuecoDetectado,
    HuecoVisual,
)

logger = logging.getLogger("osa_vision.vision")


def _normalizar_estado(valor: Any) -> EstadoPosicion:
    """El modelo no siempre repite el literal exacto que pide el prompt (acentos,
    variantes en texto libre); se normaliza por palabra clave en vez de dejar
    que pydantic reviente con un ValidationError por un typo del LLM."""
    texto = str(valor or "").lower()
    if "parcial" in texto:
        return "parcial"
    if "incorrect" in texto or "surtido" in texto or "equivocad" in texto:
        return "surtido_incorrecto"
    return "vacio"

_SYSTEM_PROMPT_TEMPLATE = """Eres un asistente experto en auditoría de anaquel (on-shelf \
availability) para tiendas de autoservicio. Vas a comparar una foto de anaquel contra su \
planograma (lo que DEBERÍA tener cada posición) y reportar qué posiciones están vacías o \
mal surtidas.

Planograma de la sección "{nombre}":
{posiciones}

Para cada posición de la lista de arriba, revisa en la foto si el producto correcto está \
presente en la cantidad esperada, y clasifica lo que encuentres en uno de tres estados — \
no los mezcles, son causas distintas para el operador:

- "vacio": el espacio está físicamente vacío, sin ningún producto.
- "parcial": el producto/SKU correcto SÍ está, pero con menos piezas visibles que las \
piezas esperadas indicadas en el planograma.
- "surtido_incorrecto": el espacio tiene producto (no está vacío), pero es un producto/SKU \
distinto al que debería ir ahí — no es un hueco físico, es un error de acomodo.

Reporta solo las posiciones que no estén completas y correctas.

Responde ÚNICAMENTE con un JSON con esta forma exacta, sin texto adicional:
{{
  "resumen": "una frase breve describiendo el estado general del anaquel",
  "huecos": [
    {{
      "posicion_id": "el id EXACTO tal como aparece en el planograma de arriba, ej. 'P3'",
      "estado": "vacio, parcial o surtido_incorrecto",
      "confianza": 0.0
    }}
  ]
}}

No inventes ids que no estén en la lista del planograma. "confianza" es un número entre \
0.0 y 1.0. Si todas las posiciones están completas, responde con "huecos": [].
"""


_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@dataclass(frozen=True)
class _ReferenciaCategoria:
    id: str
    nombre: str
    archivo: str
    # nivel -> (producto(s) esperado(s), marca(s) esperada(s)), tal como en la
    # tabla "Distribución por nivel" impresa en la hoja de referencia de esa
    # categoría. Se pasa como texto además de la imagen para que el modelo no
    # tenga que releer la tabla cada vez y no pueda inventar niveles o
    # productos que no están en la hoja — el mismo principio anti-alucinación
    # que _prompt_planograma, aplicado aquí a nivel de categoría en vez de SKU.
    niveles: dict[int, tuple[str, str]]

    def distribucion_texto(self) -> str:
        return "\n".join(
            f"Nivel {nivel}: {producto} — {marca}"
            for nivel, (producto, marca) in sorted(self.niveles.items(), reverse=True)
        )


_REFERENCIAS: dict[str, _ReferenciaCategoria] = {
    "lacteos": _ReferenciaCategoria(
        id="lacteos",
        nombre="Lácteos",
        archivo="planograma_lacteos_referencia.png",
        niveles={
            6: ("Yogur griego", "Oikos, Danone Griego, Chobani"),
            5: ("Yogur bebible", "LALA, Yoplait, Danone, Alpura"),
            4: ("Yogur natural / batido", "LALA, Yoplait, Danone, Alpura"),
            3: ("Quesos untables / rebanados", "Philadelphia, FUD, La Villita, Zwan"),
            2: ("Quesos empacados", "Panela, Oaxaca, Asadero, Chihuahua"),
            1: ("Leches", "LALA, Alpura (Entera, 2%, Light, Deslactosada)"),
        },
    ),
    "cereales": _ReferenciaCategoria(
        id="cereales",
        nombre="Cereales para desayuno",
        archivo="planograma_cereales_referencia.png",
        niveles={
            5: (
                "Special K Original, Special K Frutos Rojos, Fitness Miel, Fitness Chocolate",
                "Kellogg's, Nestlé",
            ),
            4: ("Corn Flakes, Zucaritas, Choco Krispis", "Kellogg's"),
            3: ("Cheerios Miel, Cheerios Avena, Nesquik Cereal, Zucaritas", "Nestlé, Kellogg's"),
            2: (
                "Cereal La Comer Avena, Cereal La Comer Frutas, Corn Pops, Trix",
                "Marca Propia, Kellogg's, Nestlé",
            ),
            1: ("Avena Quaker 1kg, Avena Quaker 500g, Granola Natural", "Quaker, Nature Valley"),
        },
    ),
    "refrescos": _ReferenciaCategoria(
        id="refrescos",
        nombre="Refrescos (bebidas carbonatadas)",
        archivo="planograma_refrescos_referencia.png",
        niveles={
            6: (
                "Coca-Cola Original Lata, Coca-Cola Light Lata, Sprite Lata, Fanta Naranja Lata, "
                "Schweppes Lata, Sprite Sin Azúcar Lata (355ml)",
                "Coca-Cola, Sprite, Fanta, Schweppes",
            ),
            5: (
                "Coca-Cola 600ml, Coca-Cola Sin Azúcar 600ml, Sprite 600ml, Fanta Naranja 600ml, "
                "Fanta Uva 600ml, Fresca 600ml",
                "Coca-Cola, Sprite, Fanta, Fresca",
            ),
            4: (
                "Coca-Cola 2L, Coca-Cola Sin Azúcar 2L, Sprite 2L, Fanta Naranja 2L, Fanta Uva 2L, "
                "Manzanita Sol 2L",
                "Coca-Cola, Sprite, Fanta, Manzanita Sol",
            ),
            3: (
                "Pepsi 2L, Pepsi Black 2L, 7up 2L, Mirinda Naranja 2L, Mirinda Uva 2L, "
                "Jarritos (Sabores) 2L",
                "Pepsi, 7up, Mirinda, Jarritos",
            ),
            2: (
                "Coca-Cola 2.5L, Coca-Cola Sin Azúcar 2.5L, Sprite 2.5L, Fanta Naranja 2.5L, "
                "Fanta Uva 2.5L, Manzanita Sol 2.5L",
                "Coca-Cola, Sprite, Fanta, Manzanita Sol",
            ),
            1: (
                "Multipack Coca-Cola 6x600ml, Multipack Coca-Cola Sin Azúcar 6x600ml, "
                "Multipack Sprite 6x600ml, Multipack Fanta Naranja 6x600ml, Multipack Fanta Uva 6x600ml",
                "Coca-Cola, Sprite, Fanta",
            ),
        },
    ),
}


def listar_categorias_visuales() -> list[CategoriaVisual]:
    return [CategoriaVisual(id=r.id, nombre=r.nombre) for r in _REFERENCIAS.values()]


def cargar_referencia(categoria_id: str) -> tuple[bytes, str]:
    referencia = _REFERENCIAS[categoria_id]
    return (_STATIC_DIR / referencia.archivo).read_bytes(), "image/png"


_SYSTEM_PROMPT_REFERENCIA = """Eres un asistente experto en auditoría de anaquel (on-shelf \
availability) para tiendas de autoservicio. Vas a comparar DOS imágenes:

1. La primera imagen es el PLANOGRAMA DE REFERENCIA de la sección de {categoria}: una hoja \
con el anaquel completo tal como debería lucir, organizado en {num_niveles} niveles numerados \
(del 1, abajo, al {num_niveles}, arriba). Esta es la tabla de distribución por nivel de esa hoja:

{distribucion}

2. La segunda imagen es una FOTO REAL del mismo anaquel, tomada en tienda en este momento.

Compara nivel por nivel la foto real contra el planograma de referencia y detecta qué \
posiciones no coinciden con lo que debería haber, clasificando cada una en uno de tres \
estados — son causas distintas para el operador, no los mezcles:

- "vacio": el espacio está físicamente vacío, sin ningún producto.
- "parcial": las marcas correctas están presentes, pero con menos piezas que sus vecinos \
inmediatos del mismo nivel (falta reponer, no falta variedad).
- "surtido_incorrecto": el nivel está físicamente lleno de producto — NO es un hueco — pero \
falta alguna de las marcas/sabores que la tabla de distribución dice que debería haber ahí \
(por ejemplo, el nivel está lleno de una sola marca y le falta otra marca de la lista).

Antes de reportar algo, verifica: (a) que el nivel que estás mirando en la FOTO es realmente \
el mismo nivel que en la REFERENCIA — cuenta los niveles de abajo hacia arriba en ambas \
imágenes, no asumas por la marca; (b) si el espacio tiene producto, no lo reportes como \
"vacio" — usa "surtido_incorrecto" si el problema es que falta variedad de marca, no volumen.

Para "posicion", describe la ubicación contando columnas de productos de izquierda a derecha \
dentro de ese nivel en la FOTO (ej. "posiciones 4 a 6 de aproximadamente 14"), no por nombre \
de marca — no es confiable identificar marcas específicas a esta resolución y las marcas de \
la tabla ya identifican qué debería ir ahí. Puedes agregar una referencia visual corta si \
ayuda (ej. "justo después del grupo de envases rosas"), pero el conteo de posición es el dato \
principal.

Responde ÚNICAMENTE con un JSON con esta forma exacta, sin texto adicional:
{{
  "resumen": "una frase breve describiendo el estado general del anaquel comparado contra el planograma",
  "huecos": [
    {{
      "nivel": 1,
      "posicion": "descripción breve de dónde está el hueco dentro del nivel",
      "estado": "vacio, parcial o surtido_incorrecto",
      "confianza": 0.0
    }}
  ]
}}

"nivel" debe ser un entero entre 1 y {num_niveles}, tal como en la tabla de distribución. \
"confianza" es un número entre 0.0 y 1.0. Si el anaquel real coincide completamente con el \
planograma de referencia, responde con "huecos": [].
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
    logger.info(
        "vision.analizar_imagen seccion=%s posiciones_planograma=%d huecos_crudos=%d",
        planograma.seccion_id, len(planograma.posiciones), len(payload.get("huecos", [])),
    )
    posiciones_por_id = {p.id: p for p in planograma.posiciones}

    huecos: list[HuecoDetectado] = []
    for h in payload.get("huecos", []):
        posicion = posiciones_por_id.get(h.get("posicion_id"))
        if posicion is None:
            # El modelo alucinó un id que no existe en el planograma; se descarta
            # en vez de inventar un sku/producto que no podemos respaldar.
            logger.warning(
                "vision.analizar_imagen seccion=%s posicion_id desconocido, descartado: %r",
                planograma.seccion_id, h.get("posicion_id"),
            )
            continue
        huecos.append(
            HuecoDetectado(
                posicion_id=posicion.id,
                posicion=posicion.posicion,
                sku=posicion.sku,
                producto=posicion.producto,
                facings_esperados=posicion.facings_esperados,
                estado=_normalizar_estado(h.get("estado")),
                confianza=h.get("confianza", 0.5),
            )
        )

    conteo = Counter(h.estado for h in huecos)
    logger.info(
        "vision.analizar_imagen seccion=%s resultado: %d huecos (%s)",
        planograma.seccion_id, len(huecos), dict(conteo),
    )

    return AnalizarImagenResponse(
        seccion_id=planograma.seccion_id,
        resumen=payload.get("resumen", ""),
        huecos=huecos,
    )


def _data_url(image_bytes: bytes, content_type: str) -> str:
    return f"data:{content_type};base64,{base64.b64encode(image_bytes).decode('utf-8')}"


def analizar_con_referencia(
    categoria_id: str,
    imagen_planograma: bytes,
    tipo_planograma: str,
    imagen_anaquel: bytes,
    tipo_anaquel: str,
) -> AnalizarConReferenciaResponse:
    """Detecta huecos comparando una foto real del anaquel contra una foto del
    planograma de referencia de la categoría (en vez de un catálogo de
    posiciones/SKU en texto): no requiere digitalizar el planograma como
    catálogo, el propio modelo lo lee de la imagen — a cambio, el producto/marca
    esperado no queda verificado contra un catálogo (solo el nivel sí, contra
    `_REFERENCIAS[categoria_id].niveles`).
    """
    referencia = _REFERENCIAS[categoria_id]
    num_niveles = max(referencia.niveles)
    logger.info("vision.analizar_con_referencia categoria=%s num_niveles=%d", categoria_id, num_niveles)
    system_prompt = _SYSTEM_PROMPT_REFERENCIA.format(
        categoria=referencia.nombre,
        num_niveles=num_niveles,
        distribucion=referencia.distribucion_texto(),
    )

    completion = _client().chat.completions.create(
        model=settings.groq_vision_model,
        response_format={"type": "json_object"},
        reasoning_effort="none",
        reasoning_format="hidden",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Imagen 1 de 2 — planograma de referencia:"},
                    {"type": "image_url", "image_url": {"url": _data_url(imagen_planograma, tipo_planograma)}},
                    {"type": "text", "text": "Imagen 2 de 2 — foto real del anaquel. Compara y detecta los huecos."},
                    {"type": "image_url", "image_url": {"url": _data_url(imagen_anaquel, tipo_anaquel)}},
                ],
            },
        ],
    )

    payload: dict[str, Any] = json.loads(completion.choices[0].message.content)
    logger.info(
        "vision.analizar_con_referencia categoria=%s huecos_crudos=%d",
        categoria_id, len(payload.get("huecos", [])),
    )

    huecos: list[HuecoVisual] = []
    for i, h in enumerate(payload.get("huecos", []), start=1):
        nivel = h.get("nivel")
        datos_nivel = referencia.niveles.get(nivel)
        if datos_nivel is None:
            # El modelo alucinó un nivel fuera de rango; se descarta en vez de
            # inventar una categoría/marca que no está en la hoja de referencia.
            logger.warning(
                "vision.analizar_con_referencia categoria=%s nivel fuera de rango, descartado: %r",
                categoria_id, nivel,
            )
            continue
        categoria_esperada, marcas_esperadas = datos_nivel
        huecos.append(
            HuecoVisual(
                id=f"V-{i:02d}",
                nivel=nivel,
                posicion=h.get("posicion", f"Nivel {nivel}"),
                categoria_esperada=categoria_esperada,
                marcas_esperadas=marcas_esperadas,
                estado=_normalizar_estado(h.get("estado")),
                confianza=h.get("confianza", 0.5),
            )
        )

    conteo = Counter(h.estado for h in huecos)
    logger.info(
        "vision.analizar_con_referencia categoria=%s resultado: %d huecos (%s)",
        categoria_id, len(huecos), dict(conteo),
    )

    return AnalizarConReferenciaResponse(resumen=payload.get("resumen", ""), huecos=huecos)
