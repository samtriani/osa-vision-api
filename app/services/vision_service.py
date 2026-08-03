import base64
import json
import logging
from collections import Counter
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from groq import APIStatusError, Groq
from PIL import Image, ImageChops

from app.core.config import settings
from app.models.planograma import Planograma
from app.services.planograma_service import (
    cargar_imagen_referencia as cargar_imagen_referencia_seccion,
)
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
    if "sobra" in texto or "exceso" in texto:
        return "sobrante"
    if "incorrect" in texto or "surtido" in texto or "equivocad" in texto:
        return "surtido_incorrecto"
    return "vacio"


def _estado_por_conteo(
    estado_modelo: EstadoPosicion, detectadas: int | None, esperadas: int
) -> EstadoPosicion:
    """Cuando el modelo sí contó piezas, el conteo manda sobre la etiqueta que
    escribió: es común que reporte "vacio" en una posición donde él mismo contó
    2 de 4 piezas (lo que en realidad es "parcial"). `surtido_incorrecto` es la
    excepción — habla de la identidad del producto, no de cuántos hay, así que
    no se puede derivar de un conteo."""
    if estado_modelo == "surtido_incorrecto" or detectadas is None:
        return estado_modelo
    if detectadas <= 0:
        return "vacio"
    if detectadas < esperadas:
        return "parcial"
    if detectadas > esperadas:
        return "sobrante"
    return estado_modelo


_SYSTEM_PROMPT_TEMPLATE = """Auditor de anaquel (on-shelf availability) de autoservicio. \
Compara la foto real contra el planograma y reporta solo las posiciones que NO estén \
completas y correctas.

Planograma "{nombre}" (formato `P<columna> <MARCA> <SKU> x<piezas esperadas>`):
{posiciones}

Estados (son causas distintas, no los mezcles):
- "vacio": 0 piezas, espacio físicamente vacío.
- "parcial": el SKU correcto está, pero con MENOS piezas de las esperadas.
- "sobrante": hay MÁS piezas de las esperadas (se come el espacio del vecino).
- "surtido_incorrecto": hay producto, pero es un SKU distinto al que va ahí.

Por cada posición reportada:
- "piezas_detectadas": cuántas piezas cuentas realmente en la foto (0 si vacío).
- "descripcion_visual": el envase en 3-6 palabras (color, formato, tamaño), ej. "bolsa \
amarilla 1.3L", "garrafón azul 4.8L", "frasco blanco Baby". Casi todas las posiciones \
repiten la misma marca, así que sin esto el operador no sabe a cuál te refieres. Descríbelo \
por lo que ves; no inventes nombres que no puedas leer.
- "nivel" (1 = charola inferior) y "columna_aproximada" (contando productos de izquierda a \
derecha en ese nivel, desde 1), contados sobre la foto. "posicion_id" solo si lo identificas \
con certeza; si no, null.

Sé conservador: que un empaque te resulte difícil de reconocer NO significa que falte. \
Verifica que ESA posición se vea vacía o con menos piezas que sus vecinas antes de \
reportarla. Si el anaquel está completo, responde "huecos": [].

Responde ÚNICAMENTE este JSON:
{{
  "resumen": "2-3 frases: cuántas posiciones con problema, en qué niveles y qué patrón",
  "huecos": [{{"nivel": 1, "columna_aproximada": 1, "posicion_id": null, \
"descripcion_visual": "", "piezas_detectadas": 0, "estado": "vacio", "confianza": 0.0}}]
}}
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


def _mensaje_groq(exc: APIStatusError) -> str:
    """El body de error de Groq trae el detalle util (p.ej. cuanto falta para
    que se libere la cuota); se usa tal cual si esta disponible en vez de
    conformarse con el mensaje generico de la excepcion."""
    body = exc.body
    if isinstance(body, dict):
        detalle = body.get("error", {}).get("message")
        if detalle:
            return str(detalle)
    return exc.message


def _client() -> Groq:
    if not settings.groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY no está configurada. Define GROQ_API_KEY en tu archivo .env."
        )
    # Sin esto el SDK reintenta solo hasta 2 veces ante un 429/5xx, esperando
    # el Retry-After de Groq entre cada uno (varios segundos) -- inutil contra
    # un 429 de cuota diaria agotada, que no se libera en ese lapso; mejor
    # fallar rapido y dejar que _mensaje_groq de la razon real de una vez.
    return Groq(api_key=settings.groq_api_key, max_retries=0)


def _prompt_planograma(planograma: Planograma) -> str:
    """Catálogo compacto, agrupado por nivel. Una línea por posición gastaba
    ~95 caracteres repitiendo "Nivel 5, columna 1" cuando eso ya está implícito
    en el id (N5-P1) y en el encabezado del nivel; en un tramo de 60 posiciones
    eso era ~1500 tokens de puro relleno por llamada, contra un límite de
    8000 TPM en Groq. Formato: `P<col> <MARCA> <SKU> x<piezas>`."""
    por_nivel: dict[int | None, list[str]] = {}
    for p in planograma.posiciones:
        por_nivel.setdefault(p.nivel, []).append(
            f"P{p.columna if p.columna is not None else '?'} {p.producto} "
            f"{p.sku} x{p.facings_esperados}"
        )
    lineas = []
    for nivel, items in sorted(por_nivel.items(), key=lambda kv: -(kv[0] or 0)):
        encabezado = f"Nivel {nivel}" if nivel is not None else "Posiciones"
        lineas.append(f"{encabezado}: " + " | ".join(items))
    return "\n".join(lineas)


_LADO_MAXIMO_IMAGEN = 1024
"""El costo en tokens de vision escala con la resolucion de la imagen. El
frontend dice que redimensiona a 1280px antes de subir pero nunca se
implemento -- las fotos llegaban tal cual las tomaba el celular (varios MP),
y las imagenes de referencia estaticas (hasta 1.6MP) se mandaban completas en
cada llamada. Con la cuota diaria de Groq esto se notaba rapido."""

_LADO_MAXIMO_REFERENCIA = 800
"""La lamina de planograma va como apoyo (para saber que envase corresponde a
cada posicion), no como la imagen a auditar, asi que baja mas que la foto real
para no comerse el limite de 8000 tokens por minuto. Como ademas se le recortan
los margenes de la pagina, a 800px el anaquel se ve mas grande que antes a
1024px con titulo, logos y regla incluidos."""


def _redimensionar_para_modelo(
    image_bytes: bytes, content_type: str, lado_maximo: int = _LADO_MAXIMO_IMAGEN
) -> tuple[bytes, str]:
    """Reduce el lado mayor a `lado_maximo` y recomprime a JPEG antes de
    mandarla al modelo -- no afecta lo que se le muestra al operador (los
    endpoints de referencia siguen sirviendo el archivo original), solo lo que
    se manda a Groq."""
    imagen = Image.open(BytesIO(image_bytes))
    ancho, alto = imagen.size
    if max(ancho, alto) <= lado_maximo:
        return image_bytes, content_type
    factor = lado_maximo / max(ancho, alto)
    imagen = imagen.convert("RGB").resize(
        (round(ancho * factor), round(alto * factor)), Image.LANCZOS
    )
    buffer = BytesIO()
    imagen.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue(), "image/jpeg"


def _recortar_margenes(image_bytes: bytes) -> bytes:
    """La lámina de planograma es una página carta completa: título, logos,
    regla graduada y mucho margen blanco. Todo eso consume tokens de visión
    igual que el anaquel pero no aporta nada a la comparación, y con el límite
    de 8000 TPM de Groq pesa. Se recorta al bounding box de lo no-blanco, que
    quita los márgenes sin depender de coordenadas fijas (si el recorte falla o
    sale degenerado se devuelve la imagen tal cual)."""
    try:
        imagen = Image.open(BytesIO(image_bytes)).convert("RGB")
        fondo = Image.new("RGB", imagen.size, (255, 255, 255))
        caja = ImageChops.difference(imagen, fondo).getbbox()
        if caja is None:
            return image_bytes
        ancho, alto = caja[2] - caja[0], caja[3] - caja[1]
        if ancho < imagen.width * 0.2 or alto < imagen.height * 0.2:
            return image_bytes
        buffer = BytesIO()
        imagen.crop(caja).save(buffer, format="JPEG", quality=85)
        return buffer.getvalue()
    except OSError:
        return image_bytes


def analizar_imagen(
    image_bytes: bytes, content_type: str, planograma: Planograma
) -> AnalizarImagenResponse:
    image_bytes, content_type = _redimensionar_para_modelo(image_bytes, content_type)
    data_url = f"data:{content_type};base64,{base64.b64encode(image_bytes).decode('utf-8')}"
    system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
        nombre=planograma.nombre, posiciones=_prompt_planograma(planograma)
    )

    # El catálogo en texto solo dice la MARCA de cada posición ("SUAVITEL"),
    # que se repite decenas de veces en la misma charola: con eso el modelo no
    # puede distinguir el frasco amarillo del blanco y termina reportando como
    # vacías posiciones que sí están surtidas. Mandarle además la lámina
    # fotorrealista del planograma le da el aspecto de cada posición, que es lo
    # que le faltaba para comparar de verdad. El SKU/producto de la respuesta
    # se sigue tomando del catálogo, no de la imagen — sin riesgo de alucinar.
    referencia = (
        cargar_imagen_referencia_seccion(planograma.seccion_id)
        if settings.vision_enviar_referencia
        else None
    )
    contenido_usuario: list[dict[str, Any]] = []
    if referencia is not None:
        ref_bytes, ref_tipo = _redimensionar_para_modelo(
            _recortar_margenes(referencia), "image/jpeg", _LADO_MAXIMO_REFERENCIA
        )
        contenido_usuario += [
            {
                "type": "text",
                "text": (
                    "Imagen 1 de 2 — lámina del planograma: así DEBERÍA verse este anaquel. "
                    "Úsala para saber qué envase corresponde a cada posición del catálogo."
                ),
            },
            {"type": "image_url", "image_url": {"url": _data_url(ref_bytes, ref_tipo)}},
            {
                "type": "text",
                "text": (
                    "Imagen 2 de 2 — foto real del anaquel en tienda. Compárala contra la "
                    "lámina y el catálogo, y detecta los huecos."
                ),
            },
        ]
    else:
        contenido_usuario.append(
            {
                "type": "text",
                "text": "Compara esta foto del anaquel contra el planograma y detecta los huecos.",
            }
        )
    contenido_usuario.append({"type": "image_url", "image_url": {"url": data_url}})

    try:
        completion = _client().chat.completions.create(
            model=settings.groq_vision_model,
            response_format={"type": "json_object"},
            # qwen3.6-27b es un modelo "thinking" por defecto: sin esto, gasta el
            # presupuesto de tokens razonando y el JSON final sale vacío/truncado.
            reasoning_effort="none",
            reasoning_format="hidden",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": contenido_usuario},
            ],
        )
    except APIStatusError as exc:
        if exc.status_code == 413:
            raise RuntimeError(
                "Esta sección tiene demasiadas posiciones para analizarla en una sola foto "
                f"(catálogo de {len(planograma.posiciones)} posiciones excede el límite de "
                "tamaño del modelo). Prueba con una sección más chica o un módulo más angosto."
            ) from exc
        if exc.status_code == 429:
            raise RuntimeError(f"Límite de uso del modelo de visión alcanzado: {_mensaje_groq(exc)}") from exc
        raise

    payload: dict[str, Any] = json.loads(completion.choices[0].message.content)
    logger.info(
        "vision.analizar_imagen seccion=%s posiciones_planograma=%d huecos_crudos=%d",
        planograma.seccion_id, len(planograma.posiciones), len(payload.get("huecos", [])),
    )
    posiciones_por_id = {p.id: p for p in planograma.posiciones}
    posiciones_por_nivel_col = {
        (p.nivel, p.columna): p for p in planograma.posiciones if p.nivel is not None and p.columna is not None
    }

    huecos: list[HuecoDetectado] = []
    for h in payload.get("huecos", []):
        # Con catálogos grandes (cientos de posiciones) es mucho más confiable
        # que el modelo cuente nivel/columna sobre la foto a que acierte un id
        # exacto de memoria — se intenta primero por (nivel, columna) y solo se
        # cae a posicion_id como respaldo (catálogos chicos tipo demo, sin
        # nivel/columna, donde sí funciona bien matchear por id).
        posicion = posiciones_por_nivel_col.get((h.get("nivel"), h.get("columna_aproximada")))
        if posicion is None:
            posicion = posiciones_por_id.get(h.get("posicion_id"))
        if posicion is None:
            # El modelo alucinó una posición que no existe en el planograma; se
            # descarta en vez de inventar un sku/producto que no podemos respaldar.
            logger.warning(
                "vision.analizar_imagen seccion=%s posicion desconocida, descartada: nivel=%r columna=%r posicion_id=%r",
                planograma.seccion_id, h.get("nivel"), h.get("columna_aproximada"), h.get("posicion_id"),
            )
            continue
        detectadas = h.get("piezas_detectadas")
        detectadas = int(detectadas) if isinstance(detectadas, (int, float)) else None
        if detectadas is not None:
            detectadas = max(0, detectadas)
        estado = _estado_por_conteo(
            _normalizar_estado(h.get("estado")), detectadas, posicion.facings_esperados
        )
        huecos.append(
            HuecoDetectado(
                posicion_id=posicion.id,
                posicion=posicion.posicion,
                sku=posicion.sku,
                producto=posicion.producto,
                descripcion=str(h.get("descripcion_visual") or "").strip(),
                facings_esperados=posicion.facings_esperados,
                piezas_detectadas=detectadas,
                diferencia=(
                    detectadas - posicion.facings_esperados if detectadas is not None else None
                ),
                estado=estado,
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

    imagen_planograma, tipo_planograma = _redimensionar_para_modelo(imagen_planograma, tipo_planograma)
    imagen_anaquel, tipo_anaquel = _redimensionar_para_modelo(imagen_anaquel, tipo_anaquel)

    contenido_usuario = [
        {"type": "text", "text": "Imagen 1 de 2 — planograma de referencia:"},
        {"type": "image_url", "image_url": {"url": _data_url(imagen_planograma, tipo_planograma)}},
        {"type": "text", "text": "Imagen 2 de 2 — foto real del anaquel. Compara y detecta los huecos."},
        {"type": "image_url", "image_url": {"url": _data_url(imagen_anaquel, tipo_anaquel)}},
    ]

    try:
        completion = _client().chat.completions.create(
            model=settings.groq_vision_model,
            response_format={"type": "json_object"},
            reasoning_effort="none",
            reasoning_format="hidden",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": contenido_usuario},
            ],
        )
    except APIStatusError as exc:
        if exc.status_code == 413:
            raise RuntimeError(
                "Las imágenes son demasiado grandes para analizarlas juntas. Prueba con fotos más chicas."
            ) from exc
        if exc.status_code == 429:
            raise RuntimeError(f"Límite de uso del modelo de visión alcanzado: {_mensaje_groq(exc)}") from exc
        raise

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
