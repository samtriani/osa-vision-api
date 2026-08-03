from typing import Literal

from pydantic import BaseModel, Field

EstadoPosicion = Literal["vacio", "parcial", "sobrante", "surtido_incorrecto"]


class HuecoDetectado(BaseModel):
    posicion_id: str
    posicion: str
    sku: str
    producto: str
    # El planograma solo trae la MARCA (todas las variantes de Suavitel dicen
    # "SUAVITEL"), así que el operador no puede saber a cuál de los diez
    # frascos de la charola se refiere. Esto lo llena el modelo describiendo el
    # envase tal como se ve en la imagen de referencia ("bolsa Rellena Pack
    # amarilla 1.3L"), que es como el operador realmente lo identifica en piso.
    descripcion: str = ""
    facings_esperados: int
    piezas_detectadas: int | None = None
    # detectadas - esperadas: negativo = faltan piezas, positivo = sobran.
    diferencia: int | None = None
    estado: EstadoPosicion
    confianza: float = Field(ge=0.0, le=1.0)


class AnalizarImagenResponse(BaseModel):
    seccion_id: str
    resumen: str
    huecos: list[HuecoDetectado]


class HuecoVisual(BaseModel):
    """Hueco detectado comparando dos fotos (planograma de referencia vs. anaquel
    real), sin catálogo de SKU de por medio. A diferencia de HuecoDetectado, el
    producto/categoría esperados vienen de lo que el modelo lee en la propia foto
    de referencia, no de una fuente verificada — ver vision_service.analizar_con_referencia.
    """

    id: str
    nivel: int = Field(ge=1, le=6)
    posicion: str
    categoria_esperada: str
    marcas_esperadas: str
    estado: EstadoPosicion
    confianza: float = Field(ge=0.0, le=1.0)


class AnalizarConReferenciaResponse(BaseModel):
    resumen: str
    huecos: list[HuecoVisual]


class CategoriaVisual(BaseModel):
    id: str
    nombre: str
