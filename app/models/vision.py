from typing import Literal

from pydantic import BaseModel, Field

EstadoPosicion = Literal["vacio", "parcial"]


class HuecoDetectado(BaseModel):
    posicion_id: str
    posicion: str
    sku: str
    producto: str
    facings_esperados: int
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
