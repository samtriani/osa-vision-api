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
