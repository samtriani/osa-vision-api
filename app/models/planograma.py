from pydantic import BaseModel, Field


class PosicionPlanograma(BaseModel):
    id: str
    posicion: str
    sku: str
    producto: str
    facings_esperados: int = Field(ge=1)


class Planograma(BaseModel):
    seccion_id: str
    nombre: str
    posiciones: list[PosicionPlanograma]
