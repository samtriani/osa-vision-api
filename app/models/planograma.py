from pydantic import BaseModel, Field


class PosicionPlanograma(BaseModel):
    id: str
    posicion: str
    sku: str
    producto: str
    facings_esperados: int = Field(ge=1)
    # Opcionales: solo los trae el catálogo digitalizado desde PDF real
    # (planograma_ingest.py). Permiten que el modelo ubique el hueco contando
    # nivel/columna en la foto en vez de tener que adivinar un id exacto entre
    # cientos de posiciones — ver vision_service.analizar_imagen.
    nivel: int | None = None
    columna: int | None = None


class Planograma(BaseModel):
    seccion_id: str
    nombre: str
    posiciones: list[PosicionPlanograma]
