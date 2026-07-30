import json
from pathlib import Path

from app.models.planograma import Planograma, PosicionPlanograma

# Planogramas reales, digitalizados con app.services.planograma_ingest a partir
# de los PDF oficiales de planograma (UPC/marca extraídos como texto, no
# adivinados por un VLM) — ver app/services/planograma_ingest.py.
_PLANOGRAMAS_REALES_DIR = Path(__file__).resolve().parent.parent / "static" / "planogramas"


def _cargar_planogramas_reales() -> dict[str, Planograma]:
    reales: dict[str, Planograma] = {}
    if not _PLANOGRAMAS_REALES_DIR.is_dir():
        return reales
    for archivo in sorted(_PLANOGRAMAS_REALES_DIR.glob("*.json")):
        datos = json.loads(archivo.read_text(encoding="utf-8"))
        reales[datos["seccion_id"]] = Planograma(**datos)
    return reales

# En memoria, uno por sección (mismas secciones que usa el frontend en las
# "muestras" de captura). Los SKU de lácteos 4-B coinciden con los huecos
# dummy que ya usa el frontend, para que el demo cuadre visualmente.
_PLANOGRAMAS: dict[str, Planograma] = {
    "lacteos-4b": Planograma(
        seccion_id="lacteos-4b",
        nombre="Lácteos 4-B",
        posiciones=[
            PosicionPlanograma(id="P1", posicion="Charola 1, posición 1", sku="750102-201", producto="Alpura Entera 1L", facings_esperados=4),
            PosicionPlanograma(id="P2", posicion="Charola 1, posición 2", sku="750102-210", producto="Alpura Deslactosada 1L", facings_esperados=4),
            PosicionPlanograma(id="P3", posicion="Charola 1, posición 3", sku="750102-334", producto="Alpura Light 1L", facings_esperados=4),
            PosicionPlanograma(id="P4", posicion="Charola 2, posición 1", sku="750102-410", producto="Lala Entera 1L", facings_esperados=3),
            PosicionPlanograma(id="P5", posicion="Charola 2, posición 2", sku="750102-518", producto="Alpura Yogurt Griego 1kg", facings_esperados=3),
            PosicionPlanograma(id="P6", posicion="Charola 3, posición 4", sku="750301-072", producto="Philadelphia 190g", facings_esperados=3),
        ],
    ),
    "lacteos-4c": Planograma(
        seccion_id="lacteos-4c",
        nombre="Lácteos 4-C",
        posiciones=[
            PosicionPlanograma(id="P1", posicion="Charola 1, posición 1", sku="750110-050", producto="Yoplait Fresa 1kg", facings_esperados=3),
            PosicionPlanograma(id="P2", posicion="Charola 1, posición 2", sku="750110-060", producto="Yoplait Natural 1kg", facings_esperados=3),
            PosicionPlanograma(id="P3", posicion="Charola 2, posición 1", sku="750115-020", producto="Danone Griego 900g", facings_esperados=4),
        ],
    ),
}
_PLANOGRAMAS.update(_cargar_planogramas_reales())


def obtener(seccion_id: str) -> Planograma | None:
    return _PLANOGRAMAS.get(seccion_id)


def listar() -> list[Planograma]:
    return list(_PLANOGRAMAS.values())


def cargar_imagen_referencia(seccion_id: str) -> bytes | None:
    """Foto fotorrealista del propio PDF de planograma (si existe) para esa
    sección — puramente ilustrativa: la comparación real sigue siendo por
    catálogo de SKU, esto es solo para que el operador vea de un vistazo cómo
    debería lucir el anaquel."""
    archivo = _PLANOGRAMAS_REALES_DIR / f"{seccion_id}.png"
    if not archivo.is_file():
        return None
    return archivo.read_bytes()
