from app.models.planograma import Planograma, PosicionPlanograma

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
    "abarrotes-7a": Planograma(
        seccion_id="abarrotes-7a",
        nombre="Abarrotes 7-A",
        posiciones=[
            PosicionPlanograma(id="P1", posicion="Charola 1, posición 1", sku="720045-011", producto="Aceite 123 900ml", facings_esperados=6),
            PosicionPlanograma(id="P2", posicion="Charola 2, posición 1", sku="720050-022", producto="Arroz Verde Valle 1kg", facings_esperados=5),
            PosicionPlanograma(id="P3", posicion="Charola 3, posición 1", sku="720060-033", producto="Frijol La Costeña 900g", facings_esperados=5),
        ],
    ),
}


def obtener(seccion_id: str) -> Planograma | None:
    return _PLANOGRAMAS.get(seccion_id)


def listar() -> list[Planograma]:
    return list(_PLANOGRAMAS.values())
