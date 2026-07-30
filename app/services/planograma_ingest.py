"""Ingesta de planogramas reales (PDF de La Comer) hacia el modelo
`Planograma`/`PosicionPlanograma` que ya usa el modo catálogo
(`vision_service.analizar_imagen`).

Los PDF de planograma vienen en pares de página por "tile" (segmento
horizontal del anaquel): una página fotorrealista (solo imagen, sin texto
útil) y su par "esquemático" con el UPC + marca de cada posición como texto
real embebido en el PDF — no hay que interpretar ninguna imagen con IA para
esto, es dato de texto extraíble con PyMuPDF.

Cada celda de producto en la página esquemática es un "bloque" de texto del
PDF (`words[...][5]`, el índice de bloque); casi siempre trae la etiqueta
completa (UPC + marca) en un solo bloque, pero cuando la columna es muy
angosta el renderizador la parte en 2-3 bloques contiguos que hay que volver
a unir (`_reparar_fragmentos`).

Uso:
    python -m app.services.planograma_ingest <ruta.pdf> --niveles 6 --tiles 5,6,7,8

Genera un JSON con la forma de `Planograma` en la ruta de --out (por default
junto al PDF, mismo nombre con sufijo _planograma.json).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import fitz

UPC_RE = re.compile(r"\d{8,14}")


def _celdas_de_pagina(page: "fitz.Page") -> list[dict]:
    """Reconstruye una celda por bloque de texto del PDF (aprox. una posición
    de producto), con su texto concatenado y su caja delimitadora."""
    words = page.get_text("words")  # (x0,y0,x1,y1,texto,bloque,linea,palabra)

    blocks: dict[int, list] = defaultdict(list)
    for w in words:
        blocks[w[5]].append(w)

    celdas = []
    for ws in blocks.values():
        ws.sort(key=lambda w: (w[6], w[7]))
        texto = "".join(w[4] for w in ws)
        x0 = min(w[0] for w in ws)
        y0 = min(w[1] for w in ws)
        x1 = max(w[2] for w in ws)
        y1 = max(w[3] for w in ws)
        celdas.append(
            {
                "texto": texto,
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "xc": (x0 + x1) / 2,
                "yc": (y0 + y1) / 2,
            }
        )
    return celdas


def _tiene_upc(texto: str) -> bool:
    m = UPC_RE.search(texto)
    return bool(m) and len(m.group(0)) >= 10


def _reparar_fragmentos(celdas: list[dict]) -> list[dict]:
    """Cuando una columna es muy angosta, el UPC+marca de una sola posición
    queda partido en 2-3 bloques contiguos (mismo carril vertical, sin UPC
    completo en ninguno). Los fusiona con sus vecinos más cercanos hasta
    que el texto combinado contenga un UPC completo, o hasta agotar
    candidatos cercanos."""
    pendientes = sorted(celdas, key=lambda c: (c["xc"], c["yc"]))
    usados = [False] * len(pendientes)
    resultado = []

    for i, c in enumerate(pendientes):
        if usados[i]:
            continue
        if _tiene_upc(c["texto"]):
            usados[i] = True
            resultado.append(c)
            continue

        grupo = [c]
        usados[i] = True
        texto_grupo = c["texto"]
        for j in range(i + 1, len(pendientes)):
            if usados[j] or _tiene_upc(texto_grupo):
                continue
            cj = pendientes[j]
            # candidato a fragmento hermano: mismo carril angosto (x cercana),
            # verticalmente adyacente (una franja de shelf, no un salto de nivel)
            if abs(cj["xc"] - c["xc"]) <= 6.0 and abs(cj["yc"] - c["yc"]) <= 20.0:
                grupo.append(cj)
                usados[j] = True
                texto_grupo += cj["texto"]

        x0 = min(g["x0"] for g in grupo)
        y0 = min(g["y0"] for g in grupo)
        x1 = max(g["x1"] for g in grupo)
        y1 = max(g["y1"] for g in grupo)
        resultado.append(
            {
                "texto": texto_grupo,
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "xc": (x0 + x1) / 2,
                "yc": (y0 + y1) / 2,
            }
        )
    return resultado


def _asignar_nivel(yc: float, y_top: float, y_bottom: float, num_niveles: int) -> int:
    """Convierte la coordenada vertical de una celda a un número de nivel
    (1 = charola inferior, num_niveles = charola superior), dividiendo el
    rango vertical observado del anaquel en `num_niveles` franjas iguales.
    """
    alto_nivel = (y_bottom - y_top) / num_niveles
    indice_desde_arriba = int((yc - y_top) // alto_nivel)
    nivel = num_niveles - indice_desde_arriba
    return max(1, min(num_niveles, nivel))


def _extraer_celdas_tile(page: "fitz.Page") -> list[dict]:
    """Celdas con UPC/marca ya separados para un tile, SIN nivel asignado
    todavía — el nivel se calcula después, con el rango vertical de TODOS los
    tiles juntos (ver construir_planograma). Calibrar cada tile por separado
    fue el bug original: cada página tiene su propia densidad de texto, así
    que su rango vertical detectado variaba tile a tile aunque los 4 sean
    cortes horizontales del mismo anaquel físico — terminaba llamando "nivel
    7" a alturas distintas según el tile.
    """
    celdas = _celdas_de_pagina(page)
    celdas = _reparar_fragmentos(celdas)
    celdas = [c for c in celdas if _tiene_upc(c["texto"])]

    resultado = []
    for c in celdas:
        m = UPC_RE.search(c["texto"])
        upc = m.group(0)
        resto = (c["texto"][: m.start()] + c["texto"][m.end() :]).strip()
        resultado.append(
            {
                "yc": c["yc"],
                "y0": c["y0"],
                "y1": c["y1"],
                "x0": c["x0"],
                "x1": c["x1"],
                "sku": upc,
                "marca": resto or "(sin marca)",
            }
        )
    return resultado


def _ancho_unidad_por_nivel(posiciones: list[dict]) -> dict[int, float]:
    """Estima el ancho de "una sola posición" en cada nivel usando el ancho
    mínimo observado en ese nivel — las corridas de facings repetidos del
    mismo SKU vienen fusionadas en una sola celda ancha, así que dividir su
    ancho entre este valor aproxima cuántos facings representa.
    """
    anchos: dict[int, list[float]] = defaultdict(list)
    for p in posiciones:
        anchos[p["nivel"]].append(p["x1"] - p["x0"])
    return {nivel: min(ws) for nivel, ws in anchos.items()}


def construir_planograma(
    pdf_path: Path, seccion_id: str, nombre: str, tiles: list[int], num_niveles: int
) -> dict:
    doc = fitz.open(pdf_path)
    crudas: list[dict] = []
    offset_x = 0.0
    for pagina_idx in tiles:
        page = doc[pagina_idx - 1]
        celdas = _extraer_celdas_tile(page)
        for c in celdas:
            c["x0"] += offset_x
            c["x1"] += offset_x
        crudas.extend(celdas)
        offset_x += page.rect.width  # tiles consecutivos, no se solapan
    doc.close()

    if not crudas:
        return {"seccion_id": seccion_id, "nombre": nombre, "posiciones": []}

    # Rango vertical calibrado con TODOS los tiles juntos — así "nivel 7"
    # significa la misma charola física sin importar de qué tile venga.
    y_top = min(c["y0"] for c in crudas)
    y_bottom = max(c["y1"] for c in crudas)
    for c in crudas:
        c["nivel"] = _asignar_nivel(c["yc"], y_top, y_bottom, num_niveles)

    ancho_unidad = _ancho_unidad_por_nivel(crudas)

    # ordenar por nivel (de mayor a menor, como se lee un anaquel) y luego x
    crudas.sort(key=lambda p: (-p["nivel"], p["x0"]))

    posiciones_finales = []
    contador_por_nivel: dict[int, int] = defaultdict(int)
    for p in crudas:
        contador_por_nivel[p["nivel"]] += 1
        col = contador_por_nivel[p["nivel"]]
        ancho = p["x1"] - p["x0"]
        unidad = ancho_unidad.get(p["nivel"], ancho) or ancho
        facings = max(1, round(ancho / unidad)) if unidad else 1
        posiciones_finales.append(
            {
                "id": f"N{p['nivel']}-P{col}",
                "posicion": f"Nivel {p['nivel']}, columna {col}",
                "sku": p["sku"],
                "producto": p["marca"],
                "facings_esperados": facings,
                "nivel": p["nivel"],
                "columna": col,
            }
        )

    return {"seccion_id": seccion_id, "nombre": nombre, "posiciones": posiciones_finales}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--seccion-id", default=None)
    parser.add_argument("--nombre", default=None)
    parser.add_argument("--niveles", type=int, required=True)
    parser.add_argument("--tiles", default="", help="páginas esquemáticas separadas por coma, ej. 5,6,7,8")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    tiles = [int(t) for t in args.tiles.split(",") if t.strip()]
    seccion_id = args.seccion_id or args.pdf.stem.lower().replace("_", "-")
    nombre = args.nombre or args.pdf.stem

    planograma = construir_planograma(args.pdf, seccion_id, nombre, tiles, args.niveles)

    out_path = args.out or args.pdf.with_name(args.pdf.stem + "_planograma.json")
    out_path.write_text(json.dumps(planograma, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(planograma['posiciones'])} posiciones -> {out_path}")


if __name__ == "__main__":
    main()
