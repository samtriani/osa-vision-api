"""Ingesta de planogramas reales (PDF de La Comer) hacia el modelo
`Planograma`/`PosicionPlanograma` que ya usa el modo catálogo
(`vision_service.analizar_imagen`).

Cada página "tile" del PDF (segmento horizontal del anaquel) es una
composición vectorial: cada facing físico trae su propio rectángulo delgado
(el marco de la celda, dibujado como trazo `re`) y el UPC/marca de cada
posición es texto real embebido (extraíble con PyMuPDF, sin IA de por medio).

La extracción NO se apoya en el "bloque" de texto que asigna PyMuPDF
(`words[...][5]`): en columnas angostas con texto rotado 90° el propio PDF
mete en un mismo bloque texto de dos niveles físicos distintos (se
comprobó comparando contra el PDF de suavizantes 287-14-B-M1). En su lugar:

1. Se toman las palabras individuales (con su rect propio) y se reconstruye
   cada etiqueta UPC+marca encadenando fragmentos contiguos (mismo nivel,
   separación menor a `GAP_MAX`) hasta completar 13 dígitos y absorber la
   marca que sigue pegada — ver `_agrupar_en_etiquetas`.
2. El número de facings de cada etiqueta NO se estima por ancho de texto
   (ese heurístico fallaba: subestimaba/sobrestimaba según el ancho mínimo
   observado en el nivel); se cuentan los rectángulos-facing reales que le
   quedan más cerca — ver `_contar_facings`.

Uso:
    python -m app.services.planograma_ingest <ruta.pdf> --niveles 6 --tiles 5,6,7,8

Genera un JSON con la forma de `Planograma` en la ruta de --out (por default
junto al PDF, mismo nombre con sufijo _planograma.json).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import fitz

UPC_RE = re.compile(r"\d{8,14}")
GAP_MAX = 4.0  # separación máxima (pt) entre fragmentos para tratarlos como la misma etiqueta

# El PDF a veces no repite el nombre completo de la marca en facings
# consecutivos del mismo SKU (p.ej. sólo "HILLS" en vez de "GOLDEN HILLS" en
# la 2a/3a repetición) — no es un fragmento que se pueda reconstruir por
# geometría porque el texto completo simplemente no está en la página.
_MARCA_TRUNCADA = {
    "HILLS": "GOLDEN HILLS",
}


def _rect_gap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Distancia entre dos rects (x0,y0,x1,y1); 0 si se tocan o se superponen."""
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return max(dx, dy)


def _palabras_producto(page: "fitz.Page") -> list[tuple]:
    """Palabras del área de producto, descartando la regla graduada de los
    márgenes (izquierda/inferior) y el encabezado/pie de página — esos
    dígitos sueltos (p.ej. dos números de la regla concatenados) cuelan
    "SKU" falsos y descalibran el rango vertical usado para asignar nivel."""
    return [w for w in page.get_text("words") if w[2] > 100 and w[1] < 490 and w[3] > 95]


def _slots_de_facing(page: "fitz.Page") -> list["fitz.Rect"]:
    """Cada facing dibujado en el planograma trae su propio rectángulo (el
    marco de la celda); su ancho varía según el tamaño físico del envase
    (de ~14pt para un frasco chico a ~43pt para un garrafón de 4.8L), pero
    siempre son mucho más angostos/bajos que los separadores de nivel, los
    divisores de módulo o el fondo blanco, así que se distinguen por tamaño."""
    out = []
    for d in page.get_drawings():
        if d["type"] != "s":
            continue
        r = d["rect"]
        w, h = r.x1 - r.x0, r.y1 - r.y0
        if 5 < w < 60 and 5 < h < 60 and r.x0 > 100 and r.y1 < 490 and r.y0 > 95:
            out.append(r)
    return out


def _agrupar_en_etiquetas(palabras: list[dict]) -> list[dict]:
    """Reconstruye cada etiqueta UPC+marca a partir de palabras individuales
    de un mismo nivel, encadenando por proximidad geométrica en vez de por
    bloque de PDF. Primero junta fragmentos de dígitos contiguos hasta
    completar un UPC de 13 dígitos, luego junta la marca (letras, o dígitos
    cortos tipo el "3" de "BOLD 3") que sigue pegada; se detiene apenas
    aparece un fragmento que ya parece el UPC del siguiente producto."""
    ws = sorted(palabras, key=lambda w: (w["r"][0], w["r"][1]))
    i, n = 0, len(ws)
    etiquetas = []
    while i < n:
        grupo = [ws[i]]
        digitos = re.sub(r"\D", "", ws[i]["texto"])
        j = i + 1
        while (
            len(digitos) < 13
            and j < n
            and _rect_gap(grupo[-1]["r"], ws[j]["r"]) <= GAP_MAX
            and ws[j]["texto"].isdigit()
        ):
            grupo.append(ws[j])
            digitos += ws[j]["texto"]
            j += 1
        while j < n and _rect_gap(grupo[-1]["r"], ws[j]["r"]) <= GAP_MAX:
            cand = ws[j]["texto"]
            if cand.isdigit() and len(cand) >= 8:
                break  # ya empezó el UPC del siguiente producto
            grupo.append(ws[j])
            j += 1
        # Los fragmentos de UN MISMO número van pegados sin espacio (para que
        # el regex del UPC los vea como una sola corrida de dígitos); entre
        # cualquier otro par de palabras (marca de 2 palabras tipo "GOLDEN
        # HILLS", o dígito-corto-pegado-a-marca tipo "BOLD 3") sí va espacio.
        texto = grupo[0]["texto"]
        for prev, cur in zip(grupo, grupo[1:]):
            if not (prev["texto"].isdigit() and cur["texto"].isdigit()):
                texto += " "
            texto += cur["texto"]
        m = UPC_RE.search(texto)
        if m:
            xs = [g["r"][0] for g in grupo] + [g["r"][2] for g in grupo]
            resto = (texto[: m.start()] + texto[m.end() :]).strip()
            producto = _MARCA_TRUNCADA.get(resto, resto) or "(sin marca)"
            etiquetas.append({"sku": m.group(0), "producto": producto, "x0": min(xs), "x1": max(xs)})
        i = j
    return etiquetas


def _contar_facings(etiquetas: list[dict], slots: list["fitz.Rect"]) -> dict[int, int]:
    """A cada facing dibujado se le asigna la etiqueta cuyo texto tiene más
    cerca (distancia 0 si el centro del facing cae dentro del propio
    bounding box del texto). Es más confiable que repartir por ancho
    estimado: el centro de un facing en el extremo de una etiqueta ancha
    puede caer numéricamente más cerca del centroide de la etiqueta VECINA
    que del propio, así que se compara contra el rango [x0,x1] completo de
    cada etiqueta, no contra su centro."""
    conteo: dict[int, int] = {}

    def distancia(e: dict, xc: float) -> float:
        if e["x0"] <= xc <= e["x1"]:
            return 0.0
        return min(abs(xc - e["x0"]), abs(xc - e["x1"]))

    for s in sorted(slots, key=lambda r: r.x0):
        if not etiquetas:
            continue
        xc = (s.x0 + s.x1) / 2
        idx = min(range(len(etiquetas)), key=lambda i: distancia(etiquetas[i], xc))
        conteo[idx] = conteo.get(idx, 0) + 1
    return conteo


def construir_planograma(
    pdf_path: Path, seccion_id: str, nombre: str, tiles: list[int], num_niveles: int
) -> dict:
    doc = fitz.open(pdf_path)

    # Calibración de niveles compartida entre TODOS los tiles de esta corrida:
    # calibrar cada tile por separado hace que "nivel 5" caiga en una altura
    # distinta si un tile tiene menos contenido cerca del borde que otro,
    # aunque los tiles sean cortes horizontales del mismo anaquel físico.
    palabras_por_tile: dict[int, list[dict]] = {}
    todas: list[dict] = []
    for idx in tiles:
        page = doc[idx - 1]
        pw = [{"r": (w[0], w[1], w[2], w[3]), "texto": w[4]} for w in _palabras_producto(page)]
        palabras_por_tile[idx] = pw
        todas.extend(pw)

    if not todas:
        doc.close()
        return {"seccion_id": seccion_id, "nombre": nombre, "posiciones": []}

    y_top = min(p["r"][1] for p in todas)
    y_bottom = max(p["r"][3] for p in todas)
    alto_nivel = (y_bottom - y_top) / num_niveles

    def nivel_de(yc: float) -> int:
        indice_desde_arriba = int((yc - y_top) // alto_nivel)
        return max(1, min(num_niveles, num_niveles - indice_desde_arriba))

    posiciones_por_nivel: dict[int, list[dict]] = {n: [] for n in range(1, num_niveles + 1)}
    offset_x = 0.0
    for idx in tiles:
        page = doc[idx - 1]
        pw = palabras_por_tile[idx]
        slots = _slots_de_facing(page)

        por_nivel_palabras: dict[int, list[dict]] = {}
        for p in pw:
            yc = (p["r"][1] + p["r"][3]) / 2
            por_nivel_palabras.setdefault(nivel_de(yc), []).append(p)
        por_nivel_slots: dict[int, list] = {}
        for s in slots:
            yc = (s.y0 + s.y1) / 2
            por_nivel_slots.setdefault(nivel_de(yc), []).append(s)

        for nivel in range(1, num_niveles + 1):
            etiquetas = _agrupar_en_etiquetas(por_nivel_palabras.get(nivel, []))
            etiquetas.sort(key=lambda e: e["x0"])
            conteo = _contar_facings(etiquetas, por_nivel_slots.get(nivel, []))
            for i, e in enumerate(etiquetas):
                posiciones_por_nivel[nivel].append(
                    {
                        "sku": e["sku"],
                        "producto": e["producto"],
                        "facings_esperados": conteo.get(i, 1),
                        "x_global": e["x0"] + offset_x,
                    }
                )
        offset_x += page.rect.width

    doc.close()

    posiciones_finales = []
    for nivel in range(num_niveles, 0, -1):
        fila = sorted(posiciones_por_nivel[nivel], key=lambda p: p["x_global"])
        for col, p in enumerate(fila, start=1):
            posiciones_finales.append(
                {
                    "id": f"N{nivel}-P{col}",
                    "posicion": f"Nivel {nivel}, columna {col}",
                    "sku": p["sku"],
                    "producto": p["producto"],
                    "facings_esperados": p["facings_esperados"],
                    "nivel": nivel,
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
