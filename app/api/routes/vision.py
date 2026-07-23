from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile, status
from fastapi.responses import Response

from app.api.deps import get_current_user
from app.models.planograma import Planograma
from app.models.user import UserPublic
from app.models.vision import AnalizarConReferenciaResponse, AnalizarImagenResponse, CategoriaVisual
from app.services import planograma_service, vision_service

router = APIRouter(prefix="/vision", tags=["vision"])

_TIPOS_SOPORTADOS = {"image/jpeg", "image/png", "image/webp"}
_MAX_IMAGE_BYTES = 20 * 1024 * 1024  # límite de Groq para image_url en base64


@router.get("/secciones", response_model=list[Planograma])
def secciones(_current_user: UserPublic = Depends(get_current_user)) -> list[Planograma]:
    return planograma_service.listar()


@router.post("/analizar", response_model=AnalizarImagenResponse)
async def analizar(
    imagen: UploadFile,
    seccion_id: str = Form(...),
    _current_user: UserPublic = Depends(get_current_user),
) -> AnalizarImagenResponse:
    planograma = planograma_service.obtener(seccion_id)
    if planograma is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No existe un planograma para la sección '{seccion_id}'.",
        )

    if imagen.content_type not in _TIPOS_SOPORTADOS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Formato de imagen no soportado. Usa JPEG, PNG o WebP.",
        )

    contenido = await imagen.read()
    if len(contenido) > _MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="La imagen supera el límite de 20MB.",
        )

    try:
        return vision_service.analizar_imagen(contenido, imagen.content_type, planograma)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc


@router.get("/categorias-visuales", response_model=list[CategoriaVisual])
def categorias_visuales(_current_user: UserPublic = Depends(get_current_user)) -> list[CategoriaVisual]:
    return vision_service.listar_categorias_visuales()


@router.get("/referencia/{categoria_id}")
def referencia(categoria_id: str, _current_user: UserPublic = Depends(get_current_user)) -> Response:
    try:
        contenido, tipo = vision_service.cargar_referencia(categoria_id)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No existe planograma de referencia para la categoría '{categoria_id}'.",
        )
    return Response(content=contenido, media_type=tipo)


@router.post("/analizar-con-referencia", response_model=AnalizarConReferenciaResponse)
async def analizar_con_referencia(
    imagen_anaquel: UploadFile,
    categoria_id: str = Form(...),
    _current_user: UserPublic = Depends(get_current_user),
) -> AnalizarConReferenciaResponse:
    if imagen_anaquel.content_type not in _TIPOS_SOPORTADOS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Formato de imagen no soportado. Usa JPEG, PNG o WebP.",
        )

    contenido_anaquel = await imagen_anaquel.read()
    if len(contenido_anaquel) > _MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="La imagen supera el límite de 20MB.",
        )

    try:
        imagen_planograma, tipo_planograma = vision_service.cargar_referencia(categoria_id)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No existe planograma de referencia para la categoría '{categoria_id}'.",
        )

    try:
        return vision_service.analizar_con_referencia(
            categoria_id, imagen_planograma, tipo_planograma, contenido_anaquel, imagen_anaquel.content_type
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
