from dataclasses import dataclass

from app.core.security import hash_password, verify_password
from app.models.user import RolUsuario, UserPublic


@dataclass
class _UserRecord:
    username: str
    nombre: str
    rol: RolUsuario
    hashed_password: str


# Usuarios demo en memoria — uno por rol, mismo esquema de roles que el front
# (RolUsuario en osa.models.ts). Sin base de datos todavía: este PoC usa datos
# dummy en el resto de la app, así que los usuarios siguen la misma lógica.
_USERS: dict[str, _UserRecord] = {
    record.username: record
    for record in [
        _UserRecord("operativo", "Juan Pérez", "operativo", hash_password("operativo123")),
        _UserRecord("tienda", "María Gómez", "tienda", hash_password("tienda123")),
        _UserRecord("ejecutivo", "Carlos Ruiz", "ejecutivo", hash_password("ejecutivo123")),
    ]
}


def authenticate(username: str, password: str) -> UserPublic | None:
    record = _USERS.get(username)
    if record is None or not verify_password(password, record.hashed_password):
        return None
    return UserPublic(username=record.username, nombre=record.nombre, rol=record.rol)


def get_user(username: str) -> UserPublic | None:
    record = _USERS.get(username)
    if record is None:
        return None
    return UserPublic(username=record.username, nombre=record.nombre, rol=record.rol)
