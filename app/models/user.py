from typing import Literal

from pydantic import BaseModel

RolUsuario = Literal["operativo", "tienda", "ejecutivo"]


class LoginRequest(BaseModel):
    username: str
    password: str


class UserPublic(BaseModel):
    username: str
    nombre: str
    rol: RolUsuario


class Token(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    user: UserPublic
