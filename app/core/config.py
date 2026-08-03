from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuración de la app, cargable desde variables de entorno o .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    secret_key: str = "dev-secret-key-change-me-in-.env"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 480

    cors_origins: list[str] = ["http://localhost:4200"]

    groq_api_key: str = ""
    groq_vision_model: str = "qwen/qwen3.6-27b"

    # Manda la lámina del planograma junto con la foto del anaquel. Mejora
    # bastante la precisión (sin ella el modelo solo tiene la MARCA en texto, y
    # como todas las posiciones de una charola repiten la misma marca no puede
    # distinguir una variante de otra), pero es la mitad del costo en tokens de
    # visión de cada llamada. Ponlo en false si el plan de Groq se queda corto
    # de TPM y prefieres precisión menor a quedarte sin cuota.
    vision_enviar_referencia: bool = True


settings = Settings()

if settings.secret_key == "dev-secret-key-change-me-in-.env":
    print(
        "[osa-vision-api] AVISO: usando SECRET_KEY de desarrollo por defecto. "
        "Define SECRET_KEY en un archivo .env antes de usar en producción."
    )
