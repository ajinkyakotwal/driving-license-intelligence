from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Driving Licence Intelligence"

    openrouter_api_key: str = ""
    openrouter_model: str = "inclusionai/ling-3.0-flash-fin:free"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    frontend_origin: str = "http://localhost:5173"

    max_file_size_mb: int = 12

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()