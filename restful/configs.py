from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    CACHING_TYPE: str
    CACHING_HOST: str
    CACHING_PORT: int
    CACHING_PASS: str

    class Config:
        env_file = ".env"

settings = Settings()
