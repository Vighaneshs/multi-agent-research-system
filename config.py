"""App settings loaded from .env."""

import os
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


class Settings(BaseModel):
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")

    default_model: str = os.getenv("DEFAULT_MODEL", "gemini-2.0-flash")
    temperature: float = float(os.getenv("TEMPERATURE", "0.1"))

    max_iterations: int = int(os.getenv("MAX_ITERATIONS", "10"))
    timeout_seconds: int = int(os.getenv("TIMEOUT_SECONDS", "300"))

    checkpoint_dir: str = os.getenv("CHECKPOINT_DIR", "./checkpoints")


settings = Settings()
