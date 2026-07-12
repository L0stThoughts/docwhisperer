from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file.

    Attributes:
        openai_api_key: Optional[str] - OpenAI API key for LLM access.
        ollama_url: Optional[str] - Optional local Ollama URL for fallback LLM.
        chroma_db_dir: str - Directory to store Chroma DB files.
        fastapi_host: str - Host for the FastAPI server.
        fastapi_port: int - Port for the FastAPI server.
    """

    openai_api_key: Optional[str]
    ollama_url: Optional[str] = None
    chroma_db_dir: str = "./chroma_db"
    fastapi_host: str = "127.0.0.1"
    fastapi_port: int = 8000
    ragas_eval_output: str = "./evaluation/results.json"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
