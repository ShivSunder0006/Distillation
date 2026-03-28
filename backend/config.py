import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    GROQ_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    TEACHER_MODEL: str = "llama-3.3-70b-versatile"
    STUDENT_MODEL: str = "llama-3.1-8b-instant"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    FAISS_INDEX_PATH: str = "data/faiss_index"
    
    class Config:
        env_file = ".env"

settings = Settings()
