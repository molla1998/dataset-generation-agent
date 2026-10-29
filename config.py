import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    api_key: str
    model: str


@dataclass(frozen=True)
class AppConfig:
    generation_llm: LLMConfig
    validation_llm: LLMConfig


def load_config() -> AppConfig:
    generation_base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    generation_api_key = os.getenv("VLLM_API_KEY", "EMPTY")
    generation_model = os.getenv("VLLM_MODEL", "gemma4-26B")

    return AppConfig(
        generation_llm=LLMConfig(
            base_url=generation_base_url,
            api_key=generation_api_key,
            model=generation_model,
        ),
        validation_llm=LLMConfig(
            base_url=os.getenv("VALIDATION_VLLM_BASE_URL", generation_base_url),
            api_key=os.getenv("VALIDATION_VLLM_API_KEY", generation_api_key),
            model=os.getenv("VALIDATION_VLLM_MODEL", generation_model),
        ),
    )
