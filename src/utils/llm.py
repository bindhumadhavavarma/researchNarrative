"""LLM client factory — supports both standard OpenAI and Azure OpenAI."""

from src.config import (
    OPENAI_API_KEY,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT,
    USE_AZURE_OPENAI,
    HAS_LLM,
    LLM_MODEL,
)


def get_llm_client():
    """Return an OpenAI-compatible client (Azure or standard).

    Returns None if no credentials are configured.
    """
    if USE_AZURE_OPENAI:
        from openai import AzureOpenAI
        return AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION,
        )
    elif OPENAI_API_KEY:
        from openai import OpenAI
        return OpenAI(api_key=OPENAI_API_KEY)
    return None


def get_model_name() -> str:
    """Return the model/deployment name to use in API calls."""
    if USE_AZURE_OPENAI:
        return AZURE_OPENAI_DEPLOYMENT
    return LLM_MODEL
