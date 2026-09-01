import logfire
from portkey_ai import Portkey, createHeaders, PORTKEY_GATEWAY_URL
from langchain_openai import ChatOpenAI

from app.config import settings


# If PORTKEY_CONFIG_ID is set in .env (e.g. pc-xxxx), use the saved dashboard config.
# If not set, keep it None so Portkey uses basic slug-based routing without triggering the inline config restriction.
GATEWAY_CONFIG = settings.PORTKEY_CONFIG_ID if settings.PORTKEY_CONFIG_ID else None

portkey_kwargs = {"api_key": settings.PORTKEY_API_KEY}
if GATEWAY_CONFIG:
    portkey_kwargs["config"] = GATEWAY_CONFIG

portkey_client = Portkey(**portkey_kwargs)


def get_langchain_llm(feature: str = "rag", model: str | None = None, temperature: float = 0) -> ChatOpenAI:
    """
    Returns a Portkey-backed ChatOpenAI client routed through the Portkey AI Gateway.
    Allows specifying different models (e.g., gpt-4o-mini for guardrails vs gpt-4o for RAG)
    and tags each call with feature-specific metadata.
    """
    header_kwargs = {
        "api_key": settings.PORTKEY_API_KEY,
        "metadata": {
            "feature": feature,
            "_user": feature,
            "environment": "production"
        }
    }
    if GATEWAY_CONFIG:
        header_kwargs["config"] = GATEWAY_CONFIG

    target_model = model or f"@{settings.OPENAI_SLUG}/gpt-4o"

    return ChatOpenAI(
        api_key=settings.PORTKEY_API_KEY,
        base_url=PORTKEY_GATEWAY_URL,
        model=target_model,
        temperature=temperature,
        default_headers=createHeaders(**header_kwargs)
    )

def extract_cache_status(response) -> str:
    """
    Pull cache-status from the Portkey native client response headers.
    Uses response.get_headers() (Portkey SDK v2+) with defensive fallbacks.
    """
    if hasattr(response, "get_headers") and callable(response.get_headers):
        headers = response.get_headers() or {}
        status = headers.get("cache-status") or headers.get("x-portkey-cache-status")
        if status:
            return str(status).upper()

    for attr in ("_raw_response", "_response", "_http_response", "headers"):
        raw = getattr(response, attr, None)
        if raw is not None:
            headers = getattr(raw, "headers", raw) if hasattr(raw, "headers") else (raw if isinstance(raw, dict) else {})
            status = headers.get("cache-status") or headers.get("x-portkey-cache-status")
            if status:
                return str(status).upper()

    return "MISS"