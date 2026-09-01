import logfire
from openai import AsyncOpenAI
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


def get_langchain_llm(
    feature: str = "rag",
    model: str | None = None,
    slug: str | None = None,
    config_id: str | None = None,
    temperature: float = 0
) -> ChatOpenAI:
    """
    Returns a Portkey-backed LangChain ChatOpenAI client.
    Supports individual component slugs (e.g. @guardrails/gpt-4o-mini, @planner/gpt-4o)
    and optional custom config IDs.
    """
    header_kwargs = {
        "api_key": settings.PORTKEY_API_KEY,
        "metadata": {
            "feature": feature,
            "_user": feature,
            "environment": "production"
        }
    }
    active_config = config_id or GATEWAY_CONFIG
    if active_config:
        header_kwargs["config"] = active_config

    target_slug = slug or settings.OPENAI_SLUG
    target_model = model or f"@{target_slug}/gpt-4o"

    return ChatOpenAI(
        api_key=settings.PORTKEY_API_KEY,
        base_url=PORTKEY_GATEWAY_URL,
        model=target_model,
        temperature=temperature,
        default_headers=createHeaders(**header_kwargs)
    )


def get_eval_llm(model: str = "gpt-4o", temperature: float = 0) -> ChatOpenAI:
    """
    Returns a Portkey-backed ChatOpenAI client specifically configured for
    LLM-as-a-Judge evaluations (RAGAS / DeepEval / synthetic dataset generation).
    Routes through settings.EVALS_SLUG and attaches PORTKEY_EVAL_CONFIG_ID if present.
    """
    return get_langchain_llm(
        feature="evals_judge",
        model=f"@{settings.EVALS_SLUG}/{model}",
        slug=settings.EVALS_SLUG,
        config_id=settings.PORTKEY_EVAL_CONFIG_ID if settings.PORTKEY_EVAL_CONFIG_ID else None,
        temperature=temperature
    )


def get_eval_async_client() -> AsyncOpenAI:
    """
    Returns an AsyncOpenAI client configured for Portkey AI Gateway.
    Pre-wired with PORTKEY_EVAL_CONFIG_ID and metadata tags for RAGAS InstructorLLM.
    """
    header_kwargs = {
        "api_key": settings.PORTKEY_API_KEY,
        "metadata": {
            "feature": "evals_judge",
            "_user": "evals_judge",
            "environment": "production"
        }
    }
    active_config = settings.PORTKEY_EVAL_CONFIG_ID if settings.PORTKEY_EVAL_CONFIG_ID else GATEWAY_CONFIG
    if active_config:
        header_kwargs["config"] = active_config

    return AsyncOpenAI(
        api_key=settings.PORTKEY_API_KEY,
        base_url=PORTKEY_GATEWAY_URL,
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