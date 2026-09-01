import logfire
from nemoguardrails import RailsConfig, LLMRails

from app.config import settings
from app.gateway.client import get_langchain_llm
from app.guardrails.colang_rules import COLANG_CONTENT, YAML_CONTENT, RAIL_INDICATORS


_rails: LLMRails | None = None


def initialize_rails() -> None:
    """
    Build the NeMo LLMRails singleton at app startup.
    Uses gpt-4o-mini via Portkey Gateway for fast intent classification at the gate —
    the heavier reasoning model is reserved for the RAG pipeline.
    """
    global _rails

    guard_llm = get_langchain_llm(
        feature="guardrails",
        slug=settings.GUARDRAILS_SLUG,
        model=f"@{settings.GUARDRAILS_SLUG}/gpt-4o-mini",
        config_id=settings.PORTKEY_GUARDRAILS_CONFIG_ID if settings.PORTKEY_GUARDRAILS_CONFIG_ID else None,
        temperature=0
    )

    config = RailsConfig.from_content(
        colang_content=COLANG_CONTENT,
        yaml_content=YAML_CONTENT
    )

    _rails = LLMRails(config, llm=guard_llm)
    logfire.info("🛡️ NeMo Guardrails initialised (gpt-4o-mini).")
    
    


def _clean_rail_response(text: str) -> str:
    """Strip internal NeMo scaffolding (User intent, Bot intent, Bot message prefixes)."""
    if "Bot message:" in text:
        text = text.split("Bot message:")[-1]
    elif "bot message:" in text.lower():
        idx = text.lower().find("bot message:")
        text = text[idx + len("bot message:"):]
    return text.strip().strip('"').strip("'").strip()


def guard(message: str) -> tuple[bool, str | None]:
    """
    Run a user message through the NeMo rails gate.

    Returns:
        (True,  rail_response) — a rail fired; return this response immediately,
                                skip the RAG pipeline entirely.
        (False, None)          — message is clean; proceed to LangGraph.
    """
    if _rails is None:
        logfire.warning("⚠️ Guardrails not initialised — skipping gate.")
        return False, None

    with logfire.span("🛡️ Guardrails Check"):
        result = _rails.generate(messages=[{"role": "user", "content": message}])

        # NeMo returns {'role': 'assistant', 'content': '...'} — extract text
        content = result.get("content", "") if isinstance(result, dict) else str(result)

        fired = any(indicator.lower() in content.lower() for indicator in RAIL_INDICATORS)

        if fired:
            clean_content = _clean_rail_response(content)
            logfire.info(f"🛡️ Guardrails fired | query='{message[:80]}'")
            return True, clean_content

        logfire.info("✅ Guardrails passed.")
        return False, None
