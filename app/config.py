import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def _clean(val: str) -> str:
    return val.strip().strip('"').strip("'") if isinstance(val, str) else val

class Settings:
    # --- GCP CONFIG ---
    PROJECT_ID = _clean(os.getenv("PROJECT_ID", "dmtxpress-507212"))
    LOCATION = _clean(os.getenv("LOCATION", "us-central1"))
    GCP_DOC_AI_LOCATION = _clean(os.getenv("GCP_DOC_AI_LOCATION", "us"))
    GCP_DOC_AI_PROCESSOR_ID = _clean(os.getenv("GCP_DOC_AI_PROCESSOR_ID"))
    RAW_BUCKET = _clean(os.getenv("GCP_RAW_BUCKET", "rag-data-raw-1"))
    PROCESSED_BUCKET = _clean(os.getenv("GCP_PROCESSED_BUCKET", "rag-data-processed-1"))

    # --- VECTOR DB (QDRANT) ---
    QDRANT_URL = _clean(os.getenv("QDRANT_CLUSTER_ENDPOINT"))
    QDRANT_API_KEY = _clean(os.getenv("QDRANT_API_KEY"))
    QDRANT_COLLECTION = "enterprise_rag"

    # --- REASONING ENGINE (OPENAI / GROQ) ---
    OPENAI_API_KEY = _clean(os.getenv("OPENAI_API_KEY"))
    OPENAI_MODEL = _clean(os.getenv("OPENAI_MODEL", "o3-mini"))
    GROQ_API_KEY = _clean(os.getenv("GROQ_API_KEY"))
    GROQ_MODEL = _clean(os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
    GROQ_FALLBACK_API_KEY = _clean(os.getenv("GROQ_FALLBACK_API_KEY"))

    # --- LLM GATEWAY (PORTKEY) ---
    PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")
    GROQ_SLUG =  "rag"     # primary: @rag/llama-3.3-70b-versatile
    GROQ_SLUG_2 = "brag"  # fallback: @brag/llama-3.1-8b-instant

    # --- PERSISTENCE (POSTGRES) ---
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASS = os.getenv("DB_PASS")
    DB_NAME = os.getenv("DB_NAME", "postgres")
    DB_CONNECTION_NAME = os.getenv("DB_CONNECTION_NAME")

    # --- CACHE (REDIS) ---
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    
    # --- OBSERVABILITY ---
    LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "true")
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "rag_scale_test")
    LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

# Apply LangChain environment variables for automatic tracing
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGSMITH_TRACING", "true")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "rag_scale_test")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

settings = Settings()