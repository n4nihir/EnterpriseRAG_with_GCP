import time
import vertexai
from vertexai.language_models import TextEmbeddingModel
from google.api_core.exceptions import ResourceExhausted
from app.config import settings

model = None
BATCH_SIZE = 5

def get_embedding_model():
    global model
    if model is None:
        # Initialize Vertex AI before loading the model
        vertexai.init(project=settings.PROJECT_ID, location=settings.LOCATION)
        # Reverting to TextEmbeddingModel for stability
        model = TextEmbeddingModel.from_pretrained("text-embedding-004")
    return model

def embed_query(query: str):
    """Embeds a single query string using the stable Vertex AI API."""
    model = get_embedding_model()
    embeddings = model.get_embeddings([query])
    return embeddings[0].values

def embed_texts(texts: list[str]):
    """Embeds a list of text strings in batches with rate limiting."""
    model = get_embedding_model()
    all_embeddings = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]

        # Retry with exponential backoff on quota errors
        for attempt in range(5):
            try:
                embeddings = model.get_embeddings(batch)
                all_embeddings.extend([e.values for e in embeddings])
                break
            except ResourceExhausted:
                wait_time = 2 ** attempt * 5  # 5, 10, 20, 40, 80 seconds
                print(f"    ⏳ Quota exceeded, retrying in {wait_time}s (attempt {attempt+1}/5)")
                time.sleep(wait_time)
        else:
            raise ResourceExhausted(f"Quota still exceeded after 5 retries for batch starting at index {i}")

        # Rate limit: pause between batches to stay within quota
        time.sleep(1)

    return all_embeddings
