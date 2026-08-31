import os
import sys
import uuid
import json
import time
import logfire
import vertexai

from google.cloud import storage
from google.api_core.exceptions import ResourceExhausted
from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.config import settings
from app.services.retrieval.embedding import embed_texts
from app.ingestion.loaders.pdf import parse_pdf
from app.ingestion.loaders.html import parse_html
from app.ingestion.loaders.text import parse_text
from app.ingestion.loaders.office import parse_office
from app.ingestion.chunking.splitter import chunk_text

# Initialize Logfire
logfire.configure(service_name="enterprise-ingestion-service")

# Initialize Vertex AI
vertexai.init(project=settings.PROJECT_ID, location=settings.LOCATION)

# Initialize Storage & Qdrant with generous timeouts
storage_client = storage.Client(project=settings.PROJECT_ID)
qdrant_client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
    timeout=60.0  # Prevents 'write operation timed out' on large payload uploads
)

DEFAULT_FAILED_FILES = [
    ("DATA/noisy_data/A Trip Through The Graphics Pipeline - All (2011).docx", "A Trip Through The Graphics Pipeline - All (2011).docx", "noisy"),
    ("DATA/noisy_data/A Journey in Creating an Operating System Kernel - The 539Kernel Book (Nov 2022).pdf", "A Journey in Creating an Operating System Kernel - The 539Kernel Book (Nov 2022).pdf", "noisy")
]

def upload_to_gcs_with_retry(data, bucket_name: str, destination_blob_name: str, is_json: bool = False, max_retries: int = 3):
    """Uploads file/JSON to GCS with retry mechanism."""
    with logfire.span("☁️ GCS Upload", bucket=bucket_name, blob=destination_blob_name):
        for attempt in range(1, max_retries + 1):
            try:
                bucket = storage_client.bucket(bucket_name)
                blob = bucket.blob(destination_blob_name)
                if is_json:
                    blob.upload_from_string(json.dumps(data), content_type='application/json', timeout=120)
                else:
                    blob.upload_from_filename(data, timeout=120)
                logfire.info(f"✅ Uploaded to {bucket_name}/{destination_blob_name}")
                return True
            except Exception as e:
                logfire.warning(f"⚠️ GCS upload attempt {attempt}/{max_retries} failed: {e}")
                if attempt == max_retries:
                    logfire.error(f"❌ GCS Upload Permanently Failed: {e}")
                    raise e
                time.sleep(2 ** attempt)

def try_load_existing_processed_chunks(bucket_name: str, processed_blob_name: str) -> list[str] | None:
    """Checks if processed JSON already exists in GCS to skip re-parsing."""
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(processed_blob_name)
        if blob.exists():
            logfire.info(f"📦 Found existing processed JSON in GCS: {processed_blob_name}")
            data = json.loads(blob.download_as_text())
            chunks = data.get("chunks", [])
            if chunks:
                logfire.info(f"⚡ Fast-track: Reusing {len(chunks)} pre-extracted chunks from GCS")
                return chunks
    except Exception as e:
        logfire.warning(f"Could not load pre-processed chunks: {e}")
    return None

def upsert_points_in_batches(collection_name: str, points: list, batch_size: int = 50, max_retries: int = 5):
    """Upserts points to Qdrant in safe batches with exponential backoff on timeouts."""
    total_points = len(points)
    for i in range(0, total_points, batch_size):
        batch = points[i:i + batch_size]
        batch_label = f"[{i+1}-{min(i+batch_size, total_points)}/{total_points}]"
        
        for attempt in range(1, max_retries + 1):
            try:
                qdrant_client.upsert(
                    collection_name=collection_name,
                    points=batch
                )
                logfire.info(f"  ✨ Upserted batch {batch_label} ({len(batch)} points)")
                break
            except Exception as e:
                wait_sec = 2 ** attempt
                logfire.warning(f"  ⚠️ Qdrant batch {batch_label} failed (attempt {attempt}/{max_retries}): {e}. Retrying in {wait_sec}s...")
                if attempt == max_retries:
                    logfire.error(f"  ❌ Qdrant upsert failed permanently for batch {batch_label}")
                    raise e
                time.sleep(wait_sec)

def process_single_file_failproof(file_path: str, filename: str, source_type: str, doc_index: int = 1, total_docs: int = 1) -> bool:
    """Bulletproof processing of a single file with fast-resume, rate-limiting, and error-handling."""
    with logfire.span(f"🚀 Robust Processing [{doc_index}/{total_docs}]", file=filename, source=source_type):
        logfire.info(f"📄 Target Document: {filename}")
        try:
            raw_gcs_path = f"{source_type}/{filename}"
            processed_gcs_path = f"{source_type}/{filename}.json"

            # 1. Ensure RAW file is in GCS
            upload_to_gcs_with_retry(file_path, settings.RAW_BUCKET, raw_gcs_path)

            # 2. Check if already parsed in GCS
            chunks = try_load_existing_processed_chunks(settings.PROCESSED_BUCKET, processed_gcs_path)

            # 3. If not pre-parsed, parse and chunk
            if not chunks:
                ext = filename.lower().split('.')[-1]
                logfire.info(f"Extracting text for {ext.upper()} document...")
                if ext == 'pdf':
                    full_text = parse_pdf(file_path)
                elif ext in ['html', 'htm']:
                    full_text = parse_html(file_path)
                elif ext == 'txt':
                    full_text = parse_text(file_path)
                elif ext in ['docx', 'pptx']:
                    full_text = parse_office(file_path)
                else:
                    logfire.warning(f"⏩ Unsupported file type: {filename}")
                    return False

                if not full_text or not full_text.strip():
                    logfire.warning(f"⚠️ No text extracted from {filename}")
                    return False

                chunks = chunk_text(full_text)
                if not chunks:
                    logfire.warning(f"⚠️ No chunks generated from {filename}")
                    return False

                # Save processed JSON to GCS
                processed_data = {"filename": filename, "chunks": chunks, "source_type": source_type}
                upload_to_gcs_with_retry(processed_data, settings.PROCESSED_BUCKET, processed_gcs_path, is_json=True)

            logfire.info(f"✂️ Total chunks to vectorize: {len(chunks)}")

            # 4. Embed chunks with Vertex AI
            with logfire.span("🧠 Vectorizing Chunks", count=len(chunks)):
                embeddings = embed_texts(chunks)

            # 5. Build Qdrant points
            points = []
            for chunk, vector in zip(chunks, embeddings):
                points.append(models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "text": chunk,
                        "source": filename,
                        "source_type": source_type,
                        "raw_gcs_path": f"gs://{settings.RAW_BUCKET}/{raw_gcs_path}"
                    }
                ))

            # 6. Safe batch upsert to Qdrant
            with logfire.span("💾 Indexing into Qdrant", total_points=len(points)):
                upsert_points_in_batches(settings.QDRANT_COLLECTION, points, batch_size=50)
                logfire.info(f"🎉 Fully indexed {len(points)} points into Qdrant for {filename}")

            return True

        except Exception as e:
            logfire.error(f"💥 Permanent failure processing {filename}: {e}")
            return False

def main():
    targets = DEFAULT_FAILED_FILES
    
    if len(sys.argv) > 1:
        custom_path = sys.argv[1]
        custom_name = os.path.basename(custom_path)
        custom_source = sys.argv[2] if len(sys.argv) > 2 else ("noisy" if "noisy" in custom_path else "true")
        targets = [(custom_path, custom_name, custom_source)]

    with logfire.span("🔄 Selective Fail-Proof Ingestion Job", total_targets=len(targets)):
        success_cnt = 0
        fail_cnt = 0

        for idx, (path, name, stype) in enumerate(targets, start=1):
            if not os.path.exists(path):
                logfire.error(f"File path does not exist locally: {path}")
                fail_cnt += 1
                continue

            ok = process_single_file_failproof(path, name, stype, doc_index=idx, total_docs=len(targets))
            if ok:
                success_cnt += 1
            else:
                fail_cnt += 1

        print(f"\n==========================================")
        print(f"Summary: {success_cnt}/{len(targets)} succeeded, {fail_cnt} failed.")
        print(f"==========================================\n")

if __name__ == "__main__":
    main()
