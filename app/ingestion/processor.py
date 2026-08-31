import os
import sys
import uuid
import json
import logfire
import vertexai

from typing import List
from google.cloud import storage
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Import local modules
from app.config import settings
from app.services.retrieval.embedding import embed_texts
from app.ingestion.loaders.pdf import parse_pdf
from app.ingestion.loaders.html import parse_html
from app.ingestion.loaders.text import parse_text
from app.ingestion.loaders.office import parse_office
from app.ingestion.chunking.splitter import chunk_text

# Initialize Logfire with the Enterprise Ingestion Service Name
logfire.configure(service_name="enterprise-ingestion-service")

# Initialize Vertex AI for Embeddings
vertexai.init(project=settings.PROJECT_ID, location=settings.LOCATION)

# Initialize GCS Client
storage_client = storage.Client(project=settings.PROJECT_ID)

# Initialize Qdrant Client
qdrant_client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY
)

def upload_to_gcs(data, bucket_name: str, destination_blob_name: str, is_json: bool = False):
    """
    Uploads a file or JSON data to GCS.
    """
    with logfire.span("☁️ GCS Upload", bucket=bucket_name, blob=destination_blob_name):
        try:
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(destination_blob_name)
            if is_json:
                blob.upload_from_string(json.dumps(data), content_type='application/json')
            else:
                blob.upload_from_filename(data)
            logfire.info(f"✅ Uploaded to {bucket_name}")
        except Exception as e:
            logfire.error(f"❌ GCS Upload Failed: {e}")
            raise e

def process_file(file_path: str, filename: str, source_type: str, doc_index: int = None, total_docs: int = None) -> bool | None:
    """
    Orchestrates the parsing, chunking, embedding, and indexing of a single file.
    Returns True on success, False on error, None if skipped.
    """
    doc_label = f" [{doc_index}/{total_docs}]" if doc_index is not None and total_docs is not None else ""
    doc_progress = f"{doc_index}/{total_docs}" if doc_index is not None and total_docs is not None else None

    with logfire.span(f"🚀 Processing File{doc_label}", file=filename, source=source_type, document_progress=doc_progress):
        if doc_progress:
            logfire.info(f"📄 Document {doc_progress}: {filename}")
        try:
            # 1. Upload RAW file to GCS
            raw_gcs_path = f"{source_type}/{filename}"
            upload_to_gcs(file_path, settings.RAW_BUCKET, raw_gcs_path)
            
            # 2. Extract Text based on extension
            ext = filename.lower().split('.')[-1]
            if ext == 'pdf':
                full_text = parse_pdf(file_path)
            elif ext in ['html', 'htm']:
                full_text = parse_html(file_path)
            elif ext == 'txt':
                full_text = parse_text(file_path)
            elif ext in ['docx', 'pptx']:
                full_text = parse_office(file_path)
            else:
                logfire.warning(f"⏩ Skipping unsupported file type: {filename}")
                return None

            if not full_text or not full_text.strip():
                logfire.warning(f"⚠️ No text extracted from {filename}")
                return None

            # 3. Chunk Text
            chunks = chunk_text(full_text)
            if not chunks:
                logfire.warning(f"⚠️ No chunks generated from {filename}")
                return None

            # 4. Upload PROCESSED metadata to GCS
            processed_data = {"filename": filename, "chunks": chunks, "source_type": source_type}
            processed_gcs_path = f"{source_type}/{filename}.json"
            upload_to_gcs(processed_data, settings.PROCESSED_BUCKET, processed_gcs_path, is_json=True)

            # 5. Embed and Index in Qdrant
            with logfire.span("🧠 Vectorizing & Indexing"):
                embeddings = embed_texts(chunks)
                points = []
                for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
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
                
                qdrant_client.upsert(
                    collection_name=settings.QDRANT_COLLECTION,
                    points=points
                )
                logfire.info(f"✨ Indexed {len(points)} points to Qdrant")
            return True

        except Exception as e:
            logfire.error(f"💥 Failed to process {filename}: {e}")
            return False

def run_universal_ingestion(base_dir: str, explicit_source_type: str = None, wipe: bool = False) -> dict:
    """
    Automatically scans the directory.
    If it has subfolders, maps them to source_types.
    If it has no subfolders, uses the explicit_source_type or infers from the folder name.
    """
    with logfire.span("🌍 Universal Ingestion Started", base_directory=base_dir):
        # Handle Collection Wipe
        if wipe:
            with logfire.span("🧹 Wiping Collection"):
                if qdrant_client.collection_exists(settings.QDRANT_COLLECTION):
                    qdrant_client.delete_collection(settings.QDRANT_COLLECTION)
                    logfire.info(f"🗑️ Collection {settings.QDRANT_COLLECTION} deleted")

        # Ensure Collection Exists
        if not qdrant_client.collection_exists(settings.QDRANT_COLLECTION):
            qdrant_client.create_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
            )
            logfire.info(f"🆕 Created collection {settings.QDRANT_COLLECTION}")

        # Collect all files across directories to track total progress
        tasks = []
        subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and not d.startswith('.')]
        
        if not subdirs:
            if explicit_source_type:
                source_type = explicit_source_type
            else:
                base_name = os.path.basename(os.path.normpath(base_dir)).lower()
                source_type = "true" if "true" in base_name else "noisy" if "noisy" in base_name else "general"
            
            logfire.info(f"📂 No subdirectories found, processing {base_dir} as '{source_type}'")
            files = [f for f in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir, f)) and not f.startswith('.')]
            for filename in sorted(files):
                tasks.append((os.path.join(base_dir, filename), filename, source_type))
        else:
            for subdir in sorted(subdirs):
                source_type = "true" if "true" in subdir.lower() else "noisy" if "noisy" in subdir.lower() else subdir
                dir_path = os.path.join(base_dir, subdir)
                files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f)) and not f.startswith('.')]
                for filename in sorted(files):
                    tasks.append((os.path.join(dir_path, filename), filename, source_type))

        total_docs = len(tasks)
        logfire.info(f"🔍 Discovered {total_docs} files to process")

        successful_count = 0
        failed_count = 0
        skipped_count = 0
        failed_files = []

        for idx, (file_path, filename, source_type) in enumerate(tasks, start=1):
            status = process_file(file_path, filename, source_type, doc_index=idx, total_docs=total_docs)
            if status is True:
                successful_count += 1
            elif status is False:
                failed_count += 1
                failed_files.append(filename)
            else:
                skipped_count += 1

        summary = {
            "total_files": total_docs,
            "successful": successful_count,
            "failed": failed_count,
            "skipped": skipped_count,
            "failed_files": failed_files
        }

        if failed_count > 0:
            logfire.warning(
                f"⚠️ Ingestion finished with errors: {successful_count}/{total_docs} succeeded, {failed_count} failed, {skipped_count} skipped",
                **summary
            )
        else:
            logfire.info(
                f"🎉 All files processed successfully: {successful_count}/{total_docs} succeeded, {failed_count} failed, {skipped_count} skipped",
                **summary
            )

        return summary

def process_directory(dir_path: str, source_type: str):
    """
    Processes all files in a specific directory.
    """
    with logfire.span("📁 Scanning Directory", path=dir_path, source=source_type):
        files = [f for f in sorted(os.listdir(dir_path)) if os.path.isfile(os.path.join(dir_path, f)) and not f.startswith('.')]
        logfire.info(f"🔍 Found {len(files)} files")
        
        success_cnt, fail_cnt, skip_cnt = 0, 0, 0
        for idx, filename in enumerate(files, start=1):
            file_path = os.path.join(dir_path, filename)
            res = process_file(file_path, filename, source_type, doc_index=idx, total_docs=len(files))
            if res is True:
                success_cnt += 1
            elif res is False:
                fail_cnt += 1
            else:
                skip_cnt += 1

        logfire.info(
            f"📁 Directory {dir_path} summary: {success_cnt} succeeded, {fail_cnt} failed, {skip_cnt} skipped out of {len(files)} files",
            successful=success_cnt, failed=fail_cnt, skipped=skip_cnt, total=len(files)
        )

if __name__ == "__main__":
    # Usage: python -m app.ingestion.processor [dir_path] [source_type] [--wipe]
    wipe_requested = "--wipe" in sys.argv
    clean_args = [a for a in sys.argv if a != "--wipe"]
    
    # Default to DATA/ if no path provided
    target_dir = clean_args[1] if len(clean_args) > 1 else "DATA"
    explicit_type = clean_args[2] if len(clean_args) > 2 else None
    
    if not os.path.exists(target_dir):
        print(f"Error: Path {target_dir} does not exist.")
        sys.exit(1)
        
    summary = run_universal_ingestion(target_dir, explicit_source_type=explicit_type, wipe=wipe_requested)
    logfire.info(
        f"🏁 Universal Ingestion Job Completed: {summary['successful']}/{summary['total_files']} succeeded, {summary['failed']} failed",
        **summary
    )

