from typing import List
import logfire

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        RecursiveCharacterTextSplitter = None

def chunk_text(text: str, chunk_size: int = 1500, chunk_overlap: int = 150) -> List[str]:
    """
    Splits text recursively by paragraphs, newlines, sentences, and spaces,
    guaranteeing that chunks do not exceed the specified chunk size.
    """
    with logfire.span("✂️ Text Chunking", text_length=len(text)):
        if not text or not text.strip(): 
            return []
            
        if RecursiveCharacterTextSplitter is not None:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
            )
            raw_chunks = splitter.split_text(text)
        else:
            raw_chunks = _fallback_chunk(text, chunk_size, chunk_overlap)
            
        valid_chunks = [c.strip() for c in raw_chunks if c and c.strip()]
        logfire.info(f"✅ Generated {len(valid_chunks)} chunks")
        return valid_chunks

def _fallback_chunk(text: str, chunk_size: int = 1500, chunk_overlap: int = 150) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            last_break = text.rfind("\n", start, end)
            if last_break == -1:
                last_break = text.rfind(" ", start, end)
            if last_break != -1 and last_break > start:
                end = last_break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = max(end - chunk_overlap, start + 1)
        if end >= len(text):
            break
    return chunks
