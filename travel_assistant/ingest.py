"""Index local text, Markdown, and PDF files. Run: python -m travel_assistant.ingest documents"""
import argparse
import hashlib
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .core import services


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    if not args.directory.is_dir():
        parser.error('directory does not exist')
    load_dotenv()
    docs = []
    for path in sorted(args.directory.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in ('.txt', '.md', '.pdf'):
            continue
        if path.suffix.lower() == '.pdf':
            from pypdf import PdfReader
            text = '\n'.join(page.extract_text() or '' for page in PdfReader(path).pages)
        else:
            text = path.read_text(encoding='utf-8')
        if text.strip():
            docs.append(Document(page_content=text, metadata={'source': str(path.relative_to(args.directory))}))
    if not docs:
        parser.error('no non-empty supported documents found')
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(docs)
    # Stable IDs make repeated ingestion of unchanged chunks safe.
    ids = [hashlib.sha256((d.metadata['source'] + '\0' + d.page_content).encode()).hexdigest() for d in chunks]
    store, _ = services()
    store.add_documents(chunks, ids=ids)
    print(f'Indexed {len(chunks)} chunks from {len(docs)} documents.')

if __name__ == '__main__':
    main()
