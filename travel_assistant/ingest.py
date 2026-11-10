"""Index text, Markdown, or PDF documents in demo or cloud mode."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .core import services
from .local import LocalStore, read_documents


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--mode', choices=('demo', 'cloud'), default=os.getenv('TRAVEL_MODE', 'demo'))
    args = parser.parse_args()
    if not args.directory.is_dir(): parser.error('directory does not exist')
    os.environ['TRAVEL_MODE'] = args.mode
    docs = read_documents(args.directory)
    if not docs: parser.error('no non-empty supported documents found')
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(docs)
    if args.mode == 'demo':
        LocalStore().replace(chunks)
    else:
        ids = [hashlib.sha256((d.metadata['source'] + '\0' + d.page_content).encode()).hexdigest() for d in chunks]
        store, _ = services()
        # A per-index/namespace/directory manifest removes chunks superseded by this source directory.
        scope = '|'.join((os.environ['PINECONE_INDEX_NAME'], os.getenv('PINECONE_NAMESPACE', 'travel'), str(args.directory.resolve())))
        manifest = Path('.travel') / (hashlib.sha256(scope.encode()).hexdigest() + '.json')
        old_ids = set(json.loads(manifest.read_text())) if manifest.exists() else set()
        store.add_documents(chunks, ids=ids)
        stale = old_ids - set(ids)
        if stale: store.delete(ids=sorted(stale))
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(ids))
    print(f'Indexed {len(chunks)} chunks from {len(docs)} documents in {args.mode} mode.')

if __name__ == '__main__': main()
