"""Persistent local retrieval for a useful credential-free demo."""
import json
import math
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from langchain_core.documents import Document

STOP = set('a an the is are was were be to of in on for and or at by i me my you your we our it its this that what which when where how should can do does about tell please'.split())


def tokens(text):
    return [word for word in re.findall(r'[a-z0-9]+', text.lower()) if word not in STOP and len(word) > 1]


def index_path():
    return Path(os.getenv('TRAVEL_LOCAL_INDEX', '.travel/index.json'))


def read_documents(directory):
    docs = []
    for path in sorted(Path(directory).rglob('*')):
        if not path.is_file() or path.suffix.lower() not in ('.txt', '.md', '.pdf'):
            continue
        if path.suffix.lower() == '.pdf':
            from pypdf import PdfReader
            content = '\n'.join(page.extract_text() or '' for page in PdfReader(path).pages)
        else:
            content = path.read_text(encoding='utf-8')
        if content.strip():
            docs.append(Document(page_content=content, metadata={'source': str(path.relative_to(directory))}))
    return docs


class LocalStore:
    def __init__(self, path=None):
        self.path = Path(path) if path else index_path()

    def documents(self):
        if self.path.exists():
            records = json.loads(self.path.read_text(encoding='utf-8'))
            return [Document(**record) for record in records]
        return read_documents(Path(__file__).parent / 'demo_documents')

    def replace(self, chunks):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        records = [{'page_content': d.page_content, 'metadata': d.metadata} for d in chunks]
        fd, name = tempfile.mkstemp(dir=self.path.parent, suffix='.json')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as stream:
                json.dump(records, stream, ensure_ascii=False)
            os.replace(name, self.path)
        finally:
            if os.path.exists(name): os.unlink(name)

    def similarity_search(self, query, k=4):
        docs = self.documents()
        query_words = set(tokens(query))
        corpus = [Counter(tokens(d.page_content + ' ' + str(d.metadata.get('source', '')))) for d in docs]
        total = len(docs)
        scores = []
        for i, words in enumerate(corpus):
            score = sum((1 + math.log((total + 1) / (1 + sum(word in item for item in corpus)))) * min(words[word], 3) for word in query_words if words[word])
            if score > 0: scores.append((score, i))
        return [docs[i] for _, i in sorted(scores, reverse=True)[:k]]


def local_query(question, history):
    # Resolve simple follow-ups without pretending to run an LLM.
    query = question
    if history and re.search(r'\b(it|there|that|those|also|they)\b', question, re.I):
        query = ' '.join(m['content'] for m in history[-4:] if m['role'] == 'user') + ' ' + question
    return query


def local_result(query, docs):
    sources = [{'id': i, 'source': d.metadata.get('source', 'unknown'), 'content': d.page_content} for i, d in enumerate(docs, 1)]
    if not sources:
        return {'answer': 'I could not find supporting information in the current documents. Try a more specific question or index your own travel documents.', 'sources': [], 'mode': 'demo'}
    query_words = set(tokens(query))
    excerpts = []
    for source in sources:
        lines = [line.strip().lstrip('- ') for line in source['content'].splitlines() if line.strip() and not line.startswith('#')]
        ranked = sorted(enumerate(lines), key=lambda item: len(query_words.intersection(tokens(item[1]))), reverse=True)
        selected = [line for _, line in ranked[:2] if query_words.intersection(tokens(line))]
        if selected: excerpts.append(' '.join(selected) + f" [{source['id']}]")
    return {'answer': 'From the indexed documents:\n\n' + '\n\n'.join(excerpts) + '\n\nDemo mode returns source excerpts. The bundled itinerary is fictional sample content.', 'sources': sources, 'mode': 'demo'}


def local_answer(question, history, store):
    query = local_query(question, history)
    return local_result(query, store.similarity_search(query, k=3))
