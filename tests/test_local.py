from pathlib import Path
from langchain_core.documents import Document
from travel_assistant.local import LocalStore, local_answer, read_documents


def test_bundled_documents_answer_with_sources(tmp_path):
    store = LocalStore(tmp_path / 'missing.json')
    result = local_answer('What should I pack for rain?', [], store)
    assert 'umbrella' in result['answer']
    assert result['mode'] == 'demo'
    assert result['sources']
    assert '[' in result['answer']


def test_unsupported_question_has_no_fabricated_answer(tmp_path):
    result = local_answer('What is the population of Neptune?', [], LocalStore(tmp_path / 'missing.json'))
    assert not result['sources']
    assert 'could not find' in result['answer']


def test_local_index_replaces_deleted_content(tmp_path):
    path = tmp_path / 'index.json'
    store = LocalStore(path)
    store.replace([Document(page_content='Cairo reservation breakfast included', metadata={'source': 'stay.txt'})])
    assert store.similarity_search('Cairo')
    store.replace([Document(page_content='Oslo dinner reservation', metadata={'source': 'dinner.txt'})])
    assert not store.similarity_search('Cairo')
    assert store.similarity_search('Oslo')[0].metadata['source'] == 'dinner.txt'


def test_local_followup_uses_previous_topic(tmp_path):
    result = local_answer('What else is there?', [{'role': 'user', 'content': 'Kyoto itinerary'}], LocalStore(tmp_path / 'missing.json'))
    assert result['sources']
    assert any('kyoto' in s['source'] for s in result['sources'])


def test_document_reader_ignores_unsupported_files(tmp_path):
    (tmp_path / 'stay.md').write_text('A sample booking')
    (tmp_path / 'data.csv').write_text('ignored')
    docs = read_documents(tmp_path)
    assert len(docs) == 1
    assert docs[0].metadata['source'] == 'stay.md'
