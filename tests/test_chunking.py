from src.chunking import split_document
from src.preprocess import prepare_document


def test_prepare_document_filters_short():
    assert prepare_document("T", "abc") is None


def test_prepare_document_keeps_valid():
    text = "А" * 100
    result = prepare_document("Заголовок", text)
    assert result is not None
    assert result.startswith("[Заголовок]")


def test_split_document_produces_chunks():
    text = "Слово. " * 200
    prepared = prepare_document("Тест", text)
    chunks = split_document(1, "http://test", "Тест", prepared)
    assert len(chunks) >= 1
    assert all(c.text.strip() for c in chunks)
