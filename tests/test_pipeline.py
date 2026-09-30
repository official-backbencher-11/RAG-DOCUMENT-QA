import hashlib
import os
import sys

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import src.rag_pipeline as rp  # noqa: E402


class FakeEmbeddings(Embeddings):
    """Deterministic bag-of-words hash embeddings (no network)."""
    dim = 64

    def _vec(self, text):
        v = [0.0] * self.dim
        for w in text.lower().split():
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        return v

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def make_pdf(path, pages):
    from reportlab.pdfgen import canvas
    c = canvas.Canvas(str(path))
    for text in pages:
        c.drawString(72, 750, text)
        c.showPage()
    c.save()


@pytest.fixture
def pipe(monkeypatch):
    monkeypatch.setattr(rp, "GoogleGenerativeAIEmbeddings", lambda **k: FakeEmbeddings())
    monkeypatch.setattr(rp, "ChatGoogleGenerativeAI",
                        lambda **k: GenericFakeChatModel(messages=iter([AIMessage(content="ok")] * 10)))
    for k in ("QDRANT_URL", "QDRANT_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    return rp.RAGPipeline()


def count(pipe):
    return pipe.client.count(pipe.collection_name).count


def test_ingest_dedup_and_isolation(pipe, tmp_path):
    f = tmp_path / "a.pdf"
    make_pdf(f, ["Qdrant is a vector database", "Streamlit builds web apps"])
    n = pipe.ingest_document(str(f), "a.pdf", "s1")
    assert n == 2 and count(pipe) == 2
    pipe.ingest_document(str(f), "a.pdf", "s1")          # re-upload: no duplicates
    assert count(pipe) == 2
    pipe.ingest_document(str(f), "a.pdf", "s2")          # other session: separate copy
    assert count(pipe) == 4


def test_retrieval_is_scoped_and_cited(pipe, tmp_path):
    f = tmp_path / "a.pdf"
    make_pdf(f, ["Qdrant is a vector database", "Streamlit builds web apps"])
    pipe.ingest_document(str(f), "a.pdf", "s1")
    stream, docs = pipe.answer_query("what is qdrant", "s1")
    assert "".join(stream) == "ok"
    assert docs and all(d.metadata["source"] == "a.pdf" for d in docs)
    assert docs[0].metadata["page"] == 0
    stream, docs = pipe.answer_query("what is qdrant", "other-session")
    assert docs == [] and rp.NOT_FOUND_MSG in "".join(stream)


def test_clear_session(pipe, tmp_path):
    f = tmp_path / "a.pdf"
    make_pdf(f, ["hello world"])
    pipe.ingest_document(str(f), "a.pdf", "s1")
    pipe.ingest_document(str(f), "a.pdf", "s2")
    pipe.clear_session("s1")
    assert count(pipe) == 1


def test_empty_pdf_rejected(pipe, tmp_path):
    f = tmp_path / "blank.pdf"
    make_pdf(f, [""])
    with pytest.raises(ValueError):
        pipe.ingest_document(str(f), "blank.pdf", "s1")


def test_bad_qdrant_url_falls_back(monkeypatch):
    monkeypatch.setattr(rp, "GoogleGenerativeAIEmbeddings", lambda **k: FakeEmbeddings())
    monkeypatch.setattr(rp, "ChatGoogleGenerativeAI", lambda **k: None)
    monkeypatch.setenv("QDRANT_URL", '"https://nonexistent-cluster.invalid:6333 "')
    monkeypatch.setenv("QDRANT_API_KEY", "'x'")
    p = rp.RAGPipeline()
    assert p.storage_mode == "memory" and "DNS" in p.connection_error


def test_clean():
    assert rp._clean('  "abc" ') == "abc" and rp._clean(None) is None


def test_page_limit(pipe, tmp_path, monkeypatch):
    monkeypatch.setattr(rp, "MAX_PAGES", 2)
    f = tmp_path / "big.pdf"
    make_pdf(f, ["a one", "b two", "c three"])
    with pytest.raises(ValueError, match="limit"):
        pipe.ingest_document(str(f), "big.pdf", "s1")


def test_cleanup_expired(pipe, tmp_path, monkeypatch):
    f = tmp_path / "a.pdf"
    make_pdf(f, ["hello world"])
    pipe.ingest_document(str(f), "a.pdf", "s1")
    monkeypatch.setattr(rp, "SESSION_TTL_HOURS", 0)
    pipe.cleanup_expired()
    assert count(pipe) == 0


def test_retry_then_success(monkeypatch):
    monkeypatch.setattr(rp.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise Exception("429 RESOURCE_EXHAUSTED")
        return "done"
    assert rp._with_retry(flaky) == "done" and calls["n"] == 3


def test_retry_gives_friendly_error(monkeypatch):
    monkeypatch.setattr(rp.time, "sleep", lambda s: None)
    def always():
        raise Exception("429 quota exceeded")
    with pytest.raises(RuntimeError, match="rate limit"):
        rp._with_retry(always)
    def other():
        raise KeyError("x")
    with pytest.raises(KeyError):
        rp._with_retry(other)
