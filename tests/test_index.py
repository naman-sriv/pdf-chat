import os
import pytest
import rag
from rag import Retriever, build_index, is_heading_only, md_sections

# ---------- is_heading_only ----------

@pytest.mark.parametrize("chunk, section", [
    ("Risk Tolerance", "Framing Risk > Risk Tolerance"),
    ("1.2.2Risk Tolerance", "Framing Risk > Risk Tolerance"),   # glued section number
    ("4 EXPERIMENTS", "Experiments"),                           # capitalized
    ("## Install", "Guide > Install"),                          # Markdown heading line
])
def test_heading_only_chunks_are_recognized(chunk, section):
    assert is_heading_only(chunk, section)

@pytest.mark.parametrize("chunk, section", [
    ("Risk Tolerance\n\nWhile the AI RMF can be used to prioritize risk", "Risk Tolerance"),
    ("Risk tolerance refers to readiness to bear risk", "Risk Tolerance"),
    ("Some text", ""),                                          # no section at all
])
def test_chunks_with_content_are_kept(chunk, section):
    assert not is_heading_only(chunk, section)

# ---------- md_sections: code fences ----------

def test_hash_comments_in_code_blocks_are_not_headings():
    text = "# Setup\n```bash\n# install deps\npip install x\n```\nDone\n## Usage\nRun it"
    got = [section for _, section, _ in md_sections(text)]
    assert got == ["", "Setup", "Setup > Usage"]

def test_tilde_fences_and_mismatched_markers():
    # A ``` line inside a ~~~ block does not close it
    text = "# A\n~~~\n```\n# not a heading\n~~~\n## B\nx"
    assert [s for _, s, _ in md_sections(text)] == ["", "A", "A > B"]

# ---------- build_index and Retriever (Ollama replaced by a fake) ----------

@pytest.fixture
def fake_embed(monkeypatch):
    calls = []
    def embed(model, input):
        calls.append(input)
        return {"embeddings": [[float(len(t) % 7), 1.0] for t in input]}
    monkeypatch.setattr(rag.ollama, "embed", embed)
    return calls

def write(path, text):
    with open(path, "w") as f:
        f.write(text)

def chunks_of(db, source):
    col = rag.chromadb.PersistentClient(path=db).get_collection("docs")
    return col.get(where={"source": source}, include=["documents"])["documents"]

def test_failed_reindex_keeps_previous_chunks(tmp_path, monkeypatch, fake_embed):
    docs, db = tmp_path / "docs", str(tmp_path / "db")
    docs.mkdir()
    write(docs / "a.md", "# A\nFirst version of the document.")
    write(docs / "b.md", "# B\nAnother document.")
    assert build_index(str(docs), db, log=lambda *a: None) == (2, [])

    # Ollama goes down: the edit to a.md can't be embedded
    write(docs / "a.md", "# A\nSecond version.")
    def down(model, input):
        raise ConnectionError("Ollama not running")
    monkeypatch.setattr(rag.ollama, "embed", down)
    total, failed = build_index(str(docs), db, log=lambda *a: None)

    assert failed == ["a.md", "b.md"]
    assert total == 2
    assert "First version" in chunks_of(db, "a.md")[0]

def test_documents_and_queries_get_their_task_prefixes(tmp_path, fake_embed):
    docs, db = tmp_path / "docs", str(tmp_path / "db")
    docs.mkdir()
    write(docs / "a.md", "# A\nSome text.")
    build_index(str(docs), db, log=lambda *a: None)
    Retriever(db).vector("a question", 1)
    assert all(t.startswith("search_document: ") for t in fake_embed[0])
    assert fake_embed[-1] == ["search_query: a question"]

def test_retriever_on_a_never_indexed_folder_is_empty(tmp_path):
    retriever = Retriever(str(tmp_path / "no-db-yet"))
    assert retriever.ids == []
