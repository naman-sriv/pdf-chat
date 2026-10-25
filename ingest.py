import os, ollama, chromadb
from pypdf import PdfReader

ROOT = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(ROOT, "docs")
EXTS = (".pdf", ".txt", ".md")

client = chromadb.PersistentClient(path=os.path.join(ROOT, "db"))
col = client.get_or_create_collection("docs")

def read(path):
    if path.lower().endswith(".pdf"):
        return "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)
    return open(path, encoding="utf-8").read()

def chunk(text, size=800, overlap=150):
    return [text[i:i+size] for i in range(0, len(text), size - overlap)]

os.makedirs(DOCS, exist_ok=True)
files = sorted(f for f in os.listdir(DOCS) if f.lower().endswith(EXTS))

# Drop chunks from files that were removed from docs/
indexed = {m["source"] for m in col.get(include=["metadatas"])["metadatas"]}
for gone in indexed - set(files):
    col.delete(where={"source": gone})

for fname in files:
    # Clear old chunks first so a shorter re-index leaves nothing stale behind
    col.delete(where={"source": fname})
    chunks = chunk(read(os.path.join(DOCS, fname)))
    if not chunks:
        print("Skipped", fname, "(no extractable text)")
        continue
    embs = ollama.embed(model="nomic-embed-text", input=chunks)["embeddings"]
    col.add(ids=[f"{fname}-{i}" for i in range(len(chunks))], embeddings=embs,
            documents=chunks, metadatas=[{"source": fname}] * len(chunks))
    print("Indexed", fname, f"({len(chunks)} chunks)")

print("Total:", col.count(), "chunks")
