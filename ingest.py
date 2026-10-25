import os, ollama, chromadb
from pypdf import PdfReader

client = chromadb.PersistentClient(path="./db")
col = client.get_or_create_collection("docs")

def read(path):
    if path.endswith(".pdf"):
        return "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)
    return open(path, encoding="utf-8").read()

def chunk(text, size=800, overlap=150):
    return [text[i:i+size] for  i in range(0, len(text), size - overlap)]

for fname in os.listdir("docs"):
    for i, c in enumerate(chunk(read(os.path.join("docs", fname)))):
        emb = ollama.embed(model="nomic-embed-text", input=c) ["embeddings"][0]
        col.upsert(ids=[f"{name}-{i}"], embeddings=[emb], document=[c], metadatas=[{"source": fname}])

print("Indexed", col.count(), "chunks")