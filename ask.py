import os, re, ollama, chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

ROOT = os.path.dirname(os.path.abspath(__file__))
CANDIDATES, TOP_K = 20, 4

col = chromadb.PersistentClient(path=os.path.join(ROOT, "db")).get_collection("docs")
store = col.get(include=["documents", "metadatas"])
if not store["ids"]:
    raise SystemExit("Index is empty. Add files to docs/ and run ingest.py first.")
chunks = {i: (d, m) for i, d, m in zip(store["ids"], store["documents"], store["metadatas"])}

def tokens(text):
    return re.findall(r"\w+", text.lower())

# Keyword index over the same text the vectors were built from (heading + chunk)
bm25 = BM25Okapi([tokens(f"{m['section']}\n{d}") for d, m in chunks.values()])
ids = list(chunks)
print("Loading reranker...")
reranker = CrossEncoder("BAAI/bge-reranker-base")

def retrieve(q):
    q_emb = ollama.embed(model="nomic-embed-text", input=q)["embeddings"][0]
    vec = col.query(query_embeddings=[q_emb], n_results=min(CANDIDATES, len(ids)))["ids"][0]
    scores = bm25.get_scores(tokens(q))
    kw = [ids[i] for i in sorted(range(len(ids)), key=lambda i: -scores[i])[:CANDIDATES] if scores[i] > 0]
    # Reciprocal rank fusion: rewards chunks ranked high by either search
    fused = {}
    for ranking in (vec, kw):
        for rank, cid in enumerate(ranking):
            fused[cid] = fused.get(cid, 0) + 1 / (60 + rank)
    pool = sorted(fused, key=fused.get, reverse=True)[:CANDIDATES]
    rerank = reranker.predict([(q, chunks[cid][0]) for cid in pool])
    return [cid for _, cid in sorted(zip(rerank, pool), reverse=True)[:TOP_K]]

def cite(m):
    return ", ".join(x for x in (m["source"], f"p. {m['page']}" if "page" in m else "", m["section"]) if x)

print("Ask about your documents. Ctrl+D or 'exit' to quit.")
try:
    while True:
        q = input("\nQuestion: ").strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break
        top = retrieve(q)
        context = "\n\n".join(f"[{n}] ({cite(chunks[cid][1])})\n{chunks[cid][0]}"
                              for n, cid in enumerate(top, 1))
        prompt = (f"Answer using only the numbered context below, citing passages like [1]. "
                  f"If the answer isn't there, say so.\n\n"
                  f"Context:\n{context}\n\nQuestion: {q}")
        reply = ollama.chat(model="llama3.1", messages=[{"role": "user", "content": prompt}])
        print(reply["message"]["content"])
        print("\nSources:")
        for n, cid in enumerate(top, 1):
            print(f"  [{n}] {cite(chunks[cid][1])}")
except (EOFError, KeyboardInterrupt):
    print()
