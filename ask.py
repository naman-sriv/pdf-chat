import os, ollama, chromadb

ROOT = os.path.dirname(os.path.abspath(__file__))
col = chromadb.PersistentClient(path=os.path.join(ROOT, "db")).get_collection("docs")

print("Ask about your documents. Ctrl+D or 'exit' to quit.")
try:
    while True:
        q = input("\nQuestion: ").strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break
        q_emb = ollama.embed(model="nomic-embed-text", input=q)["embeddings"][0]
        hits = col.query(query_embeddings=[q_emb], n_results=4)
        context = "\n\n".join(hits["documents"][0])
        prompt = (f"Answer using only the context below. If the answer isn't there, say so.\n\n"
                  f"Context:\n{context}\n\nQuestion: {q}")
        reply = ollama.chat(model="llama3.1", messages=[{"role": "user", "content": prompt}])
        print(reply["message"]["content"])
        print("Sources:", {m["source"] for m in hits["metadatas"][0]})
except (EOFError, KeyboardInterrupt):
    print()
