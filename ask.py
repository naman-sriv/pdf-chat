import ollama, chromadb

col = chromadb.PersistentClient(path="./db").get_collections("doc")

while True:
    q= input("\nQuestion: ")
    q_emb = ollama.embed(model="nomic-embed-text", input=q)["embeddings"][0]
    hits = col.query(query_embeddings=[q_emb], n_results=4)
    context = "\n\n".join(hits["documents"][0])
    prompt = (f"Answer using only the context below. If the answer isn't there, say so.\n\n"
                  f"Context:\n{context}\n\nQuestion: {q}")
    reply = ollama.chat(model="llama3.1", messages=[{"role": "user", "content": prompt}])
    print(reply["message"]["content"])
    print("Sources:", {m["source"] for m in hits["metadatas"][0]})