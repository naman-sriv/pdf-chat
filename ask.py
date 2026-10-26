import os, ollama
from rag import Retriever, cite

ROOT = os.path.dirname(os.path.abspath(__file__))
CHAT_MODEL, TOP_K = "llama3.1", 4

def answer(retriever, q):
    top = retriever.search(q, TOP_K)
    context = "\n\n".join(f"[{n}] ({cite(retriever.chunks[cid][1])})\n{retriever.chunks[cid][0]}"
                          for n, cid in enumerate(top, 1))
    prompt = (f"Answer using only the numbered context below, citing passages like [1]. "
              f"If the answer isn't there, say so.\n\n"
              f"Context:\n{context}\n\nQuestion: {q}")
    reply = ollama.chat(model=CHAT_MODEL, messages=[{"role": "user", "content": prompt}])
    return reply["message"]["content"], top

def main():
    retriever = Retriever(os.path.join(ROOT, "db"))
    if not retriever.ids:
        raise SystemExit("Index is empty. Add files to docs/ and run ingest.py first.")
    print("Loading reranker...")
    retriever.load_reranker()
    print("Ask about your documents. Ctrl+D or 'exit' to quit.")
    try:
        while True:
            q = input("\nQuestion: ").strip()
            if not q:
                continue
            if q.lower() in ("exit", "quit"):
                break
            text, top = answer(retriever, q)
            print(text)
            print("\nSources:")
            for n, cid in enumerate(top, 1):
                print(f"  [{n}] {cite(retriever.chunks[cid][1])}")
    except (EOFError, KeyboardInterrupt):
        print()

if __name__ == "__main__":
    main()
