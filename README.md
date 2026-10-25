# pdf-chat

Chat with your own PDFs and text files, fully locally. Documents are chunked, embedded with Ollama's `nomic-embed-text`, stored in a local Chroma database, and answered by `llama3.1` using only the retrieved context.

## How it works

- **Chunking:** each document is cut into sections at its headings (PDF bookmarks, or `#` headings in Markdown), then split on paragraphs, lines and sentences into chunks of up to 800 characters.
- **Metadata:** every chunk stores its source file, page number (PDFs) and section path, such as `Must Have Skills > Databases`. Answers cite these.
- **Retrieval:** vector search and BM25 keyword search each return 20 candidates, which are merged with reciprocal rank fusion. The `BAAI/bge-reranker-base` cross-encoder reorders them, and the top 4 go to the LLM.

## Requirements

- Python 3.14 (developed on 3.14.8)
- [Ollama](https://ollama.com), running locally

## Setup

```bash
ollama pull nomic-embed-text
ollama pull llama3.1

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

1. Put `.pdf`, `.txt` or `.md` files in `docs/` (created on first run if missing).
2. Build the index:

   ```bash
   python ingest.py
   ```

   Re-run it whenever you add, change or remove documents; each file's chunks are replaced, and files removed from `docs/` are dropped from the index.

3. Ask questions:

   ```bash
   python ask.py
   ```

   Type `exit`, or press Ctrl+D / Ctrl+C, to quit. Each answer cites numbered passages and lists their file, page and section.

   The first run downloads the reranker model (about 1.1 GB) from Hugging Face into `~/.cache/huggingface`. Later runs load it from there, which takes a few seconds at startup.

Both scripts resolve `docs/` and `db/` relative to their own location, so they can be run from any directory.

Scanned PDFs without a text layer have no extractable text and are skipped. PDFs without bookmarks are still chunked and cite page numbers, but have no section names.

## License

MIT, see [LICENSE](LICENSE).
