# pdf-chat

Chat with your own PDFs and text files, fully locally. Documents are chunked, embedded with Ollama's `nomic-embed-text`, stored in a local Chroma database, and answered by `llama3.1` using only the retrieved context.

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

   Type `exit`, or press Ctrl+D / Ctrl+C, to quit. Each answer lists the source files it drew on.

Both scripts resolve `docs/` and `db/` relative to their own location, so they can be run from any directory.

Scanned PDFs without a text layer have no extractable text and are skipped.

## License

MIT, see [LICENSE](LICENSE).
