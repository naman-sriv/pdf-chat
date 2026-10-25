# pdf-chat

Chat with your own PDFs and text files, fully locally. Documents are chunked, embedded with Ollama's `nomic-embed-text`, stored in a local Chroma database, and answered by `llama3.1` using only the retrieved context.

## How it works

- **Chunking:** each document is cut into sections at its headings (PDF bookmarks, or `#` headings in Markdown), then split on paragraphs, lines and sentences into chunks of up to 800 characters. Bookmark page numbers are often wrong, so each heading is located in the text itself, searching forward from the previous one and skipping table-of-contents pages.
- **Metadata:** every chunk stores its source file, page number (PDFs) and section path, such as `Must Have Skills > Databases`. Answers cite these.
- **Retrieval:** vector search and BM25 keyword search each return 20 candidates, which are merged with reciprocal rank fusion. The `BAAI/bge-reranker-base` cross-encoder reorders them, and the top 4 go to the LLM.

## Evaluation

[`eval/evaluate.py`](eval/evaluate.py) compares the four retrieval stages on 32 labeled questions about the [NIST AI Risk Management Framework](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf) (48 pages, 247 chunks, included in [`samples/`](samples/) as a public-domain US government work). Each question is labeled with the section holding the answer and a phrase that must appear in the retrieved chunk. 24 questions are worded differently from the document (**concept**); 8 use exact terms such as `GOVERN 6.1` (**exact-term**).

| Method | Hit@1 | Hit@4 | MRR@10 | concept (hit@4) | exact-term (hit@4) | Latency / query |
| --- | --- | --- | --- | --- | --- | --- |
| BM25 only | 44% | 59% | 0.54 | 12/24 | 7/8 | <1 ms |
| Vector only | 53% | 75% | 0.63 | 20/24 | 4/8 | 10 ms |
| Hybrid (RRF) | 50% | 78% | 0.65 | 18/24 | 7/8 | 18 ms |
| Hybrid + reranker | **62%** | **84%** | **0.74** | 19/24 | **8/8** | 342 ms |

Hit@4 is the share of questions whose answer reaches the LLM. What the numbers show:

- **The two searches fail on different questions.** Vector search misses half the exact-term questions, such as section IDs, while BM25 misses half the reworded ones. Hybrid search keeps most of both.
- **The reranker gives the biggest gain.** It lifts Hit@1 from 50% to 62%, at the cost of about a third of a second per query on a laptop CPU/GPU.
- **Vector search still wins on reworded questions.** Fusion with BM25 costs two of those, and reranking recovers only one.

Caveats: 32 questions is a small set, so a one-question difference (3 points) is noise. The questions were written after reading the document, not by independent annotators. Full per-question misses are in [`eval/results.md`](eval/results.md).

```bash
python eval/evaluate.py              # re-index samples/ and evaluate
python eval/evaluate.py --skip-index # reuse the index
```

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

Scanned PDFs without a text layer have no extractable text and are skipped. PDFs without bookmarks are still chunked and cite page numbers, but have no section names. Pages with sidebars or multiple columns can come out with lines from different columns interleaved.

## Project layout

- `rag.py`: ingestion (section detection, chunking, indexing) and the `Retriever` with all four search methods
- `ingest.py`, `ask.py`: command-line entry points
- `eval/`: labeled questions, the evaluation script and its latest results
- `samples/`: the public PDF used for evaluation

## License

MIT, see [LICENSE](LICENSE).
