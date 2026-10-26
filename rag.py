import os, re, ollama, chromadb
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

EMBED_MODEL = "nomic-embed-text"
RERANK_MODEL = "BAAI/bge-reranker-base"
EXTS = (".pdf", ".txt", ".md")
# nomic-embed-text requires a task prefix on every input, different for stored
# documents and for search queries (see its model card)
DOC_PREFIX, QUERY_PREFIX = "search_document: ", "search_query: "

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, chunk_overlap=150, separators=["\n\n", "\n", ". ", " ", ""])

# ---------- Ingest ----------

def clean(text):
    # Layout extraction pads lines with spaces; keep blank lines as paragraph breaks
    lines = [re.sub(r"[ \t]+", " ", l).strip() for l in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()

def outline(reader):
    # Flatten PDF bookmarks into (page, title, "Parent > Child") in outline order
    out = []
    def walk(items, path):
        last = None
        for it in items:
            if isinstance(it, list):
                walk(it, path + [last] if last else path)
            else:
                last = it.title.strip()
                out.append((reader.get_destination_page_number(it), last, " > ".join(path + [last])))
    try:
        walk(reader.outline, [])
    except Exception:
        return []
    return out

def heading_pattern(title):
    # Line start, optional section number ("3.1", "3.1." or glued "3.1Valid"), then the
    # title; layout extraction sometimes drops spaces, so words may run together.
    # Case-insensitive because papers often print "1 INTRODUCTION" for bookmark "Introduction"
    words = r"\s*".join(map(re.escape, title.split()))
    return re.compile(r"(?mi)^(?:\d+(?:\.\d+)*\.?\s*)?" + words + r"(?!\w)")

def find_headings(pages, marks):
    # Bookmark page numbers can be wrong, so search forward from the previous heading
    # instead of trusting them; skip table-of-contents pages, which list every heading
    pats = [heading_pattern(title) for _, title, _ in marks]
    def toc_entries(page):
        # A contents line is a heading followed by nothing but leader dots and a page number
        return sum(1 for pat in pats for m in pat.finditer(page)
                   if re.fullmatch(r"[\s.]*\d+", page[m.end():].split("\n", 1)[0]))
    toc = {i for i, p in enumerate(pages) if toc_entries(p) >= 5}
    cuts, page, offset = [], 0, 0
    for (_, _, full), pat in zip(marks, pats):
        for pno in range(page, len(pages)):
            if pno in toc:
                continue
            m = pat.search(pages[pno], offset if pno == page else 0)
            if m:
                cuts.append((pno, m.start(), full))
                page, offset = pno, m.end()
                break
    return cuts

def pdf_sections(path):
    # Yields (page, section, text); a section runs from one heading to the next
    reader = PdfReader(path)
    pages = [clean(p.extract_text(extraction_mode="layout") or "") for p in reader.pages]
    cuts = find_headings(pages, outline(reader))
    section = ""
    for pno, text in enumerate(pages):
        start = 0
        for _, pos, full in [c for c in cuts if c[0] == pno]:
            yield pno + 1, section, text[start:pos]
            start, section = pos, full
        yield pno + 1, section, text[start:]

def md_sections(text):
    path, buf, fence = [], [], None
    for line in text.splitlines():
        # A "# comment" inside a fenced code block is not a heading
        marker = re.match(r"\s*(`{3,}|~{3,})", line)
        if marker and (fence is None or marker.group(1).startswith(fence)):
            fence = marker.group(1)[:3] if fence is None else None
        m = None if fence else re.match(r"(#{1,6})\s+(.*)", line)
        if m:
            yield None, " > ".join(path), "\n".join(buf)
            path, buf = path[:len(m.group(1)) - 1] + [m.group(2).strip()], []
        buf.append(line)
    yield None, " > ".join(path), "\n".join(buf)

def sections(path):
    if path.lower().endswith(".pdf"):
        return pdf_sections(path)
    text = open(path, encoding="utf-8").read()
    return md_sections(text) if path.lower().endswith(".md") else [(None, "", text)]

def is_heading_only(chunk, section):
    # True for a chunk that is just its section's heading, however it is printed:
    # "1.2.2Risk Tolerance", "4 EXPERIMENTS" or Markdown "## Install"
    if not section:
        return False
    text = " ".join(chunk.split()).lstrip("#").strip()
    return bool(heading_pattern(section.split(" > ")[-1]).fullmatch(text))

def chunk_file(path, fname):
    chunks, metas = [], []
    for page, section, text in sections(path):
        for c in splitter.split_text(text):
            if is_heading_only(c, section):
                continue  # no body; its content follows under the same section
            chunks.append(c)
            meta = {"source": fname, "section": section}
            if page:
                meta["page"] = page
            metas.append(meta)
    return chunks, metas

def embed(texts, prefix, batch=64):
    out = []
    for i in range(0, len(texts), batch):
        out += ollama.embed(model=EMBED_MODEL, input=[prefix + t for t in texts[i:i+batch]])["embeddings"]
    return out

def build_index(docs_dir, db_path, log=print):
    col = chromadb.PersistentClient(path=db_path).get_or_create_collection("docs")
    os.makedirs(docs_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(docs_dir) if f.lower().endswith(EXTS))

    # Drop chunks from files that were removed from the docs folder
    indexed = {m["source"] for m in col.get(include=["metadatas"])["metadatas"]}
    for gone in indexed - set(files):
        col.delete(where={"source": gone})

    failed = []
    for fname in files:
        # Chunk and embed before touching the index, so a file that fails to parse or
        # embed (e.g. Ollama not running) keeps its previous chunks
        try:
            chunks, metas = chunk_file(os.path.join(docs_dir, fname), fname)
            # Prefix the heading so a chunk's topic is searchable even if the chunk never names it
            embs = embed([f"{m['section']}\n{c}" for c, m in zip(chunks, metas)], DOC_PREFIX)
        except Exception as e:
            log("Failed", fname, f"({type(e).__name__}: {e}); kept its previous chunks")
            failed.append(fname)
            continue
        # Clear old chunks so a shorter re-index leaves nothing stale behind
        col.delete(where={"source": fname})
        if not chunks:
            log("Skipped", fname, "(no extractable text)")
            continue
        col.add(ids=[f"{fname}-{i}" for i in range(len(chunks))], embeddings=embs,
                documents=chunks, metadatas=metas)
        log("Indexed", fname, f"({len(chunks)} chunks)")
    return col.count(), failed

# ---------- Retrieval ----------

def tokens(text):
    return re.findall(r"\w+", text.lower())

class Retriever:
    METHODS = ("bm25", "vector", "hybrid", "rerank")

    def __init__(self, db_path, candidates=20):
        # get_or_create so a never-indexed folder reads as an empty index instead of raising
        self.col = chromadb.PersistentClient(path=db_path).get_or_create_collection("docs")
        store = self.col.get(include=["documents", "metadatas"])
        self.ids = store["ids"]
        self.chunks = {i: (d, m) for i, d, m in zip(store["ids"], store["documents"], store["metadatas"])}
        self.candidates = candidates
        # Keyword index over the same text the vectors were built from (heading + chunk)
        self.bm25 = BM25Okapi([tokens(f"{m['section']}\n{d}") for d, m in self.chunks.values()]) if self.ids else None
        self._reranker = None

    def load_reranker(self):
        # Lazy, because importing torch and the model takes a few seconds; call it
        # up front to pay that cost before the first query
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(RERANK_MODEL)
        return self._reranker

    def vector(self, q, n):
        q_emb = embed([q], QUERY_PREFIX)[0]
        return self.col.query(query_embeddings=[q_emb], n_results=min(n, len(self.ids)))["ids"][0]

    def keyword(self, q, n):
        scores = self.bm25.get_scores(tokens(q))
        return [self.ids[i] for i in sorted(range(len(self.ids)), key=lambda i: -scores[i])[:n] if scores[i] > 0]

    def hybrid(self, q, n):
        # Reciprocal rank fusion: rewards chunks ranked high by either search
        fused = {}
        for ranking in (self.vector(q, self.candidates), self.keyword(q, self.candidates)):
            for rank, cid in enumerate(ranking):
                fused[cid] = fused.get(cid, 0) + 1 / (60 + rank)
        return sorted(fused, key=fused.get, reverse=True)[:n]

    def rerank(self, q, n):
        pool = self.hybrid(q, self.candidates)
        scores = self.load_reranker().predict([(q, self.chunks[cid][0]) for cid in pool])
        return [cid for _, cid in sorted(zip(scores, pool), reverse=True)[:n]]

    def search(self, q, n=4, method="rerank"):
        return {"bm25": self.keyword, "vector": self.vector,
                "hybrid": self.hybrid, "rerank": self.rerank}[method](q, n)

def cite(meta):
    return ", ".join(x for x in (meta["source"], f"p. {meta['page']}" if "page" in meta else "",
                                 meta["section"]) if x)
