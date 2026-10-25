import os, re, ollama, chromadb
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(ROOT, "docs")
EXTS = (".pdf", ".txt", ".md")

client = chromadb.PersistentClient(path=os.path.join(ROOT, "db"))
col = client.get_or_create_collection("docs")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, chunk_overlap=150, separators=["\n\n", "\n", ". ", " ", ""])

def clean(text):
    # Layout extraction pads lines with spaces; keep blank lines as paragraph breaks
    lines = [re.sub(r"[ \t]+", " ", l).strip() for l in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()

def outline(reader):
    # Flatten PDF bookmarks into (page, "Parent > Child") in reading order
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
    return sorted(out, key=lambda o: o[0])

def pdf_sections(path):
    # Yields (page, section, text); a section runs from one bookmark title to the next
    reader = PdfReader(path)
    marks, section = outline(reader), ""
    for pno, page in enumerate(reader.pages):
        text = clean(page.extract_text(extraction_mode="layout") or "")
        cuts = []
        for mpage, title, full in marks:
            if mpage != pno:
                continue
            m = re.search(r"\s+".join(map(re.escape, title.split())), text)
            if m:
                cuts.append((m.start(), full))
        start = 0
        for pos, full in sorted(cuts):
            yield pno + 1, section, text[start:pos]
            start, section = pos, full
        yield pno + 1, section, text[start:]

def md_sections(text):
    path, buf = [], []
    for line in text.splitlines():
        m = re.match(r"(#{1,6})\s+(.*)", line)
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

os.makedirs(DOCS, exist_ok=True)
files = sorted(f for f in os.listdir(DOCS) if f.lower().endswith(EXTS))

# Drop chunks from files that were removed from docs/
indexed = {m["source"] for m in col.get(include=["metadatas"])["metadatas"]}
for gone in indexed - set(files):
    col.delete(where={"source": gone})

for fname in files:
    # Clear old chunks first so a shorter re-index leaves nothing stale behind
    col.delete(where={"source": fname})
    chunks, metas = [], []
    for page, section, text in sections(os.path.join(DOCS, fname)):
        for c in splitter.split_text(text):
            if " ".join(c.split()) == section.split(" > ")[-1]:
                continue  # heading with no body; its content follows under the same section
            chunks.append(c)
            meta = {"source": fname, "section": section}
            if page:
                meta["page"] = page
            metas.append(meta)
    if not chunks:
        print("Skipped", fname, "(no extractable text)")
        continue
    # Prefix the heading so a chunk's topic is searchable even if the chunk never names it
    embs = ollama.embed(model="nomic-embed-text",
                        input=[f"{m['section']}\n{c}" for c, m in zip(chunks, metas)])["embeddings"]
    col.add(ids=[f"{fname}-{i}" for i in range(len(chunks))], embeddings=embs,
            documents=chunks, metadatas=metas)
    print("Indexed", fname, f"({len(chunks)} chunks)")

print("Total:", col.count(), "chunks")
