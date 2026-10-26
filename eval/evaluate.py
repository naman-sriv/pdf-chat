"""Compare retrieval methods on a labeled question set.

A retrieved chunk counts as relevant when it comes from the question's labeled section
(or one of its subsections) and contains the labeled evidence phrase.
"""
import os, sys, json, time, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from rag import build_index, Retriever

TOP_K, DEPTH = 4, 10  # TOP_K chunks go to the LLM; MRR looks DEPTH deep

def norm(s):
    return " ".join(s.split()).lower()

def relevant(retriever, cid, item):
    doc, meta = retriever.chunks[cid]
    sec = meta["section"]
    return ((sec == item["section"] or sec.startswith(item["section"] + " > "))
            and norm(item["evidence"]) in norm(doc))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default=os.path.join(ROOT, "samples"))
    ap.add_argument("--questions", default=os.path.join(HERE, "questions.json"))
    ap.add_argument("--skip-index", action="store_true", help="reuse the existing eval index")
    ap.add_argument("--out", default=os.path.join(HERE, "results.md"))
    args = ap.parse_args()

    db = os.path.join(HERE, ".db")
    if not args.skip_index:
        print("Indexing", args.docs)
        _, failed = build_index(args.docs, db, log=lambda *a: None)
        if failed:
            raise SystemExit(f"Failed to index {', '.join(failed)}; is Ollama running?")
    retriever = Retriever(db)
    items = json.load(open(args.questions))

    # Every label must point at a real chunk, or a miss would be the label's fault
    bad = [it["q"] for it in items if not any(relevant(retriever, cid, it) for cid in retriever.ids)]
    if bad:
        raise SystemExit("No chunk matches the label for:\n  " + "\n  ".join(bad))

    retriever.load_reranker()  # so model loading isn't counted in the first query's latency
    types = sorted({it["type"] for it in items})
    rows, misses = [], {}
    for method in Retriever.METHODS:
        hit1 = hitk = rr = secs = 0
        by_type = {t: [0, 0] for t in types}
        misses[method] = []
        for it in items:
            start = time.perf_counter()
            ranked = retriever.search(it["q"], DEPTH, method)
            secs += time.perf_counter() - start
            rank = next((r for r, cid in enumerate(ranked, 1) if relevant(retriever, cid, it)), None)
            hit1 += rank == 1
            hitk += bool(rank and rank <= TOP_K)
            rr += 1 / rank if rank else 0
            by_type[it["type"]][0] += bool(rank and rank <= TOP_K)
            by_type[it["type"]][1] += 1
            if not (rank and rank <= TOP_K):
                misses[method].append((it["q"], rank))
        n = len(items)
        rows.append([method, f"{hit1 / n:.0%}", f"{hitk / n:.0%}", f"{rr / n:.2f}"]
                    + [f"{h}/{c}" for h, c in by_type.values()] + [f"{secs / n * 1000:.0f} ms"])

    header = ["Method", "Hit@1", f"Hit@{TOP_K}", f"MRR@{DEPTH}"] + [f"{t} (hit@{TOP_K})" for t in types] + ["Latency"]
    table = "\n".join("| " + " | ".join(r) + " |" for r in
                      [header, ["---"] * len(header)] + rows)
    detail = "\n".join(f"- **{m}** missed {len(ms)}: " + "; ".join(
        f"\"{q}\" (rank {r or f'>{DEPTH}'})" for q, r in ms) for m, ms in misses.items() if ms)

    report = (f"{len(items)} questions over `{os.path.relpath(args.docs, ROOT)}` "
              f"({len(retriever.ids)} chunks).\n\n{table}\n\n### Misses (not in top {TOP_K})\n\n{detail}\n")
    print("\n" + report)
    with open(args.out, "w") as f:
        f.write("# Retrieval evaluation\n\n" + report)

if __name__ == "__main__":
    main()
