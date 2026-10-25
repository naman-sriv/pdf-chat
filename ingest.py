import os
from rag import build_index

ROOT = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    total = build_index(os.path.join(ROOT, "docs"), os.path.join(ROOT, "db"))
    print("Total:", total, "chunks")
