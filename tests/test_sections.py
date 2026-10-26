import os
import pytest
from rag import clean, heading_pattern, find_headings, md_sections, pdf_sections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------- clean ----------

def test_clean_collapses_padding_but_keeps_paragraph_breaks():
    raw = "   Title  \n\n\n\n    first   line\n  second line   "
    assert clean(raw) == "Title\n\nfirst line\nsecond line"

# ---------- heading_pattern ----------

@pytest.mark.parametrize("line", [
    "Valid and Reliable",
    "3.1 Valid and Reliable",
    "3.1. Valid and Reliable",
    "3.1Valid and Reliable",        # layout extraction glues the number on
    "3.1 VALID AND RELIABLE",       # papers often capitalize headings
])
def test_heading_matches_numbering_and_case_variants(line):
    assert heading_pattern("Valid and Reliable").search(f"intro\n{line}\nbody")

def test_heading_matches_when_extraction_drops_spaces():
    pat = heading_pattern("Descriptions of AI Actor Tasks from Figures 2 and 3")
    assert pat.search("Appendix A:\nDescriptions of AI Actor Tasks from Figures2and3\n")

def test_heading_must_start_a_line():
    assert not heading_pattern("Architecture").search("● Clean Architecture\n")

def test_heading_must_not_be_a_prefix_of_a_longer_word():
    assert not heading_pattern("Map").search("Mapping the context\n")

# ---------- find_headings ----------

def cut_pages(cuts):
    return {full: page for page, _, full in cuts}

def test_wrong_bookmark_page_is_corrected_by_searching_forward():
    pages = ["Cover", "Some text", "Appendix B\nBody of appendix B"]
    cuts = find_headings(pages, [(0, "Appendix B", "Appendix B")])
    assert cut_pages(cuts) == {"Appendix B": 2}

def test_table_of_contents_page_is_skipped():
    titles = ["Intro", "Method", "Results", "Discussion", "Conclusion"]
    toc = "Contents\n" + "\n".join(f"{t} . . . {i + 2}" for i, t in enumerate(titles))
    pages = [toc] + [f"{t}\nText about {t.lower()}." for t in titles]
    cuts = find_headings(pages, [(0, t, t) for t in titles])
    assert cut_pages(cuts) == {t: i + 1 for i, t in enumerate(titles)}

def test_page_with_many_real_headings_is_not_mistaken_for_contents():
    # A dense first page (like a job description) has many headings but no page numbers
    titles = ["About the Role", "Responsibilities", "Must Have Skills", "Backend", "Databases"]
    page = "\n".join(f"{t}\n● some detail" for t in titles)
    cuts = find_headings([page], [(0, t, t) for t in titles])
    assert cut_pages(cuts) == {t: 0 for t in titles}

def test_search_continues_after_previous_heading():
    # "Safe ..." appears at a line start before the real "Safe" heading
    page = "Safe practices matter.\n3.1 Valid and Reliable\nText\n3.2 Safe\nSafety text"
    cuts = find_headings([page], [(0, "Valid and Reliable", "Valid"), (0, "Safe", "Safe")])
    safe_pos = next(pos for _, pos, full in cuts if full == "Safe")
    assert page[safe_pos:].startswith("3.2 Safe")

def test_missing_heading_is_skipped_without_losing_later_ones():
    pages = ["Intro\ntext", "Results\ntext"]
    cuts = find_headings(pages, [(0, "Intro", "Intro"), (0, "Not In Text", "Gone"), (1, "Results", "Results")])
    assert cut_pages(cuts) == {"Intro": 0, "Results": 1}

# ---------- md_sections ----------

def test_markdown_sections_track_nested_heading_paths():
    text = "Preamble\n# Guide\nIntro\n## Install\nRun pip\n### Mac\nUse brew\n## Usage\nRun ask"
    got = [(section, body) for _, section, body in md_sections(text)]
    assert got == [
        ("", "Preamble"),
        ("Guide", "# Guide\nIntro"),
        ("Guide > Install", "## Install\nRun pip"),
        ("Guide > Install > Mac", "### Mac\nUse brew"),
        ("Guide > Usage", "## Usage\nRun ask"),   # going back up a level drops "Mac"
    ]

def test_markdown_sections_have_no_page():
    assert all(page is None for page, _, _ in md_sections("# A\ntext"))

# ---------- Real PDFs (regression) ----------

def section_start_pages(path):
    starts = {}
    for page, section, _ in pdf_sections(path):
        starts.setdefault(section, page)
    return starts

def test_nist_sections_start_on_their_real_pages():
    # Several bookmarks in this PDF point to the wrong page (e.g. all appendices to 38)
    starts = section_start_pages(os.path.join(ROOT, "samples", "nist-ai-rmf-1.0.pdf"))
    assert starts["Executive Summary"] == 6
    assert starts["AI Risks and Trustworthiness > Valid and Reliable"] == 18
    assert starts["AI RMF Core > Govern"] == 26
    assert starts["Appendix A: Descriptions of AI Actor Tasks from Figures 2 and 3"] == 40
    assert starts["Appendix D: Attributes of the AI RMF"] == 47
    assert len([s for s in starts if s]) == 30  # every bookmark is located in the text

def test_capitalized_paper_headings_are_found():
    # Self-RAG prints "1 INTRODUCTION" for the bookmark "Introduction"
    starts = section_start_pages(os.path.join(ROOT, "samples", "papers", "self-rag.pdf"))
    assert starts["Introduction"] == 1
    assert starts["Related Work"] == 2
