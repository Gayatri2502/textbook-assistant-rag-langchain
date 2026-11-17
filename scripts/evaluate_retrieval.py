"""Run from the project root with python -m scripts.evaluate_retrieval PDF."""
import argparse
from pathlib import Path

from travel_assistant.app import Book
from travel_assistant.retrieval import embed


CASES = [
    ("Exact term", "What does the sinoatrial node do?", [{815, 816}]),
    ("Paraphrase", "What structure sets the pace for each heartbeat?", [{815, 816}]),
    ("Multiple passages", "Compare cardiac and skeletal muscle", [{392, 413}, {813}]),
    ("Absent topic", "Who won the intergalactic xylophonium championship?", []),
]


def report(book, label):
    for name, question, expected in CASES:
        matches = book.search(question, 5)["matches"]
        pages = [hit["page"] for hit in matches]
        passed = all(set(pages) & group for group in expected) if expected else not pages
        print(f"{label:8} {name:20} {'PASS' if passed else 'MISS'} {pages}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="OpenStax Anatomy and Physiology, 1,358-page PDF")
    args = parser.parse_args()
    book = Book(args.pdf, args.pdf.name, 700, 80, "recursive", ["\n\n", "\n", " "])
    report(book, "Baseline")
    book.embeddings = embed([chunk["text"] for chunk in book.chunks])
    report(book, "Hybrid")
