"""Single CLI entrypoint for both challenges.

Usage:
  python -m agent.cli search "What are the latest specs in MacBook this year?"
  python -m agent.cli pdf-summarize path/to/doc.pdf
  python -m agent.cli pdf-ask path/to/doc.pdf "What methodology was used?"
  python -m agent.cli pdf-chat path/to/doc.pdf
"""

from __future__ import annotations

import argparse
import io
import sys

from . import web_search_agent
from .pdf_rag_agent import PdfRag


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def _cmd_search(args: argparse.Namespace) -> int:
    print(web_search_agent.run(args.query, limit=args.limit))
    return 0


def _cmd_pdf_summarize(args: argparse.Namespace) -> int:
    rag = PdfRag(args.pdf)
    print(rag.summarize())
    return 0


def _cmd_pdf_ask(args: argparse.Namespace) -> int:
    rag = PdfRag(args.pdf)
    print(rag.ask(args.question))
    return 0


def _cmd_pdf_chat(args: argparse.Namespace) -> int:
    rag = PdfRag(args.pdf)
    print(f"Loaded {args.pdf}. Type a question (or 'quit').")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            return 0
        if not q:
            continue
        if q.lower() in {"quit", "exit"}:
            return 0
        print(rag.ask(q))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agent", description="Slooze take-home: AI agents")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("search", help="Web search agent (Challenge A)")
    s.add_argument("query")
    s.add_argument("--limit", type=int, default=5)
    s.set_defaults(func=_cmd_search)

    s = sub.add_parser("pdf-summarize", help="Summarize a PDF (Challenge B)")
    s.add_argument("pdf")
    s.set_defaults(func=_cmd_pdf_summarize)

    s = sub.add_parser("pdf-ask", help="Ask a single question of a PDF (Challenge B)")
    s.add_argument("pdf")
    s.add_argument("question")
    s.set_defaults(func=_cmd_pdf_ask)

    s = sub.add_parser("pdf-chat", help="Interactive Q&A over a PDF (Challenge B)")
    s.add_argument("pdf")
    s.set_defaults(func=_cmd_pdf_chat)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
