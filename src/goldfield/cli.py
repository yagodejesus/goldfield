"""Command line: ``goldfield predictions.jsonl gold.jsonl --key id --field total=number:0.01``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .evaluate import evaluate


def _load(path: str) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8").strip()
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="goldfield", description="Field-level precision/recall/F1 for structured extraction.")
    p.add_argument("predictions", help=".jsonl or .json array")
    p.add_argument("gold", help=".jsonl or .json array")
    p.add_argument("--key", help="field that identifies a document (default: align by position)")
    p.add_argument("--field", action="append", default=[], metavar="PATH=TYPE", help="e.g. total=number:0.01, date=date, items=list[text]")
    p.add_argument("--schema", help="JSON file mapping field paths to types")
    p.add_argument("--monthfirst", action="store_true", help="read 03/04/2026 as March 4")
    p.add_argument("--format", choices=("text", "json", "markdown"), default="text")
    p.add_argument("--fail-under", type=float, help="exit 1 if micro-F1 is below this value")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    a = p.parse_args(argv)
    fields = json.loads(Path(a.schema).read_text()) if a.schema else {}
    for f in a.field:
        name, _, kind = f.partition("=")
        fields[name] = kind or "text"
    try:
        rep = evaluate(_load(a.predictions), _load(a.gold), fields or None, a.key, dayfirst=not a.monthfirst)
    except (OSError, json.JSONDecodeError) as e:
        print(f"goldfield: {e}", file=sys.stderr)
        return 2
    print(json.dumps(rep.to_dict(), indent=2, ensure_ascii=False, default=str) if a.format == "json" else rep.to_markdown() if a.format == "markdown" else rep)
    return 1 if a.fail_under is not None and rep.micro_f1 < a.fail_under else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
