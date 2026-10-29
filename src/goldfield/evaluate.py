"""Field-level evaluation of structured extraction (LLM or not) against gold labels."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .normalize import is_empty, norm_text, parse_bool, parse_date, parse_number, similar

__all__ = ["evaluate", "Report", "FieldScore", "FieldError", "match"]


def _get(d: Any, path: str) -> Any:
    for part in path.split("."):
        if isinstance(d, Mapping):
            d = d.get(part)
        else:
            return None
    return d


def _spec(kind: str) -> tuple[str, float | None]:
    name, _, arg = kind.partition(":")
    return name.strip(), float(arg) if arg else None


def match(kind: str, gold: Any, pred: Any, dayfirst: bool = True) -> bool:
    """Do two values match under a field type: ``text``, ``text:0.9`` (fuzzy), ``number``, ``number:0.01``
    (absolute tolerance), ``number:1%`` (relative), ``date``, ``bool``, ``enum``, ``exact``, ``list[...]``."""
    if kind.startswith("list"):
        inner = kind[5:-1] if kind.startswith("list[") else "text"
        g = gold if isinstance(gold, list) else [gold]
        p = list(pred) if isinstance(pred, list) else [pred]
        if len(g) != len(p):
            return False
        for gv in g:
            hit = next((i for i, pv in enumerate(p) if match(inner, gv, pv, dayfirst)), None)
            if hit is None:
                return False
            p.pop(hit)
        return True
    name, arg_raw = kind.partition(":")[0], kind.partition(":")[2]
    if name == "number":
        a, b = parse_number(gold), parse_number(pred)
        if a is None or b is None:
            return False
        if arg_raw.endswith("%"):
            return abs(a - b) <= abs(a) * float(arg_raw[:-1]) / 100
        return abs(a - b) <= (float(arg_raw) if arg_raw else 1e-9)
    if name == "date":
        a, b = parse_date(gold, dayfirst), parse_date(pred, dayfirst)
        return a is not None and a == b
    if name == "bool":
        a, b = parse_bool(gold), parse_bool(pred)
        return a is not None and a == b
    if name == "exact":
        return gold == pred
    if name == "enum":
        return norm_text(gold) == norm_text(pred)
    a, b = norm_text(gold), norm_text(pred)
    if arg_raw:
        return similar(a, b) >= float(arg_raw)
    return a == b


@dataclass
class FieldScore:
    field: str
    kind: str
    tp: int = 0
    fp: int = 0
    fn: int = 0
    wrong: int = 0
    missing: int = 0
    hallucinated: int = 0
    support: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 1.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def accuracy(self) -> float:
        return self.tp / self.support if self.support else 1.0


@dataclass(frozen=True)
class FieldError:
    id: Any
    field: str
    kind: str  # "wrong" | "missing" | "hallucinated"
    gold: Any
    pred: Any


@dataclass
class Report:
    fields: dict[str, FieldScore]
    errors: list[FieldError]
    documents: int
    perfect_documents: int
    unmatched_predictions: list[Any] = field(default_factory=list)
    unmatched_gold: list[Any] = field(default_factory=list)
    confusion: dict[str, Counter] = field(default_factory=dict)

    @property
    def document_accuracy(self) -> float:
        return self.perfect_documents / self.documents if self.documents else 1.0

    @property
    def micro_f1(self) -> float:
        tp = sum(f.tp for f in self.fields.values())
        fp = sum(f.fp for f in self.fields.values())
        fn = sum(f.fn for f in self.fields.values())
        p = tp / (tp + fp) if tp + fp else 1.0
        r = tp / (tp + fn) if tp + fn else 1.0
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def macro_f1(self) -> float:
        return sum(f.f1 for f in self.fields.values()) / len(self.fields) if self.fields else 1.0

    def worst(self, n: int = 5) -> list[FieldScore]:
        return sorted(self.fields.values(), key=lambda f: f.f1)[:n]

    def to_dict(self) -> dict[str, Any]:
        return {
            "documents": self.documents,
            "document_accuracy": round(self.document_accuracy, 4),
            "micro_f1": round(self.micro_f1, 4),
            "macro_f1": round(self.macro_f1, 4),
            "fields": {
                k: {
                    "kind": f.kind,
                    "precision": round(f.precision, 4),
                    "recall": round(f.recall, 4),
                    "f1": round(f.f1, 4),
                    "accuracy": round(f.accuracy, 4),
                    "wrong": f.wrong,
                    "missing": f.missing,
                    "hallucinated": f.hallucinated,
                    "support": f.support,
                }
                for k, f in self.fields.items()
            },
            "errors": [e.__dict__ for e in self.errors],
            "unmatched_predictions": self.unmatched_predictions,
            "unmatched_gold": self.unmatched_gold,
        }

    def __str__(self) -> str:
        w = max([len(k) for k in self.fields] + [5])
        lines = [
            f"{self.documents} documents · {self.document_accuracy:.1%} fully correct · micro-F1 {self.micro_f1:.3f} · macro-F1 {self.macro_f1:.3f}",
            "",
            f"  {'field':<{w}}  {'prec':>6} {'recall':>6} {'F1':>6}  {'wrong':>5} {'miss':>5} {'halluc':>6}",
        ]
        for k, f in self.fields.items():
            lines.append(f"  {k:<{w}}  {f.precision:>6.1%} {f.recall:>6.1%} {f.f1:>6.3f}  {f.wrong:>5} {f.missing:>5} {f.hallucinated:>6}")
        if self.errors:
            lines += ["", "Examples of errors:"]
            for e in self.errors[:8]:
                lines.append(
                    f"  [{e.id}] {e.field}: {e.kind} — gold {json.dumps(e.gold, ensure_ascii=False, default=str)} · predicted {json.dumps(e.pred, ensure_ascii=False, default=str)}"
                )
        if self.unmatched_gold:
            lines.append(f"\n⚠ {len(self.unmatched_gold)} gold document(s) without a prediction: {self.unmatched_gold[:5]}")
        return "\n".join(lines)

    def to_markdown(self) -> str:
        out = [
            f"**{self.documents} documents** · {self.document_accuracy:.1%} fully correct · micro-F1 **{self.micro_f1:.3f}**",
            "",
            "| Field | Precision | Recall | F1 | Wrong | Missing | Hallucinated |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        out += [f"| `{k}` | {f.precision:.1%} | {f.recall:.1%} | {f.f1:.3f} | {f.wrong} | {f.missing} | {f.hallucinated} |" for k, f in self.fields.items()]
        return "\n".join(out)


def _index(rows: Iterable[Mapping[str, Any]] | Mapping[Any, Mapping[str, Any]], key: str | None) -> dict[Any, Mapping[str, Any]]:
    if isinstance(rows, Mapping):
        return dict(rows)
    rows = list(rows)
    if key is None:
        return dict(enumerate(rows))
    return {r.get(key): r for r in rows}


def evaluate(
    predictions: Sequence[Mapping[str, Any]] | Mapping[Any, Mapping[str, Any]],
    gold: Sequence[Mapping[str, Any]] | Mapping[Any, Mapping[str, Any]],
    fields: Mapping[str, str] | Iterable[str] | None = None,
    key: str | None = None,
    dayfirst: bool = True,
) -> Report:
    """Score extracted records against gold records, field by field.

    ``fields`` maps (dotted) field paths to types, e.g. ``{"invoice.total": "number:0.01", "date": "date",
    "vendor": "text:0.9", "status": "enum", "items": "list[text]"}``. Without it, every key in the gold
    data is compared as text. Records are aligned by ``key`` (e.g. ``"id"``) or by position.
    """
    p_idx, g_idx = _index(predictions, key), _index(gold, key)
    if fields is None:
        names = sorted({k for g in g_idx.values() for k in g if k != key})
        fields = dict.fromkeys(names, "text")
    elif not isinstance(fields, Mapping):
        fields = dict.fromkeys(fields, "text")
    scores = {f: FieldScore(f, k) for f, k in fields.items()}
    errors: list[FieldError] = []
    confusion: dict[str, Counter] = defaultdict(Counter)
    perfect = 0
    for doc_id, g in g_idx.items():
        p = p_idx.get(doc_id, {})
        ok_doc = doc_id in p_idx
        for fname, kind in fields.items():
            s = scores[fname]
            gv, pv = _get(g, fname), _get(p, fname)
            g_empty, p_empty = is_empty(gv), is_empty(pv)
            if g_empty and p_empty:
                continue
            s.support += 1
            if kind == "enum" and not g_empty:
                confusion[fname][(norm_text(gv), "∅" if p_empty else norm_text(pv))] += 1
            if not g_empty and not p_empty and match(kind, gv, pv, dayfirst):
                s.tp += 1
                continue
            ok_doc = False
            if g_empty:
                s.fp += 1
                s.hallucinated += 1
                errors.append(FieldError(doc_id, fname, "hallucinated", gv, pv))
            elif p_empty:
                s.fn += 1
                s.missing += 1
                errors.append(FieldError(doc_id, fname, "missing", gv, pv))
            else:
                s.fp += 1
                s.fn += 1
                s.wrong += 1
                errors.append(FieldError(doc_id, fname, "wrong", gv, pv))
        perfect += ok_doc
    return Report(scores, errors, len(g_idx), perfect, [k for k in p_idx if k not in g_idx], [k for k in g_idx if k not in p_idx], dict(confusion))
