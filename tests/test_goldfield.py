import json
from datetime import date
from pathlib import Path

import pytest

from goldfield import evaluate, match, parse_date, parse_number
from goldfield.cli import main

EX = Path(__file__).parent.parent / "examples"


def load(name):
    return [json.loads(x) for x in (EX / name).read_text(encoding="utf-8").splitlines()]


SCHEMA = json.loads((EX / "schema.json").read_text())


def test_invoice_example():
    r = evaluate(load("predictions.jsonl"), load("gold.jsonl"), SCHEMA, key="id")
    f = r.fields
    assert f["total"].f1 == 1.0 and f["paid"].f1 == 1.0 and f["items"].f1 == 1.0 and f["currency"].f1 == 1.0
    assert f["date"].wrong == 1  # inv-003: 2026-08-01 vs 2026-01-08
    assert f["vendor"].wrong == 1  # "Acme Corp" vs "Acme Corporation" below the 0.85 threshold
    assert f["po_number"].missing == 1 and f["po_number"].hallucinated == 1
    assert r.documents == 4 and r.perfect_documents == 1
    assert 0.8 < r.micro_f1 < 0.95
    kinds = sorted((e.id, e.field, e.kind) for e in r.errors)
    assert ("inv-004", "po_number", "hallucinated") in kinds and ("inv-003", "po_number", "missing") in kinds


@pytest.mark.parametrize(
    "text,value",
    [
        ("R$ 1.234,56", 1234.56),
        ("$1,234.56", 1234.56),
        ("1234", 1234.0),
        ("(12.50)", -12.5),
        ("0,5", 0.5),
        ("1,234", 1234.0),
        ("12%", 12.0),
        (7, 7.0),
        ("abc", None),
        (True, None),
    ],
)
def test_numbers(text, value):
    assert parse_number(text) == value


@pytest.mark.parametrize(
    "text,expected",
    [
        ("2026-10-03", date(2026, 10, 3)),
        ("03/10/2026", date(2026, 10, 3)),
        ("31/12/26", date(2026, 12, 31)),
        ("Sep 15, 2026", date(2026, 9, 15)),
        ("15 September 2026", date(2026, 9, 15)),
        ("3 de outubro de 2026", date(2026, 10, 3)),
        ("2026-10-03T14:00:00Z", date(2026, 10, 3)),
        ("someday", None),
    ],
)
def test_dates(text, expected):
    assert parse_date(text) == expected


def test_monthfirst_and_unambiguous_dates():
    assert parse_date("03/10/2026", dayfirst=False) == date(2026, 3, 10)
    assert parse_date("25/12/2026", dayfirst=False) == date(2026, 12, 25)


def test_match_kinds():
    assert match("text", "São Paulo ", "sao paulo") and not match("text", "São Paulo", "Sao Paulos")
    assert match("text:0.9", "Northwind Traders", "Northwind Trader")
    assert match("number:1%", 100, "100.9") and not match("number:1%", 100, 102)
    assert match("number:0.01", "10.004", 10)
    assert match("list[number]", [1, "2,5"], ["2.5", 1]) and not match("list[text]", ["a"], ["a", "b"])
    assert match("exact", 1, 1) and not match("exact", "1", 1)


def test_nested_fields_and_position_alignment():
    gold = [{"a": {"b": "x"}, "c": 1}, {"a": {"b": "y"}, "c": 2}]
    pred = [{"a": {"b": "x"}, "c": 1}, {"a": {"b": "z"}}]
    r = evaluate(pred, gold, {"a.b": "text", "c": "number"})
    assert r.fields["a.b"].wrong == 1 and r.fields["c"].missing == 1 and r.perfect_documents == 1


def test_default_fields_and_unmatched_documents():
    r = evaluate([{"id": 1, "x": "a"}, {"id": 9, "x": "b"}], [{"id": 1, "x": "a"}, {"id": 2, "x": "c"}], key="id")
    assert list(r.fields) == ["x"] and r.unmatched_predictions == [9] and r.unmatched_gold == [2]
    assert r.fields["x"].missing == 1


def test_enum_confusion_and_outputs():
    r = evaluate([{"s": "paid"}, {"s": "open"}], [{"s": "paid"}, {"s": "paid"}], {"s": "enum"})
    assert r.confusion["s"][("paid", "open")] == 1
    assert "| `s` |" in r.to_markdown() and r.to_dict()["fields"]["s"]["wrong"] == 1 and "fully correct" in str(r)


def test_cli(capsys):
    args = [str(EX / "predictions.jsonl"), str(EX / "gold.jsonl"), "--key", "id", "--schema", str(EX / "schema.json")]
    assert main(args) == 0
    assert "micro-F1" in capsys.readouterr().out
    assert main([*args, "--fail-under", "0.99"]) == 1
    assert main([*args, "--format", "json"]) == 0
