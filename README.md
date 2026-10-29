<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
    <img src="assets/logo.svg" alt="goldfield" width="400">
  </picture>
</p>

<p align="center">
  <a href="https://github.com/yagodejesus/goldfield/actions/workflows/ci.yml"><img src="https://github.com/yagodejesus/goldfield/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/goldfield/"><img src="https://img.shields.io/pypi/v/goldfield.svg" alt="PyPI"></a>
  <a href="https://pypi.org/project/goldfield/"><img src="https://img.shields.io/pypi/pyversions/goldfield.svg" alt="Python"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT"></a>
</p>

**How good is your LLM at extracting data — field by field?** Precision, recall and F1 for every field of a structured extraction, with matching that understands `"R$ 1.234,56" == 1234.56`, `"Sep 15, 2026" == "2026-09-15"` and `"ACAI DA PRACA LTDA." ≈ "Açaí da Praça Ltda"`.

> 🇧🇷 *Avalia a extração de dados por LLMs campo a campo — precisão, recall e F1 — com comparação que entende moedas, datas em vários formatos, textos com e sem acento e listas fora de ordem.*

Teams use LLMs to pull fields out of invoices, contracts, forms and e-mails, then judge the result with exact string matching (which calls `"$89.90"` vs `89.9` an error) or with vibes. `goldfield` gives you the numbers that matter: which fields fail, how (wrong value, missing, or hallucinated), and examples of each.

```text
$ goldfield predictions.jsonl gold.jsonl --key id --schema schema.json
4 documents · 25.0% fully correct · micro-F1 0.875 · macro-F1 0.786

  field        prec recall     F1  wrong  miss halluc
  vendor      75.0%  75.0%  0.750      1     0      0
  date        75.0%  75.0%  0.750      1     0      0
  total      100.0% 100.0%  1.000      0     0      0
  currency   100.0% 100.0%  1.000      0     0      0
  paid       100.0% 100.0%  1.000      0     0      0
  po_number    0.0%   0.0%  0.000      0     1      1
  items      100.0% 100.0%  1.000      0     0      0

Examples of errors:
  [inv-002] vendor: wrong — gold "Acme Corp" · predicted "Acme Corporation"
  [inv-003] date: wrong — gold "2026-08-01" · predicted "2026-01-08"
  [inv-003] po_number: missing — gold "PO-7781" · predicted null
  [inv-004] po_number: hallucinated — gold null · predicted "PO-0001"
```

## Field types

```json
{ "vendor": "text:0.85", "date": "date", "total": "number:0.01", "currency": "enum",
  "paid": "bool", "po_number": "exact", "items": "list[text]", "customer.address.city": "text" }
```

| Type | Matches when |
|---|---|
| `text` | equal after normalizing case, accents, spaces and trailing punctuation |
| `text:0.9` | fuzzy similarity ≥ 0.9 |
| `number`, `number:0.01`, `number:1%` | numeric values within an absolute or relative tolerance — `R$ 1.234,56`, `$1,234.56`, `(12.50)` and `12%` are all parsed |
| `date` | same calendar day: ISO, `31/12/2026` (or month-first), `Sep 15, 2026`, `15 September 2026`, `3 de outubro de 2026`, timestamps |
| `bool` | `true/false`, `yes/no`, `sim/não`, `1/0` |
| `enum` | normalized equality, plus a confusion matrix |
| `exact` | identical values (IDs, codes) |
| `list[<type>]` | same items in any order, each compared with `<type>` |

Dotted paths (`customer.address.city`) reach into nested objects.

## How it counts

Following standard information-extraction practice, for each field of each document:

| Gold | Prediction | Counted as |
|---|---|---|
| value | matching value | true positive |
| value | different value | **wrong** (a false positive *and* a false negative) |
| value | empty | **missing** (false negative) |
| empty | value | **hallucinated** (false positive) |
| empty | empty | ignored |

You get precision, recall, F1 and accuracy per field, micro and macro F1, the share of documents with every field right, and the list of errors.

## Python

```python
from goldfield import evaluate

report = evaluate(predictions, gold, fields={"total": "number:0.01", "date": "date"}, key="id")
report.micro_f1, report.document_accuracy
report.fields["date"].recall
report.worst(3)                 # fields to fix first
report.errors                   # FieldError(id, field, kind, gold, pred)
report.to_markdown()            # for PRs and experiment logs
```

Compare prompts or models by running `evaluate()` on each and diffing the per-field F1 — and add `--fail-under 0.9` to CI so a prompt change can't silently make extraction worse.

## Install

```bash
pip install goldfield
```

Zero dependencies. Inputs are JSONL or JSON arrays (or Python lists/dicts).

## Development

```bash
git clone https://github.com/yagodejesus/goldfield && cd goldfield
pip install -e ".[dev]"
pytest && ruff check .
```

## License

MIT © Yago de Jesus Martins
