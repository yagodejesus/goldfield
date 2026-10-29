"""Value normalizers for comparing extracted fields: text, numbers, dates, booleans, enums."""

from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime
from difflib import SequenceMatcher
from typing import Any

__all__ = ["norm_text", "parse_number", "parse_date", "parse_bool", "similar", "is_empty"]

_MONTHS = {
    "jan": 1,
    "january": 1,
    "janeiro": 1,
    "feb": 2,
    "february": 2,
    "fev": 2,
    "fevereiro": 2,
    "mar": 3,
    "march": 3,
    "marco": 3,
    "apr": 4,
    "april": 4,
    "abr": 4,
    "abril": 4,
    "may": 5,
    "mai": 5,
    "maio": 5,
    "jun": 6,
    "june": 6,
    "junho": 6,
    "jul": 7,
    "july": 7,
    "julho": 7,
    "aug": 8,
    "august": 8,
    "ago": 8,
    "agosto": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "set": 9,
    "setembro": 9,
    "oct": 10,
    "october": 10,
    "out": 10,
    "outubro": 10,
    "nov": 11,
    "november": 11,
    "novembro": 11,
    "dec": 12,
    "december": 12,
    "dez": 12,
    "dezembro": 12,
}


def is_empty(v: Any) -> bool:
    return v is None or (isinstance(v, str) and not v.strip()) or (isinstance(v, (list, dict)) and not v)


def norm_text(v: Any, accents: bool = False) -> str:
    s = unicodedata.normalize("NFKC", str(v)).casefold()
    if not accents:
        s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
    s = re.sub(r"[\s\u00a0]+", " ", s)
    return s.strip(" .,;:")


def similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def parse_number(v: Any) -> float | None:
    """``"R$ 1.234,56"``, ``"$1,234.56"``, ``"(12)"``, ``"12%"``, ``1234`` → float."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    neg = s.startswith("(") and s.endswith(")") or s.startswith("-") or s.endswith("-")
    s = re.sub(r"[^\d.,]", "", s)
    if not s or not re.search(r"\d", s):
        return None
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".") if s.rfind(",") > s.rfind(".") else s.replace(",", "")
    elif "," in s:
        parts = s.split(",")
        thousands = len(parts) > 2 or (len(parts[-1]) == 3 and 1 <= len(parts[0]) <= 3 and parts[0] != "0")
        s = s.replace(",", "") if thousands else s.replace(",", ".")
    elif s.count(".") > 1:
        s = s.replace(".", "")
    try:
        x = float(s)
    except ValueError:
        return None
    return -x if neg else x


def parse_date(v: Any, dayfirst: bool = True) -> date | None:
    """ISO, ``31/12/2026``, ``12/31/2026``, ``2026-12-31T10:00``, ``Dec 31, 2026``, ``31 de dezembro de 2026``…"""
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    s = norm_text(v)
    m = re.match(r"(\d{4})-(\d{1,2})-(\d{1,2})", s)
    if m:
        y, mo, d = map(int, m.groups())
    elif m := re.match(r"(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})$", s):
        a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        y += 2000 if y < 100 else 0
        d, mo = (a, b) if (dayfirst and b <= 12) or a > 12 else (b, a)
    elif m := re.match(r"(\d{1,2})(?:st|nd|rd|th)?\s+(?:de\s+)?([a-z]+)\.?,?\s+(?:de\s+)?(\d{4})", s):
        d, mo, y = int(m.group(1)), _MONTHS.get(m.group(2)), int(m.group(3))
    elif m := re.match(r"([a-z]+)\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})", s):
        mo, d, y = _MONTHS.get(m.group(1)), int(m.group(2)), int(m.group(3))
    else:
        return None
    try:
        return date(y, mo, d) if mo else None
    except (TypeError, ValueError):
        return None


def parse_bool(v: Any) -> bool | None:
    if isinstance(v, bool):
        return v
    s = norm_text(v)
    if s in ("true", "yes", "y", "sim", "s", "1", "verdadeiro"):
        return True
    if s in ("false", "no", "n", "nao", "0", "falso"):
        return False
    return None
