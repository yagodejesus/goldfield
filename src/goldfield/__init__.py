"""goldfield — field-level evaluation for LLM data extraction: precision, recall and F1 per field, with smart value matching."""

__version__ = "0.1.0"

from .evaluate import FieldError, FieldScore, Report, evaluate, match
from .normalize import norm_text, parse_bool, parse_date, parse_number

__all__ = ["__version__", "FieldError", "FieldScore", "Report", "evaluate", "match", "norm_text", "parse_bool", "parse_date", "parse_number"]
