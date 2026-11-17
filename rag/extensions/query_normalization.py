"""Conservative normalization of spoken financial identifiers for retrieval."""
from __future__ import annotations

import re

_FOUR_HUNDRED = r"(?:four[- ]hundred(?:[- ]and)?|four[- ](?:oh|zero|o))"
_SEP = r"[- ]+"
_OPTIONAL_SEP = r"[- ]*"
_ACCOUNT_ALIASES = (
    (rf"(?:401|{_FOUR_HUNDRED}{_SEP}one){_OPTIONAL_SEP}(?:\(k\)|k|kay)", "401(k)"),
    (rf"(?:403|{_FOUR_HUNDRED}{_SEP}three){_OPTIONAL_SEP}(?:\(b\)|b|bee)", "403(b)"),
    (rf"(?:457|four{_SEP}(?:hundred(?:{_SEP}and)?{_SEP})?fifty{_SEP}seven)"
     rf"{_OPTIONAL_SEP}(?:\(b\)|b|bee)", "457(b)"),
)
_SPOKEN_ACRONYMS = (
    "FATCA", "SWIFT", "TFSA", "RRSP", "SIPP", "DRIP",
    "IRA", "ETF", "APR", "APY", "HSA", "FSA", "UAN", "EPF", "PMI", "NRE",
)
_IDENTIFIERS = tuple(
    (re.compile(rf"(?<!\w){body}(?!\w)", re.IGNORECASE), replacement)
    for body, replacement in _ACCOUNT_ALIASES
) + tuple(
    (re.compile(r"(?<!\w)" + _SEP.join(acronym) + r"(?!\w)", re.IGNORECASE), acronym)
    for acronym in _SPOKEN_ACRONYMS
)
_PE_RATIO = re.compile(r"\bprice[- ](?:to[- ])?earnings (ratio(?:s)?)\b", re.IGNORECASE)


def normalize_financial_query(text: str) -> str:
    for pattern, replacement in _IDENTIFIERS:
        text = pattern.sub(replacement, text)
    return _PE_RATIO.sub(r"price-to-earnings (P/E) \1", text)
