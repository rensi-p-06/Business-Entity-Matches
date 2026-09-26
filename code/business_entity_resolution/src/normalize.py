"""
Normalization utilities for business names and addresses.

All of this is pure string processing on the fields already provided in the
dataset (business_name, business_address, country). No external lookups,
geocoding, or reference data are used anywhere in this module or the rest of
the pipeline, per the challenge's fair-play rules.
"""
import re
import unicodedata

# --- Legal-entity suffixes, normalized to a single canonical token (or dropped) ---
# Keys and values are already lowercase; matching is done on lowercase tokens.
LEGAL_SUFFIX_MAP = {
    "inc": "inc", "incorporated": "inc",
    "corp": "corp", "corporation": "corp",
    "co": "co", "company": "co",
    "ltd": "ltd", "limited": "ltd",
    "llc": "llc", "l.l.c": "llc",
    "llp": "llp", "l.l.p": "llp",
    "lp": "lp", "l.p": "lp",
    "pvt": "pvt", "private": "pvt",
    "plc": "plc",
    "pc": "pc", "p.c": "pc",
    "sarl": "sarl", "s.a.r.l": "sarl",
    "sas": "sas", "s.a.s": "sas",
    "sa": "sa", "s.a": "sa",
    "gmbh": "gmbh",
    "and": "&", "et": "&",
}

# Address abbreviation normalization (common US/India/France forms -> canonical)
ADDRESS_ABBR_MAP = {
    "street": "st", "str": "st",
    "road": "rd",
    "avenue": "ave", "av": "ave",
    "boulevard": "blvd", "blvd.": "blvd",
    "drive": "dr",
    "lane": "ln",
    "court": "ct",
    "place": "pl",
    "square": "sq",
    "highway": "hwy",
    "apartment": "apt", "apartments": "apt",
    "building": "bldg",
    "floor": "fl",
    "suite": "ste",
    "unit": "unit",
    "north": "n", "south": "s", "east": "e", "west": "w",
    "saint": "st",
    "rue": "rue", "avenue de": "ave",
}

_PUNCT_RE = re.compile(r"[^\w\s&]", re.UNICODE)
_WS_RE = re.compile(r"\s+")
_NUM_RE = re.compile(r"\d+")


def strip_accents(text: str) -> str:
    """Fold accented/transliterated characters to their ASCII base form."""
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def basic_clean(text: str) -> str:
    """Lowercase, strip accents, drop punctuation (keep &), collapse whitespace."""
    if not isinstance(text, str) or not text:
        return ""
    text = strip_accents(text).lower()
    text = text.replace("&", " & ")
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def normalize_name(name: str) -> str:
    """Normalize a business name: clean text, canonicalize legal suffixes."""
    cleaned = basic_clean(name)
    tokens = [LEGAL_SUFFIX_MAP.get(t, t) for t in cleaned.split()]
    return " ".join(tokens)


def name_core_tokens(name: str) -> set:
    """Tokens of the normalized name with legal-suffix tokens removed.

    Used for blocking / core-similarity: "zephay labs inc" -> {"zephay","labs"}.
    """
    suffix_values = set(LEGAL_SUFFIX_MAP.values()) | {"&"}
    return {t for t in normalize_name(name).split() if t and t not in suffix_values}


def normalize_address(address: str) -> str:
    """Normalize an address string: clean text, canonicalize street abbreviations."""
    cleaned = basic_clean(address)
    tokens = [ADDRESS_ABBR_MAP.get(t, t) for t in cleaned.split()]
    return " ".join(tokens)


def address_tokens(address: str) -> set:
    """Token set of the normalized address, useful for Jaccard / blocking."""
    return {t for t in normalize_address(address).split() if t}


def extract_postal_code(address: str) -> str:
    """Best-effort extraction of a trailing numeric postal/PIN/ZIP code."""
    if not isinstance(address, str):
        return ""
    matches = _NUM_RE.findall(address)
    for m in reversed(matches):
        if len(m) in (5, 6):  # US ZIP (5) or India PIN (6)
            return m
    return matches[-1] if matches else ""


def extract_significant_tokens(tokens: set, min_len: int = 4) -> set:
    """Tokens long enough to be useful blocking keys (drops short noise words)."""
    return {t for t in tokens if len(t) >= min_len}
