# %% [markdown]
# # 01 — Text Normalization Pipeline
#
# **Phase 1, Step 1.1** — Applied identically to all sources (S1, S2, S3,
# train and test).
#
# Each normalization rule is an independently toggleable function (for
# ablation in Step 5.3). Every record is tagged with which rules fired.
#
# Sections:
# 1. Setup & configuration
# 2. Individual normalization rules (toggleable)
# 3. Composite normalization pipeline
# 4. Apply to all sources and save
# 5. Validation & statistics

# %%
from __future__ import annotations
import os
import re
import unicodedata
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import ftfy
import multiprocessing
import concurrent.futures
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

# --- Paths -------------------------------------------------------------------
REPO_ROOT = Path(os.getcwd())
for p in [Path(os.getcwd()), *Path(os.getcwd()).parents]:
    if (p / "dataset").exists():
        REPO_ROOT = p
        break

TRAIN_DIR   = REPO_ROOT / "dataset" / "train"
TEST_DIR    = REPO_ROOT / "dataset" / "test"
SPLITS_DIR  = REPO_ROOT / "dataset" / "splits"
NORM_DIR    = REPO_ROOT / "dataset" / "normalized"
NORM_DIR.mkdir(parents=True, exist_ok=True)

print(f"REPO_ROOT : {REPO_ROOT}")
print(f"NORM_DIR  : {NORM_DIR}")

# %%
# ---------------------------------------------------------------------------
# Configuration — toggle rules on/off for ablation
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# EDA-calibrated toggle table
# Source: eda_results.txt §5 (noise scans) + §7b (catch-all tokens) + §7c (full-data verify)
#
# Priority bands derived from §5 raw counts — §8 catalog not present in current EDA output:
#
# HIGH  (>10%): lowercase (S2 name 19.89%, addr 51.61%), collapse_whitespace (S2 11.01%)
# MEDIUM (1-10%): strip_accent_noise (S3 6.21%), strip_urls (S2 3.50%),
#                 collapse_duplicate_words (S2/S3 2.02%), leetspeak flag (S2 2.77%),
#                 strip_parenthetical_suffix (S3 1.44%), strip_geo_parenthetical (S1 2.01%)
# LOW   (<1%):   split_dba (S3 0.60%), strip_appended_digits (S2 0.58%),
#                strip_ms_prefix (S2 0.51%), strip_hashtag/angle/pipe (~0.31-0.35%),
#                strip_po_box (S2 1.00%), remove_null_literals (addr, S2 2.62%)
#                strip_email (subset of url_in_name — @\w+ counted in EDA url_in_name pattern)
# TRACE (<0.1%): dba_alias S1/S2, null_literal name S1/S2
# ---------------------------------------------------------------------------
RULES_ENABLED = {
    # --- Unicode / encoding (HIGH) ---
    "lowercase":              True,
    "fix_encoding":           True,
    "nfkd_normalize":         True,
    # --- Name cleaning (HIGH / MEDIUM) ---
    "strip_hashtag_prefix":   True,   # §5 S2 0.35%, S3 0.35%  [LOW]
    "strip_angle_brackets":   True,   # §5 S2 0.31%, S3 0.31%  [LOW]
    "strip_appended_digits":  True,   # §5 S2 0.58%, S3 0.57%  [LOW]
    "strip_email":            True,   # §5 url_in_name includes @\w+ (email pattern in EDA scanner)
    "strip_urls":             True,   # §5 S2 3.50%, S3 3.49%  [MEDIUM]
    "normalize_legal_suffix": True,   # 50 suffixes, limited 1.4M top
    "strip_parenthetical_suffix": True,  # §5 S2 1.31%, S3 1.44%  [MEDIUM]
    "strip_geo_parenthetical": True,  # §7b '(india': S1 2.01%, S2 1.60%, S3 1.56%  [MEDIUM]
    "normalize_ampersand":    True,
    "split_dba":              True,   # §5 S3 0.60%  [LOW]
    "collapse_duplicate_words": True, # §5 S2 2.02%, S3 2.02%  [MEDIUM]
    "collapse_whitespace":    True,   # §5 S2 11.01%, S3 10.89%  [HIGH]
    "strip_accent_noise":     True,   # §5 S2 5.77%, S3 6.21%  [MEDIUM]
    "strip_pipe_fragments":   True,   # §5 S2 0.31%, S3 0.31%  [LOW]
    # --- EDA §7b catch-all token rules ---
    "strip_ms_prefix":        True,   # §7b S2 0.51%, S3 0.55%  [LOW]
    "flag_leetspeak":         True,   # §5 S2 2.77%, S3 2.84% — detect only  [MEDIUM]
    # --- Address cleaning ---
    "expand_address_abbrev":  True,
    "remove_null_literals":   True,   # §5 addr S2 2.62%, S3 2.48%  [MEDIUM]
    "strip_po_box":           True,   # §5 addr S2 1.00%, S3 0.94%  [LOW]
}

# %% [markdown]
# ## 2. Individual Normalization Rules
#
# Each rule is a pure function: `(text: str) -> str`.
# Rules return the transformed text. A wrapper tracks which rules fired.

# %%
# ---------------------------------------------------------------------------
# 2a. Unicode & encoding fixes
# ---------------------------------------------------------------------------

def rule_fix_encoding(text: str) -> str:
    """Fix mojibake and encoding errors using ftfy."""
    return ftfy.fix_text(text)


def rule_nfkd_normalize(text: str) -> str:
    """Apply NFKD Unicode normalization (decomposes characters)."""
    return unicodedata.normalize("NFKD", text)


def rule_lowercase(text: str) -> str:
    """Lowercase everything (S2 is ALL-CAPS, S1/S3 are mixed-case)."""
    return text.lower()


# ---------------------------------------------------------------------------
# 2b. Name cleaning rules
# ---------------------------------------------------------------------------

_RE_HASHTAG_PREFIX = re.compile(r"^[#<]+\s*")

def rule_strip_hashtag_prefix(text: str) -> str:
    """Strip hashtag/angle-bracket prefixes: #, <<, ##."""
    return _RE_HASHTAG_PREFIX.sub("", text)


_RE_ANGLE_BRACKETS = re.compile(r"<<|>>")

def rule_strip_angle_brackets(text: str) -> str:
    """Remove stray angle bracket artifacts."""
    return _RE_ANGLE_BRACKETS.sub("", text)


_RE_APPENDED_DIGITS = re.compile(r"\s*[-–]\s*\d{6,}\s*$|[#]\d{5,}\s*$")

def rule_strip_appended_digits(text: str) -> str:
    """Remove appended numeric IDs: '- 2067865001', '#98825'."""
    return _RE_APPENDED_DIGITS.sub("", text)


# ---------------------------------------------------------------------------
# URL / email stripping
#
# EDA 00_eda.py §5 url_in_name pattern (line 292):
#   r"www\.|\.(com|net|org|in|co\.in)|https?://|@\w+"
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   The @\w+ branch means EMAILS are counted inside the url_in_name figure:
#   S2 176,144 (3.50%), S3 184,706 (3.49%).
#
# Two distinct sub-cases:
#   (A) Email embedded or as full name: 'info@shivshakti.com', 'john.doe@company.com'
#   (B) URL embedded in name: 'SHIVSHAKTI CORP | www.shivshakti.com'
#   (C) Whole name IS a URL: 'wilfordhancock.com' (S3 example) — extract domain stem.
# ---------------------------------------------------------------------------

# (A) Email — strip before URL rule so @domain.com doesn’t confuse URL regex
_RE_EMAIL = re.compile(
    r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b",
    re.IGNORECASE,
)

def rule_strip_email(text: str) -> str:
    r"""Strip email addresses from business names.

    EDA: The url_in_name scanner in 00_eda.py includes '@\w+' (email pattern),
    so emails are counted within the S2 3.50% / S3 3.49% url_in_name figure.
    Examples:
      'info@shivshakti.com'              -> '' (whole-name email — flag as empty)
      'Shivshakti Corp info@corp.com'    -> 'Shivshakti Corp'
    """
    return _RE_EMAIL.sub("", text).strip()


# (B+C) URL: embedded or whole-name
# Case C (whole-name IS a URL): extract domain stem instead of erasing everything.
# E.g. 'wilfordhancock.com' — S3 url_in_name example — becomes 'wilfordhancock'
_RE_WHOLE_URL = re.compile(
    r"^(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9\-]*)\.(?:com|net|org|co\.in|in|io|biz)\s*$",
    re.IGNORECASE,
)
# (B) Embedded URL fragment after pipe, space or mid-name
_RE_URL_FRAGMENT = re.compile(
    r"\s*\|?\s*(?:https?://\S+|www\.\S+|\S+\.com\b|\S+\.net\b|\S+\.org\b|\S+\.co\.in\b|\S+\.in\b)",
    re.IGNORECASE,
)

def rule_strip_urls(text: str) -> str:
    """Remove URL fragments from business names.

    EDA: url_in_name S2 176,144 (3.50%), S3 184,706 (3.49%).
    Handles three cases:
      (A) Email: handled by rule_strip_email (run first).
      (B) Embedded URL in name: 'SHIVSHAKTI CORP | www.shivshakti.com' -> 'SHIVSHAKTI CORP'
      (C) Whole name is a URL: 'wilfordhancock.com' -> 'wilfordhancock' (stem preserved)

    Also note: 'com' is the 6th most frequent token in S2 (202,143) and S3 (211,902)
    per §4 token analysis — this is URL/email domain noise, stripped here.
    """
    # Case C: whole name is a URL — return domain stem only
    m = _RE_WHOLE_URL.match(text.strip())
    if m:
        return m.group(1)  # e.g. 'wilfordhancock'
    # Case B: embedded URL fragment
    return _RE_URL_FRAGMENT.sub("", text).strip()


_RE_PIPE_FRAGMENT = re.compile(r"\s*\|.*$")

def rule_strip_pipe_fragments(text: str) -> str:
    """Remove pipe-separated trailing fragments (often URLs or metadata)."""
    return _RE_PIPE_FRAGMENT.sub("", text)


# Legal suffix normalization — canonical mappings
LEGAL_SUFFIX_MAP = {
    # Corporation family
    "corp":         "corporation",
    "corp.":        "corporation",
    "corporation":  "corporation",
    # Limited family
    "ltd":          "limited",
    "ltd.":         "limited",
    "limited":      "limited",
    "ltda":         "limited",   # Portuguese/Spanish variant
    # Private family
    "pvt":          "private",
    "pvt.":         "private",
    "private":      "private",
    # LLC family
    "llc":          "llc",
    "l.l.c":        "llc",
    "l.l.c.":       "llc",
    # Inc family
    "inc":          "inc",
    "inc.":         "inc",
    "incorporated": "inc",
    # LLP / LP
    "llp":          "llp",
    "l.l.p":        "llp",
    "l.l.p.":       "llp",
    "lp":           "lp",
    "l.p.":         "lp",
    # French legal forms
    "sarl":         "sarl",
    "s.a.r.l":      "sarl",
    "s.a.r.l.":     "sarl",
    "sci":          "sci",
    "s.c.i":        "sci",
    "s.c.i.":       "sci",
    "sas":          "sas",
    "s.a.s":        "sas",
    "s.a.s.":       "sas",
    "sa":           "sa",
    "s.a.":         "sa",
    "s.a":          "sa",
    # Other
    "plc":          "plc",
    "pllc":         "pllc",
    "co":           "co",
    "co.":          "co",
    "company":      "company",
    "gmbh":         "gmbh",
    "ag":           "ag",
    "pty":          "pty",
    "ngo":          "ngo",
    "npo":          "npo",
}

# Build regex that matches any suffix at a word boundary
_SUFFIX_PATTERN = "|".join(
    re.escape(k) for k in sorted(LEGAL_SUFFIX_MAP.keys(), key=len, reverse=True)
)
_RE_LEGAL_SUFFIX = re.compile(
    r"\b(" + _SUFFIX_PATTERN + r")\s*$",
    re.IGNORECASE,
)


def rule_normalize_legal_suffix(text: str) -> str:
    """Normalize legal suffixes to canonical forms."""
    def _replace(m):
        return LEGAL_SUFFIX_MAP.get(m.group(1).lower().rstrip(".") if m.group(1).lower().rstrip(".") in LEGAL_SUFFIX_MAP else m.group(1).lower(), m.group(1).lower())
    return _RE_LEGAL_SUFFIX.sub(_replace, text)


# Legal suffix parentheticals: e.g. 'Sweet Barbershop (Co)' → 'Sweet Barbershop co'
# §5: S2 1.31%, S3 1.44%  [MEDIUM]
_RE_PAREN_SUFFIX = re.compile(
    r"\s*\(\s*(?:LLC|LLP|Inc|Corp|Co|Ltd|Limited|Pvt|Private|SA|SAS|SARL|SCI|PC|PLLC|LP)\s*\)\s*",
    re.IGNORECASE,
)


def rule_strip_parenthetical_suffix(text: str) -> str:
    """Strip parentheses around legal suffixes: '(LLC)' -> 'llc'.

    EDA §5: S2 1.31%, S3 1.44% (MEDIUM).
    Example: 'Sweet Barbershop (Co)' -> 'Sweet Barbershop co'
    """
    def _replace(m):
        inner = m.group(0).strip().strip("()")
        canonical = LEGAL_SUFFIX_MAP.get(inner.lower(), inner.lower())
        return " " + canonical
    return _RE_PAREN_SUFFIX.sub(_replace, text)


# Geographic/country parentheticals: e.g. 'Osprey (India) Development LLP'
# §7b catch-all: S1 2.01%, S2 1.60%, S3 1.56%  [MEDIUM]
# These are country qualifiers, NOT legal suffixes — strip the whole token.
_RE_GEO_PAREN = re.compile(
    r"\s*\(\s*(?:india|usa|us|uk|uae|australia|canada|germany|france|singapore|pvt)\s*\)\s*",
    re.IGNORECASE,
)


def rule_strip_geo_parenthetical(text: str) -> str:
    """Strip parenthesised geographic/country qualifiers from business names.

    EDA §7b: '(india' token — S1 2.01%, S2 1.60%, S3 1.56% (MEDIUM).
    Example: 'Osprey (India) Development LLP' -> 'Osprey Development LLP'
    Example: 'GM (INDIA) FABRICS LTD' -> 'GM FABRICS LTD'
    """
    return _RE_GEO_PAREN.sub(" ", text).strip()


def rule_normalize_ampersand(text: str) -> str:
    """Normalize & to 'and'."""
    return text.replace("&", " and ")


# DBA splitting
_RE_DBA = re.compile(r"\s+d/?b/?a\s+", re.IGNORECASE)


def rule_split_dba(text: str) -> Tuple[str, Optional[str]]:
    """Split on 'dba' into (legal_name, alias_name).

    Returns (text, None) if no dba found.
    """
    parts = _RE_DBA.split(text, maxsplit=1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return text, None


_RE_DUPLICATE_WORD = re.compile(r"\b(\w{3,})\s+\1\b", re.IGNORECASE)


def rule_collapse_duplicate_words(text: str) -> str:
    """Collapse duplicate adjacent words: 'Crestline Crestline' -> 'Crestline'."""
    # Apply up to 3 times to handle triple+ duplicates
    for _ in range(3):
        new_text = _RE_DUPLICATE_WORD.sub(r"\1", text)
        if new_text == text:
            break
        text = new_text
    return text


def rule_collapse_whitespace(text: str) -> str:
    """Collapse multiple spaces to single space and strip."""
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# 2c. Accent / diacritic handling
# ---------------------------------------------------------------------------

# Common English words that should NOT have accents.
# If a word after accent-stripping matches one of these, strip the accent.
# This catches "Léarning" -> "Learning", "Nétwork" -> "Network", etc.
# We load a small set; the heuristic is: if the base form (sans accent) is
# a recognizable English word, strip. We keep accents on French names.

# Rather than a full dictionary, use a simple heuristic:
# Strip accents from Latin chars only when the entire name is otherwise ASCII-Latin.
# For French names (which legitimately have accents), the context (country=France
# or French address patterns) helps, but since we process all sources identically,
# we create BOTH versions: accent-stripped and accent-preserved.

def _strip_accents_latin(text: str) -> str:
    """Strip combining diacritics ONLY from Latin base characters.

    Uses NFD decomposition then removes combining marks (category 'M') that
    follow a Latin base character. Non-Latin combining marks (Devanagari matras,
    Tamil vowel signs, Gurmukhi nasalization, etc.) are PRESERVED.

    Bug fixed: previous implementation stripped ALL combining marks, which would
    corrupt the 474k non-Latin S2 records (9.42%) and 278k S3 records (5.27%).
    """
    nfd = unicodedata.normalize("NFD", text)
    out = []
    last_base_is_latin = False
    for ch in nfd:
        cat = unicodedata.category(ch)
        if cat.startswith("M"):           # combining mark of any script
            if last_base_is_latin:
                continue                  # strip: diacritic on a Latin char
            else:
                out.append(ch)           # keep: Indic/other combining mark
        else:
            # Determine if this base character is Latin
            # Latin Unicode blocks: Basic Latin (0000-007F), Latin-1 (0080-00FF),
            # Latin Extended A/B (0100-024F), IPA Extensions (0250-02AF)
            cp = ord(ch)
            last_base_is_latin = (
                cp <= 0x02AF              # Basic Latin through IPA Extensions
                and unicodedata.name(ch, "").startswith("LATIN")
            ) or (
                cat[0] == "L"            # letter
                and 0x0041 <= cp <= 0x024F  # A-z + Latin Extended range
            )
            out.append(ch)
    return "".join(out)


def rule_strip_accent_noise(text: str) -> str:
    """Strip accents that appear to be noise on English/common words.

    Strategy: strip all Latin accents. For French entities, the blocking/
    matching model will use both accent-stripped and original versions.
    """
    return _strip_accents_latin(text)


# ---------------------------------------------------------------------------
# 2d-extra. EDA catch-all rules (§7b — high-frequency anomalous tokens)
# ---------------------------------------------------------------------------

# M/s prefix — Indian business vernacular prefix, S2/S3 ~0.5%
_RE_MS_PREFIX = re.compile(r"^m/s\s+", re.IGNORECASE)

def rule_strip_ms_prefix(text: str) -> str:
    """Strip Indian 'M/s' (Messrs) prefix from business names.

    EDA: S2 0.51%, S3 0.55% prevalence (§7b anomalous tokens).
    Example: 'M/s Pearl Brothers Private' -> 'Pearl Brothers Private'
    """
    return _RE_MS_PREFIX.sub("", text)


# Leetspeak digit detector — S2 2.77%, S3 2.84% (MEDIUM priority)
# Detection-only: we flag but do NOT normalize (modifying semantics is risky).
_RE_LEETSPEAK = re.compile(
    r"(?i)\b\w*[05][a-z]\w*|\b\w*[a-z][05]\w*|\b\w*[1il][a-z]\w*"
    r"|\b[0-9]+[a-z]+[0-9]+\b"
)

def detect_leetspeak(text: str) -> bool:
    """Return True if the name contains probable leetspeak digit substitutions.

    EDA: S2 2.77%, S3 2.84% (MEDIUM).  Used as a feature flag, not cleaned.
    Example: 'Center 5uperior Co' → True
    """
    return bool(_RE_LEETSPEAK.search(text)) if isinstance(text, str) else False


# ---------------------------------------------------------------------------
# 2d. Address cleaning rules
# ---------------------------------------------------------------------------

ADDRESS_ABBREVIATIONS = {
    r"\bst\b":   "street",
    r"\brd\b":   "road",
    r"\bave\b":  "avenue",
    r"\bblvd\b": "boulevard",
    r"\bdr\b":   "drive",
    r"\bln\b":   "lane",
    r"\bct\b":   "court",
    r"\bpl\b":   "place",
    r"\bcir\b":  "circle",
    r"\bhwy\b":  "highway",
    r"\bpkwy\b": "parkway",
    r"\bsq\b":   "square",
    r"\bter\b":  "terrace",
    r"\btpke\b": "turnpike",
    r"\bexpy\b": "expressway",
    r"\bfwy\b":  "freeway",
    r"\bapt\b":  "apartment",
    r"\bste\b":  "suite",
    r"\bfl\b":   "floor",
    r"\bbldg\b": "building",
}


def rule_expand_address_abbrev(text: str) -> str:
    """Expand common address abbreviations."""
    for pattern, expansion in ADDRESS_ABBREVIATIONS.items():
        text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
    return text


_RE_NULL_LITERAL = re.compile(r"\bnull\b", re.IGNORECASE)


def rule_remove_null_literals(text: str) -> str:
    """Remove 'null' string literals from addresses."""
    return _RE_NULL_LITERAL.sub("", text)


# PO Box extraction — S2 1.00%, S3 0.94% (LOW priority)
# Strip the P.O. Box portion so it doesn't pollute TF-IDF blocking keys.
_RE_PO_BOX = re.compile(
    r"\b(?:p\.?\s*o\.?\s*box|po\s*box|post\s*office\s*box)\s*[#\d]*\b",
    re.IGNORECASE,
)

def rule_strip_po_box(text: str) -> str:
    """Remove P.O. Box fragments from address strings.

    EDA: S2 1.00%, S3 0.94% prevalence (§5 address noise scan).
    Keeps the rest of the address; the PO box itself is low-signal for blocking.
    """
    return _RE_PO_BOX.sub("", text).strip()


# ---------------------------------------------------------------------------
# 2e. Script detection (carried over from EDA, for flagging records)
# ---------------------------------------------------------------------------

SCRIPT_BEARING_CATEGORIES = {"L", "M"}

def get_char_script(ch: str) -> Optional[str]:
    """Return the script family for a letter/mark character, or None."""
    if ord(ch) < 128:
        return None
    cat = unicodedata.category(ch)
    if cat[0] not in SCRIPT_BEARING_CATEGORIES:
        return None
    uname = unicodedata.name(ch, "")
    if not uname:
        return None
    return uname.split()[0]


def detect_scripts(text: str) -> Set[str]:
    """Return all script families found in text (e.g., LATIN, DEVANAGARI)."""
    if not isinstance(text, str) or not text.strip():
        return set()
    scripts = set()
    for ch in text:
        s = get_char_script(ch)
        if s is not None:
            scripts.add(s)
    return scripts


def has_non_latin_script(text: str) -> bool:
    """True if text contains non-Latin letter/mark characters."""
    scripts = detect_scripts(text)
    scripts.discard("LATIN")
    return len(scripts) > 0


# %% [markdown]
# ## 3. Composite Normalization Pipeline
#
# Applies all enabled rules in sequence and tracks which rules fired.

# %%
NAME_COL = "business_name"
ADDR_COL = "business_address"


def normalize_name(
    raw_name: str,
    rules_enabled: Dict[str, bool] | None = None,
) -> Dict[str, Any]:
    """Normalize a business name, returning a dict of results.

    Returns
    -------
    dict with keys:
        - name_clean:     primary normalized name
        - name_no_accent: accent-stripped version (for matching)
        - alias_name:     dba alias if found, else None
        - alias_clean:    cleaned alias, else None
        - rules_fired:    list of rule names that modified the text
        - has_non_latin:  bool — contains non-Latin script
        - scripts:        set of detected script families
    """
    cfg = rules_enabled or RULES_ENABLED

    if not isinstance(raw_name, str) or not raw_name.strip():
        return {
            "name_clean": "",
            "name_no_accent": "",
            "alias_name": None,
            "alias_clean": None,
            "rules_fired": [],
            "has_non_latin": False,
            "scripts": set(),
        }

    text = raw_name
    fired: List[str] = []

    def _apply(rule_name, func, txt):
        if not cfg.get(rule_name, True):
            return txt
        result = func(txt)
        if result != txt:
            fired.append(rule_name)
        return result

    # Encoding fix first
    text = _apply("fix_encoding", rule_fix_encoding, text)

    # NFKD
    text = _apply("nfkd_normalize", rule_nfkd_normalize, text)

    # Lowercase
    text = _apply("lowercase", rule_lowercase, text)

    # M/s prefix (EDA §7b, LOW) — strip before hashtag so '#M/s' is caught
    text = _apply("strip_ms_prefix", rule_strip_ms_prefix, text)

    # Strip prefixes/artifacts
    text = _apply("strip_hashtag_prefix", rule_strip_hashtag_prefix, text)
    text = _apply("strip_angle_brackets", rule_strip_angle_brackets, text)

    # Email strip FIRST — EDA url_in_name scanner includes @\w+ so emails are counted
    # inside the S2 3.50% / S3 3.49% url_in_name figure. Must run before URL regex.
    text = _apply("strip_email", rule_strip_email, text)

    # Pipe first (pipes introduce URLs: 'Corp | www.site.com'), then URL strip.
    # rule_strip_urls handles: (B) embedded URL, (C) whole-name-is-URL → extract domain stem.
    # Note: 'com' ranks 6th in S2 (202k) and S3 (212k) token frequency (§4) — cleaned here.
    text = _apply("strip_pipe_fragments", rule_strip_pipe_fragments, text)
    text = _apply("strip_urls", rule_strip_urls, text)

    # Strip appended digits
    text = _apply("strip_appended_digits", rule_strip_appended_digits, text)

    # Legal suffix handling
    text = _apply("strip_parenthetical_suffix", rule_strip_parenthetical_suffix, text)
    # Geographic qualifiers in parens (§7b: S1 2.01%, S2 1.60%, S3 1.56% MEDIUM)
    text = _apply("strip_geo_parenthetical", rule_strip_geo_parenthetical, text)
    text = _apply("normalize_legal_suffix", rule_normalize_legal_suffix, text)

    # Ampersand
    text = _apply("normalize_ampersand", rule_normalize_ampersand, text)

    # Duplicate words
    text = _apply("collapse_duplicate_words", rule_collapse_duplicate_words, text)

    # DBA split — special: returns tuple
    alias_name = None
    alias_clean = None
    if cfg.get("split_dba", True):
        text_before = text
        text, alias_raw = rule_split_dba(text)
        if alias_raw is not None:
            fired.append("split_dba")
            alias_name = alias_raw
            # Clean the alias through the same pipeline (minus dba split)
            alias_cfg = {**cfg, "split_dba": False}
            alias_result = normalize_name(alias_raw, rules_enabled=alias_cfg)
            alias_clean = alias_result["name_clean"]

    # Whitespace cleanup
    text = _apply("collapse_whitespace", rule_collapse_whitespace, text)

    # Detect scripts BEFORE accent stripping (to know original scripts)
    scripts = detect_scripts(text)
    is_non_latin = bool(scripts - {"LATIN"})

    # Leetspeak detection (EDA §5 MEDIUM: S2 2.77%, S3 2.84%) — flag only, no rewrite
    has_leetspeak = False
    if cfg.get("flag_leetspeak", True):
        has_leetspeak = detect_leetspeak(text)

    # Create accent-stripped version
    name_no_accent = text
    if cfg.get("strip_accent_noise", True):
        name_no_accent = rule_strip_accent_noise(text)
        if name_no_accent != text:
            fired.append("strip_accent_noise")

    # Final cleanup on accent-stripped version
    name_no_accent = re.sub(r"\s+", " ", name_no_accent).strip()
    text = re.sub(r"\s+", " ", text).strip()

    return {
        "name_clean": text,
        "name_no_accent": name_no_accent,
        "alias_name": alias_name,
        "alias_clean": alias_clean,
        "rules_fired": fired,
        "has_non_latin": is_non_latin,
        "has_leetspeak": has_leetspeak,
        "scripts": scripts,
    }


def normalize_address(
    raw_addr: str,
    rules_enabled: Dict[str, bool] | None = None,
) -> Dict[str, Any]:
    """Normalize a business address.

    Returns
    -------
    dict with keys:
        - addr_clean:       normalized address
        - addr_no_accent:   accent-stripped version
        - rules_fired:      list of rule names that modified the text
        - has_addr:         bool — address is non-empty
    """
    cfg = rules_enabled or RULES_ENABLED

    if not isinstance(raw_addr, str) or not raw_addr.strip():
        return {
            "addr_clean": "",
            "addr_no_accent": "",
            "rules_fired": [],
            "has_addr": False,
            # EDA §6: S2/S3 missing addr rate 2.87–3.68%; S1 always present.
            # has_addr=False triggers name-only blocking in 02_blocking.py.
        }

    text = raw_addr
    fired: List[str] = []

    def _apply(rule_name, func, txt):
        if not cfg.get(rule_name, True):
            return txt
        result = func(txt)
        if result != txt:
            fired.append(rule_name)
        return result

    text = _apply("fix_encoding", rule_fix_encoding, text)
    text = _apply("nfkd_normalize", rule_nfkd_normalize, text)
    text = _apply("lowercase", rule_lowercase, text)
    # EDA §5: S2 2.62%, S3 2.48% null literals in addresses
    text = _apply("remove_null_literals", rule_remove_null_literals, text)
    # EDA §5: S2 1.00%, S3 0.94% PO Box fragments (LOW) — strip before abbrev expansion
    text = _apply("strip_po_box", rule_strip_po_box, text)
    text = _apply("expand_address_abbrev", rule_expand_address_abbrev, text)
    text = _apply("collapse_whitespace", rule_collapse_whitespace, text)

    # Accent-stripped version
    addr_no_accent = text
    if cfg.get("strip_accent_noise", True):
        addr_no_accent = rule_strip_accent_noise(text)

    addr_no_accent = re.sub(r"\s+", " ", addr_no_accent).strip()
    text = re.sub(r"\s+", " ", text).strip()

    return {
        "addr_clean": text,
        "addr_no_accent": addr_no_accent,
        "rules_fired": fired,
        "has_addr": bool(text.strip()),
    }


# %% [markdown]
# ## 4. Batch Normalization — Apply to DataFrames
#
# Process all sources efficiently with progress tracking.

# %%
def normalize_dataframe(
    df: pd.DataFrame,
    source_label: str = "",
    rules_enabled: Dict[str, bool] | None = None,
) -> pd.DataFrame:
    """Apply normalization to a source DataFrame in-place.

    Adds columns:
    - name_clean, name_no_accent, alias_name, alias_clean
    - has_non_latin, scripts_detected
    - addr_clean, addr_no_accent, has_addr
    - name_rules_fired, addr_rules_fired
    """
    cfg = rules_enabled or RULES_ENABLED
    n = len(df)
    label = f"[{source_label}] " if source_label else ""

    # --- Name normalization ---
    print(f"{label}Normalizing {n:,} business names (up to {N_WORKERS} workers) ...")
    names_list = df[NAME_COL].fillna("").tolist()
    name_results = _parallel_map(
        normalize_name, names_list,
        n_workers=N_WORKERS, chunksize=2000,
        desc=f"{label}Names", total=n,
    )

    df["name_clean"]       = [r["name_clean"] for r in name_results]
    df["name_no_accent"]   = [r["name_no_accent"] for r in name_results]
    df["alias_name"]       = [r["alias_name"] for r in name_results]
    df["alias_clean"]      = [r["alias_clean"] for r in name_results]
    df["has_non_latin"]    = [r["has_non_latin"] for r in name_results]
    # EDA §5: leetspeak_digit MEDIUM — S2 2.77%, S3 2.84%
    df["has_leetspeak"]    = [r.get("has_leetspeak", False) for r in name_results]
    df["scripts_detected"] = [",".join(sorted(r["scripts"])) if r["scripts"] else "" for r in name_results]
    df["name_rules_fired"] = [",".join(r["rules_fired"]) if r["rules_fired"] else "" for r in name_results]

    # --- Address normalization ---
    if ADDR_COL in df.columns:
        print(f"{label}Normalizing {n:,} business addresses (up to {N_WORKERS} workers) ...")
        addrs_list = df[ADDR_COL].fillna("").tolist()
        addr_results = _parallel_map(
            normalize_address, addrs_list,
            n_workers=N_WORKERS, chunksize=2000,
            desc=f"{label}Addrs", total=n,
        )

        df["addr_clean"]       = [r["addr_clean"] for r in addr_results]
        df["addr_no_accent"]   = [r["addr_no_accent"] for r in addr_results]
        df["has_addr"]         = [r["has_addr"] for r in addr_results]
        df["addr_rules_fired"] = [",".join(r["rules_fired"]) if r["rules_fired"] else "" for r in addr_results]
    else:
        df["addr_clean"]       = ""
        df["addr_no_accent"]   = ""
        df["has_addr"]         = False
        df["addr_rules_fired"] = ""

    return df


# %% [markdown]
# ## 5. Load, Normalize, and Save All Sources
#
# We process each source file in chunks to manage memory on large files.

# %%
# ---------------------------------------------------------------------------
# Parallelism configuration
# ---------------------------------------------------------------------------
N_WORKERS = max(1, multiprocessing.cpu_count() - 1)  # 15 on 16-core machine
CHUNK_SIZE = 500_000  # rows per chunk for large-file processing

# On Windows, ProcessPoolExecutor uses 'spawn' which re-imports the module.
# This deadlocks when called from Jupyter (no __main__ guard possible in notebooks).
# _parallel_map() handles this: tries ProcessPoolExecutor, falls back to sequential
# if spawning fails (i.e. when running inside a notebook on Windows).
def _parallel_map(fn, items, n_workers, chunksize=2000, desc="", total=None):
    """Run fn over items in parallel. Falls back to sequential in Jupyter/Windows."""
    items = list(items)
    n = total or len(items)
    if n_workers <= 1:
        return [fn(x) for x in tqdm(items, desc=desc, total=n)]
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as ex:
            return list(tqdm(ex.map(fn, items, chunksize=chunksize), desc=desc, total=n))
    except Exception as e:
        print(f"  [parallel] ProcessPoolExecutor failed ({type(e).__name__}): {e}")
        print(f"  [parallel] Falling back to single-threaded (this is normal in Jupyter on Windows).")
        print(f"  [parallel] Tip: run via 'python' from terminal for full {n_workers}x parallelism.")
        return [fn(x) for x in tqdm(items, desc=desc, total=n)]

def process_source_file(
    input_path: Path,
    output_path: Path,
    source_label: str,
    rules_enabled: Dict[str, bool] | None = None,
) -> pd.DataFrame:
    """Load a source TSV, normalize, save, and return the DataFrame.

    For very large files, processes in chunks.
    """
    cfg = rules_enabled or RULES_ENABLED

    print(f"\n{'='*60}")
    print(f"Processing {source_label}: {input_path.name}")
    print(f"{'='*60}")

    # Count total rows
    total_rows = sum(1 for _ in open(input_path, "r", encoding="utf-8")) - 1
    print(f"  Total rows: {total_rows:,}")

    if total_rows <= CHUNK_SIZE:
        # Small enough to load in one go
        df = pd.read_csv(input_path, sep="\t", dtype=str, low_memory=False)
        df = normalize_dataframe(df, source_label, cfg)
        df.to_csv(output_path, sep="\t", index=False)
        print(f"  Saved to {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")
        return df
    else:
        # Process in chunks
        chunks = []
        reader = pd.read_csv(input_path, sep="\t", dtype=str, low_memory=False, chunksize=CHUNK_SIZE)

        for i, chunk_df in enumerate(reader):
            chunk_label = f"{source_label} chunk {i+1}"
            chunk_df = normalize_dataframe(chunk_df, chunk_label, cfg)
            chunks.append(chunk_df)

        df = pd.concat(chunks, ignore_index=True)
        df.to_csv(output_path, sep="\t", index=False)
        print(f"  Saved to {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")
        return df


# %%
def main():
    # --- Process training sources ------------------------------------------------
    print("\n" + "="*70)
    print("PHASE 1: TEXT NORMALIZATION — TRAINING DATA")
    print("="*70)

    train_s1 = process_source_file(
        TRAIN_DIR / "train_source1.tsv",
        NORM_DIR / "train_source1_norm.tsv",
        "Train-S1",
    )

    train_s2 = process_source_file(
        TRAIN_DIR / "train_source2.tsv",
        NORM_DIR / "train_source2_norm.tsv",
        "Train-S2",
    )

    train_s3 = process_source_file(
        TRAIN_DIR / "train_source3.tsv",
        NORM_DIR / "train_source3_norm.tsv",
        "Train-S3",
    )

    # --- Process test sources ----------------------------------------------------
    print("\n" + "="*70)
    print("PHASE 1: TEXT NORMALIZATION — TEST DATA")
    print("="*70)

    test_s1 = process_source_file(
        TEST_DIR / "test_source1.tsv",
        NORM_DIR / "test_source1_norm.tsv",
        "Test-S1",
    )

    test_s2 = process_source_file(
        TEST_DIR / "test_source2.tsv",
        NORM_DIR / "test_source2_norm.tsv",
        "Test-S2",
    )

    test_s3 = process_source_file(
        TEST_DIR / "test_source3.tsv",
        NORM_DIR / "test_source3_norm.tsv",
        "Test-S3",
    )

    # --- Validation & Statistics -------------------------------------------------
    print("\n" + "="*70)
    print("NORMALIZATION STATISTICS")
    print("="*70)

    for label, df in [
        ("Train-S1", train_s1), ("Train-S2", train_s2), ("Train-S3", train_s3),
        ("Test-S1", test_s1),   ("Test-S2", test_s2),   ("Test-S3", test_s3),
    ]:
        n = len(df)
        name_changed = (df[NAME_COL].fillna("").str.lower() != df["name_clean"]).sum()
        non_latin    = df["has_non_latin"].sum() if "has_non_latin" in df.columns else 0
        has_alias    = df["alias_name"].notna().sum() if "alias_name" in df.columns else 0
        has_addr_n   = df["has_addr"].sum() if "has_addr" in df.columns else 0

        print(f"\n--- {label} ({n:,} rows) ---")
        print(f"  Names changed by normalization: {name_changed:,} ({name_changed/n*100:.1f}%)")
        print(f"  Non-Latin script records:       {non_latin:,} ({non_latin/n*100:.1f}%)")
        print(f"  DBA aliases found:              {has_alias:,} ({has_alias/n*100:.1f}%)")
        print(f"  Records with address:           {has_addr_n:,} ({has_addr_n/n*100:.1f}%)")

        # Rule fire rates
        if "name_rules_fired" in df.columns:
            rule_counter = Counter()
            for fired_str in df["name_rules_fired"].dropna():
                if fired_str:
                    for r in fired_str.split(","):
                        rule_counter[r.strip()] += 1
            if rule_counter:
                print(f"  Rule fire rates (name):")
                for rule, cnt in rule_counter.most_common():
                    print(f"    {rule:<30}: {cnt:>8,} ({cnt/n*100:5.2f}%)")

    # Quick spot-check: show some examples of normalized names
    print("\n" + "="*70)
    print("SPOT-CHECK EXAMPLES")
    print("="*70)

    for label, df in [("Train-S2", train_s2), ("Train-S3", train_s3)]:
        changed = df[df[NAME_COL].fillna("").str.lower() != df["name_clean"]].head(10)
        if not changed.empty:
            print(f"\n--- {label}: names that changed ---")
            for _, row in changed.iterrows():
                print(f"  BEFORE: {row[NAME_COL]}")
                print(f"  AFTER:  {row['name_clean']}")
                print(f"  RULES:  {row['name_rules_fired']}")
                if row.get("alias_name"):
                    print(f"  ALIAS:  {row['alias_name']} -> {row['alias_clean']}")
                print()

    # EDA-calibrated sanity check
    EXPECTED_FIRE_RATES_S2 = {
        "lowercase":                (0.50, 0.95),
        "collapse_whitespace":      (0.08, 0.18),
        "strip_accent_noise":       (0.08, 0.20),
        "strip_urls":               (0.025, 0.05),
        "collapse_duplicate_words": (0.014, 0.028),
        "strip_parenthetical_suffix": (0.009, 0.018),
        "strip_geo_parenthetical":  (0.011, 0.021),
        "strip_appended_digits":    (0.004, 0.009),
        "strip_hashtag_prefix":     (0.002, 0.006),
        "split_dba":                (0.000, 0.004),
        "strip_ms_prefix":          (0.003, 0.010),
    }

    if "Train-S2" in [lbl for lbl, _ in [("Train-S2", train_s2)]]:
        print("\nSanity-checking S2 rule fire rates against EDA expectations ...")
        rule_counter_s2 = Counter()
        n_s2 = len(train_s2)
        for fired_str in train_s2.get("name_rules_fired", pd.Series(dtype=str)).fillna(""):
            if fired_str:
                for r in fired_str.split(","):
                    rule_counter_s2[r.strip()] += 1

        all_ok = True
        for rule, (lo, hi) in EXPECTED_FIRE_RATES_S2.items():
            rate = rule_counter_s2.get(rule, 0) / n_s2 if n_s2 > 0 else 0
            status = "✓" if lo <= rate <= hi else "✗"
            if status == "✗":
                all_ok = False
            print(f"  {status} {rule:<30}: {rate:.4f}  (expected {lo:.3f}–{hi:.3f})")
        if all_ok:
            print("  All fire rates within EDA-expected range.")
        else:
            print("  WARNING: Some fire rates outside expected range — review normalization rules.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()

# %% [markdown]
# ## Next Steps
#
# 1. Normalized data saved to `dataset/normalized/` — these are the inputs
#    for blocking (`02_blocking.py`).
# 2. Non-Latin flagged records (`has_non_latin=True`) need embedding-based blocking (Strategy 5).
#    EDA: S2 9.42% (474k), S3 5.27% (278k) non-Latin — see 02_blocking.py EMBEDDING_* params.
# 3. DBA aliases stored separately for dual-name similarity features.
# 4. Leetspeak-flagged records (`has_leetspeak=True`, S2/S3 ~2.8%) go to feature engineering
#    as a binary feature — do NOT normalize tokens.
# 5. `has_addr=False` records (S2/S3 ~3–4%) use name-only blocking path in 02_blocking.py.
# 6. Rule fire rates feed into ablation table (Phase 5, Step 5.3).
