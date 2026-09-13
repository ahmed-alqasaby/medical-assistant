"""Arabic + Latin normalization — the single shared retrieval contract (§6).

One function runs at BOTH index time and query time so a diacritized or
code-switched Arabic query lands in the same canonical space as the corpus.
This is the cheapest retrieval-accuracy lever in the whole system; the spec
forbids branching it by side.

Rules (locked in the spec's §6 normalization contract):
  - strip diacritics/tashkeel (incl. shadda) and tatweel/kashida + ZWNJ
  - unify alef variants أ إ آ -> ا ; teh marbuta ة -> ه ; yaeh ى -> ي
  - strip punctuation relevant to matching, collapse whitespace
This file is pure; it never touches disk or models. Indexing (bge-m3 +
Chroma) lives in retrieval.py, generation in generation.py.
"""

from __future__ import annotations

import re
import unicodedata

_DIACRITICS_RE = re.compile(
    r"[\u064B-\u065F\u0670\u0671\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED\u08D3-\u08FF]"
)
_TATWEEL_RE = re.compile(r"[\u0640\u200C\u200B]")
_WS_RE = re.compile(r"\s+")
_ARABIC_PUNCT_RE = re.compile(r"[\u060C\u061B\u061F\u066A\u066B\u066C\u0640]")


def normalize_arabic_text(text: str) -> str:
    """Same normalization at index and query time. Never branch on side."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = _DIACRITICS_RE.sub("", text)
    text = _TATWEEL_RE.sub("", text)
    # Alef variants -> plain alef; teh-marbuta -> heh; yaeh -> yeh
    text = text.replace("\u0623", "\u0627").replace("\u0625", "\u0627").replace("\u0622", "\u0627")
    text = text.replace("\u0629", "\u0647")  # ة -> ه
    text = text.replace("\u0649", "\u064A")  # ى -> ي
    text = _ARABIC_PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def normalize_query(text: str) -> str:
    """Query side: Arabic normalization + Latin token folding, so a
    code-switched query (e.g. ascending 'Metformin' mixed into Arabic)
    matches its Latin-spelling index tokens regardless of Arabic diacritics."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)  # splits Latin+Arabic composed forms
    text = normalize_arabic_text(text)
    text = text.lower()  # Latin folding; Arabic case-insensitive by nature
    return text


def normalize_doc_id(value: str) -> str:
    """Doc-id registry entries (§3/§6.2): uppercase, A-Z0-9/_-, stable key."""
    return re.sub(r"[^A-Z0-9_/-]", "_", value.upper())
