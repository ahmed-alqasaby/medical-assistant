"""normalization.py — the single shared ¢6 retrieval normalization contract.

One function runs at index AND query time; the tests assert the locked rules:
diacritics/tashkeel + tatweel stripped, alef/teh-marbuta/yaeh unified, Arabic
punct -> space, Latin case folded on the query side. Pure, deterministic.
"""

from __future__ import annotations

from backend.app.normalization import normalize_arabic_text, normalize_doc_id, normalize_query


class TestNormalizeArabicText:
    def test_strips_diacritics_and_shadda(self) -> None:
        assert normalize_arabic_text("كِتَابٌ مُفِيدٌ") == "كتاب مفيد"

    def test_strips_tatweel_and_zwnj(self) -> None:
        assert normalize_arabic_text("مــركز\u200c") == "مركز"

    def test_unifies_alef_variants(self) -> None:
        assert normalize_arabic_text("أ إ آ ا") == "ا ا ا ا"

    def test_unifies_teh_marbuta(self) -> None:
        assert normalize_arabic_text("جامعةٌ") == "جامعه"  # ة -> ه

    def test_unifies_yaeh(self) -> None:
        assert normalize_arabic_text("موسى") == "موسي"  # ى -> ي

    def test_arabic_punctuation_becomes_space(self) -> None:
        assert normalize_arabic_text("سؤال؟؛،") == "سؤال"

    def test_empty_string(self) -> None:
        assert normalize_arabic_text("") == ""


class TestNormalizeQuery:
    def test_latin_case_folded(self) -> None:
        assert normalize_query("Metformin 500MG") == "metformin 500mg"

    def test_code_switched_query_folds_diacritics_and_case(self) -> None:
        # ascending 'Metformin' glued to a diacritized Arabic phrase
        assert normalize_query("أخذ الميتفورمين Metformin قرصا") == "اخذ الميتفورمين metformin قرصا"

    def test_empty_string(self) -> None:
        assert normalize_query("") == ""
        assert normalize_query("   ") == ""


class TestNormalizeDocId:
    def test_uppercases_and_keeps_allowed_chars(self) -> None:
        # allowed: A-Z 0-9 _ / - ; ':' and '.' are folded to '_'
        assert normalize_doc_id("ma::ar/train.csv") == "MA__AR/TRAIN_CSV"

    def test_strips_disallowed_chars_to_underscore(self) -> None:
        assert normalize_doc_id("rx 001!") == "RX_001_"