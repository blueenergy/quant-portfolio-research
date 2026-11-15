"""Universe display names and default index benchmark symbols."""

from __future__ import annotations

from typing import Dict, Sequence

UNIVERSE_DISPLAY_NAMES: Dict[str, str] = {
    "csi1000": "中证1000",
    "csi2000": "中证2000",
    "csi500": "中证500",
    "hs300": "沪深300",
    "a500": "中证A500",
    "star50": "科创50",
}

UNIVERSE_BENCHMARK_SYMBOLS: Dict[str, str] = {
    "csi1000": "000852.SH",
    "csi2000": "932000.CSI",
    "csi500": "000905.SH",
    "hs300": "000300.SH",
    "a500": "000510.SH",
    "star50": "000688.SH",
}

FALLBACK_INDEX_BENCHMARK_SYMBOL = "000852.SH"


def universe_slug(universe_index_codes: Sequence[str]) -> str:
    codes = [str(code).strip().lower() for code in universe_index_codes if str(code).strip()]
    return "_".join(codes) if codes else "all"


def universe_title(universe_index_codes: Sequence[str]) -> str:
    codes = [str(code).strip().lower() for code in universe_index_codes if str(code).strip()]
    if not codes:
        return "全市场"
    return " + ".join(UNIVERSE_DISPLAY_NAMES.get(code, code.upper()) for code in codes)


def default_index_benchmark_symbol(universe_index_codes: Sequence[str] | str | None) -> str:
    """Resolve the index whose returns a universe should be measured against.

    Mixed-universe runs have no single representative index, so they keep the
    fallback rather than silently adopting the first member's benchmark.
    """
    if universe_index_codes is None:
        return FALLBACK_INDEX_BENCHMARK_SYMBOL
    if isinstance(universe_index_codes, str):
        codes = [universe_index_codes.strip().lower()] if universe_index_codes.strip() else []
    else:
        codes = [str(code).strip().lower() for code in universe_index_codes if str(code).strip()]
    if len(codes) != 1:
        return FALLBACK_INDEX_BENCHMARK_SYMBOL
    return UNIVERSE_BENCHMARK_SYMBOLS.get(codes[0], FALLBACK_INDEX_BENCHMARK_SYMBOL)
