"""Shenwan (SW2021) L1 industry universe helpers for research and backtests."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd

log = logging.getLogger(__name__)

SW_SRC_DEFAULT = "SW2021"
SW_L1_CODE_COL = "sw_l1_code"
SW_L1_NAME_COL = "sw_l1_name"
# Bull/bear/uncertain label of the row's own SW L1 industry index, as of the
# row's score_date. Attached by the research worker next to ``attach_sw_l1``.
SW_L1_REGIME_COL = "sw_l1_regime"


def _normalize_symbol_key(symbol: Any) -> str:
    text = str(symbol or "").strip()
    if not text:
        return ""
    return text.split(".")[0]


def _ts_code_variants(symbol: Any) -> List[str]:
    text = str(symbol or "").strip()
    if not text:
        return []
    variants = [text]
    base = text.split(".", 1)[0]
    if base != text:
        variants.append(base)
    if "." not in text:
        for suffix in (".SH", ".SZ", ".BJ"):
            variants.append(f"{text}{suffix}")
    return list(dict.fromkeys(variants))


def _resolve_db_manager(db_manager) -> Any:
    inner = getattr(db_manager, "_db", None)
    return inner if inner is not None else db_manager


def _member_coll(db_manager) -> Any:
    root = _resolve_db_manager(db_manager)
    coll = getattr(root, "sw_index_member", None)
    if coll is not None:
        return coll
    data_db = getattr(root, "data_db", None)
    if data_db is not None:
        return data_db["sw_index_member"]
    raise AttributeError("db_manager has no sw_index_member collection")


def _classify_coll(db_manager) -> Any:
    root = _resolve_db_manager(db_manager)
    coll = getattr(root, "sw_index_classify", None)
    if coll is not None:
        return coll
    data_db = getattr(root, "data_db", None)
    if data_db is not None:
        return data_db["sw_index_classify"]
    raise AttributeError("db_manager has no sw_index_classify collection")


def sw_l1_catalog(db_manager, src: str = SW_SRC_DEFAULT) -> List[Dict[str, str]]:
    """Return sorted L1 entries ``{code, name}`` from ``sw_index_classify``."""
    coll = _classify_coll(db_manager)
    cursor = coll.find(
        {"src": src, "level": "L1"},
        {"_id": 0, "index_code": 1, "industry_name": 1, "name": 1},
    )
    rows: List[Dict[str, str]] = []
    seen: Set[str] = set()
    for doc in cursor:
        code = str(doc.get("index_code") or "").strip()
        if not code or code in seen:
            continue
        name = str(doc.get("industry_name") or doc.get("name") or code).strip()
        rows.append({"code": code, "name": name})
        seen.add(code)
    rows.sort(key=lambda item: item["code"])
    return rows


def _build_symbol_sw_l1_map(db_manager, symbols: Sequence[str], src: str = SW_SRC_DEFAULT) -> Dict[str, Dict[str, str]]:
    """Map bare symbol -> {l1_code, l1_name} using current SW membership snapshot."""
    if not symbols:
        return {}

    ts_to_symbol: Dict[str, str] = {}
    for symbol in symbols:
        bare = _normalize_symbol_key(symbol)
        if not bare:
            continue
        for variant in _ts_code_variants(symbol):
            ts_to_symbol.setdefault(variant, bare)

    if not ts_to_symbol:
        return {}

    coll = _member_coll(db_manager)
    cursor = coll.find(
        {
            "src": src,
            "is_new": "Y",
            "con_code": {"$in": list(ts_to_symbol.keys())},
        },
        {"_id": 0, "con_code": 1, "l1_code": 1, "l1_name": 1},
    )

    out: Dict[str, Dict[str, str]] = {}
    for doc in cursor:
        con_code = str(doc.get("con_code") or "").strip()
        bare = ts_to_symbol.get(con_code)
        if not bare:
            continue
        l1_code = str(doc.get("l1_code") or "").strip()
        l1_name = str(doc.get("l1_name") or "").strip()
        if not l1_code:
            continue
        out[bare] = {"l1_code": l1_code, "l1_name": l1_name}
    return out


def _member_stint_frame(
    db_manager,
    symbols: Sequence[str],
    *,
    src: str = SW_SRC_DEFAULT,
) -> pd.DataFrame:
    """Return every membership stint for ``symbols`` as a dated frame.

    ``sw_index_member`` is a stint table: one stock can carry several historical
    assignments, and ``out_date`` is the last day *in* that industry (inclusive).
    The current assignment has an empty ``out_date``.
    """
    ts_to_symbol: Dict[str, str] = {}
    for symbol in symbols:
        bare = _normalize_symbol_key(symbol)
        if not bare:
            continue
        for variant in _ts_code_variants(symbol):
            ts_to_symbol.setdefault(variant, bare)
    if not ts_to_symbol:
        return pd.DataFrame(columns=["symbol", "in_key", "out_key", SW_L1_CODE_COL, SW_L1_NAME_COL])

    coll = _member_coll(db_manager)
    cursor = coll.find(
        {"src": src, "con_code": {"$in": list(ts_to_symbol.keys())}},
        {"_id": 0, "con_code": 1, "in_date": 1, "out_date": 1, "l1_code": 1, "l1_name": 1},
    )
    rows: List[Dict[str, Any]] = []
    undated = 0
    for doc in cursor:
        bare = ts_to_symbol.get(str(doc.get("con_code") or "").strip())
        if not bare:
            continue
        in_date = str(doc.get("in_date") or "").strip()
        l1_code = str(doc.get("l1_code") or "").strip()
        if not in_date.isdigit() or not l1_code:
            undated += 1
            continue
        out_date = str(doc.get("out_date") or "").strip()
        rows.append(
            {
                "symbol": bare,
                "in_key": int(in_date),
                "out_key": int(out_date) if out_date.isdigit() else None,
                SW_L1_CODE_COL: l1_code,
                SW_L1_NAME_COL: str(doc.get("l1_name") or "").strip(),
            }
        )
    if undated:
        # Skipping these silently is how a half-migrated collection turns into
        # "the filter just stopped matching" with no trace.
        log.warning(
            "sw_index_member: %d row(s) skipped for point-in-time labelling (missing/invalid in_date)",
            undated,
        )
    if not rows:
        return pd.DataFrame(columns=["symbol", "in_key", "out_key", SW_L1_CODE_COL, SW_L1_NAME_COL])
    return pd.DataFrame(rows)


def _attach_sw_l1_pit(
    frame: pd.DataFrame,
    db_manager,
    *,
    as_of_column: str,
    src: str,
) -> pd.DataFrame:
    """Attach the industry that was in force on each row's own date."""
    if as_of_column not in frame.columns:
        raise ValueError(f"attach_sw_l1 as_of_column {as_of_column!r} is not a dataset column")

    score_keys = pd.to_numeric(frame[as_of_column].astype(str), errors="coerce")
    placeable = score_keys.notna()
    frame[SW_L1_CODE_COL] = ""
    frame[SW_L1_NAME_COL] = ""
    if not placeable.any():
        return frame

    stints = _member_stint_frame(db_manager, frame.loc[placeable, "symbol"].astype(str).tolist(), src=src)
    if stints.empty:
        return frame

    # pandas 3 refuses to match StringDtype against object on a join key, so
    # both sides are pinned to plain object strings.
    stints["symbol"] = pd.Series(stints["symbol"].astype(str).to_numpy(), dtype=object, index=stints.index)
    left = pd.DataFrame(
        {
            "_row_id": frame.index.to_numpy(),
            "_score_key": score_keys.to_numpy(),
            "symbol": pd.Series(
                frame["symbol"].map(_normalize_symbol_key).astype(str).to_numpy(),
                dtype=object,
                index=frame.index,
            ),
        }
    ).loc[placeable.to_numpy()]

    merged = pd.merge_asof(
        left.sort_values("_score_key"),
        stints.sort_values("in_key", kind="stable"),
        left_on="_score_key",
        right_on="in_key",
        by="symbol",
        direction="backward",
    )
    # A backward as-of match can land on a stint that had already ended; those
    # rows belong to no industry on that date and must stay unmapped.
    ended = merged["out_key"].notna() & (merged["out_key"] < merged["_score_key"])
    merged.loc[ended, [SW_L1_CODE_COL, SW_L1_NAME_COL]] = ""
    frame.loc[merged["_row_id"], SW_L1_CODE_COL] = (
        merged[SW_L1_CODE_COL].fillna("").astype(str).to_numpy()
    )
    frame.loc[merged["_row_id"], SW_L1_NAME_COL] = (
        merged[SW_L1_NAME_COL].fillna("").astype(str).to_numpy()
    )
    return frame


def attach_sw_l1(
    dataset: pd.DataFrame,
    db_manager,
    *,
    src: str = SW_SRC_DEFAULT,
    as_of_column: Optional[str] = None,
) -> pd.DataFrame:
    """Attach ``sw_l1_code`` / ``sw_l1_name``; unmapped symbols get empty strings.

    ``as_of_column=None`` keeps the original behaviour: the **current** industry
    of every symbol, identical for every row. That is right for display and for
    live plans, and wrong for any historical backtest -- it hands a stock the
    industry it belongs to *today* while judging a date in the past.

    Pass ``as_of_column="score_date"`` for point-in-time membership: each row
    gets the industry that was in force on its own date, resolved from the
    stint table (``in_date <= date <= out_date``, empty ``out_date`` = current).
    """
    if dataset.empty or "symbol" not in dataset.columns:
        return dataset

    if as_of_column is not None:
        return _attach_sw_l1_pit(dataset.copy(), db_manager, as_of_column=as_of_column, src=src)

    frame = dataset.copy()
    mapping = _build_symbol_sw_l1_map(db_manager, frame["symbol"].astype(str).tolist(), src=src)

    def _lookup(symbol: Any, field: str) -> str:
        bare = _normalize_symbol_key(symbol)
        entry = mapping.get(bare, {})
        return str(entry.get(field) or "")

    frame[SW_L1_CODE_COL] = frame["symbol"].map(lambda s: _lookup(s, "l1_code"))
    frame[SW_L1_NAME_COL] = frame["symbol"].map(lambda s: _lookup(s, "l1_name"))
    return frame


def resolve_industry_l1_codes(
    values: Sequence[str],
    catalog: Sequence[Dict[str, str]],
) -> List[str]:
    """Resolve CLI/API values (code or Chinese name) to canonical L1 codes."""
    if not values:
        return []

    code_set = {str(item.get("code") or "").strip() for item in catalog}
    name_to_code: Dict[str, str] = {}
    for item in catalog:
        code = str(item.get("code") or "").strip()
        name = str(item.get("name") or "").strip()
        if code and name:
            name_to_code[name] = code

    resolved: List[str] = []
    seen: Set[str] = set()
    for raw in values:
        token = str(raw or "").strip()
        if not token:
            continue
        if token in code_set:
            code = token
        elif token in name_to_code:
            code = name_to_code[token]
        else:
            raise ValueError(f"Unknown Shenwan L1 industry: {raw}")
        if code not in seen:
            resolved.append(code)
            seen.add(code)
    return resolved


def normalize_industry_l1_filter(values: Optional[Sequence[str]]) -> Tuple[str, ...]:
    """Return effective L1 codes; empty tuple means filter disabled."""
    if not values:
        return ()
    codes: List[str] = []
    for raw in values:
        token = str(raw or "").strip()
        if token:
            codes.append(token)
    return tuple(codes)


def apply_industry_filter(
    frame: pd.DataFrame,
    l1_codes: Sequence[str],
    *,
    code_col: str = SW_L1_CODE_COL,
) -> pd.DataFrame:
    """Filter rows to Shenwan L1 codes; preserves row order."""
    if not l1_codes or frame.empty:
        return frame
    if code_col not in frame.columns:
        return frame.iloc[0:0].copy()

    allowed = {str(code).strip() for code in l1_codes if str(code).strip()}
    if not allowed:
        return frame

    codes = frame[code_col].fillna("").astype(str)
    return frame.loc[codes.isin(allowed)].copy()


def sw_l1_coverage_summary(dataset: pd.DataFrame) -> Dict[str, Any]:
    """Summarize SW L1 mapping coverage on a panel."""
    if dataset.empty:
        return {
            "rows": 0,
            "mapped_rows": 0,
            "unmapped_rows": 0,
            "mapped_share": None,
            "distinct_l1": 0,
        }
    mapped = dataset[SW_L1_CODE_COL].fillna("").astype(str).str.strip() != ""
    rows = int(len(dataset))
    mapped_rows = int(mapped.sum())
    unmapped_rows = rows - mapped_rows
    mapped_share = float(mapped_rows / rows) if rows else None
    distinct_l1 = int(dataset.loc[mapped, SW_L1_CODE_COL].nunique()) if mapped_rows else 0
    return {
        "rows": rows,
        "mapped_rows": mapped_rows,
        "unmapped_rows": unmapped_rows,
        "mapped_share": mapped_share,
        "distinct_l1": distinct_l1,
    }


def is_sw_index_symbol(symbol: str) -> bool:
    return str(symbol or "").strip().upper().endswith(".SI")
