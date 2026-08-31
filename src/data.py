from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path


def load_calibration_data(path: str | Path, sentinel: float = 99999) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    required = ["Left", "Right", "PROD", "TYP", "CAT", "CLMARK", "RNG"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Chybí povinná pole: {missing}")

    df = df[df[required].notna().all(axis=1)].copy()
    invalid = (df["Right"] < sentinel) & (df["Left"] >= df["Right"])
    df = df[~invalid].copy()

    df["_row_id"] = np.arange(len(df), dtype=int)
    df["TYP_STR"] = df["TYP"].astype(str).str.replace(r"\.0$", "", regex=True)
    df["PROD_TYP"] = df["PROD"].astype(str) + "||" + df["TYP_STR"]
    df["CONSTRUCTION"] = np.where(df["CLMARK"].astype(str) == "112.12", "digital", "mechanical")
    rng = pd.to_numeric(df["RNG"], errors="coerce")
    df["RNG_GRP"] = np.where(rng.isin([150,160,200,300]), rng.astype("Int64").astype(str), "Other")
    df["COMBO"] = df["CONSTRUCTION"] + " | RNG " + df["RNG_GRP"].astype(str)
    df["NOK"] = ((df["Right"] < sentinel) & (df["Right"] > df["Left"])).astype(int)
    df["RIGHT_CENSORED"] = df["Right"] >= sentinel
    return df


def support_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    def tab(col):
        return df.groupby(col, dropna=False).agg(n=("NOK","size"), NOK=("NOK","sum")).reset_index()
    return {"combo": tab("COMBO"), "prod": tab("PROD"), "type": tab("PROD_TYP")}


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    out = df
    for field, value in filters.items():
        if value is None or field not in out.columns:
            continue
        if isinstance(value, tuple) and len(value) == 2:
            s = pd.to_datetime(out[field], errors="coerce")
            lo, hi = pd.to_datetime(value[0]), pd.to_datetime(value[1])
            out = out[(s >= lo) & (s <= hi)]
        elif isinstance(value, (list, tuple, set)):
            if len(value):
                out = out[out[field].astype(str).isin([str(x) for x in value])]
        else:
            out = out[out[field].astype(str) == str(value)]
    return out.copy()
