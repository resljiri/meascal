from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path


def load_calibration_data(path: str | Path, sentinel: float = 99999) -> pd.DataFrame:
    """Load calibration history while retaining rows with partially missing covariates.

    Only Left/Right are mandatory for reliability likelihood. Missing PROD/TYP/CAT/RNG
    no longer removes the whole record. Such a record can still inform every model term
    for which its covariate is known during a refit.
    """
    df = pd.read_csv(path).copy()
    required_time = ["Left", "Right"]
    model_fields = ["PROD", "TYP", "CAT", "CLMARK", "RNG"]
    missing = [c for c in required_time + model_fields if c not in df.columns]
    if missing:
        raise ValueError(f"Chybí povinné sloupce databáze: {missing}")

    # Reliability information itself must be usable.
    df = df[df[required_time].notna().all(axis=1)].copy()
    invalid = (df["Right"] < sentinel) & (df["Left"] >= df["Right"])
    df = df[~invalid].copy()

    df["_row_id"] = np.arange(len(df), dtype=int)

    typ = df["TYP"].where(df["TYP"].notna(), pd.NA)
    df["TYP_STR"] = typ.astype("string").str.replace(r"\.0$", "", regex=True)

    both_pt = df["PROD"].notna() & df["TYP_STR"].notna()
    df["PROD_TYP"] = pd.Series(pd.NA, index=df.index, dtype="string")
    df.loc[both_pt, "PROD_TYP"] = (
        df.loc[both_pt, "PROD"].astype(str) + "||" + df.loc[both_pt, "TYP_STR"].astype(str)
    )

    clm_known = df["CLMARK"].notna()
    df["CONSTRUCTION"] = pd.Series(pd.NA, index=df.index, dtype="string")
    df.loc[clm_known, "CONSTRUCTION"] = np.where(
        df.loc[clm_known, "CLMARK"].astype(str) == "112.12", "digital", "mechanical"
    )

    rng = pd.to_numeric(df["RNG"], errors="coerce")
    df["RNG_GRP"] = pd.Series(pd.NA, index=df.index, dtype="string")
    known_rng = rng.notna()
    standard = known_rng & rng.isin([150, 160, 200, 300])
    df.loc[standard, "RNG_GRP"] = rng.loc[standard].astype("Int64").astype(str)
    df.loc[known_rng & ~standard, "RNG_GRP"] = "Other"

    combo_known = df["CONSTRUCTION"].notna() & df["RNG_GRP"].notna()
    df["COMBO"] = pd.Series(pd.NA, index=df.index, dtype="string")
    df.loc[combo_known, "COMBO"] = (
        df.loc[combo_known, "CONSTRUCTION"].astype(str)
        + " | RNG "
        + df.loc[combo_known, "RNG_GRP"].astype(str)
    )

    df["NOK"] = ((df["Right"] < sentinel) & (df["Right"] > df["Left"])).astype(int)
    df["RIGHT_CENSORED"] = df["Right"] >= sentinel
    return df


def support_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    def tab(col):
        # Missing covariates are not a model level. They remain useful to broader known terms.
        x = df[df[col].notna()].copy()
        return x.groupby(col, dropna=True).agg(n=("NOK", "size"), NOK=("NOK", "sum")).reset_index()
    return {"combo": tab("COMBO"), "prod": tab("PROD"), "type": tab("PROD_TYP")}


def model_coverage_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Counts showing how many records can inform each hierarchical block."""
    specs = [
        ("Population / intercept", []),
        ("CAT", ["CAT"]),
        ("CLMARK × RNG", ["CLMARK", "RNG"]),
        ("PROD", ["PROD"]),
        ("PROD + TYP", ["PROD", "TYP"]),
        ("Complete profile", ["PROD", "TYP", "CAT", "CLMARK", "RNG"]),
    ]
    rows=[]
    for label, cols in specs:
        mask = pd.Series(True, index=df.index) if not cols else df[cols].notna().all(axis=1)
        d=df.loc[mask]
        rows.append({"Vrstva":label,"n":int(len(d)),"NOK":int(d["NOK"].sum())})
    return pd.DataFrame(rows)


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
