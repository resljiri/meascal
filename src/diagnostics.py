from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy import sparse

from .likelihood import interval_weibull_loglik


def profile_keys(profile: dict):
    prod = str(profile.get("PROD", ""))
    typ = str(profile.get("TYP", "")).replace(".0", "")
    cat = str(profile.get("CAT", "D"))
    clmark = str(profile.get("CLMARK", ""))
    try:
        rv = float(profile.get("RNG"))
        rng_str = str(int(rv)) if rv.is_integer() else str(rv)
    except Exception:
        rng_str = str(profile.get("RNG", ""))
    construction = "digital" if clmark in {"112.12", "112.12.0"} or profile.get("CONSTRUCTION") == "digital" else "mechanical"
    try:
        rv = float(profile.get("RNG"))
        rg = str(int(rv)) if rv in [150, 160, 200, 300] else "Other"
    except Exception:
        rg = "Other"
    return {
        "PROD": prod,
        "TYP_STR": typ,
        "CAT": cat,
        "CLMARK": clmark,
        "RNG_STR": rng_str,
        "COMBO": f"{construction} | RNG {rg}",
        "PROD_TYP": f"{prod}||{typ}",
    }


def _specified(value):
    return value is not None and str(value).strip() not in {"", "__ALL__", "ALL"}

def current_profile_subset(df: pd.DataFrame, profile: dict) -> pd.DataFrame:
    """Filter only by factors explicitly specified; __ALL__ factors are ignored."""
    out=df.copy()
    if _specified(profile.get("PROD")):
        out=out[out["PROD"].astype(str)==str(profile["PROD"])]
    if _specified(profile.get("TYP")):
        out=out[out["TYP_STR"].astype(str)==str(profile["TYP"]).replace(".0","")]
    if _specified(profile.get("CAT")):
        out=out[out["CAT"].astype(str)==str(profile["CAT"])]
    if _specified(profile.get("CLMARK")):
        out=out[out["CLMARK"].astype(str)==str(profile["CLMARK"])]
    if _specified(profile.get("RNG")):
        try:
            out=out[pd.to_numeric(out["RNG"],errors="coerce")==float(profile["RNG"])]
        except Exception:
            out=out[out["RNG"].astype(str)==str(profile["RNG"])]
    return out.copy()

def profile_data_subset(df: pd.DataFrame, profile: dict, level: str) -> pd.DataFrame:
    """Observed records for diagnostics. CURRENT follows all currently specified factors."""
    if level == "CURRENT":
        return current_profile_subset(df,profile)
    k = profile_keys(profile)
    if level == "ALL": return df.copy()
    if level == "PROD":
        if not _specified(profile.get("PROD")): return df.copy()
        return df[df["PROD"].astype(str) == k["PROD"]].copy()
    if level == "TYPE":
        if not (_specified(profile.get("PROD")) and _specified(profile.get("TYP"))): return current_profile_subset(df,profile)
        return df[df["PROD_TYP"].astype(str) == k["PROD_TYP"]].copy()
    if level == "COMBO":
        if not (_specified(profile.get("CLMARK")) and _specified(profile.get("RNG"))): return current_profile_subset(df,profile)
        return df[df["COMBO"].astype(str) == k["COMBO"]].copy()
    if level == "EXACT":
        return current_profile_subset(df,profile)
    raise ValueError(f"Neznámá datová úroveň: {level}")


def batch_predict(model, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        p = {"PROD": r["PROD"], "TYP": r["TYP_STR"], "CAT": r["CAT"], "CLMARK": r["CLMARK"], "RNG": r["RNG"]}
        q = model.predict(p, reliability_ci=False)
        rows.append({
            "_row_id": r["_row_id"], "eta": q.eta, "k": q.k,
            "t95": q.t95, "t90": q.t90, "t85": q.t85,
            "eta_level": q.backoff["ETA"], "k_level": q.backoff["K"],
            "warning": " | ".join(q.warnings),
        })
    return pd.DataFrame(rows)


def support_summary(pred: pd.DataFrame) -> pd.DataFrame:
    return pred.groupby(["eta_level", "k_level"]).size().rename("count").reset_index()


def _numeric_hessian_2d(fun, x, eps=2e-4):
    x=np.asarray(x,float); H=np.zeros((2,2)); f0=fun(x)
    for i in range(2):
        ei=np.zeros(2); ei[i]=eps
        H[i,i]=(fun(x+ei)-2*f0+fun(x-ei))/(eps**2)
    e0=np.array([eps,0.0]); e1=np.array([0.0,eps])
    H[0,1]=H[1,0]=(fun(x+e0+e1)-fun(x+e0-e1)-fun(x-e0+e1)+fun(x-e0-e1))/(4*eps**2)
    return H

def fit_direct_weibull(df: pd.DataFrame, sentinel: float = 99999.0, reliability: float = 0.90, compute_ci: bool = True, n_samples: int = 5000):
    """Direct interval-censored Weibull fit on the currently displayed observed subset.

    Returns point estimates and, when numerically available, approximate 95% CI from
    the inverse Hessian on (log eta, log k), propagated by Monte Carlo.
    """
    if len(df) < 2:
        return None
    reliability=float(reliability)
    if not 0 < reliability < 1:
        raise ValueError("reliability must be between 0 and 1")
    L = df["Left"].to_numpy(float)
    R = df["Right"].to_numpy(float)
    finite = np.r_[L[L > 0], R[R < sentinel]]
    start_eta = max(float(np.median(finite)) if len(finite) else 1000.0, 100.0)

    def nll(theta):
        eta = np.full(len(df), np.exp(theta[0]))
        k = np.full(len(df), np.exp(theta[1]))
        return -float(interval_weibull_loglik(L, R, eta, k, sentinel).sum())

    res = minimize(
        nll, [np.log(start_eta), np.log(0.9)], method="L-BFGS-B",
        bounds=[(np.log(10), np.log(200000)), (np.log(0.15), np.log(5.0))],
    )
    if not res.success:
        return None
    eta, k = float(np.exp(res.x[0])), float(np.exp(res.x[1]))
    tq=lambda rel: eta*(-np.log(rel))**(1/k)
    out={"eta": eta, "k": k, "n": len(df), "NOK": int(df["NOK"].sum()), "loglik": -float(res.fun),
         "reliability": reliability, "t_target": tq(reliability), "t90": tq(.90)}
    if compute_ci:
        try:
            H=_numeric_hessian_2d(nll,np.asarray(res.x,float))
            cov=np.linalg.pinv(H)
            if np.all(np.isfinite(cov)) and np.all(np.diag(cov)>=0):
                rng=np.random.default_rng(271828)
                sims=rng.multivariate_normal(np.asarray(res.x,float),cov,size=n_samples,check_valid="ignore")
                eta_s=np.exp(sims[:,0]); k_s=np.exp(sims[:,1])
                t_s=eta_s*(-np.log(reliability))**(1/k_s)
                out["eta_ci95"]=tuple(np.quantile(eta_s,[.025,.975]))
                out["k_ci95"]=tuple(np.quantile(k_s,[.025,.975]))
                out["t_target_ci95"]=tuple(np.quantile(t_s,[.025,.975]))
                out["covariance"]=cov
        except Exception:
            pass
    return out

def turnbull_curve(df: pd.DataFrame, times: np.ndarray, sentinel: float = 99999.0):
    """Simple Turnbull NPMLE for interval/right-censored observations.

    Intended for diagnostics/visualisation of the currently selected data layer.
    """
    if len(df) == 0:
        return None
    L = df["Left"].to_numpy(float)
    R0 = df["Right"].to_numpy(float)
    R = np.where(R0 >= sentinel, np.inf, R0)
    times = np.asarray(times, float)

    finite_R = R[np.isfinite(R)]
    support = np.unique(np.r_[L, finite_R])
    support = support[np.isfinite(support) & (support >= 0)]
    if len(support) == 0:
        return None
    tail = max(float(support.max()) * 1.25, float(times.max()) * 1.05, 10.0)
    support = np.unique(np.r_[support, tail])

    rows, cols = [], []
    for i, (l, r) in enumerate(zip(L, R)):
        if np.isinf(r):
            idx = np.where(support > l)[0]
        else:
            idx = np.where((support > l) & (support <= r))[0]
        if len(idx) == 0:
            idx = np.array([len(support) - 1]) if np.isinf(r) else np.array([np.argmin(np.abs(support - r))])
        rows.extend([i] * len(idx)); cols.extend(idx.tolist())

    A = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(L), len(support)))
    p = np.ones(len(support)) / len(support)
    for _ in range(4000):
        den = np.maximum(A.dot(p), 1e-300)
        pn = p * (A.T.dot(1.0 / den)) / len(L)
        pn /= pn.sum()
        if np.max(np.abs(pn - p)) < 1e-10:
            p = pn
            break
        p = pn
    S = np.array([p[support > t].sum() for t in times])
    return S


def batch_predict_fast(model, df: pd.DataFrame, reliability: float = 0.90) -> pd.DataFrame:
    """Fast batch prediction by evaluating each unique model profile only once.

    Many calibration-history rows share the same PROD/TYP/CAT/CLMARK/RNG profile.
    Reusing one prediction per unique profile avoids thousands of repeated Python calls
    while preserving exactly the same point-estimate logic as model.predict().
    """
    if len(df) == 0:
        return pd.DataFrame(columns=["_row_id","eta","k","t_target","eta_level","k_level","warning"])
    work=df[["_row_id","PROD","TYP_STR","CAT","CLMARK","RNG"]].copy()
    key_cols=["PROD","TYP_STR","CAT","CLMARK","RNG"]
    unique=work[key_cols].drop_duplicates().reset_index(drop=True)
    preds=[]
    for _, r in unique.iterrows():
        p={"PROD":r["PROD"],"TYP":r["TYP_STR"],"CAT":r["CAT"],"CLMARK":r["CLMARK"],"RNG":r["RNG"]}
        q=model.predict(p,reliability_ci=False,reliability=float(reliability))
        preds.append({**{c:r[c] for c in key_cols},
                      "eta":q.eta,"k":q.k,"t_target":q.t_target,
                      "eta_level":q.backoff["ETA"],"k_level":q.backoff["K"],
                      "warning":" | ".join(q.warnings)})
    lookup=pd.DataFrame(preds)
    return work.merge(lookup,on=key_cols,how="left")[["_row_id","eta","k","t_target","eta_level","k_level","warning"]]


def direct_survival_ci(direct: dict | None, times: np.ndarray, n_samples: int = 1000):
    """Pointwise 95% confidence band for a direct Weibull fit."""
    if not direct or direct.get("covariance") is None:
        return None
    cov=np.asarray(direct["covariance"],float)
    if cov.shape != (2,2) or not np.all(np.isfinite(cov)):
        return None
    rng=np.random.default_rng(314159)
    mu=np.array([np.log(float(direct["eta"])),np.log(float(direct["k"]))])
    sims=rng.multivariate_normal(mu,cov,size=max(int(n_samples),50),check_valid="ignore")
    eta=np.exp(sims[:,0])[:,None]; k=np.exp(sims[:,1])[:,None]
    t=np.asarray(times,float)[None,:]
    surv=np.exp(-np.power(np.maximum(t,0.0)/eta,k))
    lo,hi=np.quantile(surv,[.025,.975],axis=0)
    return lo,hi
