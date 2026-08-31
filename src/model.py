from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from .likelihood import interval_weibull_loglik
from .data import support_tables

@dataclass
class PredictionResult:
    eta: float
    k: float
    t90: float
    t95: float
    t85: float
    reliability: float
    t_target: float
    support: dict
    active_effects: dict
    backoff: dict
    warnings: list[str]
    contributions_eta: dict | None = None
    contributions_k: dict | None = None
    eta_ci95: tuple[float,float] | None = None
    k_ci95: tuple[float,float] | None = None
    t90_ci95: tuple[float,float] | None = None
    t_target_ci95: tuple[float,float] | None = None

class AdaptiveWeibullModel:
    ALL_TOKEN = "__ALL__"

    def __init__(self, config: dict):
        self.config = config
        self.fitted = False
        self.covariance = None
        self.parameter_names = []
        self.theta = None
        self._population_weights = None

    @classmethod
    def _is_all(cls, value):
        try:
            if pd.isna(value):
                return True
        except Exception:
            pass
        return value is None or str(value).strip() in {"", cls.ALL_TOKEN, "ALL", "VŠECHNY", "Vsechny"}

    def set_population_context(self, df: pd.DataFrame):
        """Uloží pouze četnosti potřebné pro pooled predikce; neukládá celou databázi."""
        w = {}
        for col in ["CAT", "COMBO", "PROD", "PROD_TYP", "CLMARK", "RNG_GRP"]:
            if col in df.columns:
                w[col] = df.loc[df[col].notna(), col].astype(str).value_counts().to_dict()
        if "PROD" in df.columns and "PROD_TYP" in df.columns:
            w["TYPE_BY_PROD"] = {
                str(prod): g.loc[g["PROD_TYP"].notna(), "PROD_TYP"].astype(str).value_counts().to_dict()
                for prod, g in df[df["PROD"].notna()].groupby("PROD")
            }
        if "CLMARK" in df.columns and "COMBO" in df.columns:
            w["COMBO_BY_CLMARK"] = {
                str(v): g["COMBO"].astype(str).value_counts().to_dict()
                for v, g in df.groupby(df["CLMARK"].astype(str))
            }
        if "RNG_GRP" in df.columns and "COMBO" in df.columns:
            w["COMBO_BY_RNG"] = {
                str(v): g["COMBO"].astype(str).value_counts().to_dict()
                for v, g in df.groupby(df["RNG_GRP"].astype(str))
            }
        self._population_weights = w
        return self

    @staticmethod
    def _weighted_mean_effect(index_df, weights, value_col, active_col=None):
        if index_df is None or weights is None or not len(weights):
            return 0.0
        num = den = 0.0
        for key, count in weights.items():
            if key not in index_df.index:
                continue
            row = index_df.loc[key]
            val = float(row[value_col])
            if active_col is not None and not AdaptiveWeibullModel._bool(row[active_col]):
                val = 0.0
            num += float(count) * val
            den += float(count)
        return num / den if den else 0.0

    @staticmethod
    def _bool(x):
        if isinstance(x, str): return x.lower() in {"true","1","yes"}
        return bool(x)

    def load_imported(self, model_dir: str | Path):
        p = Path(model_dir)
        self.cat_eta = pd.read_csv(p/"selected_CAT_eta_effects.csv")
        self.cat_k = pd.read_csv(p/"selected_CAT_shape_effects.csv")
        self.combo = pd.read_csv(p/"selected_combo_effects.csv")
        self.prod = pd.read_csv(p/"selected_PROD_effects.csv")
        self.typ = pd.read_csv(p/"selected_TYP_effects.csv")
        self.fitted = True
        self._build_lookup()
        return self

    def _build_lookup(self):
        self.eta_terms = dict(zip(self.cat_eta.term, self.cat_eta.beta_eta))
        self.k_terms = dict(zip(self.cat_k.term, self.cat_k.gamma_logk))
        self.combo_idx = self.combo.set_index("COMBO")
        self.prod_idx = self.prod.set_index("PROD")
        self.typ_idx = self.typ.set_index("PROD_TYP")

    def _profile_keys(self, profile: dict):
        prod = str(profile.get("PROD", ""))
        typ = str(profile.get("TYP", "")).replace(".0", "")
        cat = str(profile.get("CAT", "D"))
        clmark = str(profile.get("CLMARK", ""))
        construction = "digital" if clmark in {"112.12", "112.12.0"} or profile.get("CONSTRUCTION") == "digital" else "mechanical"
        try:
            rv = float(profile.get("RNG"))
            rg = str(int(rv)) if rv in [150,160,200,300] else "Other"
        except Exception:
            rg = "Other"
        return prod, typ, cat, f"{construction} | RNG {rg}", f"{prod}||{typ}"

    def predict(self, profile: dict, reliability_ci: bool=True, n_samples: int=3000, reliability: float=0.90) -> PredictionResult:
        if not self.fitted:
            raise RuntimeError("Model není načten ani fitován.")

        raw_prod = profile.get("PROD")
        raw_typ = profile.get("TYP")
        raw_cat = profile.get("CAT")
        raw_clm = profile.get("CLMARK")
        raw_rng = profile.get("RNG")
        prod_all, typ_all = self._is_all(raw_prod), self._is_all(raw_typ)
        cat_all, clm_all, rng_all = self._is_all(raw_cat), self._is_all(raw_clm), self._is_all(raw_rng)

        # Dummy values are used only to construct lookup keys when a factor is specified.
        key_profile = dict(profile)
        key_profile["PROD"] = "" if prod_all else raw_prod
        key_profile["TYP"] = "" if typ_all else raw_typ
        key_profile["CAT"] = self.config.get("reference_cat", "D") if cat_all else raw_cat
        key_profile["CLMARK"] = "112.11" if clm_all else raw_clm
        key_profile["RNG"] = 160 if rng_all else raw_rng
        prod, typ, cat, combo_key, typ_key = self._profile_keys(key_profile)

        reliability=float(reliability)
        if not (0.0 < reliability < 1.0):
            raise ValueError("Cílová spolehlivost musí být mezi 0 a 1.")
        warnings=[]; active={}; backoff={}; support={}
        lp_eta=float(self.eta_terms["intercept"]); lp_k=float(self.k_terms["intercept"])
        contrib_eta={"Intercept": float(self.eta_terms["intercept"])}
        contrib_k={"Intercept": float(self.k_terms["intercept"])}
        W = self._population_weights or {}

        # CAT: při "Všechny" používáme empiricky vážený průměr CAT příspěvků,
        # nikoli referenční kategorii D.
        if cat_all:
            cw = W.get("CAT", {})
            total = sum(cw.values())
            ce = ck = 0.0
            if total:
                for c, n in cw.items():
                    ce += n * float(self.eta_terms.get(f"CAT_{c}", 0.0))
                    ck += n * float(self.k_terms.get(f"CAT_{c}", 0.0))
                ce /= total; ck /= total
            lp_eta += ce; lp_k += ck
            contrib_eta["CAT (pooled)"]=ce; contrib_k["CAT (pooled)"]=ck
            active["CAT"] = "POOLED"
            support["CAT"] = {"n": int(total), "NOK": None}
        elif cat != self.config.get("reference_cat","D"):
            et=f"CAT_{cat}"
            if et in self.eta_terms:
                ce=float(self.eta_terms[et]); ck=float(self.k_terms[et])
                lp_eta += ce; lp_k += ck; contrib_eta[f"CAT {cat}"]=ce; contrib_k[f"CAT {cat}"]=ck; active["CAT"] = True
            else:
                active["CAT"] = False; warnings.append(f"CAT {cat} není v modelu; použit referenční/poolovaný základ.")
        else:
            active["CAT"] = True

        # CLMARK × RNG combined effect. If one/both factors are pooled, use a weighted pooled contribution.
        if clm_all or rng_all:
            if not clm_all and rng_all:
                weights = W.get("COMBO_BY_CLMARK", {}).get(str(raw_clm), {})
            elif clm_all and not rng_all:
                try:
                    rv=float(raw_rng); rg=str(int(rv)) if rv in [150,160,200,300] else "Other"
                except Exception:
                    rg="Other"
                weights = W.get("COMBO_BY_RNG", {}).get(rg, {})
            else:
                weights = W.get("COMBO", {})
            ce=self._weighted_mean_effect(self.combo_idx, weights, "a_eta", "eta_active")
            ck=self._weighted_mean_effect(self.combo_idx, weights, "a_k", "k_active")
            lp_eta += ce; lp_k += ck
            contrib_eta["CLMARK×RNG (pooled)"]=ce; contrib_k["CLMARK×RNG (pooled)"]=ck
            support["COMBO"]={"n":int(sum(weights.values())),"NOK":None}
            active["COMBO_ETA"]="POOLED"; active["COMBO_K"]="POOLED"
        elif combo_key in self.combo_idx.index:
            r=self.combo_idx.loc[combo_key]
            support["COMBO"]={"n":int(r.n),"NOK":int(r.NOK)}
            ae=self._bool(r.eta_active); ak=self._bool(r.k_active)
            if ae:
                ce=float(r.a_eta); lp_eta += ce; contrib_eta[f"CLMARK×RNG: {combo_key}"]=ce
            if ak:
                ck=float(r.a_k); lp_k += ck; contrib_k[f"CLMARK×RNG: {combo_key}"]=ck
            active["COMBO_ETA"]=ae; active["COMBO_K"]=ak
            if not (ae and ak): warnings.append("Kombinace konstrukce a rozsahu má omezenou datovou podporu; část efektů byla potlačena.")
        else:
            support["COMBO"]={"n":0,"NOK":0}; active["COMBO_ETA"]=active["COMBO_K"]=False
            warnings.append("Kombinace konstrukce a rozsahu není v trénovacích datech.")

        # Manufacturer
        if prod_all:
            weights=W.get("PROD", {})
            ce=self._weighted_mean_effect(self.prod_idx,weights,"u_eta","eta_active")
            ck=self._weighted_mean_effect(self.prod_idx,weights,"u_k","k_active")
            lp_eta += ce; lp_k += ck; contrib_eta["PROD (pooled)"]=ce; contrib_k["PROD (pooled)"]=ck
            support["PROD"]={"n":int(sum(weights.values())),"NOK":None}
            active["PROD_ETA"]="POOLED"; active["PROD_K"]="POOLED"
        elif prod in self.prod_idx.index:
            r=self.prod_idx.loc[prod]
            support["PROD"]={"n":int(r.n),"NOK":int(r.NOK)}
            ae=self._bool(r.eta_active); ak=self._bool(r.k_active)
            if ae:
                ce=float(r.u_eta); lp_eta += ce; contrib_eta[f"PROD: {prod}"]=ce
            if ak:
                ck=float(r.u_k); lp_k += ck; contrib_k[f"PROD: {prod}"]=ck
            active["PROD_ETA"]=ae; active["PROD_K"]=ak
            if not (ae and ak): warnings.append(f"Výrobce {prod} nemá dost podpory pro všechny parametry; model částečně back-offuje.")
        else:
            support["PROD"]={"n":0,"NOK":0}; active["PROD_ETA"]=active["PROD_K"]=False
            warnings.append(f"Výrobce {prod} není v trénovacích datech; efekt výrobce není použit.")

        # Nested type. A specific type is meaningful only with a specific producer.
        if typ_all or prod_all:
            if not prod_all:
                weights=W.get("TYPE_BY_PROD",{}).get(str(prod),{})
            else:
                weights=W.get("PROD_TYP",{})
            ce=self._weighted_mean_effect(self.typ_idx,weights,"v_eta","eta_active")
            ck=self._weighted_mean_effect(self.typ_idx,weights,"v_k","k_active")
            lp_eta += ce; lp_k += ck; contrib_eta["TYP (pooled)"]=ce; contrib_k["TYP (pooled)"]=ck
            support["TYP"]={"n":int(sum(weights.values())),"NOK":None}
            active["TYP_ETA"]="POOLED"; active["TYP_K"]="POOLED"
        elif typ_key in self.typ_idx.index:
            r=self.typ_idx.loc[typ_key]
            support["TYP"]={"n":int(r.n),"NOK":int(r.NOK)}
            ae=self._bool(r.eta_active); ak=self._bool(r.k_active)
            if ae:
                ce=float(r.v_eta); lp_eta += ce; contrib_eta[f"TYP: {typ}"]=ce
            if ak:
                ck=float(r.v_k); lp_k += ck; contrib_k[f"TYP: {typ}"]=ck
            active["TYP_ETA"]=ae; active["TYP_K"]=ak
            if not (ae and ak): warnings.append(f"Typ {typ} nemá dost podpory pro všechny parametry; použit širší kontext.")
        else:
            support["TYP"]={"n":0,"NOK":0}; active["TYP_ETA"]=active["TYP_K"]=False
            warnings.append(f"Typ {typ} není v trénovacích datech; predikce nepoužívá specifický efekt typu.")

        eta=float(np.exp(lp_eta)); k=float(np.exp(lp_k))
        tq=lambda R: eta*(-np.log(R))**(1/k)
        t90=tq(.90); t95=tq(.95); t85=tq(.85); t_target=tq(reliability)

        def level(type_key, prod_key, combo_key_name):
            if active.get(type_key) is True: return "TYP"
            if active.get(prod_key) is True: return "PROD"
            if active.get(combo_key_name) is True: return "COMBO"
            if "POOLED" in {active.get(type_key),active.get(prod_key),active.get(combo_key_name),active.get("CAT")}: return "POOLED"
            return "CAT/POPULATION"
        backoff["ETA"] = level("TYP_ETA","PROD_ETA","COMBO_ETA")
        backoff["K"] = level("TYP_K","PROD_K","COMBO_K")

        eta_ci=k_ci=t90_ci=t_target_ci=None
        if reliability_ci and self.covariance is not None and self.theta is not None:
            sims=self._simulate_prediction(profile,n_samples,reliability=reliability)
            if len(sims):
                eta_ci=tuple(np.quantile(sims[:,0],[.025,.975])); k_ci=tuple(np.quantile(sims[:,1],[.025,.975])); t90_ci=tuple(np.quantile(sims[:,2],[.025,.975])); t_target_ci=tuple(np.quantile(sims[:,3],[.025,.975]))
        elif reliability_ci:
            warnings.append("95% CI není u importovaného modelu dostupný: chybí kovarianční matice. Aktivujte přeučený kandidátní model s kovarianční maticí.")

        return PredictionResult(eta,k,t90,t95,t85,reliability,t_target,support,active,backoff,warnings,contrib_eta,contrib_k,eta_ci,k_ci,t90_ci,t_target_ci)

    def survival(self, profile: dict, times: np.ndarray):
        r=self.predict(profile,reliability_ci=False)
        times=np.asarray(times,float)
        return np.exp(-(times/r.eta)**r.k)

    def survival_ci(self, profile: dict, times: np.ndarray, n_samples: int=1000, reliability: float=0.90):
        """Approximate pointwise 95% confidence band for R(t) from parameter covariance."""
        if self.covariance is None or self.theta is None:
            return None
        sims=self._simulate_prediction(profile,max(int(n_samples),50),reliability=float(reliability))
        if len(sims)==0:
            return None
        times=np.asarray(times,float)
        eta=sims[:,0][:,None]; k=sims[:,1][:,None]
        surv=np.exp(-np.power(np.maximum(times[None,:],0.0)/eta,k))
        lo,hi=np.quantile(surv,[.025,.975],axis=0)
        return lo,hi

    # ---------------- training ----------------
    def fit(self, df: pd.DataFrame, compute_covariance: bool=True):
        cfg=self.config
        th=cfg["thresholds"]; pen=cfg["penalties"]
        sup=support_tables(df)
        combo_sup=sup["combo"].set_index("COMBO"); prod_sup=sup["prod"].set_index("PROD"); type_sup=sup["type"].set_index("PROD_TYP")

        def active(index, key, spec):
            if key not in index.index: return False
            r=index.loc[key]
            return int(r.n)>=spec["min_n"] and int(r.NOK)>=spec["min_nok"]

        cats=sorted([str(x) for x in df.loc[df.CAT.notna(),"CAT"].unique() if str(x) != cfg.get("reference_cat","D")])
        combos=sorted(df.loc[df.COMBO.notna(),"COMBO"].astype(str).unique())
        prods=sorted(df.loc[df.PROD.notna(),"PROD"].astype(str).unique())
        types=sorted(df.loc[df.PROD_TYP.notna(),"PROD_TYP"].astype(str).unique())

        eta_combo=[x for x in combos if active(combo_sup,x,th["eta_combo"])]
        k_combo=[x for x in combos if active(combo_sup,x,th["k_combo"])]
        eta_prod=[x for x in prods if active(prod_sup,x,th["eta_prod"])]
        k_prod=[x for x in prods if active(prod_sup,x,th["k_prod"])]
        eta_type=[x for x in types if active(type_sup,x,th["eta_type"])]
        k_type=[x for x in types if active(type_sup,x,th["k_type"])]

        names=["eta_intercept"]+[f"eta_CAT:{x}" for x in cats]+[f"k_intercept"]+[f"k_CAT:{x}" for x in cats]
        names += [f"eta_combo:{x}" for x in eta_combo]+[f"k_combo:{x}" for x in k_combo]
        names += [f"eta_prod:{x}" for x in eta_prod]+[f"k_prod:{x}" for x in k_prod]
        names += [f"eta_type:{x}" for x in eta_type]+[f"k_type:{x}" for x in k_type]
        pos={n:i for i,n in enumerate(names)}

        L=df.Left.to_numpy(float); R=df.Right.to_numpy(float); n=len(df)
        def build_lp(theta):
            le=np.full(n,theta[pos["eta_intercept"]]); lk=np.full(n,theta[pos["k_intercept"]])
            # Missing covariates simply do not receive that specific effect. The row still
            # contributes to the intercept and every broader/sibling factor that is known.
            catv=df.CAT.astype("string").fillna("__MISSING__").astype(str).to_numpy()
            combv=df.COMBO.astype("string").fillna("__MISSING__").astype(str).to_numpy()
            pv=df.PROD.astype("string").fillna("__MISSING__").astype(str).to_numpy()
            tv=df.PROD_TYP.astype("string").fillna("__MISSING__").astype(str).to_numpy()
            for x in cats:
                m=catv==x; le[m]+=theta[pos[f"eta_CAT:{x}"]]; lk[m]+=theta[pos[f"k_CAT:{x}"]]
            for x in eta_combo: le[combv==x]+=theta[pos[f"eta_combo:{x}"]]
            for x in k_combo: lk[combv==x]+=theta[pos[f"k_combo:{x}"]]
            for x in eta_prod: le[pv==x]+=theta[pos[f"eta_prod:{x}"]]
            for x in k_prod: lk[pv==x]+=theta[pos[f"k_prod:{x}"]]
            for x in eta_type: le[tv==x]+=theta[pos[f"eta_type:{x}"]]
            for x in k_type: lk[tv==x]+=theta[pos[f"k_type:{x}"]]
            return le,lk

        def objective(theta):
            le,lk=build_lp(theta); eta=np.exp(le); k=np.exp(np.clip(lk,np.log(cfg["k_bounds"][0]),np.log(cfg["k_bounds"][1])))
            val=-interval_weibull_loglik(L,R,eta,k,cfg.get("right_censor_sentinel",99999)).sum()
            for x in eta_combo: val += .5*(theta[pos[f"eta_combo:{x}"]]/pen["eta_combo"])**2
            for x in k_combo: val += .5*(theta[pos[f"k_combo:{x}"]]/pen["k_combo"])**2
            for x in eta_prod: val += .5*(theta[pos[f"eta_prod:{x}"]]/pen["eta_prod"])**2
            for x in k_prod: val += .5*(theta[pos[f"k_prod:{x}"]]/pen["k_prod"])**2
            for x in eta_type: val += .5*(theta[pos[f"eta_type:{x}"]]/pen["eta_type"])**2
            for x in k_type: val += .5*(theta[pos[f"k_type:{x}"]]/pen["k_type"])**2
            return float(val)

        # reasonable start
        finite=np.r_[df.Left.to_numpy(float),df.loc[df.Right<cfg.get("right_censor_sentinel",99999),"Right"].to_numpy(float)]
        med=max(np.median(finite[finite>0]),100)
        x0=np.zeros(len(names)); x0[pos["eta_intercept"]]=np.log(med); x0[pos["k_intercept"]]=np.log(.9)
        res=minimize(objective,x0,method="L-BFGS-B",options={"maxiter":1200,"ftol":1e-10})
        self.theta=np.asarray(res.x,float); self.parameter_names=names; self._training_pos=pos; self._training_sets=(cats,eta_combo,k_combo,eta_prod,k_prod,eta_type,k_type)
        self.fit_result={"success":bool(res.success),"message":str(res.message),"objective":float(res.fun),"n":len(df),"NOK":int(df.NOK.sum()),"n_params":len(names)}
        self.covariance=None
        if compute_covariance:
            self.covariance=self._numerical_covariance(objective,self.theta)
        self._export_from_theta(df,combo_sup,prod_sup,type_sup)
        self.fitted=True; self._build_lookup(); self.set_population_context(df)
        self.fit_result["covariance_available"] = self.covariance is not None
        return self

    def _numerical_covariance(self, fun, x):
        # full central-difference Hessian; fine for current balanced caliper model (~tens of active parameters)
        x=np.asarray(x,float); m=len(x); H=np.zeros((m,m)); f0=fun(x); eps=2e-4
        for i in range(m):
            ei=np.zeros(m); ei[i]=eps
            H[i,i]=(fun(x+ei)-2*f0+fun(x-ei))/(eps**2)
            for j in range(i+1,m):
                ej=np.zeros(m); ej[j]=eps
                H[i,j]=H[j,i]=(fun(x+ei+ej)-fun(x+ei-ej)-fun(x-ei+ej)+fun(x-ei-ej))/(4*eps**2)
        return np.linalg.pinv(H)

    def _export_from_theta(self,df,combo_sup,prod_sup,type_sup):
        pos=self._training_pos; cats,eta_combo,k_combo,eta_prod,k_prod,eta_type,k_type=self._training_sets; t=self.theta
        self.cat_eta=pd.DataFrame([{"term":"intercept","beta_eta":t[pos["eta_intercept"]]}]+[{"term":f"CAT_{x}","beta_eta":t[pos[f"eta_CAT:{x}"]]} for x in cats])
        self.cat_eta["TR_eta"]=np.exp(self.cat_eta.beta_eta)
        self.cat_k=pd.DataFrame([{"term":"intercept","gamma_logk":t[pos["k_intercept"]]}]+[{"term":f"CAT_{x}","gamma_logk":t[pos[f"k_CAT:{x}"]]} for x in cats])
        self.cat_k["shape_ratio"]=np.exp(self.cat_k.gamma_logk)
        rows=[]
        for x in sorted(df.loc[df.COMBO.notna(),"COMBO"].astype(str).unique()):
            r=combo_sup.loc[x]; ae=x in eta_combo; ak=x in k_combo; av=t[pos[f"eta_combo:{x}"]] if ae else 0; kv=t[pos[f"k_combo:{x}"]] if ak else 0
            rows.append({"COMBO":x,"n":int(r.n),"NOK":int(r.NOK),"a_eta":av,"eta_multiplier":np.exp(av),"a_k":kv,"k_multiplier":np.exp(kv),"eta_active":ae,"k_active":ak})
        self.combo=pd.DataFrame(rows)
        rows=[]
        for x in sorted(df.loc[df.PROD.notna(),"PROD"].astype(str).unique()):
            r=prod_sup.loc[x]; ae=x in eta_prod; ak=x in k_prod; av=t[pos[f"eta_prod:{x}"]] if ae else 0; kv=t[pos[f"k_prod:{x}"]] if ak else 0
            rows.append({"PROD":x,"n":int(r.n),"NOK":int(r.NOK),"u_eta":av,"eta_multiplier":np.exp(av),"u_k":kv,"k_multiplier":np.exp(kv),"eta_active":ae,"k_active":ak})
        self.prod=pd.DataFrame(rows)
        rows=[]
        for x in sorted(df.loc[df.PROD_TYP.notna(),"PROD_TYP"].astype(str).unique()):
            r=type_sup.loc[x]; ae=x in eta_type; ak=x in k_type; av=t[pos[f"eta_type:{x}"]] if ae else 0; kv=t[pos[f"k_type:{x}"]] if ak else 0
            rows.append({"PROD_TYP":x,"n":int(r.n),"NOK":int(r.NOK),"v_eta":av,"eta_multiplier":np.exp(av),"v_k":kv,"k_multiplier":np.exp(kv),"eta_active":ae,"k_active":ak})
        self.typ=pd.DataFrame(rows)

    def _simulate_prediction(self, profile, n_samples, reliability=0.90):
        if self.covariance is None: return np.empty((0,4))
        rng=np.random.default_rng(12345)
        sims=rng.multivariate_normal(self.theta,self.covariance,size=n_samples,check_valid="ignore")
        old=self.theta.copy(); out=[]
        for th in sims:
            self.theta=th; self._export_from_theta_sim(); r=self.predict(profile,reliability_ci=False,reliability=reliability); out.append([r.eta,r.k,r.t90,r.t_target])
        self.theta=old; self._export_from_theta_sim()
        return np.asarray(out)

    def _export_from_theta_sim(self):
        # fast reconstruction of lookup tables using current theta and existing support frames
        pos=self._training_pos; cats,eta_combo,k_combo,eta_prod,k_prod,eta_type,k_type=self._training_sets; t=self.theta
        self.eta_terms={"intercept":t[pos["eta_intercept"]],**{f"CAT_{x}":t[pos[f"eta_CAT:{x}"]] for x in cats}}
        self.k_terms={"intercept":t[pos["k_intercept"]],**{f"CAT_{x}":t[pos[f"k_CAT:{x}"]] for x in cats}}
        # mutate only coefficient values in current dataframes
        for x in eta_combo:
            self.combo_idx.loc[x,"a_eta"]=t[pos[f"eta_combo:{x}"]]
        for x in k_combo: self.combo_idx.loc[x,"a_k"]=t[pos[f"k_combo:{x}"]]
        for x in eta_prod: self.prod_idx.loc[x,"u_eta"]=t[pos[f"eta_prod:{x}"]]
        for x in k_prod: self.prod_idx.loc[x,"u_k"]=t[pos[f"k_prod:{x}"]]
        for x in eta_type: self.typ_idx.loc[x,"v_eta"]=t[pos[f"eta_type:{x}"]]
        for x in k_type: self.typ_idx.loc[x,"v_k"]=t[pos[f"k_type:{x}"]]

    def save(self, path: str|Path):
        import joblib
        Path(path).parent.mkdir(parents=True,exist_ok=True); joblib.dump(self,path)

    @classmethod
    def load(cls,path):
        import joblib
        return joblib.load(path)
