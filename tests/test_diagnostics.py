from pathlib import Path
import numpy as np
import yaml
from src.data import load_calibration_data
from src.model import AdaptiveWeibullModel
from src.diagnostics import profile_data_subset, fit_direct_weibull, turnbull_curve

ROOT=Path(__file__).resolve().parents[1]
cfg=yaml.safe_load((ROOT/'config/app.yaml').read_text())
fam=cfg['families']['CALIPER']
df=load_calibration_data(ROOT/fam['data_file'])
model=AdaptiveWeibullModel(fam['model']).load_imported(ROOT/fam['model_dir'])
p={'PROD':'Mitutoyo','TYP':'500','CAT':'C','CLMARK':'112.12','RNG':200}
exact=profile_data_subset(df,p,'EXACT')
assert len(exact)==16
assert int(exact.NOK.sum())==3
ptype=profile_data_subset(df,p,'TYPE')
assert len(ptype)>=400
direct=fit_direct_weibull(exact)
assert direct is not None and direct['eta']>0 and direct['k']>0
t=np.linspace(0,3000,80)
tb=turnbull_curve(exact,t)
assert tb is not None and len(tb)==len(t)
assert np.all((tb>=0)&(tb<=1))
print('OK diagnostics',len(exact),len(ptype),direct['eta'],direct['k'])
