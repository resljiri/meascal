from pathlib import Path
import yaml
from src.data import load_calibration_data
from src.model import AdaptiveWeibullModel

ROOT=Path(__file__).resolve().parents[1]
cfg=yaml.safe_load((ROOT/"config/app.yaml").read_text())
fam=cfg["families"]["CALIPER"]
df=load_calibration_data(ROOT/fam["data_file"])
model=AdaptiveWeibullModel(fam["model"]).load_imported(ROOT/fam["model_dir"])
r=model.predict({"PROD":"Mitutoyo","TYP":"500","CAT":"C","CLMARK":"112.12","RNG":200})
assert 12000 < r.eta < 13500
assert 0.95 < r.k < 1.10
assert 1350 < r.t90 < 1550
r2=model.predict({"PROD":"Mitutoyo","TYP":"NEW-TYPE","CAT":"C","CLMARK":"112.12","RNG":200})
assert r2.backoff["ETA"] in {"PROD","COMBO","CAT/POPULATION"}
assert r2.support["TYP"]["n"] == 0
print("OK",r.eta,r.k,r.t90,r2.backoff)
