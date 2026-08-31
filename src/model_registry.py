from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import shutil
import zipfile

from .model import AdaptiveWeibullModel

ACTIVE_FILE = Path("models") / "active_model.json"
REGISTRY_DIR = Path("models") / "registry"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def active_descriptor(root: str | Path) -> dict:
    root = Path(root)
    path = root / ACTIVE_FILE
    if not path.exists():
        return {
            "family": "CALIPER",
            "kind": "imported_csv",
            "version": "caliper_v1",
            "path": "models/caliper_v1",
            "created_at": None,
            "note": "Fallback imported model",
        }
    return json.loads(path.read_text(encoding="utf-8"))


def registry_token(root: str | Path) -> str:
    """Stable cache token that changes whenever the active descriptor/model changes."""
    root = Path(root)
    desc = active_descriptor(root)
    payload = json.dumps(desc, sort_keys=True, ensure_ascii=False).encode("utf-8")
    p = root / str(desc.get("path", ""))
    if p.is_file():
        stat = p.stat()
        payload += f"|{stat.st_mtime_ns}|{stat.st_size}".encode()
    return hashlib.sha1(payload).hexdigest()[:16]


def load_active_model(root: str | Path, model_config: dict, df=None):
    root = Path(root)
    desc = active_descriptor(root)
    kind = desc.get("kind", "imported_csv")
    path = root / desc["path"]
    if kind == "joblib":
        model = AdaptiveWeibullModel.load(path)
    elif kind == "imported_csv":
        model = AdaptiveWeibullModel(model_config).load_imported(path)
    else:
        raise ValueError(f"Neznámý typ modelového artefaktu: {kind}")
    if df is not None:
        model.set_population_context(df)
    return model, desc


def _metadata_for(model: AdaptiveWeibullModel, version: str, source: str, data_fingerprint: str | None = None) -> dict:
    fit = getattr(model, "fit_result", {}) or {}
    return {
        "family": "CALIPER",
        "version": version,
        "kind": "joblib",
        "created_at": _utc_now(),
        "source": source,
        "data_fingerprint": data_fingerprint,
        "fit": fit,
        "covariance_available": getattr(model, "covariance", None) is not None,
        "n_parameters": len(getattr(model, "parameter_names", []) or []),
    }


def save_versioned_model(
    root: str | Path,
    model: AdaptiveWeibullModel,
    version: str | None = None,
    source: str = "refit",
    data_fingerprint: str | None = None,
    activate: bool = True,
):
    root = Path(root)
    if version is None:
        version = "caliper_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = root / REGISTRY_DIR / version
    folder.mkdir(parents=True, exist_ok=False)
    model_path = folder / "model.joblib"
    model.save(model_path)

    # Human-readable coefficient exports aid auditability and dissertation reproducibility.
    exports = {
        "CAT_eta.csv": getattr(model, "cat_eta", None),
        "CAT_shape.csv": getattr(model, "cat_k", None),
        "combo_effects.csv": getattr(model, "combo", None),
        "PROD_effects.csv": getattr(model, "prod", None),
        "TYP_effects.csv": getattr(model, "typ", None),
    }
    for name, frame in exports.items():
        if frame is not None:
            frame.to_csv(folder / name, index=False)

    meta = _metadata_for(model, version, source, data_fingerprint)
    (folder / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    desc = {
        "family": "CALIPER",
        "kind": "joblib",
        "version": version,
        "path": str(model_path.relative_to(root)).replace("\\", "/"),
        "created_at": meta["created_at"],
        "source": source,
    }
    if activate:
        activate_descriptor(root, desc)
    return desc, folder


def activate_descriptor(root: str | Path, desc: dict):
    root = Path(root)
    path = root / ACTIVE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(desc, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def activate_imported(root: str | Path, version: str = "caliper_v1") -> dict:
    desc = {
        "family": "CALIPER",
        "kind": "imported_csv",
        "version": version,
        "path": f"models/{version}",
        "created_at": _utc_now(),
        "source": "manual activation",
    }
    activate_descriptor(root, desc)
    return desc


def list_versions(root: str | Path) -> list[dict]:
    root = Path(root)
    rows = []
    reg = root / REGISTRY_DIR
    if reg.exists():
        for folder in sorted(reg.iterdir(), reverse=True):
            if not folder.is_dir():
                continue
            meta_path = folder / "metadata.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception:
                    meta = {"version": folder.name}
            else:
                meta = {"version": folder.name}
            model_path = folder / "model.joblib"
            if model_path.exists():
                rows.append({**meta, "path": str(model_path.relative_to(root)).replace("\\", "/")})
    return rows


def activate_version(root: str | Path, version: str) -> dict:
    root = Path(root)
    model_path = root / REGISTRY_DIR / version / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    meta_path = model_path.parent / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    desc = {
        "family": "CALIPER",
        "kind": "joblib",
        "version": version,
        "path": str(model_path.relative_to(root)).replace("\\", "/"),
        "created_at": meta.get("created_at"),
        "source": "registry activation",
    }
    activate_descriptor(root, desc)
    return desc


def make_persistence_bundle(root: str | Path, desc: dict, output: str | Path) -> Path:
    """Bundle only files that must be committed to persist the active model after cloud reboot."""
    root = Path(root)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model_path = root / desc["path"]
    if desc.get("kind") != "joblib" or not model_path.exists():
        raise ValueError("Aktivní model není versioned joblib artefakt.")
    folder = model_path.parent
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
        for p in folder.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(root))
        z.write(root / ACTIVE_FILE, ACTIVE_FILE)
    return output
