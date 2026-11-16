"""Uniform inference wrappers ("predictors") for every trained model.

All predictors expose the same interface so that the benchmark and the
interactive demo treat every model identically:

    pred.predict_proba(texts, batch_size=64) -> np.ndarray [N, n_classes]
    pred.classes                              -> list[str]
    pred.to(device)                           -> pred   (neural models only)
    pred.n_params / pred.size_mb

Supported kinds: 'tiny' (our transformer), 'e5_ft' (fine-tuned e5-small),
'e5_frozen_lr' (frozen e5 embeddings + LogisticRegression) and 'baseline'
(the TF-IDF + LR models of src.baseline).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src import config  # noqa: F401  (pins GPU 0 before torch)
from src.model_utils import E5_MODEL_NAME, E5_PREFIX, dir_size_mb, load_json

KINDS = ("baseline", "tiny", "e5_frozen_lr", "e5_ft")
_SHARED: dict = {}      # objects shared between predictors (frozen e5 encoder)


@dataclass
class ModelSpec:
    """Where a trained model lives and how to load it."""
    kind: str          # one of KINDS
    name: str          # model_name used in the experiments table
    task: str
    path: str          # model directory (or baseline name for kind='baseline')


class BasePredictor:
    """Common helpers: device handling and top-k decoding."""
    classes: list[str]
    device: str = "cpu"
    n_params: int | None = None
    size_mb: float | None = None
    supports_cuda: bool = True

    def predict_proba(self, texts: list[str], batch_size: int = 64) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def to(self, device: str) -> "BasePredictor":
        self.device = device
        return self

    def predict(self, texts: list[str], top_k: int = 3) -> list[dict]:
        """Label + top-k (label, probability) for every text."""
        probs = self.predict_proba(texts)
        out = []
        for row in probs:
            order = np.argsort(row)[::-1][:top_k]
            out.append({"label": self.classes[order[0]],
                        "top": [(self.classes[i], float(row[i])) for i in order]})
        return out


def _torch():
    import torch
    return torch


def _length_order(texts: list[str]) -> tuple[list[str], np.ndarray]:
    """Sort texts by length so each batch has little padding; returns (sorted, inverse order)."""
    order = np.argsort([len(t) for t in texts], kind="stable")
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return [texts[i] for i in order], inverse


def _mean_pool(hidden, mask):
    """Masked mean over the sequence dimension."""
    m = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * m).sum(1) / m.sum(1).clamp_min(1.0)


class TinyTransformerPredictor(BasePredictor):
    """Our from-scratch transformer + its BPE tokenizer."""

    def __init__(self, model_dir: str | Path, device: str = "cpu"):
        torch = _torch()
        from src.bpe_tokenizer import encode_titles, load_tokenizer
        from src.tiny_transformer import TinyTransformerClassifier, count_params

        self.model_dir = Path(model_dir)
        cfg = load_json(self.model_dir / "config.json")
        l2id = load_json(self.model_dir / "label2id.json")
        self.classes = [c for c, _ in sorted(l2id.items(), key=lambda kv: kv[1])]
        self.max_len = int(cfg["max_len"])
        self.tok = load_tokenizer(self.model_dir / "tokenizer.json")
        self._encode = encode_titles
        self.model = TinyTransformerClassifier(**cfg["model"])
        state = torch.load(self.model_dir / "model.pt", map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()
        self.n_params = count_params(self.model)
        self.size_mb = dir_size_mb(self.model_dir / "model.pt") + dir_size_mb(self.model_dir / "tokenizer.json")
        self.to(device)

    def to(self, device: str) -> "TinyTransformerPredictor":
        self.device = device
        self.model.to(device)
        return self

    def predict_proba(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        torch = _torch()
        texts, inverse = _length_order(list(texts))
        out = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                ids, lens = self._encode(self.tok, texts[start:start + batch_size], self.max_len)
                ids = torch.from_numpy(ids[:, :int(lens.max())].astype(np.int64)).to(self.device)
                lens_t = torch.from_numpy(lens.astype(np.int64)).to(self.device)
                mask = torch.arange(ids.size(1), device=self.device)[None, :] < lens_t[:, None]
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device == "cuda"):
                    logits = self.model(ids, mask)
                out.append(torch.softmax(logits.float(), -1).cpu().numpy())
        return np.concatenate(out)[inverse]


class _E5Base(BasePredictor):
    """Shared tokenisation / mean-pooled encoding for the e5 predictors."""
    max_len: int = 128

    def _batches(self, texts: list[str], batch_size: int):
        torch = _torch()
        for start in range(0, len(texts), batch_size):
            batch = [E5_PREFIX + t for t in texts[start:start + batch_size]]
            enc = self.tokenizer(batch, truncation=True, max_length=self.max_len, padding=True,
                                 return_tensors="pt")
            yield enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device).bool()

    def _embed(self, encoder, texts: list[str], batch_size: int, normalize: bool) -> np.ndarray:
        torch = _torch()
        texts, inverse = _length_order(list(texts))
        out = []
        with torch.inference_mode():
            for ids, mask in self._batches(texts, batch_size):
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device == "cuda"):
                    hidden = encoder(input_ids=ids, attention_mask=mask.long()).last_hidden_state
                emb = _mean_pool(hidden.float(), mask)
                if normalize:
                    emb = torch.nn.functional.normalize(emb, dim=-1)
                out.append(emb.cpu().numpy())
        return np.concatenate(out)[inverse]


class E5FinetunedPredictor(_E5Base):
    """Fine-tuned e5-small encoder + linear head (saved by src.finetune_e5)."""

    def __init__(self, model_dir: str | Path, device: str = "cpu"):
        torch = _torch()
        from transformers import AutoModel, AutoTokenizer

        self.model_dir = Path(model_dir)
        cfg = load_json(self.model_dir / "config.json")
        l2id = load_json(self.model_dir / "label2id.json")
        self.classes = [c for c, _ in sorted(l2id.items(), key=lambda kv: kv[1])]
        self.max_len = int(cfg["max_len"])
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir / "encoder")
        self.encoder = AutoModel.from_pretrained(self.model_dir / "encoder").eval()
        self.head = torch.nn.Linear(self.encoder.config.hidden_size, len(self.classes))
        self.head.load_state_dict(torch.load(self.model_dir / "head.pt", map_location="cpu", weights_only=True))
        self.head.eval()
        self.n_params = sum(p.numel() for p in self.encoder.parameters()) + \
            sum(p.numel() for p in self.head.parameters())
        self.size_mb = dir_size_mb(self.model_dir)
        self.to(device)

    def to(self, device: str) -> "E5FinetunedPredictor":
        self.device = device
        self.encoder.to(device)
        self.head.to(device)
        return self

    def predict_proba(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        torch = _torch()
        emb = self._embed(self.encoder, list(texts), batch_size, normalize=False)
        with torch.inference_mode():
            logits = self.head(torch.from_numpy(emb).to(self.device))
            return torch.softmax(logits.float(), -1).cpu().numpy()


class E5FrozenLRPredictor(_E5Base):
    """Frozen e5-small sentence embeddings + scikit-learn LogisticRegression."""

    def __init__(self, model_dir: str | Path, device: str = "cpu"):
        import joblib
        from transformers import AutoModel, AutoTokenizer

        self.model_dir = Path(model_dir)
        cfg = load_json(self.model_dir / "config.json")
        self.max_len = int(cfg["max_len"])
        self.clf = joblib.load(self.model_dir / "lr.joblib")
        self.classes = [str(c) for c in self.clf.classes_]
        # The frozen encoder is identical for every task -> share one copy per device (saves ~470 MB RAM).
        cache_key = ("frozen_e5", device)
        if cache_key not in _SHARED:
            _SHARED[cache_key] = (AutoTokenizer.from_pretrained(E5_MODEL_NAME),
                                  AutoModel.from_pretrained(E5_MODEL_NAME).eval())
        self.tokenizer, self.encoder = _SHARED[cache_key]
        lr = self.clf[-1] if hasattr(self.clf, "steps") else self.clf
        self.n_params = sum(p.numel() for p in self.encoder.parameters()) + int(lr.coef_.size + lr.intercept_.size)
        self.size_mb = _hf_weights_mb(E5_MODEL_NAME) + dir_size_mb(self.model_dir / "lr.joblib")
        self.to(device)

    def to(self, device: str) -> "E5FrozenLRPredictor":
        self.device = device
        self.encoder.to(device)
        return self

    def predict_proba(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        emb = self._embed(self.encoder, list(texts), batch_size, normalize=True)
        return self.clf.predict_proba(emb)


class BaselinePredictor(BasePredictor):
    """TF-IDF + LogisticRegression models trained by src.baseline (CPU only)."""
    supports_cuda = False

    def __init__(self, name: str, task: str, device: str = "cpu"):
        from src import baseline

        _alias_main_symbols(baseline)
        self.name, self.task = name, task
        self._predict = baseline.predict_baseline
        self.model = baseline.load_baseline(name, task)
        _, _, classes = baseline.predict_baseline(self.model, ["тест"])
        self.classes = [str(c) for c in classes]
        self.n_params = _sklearn_params(self.model)
        path = baseline_path(name, task)
        self.size_mb = dir_size_mb(path) if path is not None else None

    def predict_proba(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        parts = []
        for start in range(0, len(texts), batch_size):
            _, probas, _ = self._predict(self.model, list(texts[start:start + batch_size]))
            parts.append(np.asarray(probas))
        return np.concatenate(parts)


def _alias_main_symbols(module) -> None:
    """Models pickled while `module` ran as __main__ (python -m src.baseline) reference
    __main__.<name>; expose the module's own classes/functions there so they unpickle anywhere."""
    import __main__
    for name, obj in vars(module).items():
        if callable(obj) and getattr(obj, "__module__", None) == module.__name__ and not hasattr(__main__, name):
            setattr(__main__, name, obj)


def _sklearn_params(model) -> int | None:
    """coef_ + intercept_ size of the (last) linear estimator inside a model/pipeline/dict."""
    candidates = [model]
    if isinstance(model, dict):
        candidates = list(model.values())
    for obj in candidates:
        est = obj
        if hasattr(est, "steps"):
            est = est.steps[-1][1]
        if hasattr(est, "coef_"):
            return int(np.asarray(est.coef_).size + np.asarray(getattr(est, "intercept_", [])).size)
    return None


def baseline_path(name: str, task: str) -> Path | None:
    """File/dir holding a baseline model (layout decided by src.baseline)."""
    # src.baseline saves to artifacts/models/<name>_<task>/model.joblib
    path = config.MODELS_DIR / f"{name}_{task}" / "model.joblib"
    return path if path.exists() else None


def _hf_weights_mb(model_name: str) -> float:
    """Size of the cached safetensors weights of a HuggingFace model."""
    try:
        from huggingface_hub import hf_hub_download
        return dir_size_mb(Path(hf_hub_download(model_name, "model.safetensors")))
    except Exception:
        return float("nan")


def load_predictor(spec: ModelSpec, device: str = "cpu") -> BasePredictor:
    """Instantiate the right predictor for a ModelSpec."""
    if spec.kind == "tiny":
        return TinyTransformerPredictor(spec.path, device)
    if spec.kind == "e5_ft":
        return E5FinetunedPredictor(spec.path, device)
    if spec.kind == "e5_frozen_lr":
        return E5FrozenLRPredictor(spec.path, device)
    if spec.kind == "baseline":
        return BaselinePredictor(spec.name, spec.task, device)
    raise ValueError(f"unknown model kind {spec.kind!r}")


def discover_models(task: str | None = None, include_quick: bool = False) -> list[ModelSpec]:
    """Find every trained model on disk (own models via config.json, baselines via SQLite)."""
    specs: list[ModelSpec] = []
    kind_by_prefix = {"tiny_transformer_": "tiny", "e5_small_ft_": "e5_ft", "e5_frozen_lr_": "e5_frozen_lr"}
    for cfg_path in sorted(config.MODELS_DIR.glob("*/config.json")):
        d = cfg_path.parent
        kind = next((k for p, k in kind_by_prefix.items() if d.name.startswith(p)), None)
        if kind is None or (d.name.endswith("_quick") and not include_quick):
            continue
        try:
            cfg = load_json(cfg_path)
        except Exception:
            continue
        if task and cfg.get("task") != task:
            continue
        specs.append(ModelSpec(kind, cfg.get("model_name") or _default_name(kind, d.name, cfg.get("task", "")),
                               cfg.get("task", ""), str(d)))
    specs.extend(_baseline_specs(task))
    return specs


def _default_name(kind: str, dir_name: str, task: str) -> str:
    """experiments.model_name for a model directory (tiny_transformer_form_4L -> tiny_transformer_4L)."""
    return dir_name.replace(f"_{task}", "", 1) if task else dir_name


def _baseline_specs(task: str | None) -> list[ModelSpec]:
    """Baseline models = rows of family baseline/tfidf in the experiments table that load."""
    try:
        from src import experiments_db
        with experiments_db.connect() as con:
            rows = con.execute("SELECT DISTINCT model_name, task FROM experiments "
                               "WHERE model_family IN ('baseline', 'tfidf')").fetchall()
    except Exception:
        return []
    return [ModelSpec("baseline", name, t, name) for name, t in rows
            if (task is None or t == task) and not name.endswith("_quick")
            and baseline_path(name, t) is not None]          # only models saved on disk
