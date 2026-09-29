"""Hugging Face bi-encoders.

Two encoding modes:

* ``encode(texts)`` encodes each text on its own (batch of one, no padding).
  The embedding of a text therefore does not depend on what else is being
  encoded or in what order, which makes a text-keyed cache safe. All new
  (v4) experiments use this mode.
* ``encode_batched(texts, batch_size)`` reproduces the legacy pipeline's
  padded batches exactly; used only by legacy_v3 for bitwise reproduction.
"""
import hashlib
import pickle

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from . import CACHE

MEDCPT_QUERY = "ncbi/MedCPT-Query-Encoder"
MEDCPT_ARTICLE = "ncbi/MedCPT-Article-Encoder"
SAPBERT = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


_LOADED = {}  # model name -> (tokenizer, model); encoders that differ only in max_length share weights


def _load(model_name):
    if model_name not in _LOADED:
        _LOADED[model_name] = (AutoTokenizer.from_pretrained(model_name),
                               AutoModel.from_pretrained(model_name).to(DEVICE).eval())
    return _LOADED[model_name]


class Encoder:
    def __init__(self, model_name, max_length, pooling="cls", normalize=True, cache=True):
        self.model_name, self.max_length = model_name, max_length
        self.pooling, self.normalize = pooling, normalize
        self.tokenizer, self.model = _load(model_name)
        slug = model_name.replace("/", "__")
        self.cache_path = CACHE / f"emb_{slug}_{max_length}_{pooling}.pkl" if cache else None
        self._cache = {}
        if self.cache_path is not None and self.cache_path.exists():
            with open(self.cache_path, "rb") as f:
                self._cache = pickle.load(f)
        self._dirty = False

    @staticmethod
    def _key(text):
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    @torch.no_grad()
    def _forward(self, batch):
        enc = self.tokenizer(batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        out = self.model(**enc).last_hidden_state
        if self.pooling == "cls":
            vec = out[:, 0]
        else:  # mean pooling over non-padding tokens
            mask = enc["attention_mask"].unsqueeze(-1).float()
            vec = (out * mask).sum(1) / mask.sum(1)
        if self.normalize:
            vec = torch.nn.functional.normalize(vec, p=2, dim=1)
        return vec.cpu().numpy().astype(np.float32)

    def encode(self, texts):
        """One text at a time, cached by text."""
        out = np.empty((len(texts), self.model.config.hidden_size), dtype=np.float32)
        for i, t in enumerate(texts):
            k = self._key(t)
            if k not in self._cache:
                self._cache[k] = self._forward([t])[0]
                self._dirty = True
            out[i] = self._cache[k]
        return out

    def encode_batched(self, texts, batch_size=16):
        """Legacy padded batching, uncached."""
        return np.concatenate([self._forward(texts[i:i + batch_size]) for i in range(0, len(texts), batch_size)])

    def n_tokens(self, text):
        """Token count including special tokens, before truncation."""
        return len(self.tokenizer(text)["input_ids"])

    def truncated(self, text):
        return self.n_tokens(text) > self.max_length

    def save(self):
        if self.cache_path is not None and self._dirty:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "wb") as f:
                pickle.dump(self._cache, f)
            self._dirty = False
