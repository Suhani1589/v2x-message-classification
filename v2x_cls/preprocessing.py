"""Text preprocessing: tokenize on train only, pad/truncate sequences."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical


class TextPreprocessor:
    """Fit tokenizer on training texts only to avoid leakage."""

    def __init__(
        self,
        label_order: Sequence[str],
        max_sequence_length: int = 12,
        oov_token: str = "<OOV>",
        lowercase: bool = True,
    ) -> None:
        self.label_order = list(label_order)
        self.label_to_index = {lab: i for i, lab in enumerate(self.label_order)}
        self.index_to_label = {i: lab for lab, i in self.label_to_index.items()}
        self.max_sequence_length = max_sequence_length
        self.oov_token = oov_token
        self.lowercase = lowercase
        self.tokenizer = Tokenizer(oov_token=oov_token, lower=lowercase)
        self._fitted = False

    @property
    def vocab_size(self) -> int:
        if not self._fitted:
            raise RuntimeError("Tokenizer has not been fitted yet.")
        # +1 for Keras embedding (indices are 1..n; 0 is padding)
        return len(self.tokenizer.word_index) + 1

    @property
    def num_classes(self) -> int:
        return len(self.label_order)

    def fit(self, texts: Sequence[str]) -> "TextPreprocessor":
        self.tokenizer.fit_on_texts(list(texts))
        self._fitted = True
        return self

    def texts_to_sequences(self, texts: Sequence[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() on training texts before transforming.")
        seqs = self.tokenizer.texts_to_sequences(list(texts))
        return pad_sequences(
            seqs,
            maxlen=self.max_sequence_length,
            padding="post",
            truncating="post",
        )

    def labels_to_indices(self, labels: Sequence[str]) -> np.ndarray:
        missing = sorted({lab for lab in labels if lab not in self.label_to_index})
        if missing:
            raise ValueError(f"Unknown labels: {missing}")
        return np.array([self.label_to_index[lab] for lab in labels], dtype=np.int32)

    def labels_to_onehot(self, labels: Sequence[str]) -> np.ndarray:
        indices = self.labels_to_indices(labels)
        return to_categorical(indices, num_classes=self.num_classes)

    def transform(
        self, texts: Sequence[str], labels: Sequence[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = self.texts_to_sequences(texts)
        y_idx = self.labels_to_indices(labels)
        y = to_categorical(y_idx, num_classes=self.num_classes)
        return x, y, y_idx

    def sequence_to_tokens(self, sequence: Sequence[int]) -> List[str]:
        index_word = self.tokenizer.index_word
        tokens = []
        for idx in sequence:
            if int(idx) == 0:
                tokens.append("<PAD>")
            else:
                tokens.append(index_word.get(int(idx), self.oov_token))
        return tokens

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "tokenizer.pkl").open("wb") as f:
            pickle.dump(self.tokenizer, f)
        meta = {
            "label_order": self.label_order,
            "label_to_index": self.label_to_index,
            "max_sequence_length": self.max_sequence_length,
            "oov_token": self.oov_token,
            "lowercase": self.lowercase,
            "vocab_size": self.vocab_size,
            "num_classes": self.num_classes,
            "formulation": "multiclass",
            "formulation_note": (
                "Paper says multi-label but uses single class ids + Softmax + "
                "CategoricalCrossentropy; we treat this as multiclass."
            ),
        }
        with (directory / "label_mapping.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, directory: str | Path) -> "TextPreprocessor":
        directory = Path(directory)
        with (directory / "label_mapping.json").open("r", encoding="utf-8") as f:
            meta = json.load(f)
        obj = cls(
            label_order=meta["label_order"],
            max_sequence_length=meta["max_sequence_length"],
            oov_token=meta.get("oov_token", "<OOV>"),
            lowercase=meta.get("lowercase", True),
        )
        with (directory / "tokenizer.pkl").open("rb") as f:
            obj.tokenizer = pickle.load(f)
        obj._fitted = True
        return obj


def build_preprocessor_from_config(cfg: Dict[str, Any]) -> TextPreprocessor:
    return TextPreprocessor(
        label_order=cfg["data"]["label_order"],
        max_sequence_length=cfg["preprocessing"]["max_sequence_length"],
        oov_token=cfg["preprocessing"]["oov_token"],
        lowercase=cfg["preprocessing"]["lowercase"],
    )
