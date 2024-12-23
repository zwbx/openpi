import abc

import numpy as np
import sentencepiece
from typing_extensions import override

import openpi.shared.download as download


class Tokenizer(abc.ABC):
    @abc.abstractmethod
    def tokenize(self, batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Tokenize a batch of prompts.

        Args:
            batch: A batch of text prompts to tokenize.

        Returns:
            A tuple containing the tokenized prompts and the corresponding masks.
        """


class PaligemmaTokenizer(Tokenizer):
    def __init__(self, max_len: int = 48):
        self._max_len = max_len

        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

    @override
    def tokenize(self, batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        batch_tokens, batch_masks = [], []

        for text in batch:
            cleaned_text = text.lower().strip().replace("_", " ").replace("\n", " ")
            # tokenize "\n" separately as the "start of answer" token
            tokens = self._tokenizer.encode(cleaned_text, add_bos=True) + self._tokenizer.encode("\n")
            tokens_len = len(tokens)
            if tokens_len < self._max_len:
                padding = [0] * (self._max_len - tokens_len)
                mask = [1] * tokens_len + padding
                tokens = tokens + padding
            else:
                tokens = tokens[: self._max_len]
                mask = [1] * self._max_len

            batch_tokens.append(tokens)
            batch_masks.append(mask)

        return np.array(batch_tokens), np.array(batch_masks)
