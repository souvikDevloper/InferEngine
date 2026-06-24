from __future__ import annotations

import os
from typing import Any, Protocol, Sequence

from inferengine.model.toy_decoder import TorchToyDecoder


class DecoderBackend(Protocol):
    name: str

    def encode(self, prompt: str) -> list[Any]: ...

    def next_tokens(self, states: Sequence[Any]) -> list[str]: ...

    def detokenize(self, tokens: Sequence[str]) -> str: ...

    def release(self, state: Any) -> None: ...


class ToyBackend:
    def __init__(self) -> None:
        self.decoder = TorchToyDecoder()

    @property
    def name(self) -> str:
        return self.decoder.name

    def encode(self, prompt: str) -> list[str]:
        from inferengine.core.tokenizer import tokenize

        return tokenize(prompt)

    def next_tokens(self, states: Sequence[Any]) -> list[str]:
        return [
            self.decoder.next_token(state.prompt, state.generated, len(state.generated)).token
            for state in states
        ]

    def detokenize(self, tokens: Sequence[str]) -> str:
        from inferengine.core.tokenizer import detokenize

        return detokenize(list(tokens))

    def release(self, state: Any) -> None:
        del state


def build_backend(kind: str | None = None, model_id: str | None = None) -> DecoderBackend:
    selected = (kind or os.getenv("INFERENGINE_BACKEND", "toy")).lower()
    if selected == "toy":
        return ToyBackend()
    if selected in {"hf", "transformers"}:
        from inferengine.model.hf_decoder import HuggingFaceContinuousDecoder

        return HuggingFaceContinuousDecoder(
            model_id=model_id or os.getenv("INFERENGINE_MODEL", "meta-llama/Meta-Llama-3-8B")
        )
    raise ValueError(f"unknown INFERENGINE_BACKEND={selected!r}; use 'toy' or 'transformers'")
