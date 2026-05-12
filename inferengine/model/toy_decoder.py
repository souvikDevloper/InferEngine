from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch


VOCAB = [
    "scalable", "batch", "cache", "token", "latency", "throughput", "queue", "kernel",
    "prefill", "decode", "memory", "scheduler", "engine", "request", "tensor", "page",
    "attention", "serve", "model", "stream", "profile", "optimize", "system", "fast",
]


@dataclass
class DecoderOutput:
    token: str
    token_id: int


class TorchToyDecoder:
    """Small deterministic decoder used to exercise the serving system.

    The model intentionally stays lightweight so the repo can run on a laptop.
    It still uses PyTorch tensor operations, making the scheduler/cache path close
    to a real serving loop without requiring a large LLM checkpoint.
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        self.weights = torch.arange(1, len(VOCAB) + 1, dtype=torch.float32, device=self.device)

    def next_token(self, prompt: str, generated: list[str], step: int) -> DecoderOutput:
        seed = f"{prompt}|{' '.join(generated)}|{step}".encode("utf-8")
        digest = hashlib.sha256(seed).digest()
        base = int.from_bytes(digest[:4], "big")
        logits = torch.sin(self.weights * ((base % 997) + 1)) + torch.cos(self.weights + step)
        token_id = int(torch.argmax(logits).item()) % len(VOCAB)
        return DecoderOutput(token=VOCAB[token_id], token_id=token_id)

    @property
    def name(self) -> str:
        return f"torch-toy-decoder/{self.device.type}"
