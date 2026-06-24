from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Sequence

import torch


@dataclass
class SequenceCache:
    layers: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    length: int
    next_input_id: int


class HuggingFaceContinuousDecoder:
    """Real causal-LM backend with batched prefill and decode.

    Per-request KV tensors remain independent while active requests are packed
    into a left-padded batch for each decode step. This gives continuous
    admission and one model invocation per active decode cohort without
    pretending that the laptop-friendly toy decoder is a GPU benchmark target.
    """

    def __init__(self, model_id: str) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("install InferEngine's gpu extra to use the transformers backend") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("the transformers performance backend requires an NVIDIA CUDA GPU")
        self.model_id = model_id
        self.device_map = os.getenv("INFERENGINE_DEVICE_MAP", "").strip() or None
        self.input_device = torch.device(os.getenv("INFERENGINE_INPUT_DEVICE", "cuda:0"))
        major, _ = torch.cuda.get_device_capability(self.input_device)
        self.dtype = self._select_dtype(major)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        model_kwargs: dict[str, Any] = {
            "torch_dtype": self.dtype,
            "low_cpu_mem_usage": True,
        }
        attn_implementation = os.getenv("INFERENGINE_ATTN_IMPLEMENTATION", "").strip()
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        if self.device_map:
            model_kwargs["device_map"] = self.device_map
            max_memory = self._parse_max_memory(os.getenv("INFERENGINE_MAX_MEMORY", ""))
            if max_memory:
                model_kwargs["max_memory"] = max_memory
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        if not self.device_map:
            self.model = self.model.to(self.input_device)
        self.input_device = next(self.model.parameters()).device
        self.model.eval()

    @property
    def name(self) -> str:
        return self.model_id

    def encode(self, prompt: str) -> list[int]:
        return self.tokenizer.encode(prompt, add_special_tokens=True)

    @torch.inference_mode()
    def next_tokens(self, states: Sequence[Any]) -> list[str]:
        outputs: dict[str, str] = {}
        prefill = [state for state in states if state.backend_state is None]
        decode = [state for state in states if state.backend_state is not None]
        if prefill:
            outputs.update(self._prefill(prefill))
        if decode:
            outputs.update(self._decode(decode))
        return [outputs[state.request_id] for state in states]

    def _prefill(self, states: Sequence[Any]) -> dict[str, str]:
        encoded = [{"input_ids": state.prompt_tokens} for state in states]
        batch = self.tokenizer.pad(encoded, padding=True, return_tensors="pt").to(self.input_device)
        result = self.model(**batch, use_cache=True, return_dict=True)
        token_ids = result.logits[:, -1, :].argmax(dim=-1)
        legacy = self._legacy_cache(result.past_key_values)
        for row, state in enumerate(states):
            length = len(state.prompt_tokens)
            layers = tuple(
                (
                    key[row : row + 1, :, -length:, :].contiguous(),
                    value[row : row + 1, :, -length:, :].contiguous(),
                )
                for key, value in legacy
            )
            state.backend_state = SequenceCache(layers=layers, length=length, next_input_id=int(token_ids[row]))
        return {state.request_id: self._decode_token(int(token_ids[row])) for row, state in enumerate(states)}

    def _decode(self, states: Sequence[Any]) -> dict[str, str]:
        caches: list[SequenceCache] = [state.backend_state for state in states]
        max_length = max(cache.length for cache in caches)
        batched_layers: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer in range(len(caches[0].layers)):
            keys, values = [], []
            for cache in caches:
                key, value = cache.layers[layer]
                padding = max_length - cache.length
                if padding:
                    key = torch.nn.functional.pad(key, (0, 0, padding, 0))
                    value = torch.nn.functional.pad(value, (0, 0, padding, 0))
                keys.append(key)
                values.append(value)
            batched_layers.append((torch.cat(keys, dim=0), torch.cat(values, dim=0)))
        input_ids = torch.tensor([[cache.next_input_id] for cache in caches], dtype=torch.long, device=self.input_device)
        attention_mask = torch.zeros((len(caches), max_length + 1), dtype=torch.long, device=self.input_device)
        for row, cache in enumerate(caches):
            attention_mask[row, -(cache.length + 1) :] = 1
        position_ids = torch.tensor([[cache.length] for cache in caches], dtype=torch.long, device=self.input_device)
        result = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=self._cache_for_model(tuple(batched_layers)),
            use_cache=True,
            return_dict=True,
        )
        token_ids = result.logits[:, -1, :].argmax(dim=-1)
        legacy = self._legacy_cache(result.past_key_values)
        for row, state in enumerate(states):
            old = caches[row]
            length = old.length + 1
            layers = tuple(
                (
                    key[row : row + 1, :, -length:, :].contiguous(),
                    value[row : row + 1, :, -length:, :].contiguous(),
                )
                for key, value in legacy
            )
            state.backend_state = SequenceCache(layers=layers, length=length, next_input_id=int(token_ids[row]))
        return {state.request_id: self._decode_token(int(token_ids[row])) for row, state in enumerate(states)}

    @staticmethod
    def _legacy_cache(cache: Any) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        if hasattr(cache, "to_legacy_cache"):
            cache = cache.to_legacy_cache()
        return tuple((layer[0], layer[1]) for layer in cache)

    @staticmethod
    def _cache_for_model(cache: tuple[tuple[torch.Tensor, torch.Tensor], ...]) -> Any:
        try:
            from transformers.cache_utils import DynamicCache

            return DynamicCache.from_legacy_cache(cache)
        except (ImportError, AttributeError):
            return cache

    @staticmethod
    def _parse_max_memory(spec: str) -> dict[int | str, str]:
        max_memory: dict[int | str, str] = {}
        for item in spec.split(","):
            if not item.strip():
                continue
            key, sep, value = item.partition(":")
            if not sep or not key.strip() or not value.strip():
                raise ValueError("INFERENGINE_MAX_MEMORY must look like '0:14GiB,1:14GiB,cpu:48GiB'")
            normalized_key: int | str = int(key) if key.isdigit() else key
            max_memory[normalized_key] = value
        return max_memory

    @staticmethod
    def _select_dtype(device_major: int) -> torch.dtype:
        requested = os.getenv("INFERENGINE_TORCH_DTYPE", "auto").strip().lower()
        if requested in {"auto", ""}:
            return torch.bfloat16 if device_major >= 8 and torch.cuda.is_bf16_supported() else torch.float16
        dtypes = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if requested not in dtypes:
            raise ValueError("INFERENGINE_TORCH_DTYPE must be auto, float16, bfloat16, or float32")
        return dtypes[requested]

    def _decode_token(self, token_id: int) -> str:
        return self.tokenizer.decode([token_id], skip_special_tokens=True) or " "

    def detokenize(self, tokens: Sequence[str]) -> str:
        return "".join(tokens)

    def release(self, state: Any) -> None:
        state.backend_state = None
