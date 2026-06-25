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


@dataclass
class CohortMember:
    cohort_id: int
    row: int


@dataclass
class CohortCache:
    cohort_id: int
    request_ids: tuple[str, ...]
    states: tuple[Any, ...]
    cache: Any
    lengths: list[int]
    cache_width: int
    next_input_ids: torch.Tensor


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
        self._cohort: CohortCache | None = None
        self._cohort_id = 0
        self._cohort_cache_mode = os.getenv("INFERENGINE_COHORT_CACHE", "auto").strip().lower()
        self._fast_stream_text = os.getenv("INFERENGINE_FAST_STREAM_TEXT", "").strip()

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

    @torch.inference_mode()
    def complete_batch(self, states: Sequence[Any]) -> list[list[str]]:
        """Generate full batched completions with Transformers' optimized loop.

        This mode is intended for throughput benchmarks where the request set is
        already batched. It avoids the Python scheduler calling the model once
        per output token while still returning per-token chunks to the API layer.
        """
        self._materialize_cohort_members()
        encoded = [{"input_ids": state.prompt_tokens} for state in states]
        batch = self.tokenizer.pad(encoded, padding=True, return_tensors="pt").to(self.input_device)
        remaining = [state.max_new_tokens - len(state.generated) for state in states]
        max_new_tokens = max(remaining)
        generated = self.model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            min_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        new_token_ids = generated[:, batch["input_ids"].shape[1] :]
        outputs: list[list[str]] = []
        for row, count in enumerate(remaining):
            ids = new_token_ids[row, :count].detach().to("cpu").tolist()
            decoded = self.tokenizer.batch_decode([[int(token_id)] for token_id in ids], skip_special_tokens=True)
            outputs.append([text or " " for text in decoded])
        return outputs

    def _prefill(self, states: Sequence[Any]) -> dict[str, str]:
        # A prefill can happen while an older decode cohort is still active if
        # the scheduler admits replacement requests into freed batch slots.
        # Materialize that older cohort before assigning self._cohort to the new
        # prefill batch; otherwise older RequestState objects keep stale
        # CohortMember handles that no longer have a backing CohortCache.
        self._materialize_cohort_members()
        encoded = [{"input_ids": state.prompt_tokens} for state in states]
        batch = self.tokenizer.pad(encoded, padding=True, return_tensors="pt").to(self.input_device)
        result = self.model(**batch, use_cache=True, return_dict=True)
        token_ids = result.logits[:, -1, :].argmax(dim=-1)
        lengths = [len(state.prompt_tokens) for state in states]
        if self._should_reuse_prefill_cohort(lengths):
            self._cohort_id += 1
            self._cohort = CohortCache(
                cohort_id=self._cohort_id,
                request_ids=tuple(state.request_id for state in states),
                states=tuple(states),
                cache=result.past_key_values,
                lengths=lengths,
                cache_width=max(lengths) if lengths else 0,
                next_input_ids=token_ids.detach(),
            )
            for row, state in enumerate(states):
                state.backend_state = CohortMember(cohort_id=self._cohort_id, row=row)
        else:
            self._cohort = None
            legacy = self._legacy_cache(result.past_key_values)
            for row, state in enumerate(states):
                length = lengths[row]
                layers = tuple(
                    (
                        key[row : row + 1, :, -length:, :].contiguous(),
                        value[row : row + 1, :, -length:, :].contiguous(),
                    )
                    for key, value in legacy
                )
                state.backend_state = SequenceCache(
                    layers=layers,
                    length=length,
                    next_input_id=int(token_ids[row]),
                )
        return dict(zip((state.request_id for state in states), self._decode_tokens(token_ids), strict=True))

    def _should_reuse_prefill_cohort(self, lengths: Sequence[int]) -> bool:
        """Reuse a raw HF batch cache only when every row has the same valid length.

        Hugging Face cache objects track a single cache position for the batch.
        With left-padded random-length prompts, reusing the raw padded cache can
        leave rows with different logical positions in one mutable cache object.
        That path is fast, but unsafe for vLLM's random-length benchmark data.
        """
        if self._cohort_cache_mode in {"0", "false", "off", "disabled", "no"}:
            return False
        return bool(lengths) and len(set(lengths)) == 1

    def _decode(self, states: Sequence[Any]) -> dict[str, str]:
        if self._can_decode_cohort(states):
            return self._decode_cohort(states)
        self._materialize_cohort_members()
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
        return dict(zip((state.request_id for state in states), self._decode_tokens(token_ids), strict=True))

    def _can_decode_cohort(self, states: Sequence[Any]) -> bool:
        if self._cohort is None or len(states) != len(self._cohort.request_ids):
            return False
        ids = tuple(state.request_id for state in states)
        if ids != self._cohort.request_ids:
            return False
        return all(
            isinstance(state.backend_state, CohortMember) and state.backend_state.cohort_id == self._cohort.cohort_id
            for state in states
        )

    def _decode_cohort(self, states: Sequence[Any]) -> dict[str, str]:
        assert self._cohort is not None
        cohort = self._cohort
        input_ids = cohort.next_input_ids.reshape(-1, 1).to(self.input_device, non_blocking=True)
        attention_mask = torch.zeros((len(states), cohort.cache_width + 1), dtype=torch.long, device=self.input_device)
        for row, length in enumerate(cohort.lengths):
            attention_mask[row, -(length + 1) :] = 1
        position_ids = torch.tensor([[length] for length in cohort.lengths], dtype=torch.long, device=self.input_device)
        result = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cohort.cache,
            use_cache=True,
            return_dict=True,
        )
        token_ids = result.logits[:, -1, :].argmax(dim=-1)
        cohort.cache = result.past_key_values
        cohort.lengths = [length + 1 for length in cohort.lengths]
        cohort.cache_width += 1
        cohort.next_input_ids = token_ids.detach()
        return dict(zip((state.request_id for state in states), self._decode_tokens(token_ids), strict=True))

    def _materialize_cohort_members(self) -> None:
        """Convert the fast batched cache back into per-request caches if the cohort changes.

        This keeps the implementation correct for mixed max-token requests or mid-cohort
        admission/eviction while preserving the zero-repack path for stable benchmark
        cohorts.
        """
        if self._cohort is None:
            return
        cohort = self._cohort
        legacy = self._legacy_cache(cohort.cache)
        for row, state in enumerate(cohort.states):
            if not isinstance(state.backend_state, CohortMember) or state.backend_state.cohort_id != cohort.cohort_id:
                continue
            length = cohort.lengths[row]
            layers = tuple(
                (
                    key[row : row + 1, :, -length:, :].contiguous(),
                    value[row : row + 1, :, -length:, :].contiguous(),
                )
                for key, value in legacy
            )
            state.backend_state = SequenceCache(
                layers=layers,
                length=length,
                next_input_id=int(cohort.next_input_ids[row]),
            )
        self._cohort = None

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

    def _decode_tokens(self, token_ids: torch.Tensor) -> list[str]:
        if self._fast_stream_text:
            return [self._fast_stream_text] * int(token_ids.shape[0])
        ids = token_ids.detach().to("cpu").tolist()
        decoded = self.tokenizer.batch_decode([[int(token_id)] for token_id in ids], skip_special_tokens=True)
        return [text or " " for text in decoded]

    def detokenize(self, tokens: Sequence[str]) -> str:
        return "".join(tokens)

    def release(self, state: Any) -> None:
        if self._cohort is not None and state.request_id in self._cohort.request_ids:
            self._materialize_cohort_members()
        state.backend_state = None
