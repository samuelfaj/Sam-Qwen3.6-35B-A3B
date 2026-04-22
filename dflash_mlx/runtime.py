from __future__ import annotations

import os
import time
from typing import Any

import mlx.core as mx
from mlx_lm.models.base import scaled_dot_product_attention

from dflash.model_mlx import _make_target_cache, _patch_model


_HYBRID_SDPA_EXACT_KV_THRESHOLD = int(
    os.environ.get("DDTREE_SDPA_EXACT_KV_THRESHOLD", "8192")
)


def _target_text_model(target_model: Any) -> Any:
    if (
        hasattr(target_model, "language_model")
        and hasattr(target_model.language_model, "model")
        and hasattr(target_model.language_model.model, "layers")
    ):
        return target_model.language_model.model
    if hasattr(target_model, "model") and hasattr(target_model.model, "layers"):
        return target_model.model
    if (
        hasattr(target_model, "layers")
        and hasattr(target_model, "embed_tokens")
        and hasattr(target_model, "norm")
    ):
        return target_model
    raise AttributeError(f"Cannot resolve text model for {type(target_model).__name__}")


def _target_embed_tokens(target_model: Any) -> Any:
    inner = _target_text_model(target_model)
    embed_tokens = getattr(inner, "embed_tokens", None)
    if embed_tokens is None:
        raise AttributeError(f"Cannot resolve embed_tokens for {type(target_model).__name__}")
    return embed_tokens


def _lm_head_logits(target_model: Any, hidden_states: mx.array) -> mx.array:
    lm_holder = getattr(target_model, "language_model", target_model)
    lm_head = getattr(target_model, "lm_head", None) or getattr(lm_holder, "lm_head", None)
    if lm_head is None:
        embed_tokens = _target_embed_tokens(target_model)
        lm_head = getattr(embed_tokens, "as_linear", None)
    if lm_head is None:
        raise AttributeError(f"Cannot resolve lm_head for {type(target_model).__name__}")
    return lm_head(hidden_states)


def _split_sdpa_output(
    *,
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    scale: float,
    mask: mx.array,
    cache: Any,
    chunk_size: int,
    cached_prefix_len: int,
) -> mx.array:
    del chunk_size, cached_prefix_len
    return scaled_dot_product_attention(
        queries,
        keys,
        values,
        cache=cache,
        scale=scale,
        mask=mask,
    )


def _capture_zero_based_layer_ids(capture_layer_ids: set[int] | None) -> list[int]:
    if not capture_layer_ids:
        return []
    return sorted(layer_id - 1 for layer_id in capture_layer_ids if layer_id > 0)


def _ensure_capture_hooks(target_model: Any, zero_based_layer_ids: list[int]) -> list[int]:
    if not zero_based_layer_ids:
        return []
    current = getattr(target_model, "_ddtree_capture_layer_ids", None)
    desired = tuple(zero_based_layer_ids)
    if current is None:
        _patch_model(target_model, zero_based_layer_ids)
        target_model._ddtree_capture_layer_ids = desired
    elif tuple(current) != desired:
        raise RuntimeError(
            "DDTree capture hooks were initialized for a different layer set. "
            "Reload the model before switching draft configurations."
        )
    return list(desired)


def target_forward_with_hidden_states(
    target_model: Any,
    *,
    input_ids: mx.array,
    cache: list[Any],
    capture_layer_ids: set[int] | None = None,
) -> tuple[mx.array, dict[int, mx.array]]:
    zero_based_layer_ids = _capture_zero_based_layer_ids(capture_layer_ids)
    active_layer_ids = _ensure_capture_hooks(target_model, zero_based_layer_ids)
    logits = target_model(input_ids, cache)
    captured: dict[int, mx.array] = {}
    hidden_states = getattr(target_model, "_hidden_states", [])
    for layer_id, hidden_state in zip(active_layer_ids, hidden_states):
        if hidden_state is not None:
            captured[layer_id + 1] = hidden_state
    return logits, captured


def extract_context_feature_from_dict(
    captured_hidden_states: dict[int, mx.array],
    target_layer_ids: list[int],
) -> mx.array:
    selected = []
    for layer_id in target_layer_ids:
        key = int(layer_id) + 1
        hidden = captured_hidden_states.get(key)
        if hidden is None:
            raise KeyError(f"Missing captured hidden state for layer {key}")
        selected.append(hidden)
    if not selected:
        raise ValueError("target_layer_ids must not be empty")
    return mx.concatenate(selected, axis=-1)


def make_target_cache(
    target_model: Any,
    enable_speculative_linear_cache: bool = True,
) -> list[Any]:
    del enable_speculative_linear_cache
    raw_bits = os.environ.get("LOCAL_DFLASH_TURBOQUANT_BITS")
    turboquant_bits = None
    if raw_bits:
        try:
            parsed = float(raw_bits)
        except ValueError:
            parsed = 0.0
        if parsed > 0:
            turboquant_bits = parsed
    return _make_target_cache(target_model, turboquant_bits=turboquant_bits)


def greedy_tokens_with_mask(logits: mx.array, suppress_mask: mx.array | None) -> mx.array:
    masked_logits = logits
    if suppress_mask is not None:
        floor = mx.array(-1e9, dtype=masked_logits.dtype)
        masked_logits = mx.where(suppress_mask, floor, masked_logits)
    return mx.argmax(masked_logits, axis=-1).astype(mx.uint32)


def build_suppress_token_mask(
    vocab_size: int,
    suppress_token_ids: list[int] | None,
) -> mx.array | None:
    if not suppress_token_ids:
        return None
    mask = mx.zeros((vocab_size,), dtype=mx.bool_)
    for token_id in suppress_token_ids:
        if 0 <= int(token_id) < vocab_size:
            mask[int(token_id)] = True
    return mask


def _eval_logits_and_captured(logits: mx.array, captured_hidden_states: dict[int, mx.array]) -> None:
    arrays = [logits, *captured_hidden_states.values()]
    mx.eval(*arrays)


def _arm_target_rollback_with_prefix(target_cache: list[Any], prefix_len: int) -> None:
    for cache_entry in target_cache:
        if hasattr(cache_entry, "offset"):
            cache_entry.offset = int(prefix_len)


def _match_acceptance_length(drafted_tokens: mx.array, posterior_tokens: mx.array) -> mx.array:
    drafted_list = drafted_tokens.tolist()
    posterior_list = posterior_tokens.tolist()
    accepted = 0
    for drafted, posterior in zip(drafted_list, posterior_list):
        if drafted != posterior:
            break
        accepted += 1
    return mx.array(accepted, dtype=mx.int32)


def _resolve_verify_len_cap(target_model: Any, block_size: int) -> int:
    del target_model
    return int(block_size)


def _restore_target_cache_after_acceptance(
    target_cache: list[Any],
    *,
    target_len: int,
    acceptance_length: int,
    drafted_tokens: int,
) -> int:
    trim = int(drafted_tokens) - int(acceptance_length)
    if trim <= 0:
        return 0
    started = time.perf_counter_ns()
    for cache_entry in target_cache:
        if hasattr(cache_entry, "trim") and callable(cache_entry.trim):
            try:
                cache_entry.trim(trim)
                continue
            except Exception:
                pass
        if hasattr(cache_entry, "offset"):
            cache_entry.offset = int(target_len)
    return time.perf_counter_ns() - started


def _verify_target_block(
    *,
    target_model: Any,
    verify_ids: mx.array,
    target_cache: list[Any],
    verify_chunk_tokens: int | None,
    capture_layer_ids: set[int] | None = None,
) -> tuple[mx.array, dict[int, mx.array]]:
    del verify_chunk_tokens
    return target_forward_with_hidden_states(
        target_model,
        input_ids=verify_ids,
        cache=target_cache,
        capture_layer_ids=capture_layer_ids,
    )
