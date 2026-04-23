import copy
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional, Tuple


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from mlx_lm.generate import generation_stream
from mlx_lm.models.cache import KVCache, RotatingKVCache, can_trim_prompt_cache, make_prompt_cache, trim_prompt_cache
from mlx_lm.models.qwen3 import MLP
from mlx_lm.models.rope_utils import initialize_rope
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper

try:
    import mlx_lm.models.gated_delta as _gd_mod
    _HAS_GDN = True
except ImportError:
    _HAS_GDN = False


_GDN_PATCH_LOCK = RLock()


@dataclass
class DFlashConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    block_size: int
    target_layer_ids: Tuple[int, ...]
    num_target_layers: int
    mask_token_id: int = 0
    rope_scaling: Optional[Dict[str, Any]] = None
    sliding_window_size: Optional[int] = None
    turboquant_bits: Optional[float] = None
    rotating_keep_tokens: int = 0


def _resolve_local_or_hub_path(model_id_or_path: str, allow_patterns: Optional[List[str]] = None) -> Path:
    path = Path(model_id_or_path).expanduser()
    if path.exists():
        return path
    return Path(snapshot_download(model_id_or_path, allow_patterns=allow_patterns))


def _infer_model_head_dim(model: Any) -> int:
    candidates = [
        model,
        getattr(model, "model", None),
        getattr(model, "language_model", None),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        args = getattr(candidate, "args", None)
        if args is None:
            continue
        head_dim = getattr(args, "head_dim", None)
        if isinstance(head_dim, int) and head_dim > 0:
            return head_dim
        hidden_size = getattr(args, "hidden_size", None)
        num_attention_heads = getattr(args, "num_attention_heads", None)
        if (
            isinstance(hidden_size, int)
            and isinstance(num_attention_heads, int)
            and num_attention_heads > 0
        ):
            return hidden_size // num_attention_heads

    for layer in _get_layers(model):
        self_attn = getattr(layer, "self_attn", None)
        head_dim = getattr(self_attn, "head_dim", None)
        if isinstance(head_dim, int) and head_dim > 0:
            return head_dim

    raise ValueError(f"Could not infer attention head_dim for {type(model).__name__}")


class _StableTurboQuantKVCache:
    def __init__(self, inner: Any):
        self._inner = inner

    def update_and_fetch(self, keys, values):
        deq_keys, deq_values = self._inner.update_and_fetch(keys, values)
        return deq_keys.astype(keys.dtype), deq_values.astype(values.dtype)

    @property
    def state(self):
        return self._inner.state

    @state.setter
    def state(self, value):
        self._inner.state = value

    @property
    def meta_state(self):
        return self._inner.meta_state

    @meta_state.setter
    def meta_state(self, value):
        self._inner.meta_state = value

    def __deepcopy__(self, memo):
        copied = type(self).__new__(type(self))
        memo[id(self)] = copied
        copied._inner = copy.deepcopy(self._inner, memo)
        return copied

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _StableRotatingTurboQuantKVCache(_StableTurboQuantKVCache):
    def __init__(self, inner: Any, *, max_size: int, keep: int = 0):
        super().__init__(inner)
        self.max_size = max_size
        self.keep = keep
        self.offset = 0

    def _trim_return(self, trim_size: int, value: mx.array) -> mx.array:
        if trim_size <= 0:
            return value
        suffix = value[..., trim_size + self.keep :, :]
        if self.keep <= 0:
            return suffix
        prefix = value[..., : self.keep, :]
        return mx.concatenate([prefix, suffix], axis=2)

    def update_and_fetch(self, keys, values):
        prev_inner_offset = self._inner.offset
        deq_keys, deq_values = self._inner.update_and_fetch(keys, values)
        self.offset += keys.shape[2]

        trim_size = max(prev_inner_offset - self.max_size + 1, 0)
        if trim_size > 0:
            trimmed = int(self._inner.trim(trim_size))
            if trimmed > 0:
                deq_keys = self._trim_return(trimmed, deq_keys)
                deq_values = self._trim_return(trimmed, deq_values)

        return deq_keys.astype(keys.dtype), deq_values.astype(values.dtype)

    def __deepcopy__(self, memo):
        copied = type(self).__new__(type(self))
        memo[id(self)] = copied
        copied._inner = copy.deepcopy(self._inner, memo)
        copied.max_size = self.max_size
        copied.keep = self.keep
        copied.offset = self.offset
        return copied


def _turboquant_cache_cls():
    try:
        from mlx_turboquant.cache import TurboQuantKVCache as _TurboQuantKVCache
    except ImportError as exc:
        raise RuntimeError(
            "TurboQuant cache requested, but mlx-turboquant is not installed in the current environment."
        ) from exc
    return _TurboQuantKVCache


def _make_turboquant_cache_entry(
    *,
    bits: float,
    head_dim: int,
    layer_index: int,
    rotating_max_size: Optional[int] = None,
    keep: int = 0,
):
    inner = _turboquant_cache_cls()(
        bits=float(bits),
        head_dim=head_dim,
        key_seed=42 + layer_index * 2,
        value_seed=43 + layer_index * 2,
    )
    if rotating_max_size is not None:
        return _StableRotatingTurboQuantKVCache(inner, max_size=rotating_max_size, keep=keep)
    return _StableTurboQuantKVCache(inner)


def estimate_memory_bytes(value: Any, seen: Optional[set[int]] = None) -> int:
    if value is None:
        return 0

    if seen is None:
        seen = set()

    obj_id = id(value)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    nbytes = getattr(value, "nbytes", None)
    if isinstance(nbytes, int):
        return nbytes

    if isinstance(value, dict):
        return sum(estimate_memory_bytes(item, seen) for item in value.values())

    if isinstance(value, (list, tuple, set)):
        return sum(estimate_memory_bytes(item, seen) for item in value)

    if hasattr(value, "__dict__"):
        return sum(estimate_memory_bytes(item, seen) for item in vars(value).values())

    return 0


def estimate_prefill_state_bytes(prefill_state: Optional["PromptPrefillState"]) -> int:
    return estimate_memory_bytes(prefill_state)


def _make_target_cache(
    model: Any,
    turboquant_bits: Optional[float] = None,
):
    cache = make_prompt_cache(model)
    if turboquant_bits is None:
        return cache

    head_dim = _infer_model_head_dim(model)
    replaced = 0
    for idx, cache_entry in enumerate(cache):
        if isinstance(cache_entry, KVCache):
            cache[idx] = _make_turboquant_cache_entry(
                bits=float(turboquant_bits),
                head_dim=head_dim,
                layer_index=idx,
            )
            replaced += 1

    if replaced == 0:
        raise RuntimeError(
            "TurboQuant cache requested, but no compatible KVCache layers were found in the target model."
        )

    return cache


def _build_rope(
    head_dim: int,
    rope_theta: float,
    max_position_embeddings: int,
    rope_scaling: Optional[Dict[str, Any]],
):
    return initialize_rope(
        dims=head_dim,
        base=rope_theta,
        traditional=False,
        scaling_config=rope_scaling,
        max_position_embeddings=max_position_embeddings,
    )


class DFlashAttention(nn.Module):
    def __init__(self, config: DFlashConfig):
        super().__init__()
        dim = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads
        self.scale = config.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, n_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * config.head_dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(config.head_dim, eps=config.rms_norm_eps)

    def __call__(self, x, x_ctx, rope, cache):
        B, L, _ = x.shape
        S = x_ctx.shape[1]
        queries = self.q_proj(x)
        ctx_keys = self.k_proj(x_ctx)
        ctx_values = self.v_proj(x_ctx)
        prop_keys = self.k_proj(x)
        prop_values = self.v_proj(x)
        queries = self.q_norm(queries.reshape(B, L, self.n_heads, -1)).transpose(0, 2, 1, 3)
        ctx_keys = self.k_norm(ctx_keys.reshape(B, S, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        ctx_values = ctx_values.reshape(B, S, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        prop_keys = self.k_norm(prop_keys.reshape(B, L, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        prop_values = prop_values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        queries = rope(queries, offset=cache.offset + S)
        ctx_keys = rope(ctx_keys, offset=cache.offset)
        prop_keys = rope(prop_keys, offset=cache.offset + S)
        keys, values = cache.update_and_fetch(ctx_keys, ctx_values)
        keys = mx.concatenate([keys, prop_keys], axis=2)
        values = mx.concatenate([values, prop_values], axis=2)
        output = mx.fast.scaled_dot_product_attention(queries, keys, values, scale=self.scale)
        return self.o_proj(output.transpose(0, 2, 1, 3).reshape(B, L, -1))


class DFlashDecoderLayer(nn.Module):
    def __init__(self, config: DFlashConfig):
        super().__init__()
        self.self_attn = DFlashAttention(config)
        self.mlp = MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x, x_ctx, rope, cache):
        h = x + self.self_attn(self.input_layernorm(x), x_ctx, rope, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class DFlashDraftModel(nn.Module):
    def __init__(self, config: DFlashConfig):
        super().__init__()
        self.config = config
        concat_dim = len(config.target_layer_ids) * config.hidden_size
        self.fc = nn.Linear(concat_dim, config.hidden_size, bias=False)
        self.hidden_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = [DFlashDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rope = _build_rope(
            config.head_dim,
            config.rope_theta,
            config.max_position_embeddings,
            config.rope_scaling,
        )
        self.embed_tokens = None
        self.lm_head = None

    def bind(self, target_model):
        if hasattr(target_model, "embed_tokens"):
            inner = target_model
        elif hasattr(target_model, "model") and hasattr(target_model.model, "embed_tokens"):
            inner = target_model.model
        elif (hasattr(target_model, "language_model") and
              hasattr(target_model.language_model, "model") and
              hasattr(target_model.language_model.model, "embed_tokens")):
            inner = target_model.language_model.model
        else:
            raise AttributeError(f"Cannot find embed_tokens in {type(target_model).__name__}")
        self.embed_tokens = inner.embed_tokens
        lm = getattr(target_model, "language_model", target_model)
        self.lm_head = getattr(target_model, "lm_head", None) or getattr(lm, "lm_head", None) or self.embed_tokens.as_linear
        return self

    def make_cache(self):
        keep_tokens = max(0, int(self.config.rotating_keep_tokens or 0))
        if self.config.turboquant_bits is not None:
            # `_StableRotatingTurboQuantKVCache` explicitly wraps the MLX
            # limitation around quantized rotating caches (see
            # lmstudio-ai/mlx-engine#177). The combination is supported here
            # via that wrapper; callers who want the stricter safety behavior
            # can set LOCAL_DFLASH_FORBID_ROTATING_TURBOQUANT=1.
            if (
                self.config.sliding_window_size is not None
                and _env_flag("LOCAL_DFLASH_FORBID_ROTATING_TURBOQUANT", False)
            ):
                raise RuntimeError(
                    "TurboQuant KV-cache quantization combined with a rotating "
                    "sliding-window cache is disabled by "
                    "LOCAL_DFLASH_FORBID_ROTATING_TURBOQUANT=1. Disable "
                    "--sliding-window-size or --target-turboquant-bits."
                )
            return [
                _make_turboquant_cache_entry(
                    bits=float(self.config.turboquant_bits),
                    head_dim=self.config.head_dim,
                    layer_index=idx,
                    rotating_max_size=self.config.sliding_window_size,
                    keep=keep_tokens,
                )
                for idx, _ in enumerate(self.layers)
            ]
        if self.config.sliding_window_size is not None:
            return [
                RotatingKVCache(max_size=self.config.sliding_window_size, keep=keep_tokens)
                for _ in self.layers
            ]
        return [KVCache() for _ in self.layers]

    def __call__(self, inputs, target_hidden, cache):
        h = self.embed_tokens(inputs)
        h_ctx = self.hidden_norm(self.fc(target_hidden))
        for layer, c in zip(self.layers, cache):
            h = layer(h, h_ctx, self.rope, c)
        return self.lm_head(self.norm(h))


def load(model_id: str):
    from mlx_lm import load as mlx_lm_load
    return mlx_lm_load(model_id)


def load_draft(
    draft_id: str,
    sliding_window_size: Optional[int] = None,
    turboquant_bits: Optional[float] = None,
    rotating_keep_tokens: int = 0,
) -> DFlashDraftModel:
    # Treat 0 / negative as "disable sliding window" (full KV retention).
    if sliding_window_size is not None and sliding_window_size <= 0:
        sliding_window_size = None
    if turboquant_bits is not None and turboquant_bits <= 0:
        turboquant_bits = None
    path = _resolve_local_or_hub_path(draft_id, allow_patterns=["*.safetensors", "*.json"])
    cfg = json.loads((path / "config.json").read_text())
    config = DFlashConfig(
        hidden_size=cfg["hidden_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg["num_key_value_heads"],
        head_dim=cfg["head_dim"],
        intermediate_size=cfg["intermediate_size"],
        vocab_size=cfg["vocab_size"],
        rms_norm_eps=cfg["rms_norm_eps"],
        rope_theta=cfg["rope_theta"],
        max_position_embeddings=cfg["max_position_embeddings"],
        block_size=cfg["block_size"],
        target_layer_ids=tuple(cfg["dflash_config"]["target_layer_ids"]),
        num_target_layers=cfg["num_target_layers"],
        mask_token_id=cfg["dflash_config"]["mask_token_id"],
        rope_scaling=cfg.get("rope_scaling"),
        sliding_window_size=sliding_window_size,
        turboquant_bits=turboquant_bits,
        rotating_keep_tokens=max(0, int(rotating_keep_tokens or 0)),
    )
    weights = {k: v for f in path.glob("*.safetensors") for k, v in mx.load(str(f)).items()}
    model = DFlashDraftModel(config)
    model.load_weights(list(weights.items()))
    return model


class _LayerHook:
    def __init__(self, layer, idx, storage):
        self._layer, self._idx, self._storage = layer, idx, storage

    def __call__(self, *args, **kwargs):
        self._storage[self._idx] = out = self._layer(*args, **kwargs)
        return out

    def __getattr__(self, name):
        return getattr(self._layer, name)


def _get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise AttributeError(f"Cannot find layers in {type(model).__name__}")


def _patch_model(model, layer_ids):
    if hasattr(model, "_hidden_states"):
        return
    model._hidden_states = [None] * len(layer_ids)
    layers = _get_layers(model)
    for i, lid in enumerate(layer_ids):
        layers[lid] = _LayerHook(layers[lid], i, model._hidden_states)


def _clear_model_hidden_states(model):
    hidden_states = getattr(model, "_hidden_states", None)
    if isinstance(hidden_states, list):
        hidden_states[:] = [None] * len(hidden_states)


class _GDNStateCapture:
    def __init__(self):
        self.conv_data = []
        self._gdn_inputs = []
        self._gdn_cls = None
        self._orig_call = None
        self._patched_call = None
        self._closed = False
        _GDN_PATCH_LOCK.acquire()
        try:
            self._patch()
        except Exception:
            _GDN_PATCH_LOCK.release()
            raise

    def _patch(self):
        from mlx_lm.models.qwen3_5 import GatedDeltaNet
        self._gdn_cls = GatedDeltaNet
        self._orig_call = GatedDeltaNet.__call__
        capture = self

        def _capturing_gdn_call(self_layer, inputs, mask=None, cache=None):
            B, S, _ = inputs.shape
            if self_layer.sharding_group is not None:
                from mlx_lm.models.qwen3_5 import sum_gradients
                inputs = sum_gradients(self_layer.sharding_group)(inputs)
            qkv = self_layer.in_proj_qkv(inputs)
            z = self_layer.in_proj_z(inputs).reshape(B, S, self_layer.num_v_heads, self_layer.head_v_dim)
            b, a = self_layer.in_proj_b(inputs), self_layer.in_proj_a(inputs)
            conv_state = cache[0] if (cache is not None and cache[0] is not None) else mx.zeros((B, self_layer.conv_kernel_size - 1, self_layer.conv_dim), dtype=inputs.dtype)
            if mask is not None:
                qkv = mx.where(mask[..., None], qkv, 0)
            conv_input = mx.concatenate([conv_state, qkv], axis=1)
            capture.conv_data.append((conv_input, self_layer.conv_kernel_size))
            if cache is not None:
                cache[0] = conv_input[:, -(self_layer.conv_kernel_size - 1):]
            conv_out = nn.silu(self_layer.conv1d(conv_input))
            q, k, v = [
                t.reshape(B, S, h, d)
                for t, h, d in zip(
                    mx.split(conv_out, [self_layer.key_dim, 2 * self_layer.key_dim], -1),
                    [self_layer.num_k_heads, self_layer.num_k_heads, self_layer.num_v_heads],
                    [self_layer.head_k_dim, self_layer.head_k_dim, self_layer.head_v_dim],
                )
            ]
            state = cache[1] if cache else None
            inv_scale = k.shape[-1] ** -0.5
            q = (inv_scale ** 2) * mx.fast.rms_norm(q, None, 1e-6)
            k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
            capture._gdn_inputs.append((q, k, v, a, b, self_layer.A_log, self_layer.dt_bias, state, mask))
            out, new_state = _gd_mod.gated_delta_update(
                q, k, v, a, b, self_layer.A_log, self_layer.dt_bias, state, mask, use_kernel=True
            )
            if cache is not None:
                cache[1] = new_state
            out = self_layer.norm(out, z)
            out = self_layer.out_proj(out.reshape(B, S, -1))
            if self_layer.sharding_group is not None:
                out = mx.distributed.all_sum(out, group=self_layer.sharding_group)
            return out

        self._patched_call = _capturing_gdn_call
        GatedDeltaNet.__call__ = _capturing_gdn_call

    def clear(self):
        self.conv_data.clear()
        self._gdn_inputs.clear()

    def close(self):
        if self._closed:
            return
        try:
            if self._gdn_cls is not None and self._gdn_cls.__call__ is self._patched_call:
                self._gdn_cls.__call__ = self._orig_call
        finally:
            self._closed = True
            self._gdn_cls = None
            self._orig_call = None
            self._patched_call = None
            _GDN_PATCH_LOCK.release()

    def rollback(self, cache, accepted, trim):
        n_non_trimmable = sum(1 for c in cache if not c.is_trimmable())
        assert n_non_trimmable == len(self._gdn_inputs), (
            f"non-trimmable cache count ({n_non_trimmable}) != "
            f"captured GDN inputs ({len(self._gdn_inputs)}); "
            "DFlash MLX rollback assumes every non-trimmable cache is a GatedDeltaNet layer"
        )
        j = 0
        for c in cache:
            if c.is_trimmable():
                c.trim(trim)
            else:
                q, k, v, a, b, A_log, dt_bias, init_state, mask = self._gdn_inputs[j]
                n = accepted + 1
                _, state = _gd_mod.gated_delta_update(
                    q[:, :n], k[:, :n], v[:, :n], a[:, :n], b[:, :n],
                    A_log, dt_bias, init_state,
                    None if mask is None else mask[:, :n],
                    use_kernel=True,
                )
                c.cache[1] = state
                conv_input, K = self.conv_data[j]
                c.cache[0] = conv_input[:, accepted + 1 : accepted + K]
                j += 1


@dataclass
class GenerationResponse:
    text: str
    tokens: List[int]
    accepted: int
    prompt_tokens: int
    prefill_seconds: float
    reused_prefix_tokens: int
    prompt_tps: float
    generation_tokens: int
    decode_seconds: float
    generation_tps: float
    peak_memory: float
    prefill_hidden_bytes: int = 0
    prefill_target_cache_bytes: int = 0
    prefill_logits_bytes: int = 0
    prefill_working_set_bytes: int = 0
    prompt_cache_state_bytes: int = 0
    speculative_steps: int = 0
    proposed_tokens: int = 0
    accepted_tokens: int = 0
    avg_acceptance_length: float = 0.0
    avg_acceptance_ratio: float = 0.0
    acceptance_lengths: Tuple[int, ...] = ()
    acceptance_ratios: Tuple[float, ...] = ()
    block_size_history: Tuple[int, ...] = ()
    adaptive_block_size: bool = False
    finish_reason: Optional[str] = None
    prefill_state: Optional["PromptPrefillState"] = None


@dataclass
class PromptPrefillState:
    prompt_tokens: Tuple[int, ...]
    target_cache: List[Any]
    hidden: mx.array
    last_logits: Optional[mx.array] = None


@dataclass(frozen=True)
class AdaptiveBlockSizeConfig:
    enabled: bool = False
    min_block_size: int = 4
    max_block_size: int = 15
    grow_threshold: float = 0.9
    shrink_threshold: float = 0.55
    grow_step: int = 1
    shrink_step: int = 1
    # Hysteresis: require this many consecutive samples above/below the
    # thresholds before we actually change block size. Prevents ping-pong
    # when the acceptance ratio oscillates right around a threshold.
    # Default 1 = legacy no-hysteresis behavior; raise to 3 for smoother
    # long-running sessions.
    grow_streak: int = 1
    shrink_streak: int = 1


@dataclass
class PromptPrefillRun:
    prompt: mx.array
    prompt_tokens: List[int]
    target_cache: List[Any]
    hidden: mx.array
    logits: mx.array
    prompt_tps: float
    prefill_seconds: float
    reused_prefix_tokens: int
    hidden_bytes: int = 0
    target_cache_bytes: int = 0
    logits_bytes: int = 0
    working_set_bytes: int = 0
    prefill_state_bytes: int = 0
    prefill_state: Optional[PromptPrefillState] = None


def _make_response(
    text,
    tokens,
    accepted,
    prompt_size,
    prefill_seconds,
    reused_prefix_tokens,
    prompt_tps,
    n,
    tic,
    finish_reason=None,
    prefill_state=None,
    speculative_steps=0,
    proposed_tokens=0,
    accepted_tokens=0,
    acceptance_lengths=(),
    acceptance_ratios=(),
    block_size_history=(),
    adaptive_block_size=False,
    snapshot_histories=True,
    prefill_hidden_bytes=0,
    prefill_target_cache_bytes=0,
    prefill_logits_bytes=0,
    prefill_working_set_bytes=0,
    prompt_cache_state_bytes=0,
):
    decode_seconds = max(time.perf_counter() - tic, 1e-9)
    generation_tps = n / decode_seconds
    avg_acceptance_length = (
        accepted_tokens / speculative_steps if speculative_steps > 0 else 0.0
    )
    avg_acceptance_ratio = (
        accepted_tokens / proposed_tokens if proposed_tokens > 0 else 0.0
    )
    return GenerationResponse(
        text=text,
        tokens=tokens,
        accepted=accepted,
        prompt_tokens=prompt_size,
        prefill_seconds=prefill_seconds,
        reused_prefix_tokens=reused_prefix_tokens,
        prompt_tps=prompt_tps,
        generation_tokens=n,
        decode_seconds=decode_seconds,
        generation_tps=generation_tps,
        peak_memory=mx.get_peak_memory() / 1e9,
        speculative_steps=speculative_steps,
        proposed_tokens=proposed_tokens,
        accepted_tokens=accepted_tokens,
        avg_acceptance_length=avg_acceptance_length,
        avg_acceptance_ratio=avg_acceptance_ratio,
        acceptance_lengths=tuple(acceptance_lengths) if snapshot_histories else (),
        acceptance_ratios=tuple(acceptance_ratios) if snapshot_histories else (),
        block_size_history=tuple(block_size_history) if snapshot_histories else (),
        adaptive_block_size=adaptive_block_size,
        finish_reason=finish_reason,
        prefill_hidden_bytes=prefill_hidden_bytes,
        prefill_target_cache_bytes=prefill_target_cache_bytes,
        prefill_logits_bytes=prefill_logits_bytes,
        prefill_working_set_bytes=prefill_working_set_bytes,
        prompt_cache_state_bytes=prompt_cache_state_bytes,
        prefill_state=prefill_state,
    )


def tokenize_prompt(tokenizer, prompt):
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    if isinstance(prompt, mx.array):
        return prompt

    if isinstance(prompt, str):
        add_special_tokens = tokenizer.bos_token is None or not prompt.startswith(tokenizer.bos_token)
        prompt = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)

    return mx.array(prompt)


def clone_prefill_state_for_reuse(prefill_state: PromptPrefillState) -> PromptPrefillState:
    return PromptPrefillState(
        prompt_tokens=tuple(prefill_state.prompt_tokens),
        target_cache=copy.deepcopy(prefill_state.target_cache),
        hidden=prefill_state.hidden,
        last_logits=prefill_state.last_logits,
    )


def _snapshot_prefill_state(prompt_tokens, target_cache, hidden, logits):
    return PromptPrefillState(
        prompt_tokens=tuple(prompt_tokens),
        target_cache=copy.deepcopy(target_cache),
        hidden=hidden,
        last_logits=None if logits is None else copy.deepcopy(logits[:, -1:]),
    )


def derive_prefill_prefix_state(
    prefill_state: PromptPrefillState,
    prefix_length: int,
) -> Optional[PromptPrefillState]:
    if prefix_length <= 0:
        return None
    total_tokens = len(prefill_state.prompt_tokens)
    if prefix_length > total_tokens:
        raise ValueError(f"prefix_length={prefix_length} exceeds prompt length={total_tokens}")
    if prefix_length == total_tokens:
        return clone_prefill_state_for_reuse(prefill_state)

    target_cache = copy.deepcopy(prefill_state.target_cache)
    if not can_trim_prompt_cache(target_cache):
        return None

    trim_prompt_cache(target_cache, total_tokens - prefix_length)
    return PromptPrefillState(
        prompt_tokens=prefill_state.prompt_tokens[:prefix_length],
        target_cache=target_cache,
        hidden=copy.deepcopy(prefill_state.hidden[:, :prefix_length, :]),
        last_logits=None,
    )


def _match_reusable_prefix(
    prompt_tokens: List[int],
    prefix_state: Optional[PromptPrefillState],
) -> tuple[Optional[PromptPrefillState], int]:
    if prefix_state is None:
        return None, 0

    cached_tokens = prefix_state.prompt_tokens
    cached_len = len(cached_tokens)
    if cached_len > len(prompt_tokens):
        return None, 0
    if tuple(prompt_tokens[:cached_len]) != cached_tokens:
        return None, 0
    if cached_len == len(prompt_tokens) and prefix_state.last_logits is None:
        return None, 0
    return clone_prefill_state_for_reuse(prefix_state), cached_len


def prefill_prompt(
    model,
    tokenizer,
    prompt,
    *,
    target_turboquant_bits: Optional[float] = None,
    prefix_state: Optional[PromptPrefillState] = None,
    capture_prefill_state: bool = False,
) -> PromptPrefillRun:
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    prompt = tokenize_prompt(tokenizer, prompt)
    prompt_tokens = prompt.tolist()
    reusable_prefix, reusable_prefix_tokens = _match_reusable_prefix(prompt_tokens, prefix_state)
    target_cache = (
        reusable_prefix.target_cache
        if reusable_prefix is not None
        else _make_target_cache(model, turboquant_bits=target_turboquant_bits)
    )

    tic = time.perf_counter()
    if reusable_prefix is None:
        _clear_model_hidden_states(model)
        with mx.stream(generation_stream):
            logits = model(prompt[None], target_cache)
            try:
                hidden = mx.concatenate(model._hidden_states, axis=-1)
            finally:
                _clear_model_hidden_states(model)
        mx.eval(logits, hidden)
    elif reusable_prefix_tokens < prompt.size:
        suffix = prompt[reusable_prefix_tokens:]
        _clear_model_hidden_states(model)
        with mx.stream(generation_stream):
            logits = model(suffix[None], target_cache)
            try:
                suffix_hidden = mx.concatenate(model._hidden_states, axis=-1)
            finally:
                _clear_model_hidden_states(model)
        hidden = mx.concatenate([reusable_prefix.hidden, suffix_hidden], axis=1)
        suffix_hidden = None
        mx.eval(logits, hidden)
    else:
        logits = reusable_prefix.last_logits
        hidden = reusable_prefix.hidden

    prefill_seconds = max(time.perf_counter() - tic, 1e-9)
    prompt_tps = prompt.size / prefill_seconds
    prefill_state = (
        _snapshot_prefill_state(prompt_tokens, target_cache, hidden, logits)
        if capture_prefill_state
        else None
    )
    hidden_bytes = estimate_memory_bytes(hidden)
    target_cache_bytes = estimate_memory_bytes(target_cache)
    logits_bytes = estimate_memory_bytes(logits)
    prefill_state_bytes = estimate_prefill_state_bytes(prefill_state)
    return PromptPrefillRun(
        prompt=prompt,
        prompt_tokens=prompt_tokens,
        target_cache=target_cache,
        hidden=hidden,
        logits=logits,
        prompt_tps=prompt_tps,
        prefill_seconds=prefill_seconds,
        reused_prefix_tokens=reusable_prefix_tokens,
        hidden_bytes=hidden_bytes,
        target_cache_bytes=target_cache_bytes,
        logits_bytes=logits_bytes,
        working_set_bytes=hidden_bytes + target_cache_bytes + logits_bytes,
        prefill_state_bytes=prefill_state_bytes,
        prefill_state=prefill_state,
    )


def _clamp_block_size(value: int, *, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


def _accepted_tokens_from_cpu_batches(
    draft_tokens: list[int],
    target_tokens: list[int],
) -> tuple[int, list[int]]:
    if not target_tokens:
        raise ValueError("target_tokens must not be empty")

    limit = min(len(draft_tokens), max(len(target_tokens) - 1, 0))
    accepted = 0
    while accepted < limit and draft_tokens[accepted] == target_tokens[accepted]:
        accepted += 1
    return accepted, draft_tokens[:accepted] + [target_tokens[accepted]]


def _acceptance_prefix_length(draft_tokens, target_tokens) -> int:
    draft_values = draft_tokens.tolist() if hasattr(draft_tokens, "tolist") else list(draft_tokens)
    target_values = target_tokens.tolist() if hasattr(target_tokens, "tolist") else list(target_tokens)
    if draft_values and isinstance(draft_values[0], (list, tuple)):
        draft_values = draft_values[0]
    if target_values and isinstance(target_values[0], (list, tuple)):
        target_values = target_values[0]
    return _accepted_tokens_from_cpu_batches(
        [int(token_id) for token_id in draft_values],
        [int(token_id) for token_id in target_values],
    )[0]


def next_adaptive_block_size(
    current_block_size: int,
    acceptance_length: int,
    proposal_tokens: int,
    config: Optional[AdaptiveBlockSizeConfig],
    hysteresis_state: Optional[dict] = None,
) -> int:
    if config is None or not config.enabled:
        return current_block_size

    current_block_size = _clamp_block_size(
        current_block_size,
        minimum=max(1, config.min_block_size),
        maximum=max(1, config.max_block_size),
    )
    ratio = acceptance_length / max(proposal_tokens, 1)
    grow_streak = max(1, int(config.grow_streak or 1))
    shrink_streak = max(1, int(config.shrink_streak or 1))

    state = hysteresis_state if hysteresis_state is not None else {}
    grow_count = int(state.get("grow", 0))
    shrink_count = int(state.get("shrink", 0))

    if ratio >= config.grow_threshold and acceptance_length >= proposal_tokens:
        grow_count += 1
        shrink_count = 0
    elif ratio <= config.shrink_threshold:
        shrink_count += 1
        grow_count = 0
    else:
        grow_count = 0
        shrink_count = 0

    new_block_size = current_block_size
    if grow_count >= grow_streak:
        new_block_size = _clamp_block_size(
            current_block_size + max(1, config.grow_step),
            minimum=max(1, config.min_block_size),
            maximum=max(1, config.max_block_size),
        )
        grow_count = 0
    elif shrink_count >= shrink_streak:
        new_block_size = _clamp_block_size(
            current_block_size - max(1, config.shrink_step),
            minimum=max(1, config.min_block_size),
            maximum=max(1, config.max_block_size),
        )
        shrink_count = 0

    state["grow"] = grow_count
    state["shrink"] = shrink_count
    return new_block_size


def _apply_logits_processors(
    processors: List[Any],
    logits: mx.array,
    tokens_context: List[int],
) -> mx.array:
    if not processors:
        return logits
    if logits.ndim == 3:
        batch, seq_len, vocab = logits.shape
        reshaped = logits.reshape(batch * seq_len, vocab)
        for proc in processors:
            reshaped = proc(tokens_context, reshaped)
        return reshaped.reshape(batch, seq_len, vocab)
    for proc in processors:
        logits = proc(tokens_context, logits)
    return logits


def stream_generate(
    model, draft, tokenizer, prompt,
    block_size=None, max_tokens=256, temperature=0.0, sampler=None,
    top_p: float = 0.0,
    top_k: int = 0,
    min_p: float = 0.0,
    presence_penalty: float = 0.0,
    repetition_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    presence_context_size: int = 20,
    repetition_context_size: int = 20,
    target_turboquant_bits: Optional[float] = None,
    prefix_state: Optional[PromptPrefillState] = None,
    capture_prefill_state: bool = False,
    adaptive_block_size: Optional[AdaptiveBlockSizeConfig] = None,
    should_stop: Optional[Any] = None,
):
    _patch_model(model, draft.config.target_layer_ids)
    block_size = block_size if block_size is not None else int(draft.config.block_size)
    if sampler is None:
        sampler = make_sampler(
            temp=temperature,
            top_p=float(top_p or 0.0),
            min_p=float(min_p or 0.0),
            top_k=int(top_k or 0),
        )
    rep_pen = float(repetition_penalty or 0.0)
    pres_pen = float(presence_penalty or 0.0)
    freq_pen = float(frequency_penalty or 0.0)
    has_rep = rep_pen and abs(rep_pen - 1.0) > 1e-9
    processors: List[Any] = []
    if has_rep or pres_pen or freq_pen:
        processors = make_logits_processors(
            repetition_penalty=rep_pen if has_rep else None,
            repetition_context_size=int(repetition_context_size or 20),
            presence_penalty=pres_pen if pres_pen else None,
            presence_context_size=int(presence_context_size or 20),
            frequency_penalty=freq_pen if freq_pen else None,
            frequency_context_size=int(presence_context_size or 20),
        )

    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    detokenizer = tokenizer.detokenizer
    mask_id = int(draft.config.mask_token_id)
    prefill = prefill_prompt(
        model,
        tokenizer,
        prompt,
        target_turboquant_bits=target_turboquant_bits,
        prefix_state=prefix_state,
        capture_prefill_state=capture_prefill_state,
    )
    prompt = prefill.prompt
    prompt_tokens = prefill.prompt_tokens
    tokens = list(prompt_tokens)
    target_cache = prefill.target_cache
    draft_cache = make_prompt_cache(draft)
    draft.bind(model)
    _target_can_trim = can_trim_prompt_cache(target_cache)
    if not _target_can_trim and not _HAS_GDN:
        raise RuntimeError(
            "This MLX model requires gated-delta rollback support, but "
            "mlx_lm.models.gated_delta is unavailable."
        )
    _capture = _GDNStateCapture() if not _target_can_trim else None

    try:
        logits = prefill.logits
        hidden = prefill.hidden
        prompt_tps = prefill.prompt_tps
        prompt_state = prefill.prefill_state

        decode_tic = time.perf_counter()
        first_logits = _apply_logits_processors(processors, logits[:, -1:], tokens)
        token = int(sampler(first_logits)[0, 0].item())
        prefill.logits = None
        logits = None
        first_logits = None
        tokens.append(token)
        n = 1
        current_block_size = int(block_size)
        proposal_history: list[int] = []
        acceptance_lengths: list[int] = []
        acceptance_ratios: list[float] = []
        _adaptive_hysteresis: dict = {"grow": 0, "shrink": 0}
        speculative_steps = 0
        proposed_tokens = 0
        accepted_tokens = 0

        if token in tokenizer.eos_token_ids:
            detokenizer.add_token(token)
            detokenizer.finalize()
            yield _make_response(
                detokenizer.last_segment,
                [token],
                1,
                prompt.size,
                prefill.prefill_seconds,
                prefill.reused_prefix_tokens,
                prompt_tps,
                n,
                decode_tic,
                "stop",
                prompt_state,
                speculative_steps=0,
                proposed_tokens=0,
                accepted_tokens=0,
                acceptance_lengths=(),
                acceptance_ratios=(),
                block_size_history=(),
                adaptive_block_size=bool(adaptive_block_size and adaptive_block_size.enabled),
                prefill_hidden_bytes=prefill.hidden_bytes,
                prefill_target_cache_bytes=prefill.target_cache_bytes,
                prefill_logits_bytes=prefill.logits_bytes,
                prefill_working_set_bytes=prefill.working_set_bytes,
                prompt_cache_state_bytes=prefill.prefill_state_bytes,
                snapshot_histories=True,
            )
            return

        detokenizer.add_token(token)
        yield _make_response(
            detokenizer.last_segment,
            [token],
            1,
            prompt.size,
            prefill.prefill_seconds,
            prefill.reused_prefix_tokens,
            prompt_tps,
            n,
            decode_tic,
            prefill_state=prompt_state,
            speculative_steps=speculative_steps,
            proposed_tokens=proposed_tokens,
            accepted_tokens=accepted_tokens,
            acceptance_lengths=acceptance_lengths,
            acceptance_ratios=acceptance_ratios,
            block_size_history=proposal_history,
            adaptive_block_size=bool(adaptive_block_size and adaptive_block_size.enabled),
            prefill_hidden_bytes=prefill.hidden_bytes,
            prefill_target_cache_bytes=prefill.target_cache_bytes,
            prefill_logits_bytes=prefill.logits_bytes,
            prefill_working_set_bytes=prefill.working_set_bytes,
            prompt_cache_state_bytes=prefill.prefill_state_bytes,
            snapshot_histories=False,
        )

        while n < max_tokens:
            if should_stop is not None:
                try:
                    if should_stop():
                        break
                except Exception:
                    pass
            bs = min(current_block_size, max_tokens - n + 1)
            if bs <= 1:
                break
            proposal_history.append(bs)

            with mx.stream(generation_stream):
                block = mx.array([[tokens[-1]] + [mask_id] * (bs - 1)])
                draft_logits = draft(block, hidden, draft_cache)
                if (
                    draft.config.sliding_window_size is None and
                    (trim_n := draft_cache[0].offset - (prompt.size + n - 1)) > 0
                ):
                    trim_prompt_cache(draft_cache, trim_n)
                draft_tokens = sampler(
                    _apply_logits_processors(processors, draft_logits[:, 1 - bs:], tokens)
                )

            if _capture is not None:
                _capture.clear()
            _clear_model_hidden_states(model)
            with mx.stream(generation_stream):
                verify_input = mx.concatenate([mx.array([[tokens[-1]]]), draft_tokens], axis=1)
                logits = model(verify_input, target_cache)
                try:
                    hidden = mx.concatenate(model._hidden_states, axis=-1)
                finally:
                    _clear_model_hidden_states(model)
                target_tokens = sampler(_apply_logits_processors(processors, logits, tokens))
            mx.eval(draft_tokens, target_tokens, hidden)

            draft_token_values = [int(token_id) for token_id in draft_tokens[0].tolist()]
            target_token_values = [int(token_id) for token_id in target_tokens[0].tolist()]
            accepted, new_tokens = _accepted_tokens_from_cpu_batches(
                draft_token_values,
                target_token_values,
            )
            new_tokens = new_tokens[:max_tokens - n]
            accepted_length = len(new_tokens)
            speculative_steps += 1
            proposed_tokens += bs
            accepted_tokens += accepted_length
            acceptance_lengths.append(accepted_length)
            acceptance_ratios.append(accepted_length / max(bs, 1))

            eos_idx = next((i for i, t in enumerate(new_tokens) if t in tokenizer.eos_token_ids), None)
            if eos_idx is not None:
                new_tokens = new_tokens[:eos_idx + 1]
                for t in new_tokens:
                    detokenizer.add_token(t)
                detokenizer.finalize()
                tokens.extend(new_tokens)
                n += len(new_tokens)
                yield _make_response(
                    detokenizer.last_segment,
                    new_tokens,
                    accepted + 1,
                    prompt.size,
                    prefill.prefill_seconds,
                    prefill.reused_prefix_tokens,
                    prompt_tps,
                    n,
                    decode_tic,
                    "stop",
                    prompt_state,
                    speculative_steps=speculative_steps,
                    proposed_tokens=proposed_tokens,
                    accepted_tokens=accepted_tokens,
                    acceptance_lengths=acceptance_lengths,
                    acceptance_ratios=acceptance_ratios,
                    block_size_history=proposal_history,
                    adaptive_block_size=bool(adaptive_block_size and adaptive_block_size.enabled),
                    prefill_hidden_bytes=prefill.hidden_bytes,
                    prefill_target_cache_bytes=prefill.target_cache_bytes,
                    prefill_logits_bytes=prefill.logits_bytes,
                    prefill_working_set_bytes=prefill.working_set_bytes,
                    prompt_cache_state_bytes=prefill.prefill_state_bytes,
                    snapshot_histories=True,
                )
                return

            for t in new_tokens:
                detokenizer.add_token(t)
            tokens.extend(new_tokens)
            n += len(new_tokens)

            yield _make_response(
                detokenizer.last_segment,
                new_tokens,
                accepted + 1,
                prompt.size,
                prefill.prefill_seconds,
                prefill.reused_prefix_tokens,
                prompt_tps,
                n,
                decode_tic,
                prefill_state=prompt_state,
                speculative_steps=speculative_steps,
                proposed_tokens=proposed_tokens,
                accepted_tokens=accepted_tokens,
                acceptance_lengths=acceptance_lengths,
                acceptance_ratios=acceptance_ratios,
                block_size_history=proposal_history,
                adaptive_block_size=bool(adaptive_block_size and adaptive_block_size.enabled),
                prefill_hidden_bytes=prefill.hidden_bytes,
                prefill_target_cache_bytes=prefill.target_cache_bytes,
                prefill_logits_bytes=prefill.logits_bytes,
                prefill_working_set_bytes=prefill.working_set_bytes,
                prompt_cache_state_bytes=prefill.prefill_state_bytes,
                snapshot_histories=False,
            )

            trim = bs - accepted - 1
            if trim > 0:
                if _target_can_trim:
                    trim_prompt_cache(target_cache, trim)
                elif _capture is not None:
                    _capture.rollback(target_cache, accepted, trim)
            hidden = hidden[:, :accepted + 1, :]
            current_block_size = next_adaptive_block_size(
                current_block_size,
                accepted_length,
                bs,
                adaptive_block_size,
                hysteresis_state=_adaptive_hysteresis,
            )

        detokenizer.finalize()
        yield _make_response(
            detokenizer.last_segment,
            [],
            0,
            prompt.size,
            prefill.prefill_seconds,
            prefill.reused_prefix_tokens,
            prompt_tps,
            n,
            decode_tic,
            "length",
            prompt_state,
            speculative_steps=speculative_steps,
            proposed_tokens=proposed_tokens,
            accepted_tokens=accepted_tokens,
            acceptance_lengths=acceptance_lengths,
            acceptance_ratios=acceptance_ratios,
            block_size_history=proposal_history,
            adaptive_block_size=bool(adaptive_block_size and adaptive_block_size.enabled),
            prefill_hidden_bytes=prefill.hidden_bytes,
            prefill_target_cache_bytes=prefill.target_cache_bytes,
            prefill_logits_bytes=prefill.logits_bytes,
            prefill_working_set_bytes=prefill.working_set_bytes,
            prompt_cache_state_bytes=prefill.prefill_state_bytes,
            snapshot_histories=True,
        )
    finally:
        prefill.logits = None
        prefill.hidden = None
        prefill.target_cache = []
        target_cache = None
        draft_cache = None
        hidden = None
        logits = None
        _clear_model_hidden_states(model)
        if _capture is not None:
            _capture.close()
