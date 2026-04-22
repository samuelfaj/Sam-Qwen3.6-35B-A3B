from __future__ import annotations

import copy
import unittest

import mlx.core as mx

from dflash import model_mlx


class _FakeCache:
    def __init__(self, offset: int) -> None:
        self.offset = offset

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        n = min(self.offset, n)
        self.offset -= n
        return n


class ModelMlxTests(unittest.TestCase):
    def test_stable_turboquant_kv_cache_supports_deepcopy(self):
        try:
            from mlx_turboquant.cache import TurboQuantKVCache
        except ImportError:
            self.skipTest("mlx-turboquant is not installed")

        wrapped = model_mlx._StableTurboQuantKVCache(
            TurboQuantKVCache(bits=4.0, head_dim=128, key_seed=42, value_seed=43)
        )

        copied = copy.deepcopy(wrapped)

        self.assertIsInstance(copied, model_mlx._StableTurboQuantKVCache)
        self.assertIsNot(copied, wrapped)
        self.assertIsNot(copied._inner, wrapped._inner)

    def test_next_adaptive_block_size_grows_and_shrinks(self):
        config = model_mlx.AdaptiveBlockSizeConfig(
            enabled=True,
            min_block_size=8,
            max_block_size=18,
            grow_threshold=0.95,
            shrink_threshold=0.6,
        )

        grown = model_mlx.next_adaptive_block_size(10, acceptance_length=10, proposal_tokens=10, config=config)
        shrunk = model_mlx.next_adaptive_block_size(10, acceptance_length=4, proposal_tokens=10, config=config)
        unchanged = model_mlx.next_adaptive_block_size(10, acceptance_length=8, proposal_tokens=10, config=config)

        self.assertEqual(grown, 11)
        self.assertEqual(shrunk, 9)
        self.assertEqual(unchanged, 10)

    def test_derive_prefill_prefix_state_trims_cache_and_hidden(self):
        state = model_mlx.PromptPrefillState(
            prompt_tokens=(10, 20, 30, 40),
            target_cache=[_FakeCache(offset=4)],
            hidden=mx.array([[[1], [2], [3], [4]]]),
            last_logits=mx.array([[[9]]]),
        )

        derived = model_mlx.derive_prefill_prefix_state(state, 2)

        self.assertIsNotNone(derived)
        assert derived is not None
        self.assertEqual(derived.prompt_tokens, (10, 20))
        self.assertEqual(derived.target_cache[0].offset, 2)
        self.assertEqual(derived.hidden.shape, (1, 2, 1))
        self.assertIsNone(derived.last_logits)


if __name__ == "__main__":
    unittest.main()
