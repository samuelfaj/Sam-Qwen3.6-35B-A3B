# TODO: Rapid-MLX Performance Gap Analysis for DFlash MLX

This document captures what Rapid-MLX appears to do for speed, what this repo
already does, what is missing here, and the concrete implementation plan to
close the gap.

Scope:

- Local repo: `/Users/samuelfajreldines/dev/dflash`
- Local serving path: `scripts/local_api_server.py`
- Local model path: `dflash/model_mlx.py`
- Local DDTree path: `dflash/ddtree_engine.py`
- External reference: <https://github.com/raullenchai/Rapid-MLX>
- Main question: what Rapid-MLX does to be faster that this repo is not using
  yet.

## Executive Summary

The biggest missing Rapid-MLX-style speed feature in this repo is not
TurboQuant or speculative decoding. This repo already has those ideas in some
form. The biggest gap is server architecture:

1. Continuous batching.
2. More general memory-aware/paged prompt cache.
3. Chunked prefill with mid-prefill snapshots.
4. Generic KV cache quantization for stored prefix entries.
5. Tool-call logits bias and recovery to reduce agent retries.

This repo already has:

- DFlash speculative generation.
- Experimental DDTree MLX generation.
- TurboQuant support for compatible target cache layers.
- Global/stable prefix cache.
- GatedDeltaNet/GDN state capture and rollback helpers.
- Streaming wrapper behavior and heartbeat-oriented serving.

But the local server currently serializes generation turns one request at a
time. That means it can be fast for one request but will waste throughput under
concurrent agent load. Rapid-MLX uses a vLLM-like engine/scheduler on top of
MLX BatchGenerator, so multiple active requests can share decode steps.

Highest-value roadmap:

1. Add continuous batching for the non-DDTree path.
2. Add chunked prefill and mid-prefill cache snapshots.
3. Replace/augment current global prefix cache with memory-aware exact/partial
   prefix matching.
4. Add optional paged cache/COW after cache semantics are proven.
5. Add tool logits bias/recovery as an agent-workflow accelerator.

## Current Local State

### Local DFlash Generation

Main file:

- `dflash/model_mlx.py`

Relevant symbols:

- `stream_generate(...)`
  - Single-request generation loop.
  - Calls `prefill_prompt(...)`.
  - Uses DFlash draft model to propose tokens.
  - Verifies proposals through target model.
  - Computes accepted token prefix.
  - Trims or rolls back target cache after acceptance.
  - Supports adaptive block size.
  - Yields metrics with accepted/proposed token counts.

- `prefill_prompt(...)`
  - Builds/reuses target cache.
  - If prefix cache state matches, prefill only suffix.
  - If no reusable prefix, runs target model over full prompt.
  - Does not implement scheduler-level chunked prefill.
  - Does not save mid-prefill snapshots.

- `PromptPrefillState`
  - Stores prompt tokens, target cache, hidden states, logits, and memory stats.
  - Used for prefix reuse.

- `derive_prefill_prefix_state(...)`
  - Trims a prefill state down to a prefix.
  - Useful for prefix cache reuse.

- `_match_reusable_prefix(...)`
  - Matches cached prefix tokens against current prompt tokens.

- `AdaptiveBlockSizeConfig`
  - Grows/shrinks speculative block size based on acceptance ratio.

### Local DDTree Generation

Main file:

- `dflash/ddtree_engine.py`

Relevant behavior:

- Builds a tree from draft logits.
- Verifies candidate tree against target model.
- Commits accepted path through tree-aware cache commit when possible.
- Falls back to slow snapshot/rollback commit when tree-aware commit is not
  available.
- Tracks acceptance ratio, tree node count, emitted tokens, cache strategy, and
  prompt prefix reuse.

This is already a meaningful speculative algorithm. It is not the Rapid-MLX
continuous batching architecture.

### Local Serving Path

Main file:

- `scripts/local_api_server.py`

Relevant behavior:

- `LocalModelServer` owns model/draft loading and generation settings.
- `_acquire_generation_turn(...)` serializes generation turns through a ticket
  system and `Condition`.
- Prefix cache fields exist:
  - response prefix states
  - global prefix states
  - stable prefix token cache
  - byte limits
  - hit/miss counters
  - pruning helpers
- `_select_prefix_state_locked(...)` selects reusable prefix state.
- `_remember_global_prefix_state_locked(...)` stores prefix state after
  generation.
- The server has many compatibility shims for OpenAI/Anthropic-style APIs,
  tool parsing, event streaming, metrics, keep-alive, and memory settings.

Key limitation:

- Requests are protected by server-level generation turn control. This avoids
  cache corruption and simplifies streaming, but it prevents throughput gains
  from multi-request decode batching.

## Rapid-MLX Speed Features

Rapid-MLX claims/implements the following relevant speed mechanisms:

- Continuous batching through `EngineCore` and `Scheduler`.
- Use of `mlx_lm.generate.BatchGenerator`.
- Prefix cache across requests.
- Trie/LRU prefix cache with exact, shorter-prefix, and longer-prefix trimming.
- Memory-aware prefix cache with memory-based eviction.
- Paged cache with fixed token blocks, ref counts, COW, hash lookup, and
  pinning.
- Chunked prefill to prevent long prompt prefill from starving active decode.
- Mid-prefill cache save so interrupted or repeated long prompts can reuse
  partially computed cache.
- DeltaNet/GatedDeltaNet state snapshots for hybrid RNN+attention models.
- TurboQuant V-cache compression.
- Generic KV cache quantization for prefix cache memory reduction.
- Tool logits bias for structured tool-call tokens.
- Tool-call recovery/parsers to convert broken text output back into structured
  tool calls.
- Optional cloud routing for requests where local prefill would be too slow.

Important: not all of these are equally relevant to this repo. The local fork is
focused on Qwen3.6/DFlash local serving, not a generic multimodal/cloud engine.

## Gap Matrix

| Area | Rapid-MLX | Local repo | Gap |
| --- | --- | --- | --- |
| Continuous batching | EngineCore + Scheduler + BatchGenerator | Server serializes generation turns | Major |
| Decode throughput under concurrency | Multiple requests advance together | One active generation turn | Major |
| Prompt cache lookup | Exact/shorter/longer match, trie/LRU | Custom global/stable prefix cache | Medium |
| Memory-aware prefix cache | Memory limit, LRU eviction by bytes | Byte limits exist, custom state cache | Medium |
| Paged cache | Block refcount/COW/hash/pinning | Not present | Medium/high |
| Chunked prefill | Scheduler monkeypatch chunks large prefills | Prefix/suffix reuse only | Major for long prompts |
| Mid-prefill snapshots | Saves cache during long prefill | Not present | Major for interrupted/repeated long prompts |
| TurboQuant | Present | Present | No gap |
| Generic KV quantization | Present for prefix cache | Not present | Medium |
| GDN/DeltaNet snapshots | Present | GDN patch/rollback present | Partial gap: integration maturity |
| Tool logits bias | Present | Not clearly present | Medium for agent reliability |
| Tool recovery/parsers | Broad parser suite | Local parser stack exists, less broad | Medium |
| MTP | Optional if model supports MTP head | Not present | Low/unknown |
| Cloud routing | Present | Not present | Low/non-goal unless desired |

## Priority 1: Continuous Batching

### Problem

Local serving currently serializes generation. This is simple and safe, but it
throws away the core throughput advantage of batching active decode requests.

Rapid-MLX uses a scheduler model:

- waiting queue
- running request set
- request IDs mapped to BatchGenerator UIDs
- one engine loop
- scheduler step advances active requests
- outputs are routed to per-request collectors

This is the biggest likely source of speed difference under real agent traffic.

### Current Local Evidence

Local server:

- `scripts/local_api_server.py`
  - `LocalModelServer`
  - `_acquire_generation_turn(...)`
  - `_generation_turn`
  - `_lock`

Local generation:

- `dflash/model_mlx.py`
  - `stream_generate(...)` handles one prompt/request at a time.

### Target Design

Add a new batched generation engine that can run beside the current path.

Possible module layout:

- `dflash_mlx/batch_engine.py`
- `dflash_mlx/scheduler.py`
- `dflash_mlx/request.py`
- `dflash_mlx/output.py`

Initial goal:

- Batch only the target model non-DDTree path.
- Keep existing DFlash/DDTree path as default until batched path is proven.
- Gate behind CLI/env:
  - `--engine serial|batched`
  - `LOCAL_DFLASH_ENGINE=serial|batched`

Do not start by batching DDTree. DDTree has more complex cache commit semantics.
Batching regular MLX generation first gives useful throughput learning with less
risk.

### Implementation Steps

- [ ] Define `BatchedRequest`.
  - request ID
  - prompt token IDs
  - sampling params
  - max tokens
  - stop tokens/sequences
  - stream flag
  - prefix boundary
  - creation time
  - output state

- [ ] Define `BatchedOutput`.
  - request ID
  - token ID
  - text delta
  - finish reason
  - usage counters
  - cache metrics

- [ ] Create scheduler queues.
  - waiting queue
  - running dict
  - finished set
  - request ID to backend UID map

- [ ] Use `mlx_lm.generate.BatchGenerator` where possible.
  - This gives MLX-native batching semantics.
  - It may not directly support DFlash draft verification.
  - First milestone can batch target-model decode only.

- [ ] Add per-request output collectors.
  - Streaming endpoint needs low-latency per-request deltas.
  - Non-stream endpoint can aggregate collector output until finish.

- [ ] Move server generation calls through engine abstraction.
  - `SerialEngine`: existing behavior.
  - `BatchedEngine`: new scheduler behavior.

- [ ] Preserve current API compatibility.
  - OpenAI chat completions.
  - Anthropic messages.
  - Responses API events.
  - Streaming heartbeat behavior.
  - Tool call normalization.

- [ ] Add graceful abort.
  - Client disconnect aborts request.
  - Finished/aborted request releases cache/output collector state.

- [ ] Add memory pressure control.
  - Periodically check MLX active memory.
  - Clear MLX cache after completed requests or threshold breach.

### Hard Parts

- Different requests may use different sampling params.
  - Option A: batch only requests with compatible sampler params.
  - Option B: recreate BatchGenerator per sampler group.
  - Option C: implement per-request samplers if backend supports it.

- Tool-call constrained decoding may require per-request logits processors.

- Prefix cache must be immutable or copied per request.

- Streaming must not block scheduler progress.

- Existing `RLock` and generation turn must be bypassed only for batched engine.

### Acceptance Criteria

- [ ] With one request, output parity with serial path for deterministic
  temperature `0`.
- [ ] With two concurrent requests, both make progress without waiting for the
  other to finish.
- [ ] p50/p95 latency does not regress badly for one request.
- [ ] Aggregate tokens/sec improves under concurrency.
- [ ] Streaming emits deltas for each request independently.
- [ ] Client disconnect aborts only that request.
- [ ] No cache corruption across 100 concurrent smoke tests.

### Benchmarks

Measure:

- TTFT cold.
- TTFT cached.
- Decode tokens/sec single request.
- Aggregate tokens/sec for 2, 4, 8 concurrent requests.
- p50/p95 time-to-first-token.
- p50/p95 full response latency.
- active memory peak.
- cache hit rate.
- failed requests / cache corruption.

Suggested benchmark scenarios:

- short prompt, 128 output tokens
- long prompt, 128 output tokens
- multi-turn agent prompt with mostly stable prefix
- concurrent Codex-like tool prompt workload

## Priority 2: Chunked Prefill

### Problem

Long prefill can block the engine. In serial mode this just means one slow
request. In batched mode it also means active decode requests can starve while a
new large prompt is being processed.

Rapid-MLX patches/uses BatchGenerator so large prompt prefills are split into
chunks. Between chunks, generation requests can continue.

### Current Local Behavior

`prefill_prompt(...)` reuses prefix state if present. If no prefix state exists,
or if there is a suffix after the cached prefix, it calls the model on that
full remaining sequence.

That is prefix/suffix reuse, not chunked prefill scheduling.

### Target Design

Add chunked prefill with explicit budget.

New config:

- `--prefill-step-size`
- `LOCAL_DFLASH_PREFILL_STEP_SIZE`
- default candidate: `2048`

For serial engine:

- Chunking can still help heartbeat responsiveness and memory pressure.

For batched engine:

- Chunking is required so decode steps can interleave with long prompt prefill.

### Implementation Steps

- [ ] Add config to server CLI/env.
- [ ] Implement chunked prefill helper.
  - input token range
  - existing cache
  - start offset
  - step size
  - callback after each chunk

- [ ] Call target model chunk by chunk.
  - ensure contiguous token arrays
  - call `mx.eval(...)` or `mx.async_eval(...)` at safe points
  - update cache
  - capture hidden state if DFlash needs it

- [ ] Keep hidden-state semantics correct.
  - DFlash draft model needs target hidden states.
  - If hidden states are captured via hooks, chunking must concatenate hidden
    chunks in order.

- [ ] Emit heartbeat between chunks.
  - Streaming clients should not time out during huge prompts.

- [ ] Add abort checks between chunks.
  - If client disconnects, stop prefill and release temporary state.

### Acceptance Criteria

- [ ] Chunked and unchunked prefill produce same next-token logits for a fixed
  prompt within acceptable numeric tolerance.
- [ ] Long prompt streaming emits heartbeat while prefill is running.
- [ ] Abort during prefill does not leak cache state.
- [ ] Memory peak is not worse than unchunked prefill.
- [ ] Batched engine can keep decode requests moving during another request's
  large prefill.

## Priority 3: Mid-Prefill Cache Snapshots

### Problem

Agent workflows often repeat a huge stable prefix and only change the final user
message. If the first long request disconnects or times out mid-prefill, current
work may be lost.

Rapid-MLX stores intermediate prompt cache snapshots during chunked prefill.
Later requests can reuse the already-prefilled prefix.

### Current Local Behavior

Local server stores prefix states after generation/prefill reaches stable
points. It does not appear to save intermediate cache state during a large
prefill loop, because no chunked prefill loop exists yet.

### Target Design

Save prefix cache snapshots while processing long prompts.

New config:

- `--mid-prefill-save-interval`
- `LOCAL_DFLASH_MID_PREFILL_SAVE_INTERVAL`
- default candidate: `8192`
- `0` disables

Save at:

- every configured interval
- known stable prefix boundary
- after tool/schema/system section if detected
- before final user suffix when safe

### Implementation Steps

- [ ] Add prompt boundary tracking.
  - stable system prompt boundary
  - tools schema boundary
  - previous conversation boundary
  - final user message boundary

- [ ] Extract cache snapshots safely.
  - Deep copy mutable cache metadata.
  - Avoid storing cache objects that will be mutated by active generation.

- [ ] Store snapshots keyed by token tuple and model/cache config.
  - model ID
  - target TurboQuant bits
  - tokenizer identity/version
  - prompt token tuple or block hash chain

- [ ] Replace older intermediate snapshot for same request if it is only a
  shorter prefix and memory pressure is high.

- [ ] Expose metrics.
  - mid-prefill snapshots saved
  - mid-prefill cache bytes
  - mid-prefill hit count
  - tokens saved

### Risks

- GDN/DeltaNet layers may not be trivially trimmable.
- Hidden state snapshots can be large.
- Storing too many snapshots can increase memory pressure and hurt performance.
- Snapshotting may force MLX evaluation and spike memory if done carelessly.

### Acceptance Criteria

- [ ] Long prefill interrupted at N tokens leaves a reusable prefix snapshot.
- [ ] Next request with same prefix skips those N tokens.
- [ ] Snapshot memory is counted and evicted under byte limits.
- [ ] No corrupted output after 50 repeated interrupted-prefill tests.

## Priority 4: Memory-Aware Prefix Cache

### Problem

The local global prefix cache is useful but custom and less general than
Rapid-MLX's memory-aware prefix cache. It should become more explicit,
measurable, and robust.

Rapid-style cache supports:

- exact prompt hit
- shorter cached prefix hit
- longer cached prefix trim
- memory-based eviction
- immutable/deep-copied fetch semantics
- hit/miss/tokens-saved stats

### Current Local Behavior

Local server already has:

- `_global_prefix_states`
- `_global_prefix_order`
- `_global_prefix_cache_bytes`
- `_global_prefix_cache_hits`
- `_global_prefix_cache_misses`
- `_stable_prefix_tokens_by_key`
- pruning helpers

This should be preserved but made more general and testable.

### Target Design

Create a real prefix cache module instead of keeping all logic inside
`scripts/local_api_server.py`.

Possible module:

- `dflash_mlx/prefix_cache.py`

Core API:

```python
class PrefixCache:
    def fetch(self, tokens: list[int], config: PrefixCacheConfig) -> PrefixHit | None:
        ...

    def store(self, tokens: list[int], state: PromptPrefillState, metadata: dict) -> bool:
        ...

    def prune(self) -> None:
        ...

    def stats(self) -> dict:
        ...
```

`PrefixHit` should include:

- matched token count
- remaining tokens
- cloned/trimmed `PromptPrefillState`
- source: exact, shorter, longer-trimmed, stable, global, mid-prefill

### Implementation Steps

- [ ] Extract current global prefix logic into a dedicated module.
- [ ] Keep current behavior behind compatibility wrapper.
- [ ] Add exact/shorter/longer prefix matching.
- [ ] Add stable token keying by token tuple/block hash, not only payload-level
  JSON/stable message interpretation.
- [ ] Track memory by MLX array shape/dtype metadata, not expensive `.nbytes`
  when it forces evaluation.
- [ ] Add byte-limit eviction with LRU order.
- [ ] Add pinning for system/tool stable prefix.
- [ ] Add tests for:
  - exact hit
  - shorter prefix hit
  - longer prefix trim hit
  - eviction
  - memory stats
  - mutation isolation after fetch

### Acceptance Criteria

- [ ] Cache fetch never returns a mutable object that corrupts stored state.
- [ ] Cached prefix can be reused across multiple requests.
- [ ] Byte limits are respected.
- [ ] Longer cached entry can be trimmed to shorter prompt when safe.
- [ ] Cache stats are visible in `/health` and `/metrics`.

## Priority 5: Paged Cache / Block-Aware Cache

### Problem

Plain prefix state snapshots duplicate KV tensors across similar prompts.
Paged/block cache can share blocks between requests and reduce memory.

Rapid-style paged cache uses:

- fixed token blocks, often 64 tokens
- chain hash per block
- ref counts
- copy-on-write
- LRU/LFU eviction
- pinning for important prefixes
- block table per request

### Why This Is Not Priority 1

Paged cache is more invasive than continuous batching and chunked prefill. It is
best added after prefix cache semantics and metrics are solid.

### Target Design

Add a block-aware backend behind the prefix cache API.

New config:

- `--prefix-cache-backend=snapshot|paged`
- `--paged-cache-block-size=64`
- `--max-cache-blocks`

### Implementation Steps

- [ ] Define block metadata.
  - block ID
  - token count
  - ref count
  - block hash
  - pinned flag
  - last access
  - cache data slices

- [ ] Define block table per request.
- [ ] Implement chain hashing.
- [ ] Implement shared prefix lookup.
- [ ] Implement COW when a request appends to shared prefix.
- [ ] Implement cache reconstruction for MLX KV cache objects.
- [ ] Add memory pressure eviction.
- [ ] Add pin/unpin for stable system/tools prefix.

### Risks

- Reconstructing cache objects may break for non-standard cache layers.
- GDN/DeltaNet/ArraysCache layers may need separate snapshot handling.
- TurboQuant cache layers need custom accounting/reconstruction.
- Debugging cache corruption is hard.

### Acceptance Criteria

- [ ] Same output as snapshot cache for deterministic prompts.
- [ ] Memory usage drops on repeated shared-prefix requests.
- [ ] Ref counts reach zero after request cleanup.
- [ ] Pinned blocks are not evicted.
- [ ] No stale blocks reused after model/config change.

## Priority 6: Generic KV Cache Quantization

### Problem

This repo has TurboQuant support, but not generic KV cache quantization for
stored prefix cache entries.

Rapid-MLX supports:

- TurboQuant V-cache compression.
- Generic KV cache quantization for prefix cache memory reduction.

Generic quantization is useful when:

- cache memory is the bottleneck
- many prompt prefixes are worth retaining
- TurboQuant is not compatible with a cache layer

### Target Design

Add optional quantization for stored prefix cache entries.

New config:

- `--kv-cache-quantization`
- `--kv-cache-quantization-bits`
- `--kv-cache-quantization-group-size`
- `--kv-cache-min-quantize-tokens`

### Implementation Steps

- [ ] Investigate MLX/MLX-LM cache quantization APIs available in current
  dependency versions.
- [ ] Implement cache-entry quantize/dequantize wrappers.
- [ ] Apply only to stored prefix entries, not active generation cache at first.
- [ ] Skip short prefixes by default.
- [ ] Track memory saved.
- [ ] Add quality/perf benchmark.

### Risks

- Dequantization overhead can hurt TTFT.
- Quantization can degrade logits if too aggressive.
- Not all cache layer types can be quantized safely.

### Acceptance Criteria

- [ ] Prefix cache memory drops for long prompts.
- [ ] Cached TTFT remains better than cold TTFT.
- [ ] Deterministic output remains stable enough for coding/tool prompts.
- [ ] Feature can be disabled fully.

## Priority 7: Tool Logits Bias and Tool Recovery

### Problem

Agent workflows lose time when a local model emits malformed tool calls. Even if
raw decode tokens/sec is high, retries and parser failures make wall-clock task
time worse.

Rapid-MLX treats tool reliability as a speed feature:

- Bias logits toward structured tool-call tokens.
- Recover malformed text tool calls into structured `tool_calls`.
- Support many parser formats.

### Local State

This repo has substantial API/tool compatibility code in
`scripts/local_api_server.py`, including Anthropic/OpenAI normalization and tool
block/event handling. But there is no clear Rapid-style tool logits processor in
the local generation loop.

### Target Design

Add optional tool-mode logits processor.

New config:

- `--enable-tool-logits-bias`
- `LOCAL_DFLASH_ENABLE_TOOL_LOGITS_BIAS`
- optional strength:
  - `--tool-logits-bias-strength`

### Implementation Steps

- [ ] Define token sequences for likely tool-call syntax for Qwen chat template.
- [ ] Add logits processor hook to `stream_generate(...)`.
- [ ] Bias only when tools are present and tool choice permits tool call.
- [ ] Avoid bias during normal prose response.
- [ ] Add recovery pass for malformed tool-call-like text.
- [ ] Emit metrics:
  - tool bias active
  - tool recovery attempted
  - tool recovery success/failure
  - malformed tool output count

### Risks

- Over-bias can force tool calls when model should answer text.
- Tokenization differs by model/template.
- Bad recovery can invent tool calls.

### Acceptance Criteria

- [ ] Tool-call pass rate improves on local harness.
- [ ] No regression for `tool_choice=none`.
- [ ] Invalid JSON tool arguments decrease.
- [ ] Recovery never runs silently; metrics/logs expose it.

## Priority 8: GDN/DeltaNet Snapshot Maturity

### Current Local State

Local repo already has GDN state capture/rollback helpers in `dflash/model_mlx.py`.

The remaining gap is likely not existence; it is integration maturity:

- cache keying
- snapshot timing
- memory accounting
- compatibility with chunked prefill
- compatibility with batching
- rollback correctness after speculative accept/reject

### Implementation Steps

- [ ] Add tests for GDN state clone/rollback under repeated prefix reuse.
- [ ] Add tests for GDN state with chunked prefill.
- [ ] Add tests for GDN state under abort/disconnect.
- [ ] Add metrics:
  - GDN snapshots created
  - GDN snapshot bytes
  - GDN rollback count
  - GDN restore latency

### Acceptance Criteria

- [ ] Same deterministic output with and without prefix reuse.
- [ ] No state leakage across requests.
- [ ] Rollback after rejected speculative proposal is correct.

## Priority 9: MTP Investigation

### Problem

Rapid-MLX has optional MTP support if a model exposes a multi-token prediction
head. This repo uses DFlash draft model and DDTree instead.

MTP may or may not matter for Qwen3.6-35B-A3B in this repo. Treat it as an
investigation, not an immediate implementation.

### Tasks

- [ ] Check whether target model exposes MTP head in MLX object.
- [ ] If present, benchmark MTP vs DFlash vs DDTree.
- [ ] If absent, document as not applicable.
- [ ] Do not mix MTP with DFlash until standalone behavior is measured.

### Acceptance Criteria

- [ ] Clear decision: implement, defer, or not applicable.

## Priority 10: Cloud Routing

### Problem

Rapid-MLX can route large-context requests to a cloud model when local prefill
would be slow.

This repo appears intentionally focused on local serving. Cloud routing may be a
non-goal.

### Decision Needed

- [ ] Decide if cloud routing belongs in this fork.
- [ ] If yes, design explicit opt-in only.
- [ ] Never silently route local/private prompts to cloud.

### Acceptance Criteria If Implemented

- [ ] Disabled by default.
- [ ] Requires explicit model/provider config.
- [ ] Logs routing decision.
- [ ] Exposes routed token counts.
- [ ] Redacts nothing silently; user owns privacy decision.

## Benchmark Plan

Benchmarks must compare current serial path against each new feature.

### Required Metrics

- Cold TTFT.
- Cached TTFT.
- Prompt tokens/sec.
- Decode tokens/sec.
- Aggregate tokens/sec under concurrency.
- p50/p95 TTFT.
- p50/p95 total latency.
- Peak MLX active memory.
- Prefix cache hit rate.
- Prefix tokens saved.
- Prefix cache bytes.
- Mid-prefill snapshots saved.
- Speculative acceptance ratio.
- Generated tokens.
- Tool-call success rate.
- Invalid tool-call rate.
- Abort cleanup success.

### Scenarios

1. Single short prompt.
   - 256 prompt tokens.
   - 128 output tokens.

2. Single long prompt.
   - 16k+ prompt tokens.
   - 128 output tokens.

3. Multi-turn stable prefix.
   - Same system/tools/context.
   - Different final user message.
   - Measures prefix reuse.

4. Concurrent short prompts.
   - 2, 4, 8 concurrent requests.
   - Measures batching throughput.

5. Concurrent mixed workload.
   - One long prefill.
   - Several active short decodes.
   - Measures chunked prefill fairness.

6. Interrupted long prefill.
   - Abort after partial prefill.
   - Repeat with same prefix.
   - Measures mid-prefill cache.

7. Tool-call harness.
   - Multiple tool schemas.
   - Multi-turn tool loop.
   - Measures tool bias/recovery impact.

### Suggested Output Table

| Engine | Cache | Chunked Prefill | Concurrency | TTFT p50 | TTFT p95 | Decode tok/s | Aggregate tok/s | Peak GB | Tool pass % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| serial | current | no | 1 | | | | | | |
| serial | memory-aware | yes | 1 | | | | | | |
| batched | memory-aware | yes | 2 | | | | | | |
| batched | memory-aware | yes | 4 | | | | | | |
| batched | paged | yes | 4 | | | | | | |

## Implementation Order

### Phase 0: Baseline

- [ ] Add or update benchmark script.
- [ ] Capture current serial numbers.
- [ ] Save baseline JSON reports.
- [ ] Add `/health` cache/scheduler fields needed for comparison.

### Phase 1: Engine Interface

- [ ] Introduce `SerialEngine` wrapper around current generation.
- [ ] Route existing server through engine abstraction.
- [ ] Confirm no behavior change.

### Phase 2: Batched Target Decode Prototype

- [ ] Add `BatchedEngine`.
- [ ] Use BatchGenerator for target-only decode.
- [ ] Support non-stream and stream.
- [ ] Gate behind env/CLI.
- [ ] Benchmark concurrency.

### Phase 3: Chunked Prefill

- [ ] Add prefill step size config.
- [ ] Implement chunk loop.
- [ ] Add heartbeat/abort checks.
- [ ] Validate logits/output parity.

### Phase 4: Mid-Prefill Snapshots

- [ ] Add save interval config.
- [ ] Save safe prefix snapshots.
- [ ] Reuse on repeated prefix.
- [ ] Add memory accounting and pruning.

### Phase 5: Prefix Cache Module

- [ ] Extract cache from server.
- [ ] Add exact/shorter/longer matching.
- [ ] Add memory-aware eviction.
- [ ] Add tests.

### Phase 6: Paged Cache

- [ ] Add block cache backend.
- [ ] Add COW/refcount.
- [ ] Add reconstruction.
- [ ] Benchmark memory savings.

### Phase 7: Agent Reliability Speedups

- [ ] Add tool logits bias.
- [ ] Add tool recovery metrics.
- [ ] Run tool-call harness.

## Non-Goals For Now

- Replacing DFlash/DDTree immediately.
- Batching DDTree before target-only batching works.
- Enabling cloud routing by default.
- Supporting every Rapid-MLX model/vision/audio feature.
- Rewriting all server compatibility code during performance work.

## Open Questions

- Can `mlx_lm.generate.BatchGenerator` integrate with DFlash draft verification,
  or should batched mode be target-only first?
- Does Qwen3.6-35B-A3B expose any MTP head in the loaded MLX model?
- Which cache layer types appear in the target model under TurboQuant enabled vs
  disabled?
- Which prompt boundary is safest for Codex/OpenCode tool-heavy prompts?
- What cache snapshot size is acceptable on the target Mac memory profile?
- Does generic KV quantization improve wall-clock TTFT after dequant overhead?
- Should global prefix cache persist only in memory or optionally on disk?

## Definition of Done

This TODO is complete only when:

- Batched engine exists behind a feature flag.
- Benchmarks show improved aggregate throughput under concurrent requests.
- Long prefill no longer starves active decode in batched mode.
- Prefix cache hits are measurable, bounded by memory limits, and mutation-safe.
- Tool-call reliability is measured before and after logits bias/recovery.
- Existing serial DFlash/DDTree behavior remains available as fallback.
