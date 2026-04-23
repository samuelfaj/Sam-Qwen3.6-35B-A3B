# DFlash MLX Local Serving Fork

This repository is a practical Apple Silicon fork of `z-lab/dflash`.

It is focused on running the MLX path locally with:

- Target model: `mlx-community/Qwen3.6-35B-A3B-4bit`
- Draft model: `z-lab/Qwen3.6-35B-A3B-DFlash`
- Backend: `MLX`
- Main use case: local coding agents and tool-using assistants

The goal is not to be a generic inference stack. The goal is to make the DFlash MLX path usable as a local serving system for real agent workflows, with good streaming behavior, controllable memory usage, and compatibility with Codex and OpenCode.

## What This Fork Adds

Compared with upstream `dflash`, this fork includes:

- A local OpenAI/Anthropic-style API wrapper in `scripts/local_api_server.py`
- Compatibility wrappers for Codex and OpenCode
- Background service helpers for the main API
- Lazy loading and keep-alive based unload behavior
- Streaming heartbeats for long prefills and long-running turns
- Broader tool-call parsing for real agent traffic
- TurboQuant support for compatible target-model KV cache layers in the MLX path
- Prefix reuse controls and health metrics for local serving
- Experimental DDTree MLX integration and test entry points

## Repository Layout

- `dflash/model_mlx.py`
  MLX DFlash generation path, model loading, cache management, and TurboQuant integration.
- `dflash/ddtree_engine.py`
  Experimental DDTree generation path for MLX.
- `dflash_mlx/runtime.py`
  MLX runtime helpers used by the DDTree integration.
- `scripts/local_api_server.py`
  Local HTTP API with OpenAI-style and Anthropic-style endpoints, streaming, health reporting, tool-call parsing, and model lifecycle controls.
- `scripts/start_local_wrapper.sh`
  Main local API launcher with the default coding-agent profile.
- `scripts/dflash.sh`
  Background service helper for the main API.
- `scripts/run_codex_local.sh`
  Writes a local Codex config and points Codex at the main API.
- `scripts/run_opencode_local.sh`
  Runs OpenCode against the main API.
- `scripts/test_qwen36_dflash_mlx.py`
  Direct MLX DFlash smoke test without the HTTP wrapper.
- `scripts/test_qwen36_ddtree_mlx.py`
  Direct MLX DDTree smoke test.
- `scripts/sweep_block_size.py`
  Replay and tuning utility for speculative block size experiments.

## Installation

Use a fresh virtual environment.

```bash
git clone git@github.com:samuelfaj/Sam-Qwen3.6-35B-A3B.git
cd Sam-Qwen3.6-35B-A3B

python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[mlx]"
```

The `mlx` extra in this fork also installs `mlx-turboquant`.

## Model Download

Example using local directories:

```bash
huggingface-cli download mlx-community/Qwen3.6-35B-A3B-4bit \
  --local-dir ~/models/Qwen3.6-35B-A3B-4bit

huggingface-cli download z-lab/Qwen3.6-35B-A3B-DFlash \
  --local-dir ~/models/Qwen3.6-35B-A3B-DFlash
```

The launcher scripts default to:

- Base model: `/Users/samuelfajreldines/dev/models/Qwen3.6-35B-A3B-4bit`
- Draft model: `/Users/samuelfajreldines/dev/models/Qwen3.6-35B-A3B-DFlash`

If your model paths differ, override them with environment variables:

```bash
export LOCAL_DFLASH_MODEL_PATH=~/models/Qwen3.6-35B-A3B-4bit
export LOCAL_DFLASH_DRAFT_PATH=~/models/Qwen3.6-35B-A3B-DFlash
```

## Quick MLX Tests

Direct DFlash MLX test:

```bash
python scripts/test_qwen36_dflash_mlx.py \
  --disable-thinking \
  --block-size 15 \
  --target-turboquant-bits 4
```

Direct DDTree MLX test:

```bash
python scripts/test_qwen36_ddtree_mlx.py \
  --disable-thinking \
  --tree-budget 4 \
  --target-turboquant-bits 4
```

## Main Local API

Start the main API in the foreground:

```bash
./scripts/start_local_wrapper.sh
```

Default endpoint:

```text
http://127.0.0.1:8010
```

### Main API Default Profile

- Model name: `qwen3.6-35b-a3b-dflash-local`
- Port: `8010`
- Block size: `12`
- Adaptive block size: enabled (min 10, max 22, grow 0.88, shrink 0.5)
- Draft sliding window: `32768`
- Rotating keep tokens: `4096`
- Context window: `98304`
- Max output tokens limit: `4096`
- Thinking mode: disabled
- TurboQuant target KV: `3.5`
- TurboQuant draft KV: `3.5`
- MLX memory limit: `32 GiB`
- MLX allocator cache limit: `4 GiB`
- MLX clear-cache threshold: `0.7`
- Keep-alive: `60s`
- Stream heartbeat: `10s`
- Preload on startup: disabled
- Response history limit: `4`
- Response prefix cache limit: `2`
- Global prefix cache limit: `2` (byte cap `4 GiB`)
- Stable prefix tokens byte cap: `2 GiB`

### Main API Service Helper

Use the background helper for longer sessions:

```bash
./scripts/dflash.sh start
./scripts/dflash.sh status
./scripts/dflash.sh logs
./scripts/dflash.sh stop
./scripts/dflash.sh restart
./scripts/dflash.sh kill
```

The helper stores:

- PID file: `.dflash.pid`
- Log file: `dflash.log`

Any extra arguments after `start` or `restart` are forwarded to `scripts/start_local_wrapper.sh`, and then to `scripts/local_api_server.py`.

### Main API Health Check

```bash
curl http://127.0.0.1:8010/health
```

Typical health fields include:

- `loaded`
- `context_window`
- `block_size`
- `disable_thinking`
- `keep_alive_seconds`
- `response_history_limit`
- `response_history_entries`
- `prefix_cache_state_limit`
- `global_prefix_cache_limit`
- `active_generation_requests`
- `queued_generation_requests`
- `stream_heartbeat_seconds`
- `target_turboquant_bits`
- `draft_turboquant_bits`
- `response_prefix_cache_bytes`
- `global_prefix_cache_bytes`
- `active_memory_gb`
- `cache_memory_gb`
- `peak_memory_gb`

## Client Wrappers

### Codex

Run Codex against the main API:

```bash
./scripts/run_codex_local.sh
```

This script writes a local Codex `config.toml` pointing at:

- Provider name: `localdflash`
- Base URL: `http://127.0.0.1:8010/v1`
- Wire API: `responses`

It also sets:

- `approval_policy = "never"`
- `sandbox_mode = "danger-full-access"`
- `model_reasoning_summary = "none"`
- `stream_idle_timeout_ms = 600000`

### OpenCode

Interactive mode:

```bash
./scripts/run_opencode_local.sh
```

One-shot mode:

```bash
./scripts/run_opencode_local.sh run --print-logs --format json --dir /tmp/test-run "Build a calculator in HTML, CSS, and JS."
```

The wrapper auto-checks `/health` and can restart the main API if needed.

Supervised autonomous mode for long-running tasks:

```bash
./scripts/run_opencode_local.sh run-auto --dir /tmp/test-run "Build a calculator in HTML, CSS, and JS."
```

Or through the helper:

```bash
./scripts/dflash.sh opencode-auto --dir /tmp/test-run "Build a calculator in HTML, CSS, and JS."
```

The watchdog mode adds:

- Stall detection based on missing progress events
- Loop detection based on repeated assistant text or repeated tool calls
- Persistent checkpoints in `.opencode-watchdog/`
- Automatic restart with a resume prompt that tells the agent to continue from the current filesystem state

Relevant knobs:

- `LOCAL_DFLASH_OPENCODE_AUTOPILOT=1` to route `run` through the watchdog automatically
- `LOCAL_DFLASH_OPENCODE_STALL_TIMEOUT_SECONDS=900`
- `LOCAL_DFLASH_OPENCODE_MAX_RESTARTS=12`
- `LOCAL_DFLASH_OPENCODE_RESTART_DELAY_SECONDS=5`
- `LOCAL_DFLASH_OPENCODE_LOOP_REPEAT_THRESHOLD=3`
- `LOCAL_DFLASH_OPENCODE_CHECKPOINT_DIR=.opencode-watchdog`

## Supported API Surfaces

The local API is intended to be usable by real agent clients, not just one-shot demos.

Supported surfaces:

- `POST /v1/responses`
- `POST /v1/chat/completions`
- `POST /v1/messages`
- `POST /v1/messages/count_tokens`
- `GET /v1/models`
- `GET /health`

Supported behavior:

- Streaming
- Tool calling
- OpenAI-style Responses API
- OpenAI-style Chat Completions API
- Anthropic-style Messages API
- Anthropic token counting

Tool-call parsing was expanded to tolerate multiple patterns commonly emitted by agent clients, including XML-style blocks, tagged JSON payloads, fenced blocks, and several common key variants.

## Memory Behavior

One of the main goals of this fork is to avoid permanently pinning model memory unless that is explicitly desired.

Main API default memory policy:

- Preload on startup: off
- Keep-alive after request: `60s`
- Unload after about one minute of inactivity

That means:

- The HTTP server can stay online even when the model is unloaded
- The next generation request can reload the model automatically
- You can keep a warm model briefly without keeping conversation state

If you want a strict MLX-side memory ceiling, set:

```bash
export LOCAL_DFLASH_MLX_MEMORY_LIMIT_GB=20
```

## Configuration Knobs

Main API launcher defaults from `scripts/start_local_wrapper.sh`:

```bash
export LOCAL_DFLASH_MODEL_PATH=/path/to/Qwen3.6-35B-A3B-4bit
export LOCAL_DFLASH_DRAFT_PATH=/path/to/Qwen3.6-35B-A3B-DFlash
export LOCAL_DFLASH_MODEL_NAME=qwen3.6-35b-a3b-dflash-local
export LOCAL_DFLASH_HOST=127.0.0.1
export LOCAL_DFLASH_PORT=8010
export LOCAL_DFLASH_BLOCK_SIZE=12
export LOCAL_DFLASH_ADAPTIVE_BLOCK_SIZE=1
export LOCAL_DFLASH_ADAPTIVE_BLOCK_SIZE_MIN=10
export LOCAL_DFLASH_ADAPTIVE_BLOCK_SIZE_MAX=22
export LOCAL_DFLASH_ADAPTIVE_BLOCK_SIZE_GROW_THRESHOLD=0.88
export LOCAL_DFLASH_ADAPTIVE_BLOCK_SIZE_SHRINK_THRESHOLD=0.5
export LOCAL_DFLASH_SLIDING_WINDOW_SIZE=32768
export LOCAL_DFLASH_ROTATING_KEEP_TOKENS=4096
export LOCAL_DFLASH_DISABLE_THINKING=1
export LOCAL_DFLASH_MAX_TOKENS=4096
export LOCAL_DFLASH_CONTEXT_RESERVE=256
export LOCAL_DFLASH_CONTEXT_WINDOW=98304
export LOCAL_DFLASH_KEEP_ALIVE=60
export LOCAL_DFLASH_STREAM_HEARTBEAT_SECONDS=10
export LOCAL_DFLASH_TURBOQUANT_BITS=3.5
export LOCAL_DFLASH_DRAFT_TURBOQUANT_BITS=3.5
export LOCAL_DFLASH_MLX_MEMORY_LIMIT_GB=32
export LOCAL_DFLASH_MLX_CACHE_LIMIT_GB=4
export LOCAL_DFLASH_MLX_CLEAR_CACHE_THRESHOLD=0.7
export LOCAL_DFLASH_NO_PRELOAD=1
export LOCAL_DFLASH_RESPONSE_HISTORY_LIMIT=4
export LOCAL_DFLASH_PREFIX_CACHE_STATE_LIMIT=2
export LOCAL_DFLASH_GLOBAL_PREFIX_CACHE_LIMIT=2
export LOCAL_DFLASH_GLOBAL_PREFIX_CACHE_BYTE_LIMIT_GB=4
export LOCAL_DFLASH_STABLE_PREFIX_TOKENS_BYTE_LIMIT_GB=2
```

## Tuning Notes

The main local profile is tuned for practical coding-agent use, not for maximum advertised context size.

Important levers:

- `LOCAL_DFLASH_BLOCK_SIZE`
- `LOCAL_DFLASH_ADAPTIVE_BLOCK_SIZE`
- `LOCAL_DFLASH_SLIDING_WINDOW_SIZE`
- `LOCAL_DFLASH_TURBOQUANT_BITS`
- `LOCAL_DFLASH_CONTEXT_WINDOW`
- `LOCAL_DFLASH_KEEP_ALIVE`

For replay-based speculative tuning, use:

```bash
python scripts/sweep_block_size.py --help
```

## Experimental DDTree Path

This repo now includes an experimental DDTree MLX path:

- Core engine: `dflash/ddtree_engine.py`
- Runtime helpers: `dflash_mlx/runtime.py`
- Direct test entry point: `scripts/test_qwen36_ddtree_mlx.py`

The DDTree path currently expects a compatible `ddtree-mlx` environment. If that package is missing, `dflash/ddtree_engine.py` raises a runtime error with an explicit install hint.

This path is experimental and separate from the HTTP serving wrappers documented above.

## Upstream

This repository is based on:

- Upstream project: `https://github.com/z-lab/dflash`
- DFlash paper: `https://arxiv.org/abs/2602.06036`
- DFlash model collection: `https://huggingface.co/collections/z-lab/dflash`

The original speculative decoding research belongs to the DFlash authors. This fork focuses on the MLX path, local serving ergonomics, and Apple Silicon agent workflows.

## Citation

If you use DFlash in research, cite the original paper:

```bibtex
@article{chen2026dflash,
  title   = {{DFlash: Block Diffusion for Flash Speculative Decoding}},
  author  = {Chen, Jian and Liang, Yesheng and Liu, Zhijian},
  journal = {arXiv preprint arXiv:2602.06036},
  year    = {2026}
}
```
