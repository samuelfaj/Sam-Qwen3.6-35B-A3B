# Sam Qwen3.6-35B-A3B

This repository is a practical local-serving fork of `z-lab/dflash`, tuned for an Apple Silicon workflow around:

- Target model: `mlx-community/Qwen3.6-35B-A3B-4bit`
- Draft model: `z-lab/Qwen3.6-35B-A3B-DFlash`
- Backend: `MLX`
- Primary use case: local agentic coding assistants such as Codex and OpenCode

The goal of this fork is not to be a generic serving stack. It is a stable local setup for running the 35B 4-bit target with DFlash enabled, exposing a local API that feels close to hosted coding models while avoiding unnecessary always-on memory usage.

## What Was Added In This Fork

Compared with upstream `dflash`, this repo now includes:

- A local MLX API wrapper exposing:
  - `POST /v1/responses`
  - `POST /v1/chat/completions`
  - `POST /v1/messages`
  - `POST /v1/messages/count_tokens`
  - `GET /v1/models`
  - `GET /health`
- Compatibility wrappers for:
  - Codex via `scripts/run_codex_local.sh`
  - OpenCode via `scripts/run_opencode_local.sh`
- A dedicated local launch script:
  - `scripts/start_local_wrapper.sh`
- A direct local test script:
  - `scripts/test_qwen36_dflash_mlx.py`
- TurboQuant support for the target-model KV cache in `dflash/model_mlx.py`
- Support for loading models from either a local directory or Hugging Face repo ID
- Streaming improvements so clients receive output earlier instead of waiting for the full generation
- SSE heartbeat comments during long prefills or reasoning pauses so agent clients appear alive while the model is still working
- Hardened tool-call parsing for multiple formats used by agent frameworks
- Lazy loading and automatic unload controls so the model does not stay resident in memory when idle

## Current Tuned Profile

The current default profile is optimized for local agentic use, not for benchmarking maximum context length at any cost.

- Base model: `mlx-community/Qwen3.6-35B-A3B-4bit`
- Draft model: `z-lab/Qwen3.6-35B-A3B-DFlash`
- DFlash speculative decoding: `ON`
- Speculative block size: `15`
- Context window: `65536`
- Output limit: `8192`
- Draft sliding window: `4096`
- Qwen thinking mode: `disabled` by default for faster, cleaner agent behavior
- TurboQuant target KV cache: `4-bit`
- Keep-alive: `0`
- Preload at startup: `disabled`
- Heartbeat interval while streaming: `2s`

This profile was chosen because it is materially more stable for real local coding-agent use than trying to force a much larger context window such as `256k` on this setup.

## Why This Setup Exists

The main requirements behind this fork were:

- Keep DFlash enabled
- Run the 35B target in 4-bit mode
- Preserve strong local quality for coding-agent tasks
- Make Codex and OpenCode work against the local model with minimal friction
- Avoid keeping 100+ GB allocated when the model is idle
- Keep the system agentic: tool calls, long-running turns, streaming, and visible progress

That is why the wrapper is designed to behave more like a serving product and less like a one-off test script.

## Repository Layout

- `dflash/model_mlx.py`
  - MLX DFlash generation path
  - local-path-or-Hub loading
  - TurboQuant KV cache integration for the target model
- `scripts/local_api_server.py`
  - local OpenAI/Anthropic-compatible API wrapper
  - streaming, heartbeats, tool parsing, lazy load/unload
- `scripts/start_local_wrapper.sh`
  - convenient launcher with the tuned default profile
- `scripts/dflash.sh`
  - service helper with `start`/`stop`/`restart`/`status`/`kill`/`logs` subcommands
- `scripts/run_codex_local.sh`
  - writes a local Codex config and points Codex at this server
- `scripts/run_opencode_local.sh`
  - runs OpenCode against this local model
- `scripts/test_qwen36_dflash_mlx.py`
  - minimal direct MLX test path without the HTTP wrapper

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

The MLX extra in this fork also installs `mlx-turboquant`.

## Download The Models

Example using local directories:

```bash
huggingface-cli download mlx-community/Qwen3.6-35B-A3B-4bit \
  --local-dir ~/models/Qwen3.6-35B-A3B-4bit

huggingface-cli download z-lab/Qwen3.6-35B-A3B-DFlash \
  --local-dir ~/models/Qwen3.6-35B-A3B-DFlash
```

The launcher scripts default to the following paths:

- Base model: `/Users/samuelfajreldines/dev/models/Qwen3.6-35B-A3B-4bit`
- Draft model: `/Users/samuelfajreldines/dev/models/Qwen3.6-35B-A3B-DFlash`

Override them with environment variables if your paths differ:

```bash
export LOCAL_DFLASH_MODEL_PATH=~/models/Qwen3.6-35B-A3B-4bit
export LOCAL_DFLASH_DRAFT_PATH=~/models/Qwen3.6-35B-A3B-DFlash
```

## Quick Local MLX Test

This bypasses the HTTP server and tests direct MLX generation:

```bash
python scripts/test_qwen36_dflash_mlx.py \
  --disable-thinking \
  --block-size 15 \
  --target-turboquant-bits 4
```

Useful flags:

- `--prompt "..."` to change the input
- `--max-tokens 256` to cap generation
- `--sliding-window-size 4096` to limit draft KV growth
- `--target-turboquant-bits 0` to disable TurboQuant for comparison

## Start The Local API Server

Foreground (logs in the terminal, Ctrl+C to stop):

```bash
./scripts/start_local_wrapper.sh
```

The wrapper exposes a local server on `127.0.0.1:8010` by default.

### Managing The Server As A Background Service

For long-running agent sessions, use the `dflash.sh` helper, which runs the server in the background, tracks the PID, and waits for `/health` to be ready:

```bash
./scripts/dflash.sh start      # start in background, wait for /health (up to 120s)
./scripts/dflash.sh status     # check running process + /health
./scripts/dflash.sh logs       # tail -f dflash.log
./scripts/dflash.sh stop       # graceful SIGTERM, wait up to 30s
./scripts/dflash.sh restart    # stop then start
./scripts/dflash.sh kill       # SIGKILL the process group, clear port squatters
./scripts/dflash.sh opencode   # launch OpenCode against the local server
./scripts/dflash.sh codex      # launch Codex against the local server
```

The `opencode` and `codex` subcommands wrap `run_opencode_local.sh` and `run_codex_local.sh` respectively. Any extra arguments are forwarded, so you can do things like `./scripts/dflash.sh opencode run --print-logs "..."` or `./scripts/dflash.sh codex exec "..."`.

Paths used:

- PID file: `.dflash.pid` at the repo root
- Log file: `dflash.log` at the repo root

Any extra arguments after `start` or `restart` are forwarded to `start_local_wrapper.sh` (and from there to `local_api_server.py`).

Health check directly:

```bash
curl http://127.0.0.1:8010/health
```

Typical health response fields include:

- `loaded`
- `context_window`
- `block_size`
- `disable_thinking`
- `keep_alive_seconds`
- `stream_heartbeat_seconds`
- `target_turboquant_bits`
- `active_memory_gb`
- `cache_memory_gb`
- `peak_memory_gb`

## Run With Codex

```bash
./scripts/run_codex_local.sh
```

This script writes a local `config.toml` for Codex and points it at:

- provider name: `localdflash`
- base URL: `http://127.0.0.1:8010/v1`
- wire API: `responses`

The Codex wrapper is configured for:

- `approval_policy = "never"`
- `sandbox_mode = "danger-full-access"`
- no reasoning summary requirement
- long stream idle timeout for agentic turns

## Run With OpenCode

Interactive mode:

```bash
./scripts/run_opencode_local.sh
```

One-shot mode:

```bash
./scripts/run_opencode_local.sh run --print-logs --format json --dir /tmp/test-run "Build a calculator in html, css, and js."
```

This fork was tuned specifically so OpenCode behaves in a more agentic way:

- tool calls are parsed more reliably
- streaming starts earlier
- heartbeat events make the run feel active during heavy prefills
- the wrapper stays compatible with long tool-heavy turns

## API Compatibility Notes

The local wrapper is meant to make local MLX serving usable by real agent clients, not only by simple text-generation demos.

Supported API surfaces:

- OpenAI-style chat completions
- OpenAI-style responses API
- Anthropic-style messages API
- tool calling
- streaming
- token counting for Anthropic-compatible clients

Tool-call parsing was expanded beyond a single XML format and now accepts multiple patterns commonly produced by agent frameworks, including:

- XML-style function blocks
- tagged JSON payloads
- tagged tool-call lists
- fenced tool-call blocks
- payloads with keys such as `function`, `tool_calls`, `function_calls`, `input`, `parameters`, `tool_use`, and `recipient_name`

The Responses API wrapper also keeps an in-process response history so clients that send `previous_response_id` plus incremental tool outputs can continue multi-step agent loops without replaying the full transcript on every turn.

## Memory Behavior

One of the main changes in this fork is that the model should not remain loaded forever unless you explicitly want that behavior.

Default memory policy:

- preload on startup: `off`
- keep-alive after request: `0`
- unload as soon as the request finishes

That means:

- memory is allocated when a request needs the model
- memory is released when the request is done
- idle RAM pressure is drastically lower than an always-loaded setup like Ollama-style serving

Relevant environment variables:

```bash
export LOCAL_DFLASH_KEEP_ALIVE=0
export LOCAL_DFLASH_NO_PRELOAD=1
export LOCAL_DFLASH_MLX_MEMORY_LIMIT_GB=
export LOCAL_DFLASH_MLX_CACHE_LIMIT_GB=0
```

If you want the model to remain warm for a short period after each request, set a positive keep-alive value:

```bash
export LOCAL_DFLASH_KEEP_ALIVE=60
```

## Tuning Knobs

The main configuration knobs exposed by `scripts/start_local_wrapper.sh` are:

```bash
export LOCAL_DFLASH_MODEL_PATH=/path/to/Qwen3.6-35B-A3B-4bit
export LOCAL_DFLASH_DRAFT_PATH=/path/to/Qwen3.6-35B-A3B-DFlash
export LOCAL_DFLASH_BLOCK_SIZE=15
export LOCAL_DFLASH_SLIDING_WINDOW_SIZE=4096
export LOCAL_DFLASH_DISABLE_THINKING=1
export LOCAL_DFLASH_MAX_TOKENS=8192
export LOCAL_DFLASH_CONTEXT_RESERVE=256
export LOCAL_DFLASH_CONTEXT_WINDOW=65536
export LOCAL_DFLASH_KEEP_ALIVE=0
export LOCAL_DFLASH_STREAM_HEARTBEAT_SECONDS=2
export LOCAL_DFLASH_TURBOQUANT_BITS=4
export LOCAL_DFLASH_MLX_MEMORY_LIMIT_GB=
export LOCAL_DFLASH_MLX_CACHE_LIMIT_GB=0
export LOCAL_DFLASH_NO_PRELOAD=1
export LOCAL_DFLASH_RESPONSE_HISTORY_LIMIT=1024
```

## TurboQuant

This fork adds TurboQuant KV cache support for the target model in the MLX path.

What changed:

- target prompt cache creation can replace compatible KV layers with TurboQuant-backed caches
- the bit width is configurable
- returned keys and values are cast back to the original dtype for stability

Current default:

- `LOCAL_DFLASH_TURBOQUANT_BITS=4`

To disable TurboQuant:

```bash
export LOCAL_DFLASH_TURBOQUANT_BITS=0
```

## Notes On Context Length

The wrapper is currently tuned to `64k` context by default.

That was an intentional choice. For this local MLX setup, forcing much larger contexts can increase latency, memory pressure, and instability without producing a better agent experience. If you want to experiment, you can raise:

```bash
export LOCAL_DFLASH_CONTEXT_WINDOW=131072
```

But the default `65536` is the balanced profile that was selected for actual local coding-agent use.

## Upstream

This repository is based on:

- Upstream project: `https://github.com/z-lab/dflash`
- DFlash paper: `https://arxiv.org/abs/2602.06036`
- DFlash models: `https://huggingface.co/collections/z-lab/dflash`

The core speculative decoding work belongs to the original DFlash authors. This fork focuses on the Apple Silicon MLX path and the local agent-serving workflow built around it.

## Citation

If you use DFlash in research, cite the original work:

```bibtex
@article{chen2026dflash,
  title   = {{DFlash: Block Diffusion for Flash Speculative Decoding}},
  author  = {Chen, Jian and Liang, Yesheng and Liu, Zhijian},
  journal = {arXiv preprint arXiv:2602.06036},
  year    = {2026}
}
```
