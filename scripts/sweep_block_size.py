#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_API_SERVER_PATH = REPO_ROOT / "scripts" / "local_api_server.py"


SPEC = importlib.util.spec_from_file_location("local_api_server", LOCAL_API_SERVER_PATH)
if SPEC is None or SPEC.loader is None:
    raise SystemExit(f"Could not import local_api_server from {LOCAL_API_SERVER_PATH}")
local_api_server = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(local_api_server)


DEFAULT_MODEL_PATH = str(REPO_ROOT.parent / "models" / "Qwen3.6-35B-A3B-4bit")
DEFAULT_DRAFT_PATH = str(REPO_ROOT.parent / "models" / "Qwen3.6-35B-A3B-DFlash")


@dataclass
class WorkloadItem:
    kind: str
    max_tokens: int
    temperature: float
    messages: list[dict[str, Any]] | None = None
    tools: list[dict[str, Any]] | None = None
    request_messages: list[dict[str, Any]] | None = None
    response_id: str | None = None
    previous_response_id: str | None = None


def _load_events(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _extract_workload(events: list[dict[str, Any]], max_samples: int | None) -> tuple[list[WorkloadItem], dict[str, int]]:
    items: list[WorkloadItem] = []
    stats = {
        "responses_completed": 0,
        "responses_request_only": 0,
        "messages_requests": 0,
        "skipped_chained_response_requests": 0,
    }
    has_completed_responses = any(event.get("kind") == "responses.completed" for event in events)

    for event in events:
        kind = event.get("kind")
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue

        if kind == "responses.completed":
            items.append(
                WorkloadItem(
                    kind="responses",
                    response_id=payload.get("response_id"),
                    previous_response_id=payload.get("previous_response_id"),
                    request_messages=payload.get("request_messages") or [],
                    tools=payload.get("tools") or [],
                    max_tokens=int(payload.get("max_output_tokens") or 512),
                    temperature=float(payload.get("temperature") or 0.0),
                )
            )
            stats["responses_completed"] += 1
        elif kind == "messages":
            req = local_api_server.AnthropicRequest.model_validate(payload)
            messages, tools = local_api_server._normalize_anthropic_messages(req)
            items.append(
                WorkloadItem(
                    kind="messages",
                    messages=messages,
                    tools=tools,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                )
            )
            stats["messages_requests"] += 1
        elif kind == "responses" and not has_completed_responses:
            req = local_api_server.ResponsesRequest.model_validate(payload)
            request_messages, request_tools = local_api_server._normalize_responses_input(req)
            if req.previous_response_id:
                stats["skipped_chained_response_requests"] += 1
                continue
            items.append(
                WorkloadItem(
                    kind="responses",
                    response_id=f"trace_req_{len(items)}",
                    previous_response_id=None,
                    request_messages=request_messages,
                    tools=request_tools,
                    max_tokens=req.max_output_tokens or 512,
                    temperature=req.temperature,
                )
            )
            stats["responses_request_only"] += 1

        if max_samples is not None and len(items) >= max_samples:
            break

    return items, stats


def _make_server(args: argparse.Namespace, block_size: int):
    adaptive_config = local_api_server.AdaptiveBlockSizeConfig(
        enabled=args.adaptive_block_size,
        min_block_size=max(1, args.adaptive_block_size_min),
        max_block_size=max(1, args.adaptive_block_size_max or block_size),
        grow_threshold=args.adaptive_block_size_grow_threshold,
        shrink_threshold=args.adaptive_block_size_shrink_threshold,
    )
    detected_context_window = local_api_server._detect_context_window(args.model_path)
    context_window = args.context_window_override or detected_context_window
    server = local_api_server.LocalModelServer(
        model_path=args.model_path,
        draft_path=args.draft_path,
        model_name=args.model_name,
        block_size=block_size,
        disable_thinking=not args.enable_thinking,
        sliding_window_size=args.sliding_window_size,
        max_tokens_limit=args.max_tokens_limit,
        context_window=context_window,
        context_reserve=args.context_reserve,
        keep_alive_seconds=None,
        target_turboquant_bits=args.target_turboquant_bits,
        adaptive_block_size_config=adaptive_config,
        global_prefix_cache_limit=max(0, args.global_prefix_cache_limit),
    )
    server.ensure_loaded()
    return server


def _replay_workload(
    server,
    workload: list[WorkloadItem],
    *,
    max_tokens_override: int | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for item in workload:
        max_tokens = max_tokens_override or item.max_tokens
        if item.kind == "responses":
            assert item.request_messages is not None
            messages, tools = server.resolve_responses_context(
                request_messages=item.request_messages,
                request_tools=item.tools or [],
                previous_response_id=item.previous_response_id,
            )
            result = server.generate(
                messages,
                max_tokens,
                item.temperature,
                tools=tools,
                previous_response_id=item.previous_response_id,
                capture_prompt_cache_state=True,
            )
            output_items = local_api_server._build_output_items(result["text"])
            server.remember_response(
                response_id=item.response_id or f"resp_trace_{len(results)}",
                previous_response_id=item.previous_response_id,
                request_messages=item.request_messages,
                tools=tools,
                output_items=output_items,
                prompt_cache_state=result.get("prompt_cache_state"),
            )
            results.append(result)
            continue

        result = server.generate(
            item.messages or [],
            max_tokens,
            item.temperature,
            tools=item.tools,
        )
        results.append(result)

    return results


def _summarize_results(block_size: int, adaptive: bool, results: list[dict[str, Any]]) -> dict[str, Any]:
    total_elapsed = sum(result["elapsed"] for result in results)
    total_output_tokens = sum(result["generated_tokens"] for result in results)
    total_prefill_seconds = sum(result["prefill_seconds"] for result in results)
    total_decode_seconds = sum(result["decode_seconds"] for result in results)
    total_proposed_tokens = sum(result.get("proposed_tokens", 0) for result in results)
    total_accepted_tokens = sum(result.get("accepted_tokens", 0) for result in results)
    total_speculative_steps = sum(result.get("speculative_steps", 0) for result in results)
    response_prefix_hits = sum(1 for result in results if result.get("prefix_cache_source") == "response")
    global_prefix_hits = sum(1 for result in results if result.get("prefix_cache_source") == "global")

    return {
        "block_size": block_size,
        "adaptive": adaptive,
        "samples": len(results),
        "throughput_tps": (total_output_tokens / max(total_elapsed, 1e-9)),
        "avg_generation_tps": statistics.mean(result["generation_tps"] for result in results),
        "avg_prompt_tps": statistics.mean(result["prompt_tps"] for result in results),
        "avg_prefill_seconds": total_prefill_seconds / max(len(results), 1),
        "avg_decode_seconds": total_decode_seconds / max(len(results), 1),
        "avg_acceptance_length": (
            total_accepted_tokens / max(total_speculative_steps, 1)
            if total_speculative_steps > 0
            else 0.0
        ),
        "avg_acceptance_ratio": (
            total_accepted_tokens / max(total_proposed_tokens, 1)
            if total_proposed_tokens > 0
            else 0.0
        ),
        "response_prefix_hits": response_prefix_hits,
        "global_prefix_hits": global_prefix_hits,
    }


def _print_summary(summaries: list[dict[str, Any]]) -> None:
    print(
        "block_size | adaptive | samples | throughput_tps | avg_gen_tps | avg_prompt_tps | "
        "avg_prefill_s | avg_decode_s | avg_accept_len | avg_accept_ratio | resp_prefix_hits | global_prefix_hits"
    )
    for summary in summaries:
        print(
            f"{summary['block_size']:>10} | "
            f"{str(summary['adaptive']):>8} | "
            f"{summary['samples']:>7} | "
            f"{summary['throughput_tps']:>14.2f} | "
            f"{summary['avg_generation_tps']:>11.2f} | "
            f"{summary['avg_prompt_tps']:>14.2f} | "
            f"{summary['avg_prefill_seconds']:>13.4f} | "
            f"{summary['avg_decode_seconds']:>12.4f} | "
            f"{summary['avg_acceptance_length']:>14.3f} | "
            f"{summary['avg_acceptance_ratio']:>16.3f} | "
            f"{summary['response_prefix_hits']:>16} | "
            f"{summary['global_prefix_hits']:>18}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep DFlash block_size on real traced workload.")
    parser.add_argument("--trace-file", type=Path, required=True, help="JSONL trace file produced by LOCAL_DFLASH_TRACE_FILE.")
    parser.add_argument("--block-sizes", default="8,10,12,15,18", help="Comma-separated block sizes to test.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on replayed requests.")
    parser.add_argument("--max-output-tokens-override", type=int, default=None, help="Optional cap applied to every replayed request.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--draft-path", default=DEFAULT_DRAFT_PATH)
    parser.add_argument("--model-name", default="qwen3.6-35b-a3b-dflash-local")
    parser.add_argument("--sliding-window-size", type=int, default=4096)
    parser.add_argument("--max-tokens-limit", type=int, default=32768)
    parser.add_argument("--context-reserve", type=int, default=256)
    parser.add_argument("--context-window-override", type=int, default=65536)
    parser.add_argument("--target-turboquant-bits", type=float, default=4.0)
    parser.add_argument("--enable-thinking", action="store_true", help="Enable thinking mode when rebuilding prompts.")
    parser.add_argument("--adaptive-block-size", action="store_true", help="Evaluate adaptive block size instead of fixed block size.")
    parser.add_argument("--adaptive-block-size-min", type=int, default=8)
    parser.add_argument("--adaptive-block-size-max", type=int, default=18)
    parser.add_argument("--adaptive-block-size-grow-threshold", type=float, default=0.95)
    parser.add_argument("--adaptive-block-size-shrink-threshold", type=float, default=0.6)
    parser.add_argument("--global-prefix-cache-limit", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.trace_file.exists():
        raise SystemExit(f"Trace file not found: {args.trace_file}")

    block_sizes = [int(part.strip()) for part in args.block_sizes.split(",") if part.strip()]
    events = _load_events(args.trace_file)
    workload, stats = _extract_workload(events, args.max_samples)
    if not workload:
        raise SystemExit(
            "No replayable workload found in trace. "
            "For responses-based agent loops, capture traces after this patch so `responses.completed` events are present."
        )

    print("Trace stats:")
    print(json.dumps(stats, indent=2))
    print(f"Replayable items: {len(workload)}")

    summaries: list[dict[str, Any]] = []
    for block_size in block_sizes:
        print(f"\nRunning sweep for block_size={block_size} adaptive={args.adaptive_block_size} ...")
        server = _make_server(args, block_size)
        results = _replay_workload(
            server,
            workload,
            max_tokens_override=args.max_output_tokens_override,
        )
        summaries.append(_summarize_results(block_size, args.adaptive_block_size, results))
        server.unload()

    print()
    _print_summary(summaries)
    best = max(summaries, key=lambda summary: summary["throughput_tps"])
    print("\nBest candidate:")
    print(json.dumps(best, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
