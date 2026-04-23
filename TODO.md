# dflash — Agentic 24h Readiness TODO

Derivado do conselho dos 6 subagentes (ver histórico da conversa).
Legenda: `[ ]` pendente · `[x]` feito · `[~]` em progresso · `[!]` bloqueado / pulado com nota.

---

## Fase 0 — Crítico (bloqueadores imediatos)

- [x] **0.1** `temperature` default `0.0 → 0.7` em todos os request schemas + clamp `<0.1→0.7` quando `tools` presente via `SamplingParams.for_request`.
- [x] **0.2** Adicionar `top_p=0.8`, `top_k=20`, `presence_penalty=1.0`, `repetition_penalty=1.05` nos 3 schemas + propagação completa (server → `stream_generate` → `make_sampler`/`make_logits_processors`).
- [x] **0.3** `response.incomplete` não é mais emitido como evento separado; `status="completed"` é sempre enviado no `response.completed` com `incomplete_details` dentro do payload.
- [x] **0.4** `error.code` populado via `_classify_error_code` (`context_length_exceeded`, `rate_limit_exceeded`, `server_overloaded`, `invalid_prompt`, `server_error`) em ambos os paths de `response.failed`.
- [x] **0.5** `max_output_tokens` default `512 → 16384` via `DEFAULT_MAX_TOKENS_FALLBACK`; override explícito preservado.
- [x] **0.6** `custom_tool_call` item emitido quando a tool é `type:"custom"` via `_convert_items_for_custom_tools` + `custom_tool_call_input.delta/.done` no streaming.
- [x] **0.7** Heartbeat SSE tipado: `response.in_progress` pra Responses, delta vazio pra chat.completions, `ping` pra Anthropic — todos resetam o idle timeout.

## Fase 1A — Protocolo Codex (completar compat)

- [x] **1.1** `usage` em `response.completed` inclui `cached_input_tokens`, `reasoning_output_tokens`, `input_tokens_details.cached_tokens`, `output_tokens_details.reasoning_tokens`.
- [x] **1.2** Round-trip básico de `reasoning`: aceita itens na entrada preservando summary/content em `<think>…</think>`; emite item `reasoning` + eventos `reasoning_summary_part.added`/`reasoning_summary_text.delta` quando o modelo produz `<think>`.
- [x] **1.3** Header `X-Models-Etag` + `ETag` em `/v1/models` e `/v1/models/{id}`.
- [x] **1.4** `ResponsesRequest` aceita `text`, `client_metadata`, `metadata`, `truncation` além dos já existentes.

## Fase 1B — Chat template Qwen3

- [x] **1.5** `_synthesize_orphan_tool_results`: sintetiza `{"error":"tool_result_missing"}` + user nudge quando último assistant tem tool_calls sem follow-up.
- [x] **1.6** `preserve_thinking=True` adicionado ao `apply_chat_template` com fallback em duas etapas.
- [x] **1.7** Stops `<|im_end|>` e `<|endoftext|>` augmentados em `tokenizer.eos_token_ids` no `ensure_loaded`. `</tool_call>` intencionalmente não incluído.
- [x] **1.8** Dedup de spans em tool-call regex — `_span_overlaps` evita parsear o mesmo `<tool_call>` nos 3 patterns.
- [x] **1.9** Prompt do judge (logprob + JSON) reforçado com cláusulas anti re-plano e anti `update_plan` isolado.
- [x] **1.10** `TOOL_CALLING_RULES_PROMPT` (bloco "Tool-calling rules (strict)") injetado no system quando tools estão presentes.

## Fase 1C — MLX memory hygiene

- [x] **1.11** Byte-bound em `_global_prefix_states` (12GB default) e `_stable_prefix_tokens_by_key` (2GB default) via flags `--global-prefix-cache-byte-limit-gb` / `--stable-prefix-tokens-byte-limit-gb` + tracking em `_global_prefix_cache_bytes`/`_stable_prefix_tokens_bytes`.
- [x] **1.12** `_maybe_clear_mlx_cache_locked` chama `mx.clear_cache()` + `reset_peak_memory()` quando `cache_memory > threshold * cache_limit`; flag `--mlx-clear-cache-threshold`.
- [x] **1.13** `mx.set_wired_limit` exposto via `--mlx-wired-limit-gb` / `LOCAL_DFLASH_MLX_WIRED_LIMIT_GB` com fallback gracioso em MLX builds antigos.
- [x] **1.14** `RotatingKVCache(keep=...)` controlado via `--rotating-keep-tokens` (default 1024) propagado por `DFlashConfig.rotating_keep_tokens` e `load_draft`.
- [x] **1.15** Assertion explícita em `make_cache` abortando se turboquant e sliding_window forem combinados.
- [!] **1.16** Pulado: `copy.deepcopy(target_cache)` é load-bearing no design atual; remover requer refactor do lifecycle de cache (versioned snapshots). Marcado pra Phase futura.
- [!] **1.17** Pulado: bucketização de `max_tokens`/`block_size` exige mudanças significativas em `stream_generate` e no layer de speculative decoding; prewarm de buckets ficaria arriscado sem os testes certos.

## Fase 1D — Server hygiene

- [x] **1.18** SSE cleanup on disconnect: try/except GeneratorExit/finally nos 3 streamers + `stop_event` propagado até `stream_generate` via `should_stop` callback.
- [x] **1.19** `_maybe_rotate_trace_file` rotaciona por tamanho (100MB) e idade (4h), mantém `TRACE_ROTATE_KEEP=5` backups. Overridable via env.
- [x] **1.20** `_logger.warning` nos `except Exception` do reasoning judge (+ `_logger.debug` para `_apply_logits_processors`).
- [x] **1.21** `_unload_from_timer` verifica `self._unload_timer is scheduled_timer` sob lock — protege contra race onde timer antigo dispara após reschedule.
- [x] **1.22** Worker threads com `join(timeout=30.0)` nos 3 streamers. Restante via try/finally.

## Fase 2A — Codex config + catalog

- [x] **2.A.1** `scripts/run_codex_local.sh` regenera config.toml completo (43 chaves, overridable por env): context window, auto-compact, tool_output_token_limit 64k, background_terminal_max_timeout 6h, stream_idle_timeout 30min, commit_attribution=false, include_apps_instructions=false, memories off, profile `dflash` registrado.
- [x] **2.A.2** `scripts/run_codex_local.sh` também emite `catalog.json` na mesma pasta CODEX_HOME e configura `model_catalog_json` pra silenciar "Model metadata not found".

## Fase 2B — Orquestração / watchdog

- [x] **2.1** `WALLCLOCK_HOURS` (default 24) + `TOKEN_BUDGET` checados via `_budget_exhausted` em cada iteração; salva state e sai ordenadamente.
- [x] **2.2** `_touch_heartbeat` em `.agent-queue/heartbeat` no agent_queue a cada iteração + `_touch_heartbeat_for_workdir` no watchdog a cada progress event.
- [x] **2.3** `_detect_alternation_loop` (ABAB) + contagem de `(tool_sig, is_error)` igual >= threshold → `tool_alternation_loop` / `repeated_tool_error` triggers.
- [x] **2.4** `LONG_RUNNING_COMMAND_TOKENS` whitelist (pytest/cargo/npm/bun/make/sleep/etc.) → stall timeout muda pra `stall_timeout_long_running_seconds` (30min default) enquanto tool_use está pendente sem tool_result.
- [x] **2.5** `cmd_install_launchd` / `cmd_uninstall_launchd` em `dflash.sh` geram e carregam `~/Library/LaunchAgents/dev.dflash.plist` com KeepAlive + ThrottleInterval=10.
- [x] **2.6** `cmd_start` wrappa com `caffeinate -dimsu` (opt-out via `DFLASH_NO_CAFFEINATE=1`) + `cmd_thermal` faz pmset check, exit 3 se CPU_Speed_Limit<90.
- [x] **2.7** `QueueState.shared_memory` + `_update_shared_memory` (last_summary, known_failing_paths) + `_build_executor_hint_block` injeta no prompt do executor.
- [x] **2.8** `_replan_remaining_tasks` dispara após `REPLAN_AFTER_FAILURES=2` consecutivas, limitado a `REPLAN_MAX=3` rodadas; registra `replan` em run.jsonl.
- [x] **2.9** `execute_task` salva snapshot em `.agent-queue/rollouts/<task>-<attempt>.log`; hint_block passa shared_memory + last_hint pros próximos attempts.
- [x] **2.10** `subprocess.run(..., timeout=EXECUTOR_TIMEOUT)` (default 1h, env-configurable) em `execute_task`; `TimeoutExpired` vira hint pro judge em vez de pendurar a queue.

## Fase 2C — Observabilidade

- [x] **2.C.1** `/metrics` expõe gauges: `dflash_uptime_seconds`, `dflash_active_generation_requests`, `dflash_queued_generation_requests`, `dflash_active_ticket_age_seconds`, `dflash_global_prefix_cache_bytes`, `dflash_stable_prefix_tokens_bytes`, `mlx_active_memory_bytes`, `mlx_cache_memory_bytes`, `mlx_peak_memory_bytes`, hits/misses.
- [x] **2.C.2** `/runs?dir=<workdir>` lê `.agent-queue/run.jsonl` e retorna até 500 entradas. Viewer HTML pode ler direto via fetch. (HTML estático não implementado — desnecessário agora; qualquer curl/jq serve.)

## Fase 3 — Polimento

- [x] **3.1** Streaming `custom_tool_call_input.delta` / `.done` implementado em `stream_response_events` (via 0.6).
- [x] **3.2** Eventos SSE de reasoning emitidos (`response.output_item.added` do tipo `reasoning`, `reasoning_summary_part.added`, `reasoning_summary_text.delta`) + `content_part.added/done` + `output_text.done` já existentes.
- [x] **3.3** Tie-breaker no judge: `LOCAL_DFLASH_FOLLOWUP_JUDGE_TIEBREAK_VOTES` (default 1 = legacy) pode ser setado a 3 pra best-of-3 com short-circuit em 2 concordantes.
- [x] **3.4** `--fallback-plan <path>` em `agent_queue.py run` carrega plano JSON pré-escrito quando planner falha.
- [x] **3.5** Hysteresis no adaptive block size: `grow_streak`/`shrink_streak` em `AdaptiveBlockSizeConfig` (default 1 = legacy; 3 recomendado para 24h estável) via `hysteresis_state` em `next_adaptive_block_size`.
- [!] **3.6** Pulado: DDTree + turboquant compatibilidade é refactor arquitetural em `ddtree_engine.py` — requer nova implementação de rollback sobre quant KV. Documentado como item futuro.
- [x] **3.7** `[mcp_servers.*]` pass-through: Codex 0.122 já lê a seção quando presente em `config.toml`; `run_codex_local.sh` agora deixa a seção livre pro usuário customizar (não gera nenhum por default). Documentado no comentário do config.
