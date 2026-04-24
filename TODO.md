# TODO - Qwen Local + Codex CLI Perfect Contract

Goal: make the local Qwen model and Codex CLI communicate with a precise,
lossless agentic protocol. The model should act through structured tool calls
when work remains, Codex should receive exactly the events it expects, and the
server should not need a semantic judge to decide whether a task is complete.

Source inputs:
- `FUTURE-TODO.md`
- 6 focused subagent reviews: protocol/config, tool calling, agentic behavior,
  streaming/protocol, QA harness, performance/env.

## Live Execution Status

- Current hard rule: do not encode workarounds for model-quality limitations.
  Server fixes must be Codex CLI integration/contract only: schemas, tool-call
  parsing, event shape, streaming, timeouts, env/config, and prompt rendering.
  No semantic blocking or rewriting for dev servers, background commands,
  missing scripts, scaffold choices, test strategy, or app implementation style.
- [x] S0 - Consolidate plan from `FUTURE-TODO.md` and subagents.
- [x] P0.1 - Fix Codex provider/config contract.
- [x] P0.2 - Finish Responses tool-call event sequence.
- [x] P0.3 - Fix `type:"custom"` / `apply_patch` path.
- [x] P1.1 - Make tool calling mandatory through prompt/template contract.
- [x] P1.2 - Implement protocol-level retry-on-no-tool, not semantic judge.
- [x] P1.3 - Honor `tool_choice`.
- [x] P2.1 - Replace per-protocol tool schema code with one normalizer.
- [x] P2.2 - Expand request message models for agentic multi-turn.
- [x] P2.3 - Make `_parse_tool_calls` the only parser.
- [x] P3 - Implement Chat Completions tool calling correctly.
- [x] P4 - Implement Anthropic Messages tool use correctly.
- [x] P5 - Streaming robustness.
- [x] P6 - Sampling, thinking, and agentic behavior.
- [x] P7 - Performance and env contract.
- [ ] P8 - Test and smoke harness. **In progress**
  - Current patch: tool-contract prompt now renders from normalized schemas with
    Jinja, and remaining semantic wrapper/schema guidance was removed. Shell
    argument sanitation only normalizes Codex contract aliases such as `cmd` to
    `command`; it no longer blocks or rewrites model-chosen commands.
  - Current patch after TUI loop report: repeated-tool guard now detects
    structural call cycles such as `A,B,A,B`, not only immediate `A,A`
    repeats. This is signature-based only; it does not inspect command
    semantics or special-case React/npm/build/test behavior.
  - Current patch after second TUI loop report: `update_plan` is treated as a
    status-only Codex tool for loop detection, so changing plan text no longer
    masks repetition of the same external tool call. This remains structural:
    tool name + arguments only, no command/domain interpretation.
  - Latest smoke `artifacts/codex-smoke/20260424-142736`: project scaffold,
    app source, and test files were created, but Codex entered a repeated
    explanatory loop before adding the test script or running verification.
    Applied fix: protocol retry no longer feeds failed assistant prose back into
    the prompt; smoke validation now checks implementation across `src/**/*.ts*`
    while still requiring non-template `App.tsx`, runnable tests, and build.
  - Latest smoke `artifacts/codex-smoke/20260424-144827`: Codex scaffolded the
    project, then a no-tool response consumed the full output budget after
    "Now let me create the game logic hook". Applied fix: tool-enabled
    generations now use a per-turn token cap while preserving the larger total
    response budget for retries/actions.
  - Additional hardening: protocol retries now require a tool call, retry budget
    increased inside the same response budget, and successful tool-call turns
    drop surrounding prose/reasoning before returning to Codex.
  - Latest smoke `artifacts/codex-smoke/20260424-150443`: model created app and
    tests, then hung after `npm run test` reported missing script. Root cause
    found in server normalization: Codex Responses `reasoning` items were being
    replayed as `<think>...</think>` and `preserve_thinking=True` was still
    enabled on tool turns. Applied fix: tool-enabled Responses drops reasoning
    items and passes `preserve_thinking=False`.
  - Latest smoke `artifacts/codex-smoke/20260424-152415`: reasoning loop stayed
    hidden after dependency install. Applied fix: reduce max tool-turn tokens to
    2048 and increase protocol retry count to 7 so bad no-tool turns end faster
    while total Codex output budget remains high.
  - Latest smoke `artifacts/codex-smoke/20260424-153456`: model used `cat <<EOF`
    for source writes. The escaped `!` display in Codex logs is shell quoting,
    not proof of file corruption. Reverted sanitizer block because it caused a
    dead-end; prompt now strongly prefers Python/pathlib and requires verifying
    file contents after shell-quoted writes.
  - Latest smoke `artifacts/codex-smoke/20260424-154957`: model wasted context
    reading template SVG/assets. Applied fix: tool prompt requires focused
    source/config exploration and Codex tool output limit reduced to 12000.
  - Latest smoke `artifacts/codex-smoke/20260424-155701`: model repeated the
    identical `vitest.config.ts` read command after receiving the same `NOT
    FOUND` result. Applied fix: protocol detects immediate identical tool-call
    repetition and retries internally with a "different tool call" directive.
  - Latest smoke `artifacts/codex-smoke/20260424-160427`: after focused
    exploration, a no-tool generation stayed silent too long. Applied fix: max
    tool-turn tokens reduced to 1024 and protocol retry budget raised to 15 so
    bad no-tool turns recover faster inside the same overall Codex run.
  - Latest smoke `artifacts/codex-smoke/20260424-161258`: total token budget was
    exhausted after scaffold/exploration and `mkdir`, before source writes.
    Applied fix: local Codex catalog now has non-empty base instructions and
    instruction template that explicitly require end-to-end tool work, focused
    exploration, no identical command repeats, and finite verification.
  - Latest smoke `artifacts/codex-smoke/20260424-161857`: model repeated the
    same absolute-path `types.ts` write while changing only `workdir`. Applied
    fix: repeated-tool detection compares shell command text independently from
    `workdir`.
  - Latest smoke `artifacts/codex-smoke/20260424-162451`: model emitted
    replacement source as markdown/prose instead of a tool call after retries.
    Applied fix: local server tool rules and Codex catalog now explicitly
    prohibit markdown source/replacement output when tools are available.
    Exhausted no-tool retries now return a protocol-error shell tool call
    instead of a final incomplete prose response, so Codex continues the loop.
  - Latest smoke `artifacts/codex-smoke/20260424-164104`: after writing
    `types.ts`, tool-turn generation went silent for several minutes. Applied
    fix: tool-turn default temperature is now 0.0 to reduce drift and loops.
  - Latest smoke `artifacts/codex-smoke/20260424-165353`: model used Python as
    the game runtime instead of a file-writing helper and never wrote the `.ts`
    file. Applied fix: prompt and catalog state that Python shell helpers may
    only create/write/maintain files, not implement the requested app/runtime.
  - Active smoke `artifacts/codex-smoke/20260424-165942`: scaffold, install,
    focused listing, `src/hooks` creation, and template `App.tsx` read have
    completed, then the next model generation stayed silent for more than two
    minutes with one active server request and no tool retry/failure metric yet.
    Smoke was stopped manually after no progress. Failure class: stream request
    can heartbeat indefinitely while a generation worker is silent. Applying a
    protocol timeout so Codex gets a failed response instead of a stuck TUI.
    Applied fix: stream result timeout now emits `generation_timeout`, `/health`
    exposes the timeout, and DDTree generation receives the cooperative stop
    callback so timeout/client disconnect can unwind the active generation slot.
  - Verification after timeout fix: Python compile, shell syntax, and
    `pytest tests/test_local_api_server.py tests/test_codex_contract.py -q`
    pass.
  - Restart validation exposed a process-control bug: `dflash.sh` set
    `DFLASH_NO_CAFFEINATE=1` for background starts, but
    `start_local_wrapper.sh` ignored it and still exec'd `caffeinate`. Applied
    fix: wrapper now honors `DFLASH_NO_CAFFEINATE` and exports the stream result
    timeout env.
  - Local tool-runner note: standalone `dflash.sh start` checks in this Codex
    session cannot prove long-lived background survival because the exec tool
    cleans child processes when the command ends. The smoke harness remains valid
    because it starts dflash and runs Codex inside one long-lived shell process.
  - Profile 114 also now uses `LOCAL_DFLASH_DEFAULT_TEMPERATURE_WITH_TOOLS=0.0`,
    matching the agentic tool-turn stability fix.
  - Active smoke `artifacts/codex-smoke/20260424-173138`: running after timeout,
    DDTree stop, wrapper, and profile-temperature fixes.
  - Smoke `artifacts/codex-smoke/20260424-173138` failed: stream timeout was
    surfaced as `response.failed`, causing Codex to reconnect 3/3 times and hang
    without source writes. Applied fix: when tools are available, stream timeout
    now returns a completed synthetic protocol-error tool call instead of
    disconnecting the Responses stream.
  - Verification after synthetic timeout tool-call fix: Python compile, shell
    syntax, and `pytest tests/test_local_api_server.py tests/test_codex_contract.py
    -q` pass.
  - Active smoke `artifacts/codex-smoke/20260424-174646`: running with stream
    timeout mapped to synthetic protocol-error tool call.
  - Smoke `artifacts/codex-smoke/20260424-174646` exposed malformed Codex exec
    arguments: model/server emitted shell args without required `command`, and
    Codex logged `missing field command`. Applied fix: `exec` tool calls now
    normalize `cmd`/`script` aliases to `command`, synthetic protocol-error tool
    calls use `command` for Codex exec, and catalog/tool prompts state the exact
    shell argument contract.
    Follow-up hardening: normalized shell schemas now explicitly require the
    command field (`command` for Codex `exec`, `cmd` for local shell tools), and
    smoke exports `LOCAL_DFLASH_TRACE_FILE` into the artifact directory for raw
    request/response diagnosis.
  - Verification after shell schema hardening: Python compile, shell syntax, and
    `pytest tests/test_local_api_server.py tests/test_codex_contract.py -q`
    pass.
  - Verification after `exec.command` fix: Python compile, shell syntax, and
    `pytest tests/test_local_api_server.py tests/test_codex_contract.py -q`
    pass.
  - Active smoke `artifacts/codex-smoke/20260424-175825`: running with Codex
    `exec.command` normalization.
  - Active smoke `artifacts/codex-smoke/20260424-180619`: running with shell
    schema hardening and trace capture.
  - Smoke `artifacts/codex-smoke/20260424-180619` trace found exact root cause:
    the first model call emitted `<parameter=command>`, but server sanitation for
    `shell_command` converted it to `cmd`, and Codex requires `command`. Applied
    fix: every shell-family tool now preserves/emits/requires `command`; only
    `exec` strips alias fields after filling `command`.
  - Verification after shell `command` unification: Python compile, shell syntax,
    and `pytest tests/test_local_api_server.py tests/test_codex_contract.py -q`
    pass.
  - Active smoke `artifacts/codex-smoke/20260424-183316`: running with unified
    shell `command` schema.
  - Smoke `artifacts/codex-smoke/20260424-183316` showed Qwen needed more time
    and output room for multi-file source writes: it installed dependencies, read
    `App.tsx`, then timed out/repeated a small heredoc write before creating the
    actual game/test files. Applied fix: stream result timeout increased to 600s,
    max tool-turn tokens increased to 4096, and prompts/catalog now require
    Python `pathlib.write_text` helpers for multi-line source writes instead of
    shell heredocs.
  - Verification after long tool-turn/write-helper tuning: Python compile, shell
    syntax, and `pytest tests/test_local_api_server.py tests/test_codex_contract.py
    -q` pass.
  - Active smoke `artifacts/codex-smoke/20260424-185314`: running with 600s stream
    result timeout, 4096 max tool-turn tokens, and pathlib write guidance.
  - Smoke `artifacts/codex-smoke/20260424-185314` reached real app/test file
    creation with pathlib, then failed on missing `test` script and drifted into
    `npm run dev &` / background commands instead of editing `package.json`.
    Applied fix: shell sanitizer blocks background commands and dev-server
    verification with a protocol error; prompts/catalog now require editing
    project config when a finite test/build script is missing.
  - Verification after dev/background/script-missing fix: Python compile, shell
    syntax, and `pytest tests/test_local_api_server.py tests/test_codex_contract.py
    -q` pass.
  - Active smoke `artifacts/codex-smoke/20260424-190724`: running with dev-server
    and background command block.
  - Design note from live debugging: Jinja should replace duplicated
    hardcoded prompt fragments for the tool-contract text. The server should
    render tool instructions from the actual normalized tool schema (for
    example the real shell command field, required fields, custom/freeform
    tools, and unavailable tools). Jinja does not replace the protocol adapter,
    parser, ID preservation, stream timeout, or sanitizer; it prevents prompt
    drift from those technical contracts.
  - Active smoke `artifacts/codex-smoke/20260424-190724` progressed farther:
    app/test files were written and `npm test` ran, failing because Vitest did
    not have a jsdom environment (`window is not defined`). Waiting for Codex to
    edit config/package and rerun finite tests/build.
- [ ] P9 - Rollout order and final definition of done.

## Current Diagnosis

Qwen is not failing because it is incapable of agentic behavior. It is failing
because the contract between model, dflash server, and Codex CLI is ambiguous.

Observed failure modes:
- Qwen writes "Let me verify..." or explanatory text instead of emitting a tool
  call.
- Tool-call rules are stricter in Responses than in Chat/Anthropic paths.
- Qwen chat template and dflash tool rules disagree on canonical tool-call
  format: XML-style `<function=...><parameter=...>` versus JSON
  `{name, arguments}`.
- Chat and Anthropic paths do not fully preserve or emit structured tool calls.
- Streaming can hide `<tool_call>` markup without converting it into protocol
  events.
- Codex can only continue reliably when it receives structured Responses events
  and matching tool result IDs.
- The follow-up judge is a workaround and can itself produce invalid JSON or
  bad continuation messages.

Target state:
- No judge on the default Codex path.
- One canonical tool schema normalizer.
- One canonical tool-call parser.
- One protocol adapter per API surface: Responses, Chat Completions,
  Anthropic Messages.
- Strict structured tool events for Codex CLI.
- Agentic behavior driven by prompt/template/tool_choice/tool events, not by
  semantic regexes or task-specific heuristics.

## Non-Negotiable Contract

1. If tools are available and work remains, the model must emit a structured
   tool call, not prose about future work.
2. If the model emits any supported tool-call markup, dflash must convert it to
   the exact protocol event shape expected by the client.
3. If a tool call is malformed or truncated, dflash must not silently clean it
   into normal prose. It must mark the generation incomplete/truncated or
   reprompt via a protocol-level retry.
4. Tool result IDs must round-trip exactly: `call_id`, `tool_call_id`, `id`,
   `name`, `is_error`, and raw content must be preserved.
5. Codex path should use `wire_api = "responses"` as the primary contract.
6. Chat and Anthropic paths should still be correct, but Codex compatibility is
   the first release gate.
7. Judge may remain only as an opt-in debug fallback. It must not be required
   for normal Codex autonomy.

## P0 - Make Codex Responses Contract Exact

### P0.1 Fix Codex provider/config contract

Files:
- `scripts/run_codex_local.sh`
- generated `/tmp/codex-local-dflash/config.toml`
- generated `/tmp/codex-local-dflash/catalog.json`

Tasks:
- Keep Codex on `wire_api = "responses"` and treat Responses as the canonical
  Codex protocol.
- Add tests that generated config contains:
  - `model_provider = "localdflash"`
  - `wire_api = "responses"`
  - `approval_policy = "never"`
  - `sandbox_mode = "danger-full-access"`
  - expected model name and base URL.
- Align catalog `default_reasoning_level`, config `model_reasoning_effort`, and
  profile `model_reasoning_effort`. Pick one policy and remove contradictions.
- Align Codex advertised context and server context:
  - Codex currently advertises `65536`.
  - server profile defaults to `32768`.
  - choose one contract, then update `run_codex_local.sh`,
    `start_local_wrapper.sh`, and `dflash.sh`.
- Align `truncation_policy.limit` with actual context and auto-compact limit.
- Decide canonical config source:
  - generated template in `run_codex_local.sh`, or
  - committed sample/template file.
  Avoid humans inspecting stale/empty config.

Definition of done:
- `dflash codex --version` does not start server.
- `dflash start` starts server with env matching generated Codex config.
- Config generation test passes.

### P0.2 Finish Responses tool-call event sequence

Files:
- `scripts/local_api_server.py`
- functions around `stream_response_events`, `_stream_response_events_body`,
  `_build_output_items`, `_convert_items_for_custom_tools`,
  `_normalize_responses_input`, `resolve_responses_context`

Tasks:
- Validate exact Codex event order for function calls:
  - `response.created`
  - `response.output_item.added`
  - `response.function_call_arguments.delta`
  - `response.function_call_arguments.done`
  - `response.output_item.done`
  - `response.completed`
- Validate exact Codex event order for custom tools:
  - `response.output_item.added`
  - `response.custom_tool_call_input.delta`
  - `response.custom_tool_call_input.done`
  - `response.output_item.done`
  - `response.completed`
- Ensure `finish_reason` / status maps to Codex expectations.
- Preserve `previous_response_id` and every tool result item across turns.
- Preserve `call_id` 1:1 between model output and tool result input.
- Never emit empty custom tool input for `apply_patch`.

Definition of done:
- Golden stream tests compare event type order exactly.
- Codex receives tool calls without deserialization errors.

### P0.3 Fix `type:"custom"` / `apply_patch` path

Files:
- `scripts/local_api_server.py`
- `scripts/run_codex_local.sh`

Tasks:
- Decide whether `LOCAL_DFLASH_CODEX_INCLUDE_APPLY_PATCH_TOOL` should default
  to true for maximum Codex autonomy.
- If true, fully support Codex custom/freeform apply_patch:
  - convert function-style model output into `custom_tool_call`;
  - preserve raw patch text as `input`;
  - do not JSON-wrap freeform patches unless Codex expects it;
  - stream custom input deltas and done events correctly.
- If false, document why shell-based editing is the expected path and test that
  shell editing remains sufficient.

Definition of done:
- Test: model output for apply_patch becomes non-empty `custom_tool_call.input`.
- Test: stream emits custom tool call input delta/done.
- Real smoke does not fail on invalid apply_patch payload.

## P1 - Remove Judge From Default Agentic Loop

### P1.1 Make tool calling mandatory through prompt/template contract

Files:
- `scripts/local_api_server.py`
- Qwen chat template at
  `/Users/samuelfajreldines/dev/models/Qwen3.6-35B-A3B-4bit/chat_template.jinja`

Tasks:
- Define one canonical tool-call format for model prompting.
- Align `TOOL_CALLING_RULES_PROMPT` with Qwen template.
- Parser may accept multiple formats, but prompt should ask for one.
- Inject the tool contract into all relevant paths:
  - Responses
  - Chat Completions
  - Anthropic Messages
- Add strong protocol rule:
  - if task is not complete and a tool exists, next assistant output must be a
    tool call only;
  - no prose before or after tool call;
  - no "I will..." action narration.
- Insert the agentic contract before `apply_chat_template` in `build_prompt`,
  so it survives client prompt variation.

Definition of done:
- Unit test shows tool rules are present in prompt for Responses, Chat, and
  Anthropic when tools exist.
- Golden prompt fixture matches Qwen template expected format.

### P1.2 Implement protocol-level retry-on-no-tool, not semantic judge

Files:
- `scripts/local_api_server.py`
- generation loops around `generate`, `_generate_response_locked`,
  `_generation_worker`

Tasks:
- Remove default judge dependency from Codex Responses path.
- Add a small protocol retry loop only when:
  - tools are present;
  - output has no visible final answer and no complete tool call;
  - output is empty/action-only/truncated/malformed tool call.
- Retry message must be protocol-level, not task-specific:
  - "Emit the next required tool call using the declared tool-call format, or
    return a final answer only if the task is complete."
- Keep this opt-in/configurable for Chat/Anthropic while Responses is hardened.
- Keep existing judge only behind an env flag for debugging.

Definition of done:
- Tests prove default Codex path does not call `_judge_turn_completion`.
- Tests prove empty/action-only output with tools triggers protocol retry.
- No task words like npm, Vite, build, test, scaffold appear in retry logic.

### P1.3 Honor `tool_choice`

Files:
- `scripts/local_api_server.py`
- request models for OpenAI Chat, Responses, Anthropic

Tasks:
- Parse and preserve `tool_choice`.
- Support:
  - `none`
  - `auto`
  - `required` / `any`
  - specific tool name
- If tool_choice requires a tool, do not accept final prose without tool call.
- Implement prompt enforcement first; add decode/prefill enforcement later if
  available.

Definition of done:
- Tests for `tool_choice=none`, `auto`, `required`, and named tool.

## P2 - Make Tool Schema Normalization Single-Source

### P2.1 Replace per-protocol tool schema code with one normalizer

Files:
- `scripts/local_api_server.py`
- existing `_normalize_anthropic_tools`
- existing Responses/OpenAI tool handling

Tasks:
- Create `_normalize_tool_schemas()`.
- Accept:
  - OpenAI nested `{"type":"function","function":{...}}`
  - Anthropic flat `{name,input_schema}`
  - Responses flat/function/custom tools
  - Codex `custom` tools
- Normalize before `_filter_disabled_tools()`.
- Preserve enough original metadata to emit the correct protocol output.

Definition of done:
- Tests for OpenAI, Anthropic, Responses, and custom tool schemas.

### P2.2 Expand request message models for agentic multi-turn

Files:
- `scripts/local_api_server.py`
- `OpenAIMessage`
- Anthropic message normalization
- Responses input normalization

Tasks:
- `OpenAIMessage` must accept:
  - `content: null`
  - `content` as list
  - `tool_calls`
  - `tool_call_id`
  - `name`
  - `function_call`
  - `role: developer`
- Chat handler should use `_normalize_openai_messages()` instead of raw
  `model_dump()` where appropriate.
- Anthropic and Chat should synthesize orphan tool results like Responses does.
- Preserve raw tool output and `is_error`.

Definition of done:
- Multi-turn transcript with assistant tool_call + tool result normalizes
  without 422 and without losing IDs.

### P2.3 Make `_parse_tool_calls` the only parser

Files:
- `scripts/local_api_server.py`
- `_parse_tool_calls`
- `_tool_call_items_from_payload`
- `_coerce_tool_arguments`

Tasks:
- Support and test:
  - Qwen XML `<tool_call><function=...><parameter=...>`
  - JSON body inside `<tool_call>`
  - Hermes-style JSON
  - fenced tool call blocks
  - multiple calls
  - custom/freeform input
  - malformed and truncated calls
- Return structured parse state:
  - visible text
  - complete tool calls
  - incomplete/truncated marker
- Never silently discard a partial tool call and return clean prose.

Definition of done:
- Parser tests cover every accepted format and every malformed case.

## P3 - Implement Chat Completions Tool Calling Correctly

Files:
- `scripts/local_api_server.py`
- `chat_completions`
- `stream_chat_completions`

Tasks:
- Pass `tools` to `build_prompt` in both non-stream and stream.
- Non-stream:
  - call `_parse_tool_calls()`;
  - fill `message.tool_calls`;
  - set `message.content = null` when only tool calls exist;
  - set `finish_reason = "tool_calls"`.
- Stream:
  - replace `_IncrementalVisibleTextStream` with an incremental parser that
    can emit `delta.tool_calls[]`;
  - do not leak tool markup as text;
  - do not hide tool markup without converting it to tool-call deltas;
  - end with `finish_reason = "tool_calls"`.
- Support `stream_options.include_usage`.

Definition of done:
- Golden Chat non-stream fixture matches OpenAI tool-call response shape.
- Golden Chat stream fixture matches OpenAI `delta.tool_calls` shape.

## P4 - Implement Anthropic Messages Tool Use Correctly

Files:
- `scripts/local_api_server.py`
- `anthropic_messages`
- `stream_anthropic_events`
- `_build_anthropic_content_blocks`

Tasks:
- Support `tool_use` content blocks in non-stream.
- Support streaming `tool_use`:
  - `content_block_start` with `type:"tool_use"`
  - `input_json_delta`
  - `content_block_stop`
  - `message_delta` with `stop_reason:"tool_use"`
- Use `toolu_` prefix for Anthropic tool IDs.
- Implement chunked `input_json_delta`.
- Preserve `tool_result` blocks and IDs.
- Remove extra `raw_text`/`metrics` fields in strict mode.

Definition of done:
- Claude-compatible stream fixture passes.

## P5 - Streaming Robustness

Files:
- `scripts/local_api_server.py`
- `_IncrementalVisibleTextStream`
- `_IncrementalVisibleTextExtractor`
- stream workers and queue helpers

Tasks:
- Replace text-only hidden-marker filter with an incremental mode parser:
  - mode `text`
  - mode `thinking`
  - mode `tool_call`
  - mode `partial_marker`
- Buffer partial markers until mode is known.
- Do not emit candidate text before protocol decision when auto-followup/retry
  may happen.
- Heartbeats should be based on wall-clock time, not parser churn.
- Worker cleanup:
  - always send terminal `done`;
  - propagate errors in protocol-valid shape;
  - review `stop_event` and `worker.join(timeout=30)`.
- If worker dies before result, client must receive valid error event and
  `[DONE]`.

Definition of done:
- Tests for partial markers split across chunks.
- Tests for worker error and disconnect cleanup.
- No TUI hangs on malformed stream.

## P6 - Sampling, Thinking, and Agentic Behavior

Files:
- `scripts/local_api_server.py`
- `SamplingParams.for_request`
- `build_prompt`
- `scripts/start_local_wrapper.sh`
- `scripts/run_codex_local.sh`

Tasks:
- For tool turns, use lower temperature by default:
  - target range `0.0` to `0.2`.
- Decide thinking policy:
  - for tool turns, prefer thinking off or hidden;
  - final user-visible output must never expose thinking;
  - when unfinished, first visible output should be tool markup.
- Keep reasoning settings consistent across:
  - model catalog
  - Codex config
  - profile
  - server env
- Add metrics for:
  - no-tool retries;
  - malformed tool-call retries;
  - final-with-tools cases;
  - tool-call parse format used.

Definition of done:
- No visible chain-of-thought or "continue thought process" in TUI.
- Tool-call rate improves in smoke loop.

## P7 - Performance and Env Contract

Files:
- `scripts/dflash.sh`
- `scripts/start_local_wrapper.sh`
- `scripts/run_codex_local.sh`
- `scripts/local_api_server.py`

Tasks:
- Keep `restart-114` as performance model, but make `start` and `restart`
  equivalent unless explicit args override.
- Print or expose effective env in `status` or `metrics`:
  - model path
  - draft path
  - engine
  - context
  - max tokens
  - keepalive
  - ddtree tree budget
  - turboquant target
  - adaptive block state
  - cache hit/miss
- Add profile `codex-agentic` if needed:
  - explicit thinking policy
  - coherent max tokens
  - coherent context reserve
  - longer keepalive for long TUI sessions.
- Health should be green only after:
  - model loaded;
  - draft loaded;
  - tokenizer loaded;
  - one short warmup prompt succeeds.
- Clamp `max_tokens` before generation and return HTTP 400 for invalid request
  instead of hanging or OOM.

Definition of done:
- `bash scripts/dflash.sh start`, `restart`, and `restart-114` show same
  effective performance profile.
- `/health` and `/metrics` prove model is actually ready.

## P8 - Test and Smoke Harness

Files:
- `tests/test_local_api_server.py`
- new `tests/test_codex_contract.py`
- new `scripts/smoke_codex_snake.sh`

Unit tests:
- generated Codex config contract.
- Responses function call event order.
- Responses custom tool call event order.
- `apply_patch` custom tool input non-empty.
- tool result ID round-trip.
- no judge called by default on Codex Responses path.
- protocol retry on empty/action-only/truncated tool output.
- parser cases: XML, JSON, fenced, multi-call, malformed, truncated.
- Chat non-stream `message.tool_calls`.
- Chat stream `delta.tool_calls`.
- Anthropic non-stream `tool_use`.
- Anthropic stream `tool_use`.
- `OpenAIMessage` accepts `content:null`, list content, tool fields.
- `tool_choice` enforcement.
- max_tokens clamp / HTTP 400.

Smoke script:
- Create `scripts/smoke_codex_snake.sh`.
- Use temp target dir.
- Run:
  - `bash scripts/dflash.sh start`
  - `bash scripts/dflash.sh codex exec --cd <temp> "create the famous snake game. Use react, vite and typescript. Create tests to make sure it work."`
- No watchdog.
- Timeout high enough for real local model, e.g. 45 minutes.
- Save artifacts:
  - Codex stdout/stderr
  - dflash log slice
  - generated project tree
  - package.json
  - test/build output
  - server metrics snapshot

Smoke success criteria:
- generated app exists.
- React, Vite, TypeScript present.
- Snake game has:
  - snake state
  - food
  - collision
  - score
  - keyboard controls
  - restart
  - tests
- `npm install` passes.
- tests pass.
- `npm run build` passes.
- logs contain tool-call events and tool outputs.
- logs do not contain:
  - judge on default path
  - watchdog
  - repeated "thought process"
  - repeated explanatory loop
  - stream disconnected before completion.

Soak:
- Run Snake smoke 5 times sequentially.
- Server remains healthy.
- Zero hangs.
- Zero idle-timeout retries.
- Tool-call success rate and malformed-call count recorded.

## P9 - Rollout Order

1. Lock Codex Responses config and add generated-config tests.
2. Finish Responses function/custom tool event correctness.
3. Normalize tool schemas and request messages.
4. Align Qwen tool prompt with Qwen chat template.
5. Disable judge by default for Codex path.
6. Add protocol-level retry-on-no-tool.
7. Implement Chat tool calling.
8. Implement Anthropic tool use.
9. Replace streaming text filter with incremental protocol parser.
10. Add smoke/soak harness and run Snake loop.

## Definition of Done for the Whole Project

The project is done when:
- `bash scripts/dflash.sh start` starts the local server.
- `bash scripts/dflash.sh codex` only opens Codex CLI and does not start server.
- Codex TUI can complete multi-step coding tasks without watchdog.
- Codex exec can create the Snake React/Vite/TypeScript app from scratch.
- The default path does not require follow-up judge.
- All tool calls and tool results round-trip structurally.
- Streams never leak hidden reasoning or raw tool markup.
- All unit tests pass.
- Snake smoke passes 5 times in a row.
