# dflash — TODO de Compatibilidade Cross-API

Este arquivo lista todo o trabalho **pendente** para tornar o dflash 100%
compatível com Codex CLI, Claude Code CLI (headless e interativo), Hermes
Agent CLI (NousResearch), e clientes OpenAI-compatíveis (aider, OpenHands,
continue.dev, opencode, LiteLLM, LangChain).

## Contexto mínimo para executar qualquer item

- **Servidor:** `dflash` é um servidor FastAPI + MLX em Python, rodando na
  porta 8010. Código principal: `/Users/samuelfajreldines/dev/dflash/scripts/local_api_server.py`.
- **Modelo:** Qwen3.6-35B-A3B (MoE, 3B active). Servido via speculative
  decoding com uma draft model. Chat template nativo é Jinja do tokenizer.
- **3 endpoints expostos:**
  - `/v1/responses` — OpenAI Responses API. Cliente: Codex CLI 0.122+.
    Handler: `responses()` por volta da linha 4219.
  - `/v1/messages` — Anthropic Messages API. Cliente: Claude Code CLI.
    Handler: `anthropic_messages()` por volta da linha 4167.
  - `/v1/chat/completions` — OpenAI Chat Completions. Clientes: Hermes
    Agent, aider, OpenHands, continue.dev, opencode, LiteLLM, LangChain.
    Handler: `chat_completions()` por volta da linha 4097.
- **Testes:** `/Users/samuelfajreldines/dev/dflash/tests/test_local_api_server.py`.
  Rodar com `.venv/bin/python3 -m pytest tests/ --tb=line -q --timeout=30`.
- **Servidor em produção local:** para restartar depois de mudanças,
  `./scripts/dflash.sh restart`. Porta 8010. Health check:
  `curl -sS http://127.0.0.1:8010/health`.
- **Comando de sintaxe rápida:** `python3 -c "import ast; ast.parse(open('scripts/local_api_server.py').read())"` ou equivalente para qualquer arquivo Python.
- **Encoding do tokenizer:** para a maioria dos itens, acesse
  `server._tokenizer` (tokenizer wrapper do mlx-lm) após garantir
  `server.ensure_loaded()`.

## Convenções usadas aqui

- Todos os itens começam `[ ]` (pendentes).
- Severidades: **CRITICAL** (endpoint quebrado), **HIGH** (features agentic
  ausentes), **MEDIUM** (qualidade/UX), **LOW** (polimento).
- Cada item contém: **Problema**, **Por que importa**, **Causa no código**,
  **Como corrigir** (passo-a-passo), **Arquivos/linhas**, **Acceptance
  criteria** (como verificar pronto), e **Riscos/edge cases**.
- Itens podem ter **Depende de:** apontando outros itens que devem ser
  feitos antes.
- Severidade `CRITICAL` tem prioridade absoluta — qualquer cliente real
  trava ou falha silenciosamente sem esses fixes.

---

## CRITICAL

### [ ] C1 — `/v1/chat/completions` encaminhar `tools` pro chat template

- **Problema:** Requests com `tools:[{type:"function",…}]` produzem prosa
  normal sem tool call algum. O endpoint é literalmente não-agentic hoje.
- **Por que importa:** aider, OpenHands, continue.dev, opencode, LiteLLM,
  LangChain — todos usam `/v1/chat/completions` com `tools`. Nenhum
  consegue invocar ferramentas pelo dflash atualmente.
- **Causa no código:** Em `scripts/local_api_server.py:4097-4165`, a função
  `chat_completions()` calcula `has_tools = bool(req.tools)` apenas para
  derivar sampling params, mas **não passa** `req.tools` para
  `server.generate(...)` ou `server.stream_chat_completions(...)`. Em
  `stream_chat_completions` (`:3010-3018`) a assinatura nem aceita o
  parâmetro `tools`; passa `None` para `_generation_worker` (`:3027`).
  O chat template do Qwen em `build_prompt` (`:2274-2286`) recebe `tools=None`
  e pula completamente o bloco `# Tools` do system prompt.
- **Como corrigir:**
  1. Em `stream_chat_completions` (`:3010-3018`), adicionar o parâmetro
     `tools: list[dict[str, Any]] | None = None` na assinatura.
  2. Propagar `tools` até `_generation_worker` no `Thread(args=...)` (`:3027`).
     Atualizar também a chamada correspondente em `_responses_generation_worker`
     e sua versão Anthropic para coerência.
  3. No handler `chat_completions()` (`:4097`), passar `tools=req.tools`
     (depois de normalização — ver H8) em todas as chamadas a
     `server.generate(...)` e `server.stream_chat_completions(...)`.
  4. Adicionar também `tool_choice` no schema `OpenAIChatRequest` (se ainda
     não estiver — ver C3/H7) e encaminhar.
- **Arquivos/linhas:** `scripts/local_api_server.py:3010-3168` (stream
  generator), `:4097-4165` (non-stream + stream handler), `:298-320` (schema).
- **Acceptance criteria:**
  - `curl http://127.0.0.1:8010/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"qwen3.6-35b-a3b-dflash-local","messages":[{"role":"user","content":"What is the weather in SF? Use the tool."}],"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]}'`
    retorna `choices[0].message.tool_calls` não-vazio.
  - `prompt_tokens` no `usage` sobe refletindo o schema das tools no prompt.
- **Riscos/edge cases:** Cuidar para não quebrar requests legados que
  passam `tools` vazio/None — comportamento inalterado nesse caso.
- **Depende de:** C3 (schema aceitar multi-part/null content), H8 (tool
  schema normalization).

---

### [ ] C2 — `/v1/chat/completions` emitir `tool_calls` em stream e non-stream

- **Problema:** Mesmo com C1 corrigido, clientes não veem tool calls. Em
  streaming, `_IncrementalVisibleTextStream(strip_edges=True)` **remove** a
  markup `<tool_call>` do texto visível mas nunca emite deltas estruturados.
  Em non-stream, `message.tool_calls` nunca é populado; texto cru (com ou
  sem markup) vai pro `content`.
- **Por que importa:** Spec OpenAI exige `message.tool_calls` (non-stream)
  ou `delta.tool_calls[]` chunks (stream) + `finish_reason:"tool_calls"`.
  Sem isso, nenhum cliente OpenAI-compatível detecta que uma tool foi
  chamada.
- **Causa no código:**
  - Non-stream em `:4140-4155`: faz `_, assistant_text = _strip_reasoning_blocks(result["text"])`
    e retorna `message.content = assistant_text`. Não chama `_parse_tool_calls`.
  - Stream em `:3010-3168`: só forwarda `delta.content`. A markup
    `<tool_call>` cai no conjunto `VISIBLE_HIDDEN_MARKERS` (`:78`) e é
    filtrada — o cliente nunca vê.
- **Como corrigir:**

  **Parte A — non-stream:**
  1. Depois de `server.generate(...)`, processar o texto:
     ```python
     reasoning, visible = _strip_reasoning_blocks(result["text"])
     assistant_text, tool_calls = _parse_tool_calls(visible)
     ```
  2. Se `tool_calls` não-vazio, montar:
     ```python
     message = {
         "role": "assistant",
         "content": assistant_text if assistant_text else None,
         "tool_calls": [
             {
                 "id": tc["call_id"],
                 "type": "function",
                 "function": {
                     "name": tc["name"],
                     "arguments": tc["arguments"],  # já é JSON string
                 },
             }
             for tc in tool_calls
         ],
     }
     finish_reason = "tool_calls"
     ```
  3. Se vazio, comportamento atual (content = assistant_text).
  4. Mapear também `finish_reason` corretamente: se `result["finish_reason"] == "length"` e havia tool markup truncada, forçar `"length"` mas limpar markup parcial antes.

  **Parte B — stream:**
  1. Substituir o pipeline `_IncrementalVisibleTextStream(strip_edges=True)`
     por um parser streaming de tool calls (classe nova
     `_IncrementalToolCallStream`). Estado:
     - `mode = "text"`: forwarda chunks como `delta.content`
     - `mode = "tool_call"` (ao ver `<tool_call>`): começa a acumular o
       corpo, emite `delta.tool_calls[{index:N, id:"call_<uuid>", type:"function", function:{name:"", arguments:""}}]`
       (primeiro frame com name/id, arguments vazio)
     - Conforme o JSON de `arguments` sai, emite `delta.tool_calls[{index:N, function:{arguments:"…chunk…"}}]`
       a cada ~20-40 bytes
     - Ao ver `</tool_call>`, fecha o bloco e volta ao `mode = "text"` (ou
       detecta próximo `<tool_call>` e incrementa `index`)
  2. No final do stream, emitir um chunk com
     `finish_reason: "tool_calls"` se houve pelo menos uma tool call, senão
     manter o finish_reason natural.
  3. Cuidado: o Qwen pode emitir o nome dentro de `<function=NAME>` (XML) ou
     dentro de `{"name":"NAME",…}` (JSON Hermes style). Usar `_parse_tool_calls`
     como fonte da verdade quando o bloco fecha, e só emitir os chunks de
     nome/id quando tiver certeza.
  4. Manter compat: se `stream_options.include_usage:true`, emitir ao final
     um chunk extra com `choices:[]` e `usage:{…}` antes do `[DONE]` (ver H10).
- **Arquivos/linhas:** `scripts/local_api_server.py:4140-4165` (non-stream),
  `:3010-3168` (stream generator e loop interno), `:891-956` (`_parse_tool_calls`),
  `:78` (`VISIBLE_HIDDEN_MARKERS`).
- **Acceptance criteria:**
  - Non-stream: resposta ao curl do C1 tem `tool_calls[0].function.name == "get_weather"` e `finish_reason == "tool_calls"`.
  - Stream: mesma requisição com `stream:true` emite em ordem:
    1. `chunk.delta.role = "assistant"`
    2. `chunk.delta.tool_calls[0] = {index:0, id:"call_…", type:"function", function:{name:"get_weather", arguments:""}}`
    3. vários chunks `chunk.delta.tool_calls[0] = {index:0, function:{arguments:"{\"city\":\"SF\""}}` (JSON streamed em partes)
    4. chunk final com `finish_reason:"tool_calls"`
    5. `[DONE]`
  - aider consegue detectar a tool call (testar com `aider --model openai/qwen3.6-35b-a3b-dflash-local --openai-api-base http://127.0.0.1:8010/v1` e fazer uma edição).
- **Riscos/edge cases:**
  - Parallel tool calls: múltiplos `<tool_call>` em um turno → enumerar `index=0,1,2,…`.
  - Tool call com arguments vazio `{}` → emitir um chunk `arguments:""` no mínimo (spec exige).
  - Markup malformado (tool call sem `</tool_call>` fechando) → tratar como
    texto normal; não emitir tool_call.
- **Depende de:** C1 (tools precisam chegar no modelo primeiro), C3 (schema).

---

### [ ] C3 — `OpenAIMessage` aceitar `content: null`, list-form e campos de tool

- **Problema:** Schema atual em `scripts/local_api_server.py:298-300`:
  ```python
  class OpenAIMessage(BaseModel):
      role: Literal["system", "user", "assistant", "tool"] | str
      content: str
  ```
  Isso retorna HTTP 422 em qualquer:
  - Assistant com `content:null` + `tool_calls:[…]` (obrigatório pela spec
    OpenAI quando há tool calls).
  - User com `content:[{"type":"text","text":"…"}, {"type":"image_url",…}]`
    (formato multimodal padrão).
  - Tool message com `tool_call_id` e `name` (dropado por `extra="ignore"`).
- **Por que importa:** Qualquer cliente agentic usa essas formas. aider,
  continue.dev, OpenHands, LiteLLM, Hermes todos mandam multi-part content
  e assistant-com-tool_calls em multi-turn.
- **Causa no código:** Schema Pydantic muito restritivo.
- **Como corrigir:**
  1. Reescrever `OpenAIMessage`:
     ```python
     class OpenAIMessage(BaseModel):
         role: Literal["system", "user", "assistant", "tool", "function", "developer"] | str
         content: str | list[dict[str, Any]] | None = None
         tool_calls: list[dict[str, Any]] | None = None
         tool_call_id: str | None = None
         name: str | None = None
         function_call: dict[str, Any] | None = None  # legacy
         reasoning_content: str | None = None  # OpenAI reasoning models
         model_config = ConfigDict(extra="allow")
     ```
  2. Adicionar ao `OpenAIChatRequest` (`:303-319`) todos os campos que
     clientes reais enviam e que hoje caem em `extra="ignore"`:
     ```python
     stop: str | list[str] | None = None
     stream_options: dict[str, Any] | None = None
     response_format: dict[str, Any] | None = None
     seed: int | None = None
     parallel_tool_calls: bool | None = None
     logprobs: bool | None = None
     top_logprobs: int | None = None
     logit_bias: dict[str, float] | None = None
     n: int | None = None
     user: str | None = None
     metadata: dict[str, Any] | None = None
     ```
  3. Criar helper `_normalize_openai_messages(messages: list[OpenAIMessage]) -> list[dict]`
     que:
     - Achata `content: list[{type:"text",text:"…"},…]` em string concatenada
       via `_extract_text_from_content` (já existe em `:485-507`).
     - Para assistant com `tool_calls`, preserva o campo no dict (o chat
       template do Qwen3 aceita `message.tool_calls` nativamente — verificar
       em `/Users/samuelfajreldines/dev/models/Qwen3.6-35B-A3B-4bit/chat_template.jinja:105-129`).
     - Para tool messages, mantém `tool_call_id` e `name`.
     - Converte role `developer` → `system`.
     - Trata `function_call` legacy convertendo pra `tool_calls[0]` com
       `id:"call_legacy_…"`.
  4. No handler `chat_completions()`, substituir
     `[m.model_dump() for m in req.messages]` por
     `_normalize_openai_messages(req.messages)`.
- **Arquivos/linhas:** `scripts/local_api_server.py:298-320` (schemas),
  `:4115` e `:4133` (handler), `:485-507` (`_extract_text_from_content`),
  `/Users/samuelfajreldines/dev/models/Qwen3.6-35B-A3B-4bit/chat_template.jinja:105-129` (referência).
- **Acceptance criteria:**
  - Request com `{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"foo","arguments":"{}"}}]}` não retorna 422.
  - Request com `{"role":"user","content":[{"type":"text","text":"oi"}]}` não retorna 422.
  - Multi-turn com tool round-trip: user → assistant+tool_calls → tool result → next turn, sem erro, modelo mantém contexto.
- **Riscos/edge cases:**
  - `content:str` legacy continua funcionando.
  - `content: list[…, {type:"image_url",…}]` — ver M10 (drop explicito ou
    placeholder, não silencioso).
- **Depende de:** nenhum (é pré-requisito de C1 e C2).

---

### [ ] C4 — `/v1/messages` streaming de tool_use não pode pendurar

- **Problema:** Request com `tools` + `stream:true` emite `message_start` +
  dezenas de `ping` e **nunca** emite `content_block_start/delta/stop` nem
  `message_stop`. Claude Code CLI e Anthropic SDK travam indefinidamente.
- **Por que importa:** É o caso de uso principal do Claude CLI headless.
  Sem fix, `claude --print --output-format=stream-json -p "use tool X"`
  é inutilizável contra dflash.
- **Causa no código:** `stream_anthropic_events` em `:3642-3921` usa
  `_IncrementalVisibleTextStream(strip_edges=False)` que trata `<tool_call>`
  como markup "oculto", suprimindo-o do output. Mas nunca faz a transição
  para emitir `content_block_start{type:"tool_use"}` + deltas estruturados.
  No final, quando a geração completa, há código que constrói
  `tool_use` blocks estaticamente mas **isso nunca chega ao stream** —
  o caminho streaming termina sem emitir os blocks.
- **Como corrigir:** Análogo a C2 mas com eventos Anthropic:
  1. Substituir `_IncrementalVisibleTextStream(strip_edges=False)` por um
     parser que transiciona entre modos "text" e "tool_use".
  2. No modo text, continuar emitindo `content_block_delta{type:"text_delta"}`
     como hoje.
  3. Ao detectar `<tool_call>`:
     - Fechar o content block de texto se estava aberto:
       `content_block_stop`, incrementar `content_block_index`.
     - Emitir `content_block_start`:
       ```json
       {"type":"content_block_start","index":N,"content_block":{"type":"tool_use","id":"toolu_<uuid>","name":"<pending>","input":{}}}
       ```
     - Nota: Anthropic emite o nome imediatamente no `content_block_start`.
       Isso significa que precisamos **bufferizar** até conseguir extrair o
       nome do `<function=NAME>` ou do JSON `{"name":"NAME"}`. Implementar:
       1. Ao ver `<tool_call>`, entrar em modo "buffering_header".
       2. Acumular até conseguir parsear o nome (via regex).
       3. Só então emitir o `content_block_start` com o nome correto.
       4. Daí em diante, emitir `content_block_delta{type:"input_json_delta", partial_json:"…"}`
          em chunks de ~20-40 bytes do JSON de `arguments`.
  4. Ao ver `</tool_call>`:
     - Emitir `content_block_stop`, incrementar index.
     - Voltar pro modo text OU para o próximo tool_call se houver.
  5. No final da geração (worker done), emitir:
     - `message_delta{stop_reason: "tool_use" if tool_calls_emitted else "end_turn", usage:{…}}`
     - `message_stop`
  6. **IMPORTANTE:** usar `id` com prefixo `toolu_` (ver M1), não `call_`.
- **Arquivos/linhas:** `scripts/local_api_server.py:3642-3921` (todo o
  streaming Anthropic), especialmente o loop de eventos em
  `_stream_anthropic_events_body`.
- **Acceptance criteria:**
  - `curl -N http://127.0.0.1:8010/v1/messages -H "Content-Type: application/json" -d '{"model":"qwen3.6-35b-a3b-dflash-local","max_tokens":200,"stream":true,"messages":[{"role":"user","content":"What is weather in SF?"}],"tools":[{"name":"get_weather","description":"…","input_schema":{…}}]}'`
    completa em &lt;30s com sequência:
    1. `event: message_start`
    2. `event: content_block_start` com `content_block.type == "tool_use"` e `content_block.id` iniciando com `toolu_`
    3. vários `event: content_block_delta` com `delta.type == "input_json_delta"` e `delta.partial_json` contendo partes do JSON
    4. `event: content_block_stop`
    5. `event: message_delta` com `delta.stop_reason == "tool_use"` e `usage` populado
    6. `event: message_stop`
  - Claude CLI (`claude --print ...`) executa sem travar.
- **Riscos/edge cases:**
  - Tool call com body muito curto (< 20 bytes) — emitir o JSON inteiro em
    um delta, ok.
  - Multiple tool calls em um turno — enumerar `index` corretamente;
    cada par start/stop é uma unidade.
  - Tool call malformado — se o buffer de header nunca consegue extrair
    nome, fallback pra emitir como texto normal e setar
    `stop_reason:"end_turn"`.
- **Depende de:** M1 (prefix `toolu_`), H10 (usage completo).

---

### [ ] C5 — `max_tokens` validar e clampar antes de travar

- **Problema:** Request com `max_tokens:10000, 100000, 999999` pendura o
  endpoint `/v1/chat/completions` sem resposta (curl timeout 30s+).
  `max_tokens:5000` completa (parece ter clamp interno em 4096 funcionando
  pra valores menores). Risco de DoS.
- **Por que importa:** Cliente mal-comportado (ou malicioso) trava o
  servidor. Também: clientes que mandam valores altos por hábito
  (tools que querem "muito espaço") recebem timeout em vez de erro claro.
- **Causa no código:** Provavelmente `_effective_max_tokens` (`:2290-2307`)
  tem lógica que não clampa para-valores-muito-grandes corretamente, ou
  algum path reserva/aloca memória com o valor cru antes do clamp. O queue
  pode entrar em espera nunca-satisfeita.
- **Como corrigir:**
  1. **Adicionar validação no schema Pydantic** para capturar cedo:
     ```python
     class OpenAIChatRequest(BaseModel):
         ...
         max_tokens: int | None = Field(default=None, ge=1, le=131072)
         max_completion_tokens: int | None = Field(default=None, ge=1, le=131072)
     ```
     (Fazer o mesmo em `AnthropicRequest` e `ResponsesRequest`.)
  2. **Adicionar clamp explícito** no handler, antes de `server.generate(...)`:
     ```python
     requested_max = req.max_completion_tokens or req.max_tokens or DEFAULT_MAX_TOKENS_FALLBACK
     hard_cap = max(1, min(server.max_tokens_limit or DEFAULT_MAX_TOKENS_FALLBACK, 131072))
     if requested_max > hard_cap:
         raise HTTPException(
             status_code=400,
             detail=f"max_tokens ({requested_max}) exceeds server limit ({hard_cap})"
         )
     ```
  3. **Investigar o hang real** — rodar com `py-spy dump` no servidor para
     ver em qual thread está preso quando `max_tokens:100000` é enviado.
     Pode ser:
     - Pre-allocação de buffer MLX (dump, verificar)
     - Queue tickets aguardando (verificar `_acquire_generation_turn`)
     - MLX alocando KV cache gigante
  4. Corrigir a causa raiz identificada, não só o sintoma.
- **Arquivos/linhas:** `scripts/local_api_server.py:303-319` (schemas),
  `:4115-4133` (handler), `:2290-2307` (`_effective_max_tokens`).
- **Acceptance criteria:**
  - `curl http://127.0.0.1:8010/v1/chat/completions -d '{"model":"qwen3.6-35b-a3b-dflash-local","messages":[{"role":"user","content":"hi"}],"max_tokens":100000}'`
    retorna HTTP 400 em &lt;1s com body
    `{"error":{"type":"invalid_request_error","code":"invalid_max_tokens","message":"max_tokens (100000) exceeds server limit (N)"}}`.
  - `max_tokens:5000` continua funcionando.
- **Riscos/edge cases:** Clientes que mandavam valor alto esperando clamp
  silencioso agora vão ver 400. Documentar. Alternativa: logar warning e
  clampar silenciosamente (menos honesto mas não-breaking).
- **Depende de:** H9 (error shapes corretos).

---

### [ ] C6 — Consertar `/v1/responses` `type:"custom"` tool output

- **Problema:** Tool registrada como
  `{type:"custom", name:"apply_patch", description:"…", format:{type:"grammar"}}`
  faz o modelo gerar ~54 tokens, mas output final é
  `{type:"message", output_text:"", content:[]}`. Nenhum `custom_tool_call`
  item é emitido. `apply_patch` freeform do Codex não funciona.
- **Por que importa:** Codex usa `apply_patch` com `type:"custom"` em
  muitas configurações. Sem isso, o modelo não consegue editar arquivos
  via Codex.
- **Causa no código:** `_convert_items_for_custom_tools` em `:767-813`
  detecta tools `type:"custom"` corretamente, mas a conversão do
  `function_call` interno em `custom_tool_call` está perdendo o payload.
  Hipóteses a investigar:
  1. O modelo não está emitindo o `<tool_call>` no formato esperado para
     tools custom — talvez precise de prompt específico.
  2. A conversão em `_convert_items_for_custom_tools` pega os
     `arguments` JSON e tenta extrair `input`/`patch`/`text`/`content`, mas
     nenhum desses keys existe, caindo no fallback de usar o raw JSON, que
     é vazio ou malformado.
  3. O stream emitter em `_stream_response_events_body` (`:2996-3060`)
     não lida com `type:"custom_tool_call"` items corretamente.
- **Como corrigir:**
  1. **Debug primeiro**: adicionar logging temporário em
     `_convert_items_for_custom_tools`:
     ```python
     _logger.info("custom_tools detected: %s", custom_names)
     _logger.info("input items: %s", items)
     _logger.info("output items: %s", converted)
     ```
     Rodar o teste do runtime report (teste 19) e ver o que acontece.
  2. **Verificar stream emission**: em `:2996-3060`, garantir que
     `pending_item["type"] == "custom_tool_call"` é tratado:
     ```python
     elif pending_item["type"] == "custom_tool_call":
         pending_item["status"] = "in_progress"
         pending_item["input"] = ""
     ```
     (Isso já existe em `:3078-3080` aparentemente; validar com os logs.)
  3. **Verificar eventos SSE**: o bloco `elif item["type"] == "custom_tool_call" and item.get("input"):`
     em `:3088-3120` precisa estar emitindo `response.custom_tool_call_input.delta/.done`.
  4. **Prompt engineering**: se o modelo não emite o formato esperado,
     pode ser que o Codex envie uma descrição da tool custom que o Qwen
     não consegue seguir. Capturar o que o Codex manda via
     `_trace_request` e ajustar o chat template ou system prompt.
  5. Adicionar teste unitário específico:
     ```python
     def test_custom_tool_call_conversion(self):
         items = [{"type":"function_call","name":"apply_patch","arguments":'{"input":"*** Begin Patch\\n…"}',"call_id":"call_1","id":"fc_1"}]
         tools = [{"type":"custom","name":"apply_patch","format":{"type":"grammar"}}]
         converted = _convert_items_for_custom_tools(items, tools)
         assert converted[0]["type"] == "custom_tool_call"
         assert converted[0]["input"] == "*** Begin Patch\n…"
     ```
- **Arquivos/linhas:** `scripts/local_api_server.py:767-813`
  (`_convert_items_for_custom_tools`, `_custom_tool_names`,
  `_make_custom_tool_call_item`), `:2996-3120` (stream emitter para custom
  tool call), `:2277` (call-site em `_generate_response_locked`).
- **Acceptance criteria:**
  - Teste unitário novo passa.
  - Runtime test 19 do conselho passa:
    ```
    curl http://127.0.0.1:8010/v1/responses -d '{"model":"qwen3.6-35b-a3b-dflash-local","input":"Fix foo.py to handle None","tools":[{"type":"custom","name":"apply_patch","description":"Apply a patch","format":{"type":"grammar"}}]}'
    ```
    retorna `output:[{type:"custom_tool_call", name:"apply_patch", input:"*** Begin Patch\n…", …}]` com `input` não-vazio.
- **Riscos/edge cases:** Se o modelo genuinamente não segue o formato
  grammar, pode precisar prompt engineering adicional. Verificar se o
  servidor recebe o `format` hint do Codex e o repassa no tool schema pro
  modelo.
- **Depende de:** nenhum. Fix isolado em `/v1/responses`.

---

## HIGH

### [ ] H1 — Judge + auto-followup loop portado pra Chat + Anthropic

- **Problema:** O mecanismo `_judge_response_needs_followup` +
  `RESPONSES_ACTION_FOLLOWUP_LIMIT` loop existe apenas em
  `_generate_response_locked` (`:2726-2784`). Clientes via
  `/v1/chat/completions` e `/v1/messages` não se beneficiam — o modelo
  pode parar em "plan-and-announce" sem ninguém forçar execução real.
- **Por que importa:** Todo o esforço anti plan-and-announce (prompt
  judge, logprob vote, tie-breaker, action prompt) só funciona no path
  Codex. Claude CLI e Hermes não têm essa rede de segurança.
- **Como corrigir:** Extrair o loop em helper protocol-agnostic.
  1. Criar dataclass `AgentLoopAdaptor`:
     ```python
     @dataclass
     class AgentLoopAdaptor:
         build_items_fn: Callable[[str, list[dict]], list[dict]]
         is_followup_candidate_fn: Callable[[dict, list[dict], list[dict] | None], bool]
         items_to_messages_fn: Callable[[list[dict]], list[dict]]
         action_prompt: str
     ```
  2. Criar `_run_agentic_generation_locked(server, messages, max_tokens, sampling, tools, adaptor, previous_response_id=None, capture_prompt_cache_state=False, should_stop=None)`
     extraindo o corpo atual de `_generate_response_locked` em função livre.
  3. `_generate_response_locked` vira thin wrapper com `RESPONSES_ADAPTOR`
     (construído a partir de `_build_output_items` +
     `_convert_items_for_custom_tools` + `_messages_from_output_items` +
     `RESPONSES_ACTION_PROMPT`).
  4. Criar `CHAT_COMPLETIONS_ADAPTOR` e `ANTHROPIC_ADAPTOR`. Ambos podem
     reutilizar `_build_output_items` + `_messages_from_output_items`
     porque produzem shape interno compatível.
  5. Nos handlers `chat_completions` e `anthropic_messages` (não-streaming
     primeiro — streaming fica pra depois), trocar `server.generate(...)`
     por `_run_agentic_generation_locked(..., adaptor=CHAT_COMPLETIONS_ADAPTOR)`.
  6. **Gate opt-in**: `LOCAL_DFLASH_CHAT_FOLLOWUP_JUDGE=0` default off.
     O judge consome tokens; não impor pra chat simples. Only fire quando
     `bool(tools)` (já há gate em `_response_is_followup_candidate`).
- **Arquivos/linhas:** `scripts/local_api_server.py:2726-2947`
  (extract + refactor), `:4097-4210` (chamar nos handlers).
- **Acceptance criteria:**
  - Com env `LOCAL_DFLASH_CHAT_FOLLOWUP_JUDGE=1` e tools presentes,
    request ao `/v1/chat/completions` que recebe "I'll call the tool now"
    sem emitir tool_call gera follow-up automático que força a tool call.
  - Sem env, comportamento atual inalterado.
  - Mesmo para `/v1/messages`.
  - Testes unitários existentes de `_generate_response_locked` continuam
    passando.
- **Riscos/edge cases:**
  - Judge consome ~128 tokens extra por turno; impacto de latência.
  - Streaming é mais complexo — fazer em item separado (ver final do TODO).
- **Depende de:** C1, C2, C3 (os handlers precisam estar emitindo tool
  calls estruturados para o judge conseguir avaliar).

---

### [ ] H2 — `_synthesize_orphan_tool_results` em Chat + Anthropic

- **Problema:** Helper existe em `:1354-1398` e é chamado apenas no handler
  `/v1/responses`. Se Chat ou Anthropic recebem conversa com assistant
  `tool_calls` sem `tool_result` subsequente, Qwen entra em prefill-only
  mode (empty-args loop a partir do turno 3+).
- **Por que importa:** Qualquer multi-turn agentic é vulnerável. Em
  produção real isso acontece quando cliente reconecta, quando sessão é
  resumida, quando há erro no tool execution.
- **Como corrigir:**
  1. Em `chat_completions()` (`:4133`), após `_normalize_openai_messages`,
     chamar:
     ```python
     messages = _synthesize_orphan_tool_results(messages)
     ```
  2. Em `anthropic_messages()` (`:4173`), logo após
     `messages, tools = _normalize_anthropic_messages(req)`:
     ```python
     messages = _synthesize_orphan_tool_results(messages)
     ```
  3. Guard: `_massage_responses_continuation_messages` existe em
     `/v1/responses` para não sintetizar quando o cliente legitimamente
     quer continuação pendente. Portar o gate:
     - Se a conversa termina em assistant com tool_calls **e o cliente
       está explicitamente continuando** (via `previous_response_id` ou
       equivalente), NÃO sintetizar.
     - Senão, sintetizar como hoje.
  4. Para Anthropic: sem `previous_response_id`, o sinal de "continuação"
     é `metadata` field ou header custom. Por ora, assumir sempre
     sintetizar (conservador).
- **Arquivos/linhas:** `scripts/local_api_server.py:1354-1398` (helper),
  `:4097-4210` (handlers), `:1190-1216` (`_massage_responses_continuation_messages`).
- **Acceptance criteria:**
  - Request ao `/v1/messages` com trailing assistant tool_use sem
    tool_result subsequente não entra em empty-args loop na próxima turno.
  - Mesma request ao `/v1/chat/completions`.
- **Riscos/edge cases:** Falso positivo em casos de continuação
  explícita — mitigar com o guard descrito.
- **Depende de:** C3.

---

### [ ] H3 — `TOOL_CALLING_RULES_PROMPT` injetado em Chat + Anthropic

- **Problema:** Bloco "Tool-calling rules (strict)" (definido em `:161-178`)
  é injetado apenas em `_normalize_responses_input` (`:1789-1790`). Claude
  CLI e clientes OpenAI passam por outros paths e não recebem, logo o
  modelo não tem instruções anti plan-and-announce.
- **Por que importa:** Essa foi a fix mais efetiva da Fase 1 contra
  loops "I will now X" sem execução. Portar pra outros paths previne
  o mesmo problema em Claude CLI / aider.
- **Como corrigir:**
  1. Extrair helper em `:1400` (ou próximo):
     ```python
     def _inject_tool_calling_rules(
         messages: list[dict[str, Any]],
         tools: list[dict[str, Any]] | None,
     ) -> list[dict[str, Any]]:
         if not (tools and TOOL_CALLING_RULES_ENABLED):
             return messages
         rules = TOOL_CALLING_RULES_PROMPT
         # Merge com system message existente ou inserir novo
         if messages and messages[0].get("role") == "system":
             existing = messages[0].get("content", "")
             if rules not in existing:
                 merged = existing.rstrip() + "\n\n" + rules
                 return [{**messages[0], "content": merged}, *messages[1:]]
             return messages
         return [{"role": "system", "content": rules}, *messages]
     ```
  2. Chamar de `_normalize_anthropic_messages` (antes do return em `:1519`).
  3. Chamar de `_normalize_openai_messages` (novo helper criado em C3).
  4. Remover a lógica ad-hoc em `_normalize_responses_input` (`:1789-1790`)
     e trocar por `messages = _inject_tool_calling_rules(messages, tools)`.
- **Arquivos/linhas:** `scripts/local_api_server.py:161-178`
  (TOOL_CALLING_RULES_PROMPT), `:1453-1519` e `:1789-1790` (normalizers).
- **Acceptance criteria:**
  - Servidor com `LOCAL_DFLASH_TOOL_CALLING_RULES=1` (default) — ao enviar
    request com tools ao `/v1/messages`, o prompt final visto no trace
    contém "Tool-calling rules (strict):".
  - Mesmo para `/v1/chat/completions`.
  - `LOCAL_DFLASH_TOOL_CALLING_RULES=0` desativa em todos os paths.
- **Riscos/edge cases:** Sistema já grande (Codex base_instructions ~10KB);
  verificar que o rules bloco cabe sem estourar context window na prática.
- **Depende de:** C3 (normalizer existe).

---

### [ ] H4 — `_classify_error_code` adaptado por protocolo

- **Problema:** `_classify_error_code` (`:1056-1079`) retorna códigos de
  Responses API (`context_length_exceeded`, `server_overloaded`, etc.).
  Anthropic e OpenAI têm taxonomias diferentes:
  - Anthropic: `invalid_request_error`, `rate_limit_error`,
    `overloaded_error`, `api_error`, `authentication_error`,
    `permission_error`, `not_found_error`, `request_too_large`.
  - OpenAI: `invalid_request_error`, `rate_limit_error`, `server_error`,
    `authentication_error`, etc. (close enough ao Responses).
- **Por que importa:** Clientes baseados em SDK oficial (Anthropic Python,
  OpenAI Python) fazem lógica de retry/classification baseada no
  `error.type`. Valores desconhecidos quebram strict validators.
- **Como corrigir:**
  1. Adicionar dois adaptors perto de `:1056`:
     ```python
     _ERROR_CODE_ANTHROPIC = {
         "context_length_exceeded": "invalid_request_error",
         "invalid_prompt": "invalid_request_error",
         "rate_limit_exceeded": "rate_limit_error",
         "server_overloaded": "overloaded_error",
         "server_error": "api_error",
     }

     def _classify_error_code_anthropic(message: str, exc: Exception | None = None) -> str:
         base = _classify_error_code(message, exc)
         return _ERROR_CODE_ANTHROPIC.get(base, "api_error")

     def _classify_error_code_openai(message: str, exc: Exception | None = None) -> str:
         return _classify_error_code(message, exc)  # já OAI-flavored
     ```
  2. Substituir todos os usos de `_classify_error_code` nos paths
     Anthropic pelo adaptor `_classify_error_code_anthropic` (buscar em
     `stream_anthropic_events` e handler `anthropic_messages`).
  3. Usar `_classify_error_code_openai` no path Chat.
  4. Plugar tanto nos handlers HTTP (via H9) quanto nos error-emission
     frames dos streams.
- **Arquivos/linhas:** `scripts/local_api_server.py:1056-1079`, e wherever
  error emission happens (`:3079-3083` chat stream, `:3743-3754`
  anthropic stream, handlers HTTPException paths).
- **Acceptance criteria:**
  - Request ao `/v1/messages` que causa `PromptTooLargeError` retorna
    `{"type":"error","error":{"type":"invalid_request_error","message":"…"}}`.
  - Mesmo para `/v1/chat/completions` mas com shape OpenAI.
- **Riscos/edge cases:** Códigos desconhecidos caem em default (`api_error`
  / `server_error`). Não retornar strings fora das enums oficiais.
- **Depende de:** H9.

---

### [ ] H5 — Reasoning round-trip (`<think>` + thinking blocks) em Chat + Anthropic

- **Problema:** `_normalize_responses_input` aceita `{type:"reasoning"}` e
  re-injeta como `<think>…</think>` no histórico (`:1853-1877`); também
  emite eventos `reasoning_summary_part.added` + `reasoning_summary_text.delta`
  no stream. Anthropic `{type:"thinking", thinking, signature}` é
  silenciosamente dropped; OpenAI `reasoning_content` idem.
- **Por que importa:** Multi-turn reasoning perde continuidade.
  Claude CLI com `/thinking` renderiza painel vazio. continue.dev e
  LiteLLM suportam `reasoning_content` mas não recebem.
- **Como corrigir:**

  **Parte A — Anthropic (input):**
  1. Em `_normalize_anthropic_messages` (`:1466-1493`, branch do assistant),
     detectar bloco `type:"thinking"`:
     ```python
     if block_type == "thinking":
         reasoning_text = block_data.get("thinking") or ""
         if reasoning_text:
             text_parts.append(f"<think>\n{reasoning_text}\n</think>\n")
         continue
     ```
  2. Também preservar `signature` opacamente em struct auxiliar (não é
     verificado localmente, mas fica pronto caso se vire proxy pra Claude
     real).

  **Parte B — Anthropic (output):**
  1. No `stream_anthropic_events`, depois de detectar que há
     `reasoning_text` no `result`, emitir antes do bloco de texto:
     ```
     content_block_start {index:0, content_block:{type:"thinking", thinking:""}}
     content_block_delta {index:0, delta:{type:"thinking_delta", thinking:"…"}}
     content_block_delta {index:0, delta:{type:"signature_delta", signature:"local-qwen"}}
     content_block_stop {index:0}
     ```
  2. Incrementar `content_block_index` antes dos blocos de texto seguintes.

  **Parte C — Chat (input):**
  1. Em `_normalize_openai_messages`, se assistant message tem
     `reasoning_content`, splice-ar como `<think>…</think>` no começo do
     `content` final.

  **Parte D — Chat (output):**
  1. No `stream_chat_completions`, separar o reasoning content do texto
     visível (usar `_strip_reasoning_blocks`).
  2. Emitir chunks `delta.reasoning_content: "…"` paralelamente a
     `delta.content` (clientes tipo continue.dev leem isso).
- **Arquivos/linhas:** `scripts/local_api_server.py:1453-1519`
  (anthropic normalizer), `:1535-1562` (`_build_anthropic_content_blocks`),
  `:3642-3921` (anthropic stream), `:3010-3168` (chat stream).
- **Acceptance criteria:**
  - Response ao `/v1/messages` com reasoning presente no modelo tem
    `content[0].type == "thinking"` seguido por `content[1].type == "text"`.
  - Stream emite `content_block_start` com `type:"thinking"` antes do texto.
  - `/v1/chat/completions` stream emite `delta.reasoning_content` além do
    `delta.content`.
  - Multi-turn: assistant com `{type:"thinking"}` replayado é aceito e
    injeta `<think>` no prompt final.
- **Riscos/edge cases:** `signature` Anthropic é HMAC real na Claude —
  nosso "local-qwen" será rejeitado se algum dia proxy-arcar pra Claude
  real. Documentar como limitação.
- **Depende de:** C3 (schema), C4 (anthropic streaming funcional).

---

### [ ] H6 — `tool_choice` enforcement em `/v1/messages`

- **Problema:** `AnthropicRequest.tool_choice` é aceito mas nunca efetivamente
  forçado. `{type:"tool", name:"X"}` deveria forçar o modelo a chamar
  aquela tool; hoje é ignorado silenciosamente (runtime test 15 confirmou).
- **Por que importa:** Claude CLI e agent frameworks usam `tool_choice`
  para restringir comportamento. Sem enforcement, behaviour é imprevisível.
- **Como corrigir:**
  1. **`tool_choice:"auto"`:** comportamento atual (default), nada a fazer.
  2. **`tool_choice:"any"`:** injetar no system prompt: "You MUST call
     exactly one tool this turn. Do not produce any text outside
     `<tool_call>…</tool_call>`." — ou usar constrained decoding.
  3. **`tool_choice:{type:"tool", name:"X"}`:** estratégia mais robusta é
     **prefill**: pré-popular `<tool_call>\n{"name":"X","arguments":` no
     prompt assistant e deixar o modelo completar. Requer suporte em
     `stream_generate` pra fazer prefill do response. Se não for viável,
     fallback de prompt injection forte.
  4. **`tool_choice:"none"`:** remover o bloco `# Tools` do system prompt
     (passar `tools=None` efetivamente quando construir o prompt, mas
     manter as tools no registry para fins de sampling).
  5. Propagar `tool_choice` desde o handler até o prompt builder.
     Passo a passo:
     - Adicionar parâmetro `tool_choice` ao schema (já é
       `dict[str, Any] | None`).
     - Passar pelo pipeline: `anthropic_messages` → `_normalize_anthropic_messages`
       → `server.generate(..., tool_choice=...)` → `build_prompt(...)`.
     - `build_prompt` aplica as estratégias 1-4.
- **Arquivos/linhas:** `scripts/local_api_server.py:320-338` (schema),
  `:1453-1519` (normalizer), `:2274-2286` (build_prompt),
  `:4167-4210` (handler).
- **Acceptance criteria:**
  - Request com `tool_choice:{type:"tool", name:"get_weather"}` produz
    response com `content[0].type == "tool_use"` e `content[0].name == "get_weather"`,
    independente da pergunta do usuário.
  - `tool_choice:"none"` + pergunta sobre tool → modelo não chama tool.
  - `tool_choice:"any"` + pergunta genérica → modelo chama alguma tool.
- **Riscos/edge cases:** Prefill via `stream_generate` precisa checar que
  o chat template aceita prefill no role assistant (verificar Jinja do
  Qwen3). Se não, usar só prompt injection.
- **Depende de:** C3, C4.

---

### [ ] H7 — Campos adicionais em `OpenAIChatRequest`

- **Problema:** `OpenAIChatRequest` hoje só tem um subconjunto dos campos
  que clientes reais enviam; o resto cai em `extra="ignore"` silenciosamente.
- **Por que importa:**
  - `stop` → Hermes, Codex, aider mandam; sem propagar, modelo pode
    ultrapassar stops desejados.
  - `stream_options.include_usage` → continue.dev sempre manda; sem
    honrar, custo calculado zera.
  - `response_format: {type:"json_object"}` → aider usa pra JSON mode.
  - `seed` → determinismo em testes.
  - `parallel_tool_calls`, `logprobs`, `top_logprobs`, `logit_bias`, `n`,
    `user`, `metadata`.
- **Como corrigir:** Parte do trabalho já feita em C3 (adicionar campos ao
  schema). Aqui focamos em **propagar** cada um:
  1. `stop` → passar pra `SamplingParams` ou direto pra `stream_generate`
     como lista de stop sequences; usar o stop trie do mlx-lm.
  2. `stream_options.include_usage` → gate em `stream_chat_completions`:
     se `True`, após o `finish_reason` chunk, emitir um chunk adicional:
     ```json
     {"id":…, "object":"chat.completion.chunk", "choices":[], "usage":{…}}
     ```
     antes de `[DONE]`.
  3. `response_format:{type:"json_object"}` → prepend instrução ao system:
     "You MUST respond with valid JSON. Do not wrap in markdown."
  4. `response_format:{type:"json_schema", json_schema:{…}}` → idem +
     passar schema se implementarmos constrained decoding (v2).
  5. `seed` → `mx.random.seed(seed)` antes da geração (opcional, custo
     baixo).
  6. `parallel_tool_calls` — gate no prompt: se false, "Emit at most one
     tool call per turn."
  7. `logprobs`, `top_logprobs` — hoje Qwen via mlx-lm não expõe; retornar
     `null` no response e documentar.
  8. `logit_bias` → pode ser aplicado via mlx-lm custom sampler; defer.
  9. `n > 1` → não suportamos; retornar 400 se n > 1.
  10. `user`, `metadata` → guardar em trace, não usar.
- **Arquivos/linhas:** `scripts/local_api_server.py:303-319` (schema),
  `:3010-3168` (stream com include_usage), handler `:4097-4165`.
- **Acceptance criteria:**
  - Request com `stream:true, stream_options:{include_usage:true}` emite
    chunk com `choices:[]` e `usage:{…}` antes de `[DONE]`.
  - Request com `stop:["\n\n"]` para no primeiro `\n\n` gerado.
  - Request com `response_format:{type:"json_object"}` retorna conteúdo
    parseável como JSON (best-effort).
  - Request com `seed:42` em dois chamados idênticos retorna outputs
    idênticos.
- **Riscos/edge cases:** `stop` com lista vazia ou strings muito curtas
  pode gerar early-stop imediato; validar.
- **Depende de:** C3.

---

### [ ] H8 — Tool schema normalization unificada

- **Problema:** `_normalize_anthropic_tools` (`:1270-1320`) converte
  Anthropic `{name, description, input_schema}` pra internal
  `{type:"function", function:{name, description, parameters}}`. Mas
  `chat_completions` e `responses` passam `req.tools` as-is. Se cliente
  manda tool no shape cruzado, quebra silenciosamente (prompt malformado).
- **Por que importa:** Proxies, LiteLLM, e clientes híbridos enviam tools
  em qualquer shape. Normalizar uma vez, aceitar todos.
- **Como corrigir:**
  1. Criar `_normalize_tool_schemas(tools: list[dict] | None) -> list[dict]`:
     ```python
     def _normalize_tool_schemas(tools):
         if not tools:
             return []
         out = []
         for t in tools:
             if not isinstance(t, dict):
                 continue
             # Custom tool (Codex freeform) — preserve
             if t.get("type") == "custom":
                 out.append(t)
                 continue
             # Anthropic shape → OpenAI function shape
             if "input_schema" in t and "parameters" not in t:
                 out.append({
                     "type": "function",
                     "function": {
                         "name": t.get("name"),
                         "description": t.get("description", ""),
                         "parameters": t["input_schema"],
                     },
                 })
                 continue
             # Already OpenAI function shape (nested)
             if t.get("type") == "function" and isinstance(t.get("function"), dict):
                 out.append(t)
                 continue
             # Flat OpenAI shape (some Responses clients)
             if "name" in t and "parameters" in t:
                 out.append({
                     "type": "function",
                     "function": {
                         "name": t["name"],
                         "description": t.get("description", ""),
                         "parameters": t["parameters"],
                     },
                 })
                 continue
             # Unknown — pass through so downstream fails visibly, not silently
             out.append(t)
         return out
     ```
  2. Chamar de `_normalize_responses_input` (substituindo `tools = req.tools or []`).
  3. Chamar do handler `chat_completions` ao montar tools.
  4. Em `_normalize_anthropic_messages`, substituir o uso de
     `_normalize_anthropic_tools` (que pode ficar como thin wrapper) por
     `_normalize_tool_schemas`.
- **Arquivos/linhas:** `scripts/local_api_server.py:1270-1320`
  (existing anthropic), `:1453-1519` (anthropic normalizer), `:1779`
  (responses normalizer), `:4097-4165` (chat handler).
- **Acceptance criteria:**
  - Request ao `/v1/chat/completions` com tool no shape Anthropic
    (`{name, description, input_schema}`) é aceito e modelo chama a tool.
  - Request ao `/v1/responses` com tool no shape nested OpenAI
    (`{type:"function", function:{name, parameters}}`) é aceito.
  - Tools `{type:"custom"}` passam sem modificação (C6 depends).
- **Riscos/edge cases:** Tool sem `name` — logar warning e dropar (ou
  passar através e deixar modelo ignorar).
- **Depende de:** nenhum. Pré-requisito de C1, H2.

---

### [ ] H9 — Error shapes corretos por protocolo

- **Problema:** FastAPI retorna `{"detail":"…"}` em 400/404/422. Clientes
  OpenAI esperam `{"error":{"message","type","code","param"}}`; Anthropic
  espera `{"type":"error","error":{"type","message"}}`.
- **Por que importa:** SDKs oficiais fazem retry/classification baseado
  na shape do erro. Strict validators rejeitam. Streaming errors hoje
  sofrem do mesmo problema.
- **Como corrigir:**
  1. Registrar exception handler global que decide shape por rota:
     ```python
     from fastapi import Request as _Req
     from fastapi.exceptions import RequestValidationError

     @app.exception_handler(HTTPException)
     async def _http_exc_handler(request: _Req, exc: HTTPException):
         path = request.url.path
         msg = str(exc.detail)
         if path.startswith("/v1/messages"):
             err_type = _classify_error_code_anthropic(msg, exc)
             return JSONResponse(
                 status_code=exc.status_code,
                 content={"type":"error","error":{"type":err_type,"message":msg}},
             )
         if path.startswith("/v1/chat/completions") or path.startswith("/v1/responses"):
             code = _classify_error_code_openai(msg, exc)
             return JSONResponse(
                 status_code=exc.status_code,
                 content={"error":{"type":"invalid_request_error","code":code,"message":msg}},
             )
         return JSONResponse(status_code=exc.status_code, content={"detail":msg})

     @app.exception_handler(RequestValidationError)
     async def _validation_exc_handler(request: _Req, exc: RequestValidationError):
         # shape error por rota como acima
         ...
     ```
  2. Nos streaming paths, trocar `{"error":{"message":payload}}` por
     shape completo com `type` e `code`.
  3. Registrar os handlers em `create_app`.
- **Arquivos/linhas:** `scripts/local_api_server.py:3850+` (create_app),
  `:3079-3083` (chat stream error), `:3743-3754` (anthropic stream error).
- **Acceptance criteria:**
  - Request ao `/v1/chat/completions` com body inválido retorna 400 com
    `{"error":{"type":"invalid_request_error","code":…,"message":…}}`.
  - Mesma coisa `/v1/messages` retorna `{"type":"error","error":…}`.
  - Stream error em `/v1/messages` emite `event: error` com shape correto.
  - SDK oficial da Anthropic (`import anthropic; client.messages.create(...)`)
    levanta `anthropic.BadRequestError` em vez de `anthropic.APIStatusError`.
- **Riscos/edge cases:** 500s inesperados caem no fallback `api_error` /
  `server_error`. Não vazar stacktrace no `detail`.
- **Depende de:** H4.

---

### [ ] H10 — `usage` completo em todos os paths

- **Problema:**
  - Anthropic `usage` só tem `input_tokens, output_tokens`. Faltam
    `cache_creation_input_tokens, cache_read_input_tokens` (SDK Python
    oficial declara nullable mas strict paths quebram).
  - Chat `usage` só tem `prompt_tokens, completion_tokens, total_tokens`.
    Faltam `prompt_tokens_details.cached_tokens,
    completion_tokens_details.reasoning_tokens`.
  - `message_start.usage.input_tokens` em streaming Anthropic sempre 0
    (deveria ser prompt_tokens real).
- **Por que importa:** Cost tracking (LiteLLM, continue.dev), rate-limit
  awareness, debugging.
- **Como corrigir:**
  1. Generalizar `_response_usage` (`:857-874`) para aceitar protocolo:
     ```python
     def _protocol_usage(result, protocol: Literal["responses","chat","anthropic"]):
         input_tokens = int(result.get("prompt_tokens") or 0)
         output_tokens = int(result.get("generated_tokens") or 0)
         cached_input_tokens = int(result.get("reused_prefix_tokens") or 0)
         reasoning_output_tokens = int(result.get("reasoning_tokens") or 0)
         if protocol == "anthropic":
             return {
                 "input_tokens": input_tokens,
                 "output_tokens": output_tokens,
                 "cache_creation_input_tokens": 0,
                 "cache_read_input_tokens": cached_input_tokens,
             }
         if protocol == "chat":
             return {
                 "prompt_tokens": input_tokens,
                 "completion_tokens": output_tokens,
                 "total_tokens": input_tokens + output_tokens,
                 "prompt_tokens_details": {"cached_tokens": cached_input_tokens, "audio_tokens": 0},
                 "completion_tokens_details": {"reasoning_tokens": reasoning_output_tokens, "audio_tokens": 0, "accepted_prediction_tokens": 0, "rejected_prediction_tokens": 0},
             }
         # responses — já está correto
         return _response_usage(result)
     ```
  2. Usar `_protocol_usage` em todos os pontos que emitem usage:
     - `anthropic_messages` handler (non-stream)
     - `stream_anthropic_events` — `message_delta.usage`
     - `chat_completions` handler (non-stream)
     - `stream_chat_completions` — chunk final quando `include_usage:true`
  3. Em `message_start` Anthropic, popular `input_tokens` de fato
     (não deixar 0). Como o streaming emite `message_start` antes do
     prefill? Se `prompt_tokens` não estiver calculado ainda, emitir 0 e
     sobrescrever no `message_delta` final. Verificar o worker thread;
     pode já ter `prompt_tokens` quando o primeiro token sai.
- **Arquivos/linhas:** `scripts/local_api_server.py:857-874`
  (`_response_usage`), handlers e streams já citados.
- **Acceptance criteria:**
  - `usage` em non-stream `/v1/messages` tem os 4 campos Anthropic.
  - `usage` em non-stream `/v1/chat/completions` tem os campos detailed.
  - `message_delta.usage` em stream Anthropic inclui `input_tokens`.
  - Com `stream_options:{include_usage:true}`, último chunk Chat tem usage.
- **Riscos/edge cases:** `reused_prefix_tokens` em non-tool requests pode
  ser 0; OK.
- **Depende de:** nenhum.

---

## MEDIUM

### [ ] M1 — `tool_use.id` com prefix `toolu_` em `/v1/messages`

- **Problema:** IDs de tool_use gerados por `_make_function_call_item`
  usam prefixo `call_…`. Anthropic exige `^toolu_`. SDKs strict rejeitam.
- **Como corrigir:**
  1. Adicionar helper `_new_anthropic_tool_use_id()` retornando
     `f"toolu_{uuid.uuid4().hex}"`.
  2. Em `_build_anthropic_content_blocks` (`:1535-1562`), quando construir
     tool_use blocks, usar esse helper para o campo `id`.
  3. No streaming (C4), idem.
- **Arquivos/linhas:** `scripts/local_api_server.py:625-639`
  (`_make_function_call_item` mantém `call_`), `:1535-1562`
  (`_build_anthropic_content_blocks`).
- **Acceptance criteria:** Response tool_use tem `id` começando com `toolu_`.
- **Depende de:** C4.

---

### [ ] M2 — Streaming de `input_json_delta` chunked

- **Problema:** Em Anthropic streaming, quando fix C4 está no lugar, o
  JSON inteiro pode sair de uma vez só em um delta. Docs Anthropic
  esperam chunks parciais.
- **Como corrigir:** No parser tool_use do C4, emitir `input_json_delta`
  em pedaços de ~20-40 bytes conforme o JSON sai do modelo, não no fim.
- **Depende de:** C4.

---

### [ ] M3 — `message_start.usage.input_tokens` populado

- **Problema:** Sempre 0. Deveria ser prompt_tokens real.
- **Como corrigir:** Ver H10 parte 3.
- **Depende de:** H10.

---

### [ ] M4 — `is_error:true` no tool_result honrado

- **Problema:** Anthropic e Responses dropam `is_error` silenciosamente.
  Modelo não sabe que a tool falhou.
- **Como corrigir:**
  1. Em `_normalize_anthropic_messages` (`:1503-1509`), antes de
     `_make_tool_message`, se `block_data.get("is_error")`, prefixar
     content com `[tool error] ` ou wrappar em `<tool_response error="true">…</tool_response>`.
  2. Atualizar `_make_tool_message` (`:529-538`) para aceitar `is_error`
     e aplicar prefix.
  3. Em `_normalize_responses_input`, aplicar o mesmo tratamento no
     branch `function_call_output` se o call-site supportar.
- **Arquivos/linhas:** `scripts/local_api_server.py:529-538`, `:1503-1509`.
- **Acceptance criteria:** Tool result com `is_error:true` faz o modelo
  reconhecer a falha ("It seems the tool returned an error...") em vez
  de tratar como sucesso.

---

### [ ] M5 — Chat streaming emitir `role` antes de content

- **Problema:** Primeiro chunk do `/v1/chat/completions` emite
  `delta:{role:"assistant", content:"Hello"}` junto. OpenAI real emite
  `delta:{role:"assistant"}` sozinho primeiro, depois começa content.
- **Como corrigir:** Em `stream_chat_completions`, emitir um chunk
  extra no início:
  ```python
  yield _data_line({
      "id": completion_id, "object":"chat.completion.chunk",
      "created":created, "model":self.model_name,
      "choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":None}]
  })
  ```
  Antes de consumir o primeiro chunk do visible_stream.
- **Depende de:** C2.

---

### [ ] M6 — Heartbeats gated por wall-clock

- **Problema:** "Say hi" produz 12+ empty deltas no Chat stream, 19+ pings
  Anthropic, 22+ `response.in_progress` Responses. Muito ruído.
- **Como corrigir:** Em cada streaming loop, manter `last_heartbeat_at`
  timestamp. Só emitir heartbeat se `time.time() - last_heartbeat_at > 1.0`.
  Reset ao emitir chunk real.
- **Arquivos/linhas:** `scripts/local_api_server.py:3010-3168` (chat),
  `:3222-3470` (responses), `:3642-3921` (anthropic).

---

### [ ] M7 — `/v1/messages/count_tokens` usar tokenizer real

- **Problema:** `anthropic_count_tokens` em `:4213-4217` faz
  `len(part.split())`. Subestima ~2-4×. Cliente pode usar o retorno pra
  budget check pré-request, e depois a request real estoura.
- **Como corrigir:**
  ```python
  @app.post("/v1/messages/count_tokens")
  def anthropic_count_tokens(req: AnthropicCountTokensRequest) -> dict[str, Any]:
      messages, _ = _normalize_anthropic_messages(req)
      server.ensure_loaded()
      # Usar o tokenizer real com o chat template
      prompt = server.build_prompt(messages, tools=None)
      tokens = server._tokenizer.encode(prompt, add_special_tokens=False)
      return {"input_tokens": len(tokens)}
  ```
- **Depende de:** nenhum.

---

### [ ] M8 — Remover `metrics` / `raw_text` do top-level response

- **Problema:** Non-stream responses em `/v1/chat/completions` e
  `/v1/messages` têm `metrics` e `raw_text` no top level. Strict clients
  (Pydantic `extra="forbid"`) rejeitam.
- **Como corrigir:**
  1. Remover `metrics` e `raw_text` do response final.
  2. Expor via header `X-Dflash-Metrics-Json: <json>` (debug only) ou
     via query param `?debug=1`.
  3. Ou só logar em `_trace_event`.
- **Arquivos/linhas:** `scripts/local_api_server.py:4161-4162` (chat),
  handler anthropic equivalente.

---

### [ ] M9 — Role `developer` → `system` em Chat + Anthropic

- **Problema:** Só Responses coerce `developer` → `system`. Chat/Anthropic
  rejeitam ou passam through sem mapeamento.
- **Como corrigir:** Em `_normalize_openai_messages` (novo de C3) e em
  `_normalize_anthropic_messages`, antes de processar role:
  ```python
  role = msg.role if msg.role != "developer" else "system"
  ```
- **Depende de:** C3.

---

### [ ] M10 — Image/document content blocks tratados explicitamente

- **Problema:** Content com `type:"image"` ou `type:"document"` é
  silenciosamente droppado por `_extract_text_from_content`.
- **Como corrigir:** Adicionar branches no `_extract_text_from_content`
  (`:485-507`):
  ```python
  elif item_type == "image":
      return "[image omitted: unsupported by local model]"
  elif item_type == "document":
      return "[document omitted: unsupported by local model]"
  ```
  Alternativa mais honesta: retornar 400 com
  `{error:"Image content not supported by local model"}`.
  Escolher a primeira (mais permissiva) para não quebrar multi-turn onde
  user já mandou uma imagem em turno anterior.
- **Depende de:** nenhum.

---

## LOW

### [ ] L1 — `/v1/props.chat_template` refletir template real

- **Problema:** `_llamacpp_props_payload` retorna
  `"chat_template":"qwen-tool-use"` mas `build_prompt` usa o Jinja do
  tokenizer direto. Cliente llama.cpp-compat que sniffa pode tomar
  decisões erradas.
- **Como corrigir:** Trocar para `"chatml"` ou `"qwen3"`, ou adicionar
  variável `TEMPLATE_ID` detectada do `tokenizer_config.json`.
- **Arquivos/linhas:** `scripts/local_api_server.py:1385-1398`.

---

### [ ] L2 — `system_fingerprint` em chunks Chat

- **Problema:** Ausente. LiteLLM usa para audit trail.
- **Como corrigir:** Em `stream_chat_completions`, incluir em todo chunk:
  ```python
  "system_fingerprint": f"local-dflash-{hashlib.sha256(server.model_name.encode()).hexdigest()[:16]}"
  ```
  Calcular uma vez e reusar.

---

### [ ] L3 — `service_tier` echo

- **Problema:** Campo silenciosamente ignorado.
- **Como corrigir:** Echo back no response com default `"default"`.
- **Depende de:** nenhum.

---

### [ ] L4 — Sampling presets opt-in (Hermes, Qwen, off)

- **Problema:** Defaults Qwen (`presence_penalty=1.5`) podem conflitar com
  recipe Hermes (0).
- **Como corrigir:** Flag `LOCAL_DFLASH_SAMPLING_PRESET=qwen|hermes|off`
  seleciona conjunto de defaults em `SamplingParams.for_request`. Override
  por request continua funcionando (passa sobre o preset).
- **Arquivos/linhas:** `scripts/local_api_server.py:181-289` (sampling defaults).

---

## Item bonus — streaming com auto-followup judge

### [ ] STREAM-H1 — Judge loop em streaming paths

- **Problema:** H1 cobre non-stream. Streaming com judge é complexo porque
  precisa reabrir o stream após o judge dizer INCOMPLETE.
- **Como corrigir:**
  1. Gate: `LOCAL_DFLASH_STREAMING_FOLLOWUP_JUDGE=0` default off.
  2. Quando enabled: streaming emite chunks normais até
     `finish_reason/stop_reason`. Roda judge. Se INCOMPLETE:
     - Emite um chunk "continuation separator" (no-op na wire, apenas
       reseta state).
     - Concatena `RESPONSES_ACTION_PROMPT` como user message.
     - Re-gera e emite chunks como se fosse o mesmo stream.
  3. Client vê um stream contínuo de deltas. No fim emite o `finish_reason`
     real.
- **Depende de:** H1.
- **Complexidade:** alta. Deixar por último.
