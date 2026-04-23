 Conselho Consolidado — dflash × 3 APIs
                                                                                                                        
  Severidade CRITICAL (quebram uso agentic real)                                                                   
                                                                                                                        
  ┌─────┬─────────────────────────────────────────────────────────────────────────────────────┬───────────┬─────────┐
  │  #  │                                         Bug                                         │    API    │  Fonte  │
  ├─────┼─────────────────────────────────────────────────────────────────────────────────────┼───────────┼─────────┤
  │ C1  │ /v1/chat/completions não encaminha tools pro chat template — endpoint não consegue  │ Chat      │ B, C, F │
  │     │ fazer tool call algum                                                               │           │         │
  ├─────┼─────────────────────────────────────────────────────────────────────────────────────┼───────────┼─────────┤   
  │ C2  │ tool_calls nunca emitido (non-stream + stream) — markup <tool_call> é strippado do  │ Chat      │ B, C, F │
  │     │ stream mas nunca convertido pra delta.tool_calls estruturado                        │           │         │   
  ├─────┼─────────────────────────────────────────────────────────────────────────────────────┼───────────┼─────────┤   
  │ C3  │ OpenAIMessage.content: str rejeita null/list — HTTP 422 em qualquer mensagem        │ Chat      │ B, C,   │   
  │     │ assistant com tool_calls ou multi-content                                           │           │ E, F    │   
  ├─────┼─────────────────────────────────────────────────────────────────────────────────────┼───────────┼─────────┤
  │ C4  │ /v1/messages streaming de tool_use PENDURA — só emite ping infinito, nunca          │ Messages  │ F       │
  │     │ content_block_start/delta/stop — Claude CLI trava indefinidamente                   │           │         │   
  ├─────┼─────────────────────────────────────────────────────────────────────────────────────┼───────────┼─────────┤
  │ C5  │ max_tokens ≥ 10000 trava /v1/chat/completions sem retornar nada (30s+ timeout) —    │ Chat      │ F       │   
  │     │ risco de DoS                                                                        │           │         │
  ├─────┼─────────────────────────────────────────────────────────────────────────────────────┼───────────┼─────────┤   
  │ C6  │ /v1/responses type:"custom" tool produz output vazio — apply_patch freeform do      │ Responses │ F       │   
  │     │ Codex silenciosamente não funciona; _convert_items_for_custom_tools tá quebrado     │           │         │
  └─────┴─────────────────────────────────────────────────────────────────────────────────────┴───────────┴─────────┘   
                                                                                                                        
  Severidade HIGH (otimizações Codex faltam nas outras APIs)      
                                                                                                                        
  ┌─────┬──────────────────────────────────────────────────────────────────────────────────────┬────────────────────┐   
  │  #  │                                      Otimização                                      │     Portar pra     │
  ├─────┼──────────────────────────────────────────────────────────────────────────────────────┼────────────────────┤   
  │ H1  │ Judge + auto-followup loop (RESPONSES_ACTION_FOLLOWUP_LIMIT)                         │ Chat, Messages     │   
  ├─────┼──────────────────────────────────────────────────────────────────────────────────────┼────────────────────┤
  │ H2  │ _synthesize_orphan_tool_results (Qwen template guard)                                │ Chat, Messages (~2 │   
  │     │                                                                                      │  call-sites)       │   
  ├─────┼──────────────────────────────────────────────────────────────────────────────────────┼────────────────────┤
  │ H3  │ TOOL_CALLING_RULES_PROMPT injection                                                  │ Chat, Messages     │   
  ├─────┼──────────────────────────────────────────────────────────────────────────────────────┼────────────────────┤
  │ H4  │ _classify_error_code (com taxonomia Anthropic/OpenAI mapeada)                        │ Chat, Messages     │
  ├─────┼──────────────────────────────────────────────────────────────────────────────────────┼────────────────────┤
  │ H5  │ Reasoning round-trip (<think> + thinking blocks)                                     │ Chat, Messages     │
  ├─────┼──────────────────────────────────────────────────────────────────────────────────────┼────────────────────┤   
  │ H6  │ tool_choice enforcement em /v1/messages (atualmente ignorado)                        │ Messages           │   
  ├─────┼──────────────────────────────────────────────────────────────────────────────────────┼────────────────────┤   
  │ H7  │ Schema: adicionar stop, stream_options, response_format, seed, tool_calls,           │ Chat               │   
  │     │ tool_call_id, name                                                                   │                    │
  ├─────┼──────────────────────────────────────────────────────────────────────────────────────┼────────────────────┤   
  │ H8  │ Tool schema normalization pra Chat + Responses (hoje só Anthropic normaliza)         │ Chat, Responses    │
  ├─────┼──────────────────────────────────────────────────────────────────────────────────────┼────────────────────┤   
  │ H9  │ Error shapes por protocolo: {error:{type, code, message}} em vez de FastAPI {detail} │ Chat, Messages,    │
  │     │                                                                                      │ Responses          │   
  ├─────┼──────────────────────────────────────────────────────────────────────────────────────┼────────────────────┤
  │     │ usage completo: cache_creation/read_input_tokens (Anthropic),                        │                    │   
  │ H10 │ prompt_tokens_details.cached_tokens + completion_tokens_details.reasoning_tokens     │ Chat, Messages     │
  │     │ (Chat)                                                                               │                    │   
  └─────┴──────────────────────────────────────────────────────────────────────────────────────┴────────────────────┘
                                                                                                                        
  Severidade MEDIUM (qualidade/UX)        
                                                                                                                        
  ┌─────┬────────────────────────────────────────────────────────────────────────────────┬──────────────────────────┐
  │  #  │                                      Item                                      │           API            │   
  ├─────┼────────────────────────────────────────────────────────────────────────────────┼──────────────────────────┤   
  │ M1  │ tool_use.id prefix: call_ → toolu_ (validação SDK Anthropic)                   │ Messages                 │
  ├─────┼────────────────────────────────────────────────────────────────────────────────┼──────────────────────────┤   
  │ M2  │ Streaming de input_json_delta chunked (hoje emite tudo de uma vez)             │ Messages                 │
  ├─────┼────────────────────────────────────────────────────────────────────────────────┼──────────────────────────┤
  │ M3  │ message_start.usage.input_tokens=0 — popular com prompt_tokens real            │ Messages                 │
  ├─────┼────────────────────────────────────────────────────────────────────────────────┼──────────────────────────┤   
  │ M4  │ is_error: true no tool_result ignorado                                         │ Messages, Responses      │
  ├─────┼────────────────────────────────────────────────────────────────────────────────┼──────────────────────────┤   
  │ M5  │ Chat streaming emite role e content juntos (OpenAI emite role sozinho          │ Chat                     │
  │     │ primeiro)                                                                      │                          │
  ├─────┼────────────────────────────────────────────────────────────────────────────────┼──────────────────────────┤   
  │ M6  │ Heartbeats excessivos (22 pings num "say hi") — gate por wall-clock            │ Chat, Messages,          │   
  │     │                                                                                │ Responses                │   
  ├─────┼────────────────────────────────────────────────────────────────────────────────┼──────────────────────────┤   
  │ M7  │ /v1/messages/count_tokens usa len(split()) em vez do tokenizer real            │ Messages                 │
  ├─────┼────────────────────────────────────────────────────────────────────────────────┼──────────────────────────┤   
  │ M8  │ Proprietary metrics/raw_text no top-level da response                          │ Chat, Messages           │
  ├─────┼────────────────────────────────────────────────────────────────────────────────┼──────────────────────────┤   
  │ M9  │ developer role → system (só Responses faz)                                     │ Chat, Messages           │
  ├─────┼────────────────────────────────────────────────────────────────────────────────┼──────────────────────────┤   
  │ M10 │ Image/document content blocks silenciosamente droppados                        │ Chat, Messages           │
  └─────┴────────────────────────────────────────────────────────────────────────────────┴──────────────────────────┘   
                                                                                         
  Severidade LOW (polimento)                                                                                            
                                                                  
  - L1: /v1/props anuncia chat_template: "qwen-tool-use" que não é aplicado                                             
  - L2: system_fingerprint ausente em chunks                                             
  - L3: service_tier echo missing                                                                                       
  - L4: Sampling defaults (presence_penalty 1.5) podem conflitar com recipe Hermes (menor prioridade; override-able por
  request)                              