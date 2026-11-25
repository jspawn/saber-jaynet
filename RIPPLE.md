# Ripple-65B Preview on Saber (experimental)

This checkout contains a host-native TP2 prototype for `ricursiveAI/Ripple-65B-Preview-NVFP4` and its official `Ripple-65B-Preview-DFlash2` drafter. It targets two AMD Radeon AI PRO R9700 GPUs (`gfx1201`).

## Current status and limits

- The 80-layer target executes on both GPUs with exact TP2 all-reduces.
- Target attention and embeddings/LM head are BF16. NVFP4 MLP weights were converted to Saber's MXFP4 W4A8 format.
- Target and drafter KV caches use dynamic per-token/per-head FP8 scales.
- DFlash2 executes five BF16-derived FP8 layers, four-tap convolutions, four local layers plus global NoPE, target hidden-state taps, and a top-32/rank-256 selector.
- Greedy and target-first sampled speculative verification preserve the target token stream, including seeded output and logprobs. The current selector computes exact top-32 candidates with the full BF16 target head; the int2 screening/rerank optimization is not implemented.
- The configured and tested server maximum is 131,072 tokens with DFlash2. A 130,029-token prompt generated `LONG_CONTEXT_OK` in 2,242.6 seconds (2,241.5 seconds prefill) and a 249-token request and 32-token DFlash2/target-only parity run passed at this allocation. A full 131,072-token prompt and 256K are not validated; 256K exceeds the drafter checkpoint's position limit and the current VRAM budget.
- `ripple-serve` is a persistent OpenAI-compatible, single-request server. Concurrent clients queue; continuous batching is not implemented. JayNet's development checkout has a `SABER_MODEL=ripple` launcher and reference preset.
- Sampling supports temperature, top-p, top-k, min-p, presence/frequency/repetition penalties, seeds, and sampled-token logprobs. Prompt logprobs and `n > 1` are refused explicitly.
- Qwen3 reasoning and tool parsing support `tools`, `tool_choice:"auto"`, JSON-in-`<tool_call>` output, OpenAI assistant/tool history, streamed tool-call deltas, and `finish_reason:"tool_calls"`. Cached-prefix counts appear in `usage.prompt_tokens_details.cached_tokens` for non-streaming responses and streaming usage chunks.
- A single-entry prefix cache tracks up to the configured context length for plain greedy requests without penalties or logprobs. Local-attention KV uses a 4,096-token preservation ring while attention remains 1,024 tokens. Reuse is allowed only when the branch's required local window remains resident; expired branches and all sampled requests use full prefill, and an unrelated prompt replaces the cached conversation.

The NVFP4-to-MXFP4 conversion is not lossless. A real-weight sample had about 11% weight-relative RMSE, so official NVFP4 output quality is not claimed. The FP8 drafter is requantized from its official BF16 checkpoint.

## Assets

Default paths on this host:

```text
Target checkpoint: /srv/models/saber-models/models/Ripple-65B-Preview-NVFP4
Drafter checkpoint: /srv/models/saber-models/models/Ripple-65B-Preview-DFlash2
Target pack: /srv/models/saber-models/saber-packs/ripple-65b-preview-mxfp4-tp2
Drafter pack: /srv/models/saber-models/saber-packs/ripple-65b-dflash2-fp8-runtime-tp2
```

The target pack has 1,443 records and about 25.915 GiB per rank. The filtered runtime drafter pack has 90 records and about 2.264 GiB per rank. The drafter shares the target's embedding and LM head at runtime.

The source safetensors were deleted during the 2026-11-24 disk cleanup. The checkpoint directories now retain only config, tokenizer, template and other small metadata files; the runnable packs remain. Reconversion requires downloading the original target and drafter checkpoints first.

## Rebuild on the host

No container or GPU is needed for conversion after restoring the source checkpoints. Python NumPy is required; this host uses `python`.

```bash
cd /srv/saber-jaynet-dev

SABER_PACKS=/srv/models/saber-models/saber-packs \
  tools/pack/build_ripple.sh --tp 2 --workers 8

SABER_PACKS=/srv/models/saber-models/saber-packs \
  tools/pack/build_ripple_draft.sh --tp 2 --workers 8
```

The scripts refuse to overwrite an existing manifest. Use `--out DIR` for a new pack, `--plan-only` to inspect the record plan, or `--bf16` on the drafter builder for the larger BF16 numerical control.

Build the GPU code objects and executables:

```bash
cargo run -q --release -p saber-kernels --bin saber-kbuild -- \
  --host ripple_spike ripple_draft
cargo build --release -p saber-cli --bin ripple-exec --bin ripple-spike --bin ripple-serve
```

## Run and stop

Before every GPU launch, confirm that both cards are free. Do not stop JayNet or another workload automatically.

```bash
cd /srv/saber-jaynet-dev
/opt/rocm/bin/rocm-smi --showpids

tools/ripple_exec_native.sh --no-build \
  --text 'Write a Python function unique_in_order(items) that preserves first occurrence order. Include one doctest.' \
  --chat --decode 96
```

Target-only control:

```bash
tools/ripple_exec_native.sh --no-build --target-only \
  --text 'Write a Python function unique_in_order(items) that preserves first occurrence order. Include one doctest.' \
  --chat --decode 96
```

Useful options and overrides:

| Option/environment | Default | Meaning |
|---|---|---|
| `--no-build` | Build enabled | Reuse `target/release/ripple-exec` |
| `--target-only` | DFlash2 enabled | Disable drafting |
| `--text TEXT` | Token IDs `1,2,3` | Prompt text |
| `--chat` | Raw text | Render a simple user/assistant chat transcript |
| `--decode N` | `8` | Maximum generated tokens; EOS stops earlier |
| `--tokens IDS` | `1,2,3` | Comma-separated token-ID input |
| `MAX_CTX` | `131072` | Per-request context allocation |
| `MAX_ROWS` | `64` | Prefill rows per chunk, up to 64 |
| `--trace-layers` | Off | Synchronize and log target/draft stages |
| `HIP_DEVICES` | `0,1` | The two physical GPU indices |
| `RIPPLE_DRAFT` | Filtered FP8 pack | Alternate drafter pack; BF16 control is supported |

The harness runs in the foreground. Press Ctrl-C in its terminal/process group and wait for both ranks to exit, then verify:

```bash
/opt/rocm/bin/rocm-smi --showpids
```

For harness service management, start it as a child in a new process group, save the PGID, and signal the whole group. Do not kill only one TP rank or use a broad `pkill`.

## Persistent OpenAI server

Start the foreground server after checking that both GPUs are free:

```bash
cd /srv/saber-jaynet-dev
/opt/rocm/bin/rocm-smi --showpids
PORT=8000 MAX_CTX=131072 tools/ripple_serve_native.sh --no-build
```

The launcher accepts `--target-only` like the one-shot launcher. `HOST`, `PORT`, `SERVED_NAME`, `MAX_CTX`, `MAX_ROWS`, `RIPPLE_PACK`, `RIPPLE_DRAFT`, `RIPPLE_CHECKPOINT`, `RIPPLE_SERVE_BIN`, `PREBUILT`, `HIP_DEVICES`, and `ROCM_PATH` are environment overrides. The bind defaults to `127.0.0.1`; `HOST` and `--host` can override it explicitly. The served model name defaults to `Ripple-65B-Preview`, with aliases `ripple` and `ripple-65b`.

Available routes include `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`, `/metrics`, `/v1/stats`, and `/v1/requests`. Streaming, stop strings, stop token IDs, `return_token_ids`, and concurrent-client queueing use Saber's existing OpenAI plane. The default temperature is zero. Temperature, top-p, top-k, min-p, presence/frequency/repetition penalties, seeds, and sampled-token logprobs are supported. Prompt logprobs and `n != 1` return HTTP 400.

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Reply with exactly OK."}],"temperature":0,"max_tokens":32}'
```

The prefix cache holds one conversation up to the configured context length and applies only to plain greedy requests without penalties or logprobs. Sampled requests always use full prefill to preserve their original numerical path. Local-attention KV remains a 4,096-token ring with a 1,024-token attention window: continuations can reuse beyond 4,096 tokens, while a branch is recomputed if its required local window has been overwritten. `cached_tokens` in `/v1/requests` and the server log reports the reused prompt tokens; at least the last prompt token is recomputed. At 128K, the target KV allocation is about 2.320 GiB per rank. The BF16 `value_seed` scratch covers only the active chunk (0.625 MiB at `MAX_ROWS=64`), rather than the whole context. The measured 128K server with DFlash2 uses 31.089 GiB and leaves 0.770 GiB free per rank.

For a background test service, save the rank-0 PID and signal only that PID for graceful coordinated shutdown:

```bash
setsid tools/ripple_serve_native.sh --no-build > ripple-serve.log 2>&1 &
RIPPLE_PID=$!
# ... use the server ...
kill -TERM "$RIPPLE_PID"
wait "$RIPPLE_PID"
/opt/rocm/bin/rocm-smi --showpids
```

Do not use a broad `pkill` or stop JayNet to free the cards.

## JayNet development integration

`/srv/orch-dev` contains `SABER_MODEL=ripple` dispatch, the native fallback `scripts/saber-ripple-native.sh`, and `presets/specialist-ripple-65b-saber-tp2.conf`. The preset uses both TP2 packs, GPUs `0,1`, loopback port `8080`, and 131,072 context. Register the preset with `caps.vision=false`, `caps.thinking=true`, `profile.think_strip=qwen3`, `profile.system_first=false`, `profile.tool_selection=all`, and deterministic sampling (`temperature=0.0`). Ripple's template accepts non-leading system messages; keeping JayNet's volatile datetime and guard notes trailing and its tool schemas stable preserves the cacheable prefix. Explicit caller tool allowlists still take precedence. It executes one request at a time; concurrent clients queue.

Dry-run or boot the reference preset from the development checkout without changing the live JayNet installation:

```bash
cd /srv/orch-dev
export JAYNET_DATA=/srv/data JAYNET_HOME=/srv/orch-dev ORCH_MODELS=/srv/models
export SABER_REPO=/srv/saber-jaynet-dev
export LLAMA_BIN=/srv/orch-dev/scripts/saber-serve.sh
scripts/start-model.sh --preset presets/specialist-ripple-65b-saber-tp2.conf --dry-run
scripts/start-model.sh --preset presets/specialist-ripple-65b-saber-tp2.conf
```

The lifecycle wrapper runs in the foreground, creates a dedicated native process group, and translates JayNet's SIGTERM into coordinated SIGINT shutdown. Installing the changed wrapper or registering/swapping a live preset is a separate deployment step; development acceptance does not edit `/srv/jaynet-orchestrator` or restart its services.

## Verification and measured results

Layer-0 real-weight GPU/host attention parity:

```text
relative RMSE 0.004716 (threshold 0.02): PASS
```

The independent BF16 drafter reference matched every draft layer with cosine similarity at least 0.9988 in the diagnostic case. The final coding A/B used the same 28-token prompt and 96 generated tokens:

```text
Target only: 16.43 tok/s
DFlash2: 13 rounds, 82 accepted draft tokens, exact 96/96 token-ID parity
DFlash2 measured runs: 74.99 to 96.58 tok/s (about 4.6x to 5.9x)
```

Persistent server validation:

```text
1024-token allocation: exact 32/32 token-ID parity with ripple-exec
65536-token allocation: exact greedy token-ID parity and exact sampled output/logprobs
131072-token allocation: 249-token multi-chunk request and exact 32/32 DFlash2/target-only token-ID parity
Four seeded sampling cases with penalties/logprobs: exact DFlash2/target-only choices and usage
Sampled streaming: matching token/logprob counts, EOS stop, [DONE]
2024-token prompt: cold 4.506 s prefill; cached repeat 0.101 s prefill (2023 cached tokens)
5016-token conversation: 5032 cached tokens; exact 24/24 continuation token-ID parity with full prefill
Long continuation wall time: 14.449 s effectively full prefill (6 generic prefix tokens) versus 0.917 s cached
Cold/cached greedy output: exact token-ID parity beyond the 1024-token attention window
4400-token prompt: active ring correct; cache expired rather than reusing overwritten entries
Generated-token conversation-prefix reuse and unrelated-prompt invalidation: PASS
Streaming, concurrent-client queueing, and /v1/completions: PASS
```

The long-prompt request's total wall time fell from 5.50 s to 1.09 s; its decode time was unchanged at about 0.99 s. The sampled suite took 0.32–0.60 s per request with DFlash2 versus 0.81–1.48 s target-only. These are short, single-request measurements, not sustained concurrency benchmarks. Target-only decoding at a real 23,839-token prompt measured 4.10 tok/s; DFlash2 throughput at that context has not been measured. Long-context attention and graph capture remain performance work.

Host checks:

```bash
cargo test -p saber-model
cargo test -p saber-cli --bin ripple-serve
python -m unittest discover -s tools/pack/tests
cargo clippy -p saber-cli --bin ripple-exec --bin ripple-spike --bin ripple-serve --no-deps -- -D warnings
```

The 20 pack-builder tests pass. `saber-model` tests pass; workspace-wide Clippy still has an unrelated upstream `template/mod.rs` warning.
