# Native Saber: start, stop, and JayNet connection

This guide covers two established host-native configurations and the validated Occamy text preview:

| Configuration | Model ID | GPUs | Default context | Startup command |
|---|---|---|---|---|
| Qwen3.8-27B TP1 | `Qwen3.8` | GPU 0 | 65,536 | `tools/serve_27b_native.sh --tp 1` |
| Qwen3.8 Flash-Next TP2 | `Qwen3.8-Flash-Next` | GPUs 0 and 1 | 262,144 | `tools/serve_flashnext_native.sh` |
| Occamy TP2 | `occamy-1.0` / `occamy` | GPUs 0 and 1 | 8,192 | `saber occamy-exec --serve 18099` |

The 27B and Flash-Next servers:

- run directly on the host, without Docker or Podman;
- run in the foreground and stop cleanly on SIGINT (`Ctrl-C`);
- listen on `0.0.0.0:8080` by default;
- expose OpenAI-compatible HTTP endpoints without authentication;
- use the GPUs, so only one model server should run at a time.

Do not expose the port to an untrusted network. For local JayNet integration, connect through loopback.

## JayNet connection settings

Use these values after the selected server reports ready:

| Setting | 27B TP1 | Flash-Next TP2 |
|---|---|---|
| OpenAI base URL | `http://127.0.0.1:8080/v1` | `http://127.0.0.1:8080/v1` |
| Model | `Qwen3.8` | `Qwen3.8-Flash-Next` |
| API key | Not required | Not required |
| Chat endpoint | `POST /v1/chat/completions` | `POST /v1/chat/completions` |
| Text completion endpoint | `POST /v1/completions` | `POST /v1/completions` |
| Readiness endpoint | `GET /v1/models` | `GET /v1/models` |
| Health/metrics | `GET /health`, `GET /metrics` | `GET /health`, `GET /metrics` |

The request's `model` value must match the served model ID. Streaming is supported with `"stream": true`.

## Before every start

```bash
cd /srv/saber-jaynet-dev
/opt/rocm/bin/rocm-smi --showpids
```

Confirm that every required GPU is free. Do not stop JayNet or another GPU workload automatically; free the cards first. Also check that no old server owns the port:

```bash
ss -ltnp | grep ':8080'
```

The launchers check for a listening port and for the legacy `qwen_prod_nvfp4` user service, but they do not otherwise reserve GPUs.

## Qwen3.8-27B TP1

### Start

The verified single-GPU command is:

```bash
cd /srv/saber-jaynet-dev
SABER_PACKS=/srv/models/saber-models/saber-packs DEVICE=0 \
  tools/serve_27b_native.sh --tp 1 --no-build --enable-prefix-caching on
```

Use GPU 1 instead with `DEVICE=1`. `DEVICE` selects one physical HIP device; do not set `HIP_DEVICES` for TP1.

Omit `--no-build` when the release binary needs to be rebuilt. Building is the default and does not require a GPU:

```bash
SABER_PACKS=/srv/models/saber-models/saber-packs DEVICE=0 \
  tools/serve_27b_native.sh --tp 1 --enable-prefix-caching on
```

### Ready check

In another terminal:

```bash
curl -fsS http://127.0.0.1:8080/v1/models
```

Wait for a successful JSON response containing `Qwen3.8` before sending JayNet traffic.

### Optional 27B TP2 mode

The launcher supports TP2, but this mode has not been verified in this checkout. It needs the separate TP2 packs from `tools/fetch_packs.sh 27b` and both GPUs:

```bash
SABER_PACKS=/srv/models/saber-models/saber-packs HIP_DEVICES=0,1 \
  tools/serve_27b_native.sh --tp 2 --no-build --enable-prefix-caching on
```

TP2 defaults to 262,144 context tokens and 8 concurrent sequences. The TP1 and TP2 packs are not interchangeable.

## Qwen3.8 Flash-Next TP2

### Start

The verified two-GPU command is:

```bash
cd /srv/saber-jaynet-dev
SABER_NVME=/srv/models/saber-models HIP_DEVICES=0,1 \
  tools/serve_flashnext_native.sh --no-build
```

`SABER_NVME=/srv/models/saber-models` is required for the current asset layout. It makes the launcher use:

- packs under `/srv/models/saber-models/saber-packs/`;
- KV and prompt-cache data under `/srv/models/saber-models/saber-kv/`;
- output under `.cache/serve-flashnext/run/` in the repository.

The checkpoint is at `/srv/models/saber-models/models/Qwen3.8-Flash-Next-Quark-MXFP4`. Its absolute path is recorded in the pack's `manifest.json` and `ple/ple_index.json`. An online refetch can restore the hosted `/mnt/nvme/...` paths; do not refetch as part of routine startup.

Omit `--no-build` to rebuild before starting. Flash-Next is TP2-only in this launcher.

### Ready check

```bash
curl -fsS http://127.0.0.1:8080/v1/models
```

Wait for a successful JSON response containing `Qwen3.8-Flash-Next`.

## Stop

### Foreground terminal

Press `Ctrl-C` once and wait for the process to exit. This sends SIGINT to the server and its TP worker. Do not close the terminal or kill only one TP rank as the normal stop method.

Confirm that the listener and GPU processes are gone:

```bash
ss -ltnp | grep ':8080' || true
/opt/rocm/bin/rocm-smi --showpids
```

Stopping the server does not require deleting `saber-kv`; its files are persistent cache data, not a lock.

### Background or service-manager operation

The scripts do not daemonize and have no `stop` argument. JayNet or a service manager should:

1. Start the selected foreground command as a child in a new process group/session.
2. Save the leader PID/PGID and keep stdout/stderr in a log.
3. Poll `GET /v1/models` until it succeeds; process existence alone does not mean model readiness.
4. On stop, send SIGINT to the whole process group and wait for exit.
5. Escalate to SIGTERM only after a graceful-shutdown timeout, then check `rocm-smi --showpids`.
6. Do not use a broad `pkill -f saber`, which could stop an unrelated server.

Example for a manually managed background server:

```bash
cd /srv/saber-jaynet-dev
setsid env SABER_NVME=/srv/models/saber-models HIP_DEVICES=0,1 \
  tools/serve_flashnext_native.sh --no-build \
  >/srv/saber-fork/flashnext.log 2>&1 &
echo $! >/srv/saber-fork/flashnext.pid
```

Inspect the saved process before stopping it:

```bash
pid=$(cat /srv/saber-fork/flashnext.pid)
ps -o pid,pgid,cmd -p "$pid"
kill -INT -- "-$pid"
```

Wait for that PID to exit before removing the PID file or starting another model. The same process-group pattern works for 27B; use a separate log and PID file for each configuration.

## Launcher parameters

Both launchers process their own options first and pass all remaining arguments to the underlying `saber` command. Their `--help` output is the short reference:

```bash
tools/serve_27b_native.sh --help
tools/serve_flashnext_native.sh --help
```

### Common parameters

| Parameter | Default | Meaning |
|---|---|---|
| `--no-build` | Build enabled | Reuse `target/release/saber`; recommended for normal restarts |
| `PORT` | `8080` | HTTP listen port |
| `ROCM_PATH` | `/opt/rocm` | ROCm installation |
| `SABER_BIN` | `$SABER_REPO/target/release/saber` | Alternate Saber executable |
| `SABER_REPO` | Repository containing `tools/` | Source/build tree |
| `PREBUILT` | `$SABER_REPO/prebuilt/gfx1201` | Shipped GPU code objects and manifests |
| `HIP_LIB` | `$ROCM_PATH/lib/libamdhip64.so.7` | HIP runtime library |
| `HSA_LIB` | `$ROCM_PATH/lib/libhsa-runtime64.so.1` | HSA runtime library; checked at startup |
| `GPU_MAX_HW_QUEUES` | `3` | ROCm hardware queues per process |
| `--help`, `-h` | — | Print launcher usage without starting |

### 27B parameters

| Parameter | Default | Meaning |
|---|---|---|
| `--tp 1` / `--tp 2` | `TP`, otherwise `2` | Tensor-parallel degree and corresponding pack layout |
| `DEVICE` | `0` | TP1 only: one physical HIP device index |
| `HIP_DEVICES` | `0,1` | TP2 only: exactly two physical HIP device indices |
| `SABER_PACKS` | `/mnt/nvme/saber-packs` | Pack directory; set to `/srv/models/saber-models/saber-packs` here |
| `MAX_CTX` | `65536` TP1; `262144` TP2 | Maximum tokens per sequence |
| `MAX_SEQS` | `4` TP1; `8` TP2 | Concurrent sequences |
| `BATCHES` | `1,2,4` TP1; `1,2,3,4,8` TP2 | Captured decode batch sizes |
| `WIDTHS` | `1,8` | Speculative verify widths |
| `DRAFT_BLOCK` | `8` | DFlash2 draft block size |
| `BUDGET_MIB` | `30500` | Per-rank device-buffer budget in MiB |
| `MODEL` | `Qwen3.8` | Served model ID |
| `TIER`, `DRAFT` | Selected by TP mode | Advanced pack-directory overrides |
| `TOKENIZER` | `.cache/template_corpus.tokenizer.json` | Tokenizer file |
| `CHAT_TEMPLATE` | `.cache/template_corpus.jinja` | Chat template |

Useful pass-through serving options:

| Option | Meaning |
|---|---|
| `--enable-prefix-caching on` | Reuse cached prefixes across compatible requests; off by default |
| `--prefix-cache-host-mib N` | Host memory for prefix snapshots per rank; default `6144` MiB |
| `--warm-prompts FILE` | Warm common system prompts at startup and enable prefix caching |
| `--vision off\|lazy\|on` | Text-only, on-demand vision, or resident vision; default `off` |
| `--tool-call-parser qwen3_coder` | Tool-call parser; this is the default |
| `--reasoning-parser qwen3` | Reasoning parser; this is the default |
| `--no-enable-auto-tool-choice` | Disable automatic tool-call parsing |
| `--verbose` | Debug logging |

All underlying options, including diagnostics and experimental controls, are listed by:

```bash
target/release/saber serve --help
```

### Flash-Next parameters

| Parameter | Default | Meaning |
|---|---|---|
| `--record-routes` | Off | Record served expert routes under `$SABER_NVME/saber-kv/routes/<date>/` |
| `SABER_RECORD_ROUTES` | `0` | Environment equivalent of `--record-routes` |
| `SABER_NVME` | `/mnt/nvme` | Parent of `saber-packs/` and `saber-kv/`; use `/srv/models/saber-models` here |
| `SABER_SHARED` | `$SABER_REPO/.cache` | Shared scratch directory |
| `PACK` | Auto-selected trunk8 pack under `$SABER_NVME/saber-packs/` | Weight-pack directory |
| `FN_TRUNK` | `w8-hc8-head` | Dense-trunk weight form |
| `FN_KV` | `fp8` | QSA KV dtype; `bf16` is the alternative |
| `FN_KV_DIR` | `$SABER_NVME/saber-kv` | KV slab and prompt-cache directory |
| `FN_ELASTIC_KV_MAX` | `0.5` | Extra elastic KV VRAM per rank in GiB |
| `FN_EXPERTS` | `tp2` | Expert layout; `ep` is the alternative |
| `FN_AGENT_DEFAULTS` | `1` | Enable agent-load tuning; set `0` to omit its flags |
| `FN_OUT` | `$SABER_SHARED/serve-flashnext/run` | Engine result/output directory |
| `HIP_DEVICES` | `0,1` | Exactly two physical HIP device indices |

With `FN_AGENT_DEFAULTS=1`, the launcher adds these unless explicitly overridden:

```text
--fn-kv-lazy 1024
--fn-prefix-tree-ram-gib 4.5
--fn-prefill-yield 1
--fn-ring-idle
--fn-kv-admit-slab-tokens 338592
--spec-couple on
```

Useful pass-through serving options:

| Option | Default | Meaning |
|---|---|---|
| `--fn-max-ctx N` | `262144` in the verified agent-default boot | Maximum tokens per sequence |
| `--fn-served-name NAME` | `Qwen3.8-Flash-Next` | Served model ID |
| `--fn-thinking on\|off` | `on` | Default thinking behavior; requests can override it |
| `--fn-prefix-tree off` | On in the agent-default boot | Disable the tiered prefix tree |
| `--fn-vision off\|lazy\|on` | Lazy where supported | Vision mode; text-only is `off` |
| `--fn-image-cache on\|off` | `on` | Content-hash image prompts in the prefix cache |
| `--fn-kv-ram-gib N` | Derived | Explicit pinned-RAM KV slab per rank; replaces lazy/admission-slab defaults |
| `--fn-kv-pcache-gib N` | `16` with prefix tree | Drive prompt-cache size per rank in GiB |
| `--warm-prompts FILE` | None | Warm common system prompts at startup |
| `--verbose` | Off | Debug logging |

All underlying options, including diagnostics and experimental controls, are listed by:

```bash
target/release/saber engine --help
```

## Occamy FP8 TP2 (validated text serving)

Occamy uses a separate command rather than either launcher above. The recommended configuration uses official block-128 FP8 routed experts, BF16 dense/shared weights and KV, and FP32 GDN state. All weights are VRAM-resident on both GPUs:

```bash
cd /srv/saber-jaynet-dev
/opt/rocm/bin/rocm-smi --showpids  # confirm both cards are free first
LD_LIBRARY_PATH=/opt/rocm/lib ./target/release/saber occamy-exec \
  --fp8-experts /srv/saber-fork/packs/occamy-fp8-all-layers \
  --serve 18099 --max-ctx 262144 --max-rows 64 --max-requests 1 \
  --hip-lib /opt/rocm/lib/libamdhip64.so.7
```

Rank 0 spawns rank 1 automatically. Startup takes about 25-30 seconds with the packs in page cache. Wait for `occamy listening`, then check:

```bash
curl -fsS http://127.0.0.1:18099/v1/models
```

The model IDs are `occamy-1.0` and the alias `occamy`. Stop with `Ctrl-C` in the foreground terminal and confirm that both ranks exit. The process-group procedure above is also appropriate for a service manager; do not kill only one rank.

For validated speculative decoding, add the official BF16 MTP head. Depth 1 is recommended because it was fastest in the short-context benchmark:

```bash
LD_LIBRARY_PATH=/opt/rocm/lib ./target/release/saber occamy-exec \
  --fp8-experts /srv/saber-fork/packs/occamy-fp8-all-layers \
  --mtp /srv/models/radconv/mtp/mtp-trained.safetensors --spec-depth 1 \
  --serve 18099 --max-ctx 262144 --max-rows 64 --max-requests 1 \
  --hip-lib /opt/rocm/lib/libamdhip64.so.7
```

MTP preserves ordinary target-model behavior for greedy and sampled requests, including temperature/top-p/top-k/min-p, presence/frequency/repetition penalties, logprobs, deterministic seeds, and streaming. Plain greedy requests retain the argmax fast path. Other requests use dense target logits and the ordinary sampler at every verification position; rank 0 makes the sampling decision and synchronizes rank 1. Sampled verification is target-first and uses position-keyed random draws, not probability-ratio acceptance, so seeded output does not change with the speculative schedule.

Depths 1, 2, and 3 matched all 32 ordinary FP8 token IDs in the 2048-token plain-greedy harness; depth 1 decoded at 69.36 tok/s versus 60.06 without MTP. The expanded serving matrix matched ordinary decoding exactly, including floating-point logprobs, for seven 64-token cases and five 256-token cases at both depths 1 and 2. However, the new dense paths are currently a correctness feature rather than a speedup: at depth 1, 256-token MTP/control ratios were 0.95x for greedy penalties, 0.91x for greedy logprobs, 0.89x for temperature 1, and 0.86x for the combined sampled case. Depth 2 was slower. Omit `--mtp` for sampled, logprob, or penalty-heavy serving when throughput is the priority. Full-context MTP allocation passed at 21.39 GiB on rank 0, but MTP throughput at a real 256K prompt has not been measured.

The server has an exact-token prefix cache with a default 4096 MiB budget per rank. It snapshots KV and GDN convolution/recurrent state at `prompt_len - 1`, restores the longest compatible prefix, and always prefills at least the final prompt token to regenerate logits. Identical repeated prompts and shared conversation prefixes can therefore hit without caching sampled answers. MTP-ready entries also snapshot the draft head's KV and are used for MTP requests only when compatible. Entries are device-resident and LRU-evicted. Set `--prefix-cache-mib 0` to disable caching or choose another per-rank budget. Rank 0 logs `occamy prefix cache: hit ...` and `miss ...`.

GPU validation passed at 8192 context. Repeated 31-token greedy MTP prompts hit 30/31 tokens; three 5303-token prompts hit 5302/5303 tokens and produced exactly the same 24-token responses cold and warm. Mean request latency fell from 3.467 seconds cold to 0.502 seconds warm, a 6.90x speedup including generation. Cached greedy MTP, seeded sampled, streaming, shared-prefix follow-up, and shorter-generation responses all matched a separate `--prefix-cache-mib 0` server exactly. An overlapping two-slot ordinary-decoding test restored 21/22 tokens into the other slot, and both 96-token responses matched the uncached control. LRU eviction kept the test cache within its 1 GiB budget. These are short-context measurements, not real 256K cache-throughput results.

The additive FP8 pack was built from `Accio-Lab/occamy-1.0-FP8` (E4M3 values and FP32 128x128 block scales). Rebuild it without changing the BF16 packs:

```bash
./target/release/saber occamy-fp8-pack --all-layers \
  --checkpoint /srv/models/radconv/occamy-1.0-FP8 \
  --out /srv/saber-fork/packs/occamy-fp8-all-layers
```

The output directory must not already exist. A two-layer pack for focused probes can be built by omitting `--all-layers`.

For the original BF16 executor, omit `--fp8-experts`. BF16 uses pinned host memory for vocabulary tables and shared experts and supports only 8192 context with one slot or 512 context with two slots; it does not fit 256K. The two-request BF16 command is:

```bash
LD_LIBRARY_PATH=/opt/rocm/lib ./target/release/saber occamy-exec \
  --serve 18099 --max-ctx 512 --max-rows 64 --max-requests 2 \
  --hip-lib /opt/rocm/lib/libamdhip64.so.7
```

Capabilities and limits:

- Text-only FP8/BF16 TP2. Vision is not implemented.
- Optional MTP uses canonical per-token attention/GDN decode verification, batched projections/MoE, per-step recurrent snapshots, and longest-prefix acceptance. Greedy, target-first sampled, penalty, logprob, non-streaming, and streaming responses matched ordinary decoding exactly. MTP requires one request slot; its dense verifier is currently slower as detailed above.
- Greedy and non-greedy sampling support temperature, `top_p`, `top_k`, `min_p`, presence/frequency/repetition penalties, and deterministic `seed`.
- Completion logprobs are supported. For chat, request `"logprobs": true` and `"top_logprobs": N`; prompt logprobs are refused.
- Non-streaming and SSE streaming responses work. One choice per request (`n: 1`) is required. Qwen3 reasoning and XML tool-call parsers are enabled, including automatic tool choice. OpenAI tool-call argument strings are decoded to objects before template rendering, so follow-up requests with tool history work. Both the config's `<|endoftext|>` and the tokenizer's `<|im_end|>` are stop tokens and are not emitted as answer text.
- `--max-requests` allocates independent KV/GDN state slots. Requests advance round-robin by prefill chunk or decode token; GPU execution is serialized, not batch-fused. Additional requests queue when all slots are occupied. Startup IPC waits are bounded, but serving disables the read deadline so five idle minutes do not kill rank 1.
- BF16 concurrency checks passed: two seeded requests at 512 context matched isolated outputs, three simultaneous submissions completed with two slots, and slot reuse preserved determinism. FP8 uses one request slot at 256K in the validated configuration.

Useful parameters:

| Parameter | Default | Meaning |
|---|---|---|
| `--serve PORT` | Off | Start the OpenAI server; without it, run the token-ID harness |
| `--host HOST` | `127.0.0.1` | HTTP bind address; keep loopback for JayNet |
| `--served-name NAME` | `occamy-1.0` | Primary model ID; `occamy` is always an alias |
| `--fp8-experts PATH` | None | Additive official FP8 routed-expert pack; also makes all weights VRAM-resident |
| `--max-ctx N` | `8192` | Per-request prompt + generation limit; use `262144` for full context |
| `--max-rows N` | `64` | Prefill rows per chunk, 1..64; not a prompt-length limit |
| `--max-requests N` | `1` | Independent request-state slots, 1..4, subject to VRAM |
| `--prefix-cache-mib N` | `4096` | Per-rank device prefix-cache budget; `0` disables caching |
| `--layers PATH` | `/srv/saber-fork/packs/occamy-bf16-all-layers` | BF16 dense/shared weights; routed records are replaced by `--fp8-experts` |
| `--globals PATH` | `/srv/saber-fork/packs/occamy-bf16-globals` | Verified embedding/head packs |
| `--checkpoint PATH` | `/srv/models/radconv/occamy-1.0` | Config, tokenizer, and chat template |
| `--device N` | `0` | Rank 0 GPU; spawned rank 1 uses N + 1 |
| `--trace-layers` | Off | Synchronize and log each layer, attention stage, and kernel dispatch |
| `--tokens IDS` | Empty | Token-ID harness input when not serving; long inputs are chunked |
| `--token-end N` | None | Harness-only generated IDs 1..N, avoiding a huge command-line argument |
| `--decode N` | `1` | Tokens generated by the token-ID harness |
| `--quality-reference PATH` | None | Compare token harness residuals and generated IDs with a saved CPU reference |
| `--mtp PATH` | Disabled | Official separate BF16 drafter; enables greedy speculative decoding |
| `--spec-depth N` | `2` | Draft tokens per round, 1..3; use `1` for the best measured throughput |

Example request:

```bash
curl -fsS http://127.0.0.1:18099/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"occamy-1.0","messages":[{"role":"user","content":"Say hello."}],"temperature":0,"max_tokens":32,"stream":false}'
```

The independent full-model CPU reference uses the original safetensors checkpoint directly and needs no Python packages:

```bash
./target/release/saber occamy-quality \
  --tokens 1,2,3 --decode 4 \
  --out /srv/saber-fork/occamy-quality-run

LD_LIBRARY_PATH=/opt/rocm/lib ./target/release/saber occamy-exec \
  --fp8-experts /srv/saber-fork/packs/occamy-fp8-all-layers \
  --tokens 1,2,3 --decode 4 --max-ctx 128 --max-rows 2 \
  --quality-reference /srv/saber-fork/occamy-quality-run \
  --hip-lib /opt/rocm/lib/libamdhip64.so.7
```

The output directory must not already exist. The 3-token/four-decode and 33-token/four-decode references matched TP2 exactly, including small-chunk prefill and persistent decode state. Real chat output is coherent after the GDN SiLU correction.

Measured with FP8: both four-token references matched exactly; a 2048-token prompt ran at 1541.05 prompt tokens/s and 60.03 decode tokens/s over 31 steps. The 262,144-token allocation uses about 20 GiB per rank, far below the 31.86 GiB available per card. A real 262,140-token prompt plus four generated tokens completed successfully in 686.163 seconds: 685.230 seconds of prefill (382.56 prompt tokens/s) and 3.21 decode tokens/s at the full boundary. The 60 tok/s figure is for short context, not 256K. BF16 measured 234.32 prompt tokens/s and 2.20 decode tokens/s, so FP8 is the recommended configuration. These are single-request benchmarks, not sustained multi-user results. JayNet integration is still not deployed and requires separate approval.

## Minimal API smoke test

Set `MODEL` to the ID returned by `/v1/models`, then run:

```bash
MODEL=Qwen3.8-Flash-Next  # use Qwen3.8 for the 27B server
curl -fsS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "'"$MODEL"'",
    "messages": [{"role": "user", "content": "Reply with OK."}],
    "max_tokens": 16,
    "stream": false
  }'
```

For JayNet, use the same endpoint and model ID with its normal OpenAI client, tool definitions, streaming setting, and coding prompts.
