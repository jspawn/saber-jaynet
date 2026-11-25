# Occamy conversion status

## Current status

Occamy has passed real-weight, full-layer BF16 reference parity for GDN layer 0 and full-attention layer 3 on both TP2 rank layouts. The strict config reader, packs, dense projections, BF16 KV append, GQA8 attention, FP32-state GDN path, and BF16 MoE path are implemented.

The text-only TP2 executor and OpenAI preview are servable. Independent full-model references match generated IDs through chunked prefill and decode, real chat output is coherent, sampling and independent request-state slots are validated, and an 8000-token prompt completed at 8192 context. The recommended configuration is now official block-128 FP8 routed experts with BF16 dense/shared weights and KV; every weight is VRAM-resident and the full 262,144-token context allocates successfully. A loopback-safe JayNet preset and native launcher are prepared in `/srv/orch-dev`; no live GPU delegation has been tested.

This is an architecture port, not a conversion of existing GGUF or `.rad` files.

- Integrated checkout: `/srv/saber-jaynet-dev`
- Integration branch: `jaynet-integration`
- Original upstream base: `c8a1e84b39e73363495e549d480a6f6c3893800c`; integrated onto `1da27661b3bf8a702a7532c8d17651de8c635e44`
- BF16 checkpoint: `/srv/models/radconv/occamy-1.0` (approximately 66 GB)
- Official FP8 checkpoint: `/srv/models/radconv/occamy-1.0-FP8` (E4M3, FP32 128x128 scales)
- Scope: **text only, TP2, FP8 experts + BF16 dense/shared/KV, MTP disabled, vision off**

The branch also contains earlier uncommitted native/JayNet launcher work. Do not reset or clean the working tree, and do not commit, push, deploy, or restart JayNet without explicit approval.

## Model facts

Occamy uses `Qwen3_5MoeForConditionalGeneration` / `qwen3_5_moe`:

- 35B total parameters and approximately 3B active parameters
- 40 text layers: 30 gated delta net (GDN) layers and 10 full-attention layers
- Hidden size 2048
- Full attention: 16 query heads, 2 KV heads, head dimension 256 (GQA8)
- GDN: 16 key heads and 32 value heads, head dimension 128, convolution width 4
- MoE: 256 routed experts, top 8, plus one shared expert
- Routed and shared expert intermediate width 512
- Untied output head and shared input embeddings
- Vision tensors and a separately trained MTP head are present but outside this phase

## Completed

### Strict config and tensor mapping

`crates/saber-model/src/qwen3_5_moe/config.rs` is an independent strict parser; Flash-Next's parser was not loosened. It accepts the released BF16 config, rejects quantization and MTP, validates TP2 geometry and the layer schedule, and records the different norm conventions.

`crates/saber-model/src/qwen3_5_moe/spike.rs` maps every source tensor and probes the real kernel registry. The real safetensors headers confirmed:

- GDN layer 0: 784 tensors with matching shapes
- Full-attention layer 3: 781 tensors with matching shapes
- GDN QKV `[8192, 2048]`, Z `[4096, 2048]`, convolution `[8192, 1, 4]`
- Gated attention Q `[8192, 2048]`, K/V `[512, 2048]`, Q/K norms `[256]`

The two-layer pack plan covers all 1,565 unique source tensors exactly once.

### Real layer 0/3 BF16 pack

Added `pack_plan.rs` and `saber occamy-pack`. The layout uses dense TP2 slices, 128 routed experts per rank, and replicated norms/router/shared-expert weights. GDN projections use the existing merged layout; `A_log` and `dt_bias` are widened from BF16 to F32. Routed and shared expert gate/up weights are merged gate-first without quantization.

Built and fully SHA-256-verified:

```text
/srv/saber-fork/packs/occamy-bf16-l0-l3/
├── rank0/dense.pack   522 records, 1,671,712,768 bytes
├── rank1/dense.pack   522 records, 1,671,712,768 bytes
└── shared/dense.pack   15 records,    14,721,024 bytes
```

Commands:

```bash
./target/release/saber occamy-pack \
  --out /srv/saber-fork/packs/occamy-bf16-l0-l3 --dry-run

./target/release/saber occamy-pack \
  --out /srv/saber-fork/packs/occamy-bf16-l0-l3
```

The builder refuses to overwrite an existing output directory.

### GQA8 attention

Added `occamy_attn.hip`, its manifest/code object, and `occamy_gfx1201.toml`:

- BF16 prefill at GQA8 with 48 query rows per workgroup
- BF16 decode at GQA8 with one through eight query tokens
- Existing split-KV combine reused
- Existing QK norm/RoPE geometry matches Occamy's head width, rotary width, interleaved M-RoPE, gated Q layout, and `1 + weight` norm convention

BF16 KV append reuses Flash-Next's paged K|V append with its indexer ring disabled. A three-row, two-block probe is byte-exact. Dense BF16 projections reuse Flash-Next's split-K and LDS GEMMs; all TP2 widths (6144, 4096, 2048, 256, 32, and 1 at K 2048) and all four row bands pass against the Rust reference.

### BF16 MoE path

Added Occamy routing and combine kernels for 256 experts, top 8, and 128 local experts per rank. The existing Flash-Next BF16 grouped GEMMs and BF16 SiLU kernel accept Occamy's runtime dimensions and are reused without quantization:

- Gate/up: N 1024, K 2048
- SiLU output width 512
- Down: N 2048, K 512

### Host validation

Passed:

```bash
cargo test -p saber-kernels
cargo test -p saber-model
cargo test -p saber-cli occamy
cargo build --release -p saber-cli
```

This includes 157 `saber-kernels` unit tests plus its integration suites, 171 `saber-model` tests, the Occamy pack tests, and the release build. Native `saber-kbuild --native` compiles the new code objects without a container.

### Focused GPU parity

Generated deterministic references without Python dependencies:

```bash
./target/release/saber occamy-reference --out /srv/saber-fork/occamy-parity
```

Ran the real kernels on the idle GPUs:

```bash
LD_LIBRARY_PATH=/opt/rocm/lib ./target/release/saber occamy-probe \
  --data /srv/saber-fork/occamy-parity \
  --prebuilt prebuilt/gfx1201 --mode all --iters 20 \
  --hip-lib /opt/rocm/lib/libamdhip64.so.7
```

Results:

- GQA8 prefill: zero values outside the 0.03 BF16 tolerance; maximum absolute error 0.005371.
- GQA8 decode plus combine: zero values outside the 0.05 tolerance; maximum absolute error 0.037842. The prefill cross-check has the same maximum, consistent with f16 split-KV partial rounding rather than a decode-only defect.
- MoE rank 0 and rank 1: exact BF16 outputs, exact top-8 IDs, exact route metadata, and zero top-8 weight errors.
- Small-case host-wall timings were approximately 17 us for prefill, 18 us for decode plus combine, and 75-91 us per MoE block. These are smoke-test measurements, not serving throughput estimates.

### Real-pack and full-layer parity

`occamy-pack-probe` reads and hash-checks the real pack records, executes them on the GPU, and compares them with independent Rust references:

```bash
LD_LIBRARY_PATH=/opt/rocm/lib ./target/release/saber occamy-pack-probe \
  --pack /srv/saber-fork/packs/occamy-bf16-l0-l3 \
  --prebuilt prebuilt/gfx1201 --mode layer0 \
  --hip-lib /opt/rocm/lib/libamdhip64.so.7

# The same command with --mode layer3, dense, or moe.
```

Verified in release mode:

- Every real dense record tested on both ranks passes; the largest absolute error is 0.000244.
- Real layer-0 MoE records execute exactly on both EP rank layouts, including the shared expert. Layer-3 MoE maximum absolute error is 0.000244.
- Layer 3 checks input/post norms, QKV, Q/K norm+RoPE, KV cache, attention core/gate/output, residuals, both MoE halves, and final output. KV copies are exact; final maximum absolute error is 0.003906 (RMS 0.000433).
- Layer 0 checks the merged projection, convolution, Q/K/V, gates, KKT solve, chunk scan, FP32 state, SiLU gated norm, output projection, residuals, both MoE halves, and final output. Maximum FP32-state error is 0.000651; final maximum absolute error is 0.003906 (RMS 0.000209).
- The GDN reference is a direct delta-rule recurrence, independent of the GPU's chunked WY scan. Occamy uses the existing `r2` KKT solver for its two value heads per key head, not Flash-Next's `r3` solver.

These focused tests use three-token, zero-initial-state prefill and run both rank layouts sequentially on one GPU with host-side reduction sums. Full TP2 prefill, decode state, and serving are validated separately below.

## Main files added

- `crates/saber-model/src/qwen3_5_moe/{config,pack_plan,spike}.rs`
- `crates/saber-model/tests/qwen3_5_moe_{config,pack,spike}.rs`
- `crates/saber-kernels/hip/occamy_{attn,moe}.hip`
- `crates/saber-kernels/rules/occamy_gfx1201.toml`
- `crates/saber-kernels/src/occamy.rs`
- `crates/saber-kernels/tests/occamy.rs`
- `crates/saber-cli/src/occamy/{pack,pack_probe,probe,reference,quality,exec,serve}.rs`
- `prebuilt/gfx1201/occamy_{attn,moe}.{hsaco,manifest.toml}`

## Completed full packs

Built and fully SHA-256-verified:

- `/srv/saber-fork/packs/occamy-bf16-all-layers`: all 31,330 decoder-layer source tensors; 10,460 records and 31.196 GiB per rank, plus 290 shared records (294,363,136-byte pack).
- `/srv/saber-fork/packs/occamy-bf16-globals`: vocabulary-sharded embedding and untied output head, 2 records and 0.947 GiB per rank, plus the replicated final norm.

Reproduction commands (the builder refuses to overwrite existing outputs):

```bash
./target/release/saber occamy-pack --all-layers \
  --out /srv/saber-fork/packs/occamy-bf16-all-layers --dry-run
./target/release/saber occamy-pack --all-layers \
  --out /srv/saber-fork/packs/occamy-bf16-all-layers
./target/release/saber occamy-pack --globals \
  --out /srv/saber-fork/packs/occamy-bf16-globals
```

## Executor, numerical validation, and serving

`crates/saber-cli/src/occamy/exec.rs` implements the resident TP2 loader and token-ID-driven 40-layer graph: vocabulary-parallel embedding, exact two-rank reductions, GDN and full-attention mixers, EP2 MoE, final norm, sharded output head, persistent FP32 GDN state, and BF16 KV state. Rank 0 spawns rank 1 as a separate process. The harness chunks prompts of any length by `--max-rows` and reports prefill and decode timing separately.

The GPUs report 34,208,743,424 bytes each. Vocabulary tables and rank-0 shared experts use pinned host memory through their existing device-pointer ABIs; rank 1 does not load unused shared-expert records. Device allocation is 31.529 GiB per rank for one slot at 8192 context and 31.549 GiB for two slots at 512 context. Two slots at 8192 context fail allocation with `hipMalloc: OutOfMemory`; that combination is not supported on these cards.

Two numerical defects were found and fixed:

- KV append initially used 8,192 elements per token instead of `2 * head_dim = 512`.
- The executor reused Flash-Next's sigmoid-only GDN norm. Occamy's published Transformers implementation requires SiLU, `z * sigmoid(z)`. The executor and layer probe now use the existing `fn_gdn_norm_bf16_t256` SiLU kernel, and the probe reference was corrected too. Before this fix, layer-0 final RMS error was 0.090 and output was incoherent; after it, the corresponding 33-token layer-0 RMS error is 0.000282.

`crates/saber-cli/src/occamy/quality.rs` is an independent, dependency-free CPU implementation over the original safetensors checkpoint. It implements direct recurrent GDN updates, attention, routing, shared experts, residuals, and the output head. It writes per-layer last-token residuals and stage traces. `occamy-exec --quality-reference` compares those traces and requires exact generated-ID agreement.

Passed on TP2:

- Three prompt tokens, two-token prefill chunks, and four decode steps exactly matched `[4, 5, 6, 24218]`.
- A 33-token prompt at 8192 context with eight-token chunks and four decode steps exactly matched `[198, 34, 1113, 11660]`. Final layer RMS difference was 0.029050; embedding was exact.
- The corrected focused layer-0 GPU probe passes on both rank layouts with final maximum absolute error 0.003906.
- Real chat requests now produce coherent English reasoning text.

Reference commands:

```bash
./target/release/saber occamy-quality \
  --tokens 1,2,3 --decode 4 --out /srv/saber-fork/occamy-quality-run
LD_LIBRARY_PATH=/opt/rocm/lib ./target/release/saber occamy-exec \
  --tokens 1,2,3 --decode 4 --max-ctx 128 --max-rows 2 \
  --quality-reference /srv/saber-fork/occamy-quality-run \
  --hip-lib /opt/rocm/lib/libamdhip64.so.7
```

`crates/saber-cli/src/occamy/serve.rs` uses the existing OpenAI router and dense Saber sampler over complete two-shard logits. It supports greedy and non-greedy sampling, temperature/top-p/top-k/min-p, presence/frequency/repetition penalties, deterministic seeds, completion logprobs, non-streaming responses, and SSE. Qwen3 reasoning and XML tool-call parsers are enabled, including automatic tool choice. `ToolArguments::ObjectsStrict` decodes OpenAI's JSON argument strings before the Qwen template iterates them, allowing requests with assistant/tool history. The engine and OpenAI layer stop on both `config.json`'s `<|endoftext|>` (248044) and the tokenizer's `<|im_end|>` (248046); neither appears in answer text. Chat logprobs use `logprobs: true` with `top_logprobs`. Prompt logprobs and `n != 1` are refused.

The server has independent per-request KV/GDN state slots selected by `--max-requests`. Active requests advance round-robin by prefill chunk or decode token; additional requests queue. Kernels remain serialized rather than batch-fused. Startup IPC waits are bounded, but serving disables the read deadline so an idle follower does not exit after five minutes. Two simultaneous seeded requests matched their isolated outputs exactly, three simultaneous submissions completed with two slots, slot reuse preserved determinism, streaming completed with `[DONE]`, and graceful Ctrl-C shutdown passed.

Performance and context checks used one slot at 8192 context:

- 2048 prompt tokens produced identical `[1, 2, 3, 4]` with 64-row and 16-row chunks. Prefill was 236.11 tokens/s at 64 rows and 85.83 tokens/s at 16 rows; decode was 2.24 tokens/s.
- 8000 prompt tokens completed in 34.141 seconds of prefill (234.32 tokens/s), followed by 2.20 decode tokens/s and the same IDs.
- These BF16 measurements are historical baselines; the FP8 section below is the current configuration and includes the 262,144-token allocation.

Final host checks pass: `cargo test -p saber-cli occamy` (4 tests), `cargo test -p saber-hip` (48 unit tests plus 1 doc test), `cargo build --release -p saber-cli`, and `git diff --check`. Test servers were stopped after validation.

## FP8 routed experts and full-context serving

`crates/saber-cli/src/occamy/fp8_pack.rs` builds an additive TP2 pack from the official FP8 release. It preserves the E4M3 values and FP32 block scales, merges gate/up records, and preshuffles weights for the gfx1201 FP8 WMMA fragments. The BF16 packs are unchanged and remain the dense/shared source and correctness fallback.

```bash
./target/release/saber occamy-fp8-pack --all-layers \
  --checkpoint /srv/models/radconv/occamy-1.0-FP8 \
  --out /srv/saber-fork/packs/occamy-fp8-all-layers
```

The full pack has 20,480 records and 15.004 GiB per rank. `occamy-exec --fp8-experts PATH` skips BF16 routed records, loads the FP8 replacements, and makes every weight VRAM-resident, including the vocabulary tables and shared experts that remain pinned host memory in BF16 mode.

Execution uses Saber's mixed W8A8/BF16 grouped MoE kernels: dynamic per-128 E4M3 activation quantization, FP8xFP8 WMMA with FP32 accumulation for routed experts, and BF16 shared experts. Attention, GDN projections, routers, embeddings, output head, and KV stay BF16; recurrent GDN state stays FP32.

Validation completed:

- Both rank layouts pass the focused FP8 MoE probe. The activation check changed 6,045 of 6,144 values, confirming that W8A8 rather than a BF16 fallback executed.
- Three-token and 33-token independent references each match all four generated IDs exactly: `[4, 5, 6, 24218]` and `[198, 34, 1113, 11660]`.
- A 2048-token prompt achieved 1541.05 prompt tokens/s and 60.03 decode tokens/s over 31 steps, compared with roughly 236/2.2 for BF16.
- The 262,144-token allocation passes with about 20 GiB per rank and preserves the exact four-token reference. A real 262,140-token prompt plus four generated tokens completed in 686.163 seconds: 685.230 seconds of prefill (382.56 prompt tokens/s) and 3.21 decode tokens/s. Short-context decode speed does not apply at the 256K attention boundary.
- Real OpenAI chat responses are coherent. Test servers were stopped after validation.

A whole-decode HIP graph experiment was rejected: ROCm returned `StreamCaptureUnsupported` at `hipStreamEndCapture`. The validated eager path is used instead.

MTP is implemented and validated with the official 19-tensor BF16 drafter at `/srv/models/radconv/mtp/mtp-trained.safetensors` (SHA-256 `56baea0ef88de178c6c858d81adee9e532983798dce81202f21a7483e59001df`). The implementation follows the official runtime hooks: batched projections and MoE, ordinary single-token attention and packed GDN decode for each verification node, no speculative prefill-state reinitialization, per-step recurrent/convolution snapshots, and commit of the longest accepted prefix. Rejected KV entries remain beyond the committed sequence length and are overwritten later. The acceptance comparison was corrected to compare target row zero with draft zero; canonical GDN calls also receive one-row `cu` metadata.

Depths 1, 2, and 3 each matched all 32 ordinary FP8 IDs for the 2048-token test. Decode was 69.36, 63.05, and 61.30 tok/s respectively, versus 60.06 without MTP, so `--spec-depth 1` is recommended. The three-token/four-decode reference `[4, 5, 6, 24218]` also matched through two-row prefill chunks. Full-context MTP allocation passed at 21.39 GiB on rank 0; no real 256K MTP throughput claim is made.

OpenAI serving requires `--max-requests 1` with MTP. Plain greedy requests retain the original device-argmax verifier. Requests with logprobs, presence/frequency/repetition penalties, or non-greedy sampling instead gather dense target logits for every verification row and run the ordinary host sampler with the exact output history and output position. Rank 0 alone samples and sends each target decision to rank 1, keeping TP state identical. Target-first sampled verification uses the existing position-keyed noise source: a matching draft is accepted, while a mismatch emits the ordinary sampled target and discards later drafts. It does not use probability-ratio/residual acceptance, so the seeded token sequence and raw-logit logprobs are schedule-invariant rather than merely distribution-equivalent. Multi-token rounds leave their last token pending so the scheduler emits it exactly once.

GPU validation covered plain greedy, greedy logprobs, all three penalties, temperature 1, top-p/top-k/min-p, deterministic seeds, and their combination. Seven 64-token cases matched cold versus warm and a separate no-MTP server exactly; SSE text, finish reason, and all 64 logprob entries also matched. Five 256-token cases then matched no-MTP output and floating-point logprobs exactly at both depths 1 and 2. Two unseeded temperature-1 requests completed independently with different answers.

The dense verifier is correct but not currently a serving speedup. Depth-1 256-token MTP/control time ratios were 0.95x for greedy penalties, 0.91x for greedy logprobs, 0.89x for temperature 1, and 0.86x for the combined sampled/logprob/penalty case; depth 2 was worse at 0.93x, 0.85x, 0.85x, and 0.77x respectively. The original plain-greedy harness result above remains the useful MTP speed case.

## Prefix caching

The serving executor now owns an exact-token, device-resident prefix cache on each TP rank. At `prompt_len - 1`, it snapshots every GDN convolution/recurrent buffer and the useful KV blocks for all full-attention layers. MTP-compatible entries also include the separate draft head's KV. Cache lookup uses the longest exact token prefix; restore translates buffer names across request slots and leaves at least one prompt token for ordinary prefill and sampling. Entries own independent GPU allocations, use LRU eviction under `--prefix-cache-mib` (default 4096 MiB per rank), and never reuse generated answers. Rank 0 logs hits, misses, reused tokens, and memory use.

Cold prefill deliberately separates the final prompt token so identical repeated prompts can hit. The first request populates the cache; a later matching request restores the state and prefills only its final token. Sampled and MTP snapshots are distinguished so a target-only snapshot cannot accidentally supply an uninitialized drafter state.

Host validation passes: nine Occamy tests, `cargo clippy -p saber-cli --no-deps -- -D warnings`, release build, and `git diff --check`.

GPU validation passed with FP8 at 8192 context:

- An identical 31-token greedy MTP request hit 30/31 tokens and matched its cold response exactly.
- Three distinct 5303-token chat prompts each hit 5302/5303 tokens on their second submission. All cold/warm `choices` and `usage` objects matched exactly, including 24 generated tokens. Mean latency was 3.467 seconds cold and 0.502 seconds warm: 6.90x faster overall, or 2.965 seconds saved on prefill-dominated work. This is end-to-end request latency, not a pure decode benchmark.
- Greedy MTP after sampled traffic, seeded non-greedy repetition, SSE reassembly, a raw-completion follow-up sharing 360 prefix tokens, and a shorter generation all matched a separate `--prefix-cache-mib 0` control exactly. Target-only and MTP-ready snapshots could not be confused.
- With two ordinary-decoding slots, an overlapping request restored 21/22 tokens captured in the other slot. Both concurrent 96-token responses matched the uncached control exactly.
- Long-prompt inserts triggered LRU eviction while reported cache memory remained within the 1 GiB test budget. Disabling the cache produced no cache log lines.
- All test servers were stopped and no KFD processes remained. Real 256K prefix-cache throughput is still unmeasured.

## Remaining work

- Prompt logprobs, multiple choices, batch-fused execution, and further performance optimization remain unfinished.
- Dense sampled/logprob/penalty verification is exact but currently slower and needs optimization. MTP concurrency, vision, and a real 256K MTP benchmark are not implemented or validated.
- JayNet integration is registered but not deployed: pull `/srv/orch-dev` into live and restart before assigning the preset. `STARTSTOP.md` documents verified startup, shutdown, quality, parameters, and VRAM combinations.

Before every GPU test or server launch, inspect `/opt/rocm/bin/rocm-smi --showpids`. Do not stop JayNet or other GPU users to obtain cards.
