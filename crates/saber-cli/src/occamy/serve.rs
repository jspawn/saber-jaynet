use std::collections::BTreeSet;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{ensure, Context};
use saber_exec::{LaunchCtx, Launcher, RankPeer};
use saber_hip::{HipThread, Stream};
use saber_serve::detok::ModelCodec;
use saber_serve::engine::{
    Engine, EngineError, EngineRequest, EngineStats, FinishReason, ModelInfo, RequestId,
    SampledLogprobs, StepEvent, Submission,
};
use saber_serve::sampler::{
    EntropyNoise, NoiseSource, PenaltyContext, RowParams, Sampler, SeededNoise,
};
use saber_serve::sampling::DefaultSampling;
use saber_serve::state::{
    ReasoningParserKind, ServerState, TemplateRenderer, ToolArguments, ToolParserKind,
};

use super::exec::{Collective, ExecArgs, Model};

#[derive(Debug)]
struct Job {
    id: RequestId,
    request: EngineRequest,
    events: tokio::sync::mpsc::Sender<StepEvent>,
}

#[derive(Debug)]
struct OccamyEngine {
    tx: std::sync::mpsc::Sender<Job>,
    model: ModelInfo,
    next_id: AtomicU64,
    running: Arc<AtomicU64>,
    aborted: Arc<Mutex<BTreeSet<RequestId>>>,
}

impl OccamyEngine {
    fn new(
        tx: std::sync::mpsc::Sender<Job>,
        model: ModelInfo,
        running: Arc<AtomicU64>,
        aborted: Arc<Mutex<BTreeSet<RequestId>>>,
    ) -> Self {
        Self {
            tx,
            model,
            next_id: AtomicU64::new(1),
            running,
            aborted,
        }
    }
}

impl Engine for OccamyEngine {
    fn submit(&self, request: EngineRequest) -> Result<Submission, EngineError> {
        if request.prompt_token_ids.is_empty() {
            return Err(EngineError::Other("empty prompts are not supported".into()));
        }
        if request.prompt_token_ids.len() > self.model.max_model_len {
            return Err(EngineError::PromptTooLong {
                got: request.prompt_token_ids.len(),
                max: self.model.max_model_len,
            });
        }
        if request.params.n != 1 {
            return Err(EngineError::Unsupported {
                param: "n",
                message: "Occamy currently serves one choice per request (n=1)".into(),
            });
        }
        if request.params.prompt_logprobs.is_some() {
            return Err(EngineError::Unsupported {
                param: "prompt_logprobs",
                message: "Occamy does not yet expose prompt logprobs".into(),
            });
        }
        let id = RequestId(self.next_id.fetch_add(1, Ordering::Relaxed));
        let (events, receiver) = tokio::sync::mpsc::channel(16);
        self.tx
            .send(Job {
                id,
                request: request.clone(),
                events,
            })
            .map_err(|_| EngineError::ShuttingDown)?;
        self.running.fetch_add(1, Ordering::AcqRel);
        let stream = futures_util::stream::unfold(receiver, |mut receiver| async move {
            receiver.recv().await.map(|event| (event, receiver))
        });
        Ok(Submission {
            id,
            events: Box::pin(stream),
        })
    }

    fn abort(&self, id: RequestId) {
        self.aborted
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(id);
    }

    fn stats(&self) -> EngineStats {
        EngineStats {
            num_requests_running: self.running.load(Ordering::Acquire),
            ..EngineStats::default()
        }
    }

    fn model(&self) -> &ModelInfo {
        &self.model
    }
}

struct Leader<'a, 'h> {
    hip: &'h HipThread,
    model: &'a mut Model,
    collective: &'a Collective,
    ctx: &'a LaunchCtx<'h>,
    launcher: &'a mut Launcher,
    stream: &'h Stream,
    peer: &'a RankPeer,
    step: u32,
    max_ctx: usize,
    eos_token_ids: Vec<u32>,
    running: Arc<AtomicU64>,
    aborted: Arc<Mutex<BTreeSet<RequestId>>>,
}

struct RequestState {
    id: RequestId,
    slot: usize,
    prompt: Vec<u32>,
    params: saber_serve::sampling::SamplingParams,
    events: tokio::sync::mpsc::Sender<StepEvent>,
    prefilled: usize,
    generated: Vec<u32>,
    next: Option<u32>,
    logprobs: Option<SampledLogprobs>,
    max_tokens: usize,
}

impl RequestState {
    fn new(job: Job, slot: usize, max_ctx: usize) -> Self {
        let max_tokens = job
            .request
            .params
            .max_tokens
            .unwrap_or(max_ctx)
            .min(max_ctx.saturating_sub(job.request.prompt_token_ids.len()));
        Self {
            id: job.id,
            slot,
            prompt: job.request.prompt_token_ids,
            params: job.request.params,
            events: job.events,
            prefilled: 0,
            generated: Vec::new(),
            next: None,
            logprobs: None,
            max_tokens,
        }
    }
}

impl Leader<'_, '_> {
    fn rpc(
        &mut self,
        slot: usize,
        flags: u8,
        tokens: &[u32],
        position: usize,
    ) -> anyhow::Result<()> {
        let mut frame = Vec::with_capacity(18 + tokens.len() * 4);
        frame.push(flags);
        frame.push(slot as u8);
        frame.extend_from_slice(&(position as u64).to_le_bytes());
        frame.extend_from_slice(&self.step.to_le_bytes());
        frame.extend_from_slice(&(tokens.len() as u32).to_le_bytes());
        for token in tokens {
            frame.extend_from_slice(&token.to_le_bytes());
        }
        self.peer
            .exchange()
            .xchg_bytes("occamy.serve.command", &frame)?;
        Ok(())
    }

    fn command(
        &mut self,
        slot: usize,
        flags: u8,
        tokens: &[u32],
        position: usize,
    ) -> anyhow::Result<Option<Vec<f32>>> {
        let decode = flags & 1 != 0;
        let with_head = flags & 2 != 0;
        let reset = flags & 4 != 0;
        self.rpc(slot, flags, tokens, position)?;
        if reset {
            self.model.reset_slot(self.hip, self.stream, slot)?;
            self.step += 1;
            return Ok(None);
        }
        self.model.forward_slot(
            self.hip,
            self.stream,
            self.ctx,
            self.launcher,
            self.collective,
            slot,
            tokens,
            position,
            decode,
            with_head,
        )?;
        let logits = if with_head {
            Some(
                self.model
                    .logits(self.hip, self.stream, self.peer, self.step)?,
            )
        } else {
            None
        };
        self.step += 1;
        Ok(logits)
    }

    fn mtp_prefill(
        &mut self,
        slot: usize,
        shifted_tokens: &[u32],
        position: usize,
        make_drafts: bool,
    ) -> anyhow::Result<()> {
        let flags = 16 | if make_drafts { 32 } else { 0 };
        self.rpc(slot, flags, shifted_tokens, position)?;
        self.model.mtp_prefill_chunk(
            self.hip,
            self.stream,
            self.ctx,
            self.launcher,
            self.collective,
            self.peer,
            slot,
            shifted_tokens,
            position,
            make_drafts,
            self.step,
        )?;
        self.step += 1;
        Ok(())
    }

    fn speculative(
        &mut self,
        state: &RequestState,
    ) -> anyhow::Result<Vec<super::exec::SpeculativeToken>> {
        let next = state.next.context("decode has no input token")?;
        let position = state.prompt.len() + state.generated.len() - 1;
        let vocab_size = self.model.config.vocab_size as usize;
        let row_params = RowParams::resolve(&state.params, vocab_size);
        let sample_target =
            !row_params.is_greedy() || !row_params.no_penalties() || row_params.logprobs.is_some();
        self.rpc(
            state.slot,
            8 | if sample_target { 32 } else { 0 },
            &[next],
            position,
        )?;
        let output_position = state.generated.len();
        let mut decide = |row: usize, drafts: &[u32], logits: Vec<f32>| {
            let history = if row_params.no_penalties() {
                None
            } else {
                let mut history = Vec::with_capacity(output_position + row);
                history.extend_from_slice(&state.generated);
                history.extend_from_slice(&drafts[..row]);
                Some(history)
            };
            let output = history.as_deref().unwrap_or(&state.generated);
            let (token, logprobs) = Self::sample_at(
                vocab_size,
                logits,
                &state.params,
                &state.prompt,
                output,
                (output_position + row) as u64,
            )?;
            Ok(super::exec::SpeculativeToken { token, logprobs })
        };
        let committed = self.model.speculative_round(
            self.hip,
            self.stream,
            self.ctx,
            self.launcher,
            self.collective,
            self.peer,
            state.slot,
            next,
            position,
            self.step * 256,
            sample_target,
            &mut decide,
        )?;
        self.step += 1;
        Ok(committed)
    }

    fn restore_prefix(
        &mut self,
        slot: usize,
        prompt: &[u32],
        use_mtp: bool,
    ) -> anyhow::Result<usize> {
        self.rpc(slot, 128 | if use_mtp { 32 } else { 0 }, prompt, 0)?;
        let prefix = self
            .model
            .restore_prefix(self.hip, self.stream, slot, prompt, use_mtp)?;
        self.step += 1;
        Ok(prefix)
    }

    fn capture_prefix(
        &mut self,
        slot: usize,
        prompt: &[u32],
        mtp_ready: bool,
    ) -> anyhow::Result<()> {
        self.rpc(slot, 64 | if mtp_ready { 32 } else { 0 }, prompt, 0)?;
        self.model
            .capture_prefix(self.hip, self.stream, slot, prompt, mtp_ready)?;
        self.step += 1;
        Ok(())
    }

    fn use_mtp(&self, _params: &saber_serve::sampling::SamplingParams) -> bool {
        self.model.mtp
    }

    fn sample_at(
        vocab_size: usize,
        mut logits: Vec<f32>,
        params: &saber_serve::sampling::SamplingParams,
        prompt: &[u32],
        output: &[u32],
        output_position: u64,
    ) -> anyhow::Result<(u32, Option<SampledLogprobs>)> {
        let row = RowParams::resolve(params, vocab_size);
        let context = PenaltyContext { prompt, output };
        let mut noise: Box<dyn NoiseSource> = match params.seed {
            Some(seed) => Box::new(SeededNoise::new(seed, output_position, 0)),
            None => Box::new(EntropyNoise::new()),
        };
        let sampled = Sampler::new()
            .sample_dense(&mut logits, &row, &context, noise.as_mut())
            .map_err(|error| anyhow::anyhow!(error))?;
        Ok((sampled.token_id, sampled.logprobs))
    }

    fn sample(
        &self,
        logits: Vec<f32>,
        params: &saber_serve::sampling::SamplingParams,
        prompt: &[u32],
        output: &[u32],
    ) -> anyhow::Result<(u32, Option<SampledLogprobs>)> {
        Self::sample_at(
            self.model.config.vocab_size as usize,
            logits,
            params,
            prompt,
            output,
            output.len() as u64,
        )
    }

    fn aborted(&self, id: RequestId) -> bool {
        self.aborted
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .contains(&id)
    }

    fn emitted_tokens(next: Option<u32>, stop: bool) -> Vec<u32> {
        if stop {
            Vec::new()
        } else {
            next.into_iter().collect()
        }
    }

    fn emit(&self, state: &mut RequestState, aborted: bool) -> anyhow::Result<bool> {
        let stop = state.next.is_some_and(|token| {
            (!state.params.ignore_eos && self.eos_token_ids.contains(&token))
                || state.params.stop_token_ids.contains(&token)
        });
        let new_tokens = Self::emitted_tokens(state.next, stop);
        let finished = if aborted {
            Some(FinishReason::Abort)
        } else if stop {
            Some(FinishReason::Stop)
        } else if state.generated.len() >= state.max_tokens {
            Some(FinishReason::Length)
        } else {
            None
        };
        let send = state.events.blocking_send(StepEvent {
            request_id: state.id,
            choice_index: 0,
            step_index: state.generated.len().saturating_sub(1) as u32,
            new_tokens,
            finished,
            logprobs: state.logprobs.clone().map(|logprobs| vec![logprobs]),
            prompt_logprobs: None,
            error: None,
            cached_tokens: None,
        });
        Ok(send.is_err() || finished.is_some())
    }

    fn advance(&mut self, state: &mut RequestState) -> anyhow::Result<bool> {
        let aborted = self.aborted(state.id);
        if !aborted && state.prefilled < state.prompt.len() {
            let start = state.prefilled;
            let mut end = (start + self.model.max_rows).min(state.prompt.len());
            if end == state.prompt.len() && end - start > 1 {
                end -= 1;
            }
            let last = end == state.prompt.len();
            let flags = if last && state.max_tokens > 0 { 2 } else { 0 };
            let logits = self.command(state.slot, flags, &state.prompt[start..end], start)?;
            state.prefilled = end;
            if let Some(logits) = logits {
                let (token, logprobs) =
                    self.sample(logits, &state.params, &state.prompt, &state.generated)?;
                state.next = Some(token);
                state.logprobs = logprobs;
                state.generated.push(token);
            }
            if self.use_mtp(&state.params) && (!last || state.next.is_some()) {
                let mut shifted = state.prompt[start + 1..end].to_vec();
                shifted.push(if last {
                    state.next.unwrap()
                } else {
                    state.prompt[end]
                });
                self.mtp_prefill(state.slot, &shifted, start, last)?;
            }
            if !last {
                if end + 1 == state.prompt.len() {
                    let mtp_ready = self.use_mtp(&state.params);
                    self.capture_prefix(state.slot, &state.prompt[..end], mtp_ready)?;
                }
                return Ok(false);
            }
        }
        if self.emit(state, aborted)? {
            return Ok(true);
        }
        let token = state.next.context("decode has no input token")?;
        let position = state.prompt.len() + state.generated.len() - 1;
        if self.use_mtp(&state.params) && position + 4 <= self.max_ctx {
            let committed = self.speculative(state)?;
            let pending = committed.len() - 1;
            for (index, decision) in committed.into_iter().enumerate() {
                state.next = Some(decision.token);
                state.logprobs = decision.logprobs;
                state.generated.push(decision.token);
                if index < pending && self.emit(state, aborted)? {
                    return Ok(true);
                }
            }
        } else {
            let logits = self
                .command(state.slot, 3, &[token], position)?
                .context("decode produced no logits")?;
            let (token, logprobs) =
                self.sample(logits, &state.params, &state.prompt, &state.generated)?;
            state.next = Some(token);
            state.logprobs = logprobs;
            state.generated.push(token);
        }
        Ok(false)
    }
}

fn follower(
    hip: &HipThread,
    model: &mut Model,
    collective: &Collective,
    ctx: &LaunchCtx<'_>,
    launcher: &mut Launcher,
    stream: &Stream,
    peer: &RankPeer,
) -> anyhow::Result<()> {
    loop {
        let frame = peer.exchange().xchg_bytes("occamy.serve.command", &[])?;
        if frame.is_empty() {
            return Ok(());
        }
        ensure!(
            frame.len() >= 18 && (frame.len() - 18) % 4 == 0,
            "invalid serve command"
        );
        let flags = frame[0];
        let slot = frame[1] as usize;
        let position = u64::from_le_bytes(frame[2..10].try_into().unwrap()) as usize;
        let step = u32::from_le_bytes(frame[10..14].try_into().unwrap());
        let count = u32::from_le_bytes(frame[14..18].try_into().unwrap()) as usize;
        ensure!(frame.len() == 18 + count * 4, "serve command token count");
        let tokens: Vec<u32> = frame[18..]
            .chunks_exact(4)
            .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        if flags & 4 != 0 {
            model.reset_slot(hip, stream, slot)?;
            continue;
        }
        if flags & 128 != 0 {
            model.restore_prefix(hip, stream, slot, &tokens, flags & 32 != 0)?;
            continue;
        }
        if flags & 64 != 0 {
            model.capture_prefix(hip, stream, slot, &tokens, flags & 32 != 0)?;
            continue;
        }
        if flags & 16 != 0 {
            model.mtp_prefill_chunk(
                hip,
                stream,
                ctx,
                launcher,
                collective,
                peer,
                slot,
                &tokens,
                position,
                flags & 32 != 0,
                step,
            )?;
            continue;
        }
        if flags & 8 != 0 {
            ensure!(
                tokens.len() == 1,
                "speculative command needs one root token"
            );
            let mut decide = |_: usize, _: &[u32], _: Vec<f32>| {
                unreachable!("rank 1 waits for rank 0's target decision")
            };
            model.speculative_round(
                hip,
                stream,
                ctx,
                launcher,
                collective,
                peer,
                slot,
                tokens[0],
                position,
                step * 256,
                flags & 32 != 0,
                &mut decide,
            )?;
            continue;
        }
        model.forward_slot(
            hip,
            stream,
            ctx,
            launcher,
            collective,
            slot,
            &tokens,
            position,
            flags & 1 != 0,
            flags & 2 != 0,
        )?;
        if flags & 2 != 0 {
            model.logits(hip, stream, peer, step)?;
        }
    }
}

fn spawn_http(
    host: String,
    port: u16,
    state: Arc<ServerState>,
) -> anyhow::Result<std::thread::JoinHandle<anyhow::Result<()>>> {
    std::thread::Builder::new()
        .name("occamy-http".into())
        .spawn(move || {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .context("building the tokio runtime")?;
            runtime.block_on(async move {
                let addr = format!("{host}:{port}");
                let listener = tokio::net::TcpListener::bind(&addr)
                    .await
                    .with_context(|| format!("binding {addr}"))?;
                println!(
                    "occamy listening http://{}/v1 (SIGINT to stop)",
                    listener.local_addr()?
                );
                saber_serve::server::serve_with_shutdown(state, listener, async {
                    let _ = tokio::signal::ctrl_c().await;
                })
                .await
                .context("serving")
            })?;
            Ok(())
        })
        .context("spawning the HTTP thread")
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run(
    hip: &HipThread,
    model: &mut Model,
    collective: &Collective,
    ctx: &LaunchCtx<'_>,
    launcher: &mut Launcher,
    stream: &Stream,
    peer: &RankPeer,
    args: &ExecArgs,
) -> anyhow::Result<()> {
    let port = args.serve.context("serve port")?;
    if args.rank != 0 {
        return follower(hip, model, collective, ctx, launcher, stream, peer);
    }
    let tokenizer = Arc::new(
        saber_model::Tokenizer::from_model_dir(&args.checkpoint)
            .map_err(|error| anyhow::anyhow!(error))?,
    );
    let renderer = Arc::new(
        TemplateRenderer::from_file(chat_template(&args.checkpoint))
            .map_err(|error| anyhow::anyhow!(error))?,
    );
    // config.json names <|endoftext|>; chat generation actually ends with the tokenizer's <|im_end|>.
    let im_end = tokenizer
        .token_to_id("<|im_end|>")
        .context("Occamy tokenizer is missing <|im_end|>")?;
    let eos_token_ids = vec![model.config.eos_token_id, im_end];
    let (tx, rx) = std::sync::mpsc::channel::<Job>();
    let running = Arc::new(AtomicU64::new(0));
    let aborted = Arc::new(Mutex::new(BTreeSet::new()));
    let info = ModelInfo {
        served_model_name: args.served_name.clone(),
        aliases: vec!["occamy".into()],
        max_model_len: args.max_ctx as usize,
        root: args.checkpoint.display().to_string(),
        eos_token_ids: eos_token_ids.clone(),
    };
    let engine = Arc::new(OccamyEngine::new(
        tx.clone(),
        info,
        Arc::clone(&running),
        Arc::clone(&aborted),
    ));
    let state = ServerState::builder(engine, Arc::new(ModelCodec::new(tokenizer)), renderer)
        .reasoning_parser(ReasoningParserKind::Qwen3)
        .tool_parser(ToolParserKind::Qwen3Coder)
        .enable_auto_tool_choice(true)
        // OpenAI history carries tool arguments as a JSON string; Occamy's template iterates an object.
        .tool_arguments(ToolArguments::ObjectsStrict)
        .default_sampling(DefaultSampling {
            temperature: 0.0,
            ..DefaultSampling::STOCK
        })
        .vocab_size(model.config.vocab_size as usize)
        .build();
    let http = spawn_http(args.host.clone(), port, state)?;
    drop(tx);
    let mut leader = Leader {
        hip,
        model,
        collective,
        ctx,
        launcher,
        stream,
        peer,
        step: 0,
        max_ctx: args.max_ctx as usize,
        eos_token_ids,
        running,
        aborted,
    };
    let mut active = std::collections::BTreeMap::<RequestId, RequestState>::new();
    let mut pending = std::collections::VecDeque::<Job>::new();
    let mut free_slots: Vec<usize> = (0..args.max_requests).rev().collect();
    let mut closed = false;
    loop {
        while !closed {
            match rx.try_recv() {
                Ok(job) => pending.push_back(job),
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => closed = true,
            }
        }
        while !free_slots.is_empty() && !pending.is_empty() {
            let job = pending.pop_front().unwrap();
            let slot = free_slots.pop().unwrap();
            let use_mtp = leader.use_mtp(&job.request.params);
            let prefilled = leader.restore_prefix(slot, &job.request.prompt_token_ids, use_mtp)?;
            if prefilled == 0 {
                leader.command(slot, 4, &[], 0)?;
            }
            let id = job.id;
            let mut state = RequestState::new(job, slot, leader.max_ctx);
            state.prefilled = prefilled;
            active.insert(id, state);
        }
        if active.is_empty() {
            if closed {
                break;
            }
            match rx.recv() {
                Ok(job) => pending.push_back(job),
                Err(_) => closed = true,
            }
            continue;
        }
        let ids: Vec<RequestId> = active.keys().copied().collect();
        for id in ids {
            let finished = {
                let state = active.get_mut(&id).unwrap();
                leader.advance(state)?
            };
            if finished {
                let state = active.remove(&id).unwrap();
                free_slots.push(state.slot);
                leader
                    .aborted
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .remove(&id);
                leader.running.fetch_sub(1, Ordering::AcqRel);
            }
        }
    }
    peer.exchange().xchg_bytes("occamy.serve.command", &[])?;
    http.join()
        .map_err(|_| anyhow::anyhow!("HTTP thread panicked"))??;
    Ok(())
}

fn chat_template(checkpoint: &Path) -> std::path::PathBuf {
    checkpoint.join("chat_template.jinja")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_tokens_are_not_emitted_as_answer_text() {
        assert_eq!(
            Leader::emitted_tokens(Some(248044), true),
            Vec::<u32>::new()
        );
        assert_eq!(Leader::emitted_tokens(Some(42), false), vec![42]);
    }

    #[test]
    fn target_first_sampling_reuses_ordinary_positions_histories_and_logprobs() {
        let params = saber_serve::sampling::SamplingParams {
            temperature: 0.73,
            top_p: 0.95,
            top_k: 4,
            min_p: 0.01,
            seed: Some(91),
            logprobs: Some(3),
            presence_penalty: 0.2,
            frequency_penalty: 0.3,
            repetition_penalty: 1.1,
            ..saber_serve::sampling::SamplingParams::default()
        };
        let prompt = [1, 2, 1];
        let mut ordinary = vec![4];
        let mut rows = Vec::new();
        for row in 0..5 {
            let logits = (0..8)
                .map(|token| ((token * 7 + row * 3) % 11) as f32 * 0.37 - 2.0)
                .collect::<Vec<f32>>();
            let position = ordinary.len();
            let (token, logprobs) = Leader::sample_at(
                8,
                logits.clone(),
                &params,
                &prompt,
                &ordinary,
                position as u64,
            )
            .unwrap();
            ordinary.push(token);
            rows.push((logits, token, logprobs));
        }

        let drafts: Vec<u32> = rows.iter().take(3).map(|row| row.1).collect();
        for row in 0..=drafts.len() {
            let mut history = vec![ordinary[0]];
            history.extend_from_slice(&drafts[..row]);
            let (token, logprobs) = Leader::sample_at(
                8,
                rows[row].0.clone(),
                &params,
                &prompt,
                &history,
                (row + 1) as u64,
            )
            .unwrap();
            assert_eq!(token, rows[row].1);
            assert_eq!(logprobs, rows[row].2);
        }

        let rejected = [drafts[0], (drafts[1] + 1) % 8];
        let history = [ordinary[0], rejected[0]];
        let (token, logprobs) =
            Leader::sample_at(8, rows[1].0.clone(), &params, &prompt, &history, 2).unwrap();
        assert_eq!(token, ordinary[2]);
        assert_eq!(logprobs, rows[1].2);
        assert_ne!(token, rejected[1]);
    }
}
