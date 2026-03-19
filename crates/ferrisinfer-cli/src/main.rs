use std::env;
use std::fs;
use std::io::{self, Write};
use std::process;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use ferrisinfer_io::{ChatMessage, HfSource, ModelSource, Tokenizer, VocabularyTokenizer};
use ferrisinfer_kernel::{Backend, CpuBackend};
use ferrisinfer_model::DecoderOnlyModel;
use ferrisinfer_runtime::{
    ExecutionMode, GenerationFinishReason, GenerationRequest, InferenceEngine, LoadedArtifacts,
    Session, SessionConfig, TokenSample,
};

const DEFAULT_HF_PATH: &str = "models/Qwen2.5-0.5B-Instruct";
const DEFAULT_SYSTEM_PROMPT_PATH: &str = "docs/system_prompt.md";
const DEFAULT_SHORT_SYSTEM_PROMPT_PATH: &str = "docs/system_prompt_short.md";

fn main() {
    let engine = InferenceEngine::new(CpuBackend::default());
    let args: Vec<String> = env::args().collect();

    let result = match args.get(1).map(String::as_str) {
        Some("plan") => {
            print_plan(&engine, &SessionConfig::default());
            Ok(())
        }
        Some("inspect-hf") => inspect_hf(args.get(2).map(String::as_str)),
        Some("profile-hf-load") => profile_hf_load(args.get(2).map(String::as_str)),
        Some("profile-hf") => profile_hf(
            &engine,
            args.get(2).map(String::as_str),
            args.get(3).map(String::as_str),
        ),
        Some("profile-hf-short") => profile_hf_with_prompt(
            &engine,
            args.get(2).map(String::as_str),
            args.get(3).map(String::as_str),
            default_short_system_prompt_path(),
        ),
        Some("chat-hf") => chat_hf(&engine, args.get(2).map(String::as_str)),
        Some("chat-hf-short") => chat_hf_with_prompt(
            &engine,
            args.get(2).map(String::as_str),
            default_short_system_prompt_path(),
        ),
        Some("render-chat-hf") => render_chat_hf(args.get(2).map(String::as_str)),
        Some("render-chat-hf-short") => render_chat_hf_with_prompt(
            args.get(2).map(String::as_str),
            default_short_system_prompt_path(),
        ),
        Some("tokenize-hf") => tokenize_hf(args.get(2).map(String::as_str)),
        Some("tokenize-file-hf") => tokenize_file_hf(args.get(2).map(String::as_str)),
        Some("smoke-hf") => smoke_hf(&engine, args.get(2).map(String::as_str)),
        Some("smoke-hf-short") => smoke_hf_with_prompt(
            &engine,
            args.get(2).map(String::as_str),
            default_short_system_prompt_path(),
        ),
        Some("generate-hf") => generate_hf(
            &engine,
            args.get(2).map(String::as_str),
            args.get(3).map(String::as_str),
        ),
        Some("generate-hf-short") => generate_hf_with_prompt(
            &engine,
            args.get(2).map(String::as_str),
            args.get(3).map(String::as_str),
            default_short_system_prompt_path(),
        ),
        Some("help") | None => {
            print_help();
            Ok(())
        }
        Some(other) => {
            eprintln!("unknown command: {other}");
            print_help();
            Err(())
        }
    };

    if result.is_err() {
        process::exit(1);
    }
}

fn print_help() {
    println!("FerrisInfer CLI");
    println!("Commands:");
    println!("  help                           Show this message");
    println!("  plan                           Print the current implementation framework");
    println!("  inspect-hf [path]              Inspect a local Hugging Face model directory");
    println!(
        "  profile-hf-load [path]         Measure config/tokenizer/weights/model load timings"
    );
    println!("  profile-hf [text] [tokens]     Measure load/session/prefill/decode timings");
    println!("  profile-hf-short [text] [tokens] Measure timings with docs/system_prompt_short.md");
    println!("  chat-hf [tokens]               Start an interactive multi-turn chat session");
    println!("  chat-hf-short [tokens]         Start chat with docs/system_prompt_short.md");
    println!(
        "  render-chat-hf [text]          Render the default Qwen chat prompt for a user message"
    );
    println!(
        "  render-chat-hf-short [text]    Render the short test system prompt for a user message"
    );
    println!(
        "  tokenize-hf [text]             Tokenize plain text with the default local HF tokenizer"
    );
    println!(
        "  tokenize-file-hf [file]        Tokenize a text file with the default local HF tokenizer"
    );
    println!("  smoke-hf [text]                Run a reference next-token smoke test on the local HF model");
    println!(
        "  smoke-hf-short [text]          Run the smoke test with docs/system_prompt_short.md"
    );
    println!(
        "  generate-hf [text] [tokens]    Run reference generation with docs/system_prompt.md"
    );
    println!(
        "  generate-hf-short [text] [tokens] Run reference generation with docs/system_prompt_short.md"
    );
}

fn print_plan(engine: &InferenceEngine<CpuBackend>, config: &SessionConfig) {
    let prefill = engine.build_plan(ExecutionMode::Prefill);
    let decode = engine.build_plan(ExecutionMode::Decode);

    println!("FerrisInfer implementation framework");
    println!("backend: {}", engine.backend().name());
    println!(
        "default max sequence length: {}",
        config.max_sequence_length
    );
    println!("prefill steps: {}", prefill.steps.len());
    println!("decode steps: {}", decode.steps.len());
    println!("layers: core -> kernel -> model -> io -> runtime -> cli");
}

fn inspect_hf(path: Option<&str>) -> Result<(), ()> {
    let path = path.unwrap_or(default_hf_path());
    let source = HfSource::new(path);
    let inspection = source.inspect().map_err(|error| {
        eprintln!(
            "failed to inspect Hugging Face model at '{}': {}",
            path, error
        );
    })?;

    println!("FerrisInfer Hugging Face inspection");
    println!("path: {}", source.path().display());
    println!("family: {}", inspection.family);
    println!("architecture: {}", inspection.architecture_name);
    println!(
        "model: hidden={} intermediate={} layers={} vocab={}",
        inspection.config.hidden_size,
        inspection.config.intermediate_size,
        inspection.config.num_layers,
        inspection.config.vocab_size
    );
    println!(
        "attention: heads={} kv_heads={} head_dim={} tie_embeddings={}",
        inspection.config.num_attention_heads,
        inspection.config.num_key_value_heads,
        inspection.config.head_dim(),
        inspection.config.tie_word_embeddings
    );
    println!(
        "rope: theta={} rotary_dims={} max_positions={}",
        inspection.config.rope.theta,
        inspection.config.rope.rotary_dims,
        inspection.config.max_position_embeddings
    );
    println!(
        "tokenizer: kind={:?} vocab={} bos={:?} eos={:?}",
        inspection.tokenizer.kind,
        inspection.tokenizer.vocab_size,
        inspection.tokenizer.bos_token_id,
        inspection.tokenizer.eos_token_id
    );
    println!(
        "weights: shards={} tensors={} bytes={}",
        inspection.shard_count, inspection.tensor_count, inspection.total_data_bytes
    );
    println!(
        "mapped tensors: required={} optional={}",
        inspection.mapped_required_tensors, inspection.mapped_optional_tensors
    );

    if inspection.missing_required_tensors.is_empty() {
        println!("missing required tensors: none");
    } else {
        println!(
            "missing required tensors: {}",
            inspection.missing_required_tensors.join(", ")
        );
    }

    if inspection.missing_optional_tensors.is_empty() {
        println!("missing optional tensors: none");
    } else {
        println!(
            "missing optional tensors: {}",
            inspection.missing_optional_tensors.join(", ")
        );
    }

    if inspection.shape_mismatches.is_empty() {
        println!("shape mismatches: none");
    } else {
        println!("shape mismatches:");
        for mismatch in &inspection.shape_mismatches {
            println!("  {mismatch}");
        }
    }

    println!(
        "compatible with current loader: {}",
        if inspection.is_compatible() {
            "yes"
        } else {
            "no"
        }
    );
    Ok(())
}

fn profile_hf_load(path: Option<&str>) -> Result<(), ()> {
    let path = path.unwrap_or(default_hf_path());
    let mut source = HfSource::new(path);
    let available_parallelism = thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1);

    let total_start = Instant::now();

    let config_start = Instant::now();
    let config = source.load_config().map_err(|error| {
        eprintln!("failed to load config from '{}': {}", path, error);
    })?;
    let config_elapsed = config_start.elapsed();

    let tokenizer_start = Instant::now();
    let tokenizer = source.load_tokenizer().map_err(|error| {
        eprintln!("failed to load tokenizer from '{}': {}", path, error);
    })?;
    let tokenizer_elapsed = tokenizer_start.elapsed();

    let weights_start = Instant::now();
    let weights = source.load_weights().map_err(|error| {
        eprintln!("failed to load weights from '{}': {}", path, error);
    })?;
    let weights_elapsed = weights_start.elapsed();
    let weight_count = weights.len();

    let model_build_start = Instant::now();
    let model = DecoderOnlyModel::new(config, weights).map_err(|error| {
        eprintln!("failed to build model from '{}': {}", path, error);
    })?;
    let model_build_elapsed = model_build_start.elapsed();
    let total_elapsed = total_start.elapsed();

    println!("FerrisInfer Hugging Face load profile");
    println!("path: {}", source.path().display());
    println!("available parallelism: {}", available_parallelism);
    println!(
        "model: hidden={} intermediate={} layers={} vocab={}",
        model.config().hidden_size,
        model.config().intermediate_size,
        model.config().num_layers,
        model.config().vocab_size
    );
    println!(
        "tokenizer: kind={:?} vocab={} bos={:?} eos={:?}",
        tokenizer.kind, tokenizer.vocab_size, tokenizer.bos_token_id, tokenizer.eos_token_id
    );
    println!("loaded tensors: {}", weight_count);
    println!("config elapsed: {:.3?}", config_elapsed);
    println!("tokenizer elapsed: {:.3?}", tokenizer_elapsed);
    println!("weights elapsed: {:.3?}", weights_elapsed);
    println!("model build elapsed: {:.3?}", model_build_elapsed);
    println!("total elapsed: {:.3?}", total_elapsed);
    Ok(())
}

fn chat_hf(engine: &InferenceEngine<CpuBackend>, max_new_tokens: Option<&str>) -> Result<(), ()> {
    chat_hf_with_prompt(engine, max_new_tokens, default_system_prompt_path())
}

fn chat_hf_with_prompt(
    engine: &InferenceEngine<CpuBackend>,
    max_new_tokens: Option<&str>,
    system_prompt_path: &str,
) -> Result<(), ()> {
    let max_new_tokens = parse_max_new_tokens(max_new_tokens, 32)?;
    let mut source = HfSource::new(default_hf_path());
    let tokenizer_asset = source.load_tokenizer().map_err(|error| {
        eprintln!(
            "failed to load tokenizer from '{}': {}",
            default_hf_path(),
            error
        );
    })?;
    let tokenizer = VocabularyTokenizer::new(tokenizer_asset);
    let system_prompt = load_system_prompt(system_prompt_path)?;
    let stop_config = resolve_chat_stop_config(&tokenizer)?;

    let load_start = Instant::now();
    let model = Arc::new(source.load_model().map_err(|error| {
        eprintln!("failed to load model artifacts: {}", error);
    })?);
    let artifacts = LoadedArtifacts {
        model,
        tokenizer: Arc::new(tokenizer.clone()) as Arc<dyn Tokenizer>,
    };
    let mut session = engine
        .create_session(&artifacts, SessionConfig::default())
        .map_err(|error| {
            eprintln!("failed to create inference session: {}", error);
        })?;
    let load_elapsed = load_start.elapsed();

    let mut messages = vec![ChatMessage::system(system_prompt)];

    println!("FerrisInfer interactive chat");
    println!("model path: {}", default_hf_path());
    println!("system prompt path: {}", system_prompt_path);
    println!("max new tokens per turn: {}", max_new_tokens);
    println!("model load elapsed: {:.3?}", load_elapsed);
    println!("commands: /reset /stats /exit");

    let stdin = io::stdin();
    loop {
        print!("user> ");
        io::stdout().flush().map_err(|error| {
            eprintln!("failed to flush stdout: {}", error);
        })?;

        let mut line = String::new();
        let bytes_read = stdin.read_line(&mut line).map_err(|error| {
            eprintln!("failed to read stdin: {}", error);
        })?;
        if bytes_read == 0 {
            println!();
            break;
        }

        let input = line.trim();
        if input.is_empty() {
            continue;
        }

        match input {
            "/exit" | "/quit" => break,
            "/reset" => {
                session.reset();
                messages.truncate(1);
                println!("session reset");
                continue;
            }
            "/stats" => {
                println!(
                    "messages={} cached_tokens={} kv_used={}",
                    messages.len(),
                    session.token_history().len(),
                    session.kv_cache().used_tokens()
                );
                continue;
            }
            _ => {}
        }

        messages.push(ChatMessage::user(input));
        let turn = run_chat_turn(
            &tokenizer,
            &mut session,
            &mut messages,
            max_new_tokens,
            &stop_config,
        )?;

        println!();
        println!(
            "turn stats: reused_prompt_tokens={} appended_prompt_tokens={} generated_tokens={} finish={:?} render={:.3?} tokenize={:.3?} sync={:.3?} decode={:.3?}",
            turn.reused_prompt_tokens,
            turn.appended_prompt_tokens,
            turn.generated_tokens,
            turn.finish_reason,
            turn.render_elapsed,
            turn.tokenize_elapsed,
            turn.sync_elapsed,
            turn.decode_elapsed
        );
    }

    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct ChatStopConfig {
    primary_stop_token_id: Option<u32>,
    fallback_stop_token_id: Option<u32>,
}

#[derive(Debug, Clone)]
struct ChatTurnReport {
    reused_prompt_tokens: usize,
    appended_prompt_tokens: usize,
    generated_tokens: usize,
    finish_reason: GenerationFinishReason,
    render_elapsed: std::time::Duration,
    tokenize_elapsed: std::time::Duration,
    sync_elapsed: std::time::Duration,
    decode_elapsed: std::time::Duration,
}

fn run_chat_turn(
    tokenizer: &VocabularyTokenizer,
    session: &mut Session,
    messages: &mut Vec<ChatMessage>,
    max_new_tokens: usize,
    stop_config: &ChatStopConfig,
) -> Result<ChatTurnReport, ()> {
    print!("assistant> ");
    io::stdout().flush().map_err(|error| {
        eprintln!("failed to flush stdout: {}", error);
    })?;

    let render_start = Instant::now();
    let rendered = tokenizer.render_chat(messages, true).map_err(|error| {
        eprintln!("failed to render chat prompt: {}", error);
    })?;
    let render_elapsed = render_start.elapsed();

    let tokenize_start = Instant::now();
    let prompt_token_ids = tokenizer.encode(&rendered, false).map_err(|error| {
        eprintln!("failed to tokenize rendered prompt: {}", error);
    })?;
    let tokenize_elapsed = tokenize_start.elapsed();

    let sync_start = Instant::now();
    let (reused_prompt_tokens, appended_prompt_tokens) =
        sync_session_to_prompt(session, &prompt_token_ids)?;
    let sync_elapsed = sync_start.elapsed();

    let decode_start = Instant::now();
    let (generated_tokens, assistant_text, finish_reason) =
        stream_reference_tokens_with_fallback_stop(
            tokenizer,
            session,
            max_new_tokens,
            stop_config,
        )?;
    let decode_elapsed = decode_start.elapsed();

    messages.push(ChatMessage::assistant(assistant_text.clone()));

    Ok(ChatTurnReport {
        reused_prompt_tokens,
        appended_prompt_tokens,
        generated_tokens: generated_tokens.len(),
        finish_reason,
        render_elapsed,
        tokenize_elapsed,
        sync_elapsed,
        decode_elapsed,
    })
}

fn sync_session_to_prompt(
    session: &mut Session,
    prompt_token_ids: &[u32],
) -> Result<(usize, usize), ()> {
    let history = session.token_history();
    let common_prefix = common_prefix_len(history, prompt_token_ids);
    let reset_required = common_prefix < history.len();
    let prefill_start = if reset_required {
        session.reset();
        0
    } else {
        common_prefix
    };

    if prefill_start < prompt_token_ids.len() {
        session
            .prefill_tokens(&prompt_token_ids[prefill_start..])
            .map_err(|error| {
                eprintln!("failed to prefill chat prompt tokens: {}", error);
            })?;
    }

    Ok((
        if reset_required { 0 } else { common_prefix },
        prompt_token_ids.len() - prefill_start,
    ))
}

fn common_prefix_len(lhs: &[u32], rhs: &[u32]) -> usize {
    lhs.iter()
        .zip(rhs.iter())
        .take_while(|(left, right)| left == right)
        .count()
}

fn resolve_chat_stop_config(tokenizer: &VocabularyTokenizer) -> Result<ChatStopConfig, ()> {
    let primary_stop_token_id = tokenizer
        .encode("<|im_end|>", false)
        .map_err(|error| {
            eprintln!("failed to tokenize chat stop token: {}", error);
        })?
        .as_slice()
        .try_into()
        .ok()
        .map(|[token_id]: [u32; 1]| token_id);

    Ok(ChatStopConfig {
        primary_stop_token_id,
        fallback_stop_token_id: tokenizer.asset().eos_token_id,
    })
}

fn stream_reference_tokens_with_fallback_stop(
    tokenizer: &VocabularyTokenizer,
    session: &mut Session,
    max_new_tokens: usize,
    stop_config: &ChatStopConfig,
) -> Result<(Vec<TokenSample>, String, GenerationFinishReason), ()> {
    let generation_budget = max_new_tokens.min(session.config().max_generated_tokens);
    let mut generated_tokens = Vec::with_capacity(generation_budget);
    let mut displayed_text = String::new();
    let mut finish_reason = if generation_budget < max_new_tokens {
        GenerationFinishReason::SessionLimit
    } else {
        GenerationFinishReason::MaxNewTokens
    };

    for _ in 0..generation_budget {
        if session.position() >= session.config().max_sequence_length {
            finish_reason = GenerationFinishReason::SequenceLength;
            break;
        }

        let sample = session.step_reference().map_err(|error| {
            eprintln!("chat decode failed: {}", error);
        })?;
        generated_tokens.push(sample);

        let assistant_token_ids = assistant_display_token_ids(&generated_tokens, stop_config);
        let decoded = tokenizer.decode(&assistant_token_ids).map_err(|error| {
            eprintln!("failed to decode assistant tokens: {}", error);
        })?;
        if let Some(delta) = decoded.strip_prefix(&displayed_text) {
            if !delta.is_empty() {
                print!("{delta}");
                io::stdout().flush().map_err(|error| {
                    eprintln!("failed to flush stdout: {}", error);
                })?;
            }
        } else if decoded != displayed_text {
            print!("\n[decode resync]\n{decoded}");
            io::stdout().flush().map_err(|error| {
                eprintln!("failed to flush stdout: {}", error);
            })?;
        }
        displayed_text = decoded;

        if stop_config
            .primary_stop_token_id
            .is_some_and(|stop| stop == sample.token_id)
            || stop_config
                .fallback_stop_token_id
                .is_some_and(|stop| stop == sample.token_id)
        {
            finish_reason = GenerationFinishReason::StopToken;
            break;
        }
    }

    Ok((generated_tokens, displayed_text, finish_reason))
}

fn assistant_display_token_ids(samples: &[TokenSample], stop_config: &ChatStopConfig) -> Vec<u32> {
    let mut token_ids = samples
        .iter()
        .map(|sample| sample.token_id)
        .collect::<Vec<_>>();
    if token_ids.last().copied().is_some_and(|token_id| {
        stop_config
            .primary_stop_token_id
            .is_some_and(|stop| stop == token_id)
            || stop_config
                .fallback_stop_token_id
                .is_some_and(|stop| stop == token_id)
    }) {
        token_ids.pop();
    }
    token_ids
}

fn render_chat_hf(text: Option<&str>) -> Result<(), ()> {
    render_chat_hf_with_prompt(text, default_system_prompt_path())
}

fn render_chat_hf_with_prompt(text: Option<&str>, system_prompt_path: &str) -> Result<(), ()> {
    let tokenizer = load_default_hf_tokenizer()?;
    let user_text = text.unwrap_or("请你用一句话概括这份 system prompt 的核心约束。");
    let system_prompt = load_system_prompt(system_prompt_path)?;
    let rendered = render_default_chat_prompt(&tokenizer, user_text, &system_prompt)?;

    println!("FerrisInfer rendered chat prompt");
    println!("model path: {}", default_hf_path());
    println!("system prompt path: {}", system_prompt_path);
    println!("system prompt chars: {}", system_prompt.chars().count());
    println!("chars: {}", rendered.chars().count());
    println!("bytes: {}", rendered.len());
    println!("preview: {}", preview_text(&rendered, 240));
    Ok(())
}

fn tokenize_hf(text: Option<&str>) -> Result<(), ()> {
    let tokenizer = load_default_hf_tokenizer()?;
    let text = text.unwrap_or("你好，FerrisInfer。请输出一段简短的自检结果。");
    let tokens = tokenizer.encode(text, false).map_err(|error| {
        eprintln!("failed to tokenize text: {}", error);
    })?;
    let decoded = tokenizer.decode(&tokens).map_err(|error| {
        eprintln!("failed to decode tokens: {}", error);
    })?;

    print_token_report("inline text", text, &tokens, &decoded);
    Ok(())
}

fn tokenize_file_hf(path: Option<&str>) -> Result<(), ()> {
    let file_path = path.unwrap_or(default_system_prompt_path());
    let text = fs::read_to_string(file_path).map_err(|error| {
        eprintln!("failed to read text file '{}': {}", file_path, error);
    })?;
    let tokenizer = load_default_hf_tokenizer()?;
    let tokens = tokenizer.encode(&text, false).map_err(|error| {
        eprintln!("failed to tokenize file '{}': {}", file_path, error);
    })?;
    let decoded = tokenizer.decode(&tokens).map_err(|error| {
        eprintln!("failed to decode tokens for '{}': {}", file_path, error);
    })?;

    print_token_report(file_path, &text, &tokens, &decoded);
    Ok(())
}

fn smoke_hf(engine: &InferenceEngine<CpuBackend>, text: Option<&str>) -> Result<(), ()> {
    smoke_hf_with_prompt(engine, text, default_system_prompt_path())
}

fn smoke_hf_with_prompt(
    engine: &InferenceEngine<CpuBackend>,
    text: Option<&str>,
    system_prompt_path: &str,
) -> Result<(), ()> {
    let user_text = text.unwrap_or("请只回复一个词：OK");
    let report = run_reference_generation(engine, user_text, 1, system_prompt_path)?;

    println!("FerrisInfer reference smoke test");
    println!("model path: {}", default_hf_path());
    println!("system prompt path: {}", system_prompt_path);
    println!("user text: {}", preview_text(user_text, 120));
    println!("rendered prompt chars: {}", report.rendered.chars().count());
    println!("prompt tokens: {}", report.prompt_token_ids.len());
    println!(
        "prompt token ids (first 32): {}",
        preview_token_ids(&report.prompt_token_ids, 32)
    );
    println!("model load elapsed: {:.3?}", report.load_elapsed);
    println!("forward elapsed: {:.3?}", report.forward_elapsed);
    println!(
        "generation finish reason: {:?}",
        report.output.finish_reason
    );

    let Some(sample) = report.output.generated_tokens.first().copied() else {
        eprintln!("reference generation did not produce any token");
        return Err(());
    };

    println!("predicted token id: {}", sample.token_id);
    println!("predicted token probability: {:.6}", sample.probability);
    println!(
        "predicted token text: {}",
        preview_text(&report.output.generated_text, 80)
    );
    Ok(())
}

fn generate_hf(
    engine: &InferenceEngine<CpuBackend>,
    text: Option<&str>,
    max_new_tokens: Option<&str>,
) -> Result<(), ()> {
    generate_hf_with_prompt(engine, text, max_new_tokens, default_system_prompt_path())
}

fn generate_hf_with_prompt(
    engine: &InferenceEngine<CpuBackend>,
    text: Option<&str>,
    max_new_tokens: Option<&str>,
    system_prompt_path: &str,
) -> Result<(), ()> {
    let user_text = text.unwrap_or("请用一句话概括这份 system prompt 的核心限制。");
    let max_new_tokens = parse_max_new_tokens(max_new_tokens, 4)?;
    let report = run_reference_generation(engine, user_text, max_new_tokens, system_prompt_path)?;

    println!("FerrisInfer reference generation");
    println!("model path: {}", default_hf_path());
    println!("system prompt path: {}", system_prompt_path);
    println!("user text: {}", preview_text(user_text, 120));
    println!("requested max new tokens: {}", max_new_tokens);
    println!("rendered prompt chars: {}", report.rendered.chars().count());
    println!("prompt tokens: {}", report.prompt_token_ids.len());
    println!("generated tokens: {}", report.output.generated_tokens.len());
    println!("model load elapsed: {:.3?}", report.load_elapsed);
    println!("forward elapsed: {:.3?}", report.forward_elapsed);
    println!(
        "generation finish reason: {:?}",
        report.output.finish_reason
    );
    println!(
        "generated token ids (first 32): {}",
        preview_token_ids(&report.output.generated_token_ids(), 32)
    );
    println!(
        "generated text: {}",
        preview_text(&report.output.generated_text, 240)
    );
    Ok(())
}

fn profile_hf(
    engine: &InferenceEngine<CpuBackend>,
    text: Option<&str>,
    max_new_tokens: Option<&str>,
) -> Result<(), ()> {
    profile_hf_with_prompt(engine, text, max_new_tokens, default_system_prompt_path())
}

fn profile_hf_with_prompt(
    engine: &InferenceEngine<CpuBackend>,
    text: Option<&str>,
    max_new_tokens: Option<&str>,
    system_prompt_path: &str,
) -> Result<(), ()> {
    let user_text = text.unwrap_or("请只回复一个词：OK");
    let max_new_tokens = parse_max_new_tokens_allow_zero(max_new_tokens, 1)?;
    let report =
        run_reference_generation_profile(engine, user_text, max_new_tokens, system_prompt_path)?;

    println!("FerrisInfer reference generation profile");
    println!("model path: {}", default_hf_path());
    println!("system prompt path: {}", system_prompt_path);
    println!("user text: {}", preview_text(user_text, 120));
    println!("requested max new tokens: {}", max_new_tokens);
    println!("rendered prompt chars: {}", report.rendered.chars().count());
    println!("prompt tokens: {}", report.prompt_token_ids.len());
    println!("generated tokens: {}", report.generated_tokens.len());
    println!("render prompt elapsed: {:.3?}", report.render_elapsed);
    println!("tokenize prompt elapsed: {:.3?}", report.tokenize_elapsed);
    println!("model load elapsed: {:.3?}", report.load_elapsed);
    println!(
        "session create elapsed: {:.3?}",
        report.session_create_elapsed
    );
    println!("prefill elapsed: {:.3?}", report.prefill_elapsed);
    println!("decode elapsed: {:.3?}", report.decode_elapsed);
    println!("total elapsed: {:.3?}", report.total_elapsed);
    println!("generation finish reason: {:?}", report.finish_reason);
    println!(
        "generated token ids (first 32): {}",
        preview_token_ids(
            &report
                .generated_tokens
                .iter()
                .map(|sample| sample.token_id)
                .collect::<Vec<_>>(),
            32
        )
    );
    println!(
        "generated text: {}",
        preview_text(&report.generated_text, 240)
    );
    Ok(())
}

struct ReferenceGenerationReport {
    rendered: String,
    prompt_token_ids: Vec<u32>,
    output: ferrisinfer_runtime::GenerationOutput,
    load_elapsed: std::time::Duration,
    forward_elapsed: std::time::Duration,
}

struct ReferenceGenerationProfileReport {
    rendered: String,
    prompt_token_ids: Vec<u32>,
    generated_tokens: Vec<TokenSample>,
    generated_text: String,
    finish_reason: GenerationFinishReason,
    render_elapsed: std::time::Duration,
    tokenize_elapsed: std::time::Duration,
    load_elapsed: std::time::Duration,
    session_create_elapsed: std::time::Duration,
    prefill_elapsed: std::time::Duration,
    decode_elapsed: std::time::Duration,
    total_elapsed: std::time::Duration,
}

fn run_reference_generation(
    engine: &InferenceEngine<CpuBackend>,
    user_text: &str,
    max_new_tokens: usize,
    system_prompt_path: &str,
) -> Result<ReferenceGenerationReport, ()> {
    let mut source = HfSource::new(default_hf_path());
    let tokenizer_asset = source.load_tokenizer().map_err(|error| {
        eprintln!(
            "failed to load tokenizer from '{}': {}",
            default_hf_path(),
            error
        );
    })?;
    let tokenizer = VocabularyTokenizer::new(tokenizer_asset);
    let system_prompt = load_system_prompt(system_prompt_path)?;
    let rendered = render_default_chat_prompt(&tokenizer, user_text, &system_prompt)?;
    let token_ids = tokenizer.encode(&rendered, false).map_err(|error| {
        eprintln!("failed to tokenize rendered prompt: {}", error);
    })?;
    let stop_token_id = tokenizer.asset().eos_token_id;

    let load_start = Instant::now();
    let model = Arc::new(source.load_model().map_err(|error| {
        eprintln!("failed to load model artifacts: {}", error);
    })?);
    let artifacts = LoadedArtifacts {
        model,
        tokenizer: Arc::new(tokenizer) as Arc<dyn Tokenizer>,
    };
    let load_elapsed = load_start.elapsed();

    let forward_start = Instant::now();
    let mut session = engine
        .create_session(&artifacts, SessionConfig::default())
        .map_err(|error| {
            eprintln!("failed to create inference session: {}", error);
        })?;
    let output = session
        .generate_reference(&GenerationRequest {
            prompt: rendered.clone(),
            max_new_tokens,
            add_bos: false,
            stop_token_id,
        })
        .map_err(|error| {
            eprintln!("reference generation failed: {}", error);
        })?;
    let forward_elapsed = forward_start.elapsed();

    if output.prompt_token_ids != token_ids {
        eprintln!("warning: session prompt tokenization diverged from CLI preview");
    }

    Ok(ReferenceGenerationReport {
        rendered,
        prompt_token_ids: token_ids,
        output,
        load_elapsed,
        forward_elapsed,
    })
}

fn run_reference_generation_profile(
    engine: &InferenceEngine<CpuBackend>,
    user_text: &str,
    max_new_tokens: usize,
    system_prompt_path: &str,
) -> Result<ReferenceGenerationProfileReport, ()> {
    let total_start = Instant::now();
    let mut source = HfSource::new(default_hf_path());
    let tokenizer_asset = source.load_tokenizer().map_err(|error| {
        eprintln!(
            "failed to load tokenizer from '{}': {}",
            default_hf_path(),
            error
        );
    })?;
    let tokenizer = VocabularyTokenizer::new(tokenizer_asset);
    let system_prompt = load_system_prompt(system_prompt_path)?;

    let render_start = Instant::now();
    let rendered = render_default_chat_prompt(&tokenizer, user_text, &system_prompt)?;
    let render_elapsed = render_start.elapsed();

    let tokenize_start = Instant::now();
    let prompt_token_ids = tokenizer.encode(&rendered, false).map_err(|error| {
        eprintln!("failed to tokenize rendered prompt: {}", error);
    })?;
    let tokenize_elapsed = tokenize_start.elapsed();
    let stop_token_id = tokenizer.asset().eos_token_id;

    let load_start = Instant::now();
    let model = Arc::new(source.load_model().map_err(|error| {
        eprintln!("failed to load model artifacts: {}", error);
    })?);
    let artifacts = LoadedArtifacts {
        model,
        tokenizer: Arc::new(tokenizer) as Arc<dyn Tokenizer>,
    };
    let load_elapsed = load_start.elapsed();

    let session_create_start = Instant::now();
    let mut session = engine
        .create_session(&artifacts, SessionConfig::default())
        .map_err(|error| {
            eprintln!("failed to create inference session: {}", error);
        })?;
    let session_create_elapsed = session_create_start.elapsed();

    let prefill_start = Instant::now();
    session.prefill_tokens(&prompt_token_ids).map_err(|error| {
        eprintln!("reference prefill failed: {}", error);
    })?;
    let prefill_elapsed = prefill_start.elapsed();

    let decode_start = Instant::now();
    let (generated_tokens, finish_reason) =
        step_reference_tokens(&mut session, max_new_tokens, stop_token_id)?;
    let decode_elapsed = decode_start.elapsed();

    let generated_text = artifacts
        .tokenizer
        .decode(
            &generated_tokens
                .iter()
                .map(|sample| sample.token_id)
                .collect::<Vec<_>>(),
        )
        .map_err(|error| {
            eprintln!("failed to decode generated tokens: {}", error);
        })?;
    let total_elapsed = total_start.elapsed();

    Ok(ReferenceGenerationProfileReport {
        rendered,
        prompt_token_ids,
        generated_tokens,
        generated_text,
        finish_reason,
        render_elapsed,
        tokenize_elapsed,
        load_elapsed,
        session_create_elapsed,
        prefill_elapsed,
        decode_elapsed,
        total_elapsed,
    })
}

fn step_reference_tokens(
    session: &mut Session,
    max_new_tokens: usize,
    stop_token_id: Option<u32>,
) -> Result<(Vec<TokenSample>, GenerationFinishReason), ()> {
    let generation_budget = max_new_tokens.min(session.config().max_generated_tokens);
    let mut generated_tokens = Vec::with_capacity(generation_budget);
    let mut finish_reason = if generation_budget < max_new_tokens {
        GenerationFinishReason::SessionLimit
    } else {
        GenerationFinishReason::MaxNewTokens
    };

    for _ in 0..generation_budget {
        if session.position() >= session.config().max_sequence_length {
            finish_reason = GenerationFinishReason::SequenceLength;
            break;
        }

        let sample = session.step_reference().map_err(|error| {
            eprintln!("reference decode failed: {}", error);
        })?;
        generated_tokens.push(sample);

        if stop_token_id.is_some_and(|stop| stop == sample.token_id) {
            finish_reason = GenerationFinishReason::StopToken;
            break;
        }
    }

    Ok((generated_tokens, finish_reason))
}

fn load_default_hf_tokenizer() -> Result<VocabularyTokenizer, ()> {
    let mut source = HfSource::new(default_hf_path());
    let asset = source.load_tokenizer().map_err(|error| {
        eprintln!(
            "failed to load tokenizer from '{}': {}",
            default_hf_path(),
            error
        );
    })?;
    Ok(VocabularyTokenizer::new(asset))
}

fn load_system_prompt(path: &str) -> Result<String, ()> {
    let text = fs::read_to_string(path).map_err(|error| {
        eprintln!("failed to read system prompt '{}': {}", path, error);
    })?;
    let trimmed = text.trim();
    if trimmed.is_empty() {
        eprintln!(
            "system prompt '{}' is empty after trimming whitespace",
            path
        );
        return Err(());
    }
    Ok(trimmed.to_string())
}

fn render_default_chat_prompt(
    tokenizer: &VocabularyTokenizer,
    user_text: &str,
    system_prompt: &str,
) -> Result<String, ()> {
    tokenizer
        .render_chat(
            &[
                ChatMessage::system(system_prompt),
                ChatMessage::user(user_text),
            ],
            true,
        )
        .map_err(|error| {
            eprintln!("failed to render chat prompt: {}", error);
        })
}

fn parse_max_new_tokens(raw: Option<&str>, default: usize) -> Result<usize, ()> {
    let Some(raw) = raw else {
        return Ok(default);
    };

    let parsed = raw.parse::<usize>().map_err(|error| {
        eprintln!("failed to parse max_new_tokens '{}': {}", raw, error);
    })?;
    if parsed == 0 {
        eprintln!("max_new_tokens must be greater than zero");
        return Err(());
    }

    Ok(parsed)
}

fn parse_max_new_tokens_allow_zero(raw: Option<&str>, default: usize) -> Result<usize, ()> {
    let Some(raw) = raw else {
        return Ok(default);
    };

    raw.parse::<usize>().map_err(|error| {
        eprintln!("failed to parse max_new_tokens '{}': {}", raw, error);
    })
}

fn print_token_report(label: &str, original: &str, tokens: &[u32], decoded: &str) {
    println!("FerrisInfer tokenizer report");
    println!("source: {}", label);
    println!("chars: {}", original.chars().count());
    println!("bytes: {}", original.len());
    println!("tokens: {}", tokens.len());
    println!(
        "roundtrip exact: {}",
        if decoded == original { "yes" } else { "no" }
    );
    println!("preview: {}", preview_text(original, 120));
    println!("token ids (first 32): {}", preview_token_ids(tokens, 32));
}

fn preview_text(text: &str, max_chars: usize) -> String {
    let total = text.chars().count();
    let preview = text.chars().take(max_chars).collect::<String>();
    if total > max_chars {
        format!("{}...", preview.replace('\n', "\\n"))
    } else {
        preview.replace('\n', "\\n")
    }
}

fn default_short_system_prompt_path() -> &'static str {
    DEFAULT_SHORT_SYSTEM_PROMPT_PATH
}

fn preview_token_ids(tokens: &[u32], max_items: usize) -> String {
    let preview = tokens
        .iter()
        .take(max_items)
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    if tokens.len() > max_items {
        format!("[{preview}, ...]")
    } else {
        format!("[{preview}]")
    }
}

fn default_hf_path() -> &'static str {
    DEFAULT_HF_PATH
}

fn default_system_prompt_path() -> &'static str {
    DEFAULT_SYSTEM_PROMPT_PATH
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn common_prefix_len_counts_matching_prefix_tokens() {
        assert_eq!(common_prefix_len(&[1, 2, 3, 4], &[1, 2, 9]), 2);
        assert_eq!(common_prefix_len(&[1, 2], &[1, 2, 3]), 2);
        assert_eq!(common_prefix_len(&[7, 8], &[9, 8]), 0);
    }

    #[test]
    fn assistant_display_token_ids_strips_terminal_stop_token() {
        let stop_config = ChatStopConfig {
            primary_stop_token_id: Some(42),
            fallback_stop_token_id: Some(99),
        };
        let samples = vec![
            TokenSample {
                token_id: 1,
                probability: 0.6,
            },
            TokenSample {
                token_id: 42,
                probability: 0.4,
            },
        ];

        assert_eq!(assistant_display_token_ids(&samples, &stop_config), vec![1]);
    }
}
