use std::env;
use std::fs;
use std::process;
use std::thread;
use std::time::Instant;

use ferrisinfer_io::{ChatMessage, HfSource, ModelSource, Tokenizer, VocabularyTokenizer};
use ferrisinfer_kernel::{Backend, CpuBackend};
use ferrisinfer_model::DecoderOnlyModel;
use ferrisinfer_runtime::{ExecutionMode, GenerationRequest, InferenceEngine, SessionConfig};

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

struct ReferenceGenerationReport {
    rendered: String,
    prompt_token_ids: Vec<u32>,
    output: ferrisinfer_runtime::GenerationOutput,
    load_elapsed: std::time::Duration,
    forward_elapsed: std::time::Duration,
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
    let artifacts = engine.load_from_source(&mut source).map_err(|error| {
        eprintln!("failed to load model artifacts: {}", error);
    })?;
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
