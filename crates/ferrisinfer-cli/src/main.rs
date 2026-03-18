use std::env;
use std::fs;
use std::process;
use std::time::Instant;

use ferrisinfer_io::{ChatMessage, HfSource, ModelSource, Tokenizer, VocabularyTokenizer};
use ferrisinfer_kernel::{Backend, CpuBackend};
use ferrisinfer_runtime::{ExecutionMode, GenerationRequest, InferenceEngine, SessionConfig};

fn main() {
    let engine = InferenceEngine::new(CpuBackend::default());
    let args: Vec<String> = env::args().collect();

    let result = match args.get(1).map(String::as_str) {
        Some("plan") => {
            print_plan(&engine, &SessionConfig::default());
            Ok(())
        }
        Some("inspect-hf") => inspect_hf(args.get(2).map(String::as_str)),
        Some("render-chat-hf") => render_chat_hf(args.get(2).map(String::as_str)),
        Some("tokenize-hf") => tokenize_hf(args.get(2).map(String::as_str)),
        Some("tokenize-file-hf") => tokenize_file_hf(args.get(2).map(String::as_str)),
        Some("smoke-hf") => smoke_hf(&engine, args.get(2).map(String::as_str)),
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
        "  render-chat-hf [text]          Render the default Qwen chat prompt for a user message"
    );
    println!(
        "  tokenize-hf [text]             Tokenize plain text with the default local HF tokenizer"
    );
    println!(
        "  tokenize-file-hf [file]        Tokenize a text file with the default local HF tokenizer"
    );
    println!("  smoke-hf [text]                Run a reference next-token smoke test on the local HF model");
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

fn render_chat_hf(text: Option<&str>) -> Result<(), ()> {
    let tokenizer = load_default_hf_tokenizer()?;
    let user_text = text.unwrap_or("请你用一句话说明 FerrisInfer 当前最重要的工程目标。");
    let rendered = tokenizer
        .render_chat(&[ChatMessage::user(user_text)], true)
        .map_err(|error| {
            eprintln!("failed to render chat prompt: {}", error);
        })?;

    println!("FerrisInfer rendered chat prompt");
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
    let file_path = path.unwrap_or("docs/参考提示词.md");
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
    let user_text = text.unwrap_or("请只回复一个词：OK");
    let mut source = HfSource::new(default_hf_path());
    let tokenizer_asset = source.load_tokenizer().map_err(|error| {
        eprintln!(
            "failed to load tokenizer from '{}': {}",
            default_hf_path(),
            error
        );
    })?;
    let tokenizer = VocabularyTokenizer::new(tokenizer_asset);

    let rendered = tokenizer
        .render_chat(&[ChatMessage::user(user_text)], true)
        .map_err(|error| {
            eprintln!("failed to render chat prompt: {}", error);
        })?;
    let token_ids = tokenizer.encode(&rendered, false).map_err(|error| {
        eprintln!("failed to tokenize rendered prompt: {}", error);
    })?;
    let stop_token_id = tokenizer.asset().eos_token_id;

    println!("FerrisInfer reference smoke test");
    println!("model path: {}", default_hf_path());
    println!("user text: {}", preview_text(user_text, 120));
    println!("rendered prompt chars: {}", rendered.chars().count());
    println!("prompt tokens: {}", token_ids.len());
    println!(
        "prompt token ids (first 32): {}",
        preview_token_ids(&token_ids, 32)
    );

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
            prompt: rendered,
            max_new_tokens: 1,
            add_bos: false,
            stop_token_id,
        })
        .map_err(|error| {
            eprintln!("reference generation failed: {}", error);
        })?;
    let forward_elapsed = forward_start.elapsed();

    let Some(sample) = output.generated_tokens.first().copied() else {
        eprintln!("reference generation did not produce any token");
        return Err(());
    };

    if output.prompt_token_ids != token_ids {
        eprintln!("warning: session prompt tokenization diverged from CLI preview");
    }

    println!("model load elapsed: {:.3?}", load_elapsed);
    println!("forward elapsed: {:.3?}", forward_elapsed);
    println!("generation finish reason: {:?}", output.finish_reason);
    println!("predicted token id: {}", sample.token_id);
    println!("predicted token probability: {:.6}", sample.probability);
    println!(
        "predicted token text: {}",
        preview_text(&output.generated_text, 80)
    );
    Ok(())
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
    "models/Qwen2.5-0.5B-Instruct"
}
