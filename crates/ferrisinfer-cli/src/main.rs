use std::env;
use std::fs;
use std::io::{self, Write};
use std::process;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use ferrisinfer_core::{Shape, Tensor};
use ferrisinfer_io::{ChatMessage, HfSource, ModelSource, Tokenizer, VocabularyTokenizer};
use ferrisinfer_kernel::{Backend, CpuBackend, NvidiaCudaBackend, NvidiaCudaProbe};
use ferrisinfer_model::DecoderOnlyModel;
use ferrisinfer_runtime::{
    ExecutionMode, GenerationFinishReason, GenerationRequest, InferenceEngine, LoadedArtifacts,
    SamplerConfig, SchedulerBatchKind, SchedulerConfig, SequenceFinishReason, SequenceId,
    SequenceSubmitRequest, Session, SessionConfig, SessionKvCacheConfig, TokenSample,
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
        Some("probe-cuda") => probe_cuda(),
        Some("smoke-cuda") => smoke_cuda(args.get(2).map(String::as_str)),
        Some("smoke-cuda-tensor") => smoke_cuda_tensor(args.get(2).map(String::as_str)),
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
        Some("profile-chat-hf") => profile_chat_hf(&engine, args.get(2).map(String::as_str)),
        Some("profile-chat-hf-short") => profile_chat_hf_with_prompt(
            &engine,
            args.get(2).map(String::as_str),
            default_short_system_prompt_path(),
        ),
        Some("profile-continuous-hf") => {
            profile_continuous_hf(&engine, args.get(2).map(String::as_str))
        }
        Some("profile-continuous-hf-short") => profile_continuous_hf_with_prompt(
            &engine,
            args.get(2).map(String::as_str),
            default_short_system_prompt_path(),
        ),
        Some("chat-hf") => chat_hf(&engine, args.get(2).map(String::as_str)),
        Some("chat-hf-greedy") => chat_hf_with_sampler(
            &engine,
            args.get(2).map(String::as_str),
            default_system_prompt_path(),
            SamplerConfig::greedy(),
        ),
        Some("chat-hf-short") => chat_hf_with_prompt(
            &engine,
            args.get(2).map(String::as_str),
            default_short_system_prompt_path(),
        ),
        Some("chat-hf-short-greedy") => chat_hf_with_sampler(
            &engine,
            args.get(2).map(String::as_str),
            default_short_system_prompt_path(),
            SamplerConfig::greedy(),
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
    println!(
        "Use `cargo run --release -p ferrisinfer-cli -- <command>` for meaningful CPU performance measurements."
    );
    println!("Commands:");
    println!("  help                           Show this message");
    println!("  plan                           Print the current implementation framework");
    println!("  probe-cuda                     Probe the NVIDIA CUDA driver and enumerate devices");
    println!("  smoke-cuda [bytes]             Allocate CUDA memory and verify host-device copies");
    println!("  smoke-cuda-tensor [elements]   Verify CUDA tensor upload/download/zero with f32 payloads");
    println!("  inspect-hf [path]              Inspect a local Hugging Face model directory");
    println!(
        "  profile-hf-load [path]         Measure config/tokenizer/weights/model load timings"
    );
    println!("  profile-hf [text] [tokens]     Measure load/session/prefill/decode timings");
    println!("  profile-hf-short [text] [tokens] Measure timings with docs/system_prompt_short.md");
    println!("  profile-chat-hf [tokens]       Measure a built-in multi-turn chat benchmark");
    println!("  profile-chat-hf-short [tokens] Measure the chat benchmark with docs/system_prompt_short.md");
    println!("  profile-continuous-hf [tokens] Measure a built-in continuous batching benchmark");
    println!("  profile-continuous-hf-short [tokens] Measure the benchmark with docs/system_prompt_short.md");
    println!("  chat-hf [tokens]               Start an interactive multi-turn chat session");
    println!("  chat-hf-greedy [tokens]        Start chat with greedy decoding");
    println!("  chat-hf-short [tokens]         Start chat with docs/system_prompt_short.md");
    println!("  chat-hf-short-greedy [tokens]  Start short-prompt chat with greedy decoding");
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
    let capabilities = engine.backend().capabilities();
    let availability = engine.backend().availability();
    let nvidia_cuda = NvidiaCudaBackend::default();
    let nvidia_cuda_probe = nvidia_cuda.probe();

    println!("FerrisInfer implementation framework");
    println!("backend: {}", engine.backend().name());
    println!("device: {:?}", engine.backend().device());
    println!(
        "backend available: {}",
        if availability.available { "yes" } else { "no" }
    );
    if let Some(reason) = availability.reason {
        println!("backend availability note: {}", reason);
    }
    println!(
        "backend capabilities: simd={} multithreaded={} quantized={} device_memory={} graph_capture={}",
        capabilities.simd,
        capabilities.multithreaded,
        capabilities.quantized_kernels,
        capabilities.device_memory,
        capabilities.graph_capture,
    );
    print_nvidia_cuda_probe_summary(nvidia_cuda_probe);
    println!(
        "nvidia cuda runtime base: context + device buffer + host-device copy prototype available"
    );
    println!(
        "nvidia cuda tensor storage: dtype + shape + host-device tensor upload/download prototype available"
    );
    println!(
        "default max sequence length: {}",
        config.max_sequence_length
    );
    println!("runtime scheduler: sequence state + reference scheduler available");
    println!("runtime batching: continuous batching reference benchmark available");
    println!("kv cache storage: contiguous default, paged prototype + logical-physical page allocator available");
    println!(
        "reference attention: paged prefill/decode reads committed KV prefixes without relying on contiguous layer views"
    );
    println!("kv prefix reuse: paged block table + prefix handle + prefix index available");
    println!("runtime lifecycle: finished sequence session/KV release + prefix refcount release available");
    println!("prefill scheduling: chunked prefill reference path available");
    println!(
        "batch scheduling: decode fairness + compaction + prefix-indexed prompt reuse available"
    );
    println!("prefill steps: {}", prefill.steps.len());
    println!("decode steps: {}", decode.steps.len());
    println!("layers: core -> kernel -> model -> io -> runtime -> cli");
}

fn probe_cuda() -> Result<(), ()> {
    let backend = NvidiaCudaBackend::default();
    println!("FerrisInfer NVIDIA CUDA probe");
    print_nvidia_cuda_probe(backend.probe());
    Ok(())
}

fn smoke_cuda(size_bytes: Option<&str>) -> Result<(), ()> {
    let size_bytes = parse_cuda_smoke_bytes(size_bytes)?;
    let backend = NvidiaCudaBackend::default();

    println!("FerrisInfer NVIDIA CUDA memory smoke");
    print_nvidia_cuda_probe(backend.probe());

    let context = backend.create_context(0).map_err(|error| {
        eprintln!("failed to create NVIDIA CUDA context: {}", error);
    })?;
    let payload = build_cuda_smoke_payload(size_bytes);

    let upload_start = Instant::now();
    let mut buffer = context.upload_bytes(&payload).map_err(|error| {
        eprintln!("failed to allocate/upload CUDA device buffer: {}", error);
    })?;
    context.synchronize().map_err(|error| {
        eprintln!("failed to synchronize CUDA context after upload: {}", error);
    })?;
    let upload_elapsed = upload_start.elapsed();

    let download_start = Instant::now();
    let downloaded = buffer.download_to_vec().map_err(|error| {
        eprintln!("failed to download CUDA device buffer: {}", error);
    })?;
    context.synchronize().map_err(|error| {
        eprintln!(
            "failed to synchronize CUDA context after download: {}",
            error
        );
    })?;
    let download_elapsed = download_start.elapsed();
    let roundtrip_exact = downloaded == payload;

    let zero_start = Instant::now();
    buffer.fill_zero().map_err(|error| {
        eprintln!("failed to zero CUDA device buffer: {}", error);
    })?;
    context.synchronize().map_err(|error| {
        eprintln!(
            "failed to synchronize CUDA context after zero fill: {}",
            error
        );
    })?;
    let zeroed = buffer.download_to_vec().map_err(|error| {
        eprintln!("failed to download zeroed CUDA device buffer: {}", error);
    })?;
    let zero_elapsed = zero_start.elapsed();
    let zero_fill_exact = zeroed.iter().all(|&byte| byte == 0);

    println!(
        "selected cuda device: cuda:{} {} cc={}.{} vram_mib={}",
        context.device().ordinal(),
        context.device().name(),
        context.device().compute_capability_major(),
        context.device().compute_capability_minor(),
        bytes_to_mib(context.device().total_memory_bytes())
    );
    println!("buffer bytes: {}", size_bytes);
    println!("upload+alloc elapsed: {:.3?}", upload_elapsed);
    println!("download elapsed: {:.3?}", download_elapsed);
    println!("zero+verify elapsed: {:.3?}", zero_elapsed);
    println!(
        "roundtrip exact: {}",
        if roundtrip_exact { "yes" } else { "no" }
    );
    println!(
        "zero fill exact: {}",
        if zero_fill_exact { "yes" } else { "no" }
    );

    if !roundtrip_exact || !zero_fill_exact {
        eprintln!("CUDA memory smoke verification failed");
        return Err(());
    }

    Ok(())
}

fn smoke_cuda_tensor(element_count: Option<&str>) -> Result<(), ()> {
    let element_count = parse_cuda_smoke_elements(element_count)?;
    let backend = NvidiaCudaBackend::default();

    println!("FerrisInfer NVIDIA CUDA tensor smoke");
    print_nvidia_cuda_probe(backend.probe());

    let context = backend.create_context(0).map_err(|error| {
        eprintln!("failed to create NVIDIA CUDA context: {}", error);
    })?;
    let payload = build_cuda_smoke_tensor_payload(element_count);
    let shape = Shape::from_slice(&[element_count]).map_err(|error| {
        eprintln!("failed to build CUDA smoke tensor shape: {}", error);
    })?;
    let host_tensor = Tensor::from_f32_vec(shape, payload.clone()).map_err(|error| {
        eprintln!("failed to build CUDA smoke host tensor: {}", error);
    })?;

    let upload_start = Instant::now();
    let mut device_tensor = context.upload_tensor(&host_tensor).map_err(|error| {
        eprintln!("failed to upload CUDA tensor: {}", error);
    })?;
    context.synchronize().map_err(|error| {
        eprintln!(
            "failed to synchronize CUDA context after tensor upload: {}",
            error
        );
    })?;
    let upload_elapsed = upload_start.elapsed();

    let download_start = Instant::now();
    let downloaded = device_tensor.download_to_tensor().map_err(|error| {
        eprintln!("failed to download CUDA tensor: {}", error);
    })?;
    context.synchronize().map_err(|error| {
        eprintln!(
            "failed to synchronize CUDA context after tensor download: {}",
            error
        );
    })?;
    let download_elapsed = download_start.elapsed();
    let roundtrip_exact = downloaded.to_vec_f32().map_err(|error| {
        eprintln!("failed to decode downloaded CUDA tensor payload: {}", error);
    })? == payload;

    let zero_start = Instant::now();
    device_tensor.fill_zero().map_err(|error| {
        eprintln!("failed to zero CUDA tensor: {}", error);
    })?;
    context.synchronize().map_err(|error| {
        eprintln!(
            "failed to synchronize CUDA context after tensor zero fill: {}",
            error
        );
    })?;
    let zeroed = device_tensor.download_to_tensor().map_err(|error| {
        eprintln!("failed to download zeroed CUDA tensor: {}", error);
    })?;
    context.synchronize().map_err(|error| {
        eprintln!(
            "failed to synchronize CUDA context after zeroed tensor download: {}",
            error
        );
    })?;
    let zero_elapsed = zero_start.elapsed();
    let zero_fill_exact = zeroed
        .to_vec_f32()
        .map_err(|error| {
            eprintln!("failed to decode zeroed CUDA tensor payload: {}", error);
        })?
        .iter()
        .all(|&value| value == 0.0);

    println!(
        "selected cuda device: cuda:{} {} cc={}.{} vram_mib={}",
        context.device().ordinal(),
        context.device().name(),
        context.device().compute_capability_major(),
        context.device().compute_capability_minor(),
        bytes_to_mib(context.device().total_memory_bytes())
    );
    println!("tensor dtype: {}", host_tensor.dtype().name());
    println!("tensor elements: {}", element_count);
    println!("tensor bytes: {}", device_tensor.len_bytes());
    println!("upload+alloc elapsed: {:.3?}", upload_elapsed);
    println!("download elapsed: {:.3?}", download_elapsed);
    println!("zero+verify elapsed: {:.3?}", zero_elapsed);
    println!(
        "roundtrip exact: {}",
        if roundtrip_exact { "yes" } else { "no" }
    );
    println!(
        "zero fill exact: {}",
        if zero_fill_exact { "yes" } else { "no" }
    );

    if !roundtrip_exact || !zero_fill_exact {
        eprintln!("CUDA tensor smoke verification failed");
        return Err(());
    }

    Ok(())
}

fn print_nvidia_cuda_probe_summary(probe: &NvidiaCudaProbe) {
    let availability = probe.availability();
    println!(
        "nvidia cuda backend available: {}",
        if availability.available { "yes" } else { "no" }
    );
    if let Some(reason) = availability.reason {
        println!("nvidia cuda availability note: {}", reason);
    }
    if let Some(detail) = probe.detail() {
        println!("nvidia cuda detail: {}", detail);
    }
    if let Some(library) = probe.driver_library() {
        println!("nvidia cuda driver library: {}", library);
    }
    if let Some(version) = probe.driver_version() {
        println!(
            "nvidia cuda driver version: {}.{}",
            version.major(),
            version.minor()
        );
    }
    println!("nvidia cuda devices: {}", probe.devices().len());
}

fn print_nvidia_cuda_probe(probe: &NvidiaCudaProbe) {
    print_nvidia_cuda_probe_summary(probe);
    for device in probe.devices() {
        println!(
            "cuda:{} name={} cc={}.{} vram_mib={}",
            device.ordinal(),
            device.name(),
            device.compute_capability_major(),
            device.compute_capability_minor(),
            bytes_to_mib(device.total_memory_bytes())
        );
    }
}

fn bytes_to_mib(bytes: u64) -> u64 {
    bytes / (1024 * 1024)
}

fn parse_cuda_smoke_bytes(raw: Option<&str>) -> Result<usize, ()> {
    let Some(raw) = raw else {
        return Ok(256);
    };

    let size_bytes = raw.parse::<usize>().map_err(|error| {
        eprintln!("failed to parse CUDA smoke byte size '{}': {}", raw, error);
    })?;
    if size_bytes == 0 {
        eprintln!("CUDA smoke byte size must be greater than zero");
        return Err(());
    }

    Ok(size_bytes)
}

fn parse_cuda_smoke_elements(raw: Option<&str>) -> Result<usize, ()> {
    let Some(raw) = raw else {
        return Ok(64);
    };

    let element_count = raw.parse::<usize>().map_err(|error| {
        eprintln!(
            "failed to parse CUDA smoke tensor element count '{}': {}",
            raw, error
        );
    })?;
    if element_count == 0 {
        eprintln!("CUDA smoke tensor element count must be greater than zero");
        return Err(());
    }

    Ok(element_count)
}

fn build_cuda_smoke_payload(size_bytes: usize) -> Vec<u8> {
    (0..size_bytes)
        .map(|index| ((index.saturating_mul(31) + 17) % 251) as u8)
        .collect()
}

fn build_cuda_smoke_tensor_payload(element_count: usize) -> Vec<f32> {
    (0..element_count)
        .map(|index| index as f32 * 0.25 - 1.5)
        .collect()
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
    warn_if_debug_profile();
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
    chat_hf_with_sampler(
        engine,
        max_new_tokens,
        default_system_prompt_path(),
        SamplerConfig::chat_default(),
    )
}

fn chat_hf_with_prompt(
    engine: &InferenceEngine<CpuBackend>,
    max_new_tokens: Option<&str>,
    system_prompt_path: &str,
) -> Result<(), ()> {
    chat_hf_with_sampler(
        engine,
        max_new_tokens,
        system_prompt_path,
        SamplerConfig::chat_default(),
    )
}

fn chat_hf_with_sampler(
    engine: &InferenceEngine<CpuBackend>,
    max_new_tokens: Option<&str>,
    system_prompt_path: &str,
    sampler: SamplerConfig,
) -> Result<(), ()> {
    warn_if_debug_profile();
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
    let mut session_config = SessionConfig::default();
    session_config.sampler = sampler;
    let mut session = engine
        .create_session(&artifacts, session_config.clone())
        .map_err(|error| {
            eprintln!("failed to create inference session: {}", error);
        })?;
    let load_elapsed = load_start.elapsed();

    let mut messages = vec![ChatMessage::system(system_prompt)];

    println!("FerrisInfer interactive chat");
    println!("model path: {}", default_hf_path());
    println!("system prompt path: {}", system_prompt_path);
    println!("max new tokens per turn: {}", max_new_tokens);
    if session_config.sampler.is_greedy() {
        println!("sampler: greedy");
    } else {
        println!(
            "sampler: temperature={:.2} top_k={} top_p={:.2} seed={}",
            session_config.sampler.temperature,
            session_config.sampler.top_k,
            session_config.sampler.top_p,
            session_config.sampler.seed
        );
    }
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
            "turn stats: reused_prompt_tokens={} appended_prompt_tokens={} generated_tokens={} finish={:?} render={:.3?} tokenize={:.3?} sync={:.3?} prefill_tok_s={:.2} decode={:.3?} decode_tok_s={:.2}",
            turn.reused_prompt_tokens,
            turn.appended_prompt_tokens,
            turn.generated_tokens,
            turn.finish_reason,
            turn.render_elapsed,
            turn.tokenize_elapsed,
            turn.sync_elapsed,
            throughput_per_second(turn.appended_prompt_tokens, turn.sync_elapsed),
            turn.decode_elapsed,
            throughput_per_second(turn.generated_tokens, turn.decode_elapsed),
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
    generate_reference_tokens_with_fallback_stop(
        tokenizer,
        session,
        max_new_tokens,
        stop_config,
        true,
    )
}

fn collect_reference_tokens_with_fallback_stop(
    tokenizer: &VocabularyTokenizer,
    session: &mut Session,
    max_new_tokens: usize,
    stop_config: &ChatStopConfig,
) -> Result<(Vec<TokenSample>, String, GenerationFinishReason), ()> {
    generate_reference_tokens_with_fallback_stop(
        tokenizer,
        session,
        max_new_tokens,
        stop_config,
        false,
    )
}

fn generate_reference_tokens_with_fallback_stop(
    tokenizer: &VocabularyTokenizer,
    session: &mut Session,
    max_new_tokens: usize,
    stop_config: &ChatStopConfig,
    stream_output: bool,
) -> Result<(Vec<TokenSample>, String, GenerationFinishReason), ()> {
    let generation_budget = max_new_tokens.min(session.config().max_generated_tokens);
    let mut generated_tokens = Vec::with_capacity(generation_budget);
    let mut displayed_text = String::new();
    let mut pending_bytes = Vec::new();
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

        if is_chat_stop_token(sample.token_id, stop_config) {
            finish_reason = GenerationFinishReason::StopToken;
            break;
        }

        let token_bytes = tokenizer
            .decode_token_bytes(sample.token_id)
            .map_err(|error| {
                eprintln!("failed to decode assistant token bytes: {}", error);
            })?;
        pending_bytes.extend_from_slice(&token_bytes);
        flush_streamable_utf8(&mut pending_bytes, &mut displayed_text, stream_output)?;
    }

    if !pending_bytes.is_empty() {
        let tail = String::from_utf8(pending_bytes).map_err(|error| {
            eprintln!("assistant stream decode failed: {}", error);
        })?;
        if stream_output {
            print!("{tail}");
            io::stdout().flush().map_err(|error| {
                eprintln!("failed to flush stdout: {}", error);
            })?;
        }
        displayed_text.push_str(&tail);
    }

    Ok((generated_tokens, displayed_text, finish_reason))
}

fn is_chat_stop_token(token_id: u32, stop_config: &ChatStopConfig) -> bool {
    stop_config
        .primary_stop_token_id
        .is_some_and(|stop| stop == token_id)
        || stop_config
            .fallback_stop_token_id
            .is_some_and(|stop| stop == token_id)
}

fn flush_streamable_utf8(
    pending_bytes: &mut Vec<u8>,
    displayed_text: &mut String,
    stream_output: bool,
) -> Result<(), ()> {
    let delta = take_valid_utf8_prefix(pending_bytes).map_err(|error| {
        eprintln!("assistant stream decode failed: {}", error);
    })?;
    if delta.is_empty() {
        return Ok(());
    }

    if stream_output {
        print!("{delta}");
        io::stdout().flush().map_err(|error| {
            eprintln!("failed to flush stdout: {}", error);
        })?;
    }
    displayed_text.push_str(&delta);
    Ok(())
}

fn take_valid_utf8_prefix(bytes: &mut Vec<u8>) -> Result<String, String> {
    if bytes.is_empty() {
        return Ok(String::new());
    }

    match std::str::from_utf8(bytes) {
        Ok(text) => {
            let prefix = text.to_string();
            bytes.clear();
            Ok(prefix)
        }
        Err(error) if error.error_len().is_none() => {
            let valid_up_to = error.valid_up_to();
            if valid_up_to == 0 {
                return Ok(String::new());
            }

            let prefix_bytes = bytes.drain(..valid_up_to).collect::<Vec<_>>();
            String::from_utf8(prefix_bytes).map_err(|utf8_error| {
                format!("valid UTF-8 prefix could not be reconstructed: {utf8_error}")
            })
        }
        Err(error) => Err(format!(
            "invalid UTF-8 sequence while streaming assistant output at byte {}",
            error.valid_up_to()
        )),
    }
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
    warn_if_debug_profile();
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
    warn_if_debug_profile();
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
    warn_if_debug_profile();
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
    println!(
        "prefill tok/s: {:.2}",
        throughput_per_second(report.prompt_token_ids.len(), report.prefill_elapsed)
    );
    println!("decode elapsed: {:.3?}", report.decode_elapsed);
    println!(
        "decode tok/s: {:.2}",
        throughput_per_second(report.generated_tokens.len(), report.decode_elapsed)
    );
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

#[derive(Debug, Clone)]
struct ScriptedChatTurnProfile {
    turn_index: usize,
    user_text: String,
    assistant_text: String,
    report: ChatTurnReport,
}

struct ScriptedChatProfileReport {
    turns: Vec<ScriptedChatTurnProfile>,
    load_elapsed: std::time::Duration,
    session_create_elapsed: std::time::Duration,
    total_elapsed: std::time::Duration,
}

fn profile_chat_hf(
    engine: &InferenceEngine<CpuBackend>,
    max_new_tokens: Option<&str>,
) -> Result<(), ()> {
    profile_chat_hf_with_prompt(engine, max_new_tokens, default_system_prompt_path())
}

fn profile_chat_hf_with_prompt(
    engine: &InferenceEngine<CpuBackend>,
    max_new_tokens: Option<&str>,
    system_prompt_path: &str,
) -> Result<(), ()> {
    warn_if_debug_profile();
    let max_new_tokens = parse_max_new_tokens(max_new_tokens, 16)?;
    let report = run_scripted_chat_profile(engine, max_new_tokens, system_prompt_path)?;
    let total_reused_prompt_tokens = report
        .turns
        .iter()
        .map(|turn| turn.report.reused_prompt_tokens)
        .sum::<usize>();
    let total_appended_prompt_tokens = report
        .turns
        .iter()
        .map(|turn| turn.report.appended_prompt_tokens)
        .sum::<usize>();
    let total_generated_tokens = report
        .turns
        .iter()
        .map(|turn| turn.report.generated_tokens)
        .sum::<usize>();
    let total_render_elapsed = report
        .turns
        .iter()
        .fold(std::time::Duration::ZERO, |acc, turn| {
            acc + turn.report.render_elapsed
        });
    let total_tokenize_elapsed = report
        .turns
        .iter()
        .fold(std::time::Duration::ZERO, |acc, turn| {
            acc + turn.report.tokenize_elapsed
        });
    let total_sync_elapsed = report
        .turns
        .iter()
        .fold(std::time::Duration::ZERO, |acc, turn| {
            acc + turn.report.sync_elapsed
        });
    let total_decode_elapsed = report
        .turns
        .iter()
        .fold(std::time::Duration::ZERO, |acc, turn| {
            acc + turn.report.decode_elapsed
        });
    let later_turns = report.turns.iter().skip(1).collect::<Vec<_>>();
    let later_turn_appended_tokens = later_turns
        .iter()
        .map(|turn| turn.report.appended_prompt_tokens)
        .sum::<usize>();
    let later_turn_sync_elapsed = later_turns
        .iter()
        .fold(std::time::Duration::ZERO, |acc, turn| {
            acc + turn.report.sync_elapsed
        });

    println!("FerrisInfer scripted chat profile");
    println!("model path: {}", default_hf_path());
    println!("system prompt path: {}", system_prompt_path);
    println!("turn source: built-in scripted benchmark");
    println!("scripted turns: {}", report.turns.len());
    println!("max new tokens per turn: {}", max_new_tokens);
    println!("sampler: greedy");
    println!("model load elapsed: {:.3?}", report.load_elapsed);
    println!(
        "session create elapsed: {:.3?}",
        report.session_create_elapsed
    );

    for turn in &report.turns {
        println!(
            "turn {}: user_chars={} reused_prompt_tokens={} appended_prompt_tokens={} generated_tokens={} finish={:?} render={:.3?} tokenize={:.3?} sync={:.3?} prefill_tok_s={:.2} decode={:.3?} decode_tok_s={:.2} reply={}",
            turn.turn_index,
            turn.user_text.chars().count(),
            turn.report.reused_prompt_tokens,
            turn.report.appended_prompt_tokens,
            turn.report.generated_tokens,
            turn.report.finish_reason,
            turn.report.render_elapsed,
            turn.report.tokenize_elapsed,
            turn.report.sync_elapsed,
            throughput_per_second(turn.report.appended_prompt_tokens, turn.report.sync_elapsed),
            turn.report.decode_elapsed,
            throughput_per_second(turn.report.generated_tokens, turn.report.decode_elapsed),
            preview_text(&turn.assistant_text, 72),
        );
    }

    println!("total reused prompt tokens: {}", total_reused_prompt_tokens);
    println!(
        "total appended prompt tokens: {}",
        total_appended_prompt_tokens
    );
    println!("total generated tokens: {}", total_generated_tokens);
    println!("total render elapsed: {:.3?}", total_render_elapsed);
    println!("total tokenize elapsed: {:.3?}", total_tokenize_elapsed);
    println!("total sync elapsed: {:.3?}", total_sync_elapsed);
    println!(
        "overall append-prefill tok/s: {:.2}",
        throughput_per_second(total_appended_prompt_tokens, total_sync_elapsed)
    );
    println!("total decode elapsed: {:.3?}", total_decode_elapsed);
    println!(
        "overall decode tok/s: {:.2}",
        throughput_per_second(total_generated_tokens, total_decode_elapsed)
    );
    if let Some(first_turn) = report.turns.first() {
        println!(
            "first turn sync elapsed: {:.3?}",
            first_turn.report.sync_elapsed
        );
    }
    if !later_turns.is_empty() {
        println!("later turns sync elapsed: {:.3?}", later_turn_sync_elapsed);
        println!(
            "later turns append-prefill tok/s: {:.2}",
            throughput_per_second(later_turn_appended_tokens, later_turn_sync_elapsed)
        );
    }
    println!("total elapsed: {:.3?}", report.total_elapsed);
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct ContinuousBatchRequestSpec {
    arrival_tick: usize,
    user_text: &'static str,
}

#[derive(Debug, Clone)]
struct PreparedContinuousBatchRequest {
    request_index: usize,
    arrival_tick: usize,
    user_text: String,
    prompt_token_ids: Vec<u32>,
}

#[derive(Debug, Clone)]
struct ContinuousBatchRequestProfile {
    request_index: usize,
    arrival_tick: usize,
    sequence_id: SequenceId,
    user_text: String,
    prompt_tokens: usize,
    generated_tokens: Vec<TokenSample>,
    first_token_tick: Option<usize>,
    finish_tick: Option<usize>,
    finish_reason: Option<SequenceFinishReason>,
    assistant_text: String,
}

#[derive(Debug, Clone)]
struct ContinuousBatchTickProfile {
    tick_index: usize,
    submitted_requests: usize,
    batch_kind: SchedulerBatchKind,
    batch_size: usize,
    active_sequences: usize,
    reused_prompt_tokens: usize,
    appended_prompt_tokens: usize,
    generated_tokens: usize,
    finished_sequences: usize,
    elapsed: std::time::Duration,
}

struct ContinuousBatchProfileReport {
    requests: Vec<ContinuousBatchRequestProfile>,
    ticks: Vec<ContinuousBatchTickProfile>,
    load_elapsed: std::time::Duration,
    scheduler_create_elapsed: std::time::Duration,
    total_elapsed: std::time::Duration,
    scheduler_config: SchedulerConfig,
    kv_cache: SessionKvCacheConfig,
}

fn profile_continuous_hf(
    engine: &InferenceEngine<CpuBackend>,
    max_new_tokens: Option<&str>,
) -> Result<(), ()> {
    profile_continuous_hf_with_prompt(engine, max_new_tokens, default_system_prompt_path())
}

fn profile_continuous_hf_with_prompt(
    engine: &InferenceEngine<CpuBackend>,
    max_new_tokens: Option<&str>,
    system_prompt_path: &str,
) -> Result<(), ()> {
    warn_if_debug_profile();
    let max_new_tokens = parse_max_new_tokens(max_new_tokens, 16)?;
    let report = run_continuous_batch_profile(engine, max_new_tokens, system_prompt_path)?;
    let total_prompt_tokens = report
        .requests
        .iter()
        .map(|request| request.prompt_tokens)
        .sum::<usize>();
    let total_generated_tokens = report
        .requests
        .iter()
        .map(|request| request.generated_tokens.len())
        .sum::<usize>();
    let prefill_ticks = report
        .ticks
        .iter()
        .filter(|tick| tick.batch_kind == SchedulerBatchKind::Prefill)
        .collect::<Vec<_>>();
    let decode_ticks = report
        .ticks
        .iter()
        .filter(|tick| tick.batch_kind == SchedulerBatchKind::Decode)
        .collect::<Vec<_>>();
    let total_prefill_elapsed = prefill_ticks
        .iter()
        .fold(std::time::Duration::ZERO, |acc, tick| acc + tick.elapsed);
    let total_decode_elapsed = decode_ticks
        .iter()
        .fold(std::time::Duration::ZERO, |acc, tick| acc + tick.elapsed);
    let total_reused_prompt_tokens = report
        .ticks
        .iter()
        .map(|tick| tick.reused_prompt_tokens)
        .sum::<usize>();
    let total_prefill_tokens = report
        .ticks
        .iter()
        .map(|tick| tick.appended_prompt_tokens)
        .sum::<usize>();
    let max_active_sequences = report
        .ticks
        .iter()
        .map(|tick| tick.active_sequences)
        .max()
        .unwrap_or(0);
    let average_batch_size = if report.ticks.is_empty() {
        0.0
    } else {
        report
            .ticks
            .iter()
            .map(|tick| tick.batch_size)
            .sum::<usize>() as f64
            / report.ticks.len() as f64
    };
    let average_decode_batch_size = if decode_ticks.is_empty() {
        0.0
    } else {
        decode_ticks
            .iter()
            .map(|tick| tick.batch_size)
            .sum::<usize>() as f64
            / decode_ticks.len() as f64
    };

    println!("FerrisInfer continuous batching profile");
    println!("model path: {}", default_hf_path());
    println!("system prompt path: {}", system_prompt_path);
    println!("request source: built-in staggered benchmark");
    println!("requests: {}", report.requests.len());
    println!("max new tokens per request: {}", max_new_tokens);
    println!(
        "scheduler batch size: {}",
        report.scheduler_config.max_batch_size
    );
    println!(
        "scheduler prefill chunk tokens: {}",
        report.scheduler_config.max_prefill_chunk_tokens
    );
    println!(
        "scheduler prefill batch token budget: {}",
        report.scheduler_config.max_prefill_batch_tokens
    );
    println!(
        "scheduler max consecutive prefill ticks: {}",
        report.scheduler_config.max_consecutive_prefill_ticks
    );
    println!(
        "scheduler min prefix share tokens: {}",
        report.scheduler_config.min_prefix_share_tokens
    );
    println!("kv cache: {:?}", report.kv_cache);
    println!("model load elapsed: {:.3?}", report.load_elapsed);
    println!(
        "scheduler create elapsed: {:.3?}",
        report.scheduler_create_elapsed
    );

    for tick in &report.ticks {
        println!(
            "tick {}: arrivals={} kind={:?} batch={} active={} reused_prompt_tokens={} appended_prompt_tokens={} generated_tokens={} finished={} elapsed={:.3?}",
            tick.tick_index,
            tick.submitted_requests,
            tick.batch_kind,
            tick.batch_size,
            tick.active_sequences,
            tick.reused_prompt_tokens,
            tick.appended_prompt_tokens,
            tick.generated_tokens,
            tick.finished_sequences,
            tick.elapsed,
        );
    }

    println!("total prompt tokens: {}", total_prompt_tokens);
    println!("total reused prompt tokens: {}", total_reused_prompt_tokens);
    println!("total generated tokens: {}", total_generated_tokens);
    println!("prefill ticks: {}", prefill_ticks.len());
    println!("decode ticks: {}", decode_ticks.len());
    println!("max active sequences: {}", max_active_sequences);
    println!("average tick batch size: {:.2}", average_batch_size);
    println!(
        "average decode batch size: {:.2}",
        average_decode_batch_size
    );
    println!("total prefill elapsed: {:.3?}", total_prefill_elapsed);
    println!(
        "effective prompt coverage tok/s: {:.2}",
        throughput_per_second(total_prompt_tokens, total_prefill_elapsed)
    );
    println!(
        "materialized prefill tok/s: {:.2}",
        throughput_per_second(total_prefill_tokens, total_prefill_elapsed)
    );
    println!("total decode elapsed: {:.3?}", total_decode_elapsed);
    println!(
        "continuous decode tok/s: {:.2}",
        throughput_per_second(total_generated_tokens, total_decode_elapsed)
    );
    println!("total elapsed: {:.3?}", report.total_elapsed);

    for request in &report.requests {
        println!(
            "request {}: arrival_tick={} prompt_tokens={} generated_tokens={} first_token_tick={} finish_tick={} finish={:?} user={} reply={}",
            request.request_index,
            request.arrival_tick,
            request.prompt_tokens,
            request.generated_tokens.len(),
            request
                .first_token_tick
                .map(|tick| tick.to_string())
                .unwrap_or_else(|| "-".to_string()),
            request
                .finish_tick
                .map(|tick| tick.to_string())
                .unwrap_or_else(|| "-".to_string()),
            request.finish_reason,
            preview_text(&request.user_text, 48),
            preview_text(&request.assistant_text, 72),
        );
    }

    Ok(())
}

fn run_continuous_batch_profile(
    engine: &InferenceEngine<CpuBackend>,
    max_new_tokens: usize,
    system_prompt_path: &str,
) -> Result<ContinuousBatchProfileReport, ()> {
    const CONTINUOUS_BATCH_PAGE_SIZE: usize = 32;
    const CONTINUOUS_BATCH_SIZE: usize = 4;
    const CONTINUOUS_BATCH_PREFILL_CHUNK_TOKENS: usize = 64;
    const CONTINUOUS_BATCH_PREFILL_BATCH_TOKENS: usize = 128;
    const CONTINUOUS_BATCH_MAX_CONSECUTIVE_PREFILL_TICKS: usize = 2;
    const CONTINUOUS_BATCH_MIN_PREFIX_SHARE_TOKENS: usize = 64;

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
    let stop_config = resolve_chat_stop_config(&tokenizer)?;
    let stop_token_id = stop_config
        .primary_stop_token_id
        .or(stop_config.fallback_stop_token_id);
    let prepared_requests =
        build_continuous_batch_requests(&tokenizer, &system_prompt, stop_token_id)?;

    let load_start = Instant::now();
    let model = Arc::new(source.load_model().map_err(|error| {
        eprintln!("failed to load model artifacts: {}", error);
    })?);
    let artifacts = LoadedArtifacts {
        model,
        tokenizer: Arc::new(tokenizer.clone()) as Arc<dyn Tokenizer>,
    };
    let load_elapsed = load_start.elapsed();

    let scheduler_session_config = SessionConfig {
        max_generated_tokens: max_new_tokens,
        sampler: SamplerConfig::greedy(),
        kv_cache: SessionKvCacheConfig::Paged {
            page_size: CONTINUOUS_BATCH_PAGE_SIZE,
        },
        ..SessionConfig::default()
    };
    let scheduler_config = SchedulerConfig {
        max_batch_size: CONTINUOUS_BATCH_SIZE,
        max_prefill_chunk_tokens: CONTINUOUS_BATCH_PREFILL_CHUNK_TOKENS,
        max_prefill_batch_tokens: CONTINUOUS_BATCH_PREFILL_BATCH_TOKENS,
        max_consecutive_prefill_ticks: CONTINUOUS_BATCH_MAX_CONSECUTIVE_PREFILL_TICKS,
        min_prefix_share_tokens: CONTINUOUS_BATCH_MIN_PREFIX_SHARE_TOKENS,
    };
    let scheduler_create_start = Instant::now();
    let mut scheduler = engine
        .create_reference_scheduler(
            &artifacts,
            scheduler_session_config.clone(),
            scheduler_config.clone(),
        )
        .map_err(|error| {
            eprintln!("failed to create reference scheduler: {}", error);
        })?;
    let scheduler_create_elapsed = scheduler_create_start.elapsed();

    let mut request_profiles = Vec::with_capacity(prepared_requests.len());
    let mut ticks = Vec::new();
    let mut next_request_index = 0usize;
    let mut tick_index = 0usize;

    while next_request_index < prepared_requests.len() || scheduler.has_pending() {
        while next_request_index < prepared_requests.len()
            && prepared_requests[next_request_index].arrival_tick < tick_index
            && !scheduler.has_pending()
        {
            tick_index = prepared_requests[next_request_index].arrival_tick;
        }

        let mut submitted_requests = 0usize;
        while next_request_index < prepared_requests.len()
            && prepared_requests[next_request_index].arrival_tick == tick_index
        {
            let request = &prepared_requests[next_request_index];
            let sequence_id = scheduler
                .submit(SequenceSubmitRequest {
                    prompt_token_ids: request.prompt_token_ids.clone(),
                    max_new_tokens,
                    stop_token_id,
                })
                .map_err(|error| {
                    eprintln!("failed to submit continuous batching request: {}", error);
                })?;
            request_profiles.push(ContinuousBatchRequestProfile {
                request_index: request.request_index,
                arrival_tick: request.arrival_tick,
                sequence_id,
                user_text: request.user_text.clone(),
                prompt_tokens: request.prompt_token_ids.len(),
                generated_tokens: Vec::new(),
                first_token_tick: None,
                finish_tick: None,
                finish_reason: None,
                assistant_text: String::new(),
            });
            next_request_index += 1;
            submitted_requests += 1;
        }

        let active_sequences = scheduler
            .sequence_states()
            .filter(|state| !state.is_finished())
            .count();
        let tick_start = Instant::now();
        let tick_report = scheduler.execute_next_tick().map_err(|error| {
            eprintln!("continuous batching scheduler tick failed: {}", error);
        })?;
        let tick_elapsed = tick_start.elapsed();

        let Some(tick_report) = tick_report else {
            if next_request_index < prepared_requests.len() {
                tick_index = prepared_requests[next_request_index].arrival_tick;
                continue;
            }
            break;
        };

        let reused_prompt_tokens = tick_report
            .updates
            .iter()
            .map(|update| update.reused_prompt_tokens)
            .sum::<usize>();
        let appended_prompt_tokens = tick_report
            .updates
            .iter()
            .map(|update| update.appended_prompt_tokens)
            .sum::<usize>();
        let generated_tokens = tick_report
            .updates
            .iter()
            .filter(|update| update.generated_token.is_some())
            .count();
        let finished_sequences = tick_report
            .updates
            .iter()
            .filter(|update| update.finish_reason.is_some())
            .count();

        for update in &tick_report.updates {
            let request = request_profiles
                .iter_mut()
                .find(|request| request.sequence_id == update.sequence_id)
                .ok_or_else(|| {
                    eprintln!(
                        "continuous batching request profile missing for sequence {}",
                        update.sequence_id.raw()
                    );
                })?;

            if let Some(sample) = update.generated_token {
                request.generated_tokens.push(sample);
                if request.first_token_tick.is_none() {
                    request.first_token_tick = Some(tick_index);
                }
            }

            if let Some(reason) = update.finish_reason {
                request.finish_reason = Some(reason);
                request.finish_tick = Some(tick_index);
            }
        }

        ticks.push(ContinuousBatchTickProfile {
            tick_index,
            submitted_requests,
            batch_kind: tick_report.tick.batch_kind(),
            batch_size: tick_report.tick.sequence_ids().len(),
            active_sequences,
            reused_prompt_tokens,
            appended_prompt_tokens,
            generated_tokens,
            finished_sequences,
            elapsed: tick_elapsed,
        });
        tick_index += 1;
    }

    for request in &mut request_profiles {
        request.assistant_text = artifacts
            .tokenizer
            .decode(
                &request
                    .generated_tokens
                    .iter()
                    .map(|sample| sample.token_id)
                    .collect::<Vec<_>>(),
            )
            .map_err(|error| {
                eprintln!("failed to decode continuous batching tokens: {}", error);
            })?;
    }

    Ok(ContinuousBatchProfileReport {
        requests: request_profiles,
        ticks,
        load_elapsed,
        scheduler_create_elapsed,
        total_elapsed: total_start.elapsed(),
        scheduler_config,
        kv_cache: scheduler_session_config.kv_cache,
    })
}

fn build_continuous_batch_requests(
    tokenizer: &VocabularyTokenizer,
    system_prompt: &str,
    _stop_token_id: Option<u32>,
) -> Result<Vec<PreparedContinuousBatchRequest>, ()> {
    let mut prepared = Vec::with_capacity(default_continuous_batch_requests().len());
    for (request_index, request) in default_continuous_batch_requests()
        .iter()
        .copied()
        .enumerate()
    {
        let rendered = render_default_chat_prompt(tokenizer, request.user_text, system_prompt)?;
        let prompt_token_ids = tokenizer.encode(&rendered, false).map_err(|error| {
            eprintln!("failed to tokenize continuous batching prompt: {}", error);
        })?;
        prepared.push(PreparedContinuousBatchRequest {
            request_index: request_index + 1,
            arrival_tick: request.arrival_tick,
            user_text: request.user_text.to_string(),
            prompt_token_ids,
        });
    }
    Ok(prepared)
}

fn default_continuous_batch_requests() -> &'static [ContinuousBatchRequestSpec] {
    &[
        ContinuousBatchRequestSpec {
            arrival_tick: 0,
            user_text: "你好",
        },
        ContinuousBatchRequestSpec {
            arrival_tick: 2,
            user_text: "请用一句话介绍你自己。",
        },
        ContinuousBatchRequestSpec {
            arrival_tick: 4,
            user_text: "列出三个关键词描述 Rust。",
        },
        ContinuousBatchRequestSpec {
            arrival_tick: 6,
            user_text: "把 FerrisInfer 的目标压缩成一句话。",
        },
    ]
}
fn run_scripted_chat_profile(
    engine: &InferenceEngine<CpuBackend>,
    max_new_tokens: usize,
    system_prompt_path: &str,
) -> Result<ScriptedChatProfileReport, ()> {
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
    let stop_config = resolve_chat_stop_config(&tokenizer)?;

    let load_start = Instant::now();
    let model = Arc::new(source.load_model().map_err(|error| {
        eprintln!("failed to load model artifacts: {}", error);
    })?);
    let artifacts = LoadedArtifacts {
        model,
        tokenizer: Arc::new(tokenizer.clone()) as Arc<dyn Tokenizer>,
    };
    let load_elapsed = load_start.elapsed();

    let session_create_start = Instant::now();
    let mut session_config = SessionConfig::default();
    session_config.sampler = SamplerConfig::greedy();
    let mut session = engine
        .create_session(&artifacts, session_config)
        .map_err(|error| {
            eprintln!("failed to create inference session: {}", error);
        })?;
    let session_create_elapsed = session_create_start.elapsed();

    let mut messages = vec![ChatMessage::system(system_prompt)];
    let mut turns = Vec::with_capacity(default_profile_chat_turns().len());

    for (turn_index, user_text) in default_profile_chat_turns().iter().copied().enumerate() {
        messages.push(ChatMessage::user(user_text));
        let (report, assistant_text) = run_chat_turn_profile(
            &tokenizer,
            &mut session,
            &mut messages,
            max_new_tokens,
            &stop_config,
        )?;
        turns.push(ScriptedChatTurnProfile {
            turn_index: turn_index + 1,
            user_text: user_text.to_string(),
            assistant_text,
            report,
        });
    }

    Ok(ScriptedChatProfileReport {
        turns,
        load_elapsed,
        session_create_elapsed,
        total_elapsed: total_start.elapsed(),
    })
}

fn run_chat_turn_profile(
    tokenizer: &VocabularyTokenizer,
    session: &mut Session,
    messages: &mut Vec<ChatMessage>,
    max_new_tokens: usize,
    stop_config: &ChatStopConfig,
) -> Result<(ChatTurnReport, String), ()> {
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
        collect_reference_tokens_with_fallback_stop(
            tokenizer,
            session,
            max_new_tokens,
            stop_config,
        )?;
    let decode_elapsed = decode_start.elapsed();

    messages.push(ChatMessage::assistant(assistant_text.clone()));

    Ok((
        ChatTurnReport {
            reused_prompt_tokens,
            appended_prompt_tokens,
            generated_tokens: generated_tokens.len(),
            finish_reason,
            render_elapsed,
            tokenize_elapsed,
            sync_elapsed,
            decode_elapsed,
        },
        assistant_text,
    ))
}

fn default_profile_chat_turns() -> &'static [&'static str] {
    &[
        "你好",
        "请用一句话介绍你自己。",
        "把上一句压缩成十个字以内。",
        "总结一下我们刚才这段对话的重点。",
    ]
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

fn warn_if_debug_profile() {
    if cfg!(debug_assertions) {
        eprintln!(
            "warning: ferrisinfer-cli is running in the debug profile; CPU inference is much slower here. Use `cargo run --release -p ferrisinfer-cli -- <command>` for meaningful timing."
        );
    }
}

fn throughput_per_second(tokens: usize, elapsed: std::time::Duration) -> f64 {
    let seconds = elapsed.as_secs_f64();
    if seconds <= f64::EPSILON {
        return 0.0;
    }
    tokens as f64 / seconds
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
    fn build_cuda_smoke_payload_is_nonzero_and_deterministic() {
        let payload = build_cuda_smoke_payload(8);
        assert_eq!(payload, vec![17, 48, 79, 110, 141, 172, 203, 234]);
    }

    #[test]
    fn build_cuda_smoke_tensor_payload_is_deterministic() {
        let payload = build_cuda_smoke_tensor_payload(6);
        assert_eq!(payload, vec![-1.5, -1.25, -1.0, -0.75, -0.5, -0.25]);
    }

    #[test]
    fn default_profile_chat_turns_are_nonempty() {
        assert!(!default_profile_chat_turns().is_empty());
        assert!(default_profile_chat_turns()
            .iter()
            .all(|turn| !turn.trim().is_empty()));
    }

    #[test]
    fn default_continuous_batch_requests_are_nonempty_and_sorted() {
        assert!(!default_continuous_batch_requests().is_empty());
        assert!(default_continuous_batch_requests()
            .iter()
            .all(|request| !request.user_text.trim().is_empty()));
        assert!(default_continuous_batch_requests()
            .windows(2)
            .all(|pair| pair[0].arrival_tick <= pair[1].arrival_tick));
    }

    #[test]
    fn take_valid_utf8_prefix_buffers_incomplete_multibyte_sequence() {
        let mut bytes = vec![0xE5, 0x8A];
        assert_eq!(take_valid_utf8_prefix(&mut bytes).unwrap(), "");
        assert_eq!(bytes, vec![0xE5, 0x8A]);
    }

    #[test]
    fn take_valid_utf8_prefix_flushes_completed_multibyte_sequence() {
        let mut bytes = vec![0xE5, 0x8A, 0xA9, b' ', b'A'];
        assert_eq!(take_valid_utf8_prefix(&mut bytes).unwrap(), "助 A");
        assert!(bytes.is_empty());
    }
}
