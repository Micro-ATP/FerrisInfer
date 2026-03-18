use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use ferrisinfer_core::{ErrorKind, FerrisError, Result};
use ferrisinfer_model::{
    ActivationKind, ArchitectureKind, AttentionLayout, AttentionSpec, MlpSpec, ModelConfig,
    NormKind, NormSpec, RopeScalingKind, RopeSpec, WeightMap,
};

use crate::json::{parse_json, JsonValue};
use crate::safetensors::SafeTensorsRepository;
use crate::source::ModelSource;
use crate::tokenizer::{
    BytePairTokenizerModel, TokenizerAsset, TokenizerKind, TokenizerModelAsset,
};

#[derive(Debug, Clone)]
pub struct HfSource {
    path: PathBuf,
}

#[derive(Debug, Clone)]
pub struct HfInspection {
    pub family: String,
    pub architecture_name: String,
    pub config: ModelConfig,
    pub tokenizer: TokenizerAsset,
    pub shard_count: usize,
    pub tensor_count: usize,
    pub total_data_bytes: u64,
    pub mapped_required_tensors: usize,
    pub mapped_optional_tensors: usize,
    pub missing_required_tensors: Vec<String>,
    pub missing_optional_tensors: Vec<String>,
    pub shape_mismatches: Vec<String>,
}

impl HfInspection {
    pub fn is_compatible(&self) -> bool {
        self.missing_required_tensors.is_empty() && self.shape_mismatches.is_empty()
    }
}

#[derive(Debug, Clone)]
struct TensorPlan {
    external_name: String,
    internal_name: String,
    expected_source_shape: Vec<usize>,
    transpose_2d: bool,
    required: bool,
}

impl HfSource {
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn inspect(&self) -> Result<HfInspection> {
        let config_root = self.read_config_root()?;
        let family = read_model_family(&config_root)?;
        let architecture_name = read_architecture_name(&config_root)?;
        let config = parse_model_config(&config_root)?;
        let tokenizer = self.read_tokenizer_asset(&config_root, &family)?;
        let repository = SafeTensorsRepository::open(&self.path)?;
        let plans = tensor_plans_for_family(&family, &config)?;

        let mut mapped_required_tensors = 0usize;
        let mut mapped_optional_tensors = 0usize;
        let mut missing_required_tensors = Vec::new();
        let mut missing_optional_tensors = Vec::new();
        let mut shape_mismatches = Vec::new();

        for plan in &plans {
            match repository.get(&plan.external_name) {
                Some(entry) => {
                    if entry.shape() != plan.expected_source_shape.as_slice() {
                        shape_mismatches.push(format!(
                            "{} expected {:?} but found {:?}",
                            plan.external_name,
                            plan.expected_source_shape,
                            entry.shape()
                        ));
                    }

                    if plan.required {
                        mapped_required_tensors += 1;
                    } else {
                        mapped_optional_tensors += 1;
                    }
                }
                None if plan.required => missing_required_tensors.push(plan.external_name.clone()),
                None => missing_optional_tensors.push(plan.external_name.clone()),
            }
        }

        Ok(HfInspection {
            family,
            architecture_name,
            config,
            tokenizer,
            shard_count: repository.shard_count(),
            tensor_count: repository.tensor_count(),
            total_data_bytes: repository.total_data_bytes(),
            mapped_required_tensors,
            mapped_optional_tensors,
            missing_required_tensors,
            missing_optional_tensors,
            shape_mismatches,
        })
    }

    fn read_config_root(&self) -> Result<JsonValue> {
        let config_path = self.path.join("config.json");
        let text = fs::read_to_string(&config_path)?;
        parse_json(&text)
    }

    fn read_tokenizer_asset(
        &self,
        config_root: &JsonValue,
        family: &str,
    ) -> Result<TokenizerAsset> {
        let tokenizer_json = self.path.join("tokenizer.json");
        let vocab_json = self.path.join("vocab.json");
        let merges_txt = self.path.join("merges.txt");
        if !tokenizer_json.is_file() && !(vocab_json.is_file() && merges_txt.is_file()) {
            return Err(FerrisError::new(
                ErrorKind::Io,
                format!("no tokenizer assets found under {}", self.path.display()),
            ));
        }

        let tokenizer_config_root = self.read_tokenizer_config_root()?;

        match family {
            "qwen2" => self.load_qwen2_tokenizer_asset(config_root, &tokenizer_config_root),
            other => Err(FerrisError::unsupported(format!(
                "tokenizer loading for model family '{other}' is not implemented yet"
            ))),
        }
    }

    fn read_tokenizer_config_root(&self) -> Result<JsonValue> {
        let path = self.path.join("tokenizer_config.json");
        if !path.is_file() {
            return parse_json("{}");
        }

        let text = fs::read_to_string(path)?;
        parse_json(&text)
    }

    fn load_qwen2_tokenizer_asset(
        &self,
        config_root: &JsonValue,
        tokenizer_config_root: &JsonValue,
    ) -> Result<TokenizerAsset> {
        let token_to_id = load_vocab_json(&self.path.join("vocab.json"))?;
        let merge_ranks = load_merges_txt(&self.path.join("merges.txt"))?;
        let added_token_to_id = load_added_tokens(tokenizer_config_root)?;

        let asset = TokenizerAsset::new(
            TokenizerKind::BytePair,
            required_usize(config_root, "vocab_size")?,
            optional_u32(config_root, "bos_token_id")?,
            optional_u32(config_root, "eos_token_id")?,
            optional_u32(tokenizer_config_root, "unk_token_id")?,
        )
        .with_chat_template(optional_string_owned(
            tokenizer_config_root,
            "chat_template",
        )?)
        .with_model(TokenizerModelAsset::BytePair(BytePairTokenizerModel::new(
            token_to_id,
            merge_ranks,
            added_token_to_id,
            false,
        )));

        Ok(asset)
    }
}

impl ModelSource for HfSource {
    fn format_name(&self) -> &'static str {
        "huggingface"
    }

    fn load_config(&mut self) -> Result<ModelConfig> {
        let config_root = self.read_config_root()?;
        parse_model_config(&config_root)
    }

    fn load_tokenizer(&mut self) -> Result<TokenizerAsset> {
        let config_root = self.read_config_root()?;
        let family = read_model_family(&config_root)?;
        self.read_tokenizer_asset(&config_root, &family)
    }

    fn load_weights(&mut self) -> Result<WeightMap> {
        let config_root = self.read_config_root()?;
        let family = read_model_family(&config_root)?;
        let config = parse_model_config(&config_root)?;
        let repository = SafeTensorsRepository::open(&self.path)?;
        let plans = tensor_plans_for_family(&family, &config)?;

        let mut weights = WeightMap::new();
        for plan in plans {
            let Some(entry) = repository.get(&plan.external_name) else {
                if plan.required {
                    return Err(FerrisError::new(
                        ErrorKind::MissingWeight,
                        format!("missing required tensor '{}'", plan.external_name),
                    ));
                }
                continue;
            };

            if entry.shape() != plan.expected_source_shape.as_slice() {
                return Err(FerrisError::new(
                    ErrorKind::InvalidShape,
                    format!(
                        "tensor '{}' expected shape {:?} but found {:?}",
                        plan.external_name,
                        plan.expected_source_shape,
                        entry.shape()
                    ),
                ));
            }

            let tensor = repository.load_tensor_f32(&plan.external_name, plan.transpose_2d)?;
            weights.insert(plan.internal_name, tensor);
        }

        Ok(weights)
    }
}

fn parse_model_config(root: &JsonValue) -> Result<ModelConfig> {
    match read_model_family(root)?.as_str() {
        "qwen2" => parse_qwen2_config(root),
        other => Err(FerrisError::unsupported(format!(
            "model family '{other}' is not supported yet"
        ))),
    }
}

fn parse_qwen2_config(root: &JsonValue) -> Result<ModelConfig> {
    let hidden_size = required_usize(root, "hidden_size")?;
    let intermediate_size = required_usize(root, "intermediate_size")?;
    let num_attention_heads = required_usize(root, "num_attention_heads")?;
    let head_dim = hidden_size
        .checked_div(num_attention_heads)
        .ok_or_else(|| {
            FerrisError::new(
                ErrorKind::InvalidConfig,
                "hidden_size must be divisible by num_attention_heads",
            )
        })?;
    if hidden_size % num_attention_heads != 0 {
        return Err(FerrisError::new(
            ErrorKind::InvalidConfig,
            "hidden_size must be divisible by num_attention_heads",
        ));
    }

    let partial_rotary_factor = optional_f32(root, "partial_rotary_factor")?.unwrap_or(1.0);
    let rotary_dims = ((head_dim as f32) * partial_rotary_factor).round() as usize;

    Ok(ModelConfig {
        architecture: ArchitectureKind::Qwen2,
        hidden_size,
        intermediate_size,
        num_layers: required_usize(root, "num_hidden_layers")?,
        num_attention_heads,
        num_key_value_heads: optional_usize(root, "num_key_value_heads")?
            .unwrap_or(num_attention_heads),
        vocab_size: required_usize(root, "vocab_size")?,
        max_position_embeddings: required_usize(root, "max_position_embeddings")?,
        norm: NormSpec {
            kind: NormKind::RmsNorm,
            epsilon: required_f32(root, "rms_norm_eps")?,
        },
        rope: RopeSpec {
            theta: optional_f32(root, "rope_theta")?.unwrap_or(10000.0),
            scaling: RopeScalingKind::None,
            scaling_factor: 1.0,
            rotary_dims,
        },
        attention: AttentionSpec {
            layout: AttentionLayout::SeparateQkv,
            causal: true,
            use_qk_norm: false,
            head_dim,
        },
        mlp: MlpSpec {
            hidden_act: parse_activation(required_string(root, "hidden_act")?)?,
            gated: true,
        },
        tie_word_embeddings: optional_bool(root, "tie_word_embeddings")?.unwrap_or(false),
    })
}

fn tensor_plans_for_family(family: &str, config: &ModelConfig) -> Result<Vec<TensorPlan>> {
    match family {
        "qwen2" => Ok(qwen2_tensor_plans(config)),
        other => Err(FerrisError::unsupported(format!(
            "tensor mapping for model family '{other}' is not implemented yet"
        ))),
    }
}

fn qwen2_tensor_plans(config: &ModelConfig) -> Vec<TensorPlan> {
    let hidden = config.hidden_size;
    let intermediate = config.intermediate_size;
    let kv_hidden = config.num_key_value_heads * config.head_dim();
    let mut plans = Vec::new();

    plans.push(TensorPlan {
        external_name: "model.embed_tokens.weight".to_string(),
        internal_name: "tok_embeddings.weight".to_string(),
        expected_source_shape: vec![config.vocab_size, hidden],
        transpose_2d: false,
        required: true,
    });

    for layer in 0..config.num_layers {
        plans.push(TensorPlan {
            external_name: format!("model.layers.{layer}.input_layernorm.weight"),
            internal_name: format!("layers.{layer}.attention_norm.weight"),
            expected_source_shape: vec![hidden],
            transpose_2d: false,
            required: true,
        });
        plans.push(TensorPlan {
            external_name: format!("model.layers.{layer}.self_attn.q_proj.weight"),
            internal_name: format!("layers.{layer}.attention.wq.weight"),
            expected_source_shape: vec![hidden, hidden],
            transpose_2d: true,
            required: true,
        });
        plans.push(TensorPlan {
            external_name: format!("model.layers.{layer}.self_attn.k_proj.weight"),
            internal_name: format!("layers.{layer}.attention.wk.weight"),
            expected_source_shape: vec![kv_hidden, hidden],
            transpose_2d: true,
            required: true,
        });
        plans.push(TensorPlan {
            external_name: format!("model.layers.{layer}.self_attn.v_proj.weight"),
            internal_name: format!("layers.{layer}.attention.wv.weight"),
            expected_source_shape: vec![kv_hidden, hidden],
            transpose_2d: true,
            required: true,
        });
        plans.push(TensorPlan {
            external_name: format!("model.layers.{layer}.self_attn.o_proj.weight"),
            internal_name: format!("layers.{layer}.attention.wo.weight"),
            expected_source_shape: vec![hidden, hidden],
            transpose_2d: true,
            required: true,
        });
        plans.push(TensorPlan {
            external_name: format!("model.layers.{layer}.post_attention_layernorm.weight"),
            internal_name: format!("layers.{layer}.ffn_norm.weight"),
            expected_source_shape: vec![hidden],
            transpose_2d: false,
            required: true,
        });
        plans.push(TensorPlan {
            external_name: format!("model.layers.{layer}.mlp.up_proj.weight"),
            internal_name: format!("layers.{layer}.feed_forward.w1.weight"),
            expected_source_shape: vec![intermediate, hidden],
            transpose_2d: true,
            required: true,
        });
        plans.push(TensorPlan {
            external_name: format!("model.layers.{layer}.mlp.down_proj.weight"),
            internal_name: format!("layers.{layer}.feed_forward.w2.weight"),
            expected_source_shape: vec![hidden, intermediate],
            transpose_2d: true,
            required: true,
        });
        plans.push(TensorPlan {
            external_name: format!("model.layers.{layer}.mlp.gate_proj.weight"),
            internal_name: format!("layers.{layer}.feed_forward.w3.weight"),
            expected_source_shape: vec![intermediate, hidden],
            transpose_2d: true,
            required: true,
        });
        plans.push(TensorPlan {
            external_name: format!("model.layers.{layer}.self_attn.q_proj.bias"),
            internal_name: format!("layers.{layer}.attention.wq.bias"),
            expected_source_shape: vec![hidden],
            transpose_2d: false,
            required: false,
        });
        plans.push(TensorPlan {
            external_name: format!("model.layers.{layer}.self_attn.k_proj.bias"),
            internal_name: format!("layers.{layer}.attention.wk.bias"),
            expected_source_shape: vec![kv_hidden],
            transpose_2d: false,
            required: false,
        });
        plans.push(TensorPlan {
            external_name: format!("model.layers.{layer}.self_attn.v_proj.bias"),
            internal_name: format!("layers.{layer}.attention.wv.bias"),
            expected_source_shape: vec![kv_hidden],
            transpose_2d: false,
            required: false,
        });
    }

    plans.push(TensorPlan {
        external_name: "model.norm.weight".to_string(),
        internal_name: "norm.weight".to_string(),
        expected_source_shape: vec![hidden],
        transpose_2d: false,
        required: true,
    });

    if !config.tie_word_embeddings {
        plans.push(TensorPlan {
            external_name: "lm_head.weight".to_string(),
            internal_name: "output.weight".to_string(),
            expected_source_shape: vec![config.vocab_size, hidden],
            transpose_2d: false,
            required: true,
        });
    }

    plans
}

fn read_model_family(root: &JsonValue) -> Result<String> {
    let model_type = required_string(root, "model_type")?;
    match model_type {
        "qwen2" => Ok(model_type.to_string()),
        other => Err(FerrisError::unsupported(format!(
            "model_type '{other}' is not supported yet"
        ))),
    }
}

fn read_architecture_name(root: &JsonValue) -> Result<String> {
    let architectures = root.get("architectures")?.as_array()?;
    let first = architectures.first().ok_or_else(|| {
        FerrisError::new(
            ErrorKind::Parse,
            "config architectures array must contain at least one value",
        )
    })?;
    Ok(first.as_str()?.to_string())
}

fn parse_activation(value: &str) -> Result<ActivationKind> {
    match value {
        "silu" | "swish" => Ok(ActivationKind::Silu),
        "gelu" => Ok(ActivationKind::Gelu),
        "relu" => Ok(ActivationKind::Relu),
        other => Err(FerrisError::unsupported(format!(
            "activation '{other}' is not supported yet"
        ))),
    }
}

fn load_vocab_json(path: &Path) -> Result<HashMap<String, u32>> {
    let text = fs::read_to_string(path)?;
    let root = parse_json(&text)?;
    let object = root.as_object()?;

    let mut token_to_id = HashMap::with_capacity(object.len());
    for (token, value) in object {
        let id = value.as_number()?.as_u64()?;
        let id = u32::try_from(id).map_err(|_| {
            FerrisError::new(
                ErrorKind::Runtime,
                format!("token id for '{token}' does not fit into u32"),
            )
        })?;
        token_to_id.insert(token.clone(), id);
    }

    Ok(token_to_id)
}

fn load_merges_txt(path: &Path) -> Result<HashMap<(String, String), usize>> {
    let text = fs::read_to_string(path)?;
    let mut merge_ranks = HashMap::new();

    for (rank, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let mut parts = trimmed.split(' ');
        let left = parts.next().ok_or_else(|| {
            FerrisError::new(ErrorKind::Parse, "merge line is missing left token")
        })?;
        let right = parts.next().ok_or_else(|| {
            FerrisError::new(ErrorKind::Parse, "merge line is missing right token")
        })?;
        if parts.next().is_some() {
            return Err(FerrisError::new(
                ErrorKind::Parse,
                format!("merge line '{trimmed}' contains too many tokens"),
            ));
        }

        merge_ranks.insert((left.to_string(), right.to_string()), rank);
    }

    Ok(merge_ranks)
}

fn load_added_tokens(root: &JsonValue) -> Result<HashMap<String, u32>> {
    let Some(added_tokens) = root.as_object()?.get("added_tokens_decoder") else {
        return Ok(HashMap::new());
    };

    let mut token_to_id = HashMap::new();
    for (id_string, token_value) in added_tokens.as_object()? {
        let id = id_string.parse::<u32>().map_err(|_| {
            FerrisError::new(
                ErrorKind::Parse,
                format!("added token id '{id_string}' is not a valid u32"),
            )
        })?;
        let content = token_value.get("content")?.as_str()?.to_string();
        token_to_id.insert(content, id);
    }

    Ok(token_to_id)
}

fn required_string<'a>(root: &'a JsonValue, key: &str) -> Result<&'a str> {
    root.get(key)?.as_str()
}

fn required_usize(root: &JsonValue, key: &str) -> Result<usize> {
    let value = root.get(key)?.as_number()?.as_u64()?;
    usize::try_from(value).map_err(|_| {
        FerrisError::new(
            ErrorKind::Runtime,
            format!("config key '{key}' does not fit into usize"),
        )
    })
}

fn required_f32(root: &JsonValue, key: &str) -> Result<f32> {
    let value = root.get(key)?.as_number()?.as_f64()?;
    if !value.is_finite() {
        return Err(FerrisError::new(
            ErrorKind::Parse,
            format!("config key '{key}' must be finite"),
        ));
    }
    Ok(value as f32)
}

fn optional_usize(root: &JsonValue, key: &str) -> Result<Option<usize>> {
    match root.as_object()?.get(key) {
        Some(JsonValue::Null) | None => Ok(None),
        Some(value) => {
            let number = value.as_number()?.as_u64()?;
            Ok(Some(usize::try_from(number).map_err(|_| {
                FerrisError::new(
                    ErrorKind::Runtime,
                    format!("config key '{key}' does not fit into usize"),
                )
            })?))
        }
    }
}

fn optional_u32(root: &JsonValue, key: &str) -> Result<Option<u32>> {
    match root.as_object()?.get(key) {
        Some(JsonValue::Null) | None => Ok(None),
        Some(value) => {
            let number = value.as_number()?.as_u64()?;
            Ok(Some(u32::try_from(number).map_err(|_| {
                FerrisError::new(
                    ErrorKind::Runtime,
                    format!("config key '{key}' does not fit into u32"),
                )
            })?))
        }
    }
}

fn optional_f32(root: &JsonValue, key: &str) -> Result<Option<f32>> {
    match root.as_object()?.get(key) {
        Some(JsonValue::Null) | None => Ok(None),
        Some(value) => {
            let number = value.as_number()?.as_f64()?;
            if !number.is_finite() {
                return Err(FerrisError::new(
                    ErrorKind::Parse,
                    format!("config key '{key}' must be finite"),
                ));
            }
            Ok(Some(number as f32))
        }
    }
}

fn optional_bool(root: &JsonValue, key: &str) -> Result<Option<bool>> {
    match root.as_object()?.get(key) {
        Some(JsonValue::Null) | None => Ok(None),
        Some(value) => Ok(Some(value.as_bool()?)),
    }
}

fn optional_string_owned(root: &JsonValue, key: &str) -> Result<Option<String>> {
    match root.as_object()?.get(key) {
        Some(JsonValue::Null) | None => Ok(None),
        Some(value) => Ok(Some(value.as_str()?.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    use ferrisinfer_model::DecoderOnlyModel;

    use super::*;

    static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

    #[test]
    fn hf_source_parses_qwen2_config_and_loads_weights() {
        let temp = TestDir::new();
        write_test_qwen2_model(temp.path()).unwrap();

        let mut source = HfSource::new(temp.path());
        let inspection = source.inspect().unwrap();
        let config = source.load_config().unwrap();
        let tokenizer = source.load_tokenizer().unwrap();
        let weights = source.load_weights().unwrap();
        let model = DecoderOnlyModel::new(config.clone(), weights).unwrap();

        assert_eq!(inspection.family, "qwen2");
        assert!(inspection.is_compatible());
        assert_eq!(config.hidden_size, 4);
        assert_eq!(config.num_attention_heads, 2);
        assert_eq!(config.num_key_value_heads, 1);
        assert_eq!(tokenizer.vocab_size, 16);
        assert_eq!(tokenizer.kind, TokenizerKind::BytePair);
        assert_eq!(
            model
                .weights()
                .get("layers.0.attention.wk.weight")
                .unwrap()
                .shape()
                .dims(),
            &[4, 2]
        );
        assert_eq!(
            model
                .weights()
                .get("layers.0.attention.wq.bias")
                .unwrap()
                .shape()
                .dims(),
            &[4]
        );
    }

    fn write_test_qwen2_model(dir: &Path) -> Result<()> {
        let config = r#"{
            "architectures": ["Qwen2ForCausalLM"],
            "model_type": "qwen2",
            "hidden_act": "silu",
            "hidden_size": 4,
            "intermediate_size": 8,
            "max_position_embeddings": 128,
            "num_attention_heads": 2,
            "num_hidden_layers": 1,
            "num_key_value_heads": 1,
            "rms_norm_eps": 0.000001,
            "rope_theta": 1000000.0,
            "tie_word_embeddings": true,
            "vocab_size": 16,
            "bos_token_id": 1,
            "eos_token_id": 2
        }"#;
        fs::write(dir.join("config.json"), config)?;
        fs::write(dir.join("tokenizer.json"), "{}")?;
        fs::write(dir.join("tokenizer_config.json"), "{}")?;
        fs::write(dir.join("vocab.json"), "{}")?;
        fs::write(dir.join("merges.txt"), "")?;

        write_safetensors_file(
            &dir.join("model.safetensors"),
            &[
                tensor("model.embed_tokens.weight", &[16, 4], seq_f32(0.0, 64)),
                tensor(
                    "model.layers.0.input_layernorm.weight",
                    &[4],
                    seq_f32(100.0, 4),
                ),
                tensor(
                    "model.layers.0.self_attn.q_proj.weight",
                    &[4, 4],
                    seq_f32(200.0, 16),
                ),
                tensor(
                    "model.layers.0.self_attn.k_proj.weight",
                    &[2, 4],
                    seq_f32(300.0, 8),
                ),
                tensor(
                    "model.layers.0.self_attn.v_proj.weight",
                    &[2, 4],
                    seq_f32(400.0, 8),
                ),
                tensor(
                    "model.layers.0.self_attn.o_proj.weight",
                    &[4, 4],
                    seq_f32(500.0, 16),
                ),
                tensor(
                    "model.layers.0.post_attention_layernorm.weight",
                    &[4],
                    seq_f32(600.0, 4),
                ),
                tensor(
                    "model.layers.0.mlp.up_proj.weight",
                    &[8, 4],
                    seq_f32(700.0, 32),
                ),
                tensor(
                    "model.layers.0.mlp.down_proj.weight",
                    &[4, 8],
                    seq_f32(800.0, 32),
                ),
                tensor(
                    "model.layers.0.mlp.gate_proj.weight",
                    &[8, 4],
                    seq_f32(900.0, 32),
                ),
                tensor(
                    "model.layers.0.self_attn.q_proj.bias",
                    &[4],
                    seq_f32(1000.0, 4),
                ),
                tensor(
                    "model.layers.0.self_attn.k_proj.bias",
                    &[2],
                    seq_f32(1100.0, 2),
                ),
                tensor(
                    "model.layers.0.self_attn.v_proj.bias",
                    &[2],
                    seq_f32(1200.0, 2),
                ),
                tensor("model.norm.weight", &[4], seq_f32(1300.0, 4)),
            ],
        )
    }

    #[derive(Clone)]
    struct TensorSpec {
        name: &'static str,
        shape: Vec<usize>,
        bytes: Vec<u8>,
    }

    fn tensor(name: &'static str, shape: &[usize], values: Vec<f32>) -> TensorSpec {
        TensorSpec {
            name,
            shape: shape.to_vec(),
            bytes: f32_bytes(&values),
        }
    }

    fn seq_f32(start: f32, count: usize) -> Vec<f32> {
        (0..count).map(|index| start + index as f32).collect()
    }

    fn write_safetensors_file(path: &Path, tensors: &[TensorSpec]) -> Result<()> {
        let mut data_offset = 0u64;
        let mut header = String::from("{\"__metadata__\":{\"format\":\"pt\"}");
        let mut data = Vec::new();

        for tensor in tensors {
            let start = data_offset;
            let end = start
                .checked_add(u64::try_from(tensor.bytes.len()).map_err(|_| {
                    FerrisError::new(
                        ErrorKind::Runtime,
                        "test tensor byte length does not fit into u64",
                    )
                })?)
                .ok_or_else(|| {
                    FerrisError::new(ErrorKind::Runtime, "test tensor offset overflow")
                })?;
            data_offset = end;

            let shape = tensor
                .shape
                .iter()
                .map(|dimension| dimension.to_string())
                .collect::<Vec<_>>()
                .join(",");
            header.push_str(&format!(
                ",\"{}\":{{\"dtype\":\"F32\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
                tensor.name, shape, start, end
            ));
            data.extend_from_slice(&tensor.bytes);
        }

        header.push('}');
        let header_bytes = header.into_bytes();
        let mut file_bytes = Vec::new();
        file_bytes.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        file_bytes.extend_from_slice(&header_bytes);
        file_bytes.extend_from_slice(&data);
        fs::write(path, file_bytes)?;
        Ok(())
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    struct TestDir {
        path: PathBuf,
    }

    impl TestDir {
        fn new() -> Self {
            let unique = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "ferrisinfer-hf-test-{}-{}-{}",
                std::process::id(),
                timestamp,
                unique
            ));
            fs::create_dir_all(&path).unwrap();
            Self { path }
        }

        fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}
