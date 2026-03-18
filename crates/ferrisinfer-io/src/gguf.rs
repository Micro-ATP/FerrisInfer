use std::path::{Path, PathBuf};

use ferrisinfer_core::{FerrisError, Result};
use ferrisinfer_model::{ModelConfig, WeightMap};

use crate::source::ModelSource;
use crate::tokenizer::TokenizerAsset;

#[derive(Debug, Clone)]
pub struct GgufSource {
    path: PathBuf,
}

impl GgufSource {
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl ModelSource for GgufSource {
    fn format_name(&self) -> &'static str {
        "gguf"
    }

    fn load_config(&mut self) -> Result<ModelConfig> {
        Err(FerrisError::unsupported(
            "GGUF config loading is not implemented yet",
        ))
    }

    fn load_tokenizer(&mut self) -> Result<TokenizerAsset> {
        Err(FerrisError::unsupported(
            "GGUF tokenizer loading is not implemented yet",
        ))
    }

    fn load_weights(&mut self) -> Result<WeightMap> {
        Err(FerrisError::unsupported(
            "GGUF weight loading is not implemented yet",
        ))
    }
}
