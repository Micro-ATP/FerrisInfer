use std::path::{Path, PathBuf};

use ferrisinfer_core::{FerrisError, Result};
use ferrisinfer_model::{ModelConfig, WeightMap};

use crate::source::ModelSource;
use crate::tokenizer::TokenizerAsset;

#[derive(Debug, Clone)]
pub struct FerrisSource {
    path: PathBuf,
}

impl FerrisSource {
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl ModelSource for FerrisSource {
    fn format_name(&self) -> &'static str {
        "ferris"
    }

    fn load_config(&mut self) -> Result<ModelConfig> {
        Err(FerrisError::unsupported(
            "Ferris model config loading is not implemented yet",
        ))
    }

    fn load_tokenizer(&mut self) -> Result<TokenizerAsset> {
        Err(FerrisError::unsupported(
            "Ferris tokenizer loading is not implemented yet",
        ))
    }

    fn load_weights(&mut self) -> Result<WeightMap> {
        Err(FerrisError::unsupported(
            "Ferris weight loading is not implemented yet",
        ))
    }
}
