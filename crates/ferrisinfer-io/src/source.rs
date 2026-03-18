use ferrisinfer_core::Result;
use ferrisinfer_model::{DecoderOnlyModel, ModelConfig, WeightMap};

use crate::tokenizer::TokenizerAsset;

pub trait ModelSource {
    fn format_name(&self) -> &'static str;
    fn load_config(&mut self) -> Result<ModelConfig>;
    fn load_tokenizer(&mut self) -> Result<TokenizerAsset>;
    fn load_weights(&mut self) -> Result<WeightMap>;

    fn load_model(&mut self) -> Result<DecoderOnlyModel> {
        let config = self.load_config()?;
        let weights = self.load_weights()?;
        DecoderOnlyModel::new(config, weights)
    }
}
