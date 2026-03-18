use ferrisinfer_core::{FerrisError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerKind {
    BytePair,
    SentencePiece,
}

#[derive(Debug, Clone)]
pub struct TokenizerAsset {
    pub kind: TokenizerKind,
    pub vocab_size: usize,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub unk_token_id: Option<u32>,
}

pub trait Tokenizer: Send + Sync {
    fn kind(&self) -> TokenizerKind;
    fn vocab_size(&self) -> usize;
    fn encode(&self, text: &str, add_bos: bool) -> Result<Vec<u32>>;
    fn decode(&self, tokens: &[u32]) -> Result<String>;
}

#[derive(Debug, Clone)]
pub struct VocabularyTokenizer {
    asset: TokenizerAsset,
}

impl VocabularyTokenizer {
    pub fn new(asset: TokenizerAsset) -> Self {
        Self { asset }
    }

    pub fn asset(&self) -> &TokenizerAsset {
        &self.asset
    }
}

impl Tokenizer for VocabularyTokenizer {
    fn kind(&self) -> TokenizerKind {
        self.asset.kind
    }

    fn vocab_size(&self) -> usize {
        self.asset.vocab_size
    }

    fn encode(&self, _text: &str, _add_bos: bool) -> Result<Vec<u32>> {
        Err(FerrisError::unsupported(
            "tokenizer encode is not implemented yet",
        ))
    }

    fn decode(&self, _tokens: &[u32]) -> Result<String> {
        Err(FerrisError::unsupported(
            "tokenizer decode is not implemented yet",
        ))
    }
}
