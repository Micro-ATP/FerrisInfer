use std::sync::Arc;

use ferrisinfer_core::Result;
use ferrisinfer_io::{ModelSource, Tokenizer, VocabularyTokenizer};
use ferrisinfer_kernel::Backend;
use ferrisinfer_model::DecoderOnlyModel;

use crate::plan::{ExecutionMode, ExecutionPlan};
use crate::scheduler::{ReferenceScheduler, SchedulerConfig};
use crate::session::{Session, SessionConfig};

pub struct LoadedArtifacts {
    pub model: Arc<DecoderOnlyModel>,
    pub tokenizer: Arc<dyn Tokenizer>,
}

pub struct InferenceEngine<B: Backend> {
    backend: B,
}

impl<B: Backend> InferenceEngine<B> {
    pub fn new(backend: B) -> Self {
        Self { backend }
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    pub fn build_plan(&self, mode: ExecutionMode) -> ExecutionPlan {
        match mode {
            ExecutionMode::Prefill => ExecutionPlan::prefill(self.backend.device()),
            ExecutionMode::Decode => ExecutionPlan::decode(self.backend.device()),
        }
    }

    pub fn load_from_source<S: ModelSource>(&self, source: &mut S) -> Result<LoadedArtifacts> {
        let model = Arc::new(source.load_model()?);
        let tokenizer =
            Arc::new(VocabularyTokenizer::new(source.load_tokenizer()?)) as Arc<dyn Tokenizer>;

        Ok(LoadedArtifacts { model, tokenizer })
    }

    pub fn create_session(
        &self,
        artifacts: &LoadedArtifacts,
        config: SessionConfig,
    ) -> Result<Session> {
        Session::new(
            Arc::clone(&artifacts.model),
            Arc::clone(&artifacts.tokenizer),
            config,
        )
    }

    pub fn create_reference_scheduler(
        &self,
        artifacts: &LoadedArtifacts,
        session_config: SessionConfig,
        scheduler_config: SchedulerConfig,
    ) -> Result<ReferenceScheduler> {
        ReferenceScheduler::new(
            Arc::clone(&artifacts.model),
            Arc::clone(&artifacts.tokenizer),
            session_config,
            scheduler_config,
        )
    }
}
