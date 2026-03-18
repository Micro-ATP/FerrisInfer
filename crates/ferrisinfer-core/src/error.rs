use std::fmt::{Display, Formatter};

pub type Result<T> = std::result::Result<T, FerrisError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    Io,
    Parse,
    InvalidShape,
    InvalidLayout,
    InvalidType,
    InvalidConfig,
    MissingWeight,
    Unsupported,
    Runtime,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FerrisError {
    kind: ErrorKind,
    message: String,
}

impl FerrisError {
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    pub fn unsupported(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Unsupported, message)
    }

    pub fn kind(&self) -> ErrorKind {
        self.kind
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl Display for FerrisError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)
    }
}

impl std::error::Error for FerrisError {}

impl From<std::io::Error> for FerrisError {
    fn from(error: std::io::Error) -> Self {
        Self::new(ErrorKind::Io, error.to_string())
    }
}
