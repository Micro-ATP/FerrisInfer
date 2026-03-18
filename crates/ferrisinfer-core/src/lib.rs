pub mod device;
pub mod dtype;
pub mod error;
pub mod layout;
pub mod shape;
pub mod tensor;

pub use device::{DeviceKind, ExecutionConfig};
pub use dtype::DType;
pub use error::{ErrorKind, FerrisError, Result};
pub use layout::Layout;
pub use shape::Shape;
pub use tensor::{Storage, Tensor};
