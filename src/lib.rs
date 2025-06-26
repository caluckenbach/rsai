pub mod core;
pub mod provider;

pub use core::builder::ApiKey;
pub use core::{
    builder::llm,
    types::{ChatRole, Message},
};
pub use rsai_macros::completion_schema;
