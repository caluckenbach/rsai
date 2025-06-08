pub mod core;
pub mod provider;

pub use rsai_macros::completion_schema;
pub use core::builder::ApiKey;
pub use core::{
    builder::llm,
    types::{ChatRole, Message},
};
