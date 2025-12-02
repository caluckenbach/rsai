//! Generic completion API abstraction for providers that don't use OpenAI's responses API.
//!
//! This module provides infrastructure for completion-style APIs like Google Gemini.

pub mod client;

pub use client::{
    CompletionClient, CompletionProviderConfig, CompletionRequestBuilder, ConversationItem,
};
