# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands
- Build: `cargo build`
- Run tests: `cargo test`
- Run specific test: `cargo test test_name`
- Lint: `cargo clippy`
- Format: `cargo fmt`
- Run example: `cargo run --example <name>` (e.g., `cargo run --example function-calling`)

## Architecture Overview

rsai is a Rust library for type-safe LLM completions with structured output. It prioritizes compile-time type safety over streaming/TTFT.

### Core Flow
```
llm::with(Provider) → LlmBuilder → .complete::<T>() → StructuredResponse<T>
```

### Module Structure
- **src/lib.rs**: Public API exports
- **src/core/**: Core abstractions
  - `builder.rs`: Type-state builder pattern (ProviderSet → ApiKeySet → Configuring → MessagesSet → ToolsSet)
  - `traits.rs`: `LlmProvider`, `CompletionTarget`, `ToolFunction` traits
  - `types.rs`: `Message`, `ToolRegistry`, `StructuredResponse`, `TextResponse`
  - `error.rs`: `LlmError` enum with `thiserror`
- **src/provider/**: Provider implementations (OpenAI, OpenRouter, Gemini)
  - Each provider implements `LlmProvider` trait and `ResponsesProviderConfig`
- **src/responses/**: HTTP client, request/response serialization, format handling
- **macros/**: Procedural macros crate
  - `#[completion_schema]`: Adds `Deserialize`, `JsonSchema`, `deny_unknown_fields`
  - `#[tool]`: Transforms functions into `ToolFunction` implementations
  - `toolset![]`: Creates `ToolSet` from tool functions

### Key Patterns
- **Type-state builder**: Enforces correct construction order at compile time via phantom types
- **Trait-based providers**: All providers implement `LlmProvider` async trait
- **CompletionTarget**: Defines how response types are parsed (blanket impl for `DeserializeOwned + JsonSchema`)

## Code Style
- Prefer functional style (map/filter/fold over loops)
- Use `LlmError` enum for errors (not `AIError`)
- Direct module files (e.g., `openai.rs`) not `mod.rs` pattern
- Use `async_trait` for async trait methods
- Doc comments only for public APIs; inline comments only for non-obvious behavior

## Provider Implementation
When adding a new provider:
1. Create `src/provider/<name>.rs` implementing `LlmProvider` and `ResponsesProviderConfig`
2. Add to `Provider` enum in `src/provider/mod.rs`
3. Add API constants in `src/provider/constants.rs`
4. Add builder integration in `src/core/builder.rs` match arms
5. Add example in `examples/` directory