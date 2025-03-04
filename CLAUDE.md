# Rust Agent Development Guide

## Build & Test Commands
- Build: `cargo build`
- Run: `cargo run`
- Release build: `cargo build --release`
- Run all tests: `cargo test`
- Run specific test: `cargo test test_name`
- Code linting: `cargo clippy`
- Code formatting: `cargo fmt`

## Code Style Guidelines
- **Paradigm**: Prefer functional style when available (map/filter/fold vs loops)
- **Imports**: Group by module, specific imports for common items
- **Formatting**: Follow standard Rust formatting (use `cargo fmt`)
- **Types**: Use strong typing with enums; prefer `Option<T>` over nulls
- **Naming**: CamelCase for types/structs, snake_case for functions/variables
- **Error Handling**: Use custom `AIError` enum with `thiserror`; return descriptive errors
- **Documentation**: Add doc comments for public functions and modules only
- **Comments**: Add code comments only for unexpected or non-obvious behavior
- **Module Structure**: Use direct files (not mod.rs) for module declarations
- **Architecture**: Use trait-based abstractions with builder pattern
- **Async**: Use `async_trait` for async trait methods