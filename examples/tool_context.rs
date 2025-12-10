//! Example demonstrating dependency injection for tools using context.
//!
//! This example shows how to pass external resources (like database connections,
//! HTTP clients, or configuration) to tools via context injection instead of
//! relying on global state.
//!
//! Run with: `cargo run --example tool-context`

use rsai::{tool, toolset, Ctx, ToolCall, ToolSet};
use std::sync::atomic::{AtomicU32, Ordering};

// =============================================================================
// Mock Resources
// =============================================================================

struct DatabasePool {
    query_count: AtomicU32,
}

impl DatabasePool {
    fn new() -> Self {
        Self { query_count: AtomicU32::new(0) }
    }

    fn search(&self, query: &str) -> Vec<String> {
        let count = self.query_count.fetch_add(1, Ordering::Relaxed);
        vec![
            format!("Result 1 for '{}'", query),
            format!("Result 2 for '{}'", query),
            format!("(Query #{} on this connection)", count + 1),
        ]
    }
}

// =============================================================================
// Application Context
// =============================================================================

struct AppContext {
    db: DatabasePool,
}

impl AsRef<DatabasePool> for AppContext {
    fn as_ref(&self) -> &DatabasePool {
        &self.db
    }
}

// =============================================================================
// Tools
// =============================================================================

#[tool]
/// Search the database for documents.
/// query: The search query.
fn search(db: Ctx<&DatabasePool>, query: String) -> Vec<String> {
    db.search(&query)
}

#[tool]
/// Add two numbers (no context needed).
/// a: First number.
/// b: Second number.
fn add(a: i32, b: i32) -> i32 {
    a + b
}

// =============================================================================
// Main
// =============================================================================

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create context with dependencies
    let context = AppContext { db: DatabasePool::new() };

    // Create toolset with context: `ContextType => tools...`
    let tools: ToolSet<AppContext> = toolset![AppContext => search, add]
        .with_context(context);

    // Show tool schemas (context params are excluded)
    println!("=== Tool Schemas ===");
    for schema in tools.tools()? {
        println!("  {} - {:?}", schema.name, schema.description);
        println!("    params: {}", schema.parameters);
    }

    // Execute tools directly (simulating what LLM would do)
    println!("\n=== Executing Tools ===");

    let result = tools.registry.execute(&ToolCall {
        id: "1".into(),
        call_id: "1".into(),
        name: "search".into(),
        arguments: serde_json::json!({ "query": "rust programming" }),
    }).await?;
    println!("search('rust programming') = {}", result);

    let result = tools.registry.execute(&ToolCall {
        id: "2".into(),
        call_id: "2".into(),
        name: "add".into(),
        arguments: serde_json::json!({ "a": 10, "b": 32 }),
    }).await?;
    println!("add(10, 32) = {}", result);

    // Execute search again to show context is shared (query count increases)
    let result = tools.registry.execute(&ToolCall {
        id: "3".into(),
        call_id: "3".into(),
        name: "search".into(),
        arguments: serde_json::json!({ "query": "dependency injection" }),
    }).await?;
    println!("search('dependency injection') = {}", result);

    Ok(())
}
