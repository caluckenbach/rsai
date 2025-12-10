use rsai::{BoxFuture, LlmError, Tool, ToolCall, ToolFunction, ToolRegistry, tool};
use serde_json::json;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;

#[tool]
/// Deterministic tool A that returns a static payload.
/// input: Arbitrary string payload echoed for testing.
fn test_tool_a(input: String) -> serde_json::Value {
    let _ = input;
    json!({ "result": "success_a" })
}

#[tool]
/// Deterministic tool B that simulates async work.
/// value: Numeric value passed from the caller.
async fn test_tool_b(value: f64) -> serde_json::Value {
    let _ = value;
    sleep(Duration::from_millis(1)).await;
    json!({ "result": "success_b" })
}

fn tool_a() -> Arc<dyn ToolFunction<()>> {
    Arc::new(TestToolATool)
}

fn tool_b() -> Arc<dyn ToolFunction<()>> {
    Arc::new(TestToolBTool)
}

struct AlternateToolA;

impl ToolFunction<()> for AlternateToolA {
    fn schema(&self) -> Tool {
        Tool {
            name: "test_tool_a".to_string(),
            description: Some("Alternate implementation of tool A".to_string()),
            parameters: json!({
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "Arbitrary payload"
                    }
                },
                "required": ["input"],
                "additionalProperties": false
            }),
            strict: Some(true),
        }
    }

    fn execute<'a>(
        &'a self,
        _ctx: &'a (),
        _params: serde_json::Value,
    ) -> BoxFuture<'a, Result<serde_json::Value, LlmError>> {
        Box::pin(async move { Ok(json!({ "result": "alternate_success" })) })
    }
}

fn alternate_tool_a() -> Arc<dyn ToolFunction<()>> {
    Arc::new(AlternateToolA)
}

// ============================================================================
// BASIC TESTS
// ============================================================================

#[tokio::test]
async fn test_registration() {
    let registry = ToolRegistry::new();
    let tool = tool_a();

    let result = registry.register(tool);
    assert!(result.is_ok(), "Tool registration should succeed");

    let schemas = registry.get_schemas().expect("Failed to get schemas");
    assert_eq!(schemas.len(), 1);
    assert_eq!(schemas[0].name, "test_tool_a");
}

#[tokio::test]
async fn test_duplicate_detection() {
    let registry = ToolRegistry::new();

    assert!(registry.register(tool_a()).is_ok());

    let result = registry.register(tool_a());
    assert!(result.is_err(), "Duplicate tool registration should fail");

    if let Err(LlmError::ToolRegistration { tool_name, message }) = result {
        assert_eq!(tool_name, "test_tool_a");
        assert!(message.contains("already registered"));
    } else {
        panic!("Expected ToolRegistration error");
    }
}

#[tokio::test]
async fn test_different_tools_same_name() {
    let registry = ToolRegistry::new();

    assert!(registry.register(tool_a()).is_ok());

    let result = registry.register(alternate_tool_a());
    assert!(result.is_err(), "Registration with same name should fail");

    if let Err(LlmError::ToolRegistration { tool_name, message }) = result {
        assert_eq!(tool_name, "test_tool_a");
        assert!(message.contains("already registered"));
    } else {
        panic!("Expected ToolRegistration error");
    }
}

#[tokio::test]
async fn test_error_message_clarity() {
    let registry = ToolRegistry::new();

    registry.register(tool_a()).unwrap();
    let error = registry.register(tool_a()).unwrap_err();

    match error {
        LlmError::ToolRegistration { tool_name, message } => {
            assert_eq!(tool_name, "test_tool_a");
            assert!(message.contains("already"));
            assert!(message.contains("registered"));
            assert!(message.contains("test_tool_a"));
        }
        _ => panic!("Expected ToolRegistration error"),
    }
}

#[tokio::test]
async fn test_overwrite_functionality() {
    let registry = ToolRegistry::new();

    assert!(registry.register(tool_a()).is_ok());

    let result = registry.overwrite(alternate_tool_a());
    assert!(result.is_ok(), "Overwrite should succeed");

    let schemas = registry.get_schemas().expect("Failed to get schemas");
    assert_eq!(schemas.len(), 1);
    assert_eq!(
        schemas[0].description,
        Some("Alternate implementation of tool A".to_string())
    );
}

// ============================================================================
// CONCURRENCY TESTS
// ============================================================================

const DIFFERENT_TOOL_WORKERS: usize = 4;
const SAME_TOOL_WORKERS: usize = 4;
const READERS_DURING_WRITE: usize = 3;
const READERS_DURING_REGISTRATION: usize = 4;

#[tokio::test]
async fn test_concurrent_registration_different_tools() {
    let registry = Arc::new(ToolRegistry::new());

    let handles: Vec<_> = (0..DIFFERENT_TOOL_WORKERS)
        .map(|i| {
            let registry_clone = Arc::clone(&registry);
            tokio::spawn(async move {
                let tool = if i % 2 == 0 { tool_a() } else { tool_b() };
                registry_clone.register(tool)
            })
        })
        .collect();

    let results = futures::future::join_all(handles).await;
    let success_count = results.iter().filter(|r| matches!(r, Ok(Ok(())))).count();

    assert!(
        success_count >= 2,
        "At least two different tools should register successfully"
    );
}

#[tokio::test]
async fn test_concurrent_registration_same_tool() {
    let registry = Arc::new(ToolRegistry::new());

    let handles: Vec<_> = (0..SAME_TOOL_WORKERS)
        .map(|_| {
            let registry_clone = Arc::clone(&registry);
            tokio::spawn(async move { registry_clone.register(tool_a()) })
        })
        .collect();

    let results = futures::future::join_all(handles).await;
    let success_count = results.iter().filter(|r| matches!(r, Ok(Ok(())))).count();
    let duplicate_error_count = results
        .iter()
        .filter(|r| matches!(r, Ok(Err(LlmError::ToolRegistration { .. }))))
        .count();

    assert_eq!(success_count, 1, "Exactly one registration should succeed");
    assert_eq!(
        duplicate_error_count,
        SAME_TOOL_WORKERS - 1,
        "Remaining registrations should fail with duplicate error"
    );
}

#[tokio::test]
async fn test_concurrent_read_write() {
    let registry = Arc::new(ToolRegistry::new());

    let registry_clone = Arc::clone(&registry);
    let register_handle = tokio::spawn(async move {
        sleep(Duration::from_millis(10)).await;
        registry_clone.register(tool_a())
    });

    let mut read_handles = vec![];
    for _ in 0..READERS_DURING_WRITE {
        let registry_clone = Arc::clone(&registry);
        read_handles.push(tokio::spawn(async move {
            sleep(Duration::from_millis(5)).await;
            registry_clone.get_schemas()
        }));
    }

    assert!(register_handle.await.unwrap().is_ok());

    let read_results = futures::future::join_all(read_handles).await;
    for result in read_results {
        assert!(result.unwrap().is_ok());
    }
}

#[tokio::test]
async fn test_concurrent_reads_during_registration() {
    let registry = Arc::new(ToolRegistry::new());
    registry.register(tool_a()).unwrap();

    let mut read_handles = vec![];
    for _ in 0..READERS_DURING_REGISTRATION {
        let registry_clone = Arc::clone(&registry);
        read_handles.push(tokio::spawn(async move { registry_clone.get_schemas() }));
    }

    let registry_clone = Arc::clone(&registry);
    let write_handle = tokio::spawn(async move { registry_clone.register(tool_b()) });

    let read_results = futures::future::join_all(read_handles).await;
    let write_result = write_handle.await.unwrap();

    for result in read_results {
        let schemas = result.unwrap().unwrap();
        assert!(!schemas.is_empty());
    }
    assert!(write_result.is_ok());
}

#[tokio::test]
async fn test_high_concurrency_stress_test() {
    let registry = Arc::new(ToolRegistry::new());

    let handles: Vec<_> = (0..100)
        .map(|i| {
            let registry_clone = Arc::clone(&registry);
            tokio::spawn(async move {
                let tool = if i % 2 == 0 { tool_a() } else { tool_b() };
                registry_clone.register(tool)
            })
        })
        .collect();

    let results = futures::future::join_all(handles).await;
    let success_count = results.iter().filter(|r| matches!(r, Ok(Ok(())))).count();

    assert_eq!(
        success_count, 2,
        "Exactly two unique tools should be registered"
    );

    let schemas = registry.get_schemas().unwrap();
    assert_eq!(schemas.len(), 2);
}

// ============================================================================
// REGISTRATION STATE TESTS
// ============================================================================

#[tokio::test]
async fn test_register_same_tool_twice() {
    let registry = ToolRegistry::new();

    assert!(registry.register(tool_a()).is_ok());

    let result = registry.register(tool_a());
    assert!(result.is_err());

    if let Err(LlmError::ToolRegistration { message, .. }) = result {
        assert!(message.contains("already registered"));
    } else {
        panic!("Expected ToolRegistration error");
    }
}

// ============================================================================
// EXECUTION TESTS
// ============================================================================

#[tokio::test]
async fn test_tool_execution_after_registration() {
    let registry = ToolRegistry::new();
    registry.register(tool_a()).unwrap();

    let tool_call = ToolCall {
        id: "test_id".to_string(),
        call_id: "call_123".to_string(),
        name: "test_tool_a".to_string(),
        arguments: json!({ "input": "test data" }),
    };

    let result = registry.execute(&tool_call).await.unwrap();
    assert_eq!(result["result"], "success_a");
}

#[tokio::test]
async fn test_tool_not_found_error() {
    let registry = ToolRegistry::new();

    let tool_call = ToolCall {
        id: "test_id".to_string(),
        call_id: "call_123".to_string(),
        name: "nonexistent_tool".to_string(),
        arguments: json!({}),
    };

    let result = registry.execute(&tool_call).await;
    assert!(result.is_err());

    if let Err(LlmError::ToolNotFound(name)) = result {
        assert_eq!(name, "nonexistent_tool");
    } else {
        panic!("Expected ToolNotFound error");
    }
}

// ============================================================================
// SCHEMA RETRIEVAL TESTS
// ============================================================================

#[tokio::test]
async fn test_get_schemas_empty_registry() {
    let registry = ToolRegistry::new();
    assert_eq!(registry.get_schemas().unwrap().len(), 0);
}

#[tokio::test]
async fn test_get_schemas_multiple_tools() {
    let registry = ToolRegistry::new();

    registry.register(tool_a()).unwrap();
    registry.register(tool_b()).unwrap();

    let schemas = registry.get_schemas().unwrap();
    assert_eq!(schemas.len(), 2);

    let names: Vec<_> = schemas.iter().map(|s| &s.name).collect();
    assert!(names.contains(&&"test_tool_a".to_string()));
    assert!(names.contains(&&"test_tool_b".to_string()));
}
