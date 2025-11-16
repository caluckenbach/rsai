use async_trait::async_trait;
use rsai::{LlmError, Tool, ToolCall, ToolFunction, ToolRegistry};
use serde_json::json;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;

// Mock tool implementations for testing
struct TestToolA;
#[async_trait]
impl ToolFunction for TestToolA {
    fn schema(&self) -> Tool {
        Tool {
            name: "test_tool_a".to_string(),
            description: Some("Test tool A".to_string()),
            parameters: json!({
                "type": "object",
                "properties": {
                    "input": {"type": "string"}
                }
            }),
            strict: Some(true),
        }
    }

    fn execute<'a>(
        &'a self,
        _params: serde_json::Value,
    ) -> rsai::BoxFuture<'a, Result<serde_json::Value, LlmError>> {
        Box::pin(async move { Ok(json!({"result": "success_a"})) })
    }
}

struct TestToolB;
#[async_trait]
impl ToolFunction for TestToolB {
    fn schema(&self) -> Tool {
        Tool {
            name: "test_tool_b".to_string(),
            description: Some("Test tool B".to_string()),
            parameters: json!({
                "type": "object",
                "properties": {
                    "value": {"type": "number"}
                }
            }),
            strict: Some(false),
        }
    }

    fn execute<'a>(
        &'a self,
        _params: serde_json::Value,
    ) -> rsai::BoxFuture<'a, Result<serde_json::Value, LlmError>> {
        Box::pin(async move { Ok(json!({"result": "success_b"})) })
    }
}

struct TestToolC;
#[async_trait]
impl ToolFunction for TestToolC {
    fn schema(&self) -> Tool {
        Tool {
            name: "test_tool_a".to_string(), // Same name as TestToolA
            description: Some("Test tool C - has same name as A".to_string()),
            parameters: json!({
                "type": "object",
                "properties": {}
            }),
            strict: Some(true),
        }
    }

    fn execute<'a>(
        &'a self,
        _params: serde_json::Value,
    ) -> rsai::BoxFuture<'a, Result<serde_json::Value, LlmError>> {
        Box::pin(async move { Ok(json!({"result": "success_c"})) })
    }
}

// ============================================================================
// BASIC TESTS
// ============================================================================

#[tokio::test]
async fn test_registration() {
    let registry = ToolRegistry::new();
    let tool = Arc::new(TestToolA);

    // Successful tool registration
    let result = registry.register(tool);
    assert!(result.is_ok(), "Tool registration should succeed");

    // Verify tool was registered by checking schemas
    let schemas = registry.get_schemas().expect("Failed to get schemas");
    assert_eq!(schemas.len(), 1);
    assert_eq!(schemas[0].name, "test_tool_a");
}

#[tokio::test]
async fn test_duplicate_detection() {
    let registry = ToolRegistry::new();
    let tool_a = Arc::new(TestToolA);
    let tool_a_duplicate = Arc::new(TestToolA);

    // First registration should succeed
    let result1 = registry.register(tool_a);
    assert!(result1.is_ok(), "First tool registration should succeed");

    // Second registration with same tool should fail
    let result2 = registry.register(tool_a_duplicate);
    assert!(result2.is_err(), "Duplicate tool registration should fail");

    // Verify error type and message
    if let Err(LlmError::ToolRegistration { tool_name, message }) = result2 {
        assert_eq!(tool_name, "test_tool_a");
        assert!(message.contains("already registered"));
    } else {
        panic!("Expected ToolRegistration error");
    }
}

#[tokio::test]
async fn test_different_tools_same_name() {
    let registry = ToolRegistry::new();
    let tool_a = Arc::new(TestToolA);

    // First registration should succeed
    let result1 = registry.register(tool_a);
    assert!(result1.is_ok(), "First tool registration should succeed");

    // Second registration with different tool but same name should fail
    let result2 = registry.register(Arc::new(TestToolC));
    assert!(result2.is_err(), "Registration with same name should fail");

    // Verify error details
    if let Err(LlmError::ToolRegistration { tool_name, message }) = result2 {
        assert_eq!(tool_name, "test_tool_a");
        assert!(message.contains("already registered"));
    } else {
        panic!("Expected ToolRegistration error");
    }
}

#[tokio::test]
async fn test_error_message_clarity() {
    let registry = ToolRegistry::new();
    let tool_a = Arc::new(TestToolA);
    let tool_a_duplicate = Arc::new(TestToolA);

    registry.register(tool_a).unwrap();
    let error = registry.register(tool_a_duplicate).unwrap_err();

    match error {
        LlmError::ToolRegistration { tool_name, message } => {
            // Error should contain tool name
            assert_eq!(tool_name, "test_tool_a");
            // Error message should be clear and actionable
            assert!(
                message.contains("already"),
                "Error should mention 'already'"
            );
            assert!(
                message.contains("registered"),
                "Error should mention 'registered'"
            );
            assert!(
                message.contains("test_tool_a"),
                "Error should include tool name in message"
            );
        }
        _ => panic!("Expected ToolRegistration error"),
    }
}

#[tokio::test]
async fn test_overwrite_functionality() {
    let registry = ToolRegistry::new();
    let tool_a = Arc::new(TestToolA);

    // First registration
    let result1 = registry.register(tool_a);
    assert!(result1.is_ok());

    // Overwrite should succeed
    let result2 = registry.overwrite(Arc::new(TestToolC));
    assert!(result2.is_ok(), "Overwrite should succeed");

    // Verify the tool was overwritten
    let schemas = registry.get_schemas().expect("Failed to get schemas");
    assert_eq!(schemas.len(), 1);
    assert_eq!(
        schemas[0].description,
        Some("Test tool C - has same name as A".to_string())
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
                let tool: Arc<dyn ToolFunction> = match i % 2 {
                    0 => Arc::new(TestToolA),
                    _ => Arc::new(TestToolB),
                };
                registry_clone.register(tool)
            })
        })
        .collect();

    let results: Vec<_> = futures::future::join_all(handles).await;

    // Count successful registrations
    let success_count = results.iter().filter(|r| matches!(r, Ok(Ok(())))).count();

    // Either 2 (both tools registered successfully) or all threads succeeded with different tools
    // In practice, all should succeed since we're using different tools
    assert!(
        success_count >= 2,
        "At least 2 different tools should be registered"
    );
}

#[tokio::test]
async fn test_concurrent_registration_same_tool() {
    let registry = Arc::new(ToolRegistry::new());

    let handles: Vec<_> = (0..SAME_TOOL_WORKERS)
        .map(|_| {
            let registry_clone = Arc::clone(&registry);
            tokio::spawn(async move {
                let tool: Arc<dyn ToolFunction> = Arc::new(TestToolA);
                registry_clone.register(tool)
            })
        })
        .collect();

    let results = futures::future::join_all(handles).await;

    // Count successful registrations
    let success_count = results.iter().filter(|r| matches!(r, Ok(Ok(())))).count();

    // Count failed registrations due to duplicate
    let duplicate_error_count = results
        .iter()
        .filter(|r| matches!(r, Ok(Err(LlmError::ToolRegistration { .. }))))
        .count();

    // Exactly one should succeed
    assert_eq!(success_count, 1, "Exactly one registration should succeed");
    // All others should fail with duplicate error
    assert_eq!(
        duplicate_error_count,
        SAME_TOOL_WORKERS - 1,
        "Remaining registrations should fail with duplicate error"
    );
}

#[tokio::test]
async fn test_concurrent_read_write() {
    let registry = Arc::new(ToolRegistry::new());

    // Start registration task
    let registry_clone = Arc::clone(&registry);
    let register_handle = tokio::spawn(async move {
        sleep(Duration::from_millis(10)).await;
        let tool: Arc<dyn ToolFunction> = Arc::new(TestToolA);
        registry_clone.register(tool)
    });

    // Start multiple read tasks that try to read during registration
    let mut read_handles = vec![];
    for _ in 0..READERS_DURING_WRITE {
        let registry_clone = Arc::clone(&registry);
        let handle = tokio::spawn(async move {
            sleep(Duration::from_millis(5)).await;
            registry_clone.get_schemas()
        });
        read_handles.push(handle);
    }

    let register_result = register_handle.await.unwrap();
    assert!(register_result.is_ok(), "Registration should succeed");

    let read_results = futures::future::join_all(read_handles).await;

    // All reads should succeed (readers don't block readers)
    for result in read_results {
        let schema_result = result.unwrap();
        assert!(schema_result.is_ok());
    }
}

#[tokio::test]
async fn test_concurrent_reads_during_registration() {
    let registry = Arc::new(ToolRegistry::new());

    // Pre-register a tool
    {
        registry.register(Arc::new(TestToolA)).unwrap();
    }

    // Start multiple concurrent reads
    let mut read_handles = vec![];
    for _ in 0..READERS_DURING_REGISTRATION {
        let registry_clone = Arc::clone(&registry);
        let handle = tokio::spawn(async move { registry_clone.get_schemas() });
        read_handles.push(handle);
    }

    // While reads are happening, try to write
    let write_handle = tokio::spawn(async move { registry.register(Arc::new(TestToolB)) });

    let read_results = futures::future::join_all(read_handles).await;
    let write_result = write_handle.await.unwrap();

    // All reads should succeed and see consistent state
    for result in read_results {
        let schemas = result.unwrap().unwrap();
        assert!(!schemas.is_empty()); // Should see at least the first tool
    }

    // Write should succeed
    assert!(write_result.is_ok());
}

#[tokio::test]
async fn test_high_concurrency_stress_test() {
    let registry = Arc::new(ToolRegistry::new());

    let handles: Vec<_> = (0..100)
        .map(|i| {
            let registry_clone = Arc::clone(&registry);
            tokio::spawn(async move {
                // Alternate between tool A and B
                let tool: Arc<dyn ToolFunction> = match i % 2 {
                    0 => Arc::new(TestToolA),
                    _ => Arc::new(TestToolB),
                };
                registry_clone.register(tool)
            })
        })
        .collect();

    let results = futures::future::join_all(handles).await;

    // Count successful registrations
    let success_count = results.iter().filter(|r| matches!(r, Ok(Ok(())))).count();

    // With 100 threads and 2 unique tools, we should have exactly 2 successes
    assert_eq!(
        success_count, 2,
        "Exactly 2 registrations should succeed (one for each tool)"
    );

    // Verify final state
    let schemas = registry.get_schemas().unwrap();
    assert_eq!(schemas.len(), 2, "Should have exactly 2 tools registered");
}

// ============================================================================
// REGISTRATION STATE TESTS
// ============================================================================

#[tokio::test]
async fn test_register_same_tool_twice() {
    let registry = ToolRegistry::new();
    let tool_a = Arc::new(TestToolA);
    let tool_a_clone = Arc::new(TestToolA);

    // First registration succeeds
    assert!(registry.register(tool_a).is_ok());

    // Second registration fails
    let result = registry.register(tool_a_clone);
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
    let tool = Arc::new(TestToolA);

    registry.register(tool).unwrap();

    let tool_call = ToolCall {
        id: "test_id".to_string(),
        call_id: "call_123".to_string(),
        name: "test_tool_a".to_string(),
        arguments: json!({"input": "test data"}),
    };

    let result = registry.execute(&tool_call).await;
    assert!(result.is_ok());

    let output = result.unwrap();
    assert_eq!(output["result"], "success_a");
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

    let schemas = registry.get_schemas().unwrap();
    assert_eq!(schemas.len(), 0);
}

#[tokio::test]
async fn test_get_schemas_multiple_tools() {
    let registry = ToolRegistry::new();

    registry.register(Arc::new(TestToolA)).unwrap();
    registry.register(Arc::new(TestToolB)).unwrap();

    let schemas = registry.get_schemas().unwrap();
    assert_eq!(schemas.len(), 2);

    let names: Vec<_> = schemas.iter().map(|s| &s.name).collect();
    assert!(names.contains(&&"test_tool_a".to_string()));
    assert!(names.contains(&&"test_tool_b".to_string()));
}
