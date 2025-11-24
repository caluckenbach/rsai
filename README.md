> **⚠️ WARNING**: This is a pre-release version with an unstable API. Breaking changes may occur between versions. Use with caution and pin to specific versions in production applications.

## Quick Start

```rust
use rsai::{llm, Message, ChatRole, ApiKey, Provider, TextResponse, completion_schema};

#[completion_schema]
struct Analysis {
    sentiment: String,
    confidence: f32,
}

let analysis = llm::with(Provider::OpenAI)
    .api_key(ApiKey::Default)?
    .model("gpt-4o-mini")
    .messages(vec![Message {
        role: ChatRole::User,
        content: "Analyze: 'This library is amazing!'".to_string(),
    }])
    .complete::<Analysis>()
    .await?;

let reply = llm::with(Provider::OpenAI)
    .api_key(ApiKey::Default)?
    .model("gpt-4o-mini")
    .messages(vec![
        Message {
            role: ChatRole::System,
            content: "You are friendly and concise.".to_string(),
        },
        Message {
            role: ChatRole::User,
            content: "Share a fun fact about Rust.".to_string(),
        },
    ])
    .complete::<TextResponse>()
    .await?;

println!("{}", reply.text);
```

## Structured Generation

Get AI responses that conform to your exact Rust types - no JSON parsing required!

The `#[completion_schema]` macro automatically adds the necessary derives and attributes for structured output:

- `#[derive(serde::Deserialize, schemars::JsonSchema)]`
- `#[schemars(deny_unknown_fields)]`

### Structs

```rust
#[completion_schema]
struct MovieReview {
    title: String,
    rating: u8,
    summary: String,
}

let review = llm::with(Provider::OpenAI)
    .api_key(ApiKey::Default)?
    .model("gpt-4o-mini")
    .messages(vec![Message {
        role: ChatRole::User,
        content: "Review the movie 'Inception' in 50 words".to_string(),
    }])
    .complete::<MovieReview>()
    .await?;
```

### Simple Enums

```rust
#[completion_schema]
enum Priority {
    Low,
    Medium,
    High,
    Critical,
}

let priority = llm::with(Provider::OpenAI)
    .api_key(ApiKey::Default)?
    .model("gpt-4o-mini")
    .messages(vec![Message {
        role: ChatRole::User,
        content: "What priority for a button color change request?".to_string(),
    }])
    .complete::<Priority>()
    .await?;
```

### Enums with Data

```rust
#[completion_schema]
enum TaskStatus {
    NotStarted,
    InProgress { percentage: u8 },
    Completed { date: String },
    Blocked { reason: String },
}

let status = llm::with(Provider::OpenAI)
    .api_key(ApiKey::Default)?
    .model("gpt-4o-mini")
    .messages(vec![Message {
        role: ChatRole::User,
        content: "Status of a task that's 75% complete?".to_string(),
    }])
    .complete::<TaskStatus>()
    .await?;
```

### Manual Schema Definition (Alternative)

If you prefer manual control, you can still use the traditional approach:

```rust
use serde::Deserialize;
use schemars::JsonSchema;

#[derive(Deserialize, JsonSchema)]
#[schemars(deny_unknown_fields)]
struct CustomType {
    field: String,
}
```

> **Note**: The library automatically handles provider-specific requirements. For example, OpenAI requires root schemas to be objects, so non-object types like enums are transparently wrapped and unwrapped.

## Text Generation

To obtain plain-text output without defining a schema, target `TextResponse`:

```rust
use rsai::{llm, ApiKey, Provider, Message, ChatRole, TextResponse};

let fact = llm::with(Provider::OpenAI)
    .api_key(ApiKey::Default)?
    .model("gpt-4o-mini")
    .messages(vec![
        Message {
            role: ChatRole::System,
            content: "You explain things clearly but briefly.".to_string(),
        },
        Message {
            role: ChatRole::User,
            content: "What makes Rust's borrow checker special?".to_string(),
        },
    ])
    .complete::<TextResponse>()
    .await?;

println!("{}", fact.text);
```

See `examples/text_generation.rs` for a runnable version.

## Known Issues

- ..

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
