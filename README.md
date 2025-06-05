# Rust AI-SDK

A Rust library for creating AI-powered agents with tool usage capabilities.

## Example Usage

```rust
#[derive(Deserialize, JsonSchema)]
#[schemars(deny_unknown_fields)]
struct Foo {
    bar: i32,
}

let messages = vec![Message {
    role: ChatRole::User,
    content: "Provide a random number".to_string(),
}];

let foo = llm::call()
    .provider("openai")?
    .api_key(ApiKey::Default)?
    .model("gpt-4o-mini")
    .messages(messages)
    .complete::<Foo>()
    .await?;
```

> **Important**: When using structured output, you must annotate your structs with `#[schemars(deny_unknown_fields)]`. This is currently required for the structured output functionality to work correctly.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
