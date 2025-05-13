use ai::core::{
    builder::llm,
    types::{ChatRole, Message},
};
use dotenv::dotenv;
use serde::Deserialize;
use std::env;

#[derive(Deserialize)]
struct Foo {
    bar: i32,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load .env file if present
    dotenv().ok();

    // Get API key from environment
    let api_key = env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set");

    let messages: Vec<Message> = vec![Message {
        role: ChatRole::User,
        content: "Provide a random number".to_string(),
    }];

    let foo = llm::call()
        .provider("openai")?
        .model("gpt-4o-mini")
        .response_model::<Foo>()
        .messages(messages)
        .send()
        .await?;

    println!("bar: {:?}", foo.bar);

    Ok(())
}
