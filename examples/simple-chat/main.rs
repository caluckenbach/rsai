use ai::completion_schema;
use ai::core::{
    builder::{ApiKey, llm},
    types::{ChatRole, Message},
};
use dotenv::dotenv;

#[completion_schema]
struct Foo {
    bar: i32,
}

#[derive(Debug)]
#[completion_schema]
enum Model {
    Default,
    Reasoning,
    Research,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();

    let messages: Vec<Message> = vec![Message {
        role: ChatRole::User,
        content: "Pick the AI model that fits best for the following query: Which city is the capital of Germany".to_string(),
    }];

    let result = llm::call()
        .provider("openai")?
        .api_key(ApiKey::Default)?
        .model("gpt-4o-mini")
        .messages(messages)
        .complete::<Foo>()
        .await?;

    println!("Selected model: {:?}", result.bar);

    Ok(())
}
