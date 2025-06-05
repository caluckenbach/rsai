use ai::core::{
    builder::{ApiKey, llm},
    types::{ChatRole, Message},
};
use dotenv::dotenv;
use serde::Deserialize;

#[derive(Deserialize)]
struct Foo {
    bar: i32,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();

    let messages: Vec<Message> = vec![Message {
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

    println!("bar: {:?}", foo.bar);

    Ok(())
}
