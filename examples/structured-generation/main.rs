use ai::{completion_schema, llm, Message, ChatRole, ApiKey};
use dotenv::dotenv;

#[completion_schema]
struct Analysis {
    sentiment: String,
    confidence: f32,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();

    let analysis = llm::call()
        .provider("openai")?
        .api_key(ApiKey::Default)?
        .model("gpt-4o-mini")
        .messages(vec![Message {
            role: ChatRole::User,
            content: "Analyze: 'This library is amazing!'".to_string(),
        }])
        .complete::<Analysis>()
        .await?;

    println!("Sentiment: {}, Confidence: {}", analysis.sentiment, analysis.confidence);

    Ok(())
}
