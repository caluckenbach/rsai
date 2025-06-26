use rsai::{ApiKey, ChatRole, Message, completion_schema, llm};
use dotenv::dotenv;

#[completion_schema]
struct Analysis {
    sentiment: String,
    confidence: f32,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();

    let analysis = llm::with("openai")?
        .api_key(ApiKey::Default)?
        .model("gpt-4o-mini")
        .messages(vec![Message {
            role: ChatRole::User,
            content: "Analyze: 'This library is amazing!'".to_string(),
        }])
        .complete::<Analysis>()
        .await?;

    println!(
        "Sentiment: {}, Confidence: {}",
        analysis.content.sentiment, analysis.content.confidence
    );

    Ok(())
}
