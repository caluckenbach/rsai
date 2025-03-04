use ai::{
    model::{ChatSettings, chat::ChatModel},
    provider::gemini::{GeminiProvider, GeminiSettings},
};
use dotenv::dotenv;
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load .env file if present
    dotenv().ok();

    // Get API key from environment
    let api_key = env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set");

    // Create a Gemini provider with the default model (gemini-2.0-flash)
    let provider = GeminiProvider::default(&api_key);

    // Create Gemini-specific settings
    let gemini_settings = GeminiSettings::new()
        .use_search_grounding(true)
        .into_provider_options();

    // Create chat settings with a system message
    let settings = ChatSettings::new()
        .system_prompt("You are a helpful assistant that provides concise answers.")
        .temperature(0.7)
        .max_tokens(1024)
        .provider_options(gemini_settings);

    // Create the chat model with the provider and settings
    let model = ChatModel::new(provider, settings);

    // Generate text for a simple prompt
    let prompt = "What are the three largest cities in France?";
    println!("Sending prompt: {}", prompt);

    // Call the model and get the response
    let response = model.generate_text(prompt).await?;

    // Print the response
    println!("\nAI Response:");
    println!("{:#?}", response);

    Ok(())
}
