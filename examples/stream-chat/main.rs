use ai::{
    model::{ChatSettings, chat::ChatModel},
    provider::gemini::{GeminiProvider, GeminiSettings},
};
use dotenv::dotenv;
use futures::StreamExt;
use std::env;
use std::io::{self, Write};

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
        .system_prompt("You are a helpful assistant that provides detailed, thoughtful answers.")
        .temperature(0.7)
        .max_tokens(1024)
        .provider_options(gemini_settings);

    // Create the chat model with the provider and settings
    let model = ChatModel::new(provider, settings);

    // Generate text for a simple prompt
    let prompt = "Explain the process of photosynthesis step by step, in detail.";
    println!("Sending prompt: {}", prompt);
    println!("\nAI Response (streaming):");

    // Get a stream of text chunks from the model
    let mut stream = model.stream_text(prompt).await?;

    // Process the stream chunks as they arrive
    while let Some(chunk_result) = stream.next().await {
        match chunk_result {
            Ok(chunk) => {
                // Print the chunk of text without a newline and flush immediately
                print!("{}", chunk.text);
                io::stdout().flush()?;

                // If this is the last chunk, print usage statistics
                if let Some(usage) = chunk.usage {
                    if chunk.text.is_empty() && !matches!(chunk.finish_reason, ai::model::chat::FinishReason::Unknown) {
                        println!("\n\nCompletion finished. Usage statistics:");
                        println!("  Prompt tokens: {}", usage.prompt_tokens);
                        println!("  Completion tokens: {}", usage.completion_tokens);
                        println!("  Total tokens: {}", usage.total_tokens);
                        println!("  Finish reason: {:?}", chunk.finish_reason);
                    }
                }
            },
            Err(e) => {
                eprintln!("\nError during streaming: {}", e);
                break;
            }
        }
    }

    println!("\n\nStreaming complete!");

    Ok(())
}