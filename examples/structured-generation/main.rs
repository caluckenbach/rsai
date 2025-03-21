use ai::{
    model::{
        ChatSettings,
        chat::{ChatModel, Mode, OutputType, Schema, StructuredOutputParameters, StructuredResult},
    },
    provider::gemini::{GeminiProvider, GeminiSettings},
};
use dotenv::dotenv;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::env;

// Define a struct for our structured output
#[derive(Serialize, Deserialize, JsonSchema, Debug)]
struct Recipe {
    name: String,
    ingredients: Vec<String>,
    instructions: Vec<String>,
    prep_time_minutes: i32,
    difficulty: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load .env file if present
    dotenv().ok();

    // Get API key from environment
    let api_key = env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set");

    // Create a Gemini provider with the default model (gemini-2.0-flash)
    let provider = GeminiProvider::default(&api_key);

    // Create Gemini-specific settings with structured outputs enabled
    let gemini_settings = GeminiSettings::new()
        .structured_outputs(true)
        .into_provider_options();

    // Create chat settings
    let settings = ChatSettings::new()
        .system_prompt("You are a helpful assistant specializing in creating recipes.")
        .provider_options(gemini_settings);

    // Create the chat model with the provider and settings
    let model = ChatModel::new(provider, settings);

    // Define the prompt for generating a structured recipe
    let prompt =
        "Create a recipe for a delicious vegetarian pasta dish that uses seasonal ingredients.";
    println!("Sending prompt: {}", prompt);

    // Set up parameters for structured output
    let parameters = StructuredOutputParameters {
        output: OutputType::Object,
        mode: Some(Mode::Json),
        schema: Some(Schema::<Recipe>::new()),
        schema_name: Some("Recipe".to_string()),
        schema_description: Some("A cooking recipe with ingredients and instructions".to_string()),
        enum_values: None,
    };

    // Generate the structured output
    let response = model.generate_object(prompt, &parameters).await?;

    // Print the response based on the structured result type
    println!("\nAI Response (Structured Output):");
    match &response.value {
        StructuredResult::Object(recipe) => {
            let recipe: &Recipe = recipe;
            println!("Recipe: {}", recipe.name);
            println!("\nDifficulty: {}", recipe.difficulty);
            println!("Prep Time: {} minutes", recipe.prep_time_minutes);

            println!("\nIngredients:");
            for (i, ingredient) in recipe.ingredients.iter().enumerate() {
                println!("  {}. {}", i + 1, ingredient);
            }

            println!("\nInstructions:");
            for (i, step) in recipe.instructions.iter().enumerate() {
                println!("  {}. {}", i + 1, step);
            }
        }
        StructuredResult::Array(items) => {
            println!("Received an array of {} recipes:", items.len());
            // Handle array result if needed
        }
        StructuredResult::Enum(value) => {
            println!("Received enum value: {}", value);
        }
        StructuredResult::NoSchema(value) => {
            println!("Received unstructured data: {:#?}", value);
        }
    }

    // Print usage statistics
    println!("\nUsage Statistics:");
    println!("  Prompt tokens: {}", response.usage.prompt_tokens);
    println!("  Completion tokens: {}", response.usage.completion_tokens);
    println!("  Total tokens: {}", response.usage.total_tokens);
    println!("  Finish reason: {:?}", response.finish_reason);

    Ok(())
}
