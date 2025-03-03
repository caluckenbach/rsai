use ai::{core::llm, tool};
use dotenv::dotenv;
use reqwest::Client;
use serde_json::json;
use std::collections::HashMap;
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();

    let api_key = env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set");
    let client = Client::new();

    // Create a weather tool
    let weather_tool = tool::tool(
        "get_weather",
        "Get the weather in a location",
        {
            let mut params = HashMap::new();
            params.insert(
                "location".to_string(),
                (
                    "string".to_string(),
                    "The location to get the weather for".to_string(),
                ),
            );
            params
        },
        vec!["location"],
        |params| {
            let binding = "unknown".to_string();
            let location = params.get("location").unwrap_or(&binding);
            let temperature = 24;

            json!({
                "location": location,
                "temperature": temperature,
                "unit": "celsius",
                "condition": "sunny"
            })
        },
    );

    let prompt = "What's the weather like in San Francisco?";

    // First turn: Get function call from LLM
    println!("Sending initial prompt: {}", prompt);
    let response =
        llm::generate_content_with_tools(&client, &api_key, prompt, &[weather_tool.clone()])
            .await?;

    // Check for function calls in response
    if let Some((function_name, args)) = llm::extract_function_call(&response) {
        println!("Function call detected:");
        println!("Function: {}", function_name);
        println!("Args: {}", args);

        // Execute the tool
        if let Some(result) =
            tool::process_function_call(&[weather_tool.clone()], &function_name, &args)
        {
            println!("\nTool execution result:");
            println!("{}", result);

            // Second turn: Send function result back to LLM
            println!("\nSending function result back to LLM...");

            // Send request with function result
            let final_response = llm::send_function_results(
                &client,
                &api_key,
                prompt,
                &function_name,
                &result,
                &[weather_tool],
            )
            .await?;

            // Process the final response
            if let Some((second_function, second_args)) =
                llm::extract_function_call(&final_response)
            {
                println!("\nAnother function call detected (could process recursively):");
                println!("Function: {}", second_function);
                println!("Args: {}", second_args);
            } else if let Some(text) = llm::extract_text_response(&final_response) {
                println!("\nFinal AI Response:");
                println!("{}", text);
            } else {
                println!("Failed to parse final response");
            }
        }
    } else if let Some(text) = llm::extract_text_response(&response) {
        println!("AI Response (no function call):\n{}", text);
    } else {
        println!("Failed to parse initial response");
    }

    Ok(())
}
