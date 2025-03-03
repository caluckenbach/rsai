use dotenv::dotenv;
use reqwest::Client;
use serde_json::{json, Value};
use std::env;

async fn generate_content(client: &Client, api_key: &str, prompt: &str) -> Result<Value, Box<dyn std::error::Error>> {
    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={}",
        api_key
    );
    
    let request_body = json!({
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    });
    
    let response = client.post(&url)
        .header("Content-Type", "application/json")
        .json(&request_body)
        .send()
        .await?
        .json::<Value>()
        .await?;
    
    Ok(response)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();
    
    let api_key = env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set");
    let client = Client::new();
    
    let prompt = "Explain how AI works";
    let response = generate_content(&client, &api_key, prompt).await?;
    
    // Extract the text response from Gemini
    if let Some(text) = response
        .get("candidates")
        .and_then(|candidates| candidates[0].get("content"))
        .and_then(|content| content.get("parts"))
        .and_then(|parts| parts[0].get("text"))
        .and_then(|text| text.as_str()) {
        
        println!("AI Response:\n{}", text);
    } else {
        println!("Failed to parse response. Raw response:\n{:#?}", response);
    }
    
    Ok(())
}
